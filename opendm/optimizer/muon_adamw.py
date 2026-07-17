"""Muon for Action Expert matrices and AdamW for all other parameters."""

import math

import torch
import torch.distributed as dist

_MUON_SHAPE = "_opendm_muon_shape"


def mark_muon_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if (
            param.requires_grad
            and param.ndim == 2
            and name.startswith("model.action_expert.layers.")
            and name.endswith(".weight")
        ):
            setattr(param, _MUON_SHAPE, tuple(param.shape))


def zeropower_via_newtonschulz5(
    matrix: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    """Approximate the matrix zeroth power used by Muon."""
    update = matrix.float()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.T
    update /= update.norm() + 1e-7

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * gram @ gram) @ update

    return update.T if transposed else update


class MuonAdamW(torch.optim.Optimizer):
    """A single optimizer containing Muon and AdamW parameter groups."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_lr_scale: float = 1.0,
        muon_moonlight_coefficient: float = 0.2,
    ) -> None:
        adamw_params = []
        muon_params = []
        muon_shapes = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            shape = getattr(param, _MUON_SHAPE, None)
            if shape is None:
                adamw_params.append(param)
            else:
                muon_params.append(param)
                muon_shapes.append(shape)

        groups = [
            {"params": adamw_params, "use_muon": False},
            {
                "params": muon_params,
                "use_muon": True,
                "muon_shapes": muon_shapes,
            },
        ]
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "muon_momentum": muon_momentum,
            "muon_nesterov": muon_nesterov,
            "muon_ns_steps": muon_ns_steps,
            "muon_lr_scale": muon_lr_scale,
            "muon_moonlight_coefficient": muon_moonlight_coefficient,
        }
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _adamw_step(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        for param in group["params"]:
            if param.grad is None:
                continue

            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

            state["step"] += 1
            grad = param.grad.float()
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            param.mul_(1 - group["lr"] * group["weight_decay"])
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(group["eps"])
            update = exp_avg.div(denom).mul_(group["lr"] / bias1)
            param.add_(update.to(param.dtype), alpha=-1)

    def _muon_step(self, group: dict) -> None:
        momentum = group["muon_momentum"]
        for param, matrix_shape in zip(
            group["params"], group["muon_shapes"], strict=True
        ):
            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                # Match AdamW's state schema so FSDP can rebuild flat states.
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

            state["step"] += 1
            grad = (
                param.grad.float()
                if param.grad is not None
                else torch.zeros_like(param, dtype=torch.float32)
            )
            buffer = state["exp_avg"]
            buffer.mul_(momentum).add_(grad)
            direction = (
                grad.add(buffer, alpha=momentum) if group["muon_nesterov"] else buffer
            )

            param.mul_(1 - group["lr"] * group["weight_decay"])
            full_direction, offset = self._gather_matrix(direction, matrix_shape)
            update = zeropower_via_newtonschulz5(
                full_direction.reshape(matrix_shape),
                steps=group["muon_ns_steps"],
            ).reshape(-1)
            local_update = update[offset : offset + param.numel()].reshape_as(param)
            effective_lr = (
                group["lr"]
                * group["muon_lr_scale"]
                * group["muon_moonlight_coefficient"]
                * math.sqrt(max(matrix_shape))
            )
            param.add_(local_update.to(param.dtype), alpha=-effective_lr)

    @staticmethod
    def _gather_matrix(
        local: torch.Tensor,
        matrix_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, int]:
        local = local.reshape(-1)
        if not dist.is_initialized():
            return local, 0

        size = torch.tensor(local.numel(), device=local.device)
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)
        sizes = [int(item) for item in sizes]
        full_numel = math.prod(matrix_shape)
        if all(item == full_numel for item in sizes):
            return local, 0

        padded = local.new_zeros(max(sizes))
        padded[: local.numel()] = local
        shards = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(shards, padded)
        rank = dist.get_rank()
        return torch.cat(
            [
                shard[:shard_size]
                for shard, shard_size in zip(shards, sizes, strict=True)
            ]
        ), sum(sizes[:rank])
