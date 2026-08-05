"""Muon for Action Expert matrices and AdamW for all other parameters."""

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

_MUON_SHAPE = "_opendm_muon_shape"
_MUON_NAME = "_opendm_muon_name"


@dataclass(frozen=True)
class MuonShardPlan:
    kind: str
    full_numel: int
    local_numel: int
    sizes: tuple[int, ...]
    active_ranks: tuple[int, ...]
    local_offset: int
    process_group: object | None = None


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _normalize_parameter_name(name: str) -> str:
    prefixes = ("module.", "_fsdp_wrapped_module.", "_orig_mod.")
    while True:
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        else:
            return name


def is_default_muon_parameter(name: str, param: torch.nn.Parameter) -> bool:
    """Select trainable Action Expert layer matrices for Muon."""
    if not param.requires_grad or param.ndim != 2:
        return False

    normalized = _normalize_parameter_name(name)
    if not (
        normalized.startswith("model.action_expert.")
        or normalized.startswith("action_expert.")
        or ".model.action_expert." in normalized
    ):
        return False

    lowered = normalized.lower()
    if ".layers." not in lowered or not lowered.endswith(".weight"):
        return False
    excluded = (
        "embed_tokens",
        "embedding",
        "layernorm.weight",
        "_norm.weight",
        ".norm.weight",
        ".norm.",
    )
    return not any(marker in lowered for marker in excluded)


def mark_muon_parameters(
    model: torch.nn.Module,
    predicate: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> list[str]:
    """Record Muon selection and pre-FSDP matrix shapes on model parameters."""
    predicate = predicate or is_default_muon_parameter
    selected = []
    seen_param_ids: set[int] = set()

    for name, param in model.named_parameters():
        if id(param) in seen_param_ids:
            continue
        seen_param_ids.add(id(param))

        for attr in (_MUON_SHAPE, _MUON_NAME):
            if hasattr(param, attr):
                delattr(param, attr)
        if not predicate(name, param):
            continue
        if param.ndim != 2:
            raise RuntimeError(
                f"Muon parameter {name} must be 2D, got shape {tuple(param.shape)}."
            )

        setattr(param, _MUON_SHAPE, tuple(int(dim) for dim in param.shape))
        setattr(param, _MUON_NAME, name)
        selected.append(name)

    if not selected:
        raise RuntimeError("MuonAdamW selected 0 parameters.")
    return selected


def zeropower_via_newtonschulz5(
    matrix: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate the matrix zeroth power used by Muon."""
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2D tensor.")

    update = matrix.float()
    if update.numel() == 0:
        return update

    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.T

    norm = update.norm()
    if not torch.isfinite(norm) or norm <= 0:
        return torch.zeros_like(matrix, dtype=torch.float32)
    update = update / norm.clamp_min(eps)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(int(steps), 0)):
        gram = update @ update.T
        update = a * update + (b * gram + c * (gram @ gram)) @ update

    return update.T if transposed else update


def _load_adamw_functional():
    try:
        from torch.optim.adamw import adamw
    except ImportError:
        from torch.optim._functional import adamw
    return adamw


_ADAMW_FUNCTIONAL = _load_adamw_functional()
_ADAMW_ARGUMENTS = inspect.signature(_ADAMW_FUNCTIONAL).parameters


def _call_adamw_functional(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    group: dict,
) -> None:
    beta1, beta2 = group["betas"]
    kwargs = {
        "amsgrad": False,
        "beta1": float(beta1),
        "beta2": float(beta2),
        "lr": float(group["lr"]),
        "weight_decay": float(group.get("weight_decay", 0.0) or 0.0),
        "eps": float(group["eps"]),
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "grad_scale": None,
        "found_inf": None,
        "has_complex": False,
    }
    kwargs = {key: value for key, value in kwargs.items() if key in _ADAMW_ARGUMENTS}
    _ADAMW_FUNCTIONAL(
        [param],
        [grad],
        [state["exp_avg"]],
        [state["exp_avg_sq"]],
        [],
        [state["step"]],
        **kwargs,
    )


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
        muon_names = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            shape = getattr(param, _MUON_SHAPE, None)
            if shape is None:
                adamw_params.append(param)
            else:
                muon_params.append(param)
                muon_shapes.append(tuple(int(dim) for dim in shape))
                muon_names.append(getattr(param, _MUON_NAME, None))

        if not muon_params:
            raise RuntimeError(
                "MuonAdamW metadata is missing; mark parameters before FSDP wrapping."
            )

        groups = [
            {"params": adamw_params, "use_muon": False},
            {
                "params": muon_params,
                "use_muon": True,
                "muon_shapes": muon_shapes,
                "muon_names": muon_names,
            },
        ]
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_muon": False,
            "muon_momentum": muon_momentum,
            "muon_nesterov": muon_nesterov,
            "muon_ns_steps": muon_ns_steps,
            "muon_lr_scale": muon_lr_scale,
            "muon_moonlight_coefficient": muon_moonlight_coefficient,
        }
        super().__init__(groups, defaults)
        self._muon_shard_plan_cache: dict[int, list[MuonShardPlan]] = {}
        self._normalize_param_groups()

    def _normalize_param_groups(self) -> None:
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
            if not group.get("use_muon", False):
                continue

            shapes = group.get("muon_shapes") or group.pop("muon_matrix_shapes", None)
            names = group.get("muon_names")
            infos = group.pop("muon_param_infos", None)
            if shapes is None and infos is not None:
                shapes = []
                for info in infos:
                    if isinstance(info, dict):
                        shapes.append(info["matrix_shape"])
                    elif hasattr(info, "matrix_shape"):
                        shapes.append(info.matrix_shape)
                    else:
                        shapes.append(info[1])
            if names is None and infos is not None:
                names = [
                    (
                        info.get("name")
                        if isinstance(info, dict)
                        else getattr(info, "name", None)
                    )
                    for info in infos
                ]
            if shapes is None or len(shapes) != len(group["params"]):
                raise RuntimeError("Muon parameter metadata does not match parameters.")
            group["muon_shapes"] = [
                tuple(int(dim) for dim in shape) for shape in shapes
            ]
            names = list(names or [None] * len(shapes))
            if len(names) != len(shapes):
                raise RuntimeError("Muon parameter names do not match parameters.")
            group["muon_names"] = names

    @staticmethod
    def _state_for_param(param: torch.Tensor, state: dict) -> dict:
        if "momentum_buffer" in state:
            state.setdefault("exp_avg", state.pop("momentum_buffer"))
        if "step" not in state:
            state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
        elif not torch.is_tensor(state["step"]):
            state["step"] = torch.tensor(
                float(state["step"]), dtype=torch.float32, device=param.device
            )
        else:
            if state["step"].numel() != 1:
                raise RuntimeError("Optimizer step state must contain one value.")
            state["step"] = (
                state["step"]
                .detach()
                .to(device=param.device, dtype=torch.float32)
                .reshape(())
            )

        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if value is None:
                state[key] = torch.zeros_like(
                    param, dtype=torch.float32, memory_format=torch.preserve_format
                )
            elif value.dtype != torch.float32 or value.device != param.device:
                state[key] = value.to(device=param.device, dtype=torch.float32)
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load current, legacy OpenDM, or compatible Dexbotic optimizer state."""
        super().load_state_dict(state_dict)
        self._normalize_param_groups()
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state.get(param)
                if state:
                    self._state_for_param(param, state)
        self._muon_shard_plan_cache.clear()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _adamw_step(self, group: dict) -> None:
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients.")

            state = self._state_for_param(param, self.state[param])
            _call_adamw_functional(param, grad.detach().float(), state, group)

    @staticmethod
    def _device_for_params(params: list[torch.Tensor]) -> torch.device:
        if params:
            return params[0].device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _muon_group_is_active(self, params: list[torch.Tensor]) -> bool:
        local_active = any(param.grad is not None for param in params)
        if not _dist_ready():
            return local_active
        flag = torch.tensor(
            int(local_active),
            dtype=torch.int32,
            device=self._device_for_params(params),
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _muon_step(self, group: dict) -> None:
        if not self._muon_group_is_active(group["params"]):
            return

        momentum = float(group["muon_momentum"])
        plans = self._get_muon_shard_plans(group)
        for index, (param, matrix_shape, plan) in enumerate(
            zip(group["params"], group["muon_shapes"], plans, strict=True)
        ):
            if plan.local_numel == 0:
                continue

            grad = param.grad
            if grad is not None and grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients.")
            if grad is not None and grad.numel() != param.numel():
                name = group["muon_names"][index] or index
                raise RuntimeError(f"Muon gradient shape mismatch for {name}.")

            state = self._state_for_param(param, self.state[param])
            state["step"].add_(1)
            local_grad = (
                grad.detach().float()
                if grad is not None
                else torch.zeros_like(param, dtype=torch.float32)
            )
            buffer = state["exp_avg"]
            buffer.mul_(momentum).add_(local_grad)
            direction = (
                local_grad.clone().add_(buffer, alpha=momentum)
                if group["muon_nesterov"]
                else buffer.detach().clone()
            )

            weight_decay = float(group.get("weight_decay", 0.0) or 0.0)
            if weight_decay:
                param.mul_(1.0 - float(group["lr"]) * weight_decay)

            full_direction, offset = self._gather_matrix(direction, plan)
            if full_direction.numel() == 0:
                continue
            update = zeropower_via_newtonschulz5(
                full_direction.reshape(matrix_shape),
                steps=group["muon_ns_steps"],
            ).reshape(-1)
            local_update = update.narrow(0, offset, plan.local_numel).reshape_as(param)
            effective_lr = (
                float(group["lr"])
                * float(group["muon_lr_scale"])
                * float(group["muon_moonlight_coefficient"])
                * math.sqrt(max(matrix_shape))
            )
            param.add_(local_update.to(param.dtype), alpha=-effective_lr)

    @staticmethod
    def _plan_matches_group(group: dict, plans: list[MuonShardPlan]) -> bool:
        if len(plans) != len(group["params"]):
            return False
        world_size = dist.get_world_size() if _dist_ready() else 1
        return all(
            plan.full_numel == math.prod(shape)
            and plan.local_numel == param.numel()
            and len(plan.sizes) == world_size
            for param, shape, plan in zip(
                group["params"], group["muon_shapes"], plans, strict=True
            )
        )

    def _get_muon_shard_plans(self, group: dict) -> list[MuonShardPlan]:
        cache_key = id(group)
        plans = self._muon_shard_plan_cache.get(cache_key)
        if plans is None or not self._plan_matches_group(group, plans):
            plans = [
                self._build_muon_shard_plan(param, matrix_shape)
                for param, matrix_shape in zip(
                    group["params"], group["muon_shapes"], strict=True
                )
            ]
            self._muon_shard_plan_cache[cache_key] = plans
        return plans

    @staticmethod
    def _build_muon_shard_plan(
        param: torch.Tensor,
        matrix_shape: tuple[int, int],
    ) -> MuonShardPlan:
        local_numel = int(param.numel())
        full_numel = math.prod(matrix_shape)
        if local_numel > full_numel:
            raise RuntimeError(
                f"Muon matrix {matrix_shape} has only {full_numel} elements, "
                f"but the local tensor has {local_numel}."
            )

        if not _dist_ready():
            if local_numel not in (0, full_numel):
                raise RuntimeError(
                    f"Muon matrix {matrix_shape} has {local_numel} local elements, "
                    f"expected 0 or {full_numel}."
                )
            return MuonShardPlan(
                "empty" if local_numel == 0 else "local",
                full_numel,
                local_numel,
                (local_numel,),
                (0,) if local_numel else (),
                0,
            )

        size = torch.tensor(local_numel, device=param.device, dtype=torch.long)
        gathered = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, size)
        sizes = tuple(int(item) for item in gathered)
        active_ranks = tuple(rank for rank, size in enumerate(sizes) if size > 0)
        total_numel = sum(sizes)
        process_group = None

        if total_numel == 0:
            kind = "empty"
        elif total_numel == full_numel:
            kind = "owner_only" if len(active_ranks) == 1 else "partial"
            if kind == "partial" and len(active_ranks) < dist.get_world_size():
                process_group = dist.new_group(ranks=list(active_ranks))
        elif all(size == full_numel for size in sizes):
            kind = "replicated"
        else:
            raise RuntimeError(
                f"Muon matrix {matrix_shape} cannot be reconstructed from "
                f"local shard sizes {sizes}."
            )

        return MuonShardPlan(
            kind=kind,
            full_numel=full_numel,
            local_numel=local_numel,
            sizes=sizes,
            active_ranks=active_ranks,
            local_offset=sum(sizes[: dist.get_rank()]),
            process_group=process_group,
        )

    @staticmethod
    def _gather_matrix(
        local: torch.Tensor,
        plan: MuonShardPlan,
    ) -> tuple[torch.Tensor, int]:
        local = local.reshape(-1).contiguous()
        if local.numel() != plan.local_numel:
            raise RuntimeError("Muon local tensor no longer matches its shard plan.")
        if plan.kind == "empty":
            return local.new_empty(0), 0
        if plan.kind in {"local", "owner_only", "replicated"}:
            return local, 0
        if plan.kind != "partial":
            raise RuntimeError(f"Unknown Muon shard plan kind: {plan.kind}.")

        rank = dist.get_rank()
        if rank not in plan.active_ranks:
            return local.new_empty(0), plan.local_offset

        active_sizes = tuple(plan.sizes[rank] for rank in plan.active_ranks)
        padded = local.new_zeros(max(active_sizes))
        padded[: local.numel()] = local
        shards = [torch.empty_like(padded) for _ in active_sizes]
        dist.all_gather(shards, padded, group=plan.process_group)
        return (
            torch.cat(
                [
                    shard[:shard_size]
                    for shard, shard_size in zip(shards, active_sizes, strict=True)
                ]
            ),
            plan.local_offset,
        )
