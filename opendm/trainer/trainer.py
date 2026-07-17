import os
import shutil
from typing import Any

import torch
import transformers
from loguru import logger
from transformers import Trainer, TrainingArguments


class DMTrainer(Trainer):
    """Trainer extension for OpenDM model training."""

    def __init__(self, *args, **kwargs):
        self.exp_config: Any = kwargs.pop("exp_config")
        training_args = self._link_exp_config()
        super().__init__(*args, args=training_args, **kwargs)
        self.loss_cache = {}

    def _prepare_for_training(self, *args, **kwargs):
        if not getattr(self.exp_config, "use_lora", False):
            return super()._prepare_for_training(*args, **kwargs)

        original_update = getattr(transformers.trainer, "update_fsdp_plugin_peft", None)
        if original_update is None:
            return super()._prepare_for_training(*args, **kwargs)

        def _skip_update_fsdp_plugin_peft(model, accelerator):
            logger.info(
                "Skipping Transformers PEFT FSDP auto-wrap update for DM05 LoRA."
            )

        transformers.trainer.update_fsdp_plugin_peft = _skip_update_fsdp_plugin_peft
        try:
            return super()._prepare_for_training(*args, **kwargs)
        finally:
            transformers.trainer.update_fsdp_plugin_peft = original_update

    def _fsdp_qlora_plugin_updates(self):
        if getattr(self.exp_config, "use_lora", False):
            logger.info(
                "Skipping Transformers PEFT FSDP auto-wrap update for DM05 LoRA."
            )
            return
        return super()._fsdp_qlora_plugin_updates()

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        opt_model: torch.nn.Module = model if model is not None else self.model

        if self.optimizer is None:
            if self.exp_config.optimizer_config.optim == "muon_adamw":
                self.optimizer = self.exp_config.optimizer_config.build_muon_adamw(
                    opt_model
                )
                return self.optimizer
            optimizer_grouped_parameters = (
                self.exp_config.optimizer_config._get_optimizer_grouped_parameters(
                    opt_model
                )
            )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        logger.info(f"Saving checkpoint at step {self.state.global_step}")
        super()._save_checkpoint(model, trial)

        if self.args.should_save:
            norm_stats_path = self.exp_config.data_config.norm_stats_path(
                self.exp_config.model_config.chunk_size
            )
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.state.global_step}",
            )
            shutil.copyfile(
                norm_stats_path,
                os.path.join(checkpoint_dir, "norm_stats.json"),
            )
        return  # Checkpoint saved successfully.

    def _link_exp_config(self) -> TrainingArguments:
        """Build Hugging Face training arguments from the experiment config."""
        linked_args = {
            "output_dir": self.exp_config.trainer_config.output_dir,
            "max_steps": self.exp_config.trainer_config.num_train_steps,
            "per_device_train_batch_size": self.exp_config.trainer_config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.exp_config.trainer_config.gradient_accumulation_steps,
            "save_strategy": self.exp_config.trainer_config.save_strategy,
            "save_steps": self.exp_config.trainer_config.save_steps,
            "save_total_limit": self.exp_config.trainer_config.save_total_limit,
            "save_only_model": self.exp_config.trainer_config.save_only_model,
            "logging_steps": self.exp_config.trainer_config.logging_steps,
            "dataloader_num_workers": self.exp_config.trainer_config.dataloader_num_workers,
            "dataloader_persistent_workers": getattr(
                self.exp_config.trainer_config, "dataloader_persistent_workers", False
            ),
            "dataloader_prefetch_factor": getattr(
                self.exp_config.trainer_config, "dataloader_prefetch_factor", None
            ),
            "dataloader_drop_last": True,
            # Keep model_max_length controlled by the model or processor config.
            "bf16": self.exp_config.trainer_config.bf16,
            "tf32": self.exp_config.trainer_config.tf32,
            "lr_scheduler_type": self.exp_config.trainer_config.lr_scheduler_type,
            "lr_scheduler_kwargs": self.exp_config.trainer_config.lr_scheduler_kwargs,
            "run_name": self.exp_config.trainer_config.run_name,
            "remove_unused_columns": False,
            "learning_rate": self.exp_config.optimizer_config.base_lr,
            "adam_beta1": self.exp_config.optimizer_config.adam_beta1,
            "adam_beta2": self.exp_config.optimizer_config.adam_beta2,
            "warmup_steps": self.exp_config.optimizer_config.warmup_steps,
            "weight_decay": self.exp_config.optimizer_config.weight_decay,
            "seed": getattr(self.exp_config.trainer_config, "seed", 42),
            "data_seed": getattr(self.exp_config.trainer_config, "seed", 42),
        }
        linked_args["max_grad_norm"] = 1.0

        wandb_project = self.exp_config.trainer_config.wandb_project
        if wandb_project:
            linked_args["report_to"] = ["wandb"]
            if os.environ.get("WANDB_DISABLED", "").lower() in {
                "true",
                "1",
                "yes",
                "on",
            }:
                os.environ["WANDB_DISABLED"] = "false"
        else:
            linked_args["report_to"] = []

        world_size = int(os.environ["WORLD_SIZE"])
        if self.exp_config.trainer_config.fsdp1 and world_size > 1:
            linked_args["fsdp"] = "shard_grad_op"
            linked_args["fsdp_config"] = {
                "backward_prefetch": "BACKWARD_PRE",
                "use_orig_params": True,
                "sync_module_states": True,
            }
            linked_args["ddp_find_unused_parameters"] = False
            linked_args["seed"] += int(os.environ.get("RANK", 0))
        else:
            linked_args["ddp_find_unused_parameters"] = True
        training_args = TrainingArguments(**linked_args)
        return training_args

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_keys = [_ for _ in outputs if _.endswith("_loss")]

        for loss_key in loss_keys:
            # Cache all *_loss metrics for logging, averaging them across
            # distributed workers when torch.distributed is initialized.
            raw = outputs[loss_key]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                buf = torch.tensor(
                    [
                        raw.detach().item() if raw is not None else 0.0,
                        1.0 if raw is not None else 0.0,
                    ],
                    device=loss.device,
                )
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
                total_val, total_count = buf[0].item(), buf[1].item()
                if total_count > 0:
                    self.loss_cache[loss_key] = total_val / total_count
                elif loss_key not in self.loss_cache:
                    self.loss_cache[loss_key] = 0.0
            else:
                if raw is None:
                    if loss_key not in self.loss_cache:
                        self.loss_cache[loss_key] = 0.0
                else:
                    self.loss_cache[loss_key] = raw.detach().item()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs.update(self.loss_cache)
        logs["step"] = self.state.global_step
        super().log(logs, start_time)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str) -> None:
    """Collect the model state dict on CPU and save it with the Trainer."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
