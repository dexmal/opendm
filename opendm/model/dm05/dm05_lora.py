"""DM05 LoRA helpers."""

import json
import os
from dataclasses import dataclass, field

import torch
from loguru import logger

from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration


def _is_lora_checkpoint(model_name_or_path: str | None) -> bool:
    if not model_name_or_path:
        return False
    return os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))


def _read_lora_base_model_path(adapter_path: str) -> str | None:
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        return None
    with open(adapter_config, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("base_model_name_or_path") or None


def _load_dm05_config_for_inference(
    model_config_path: str,
    config_overrides: dict | None = None,
) -> DM05Config:
    config = DM05Config.from_pretrained(model_config_path)
    for attr, value in (config_overrides or {}).items():
        if hasattr(config, attr):
            setattr(config, attr, value)
            logger.info("Overriding DM05 inference config {}={}", attr, value)
        else:
            logger.warning(
                "Ignoring unknown DM05 inference config override {}={}",
                attr,
                value,
            )
    return config


@dataclass
class DM05LoraConfig:
    style: str = field(default="openvla_e1_alllinear_time_mod")
    r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    bias: str = field(default="none")
    target_modules: list[str] | str = field(default="all-linear")
    modules_to_save: list[str] = field(
        default_factory=lambda: [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "dm05_time_modulators",
        ]
    )
    dump_trainable_path: str | None = field(
        default=(
            "user_checkpoints/dm05_sft/trainable_summaries/"
            "dm05_lora_libero_pi0_all.json"
        )
    )


_DM05_TIME_MODULATOR_ALIASES = {
    "dm05_time_modulators",
    "dm05_action_time_modulators",
    "action_time_modulators",
}


def unwrap_dm05_model(model: torch.nn.Module) -> DM05ForConditionalGeneration:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _expand_modules_to_save(
    model: DM05ForConditionalGeneration,
    modules_to_save: list[str],
) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    def append(name: str) -> None:
        if name not in seen:
            expanded.append(name)
            seen.add(name)

    for name in modules_to_save:
        if name not in _DM05_TIME_MODULATOR_ALIASES:
            append(name)
            continue

        action_expert = getattr(getattr(model, "model", model), "action_expert", None)
        input_modulators = getattr(action_expert, "input_time_modulators", None)
        mlp_modulators = getattr(action_expert, "mlp_time_modulators", None)
        final_modulator = getattr(action_expert, "final_time_modulator", None)
        if (
            input_modulators is None
            or mlp_modulators is None
            or final_modulator is None
        ):
            raise ValueError(
                f"Cannot expand LoRA modules_to_save alias {name!r}: "
                "DM05 action time modulators are not present on the model."
            )
        if len(input_modulators) != len(mlp_modulators):
            raise ValueError(
                "DM05 action time modulator lists have inconsistent lengths: "
                f"input={len(input_modulators)}, mlp={len(mlp_modulators)}"
            )

        for idx in range(len(input_modulators)):
            append(f"input_time_modulators.{idx}")
        for idx in range(len(mlp_modulators)):
            append(f"mlp_time_modulators.{idx}")
        append("final_time_modulator")

    return expanded


def _is_all_linear_target(target_modules: list[str] | str) -> bool:
    if target_modules == "all-linear":
        return True
    return (
        isinstance(target_modules, list)
        and len(target_modules) == 1
        and target_modules[0] == "all-linear"
    )


def _resolve_target_modules(
    model: DM05ForConditionalGeneration,
    target_modules: list[str] | str,
) -> list[str] | str:
    if not _is_all_linear_target(target_modules):
        return target_modules

    supported_types: tuple[type[torch.nn.Module], ...] = (torch.nn.Linear,)
    try:
        from transformers.pytorch_utils import Conv1D

        supported_types = supported_types + (Conv1D,)
    except ImportError:
        pass

    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()

    resolved: set[str] = set()
    for name, module in model.named_modules():
        if not name or not isinstance(module, supported_types):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if name.endswith("lm_head"):
            continue
        parts = name.split(".")
        if parts[-1].isdigit() and len(parts) >= 2:
            resolved.add(".".join(parts[-2:]))
        else:
            resolved.add(parts[-1])

    if not resolved:
        raise ValueError(
            "DM05 LoRA target_modules='all-linear' found no Linear modules."
        )

    logger.info(
        "Resolved DM05 LoRA target_modules='all-linear' to {} target suffixes.",
        len(resolved),
    )
    return sorted(resolved)


def _patch_peft_tied_weights_keys_compat(model: torch.nn.Module) -> None:
    root_tied_keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(root_tied_keys, dict):
        return

    patched: list[str] = []
    for module_name, module in model.named_modules():
        if module is model:
            continue
        module_tied_keys = getattr(module, "_tied_weights_keys", None)
        if isinstance(module_tied_keys, list):
            module._tied_weights_keys = None
            patched.append(module_name)

    if patched:
        logger.info(
            "Patched PEFT tied-weight metadata for {} submodules before LoRA wrap.",
            len(patched),
        )


def _patch_peft_fsdp_tied_weights(model: torch.nn.Module) -> None:
    patched: list[str] = []
    for module_name, module in model.named_modules():
        if getattr(module, "_tied_weights_keys", None):
            module._tied_weights_keys = None
            patched.append(module_name or "<root>")

    if patched:
        logger.info("Cleared tied-weight metadata for DM05 LoRA/FSDP compatibility.")


def _cast_trainable_params_to_model_dtype(model: torch.nn.Module) -> None:
    target_dtype = None
    for param in model.parameters():
        if not param.requires_grad and param.is_floating_point():
            target_dtype = param.dtype
            break
    if target_dtype is None:
        return

    cast_count = 0
    for param in model.parameters():
        if not param.requires_grad or not param.is_floating_point():
            continue
        if param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=target_dtype)
            cast_count += 1

    if cast_count:
        logger.info(
            "Cast {} DM05 LoRA trainable parameters to model dtype {} for FSDP.",
            cast_count,
            target_dtype,
        )


def _dump_trainable_summary(
    model: torch.nn.Module,
    lora_config: DM05LoraConfig,
    target_modules,
) -> None:
    trainable = []
    frozen = 0
    trainable_total = 0
    lora_modules = []
    unexpected_trainable = []
    allowed_markers = tuple(
        ["lora_", "modules_to_save"] + list(lora_config.modules_to_save)
    )
    for name, param in model.named_parameters():
        count = int(param.numel())
        if param.requires_grad:
            trainable_total += count
            trainable.append({"name": name, "shape": list(param.shape), "numel": count})
            if not any(marker in name for marker in allowed_markers):
                unexpected_trainable.append(name)
        else:
            frozen += count
        if "lora_" in name:
            lora_modules.append(name)

    summary = {
        "style": lora_config.style,
        "target_modules": target_modules,
        "modules_to_save": lora_config.modules_to_save,
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "trainable_total": trainable_total,
        "frozen_total": frozen,
        "trainable_ratio": trainable_total / max(trainable_total + frozen, 1),
        "trainable_parameters": trainable,
        "lora_parameter_names": lora_modules,
        "unexpected_trainable_parameters": unexpected_trainable,
    }
    logger.info(
        "DM05 LoRA trainable summary: style={}, trainable={} frozen={} ratio={:.6f}",
        lora_config.style,
        trainable_total,
        frozen,
        summary["trainable_ratio"],
    )
    if lora_config.dump_trainable_path:
        dump_dir = os.path.dirname(lora_config.dump_trainable_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(lora_config.dump_trainable_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(
            "Wrote DM05 LoRA trainable summary to {}",
            lora_config.dump_trainable_path,
        )


def apply_lora_to_dm05_model(
    model: DM05ForConditionalGeneration,
    lora_config: DM05LoraConfig,
    base_model_name_or_path: str | None,
) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "DM05 LoRA requires `peft` to be installed in the training environment"
        ) from exc

    target_modules = _resolve_target_modules(model, lora_config.target_modules)
    modules_to_save = _expand_modules_to_save(model, list(lora_config.modules_to_save))
    if modules_to_save != lora_config.modules_to_save:
        logger.info(
            "Expanded DM05 LoRA modules_to_save: raw={} expanded_count={}",
            lora_config.modules_to_save,
            len(modules_to_save),
        )
        lora_config.modules_to_save = modules_to_save

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=target_modules,
        modules_to_save=lora_config.modules_to_save,
    )
    peft_config.base_model_name_or_path = base_model_name_or_path or ""

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    _patch_peft_tied_weights_keys_compat(model)
    model = get_peft_model(model, peft_config)
    _patch_peft_fsdp_tied_weights(model)
    _cast_trainable_params_to_model_dtype(model)
    logger.info(
        "Enabled DM05 LoRA: style={}, r={}, alpha={}, target_modules={}, "
        "modules_to_save={}",
        lora_config.style,
        lora_config.r,
        lora_config.lora_alpha,
        target_modules,
        lora_config.modules_to_save,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    _dump_trainable_summary(model, lora_config, target_modules)
    return model


def load_dm05_model_for_inference(
    model_name_or_path: str,
    config_overrides: dict | None = None,
    **from_pretrained_kwargs,
) -> torch.nn.Module:
    if not _is_lora_checkpoint(model_name_or_path):
        config = _load_dm05_config_for_inference(model_name_or_path, config_overrides)
        return DM05ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            config=config,
            **from_pretrained_kwargs,
        )

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading a LoRA DM05 checkpoint requires `peft`") from exc

    base_model_path = _read_lora_base_model_path(model_name_or_path)
    if not base_model_path:
        raise ValueError(
            "LoRA adapter checkpoint does not record base_model_name_or_path."
        )

    logger.info("Loading DM05 LoRA base model from {}", base_model_path)
    config_source = (
        model_name_or_path
        if os.path.exists(os.path.join(model_name_or_path, "config.json"))
        else base_model_path
    )
    config = _load_dm05_config_for_inference(config_source, config_overrides)
    base_model = DM05ForConditionalGeneration.from_pretrained(
        base_model_path,
        config=config,
        **from_pretrained_kwargs,
    )
    logger.info("Loading DM05 LoRA adapter from {}", model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_name_or_path)
    logger.info("Merging DM05 LoRA adapter into the base model for inference")
    return model.merge_and_unload()
