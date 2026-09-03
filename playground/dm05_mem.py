from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Literal

import torch
import tyro
from loguru import logger

from opendm.exp.dm05_exp import (
    DM05DataConfig as _DM05DataConfig,
)
from opendm.exp.dm05_exp import (
    DM05Exp as _DM05Exp,
)
from opendm.exp.dm05_exp import (
    DM05InferenceConfig as _DM05InferenceConfig,
)
from opendm.exp.dm05_exp import (
    DM05ModelConfig as _DM05ModelConfig,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)

DEFAULT_CKPT = "./checkpoints/DM05-MEM"


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    model_name_or_path: str | None = field(default=DEFAULT_CKPT)
    chunk_size: int = field(default=50)
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="sdpa"
    )
    vision_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2"
    ] = field(default="flash_attention_2")
    action_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = (
        field(default="sdpa")
    )
    liger_kernel: bool = field(default=False)


@dataclass
class DM05DataConfig(_DM05DataConfig):
    is_history: bool = field(default=True)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    model_max_length: int = field(default=2048)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=14)
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )
    backend: Literal["default", "fast"] = field(default="default")
    clip_to_bounds: bool = field(default=False)
    compose_eef_rot: bool = field(default=True)
    full_action_mask: bool = field(default=True)
    use_transformed_state: bool = field(default=True)
    state_layout: str = field(default="identity")
    denorm_state: bool = field(default=True)
    max_history_images: int = field(default=32)
    fast_prefix_len: int = field(default=2048)
    prefix_seq_len_buckets: list[int] | None = field(default_factory=lambda: [2048])
    vision_trt_engine_path: str | None = field(
        default_factory=lambda: f"{DEFAULT_CKPT}/dm05_vision_n35.engine"
    )

    def _request_default_overrides(self) -> dict:
        return {
            "default_robot_type": "DOS W1",
            "default_speed": "0.1",
            "default_control_mode": "joint",
        }


@dataclass
class DM05Exp(_DM05Exp):
    task: Literal["inference"] = field(default="inference")
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)

    def _initialize_inference_runtime(self) -> None:
        if self.inference_config.backend == "fast":
            from opendm.infer.dm05_infer import DM05FastInferRuntime

            DM05FastInferRuntime.resolve_prefix_seq_len_buckets(
                self.inference_config.prefix_seq_len_buckets,
                fast_backend=True,
                prefix_capacity=int(self.inference_config.fast_prefix_len),
            )
            self.model_config.llm_attn_implementation = "flex_attention"
        ckpt = pathlib.Path(self.model_config.model_name_or_path)
        norm_stats_path = ckpt / "norm_stats.json"
        if not norm_stats_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is missing {norm_stats_path}; "
                "set --model-config.model-name-or-path"
            )
        logger.info(f"Loading model from {ckpt}")
        model = self.model_config.build_model(use_lora=False)
        self.inference_config._initialize(
            model=model,
            model_name_or_path=str(ckpt),
            norm_stats_path=str(norm_stats_path),
            n_bins=self.data_config.n_bins,
            model_max_length=self.trainer_config.model_max_length,
            use_absolute_action=True,
            add_state=self.data_config.add_state,
            is_history=self.data_config.is_history,
        )
        if self.inference_config.backend != "fast":
            _ia = self.inference_config.model.inference_action
            _device_type = self.inference_config.device.type

            def _inference_action(*args, **kwargs):
                with torch.autocast(device_type=_device_type, dtype=torch.bfloat16):
                    return _ia(*args, **kwargs)

            self.inference_config.model.inference_action = _inference_action


if __name__ == "__main__":
    tyro.cli(DM05Exp).inference()
