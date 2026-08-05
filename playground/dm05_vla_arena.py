import json
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import tyro
from flask import request
from PIL import Image

from opendm.constants.robot import ActionMode, RobotType
from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import (
    ChatTokenization,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from opendm.dataset.vla_arena import VLA_ARENA_EEF_STATE_DESC
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
    DM05OptimizerConfig as _DM05OptimizerConfig,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)


@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="vla_arena_eef_L0_L")
    action_mode: ActionMode = field(default=ActionMode.ABSOLUTE)
    add_state: bool = field(default=False)

    def build_dataset(
        self,
        processor,
        action_horizon: int,
        tokenizer_max_length: int = 1024,
    ) -> tuple:
        dataset_info = self._dataset_info()
        image_keys = dataset_info["image_keys"]
        image_prompts = dataset_info["image_prompts"]
        pipeline = Pipeline(
            [
                self._action_transform(action_horizon),
                LoadImages(image_keys=image_keys, image_dir=dataset_info["image_dir"]),
                PixelTransform(
                    transform_pipeline=NoAugmentationPipeline(),
                ),
                Normalize(
                    norm_stats_path=str(self.norm_stats_path(action_horizon)),
                    norm_keys=["state", "action"],
                    use_quantiles=True,
                ),
                ChatTokenization(
                    processor=processor,
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=image_prompts,
                    add_state=self.add_state,
                ),
                PadAction(32),
            ]
        )
        dataset = JsonlDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=pipeline,
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(dataset_info),
        )

        collator = TrainingCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            max_length=tokenizer_max_length,
        )

        return dataset, collator


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    model_name_or_path: str | None = field(default=("./checkpoints/DM05"))
    chunk_size: int = field(default=20)
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="flex_attention"
    )
    vision_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2"
    ] = field(default="flash_attention_2")
    action_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = (
        field(default="sdpa")
    )
    vlm_gradient_checkpointing: bool = field(default=True)
    ae_gradient_checkpointing: bool = field(default=True)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    optim: Literal["adamw", "muon_adamw"] = field(default="muon_adamw")
    base_lr: float = field(default=2e-5)
    warmup_steps: int = field(default=1000)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    fsdp1: bool | None = field(default=True)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    save_steps: int = field(default=5000)
    num_train_steps: int = field(default=60000)
    save_only_model: bool = field(default=False)
    model_max_length: int = field(default=1024)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=7)
    image_prompts: list[str] = field(default_factory=lambda: ["Head", "Left wrist"])

    def _prepare_input_legacy(self) -> dict:
        images = request.files.getlist("image", None)
        states = request.form.get("states", None)
        text = request.form.get("text", "")
        robot_type = request.form.get("robot_type", "Franka")
        speed = request.form.get("speed", "0.5")
        control_mode = request.form.get("control_mode")

        assert robot_type == RobotType.FRANKA.value, (
            f"Unsupported robot_type {robot_type!r}. Only {RobotType.FRANKA.value} is supported."
        )
        state_desc = VLA_ARENA_EEF_STATE_DESC

        pil_images = [Image.open(img).convert("RGB") for img in images]
        state = np.array(json.loads(states), dtype=np.float32)

        return {
            "images": pil_images,
            "prompt": text,
            "state": state,
            "meta_data": {
                "robot_type": robot_type,
                "speed": str(speed),
                "control_mode": control_mode,
                "state_desc": state_desc,
            },
        }


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")
