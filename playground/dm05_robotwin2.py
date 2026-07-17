import json
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import tyro
from flask import request
from PIL import Image

from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
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
    DM05OptimizerConfig as _DM05OptimizerConfig,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)


@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="robotwin2_generalist")
    action_mode: ActionMode = field(default=ActionMode.ABSOLUTE)

    def build_dataset(
        self,
        processor,
        action_horizon: int,
        tokenizer_max_length: int = 1024,
    ) -> tuple:
        dataset_info = self._dataset_info()
        image_keys = dataset_info["image_keys"]
        pipeline = Pipeline(
            [
                self._action_transform(action_horizon),
                LoadImages(image_keys=image_keys, image_dir=dataset_info["image_dir"]),
                PixelTransform(
                    transform_pipeline=NoAugmentationPipeline(),
                    image_keys=image_keys,
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
                    image_keys=image_keys,
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
class DM05OptimizerConfig(_DM05OptimizerConfig):
    base_lr: float = field(default=4e-5)
    optim: Literal["adamw", "muon_adamw"] = field(default="muon_adamw")


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    per_device_train_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=1)
    save_steps: int = field(default=10000)
    num_train_steps: int = field(default=100000)
    save_only_model: bool = field(default=False)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=14)
    image_keys: list[str] = field(
        default_factory=lambda: ["images_1", "images_2", "images_3"]
    )

    def _prepare_input(self) -> dict:
        images = request.files.getlist("image", None)
        states = request.form.get("states", None)
        text = request.form.get("text", "")
        robot_type = request.form.get("robot_type", "Aloha RoboTwin2")
        speed = request.form.get("speed", "0.5")
        control_mode = request.form.get("control_mode")

        assert robot_type == RobotType.ALOHA_ROBOTWIN2.value, (
            f"Unsupported robot_type {robot_type!r}. Only {RobotType.ALOHA_ROBOTWIN2.value} is supported."
        )
        state_desc = ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2]

        pil_images = {
            self.image_keys[i]: Image.open(img).convert("RGB")
            for i, img in enumerate(images)
        }
        state = np.array(json.loads(states), dtype=np.float32)

        return {
            **pil_images,
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
