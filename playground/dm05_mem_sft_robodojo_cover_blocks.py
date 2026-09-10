import os
from dataclasses import dataclass, field
from typing import Literal

import tyro

from opendm.constants.robot import ActionMode, RobotType
from opendm.data.augmentations import TrainingTransformPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import (
    ChatTokenization,
    LoadHistory,
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
    DM05ModelConfig as _DM05ModelConfig,
)
from opendm.exp.dm05_exp import (
    DM05OptimizerConfig as _DM05OptimizerConfig,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)

DEFAULT_CKPT = os.environ.get(
    "DM05_MEM_CHECKPOINT",
    "./checkpoints/DM05-MEM",
)
HISTORY_SLOTS = 20


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    model_name_or_path: str | None = field(default=DEFAULT_CKPT)
    chunk_size: int = field(default=25)
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="flex_attention"
    )
    vision_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2"
    ] = field(default="flash_attention_2")
    action_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = (
        field(default="sdpa")
    )
    liger_kernel: bool = field(default=False)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    optim: Literal["adamw", "muon_adamw"] = field(default="muon_adamw")
    base_lr: float = field(default=2.5e-5)
    warmup_steps: int = field(default=1000)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    wandb_project: str | None = field(default="dm05_sft")
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=10000)
    per_device_train_batch_size: int = field(default=4)
    model_max_length: int = field(default=1536)


@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="robodojo_sim_cover_blocks")
    action_mode: ActionMode = field(default=ActionMode.ABSOLUTE)
    is_history: bool = field(default=True)

    def build_dataset(
        self,
        processor,
        action_horizon: int,
        tokenizer_max_length: int = 1536,
    ) -> tuple:
        dataset_info = self._dataset_info()
        image_keys = dataset_info["image_keys"]
        image_prompts = dataset_info["image_prompts"]
        pipeline = Pipeline(
            [
                self._action_transform(action_horizon),
                LoadImages(image_keys=image_keys, image_dir=dataset_info["image_dir"]),
                LoadHistory(
                    image_key=image_keys[0],
                    image_dir=dataset_info["image_dir"],
                    max_history_images=HISTORY_SLOTS,
                ),
                PixelTransform(
                    transform_pipeline=TrainingTransformPipeline(p=0.5),
                ),
                Normalize(
                    norm_stats_path=str(self.norm_stats_path(action_horizon)),
                    norm_keys=["state", "action"],
                    use_quantiles=True,
                    clip_to_bounds=False,
                ),
                ChatTokenization(
                    processor=processor,
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=image_prompts,
                    add_state=self.add_state,
                    is_history=True,
                    max_history_images=HISTORY_SLOTS,
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
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=14)
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )
    max_history_images: int = field(default=HISTORY_SLOTS)
    fast_prefix_len: int = field(default=2048)
    prefix_seq_len_buckets: list[int] | None = field(default_factory=lambda: [2048])
    clip_to_bounds: bool = field(default=False)

    def _request_default_overrides(self) -> dict:
        return {
            "default_robot_type": RobotType.ALOHA.value,
            "default_speed": "0.5",
        }


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)

    def _initialize_inference_runtime(self) -> None:
        if self.inference_config.backend != "fast":
            self.inference_config.prefix_seq_len_buckets = None
        else:
            self.model_config.liger_kernel = False
        super()._initialize_inference_runtime()


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")
