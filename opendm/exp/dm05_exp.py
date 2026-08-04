import base64
import binascii
import hashlib
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Literal

import numpy as np
import torch
import tyro
from flask import Flask, jsonify, request
from loguru import logger
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from werkzeug.exceptions import BadRequest

import opendm.data.normalize as normalize
from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
from opendm.data.augmentations import NoAugmentationPipeline, TrainingTransformPipeline
from opendm.data.collator import NormStatsCollator, TrainingCollator
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import (
    ActionAbsolute,
    BuildAction,
    ChatTokenization,
    Denormalize,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
    ToDevice,
)
from opendm.dataset.register import CONVERSATION_DATA
from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
from opendm.model.dm05.dm05_lora import (
    DM05LoraConfig,
    _is_lora_checkpoint,
    apply_lora_to_dm05_model,
    load_dm05_model_for_inference,
    unwrap_dm05_model,
)
from opendm.optimizer import MuonAdamW, mark_muon_parameters
from opendm.optimizer.muon_adamw import is_default_muon_parameter
from opendm.trainer.trainer import DMTrainer, safe_save_model_for_hf_trainer


@dataclass
class Config:
    pass


@dataclass
class DM05ModelConfig(Config):
    model_name_or_path: str | None = field(default="./checkpoints/DM05")
    chunk_size: int = field(default=50)
    bf16: bool = field(default=True)
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="flex_attention"
    )
    vision_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2"
    ] = field(default="flash_attention_2")
    action_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = (
        field(default="sdpa")
    )
    liger_kernel: bool = field(default=True)
    freeze_vlm_embedding: bool = field(default=True)
    vlm_gradient_checkpointing: bool = field(default=True)
    ae_gradient_checkpointing: bool = field(default=True)
    lora_config: DM05LoraConfig = field(default_factory=DM05LoraConfig)

    def _config_overrides(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
        }

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.bf16 else torch.float32

    def _load_base_checkpoint_model(self) -> DM05ForConditionalGeneration:
        config = DM05Config.from_pretrained(self.model_name_or_path)
        for attr, value in self._config_overrides().items():
            setattr(config, attr, value)
        return DM05ForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            config=config,
            torch_dtype=self._torch_dtype(),
        )

    def _load_adapter_checkpoint_model(self) -> torch.nn.Module:
        return load_dm05_model_for_inference(
            self.model_name_or_path,
            config_overrides=self._config_overrides(),
            trust_remote_code=True,
            torch_dtype=self._torch_dtype(),
        )

    def _apply_runtime_model_options(
        self,
        model: torch.nn.Module,
    ) -> None:
        dm05_model = unwrap_dm05_model(model)
        dm05_model.set_attention_implementation(
            llm_attn_implementation=self.llm_attn_implementation,
            vision_attn_implementation=self.vision_attn_implementation,
            action_attn_implementation=self.action_attn_implementation,
            bf16=self.bf16,
        )
        dm05_model.enable_gradient_checkpointing(
            vlm_gradient_checkpointing=self.vlm_gradient_checkpointing,
            ae_gradient_checkpointing=self.ae_gradient_checkpointing,
        )
        if self.freeze_vlm_embedding:
            dm05_model.freeze_vlm_embedding()

    def _apply_full_model_runtime_optimizations(
        self, model: DM05ForConditionalGeneration
    ) -> None:
        if self.liger_kernel and torch.cuda.is_available():
            model._apply_liger_kernel()
        elif self.liger_kernel:
            logger.info("CUDA unavailable; skipping Liger Kernel for DM05 model.")

    def _wrap_new_lora_for_training(
        self, model: DM05ForConditionalGeneration
    ) -> torch.nn.Module:
        return apply_lora_to_dm05_model(
            model,
            self.lora_config,
            base_model_name_or_path=self.model_name_or_path or "",
        )

    def build_model(self, use_lora: bool = False) -> torch.nn.Module:
        if _is_lora_checkpoint(self.model_name_or_path):
            model = self._load_adapter_checkpoint_model()
            self._apply_runtime_model_options(model)
            return model

        model = self._load_base_checkpoint_model()
        self._apply_runtime_model_options(model)
        self._apply_full_model_runtime_optimizations(model)

        if not use_lora:
            return model
        return self._wrap_new_lora_for_training(model)


@dataclass
class DM05OptimizerConfig(Config):
    optim: Literal["adamw", "muon_adamw"] = field(default="adamw")
    base_lr: float = field(default=2.5e-5)
    weight_decay: float = field(default=1e-10)
    warmup_steps: int = field(default=1000)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=1e-8)
    muon_momentum: float = field(default=0.95)
    muon_nesterov: bool = field(default=True)
    muon_ns_steps: int = field(default=5)
    muon_lr_scale: float = field(default=1.0)
    muon_moonlight_coefficient: float = field(default=0.2)

    def should_use_muon(
        self,
        name: str,
        param: torch.nn.Parameter,
    ) -> bool:
        return is_default_muon_parameter(name, param)

    def prepare_model(self, model: torch.nn.Module) -> None:
        if self.optim == "muon_adamw":
            mark_muon_parameters(model, predicate=self.should_use_muon)

    def build_muon_adamw(self, model: torch.nn.Module) -> MuonAdamW:
        return MuonAdamW(
            model,
            lr=self.base_lr,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
            muon_momentum=self.muon_momentum,
            muon_nesterov=self.muon_nesterov,
            muon_ns_steps=self.muon_ns_steps,
            muon_lr_scale=self.muon_lr_scale,
            muon_moonlight_coefficient=self.muon_moonlight_coefficient,
        )

    def _get_optimizer_grouped_parameters(self, model: torch.nn.Module) -> list:
        parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.weight_decay,
            }
        ]
        return parameters


@dataclass
class DM05TrainerConfig(Config):
    fsdp1: bool | None = field(default=True)
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    num_train_steps: int = field(default=100000)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=5000)
    save_total_limit: int = field(default=50)
    save_only_model: bool = field(default=False)
    logging_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    dataloader_persistent_workers: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=2)
    model_max_length: int = field(default=1024)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})
    wandb_project: str | None = field(default="")

    def __post_init__(self):
        if self.output_dir is not None:
            self.run_name = os.path.basename(self.output_dir)
        if self.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = self.wandb_project


@dataclass
class DM05DataConfig(Config):
    dataset_name: str = field(default="demo")
    n_bins: int = field(default=256)
    action_mode: ActionMode = field(default=ActionMode.RELATIVE)
    compute_norm_stats_max_batches: int | None = field(default=None)
    norm_stats_root: str = field(default="./norm_stats")
    add_state: bool = field(default=True)
    is_history: bool = field(default=False)

    def _dataset_info(self) -> dict:
        assert self.dataset_name in CONVERSATION_DATA
        dataset_info = CONVERSATION_DATA[self.dataset_name]
        return {
            k: val.value if isinstance(val, Enum) else val
            for k, val in dataset_info.items()
        }

    def _dataset_meta(self, dataset_info: dict) -> dict:
        return {
            k: v
            for k, v in dataset_info.items()
            if k
            not in (
                "jsonl_dir",
                "image_dir",
                "image_keys",
                "image_prompts",
            )
        }

    def _action_transform(self, action_horizon: int) -> BuildAction:
        return BuildAction(
            action_horizon=action_horizon,
            action_mode=self.action_mode,
        )

    def norm_stats_path(self, action_horizon: int) -> pathlib.Path:
        action_transform = self._action_transform(action_horizon)
        digest = hashlib.sha256(
            f"{self.dataset_name}|{action_transform}".encode()
        ).hexdigest()[:16]
        return pathlib.Path(self.norm_stats_root) / (
            f"{self.dataset_name}_{digest}.json"
        )

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
                    transform_pipeline=TrainingTransformPipeline(p=0.5),
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

    def compute_norm_stats(
        self,
        action_horizon: int,
        batch_size: int = 128,
        num_workers: int = 32,
    ) -> None:
        norm_keys = ["state", "action"]
        dataset_info = self._dataset_info()
        dataset = JsonlDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=Pipeline([self._action_transform(action_horizon)]),
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(dataset_info),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=NormStatsCollator(),
            persistent_workers=True,
            prefetch_factor=4,
        )
        stats = {key: normalize.RunningStats() for key in norm_keys}
        max_batches = self.compute_norm_stats_max_batches or len(dataloader)
        total = min(len(dataloader), max_batches)
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            total=total,
            desc=f"Computing norm stats [{self.dataset_name}]",
        ):
            if batch_idx >= max_batches:
                break
            for key in norm_keys:
                values = batch[key]
                stats[key].update(values.reshape(-1, values.shape[-1]))

        norm_stats = {}
        for key, stat in stats.items():
            stat = stat.get_statistics()
            norm_stats[key] = {
                name: getattr(stat, name).tolist()
                for name in ("mean", "std", "q01", "q99")
            }

        output_path = self.norm_stats_path(action_horizon)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"norm_stats": norm_stats}, indent=2))
        logger.info(f"Saved norm stats for {self.dataset_name} to {output_path}")


@dataclass
class DM05InferenceConfig(Config):
    port: int = field(default=7891)
    diffusion_steps: int = field(default=10)
    output_action_dim: int = field(default=14)
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )
    backend: Literal["default", "fast"] = field(default="default")
    vision_trt_engine_path: str | None = field(default=None)
    force_rebuild: bool = field(default=False)
    prefix_seq_len_buckets: list[int] | None = field(default=None)

    @staticmethod
    def _norm_vector_dim(stats: dict, field_name: str) -> int:
        for key in ("q01", "q99", "mean", "std", "min", "max"):
            values = stats.get(key)
            if values is not None:
                return int(np.asarray(values).size)
        raise ValueError(f"Cannot infer normalization dimension for {field_name!r}.")

    def _request_default_overrides(self) -> dict:
        return {}

    def _request_default(self, name: str, fallback=None):
        if hasattr(self, name):
            value = getattr(self, name)
        else:
            overrides = self._request_default_overrides()
            value = overrides.get(name, fallback)
        return fallback if value is None else value

    def _initialize(
        self,
        model: DM05ForConditionalGeneration,
        model_name_or_path: str | None,
        norm_stats_path: str,
        n_bins: int,
        model_max_length: int,
        use_absolute_action: bool,
        add_state: bool = True,
        is_history: bool = False,
    ) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_absolute_action = use_absolute_action
        self.last_model_latency_sec = None
        self.default_robot_type = self._request_default("default_robot_type")
        self.default_state_desc = self._request_default("default_state_desc")
        self.default_speed = str(self._request_default("default_speed", "0.5"))
        self.default_control_mode = self._request_default("default_control_mode")
        model.to(self.device)
        model.eval()
        self.model = model
        self.is_history = is_history
        self.fast_runtime = None
        transform_max_length = model_max_length
        if self.backend == "fast":
            from opendm.infer.dm05_infer import DM05FastInferRuntime
            from opendm.infer.dm05_trt_utils import ensure_dm05_fast_inference_engine

            dm05_model = unwrap_dm05_model(model)
            num_vision_images = len(self.image_prompts)
            if is_history:
                from opendm.infer.dm05_trt_utils import MAX_HISTORY_IMAGES

                num_vision_images += MAX_HISTORY_IMAGES
            engine_path = ensure_dm05_fast_inference_engine(
                checkpoint=model_name_or_path,
                engine_path=self.vision_trt_engine_path,
                num_images=num_vision_images,
                force_rebuild=self.force_rebuild,
            )
            self.vision_trt_engine_path = str(engine_path)
            self.fast_runtime = DM05FastInferRuntime(
                dm05_model,
                vision_trt_engine_path=engine_path,
                prefix_seq_len_buckets=self.prefix_seq_len_buckets,
                diffusion_steps=self.diffusion_steps,
                is_history=is_history,
            )
            vlm_model = dm05_model.model.vlm.model
            self.fast_image_token_id = int(vlm_model.config.image_token_id)
            self.fast_mm_tokens_per_image = int(vlm_model.config.mm_tokens_per_image)
            transform_max_length = min(
                transform_max_length, self.fast_runtime.prefix_len
            )

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")

        input_normalize = Normalize(
            norm_stats_path=norm_stats_path,
            norm_keys=["state"],
            use_quantiles=True,
        )
        output_denormalize = Denormalize(
            norm_stats_path=norm_stats_path,
            norm_keys=["action"],
            use_quantiles=True,
        )
        self.expected_state_dim = self._norm_vector_dim(
            input_normalize.norm_stats["state"],
            "state",
        )
        self.expected_action_dim = self._norm_vector_dim(
            output_denormalize.norm_stats["action"],
            "action",
        )
        if self.output_action_dim <= 0:
            raise ValueError(
                f"inference output_action_dim must be positive, got {self.output_action_dim}."
            )
        if self.output_action_dim != self.expected_action_dim:
            raise ValueError(
                "inference output_action_dim must match action normalization "
                f"dimension, got output_action_dim={self.output_action_dim}, "
                f"action_dim={self.expected_action_dim}."
            )
        if (
            self.use_absolute_action
            and self.expected_state_dim != self.expected_action_dim
        ):
            raise ValueError(
                "Absolute action output requires matching state and action "
                f"dimensions, got state_dim={self.expected_state_dim}, "
                f"action_dim={self.expected_action_dim}."
            )

        self.input_transform = Pipeline(
            [
                PixelTransform(
                    transform_pipeline=NoAugmentationPipeline(),
                ),
                input_normalize,
                ChatTokenization(
                    processor=self.processor,
                    n_bins=n_bins,
                    max_length=transform_max_length,
                    image_prompts=self.image_prompts,
                    add_state=add_state,
                    is_history=is_history,
                    enable_logging=False,
                ),
                ToDevice(device=self.device),
            ]
        )
        self.output_transform = Pipeline(
            [
                output_denormalize,
                *([ActionAbsolute()] if use_absolute_action else []),
            ]
        )

    def _supported_robot_types(self) -> list[str]:
        supported = [robot_type.value for robot_type in ROBOT_STATE_DESCS]
        default_robot_type = self._request_default("default_robot_type")
        if default_robot_type is not None and default_robot_type not in supported:
            supported.append(default_robot_type)
        return supported

    def _load_image(self, source, field: str) -> Image.Image:
        try:
            if isinstance(source, Image.Image):
                return source.convert("RGB")
            if hasattr(source, "read"):
                return Image.open(source).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"{field} is not a valid image") from exc
        raise ValueError(f"{field} must be an uploaded file or PIL image")

    def _decode_b64_image(self, value, field: str) -> Image.Image:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a base64 string")
        if value.startswith("data:"):
            if "," not in value:
                raise ValueError(f"{field} data URL is missing base64 payload")
            value = value.split(",", 1)[1]

        try:
            raw = base64.b64decode(value, validate=True)
            return Image.open(BytesIO(raw)).convert("RGB")
        except (binascii.Error, OSError, ValueError) as exc:
            raise ValueError(f"{field} is not a valid base64 image") from exc

    def _parse_state(self, states) -> np.ndarray:
        if states is None:
            raise ValueError("observation.state is required")
        if isinstance(states, str):
            try:
                states = json.loads(states)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "observation.state must be a valid JSON array"
                ) from exc
        if not isinstance(states, list):
            raise ValueError("observation.state must be a JSON array")

        try:
            state = np.asarray(states, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation.state must contain numeric values") from exc
        if state.ndim != 1:
            raise ValueError("observation.state must be a 1D JSON array")
        if state.size == 0:
            raise ValueError("observation.state must not be empty")
        return state

    def _resolve_state_desc(self, robot_type: str | None):
        default_robot_type = self._request_default("default_robot_type")
        default_state_desc = self._request_default("default_state_desc")
        if robot_type is None:
            return default_state_desc

        try:
            robot_type_enum = RobotType(robot_type)
        except ValueError as exc:
            if robot_type == default_robot_type and default_state_desc is not None:
                return default_state_desc
            raise ValueError(
                f"Unsupported robot_type {robot_type!r}. Supported: {self._supported_robot_types()}"
            ) from exc

        if robot_type_enum in ROBOT_STATE_DESCS:
            return ROBOT_STATE_DESCS[robot_type_enum]
        if robot_type == default_robot_type and default_state_desc is not None:
            return default_state_desc
        raise ValueError(
            f"Unsupported robot_type {robot_type!r}. Supported: {self._supported_robot_types()}"
        )

    def _prepare_model_input(
        self,
        *,
        text: str,
        images: list,
        states,
        robot_type: str | None,
        speed: str | None = None,
        control_mode: str | None = None,
        history_images: list | None = None,
    ) -> dict:
        if len(images) != len(self.image_prompts):
            raise ValueError(
                f"Expected {len(self.image_prompts)} images for "
                f"image_prompts={self.image_prompts!r}, got {len(images)}."
            )

        robot_type = (
            self._request_default("default_robot_type")
            if robot_type is None
            else str(robot_type)
        )
        speed = str(
            self._request_default("default_speed", "0.5") if speed is None else speed
        )
        default_control_mode = self._request_default("default_control_mode")
        if control_mode is None:
            control_mode = default_control_mode
        elif control_mode is not None:
            control_mode = str(control_mode)
        state_desc = self._resolve_state_desc(robot_type)
        if self.use_absolute_action and state_desc is None:
            raise ValueError(
                "observation.robot_type is required unless the selected dataset "
                "registration provides state_desc for absolute-action reconstruction."
            )

        if history_images and not self.is_history:
            raise ValueError(
                "history_images were provided but the service was started without "
                "--data-config.is-history."
            )

        pil_images = [
            self._load_image(img, f"image[{i}]") for i, img in enumerate(images)
        ]
        pil_history_images = [
            self._load_image(img, f"history_images[{i}]")
            for i, img in enumerate(history_images or [])
        ]
        state = self._parse_state(states)

        return {
            "images": pil_images,
            "history_images": pil_history_images,
            "prompt": "" if text is None else str(text),
            "state": state,
            "meta_data": {
                "robot_type": robot_type,
                "speed": speed,
                "control_mode": control_mode,
                "state_desc": state_desc,
            },
        }

    def _predict(self, data: dict) -> np.ndarray:
        self.last_model_latency_sec = None
        state = data["state"]
        if state.shape[0] != self.expected_state_dim:
            raise BadRequest(
                "State dimension does not match normalization stats: "
                f"expected {self.expected_state_dim}, got {state.shape[0]}."
            )

        model_input = self.input_transform(data)
        inference_seed = os.environ.get("DM05_INFERENCE_SEED")
        if inference_seed is not None:
            seed = int(inference_seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        prefix_len = int(model_input["input_ids"].shape[1])
        if self.backend == "fast" and prefix_len > self.fast_runtime.prefix_len:
            raise BadRequest(
                f"Prefix length {prefix_len} exceeds the fast backend limit of "
                f"{self.fast_runtime.prefix_len} tokens. Shorten the prompt or "
                "reduce model_max_length."
            )

        dm05_model = unwrap_dm05_model(self.model)
        action_dim = dm05_model.model.config.action_dim
        action_mask = torch.zeros(
            1,
            1,
            action_dim,
            device=self.device,
            dtype=dm05_model.model.action_in_proj.weight.dtype,
        )
        action_mask[..., : self.output_action_dim] = 1.0
        inference_kwargs = {
            "input_ids": model_input["input_ids"],
            "attention_mask": model_input["attention_mask"],
            "pixel_values": model_input["pixel_values"],
            "token_type_ids": model_input["token_type_ids"],
            "diffusion_steps": self.diffusion_steps,
            "action_mask": action_mask,
            "history_pixel_values": model_input.get("history_pixel_values"),
            "history_mask": model_input.get("history_mask"),
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_t0 = time.perf_counter()
        if self.backend == "fast":
            actions = self.fast_runtime.inference_action(
                **inference_kwargs,
            )
        else:
            actions = self.model.inference_action(**inference_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_model_latency_sec = time.perf_counter() - model_t0

        if int(actions.shape[-1]) < self.output_action_dim:
            raise RuntimeError(
                "Model action dimension is smaller than configured output_action_dim: "
                f"model returned {int(actions.shape[-1])}, "
                f"configured {self.output_action_dim}."
            )

        raw_actions = (
            actions.to(torch.float32)
            .detach()
            .cpu()
            .numpy()[0, :, : self.output_action_dim]
        )

        outputs = {
            "action": raw_actions,
            "state": state,
            "meta_data": data["meta_data"],
        }
        outputs = self.output_transform(outputs)

        return outputs["action"]

    def _apply_v1_sampling(self, sampling) -> None:
        if sampling is None:
            return
        if not isinstance(sampling, dict):
            raise ValueError("sampling must be a JSON object")

        num_steps = sampling.get("num_steps")
        if num_steps is not None and int(num_steps) != self.diffusion_steps:
            raise ValueError(
                f"Inference uses fixed num_steps={self.diffusion_steps}, got {num_steps}."
            )

        seed = sampling.get("seed")
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _parse_image_slot(self, slot) -> int:
        try:
            slot = int(slot)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"image slot must be a 1-based numeric string, got {slot!r}"
            ) from exc

        if slot < 1:
            raise ValueError(f"image slot must be 1-based and positive, got {slot}")
        return slot

    def _prepare_input(self, body: dict) -> dict:
        observation = body.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("observation must be a JSON object")

        images_raw = observation.get("images")
        if not isinstance(images_raw, dict):
            raise ValueError("observation.images must be a JSON object")

        images = []
        sorted_images = sorted(
            images_raw.items(), key=lambda item: self._parse_image_slot(item[0])
        )
        for expected_slot, (slot, image_b64) in enumerate(sorted_images, start=1):
            parsed_slot = self._parse_image_slot(slot)
            if parsed_slot != expected_slot:
                raise ValueError(
                    "observation.images slots must be contiguous 1-based "
                    f"strings; expected {expected_slot}, got {slot!r}"
                )
            images.append(self._decode_b64_image(image_b64, f"images.{slot}"))

        history_images_raw = observation.get("history_images")
        history_images = []
        if history_images_raw is not None:
            if not isinstance(history_images_raw, list):
                raise ValueError(
                    "observation.history_images must be a JSON array of base64 images"
                )
            history_images = [
                self._decode_b64_image(image_b64, f"history_images[{i}]")
                for i, image_b64 in enumerate(history_images_raw)
            ]

        return self._prepare_model_input(
            text=observation.get("prompt", ""),
            images=images,
            states=observation.get("state"),
            robot_type=observation.get("robot_type"),
            speed=observation.get("speed"),
            control_mode=observation.get("control_mode"),
            history_images=history_images,
        )

    def _prepare_input_legacy(self) -> dict:
        states = request.form.get("states")
        if states is None:
            raise BadRequest("Form field 'states' is required.")

        return self._prepare_model_input(
            text=request.form.get("text", ""),
            images=request.files.getlist("image"),
            states=states,
            robot_type=request.form.get("robot_type") or None,
            speed=request.form.get("speed") or None,
            control_mode=request.form.get("control_mode") or None,
            history_images=request.files.getlist("history_images"),
        )

    def _infer(self):
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")

            self._apply_v1_sampling(body.get("sampling"))
            t0 = time.monotonic()
            data = self._prepare_input(body)
            actions = self._predict(data)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            logger.info(
                "v1/infer processing time: {:.3f}s, model_latency_ms={}",
                latency_ms / 1000.0,
                None
                if self.last_model_latency_sec is None
                else round(self.last_model_latency_sec * 1000.0, 3),
            )
            return jsonify(
                {
                    "actions": actions.tolist(),
                    "metadata": {"latency_ms": latency_ms},
                }
            )
        except (BadRequest, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = exc.description if isinstance(exc, BadRequest) else str(exc)
            logger.warning(f"v1/infer bad request: {message}")
            return jsonify({"error": message}), 400

    def _infer_legacy(self):
        try:
            t0 = time.monotonic()
            try:
                data = self._prepare_input_legacy()
            except (AssertionError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise BadRequest(str(exc)) from exc

            actions = self._predict(data)
            api_latency_sec = time.monotonic() - t0
            logger.info(
                "process_frame processing time: {:.3f}s, model_latency_ms={}",
                api_latency_sec,
                None
                if self.last_model_latency_sec is None
                else round(self.last_model_latency_sec * 1000.0, 3),
            )
            return jsonify(
                {
                    "response": actions.tolist(),
                    "model_latency_ms": (
                        None
                        if self.last_model_latency_sec is None
                        else round(self.last_model_latency_sec * 1000.0, 3)
                    ),
                }
            )
        except BadRequest as exc:
            message = exc.description
            logger.warning(f"process_frame bad request: {message}")
            return jsonify({"error": message}), 400


@dataclass
class DM05Exp(Config):
    use_lora: bool | None = field(default=False)
    task: Literal["train", "inference"] = field(default="train")
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)
    logger_level: str = field(default="INFO")

    def __post_init__(self):
        logger.remove()
        logger.add(sys.stdout, level=self.logger_level)

    def _ensure_norm_stats(self) -> None:
        norm_stats_path = self.data_config.norm_stats_path(self.model_config.chunk_size)
        if norm_stats_path.exists():
            logger.info(f"Using norm stats from {norm_stats_path}")
            return

        if self.local_rank == 0:
            self.data_config.compute_norm_stats(
                action_horizon=self.model_config.chunk_size
            )
        else:
            while not norm_stats_path.exists():
                time.sleep(5)
        assert norm_stats_path.exists()

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)

        self._ensure_norm_stats()

        model = self.model_config.build_model(use_lora=self.use_lora)
        self.model = model
        self.optimizer_config.prepare_model(model)
        dm05_model = unwrap_dm05_model(self.model)
        dm05_model.config.use_cache = False
        dm05_model.model.vlm.config.use_cache = False

        processor = AutoProcessor.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=True,
        )
        self.processor = processor
        self.tokenizer = processor.tokenizer

        train_dataset, data_collator = self.data_config.build_dataset(
            processor=processor,
            action_horizon=self.model_config.chunk_size,
            tokenizer_max_length=self.trainer_config.model_max_length,
        )

        trainer_kwargs = {
            "model": self.model,
            "processing_class": self.processor,
            "exp_config": self,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
        }
        trainer = DMTrainer(**trainer_kwargs)
        self.trainer = trainer

    def train(self):
        self._initialize_train()

        if list(pathlib.Path(self.trainer_config.output_dir).glob("checkpoint-*")):
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()

        self.trainer.save_state()
        dm05_model = unwrap_dm05_model(self.model)
        dm05_model.config.use_cache = True
        dm05_model.model.vlm.config.use_cache = True
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        using_lora = self.use_lora
        if using_lora and world_size > 1:
            logger.info(
                "Skipping final top-level PEFT save for LoRA/FSDP; "
                "step checkpoints remain the canonical eval artifacts."
            )
        else:
            safe_save_model_for_hf_trainer(
                trainer=self.trainer,
                output_dir=self.trainer_config.output_dir,
            )
        logger.info(
            f"Training completed and model saved to {self.trainer_config.output_dir}"
        )

    def _initialize_inference_runtime(self) -> None:
        """Build the model and transforms shared by inference entry points."""
        use_fast_backend = self.inference_config.backend == "fast"
        if use_fast_backend or self.inference_config.prefix_seq_len_buckets:
            from opendm.infer.dm05_infer import DM05FastInferRuntime

            DM05FastInferRuntime.resolve_prefix_seq_len_buckets(
                self.inference_config.prefix_seq_len_buckets,
                fast_backend=use_fast_backend,
            )
        self.model_config.llm_attn_implementation = (
            "flex_attention" if use_fast_backend else "eager"
        )
        logger.info(f"Loading model from {self.model_config.model_name_or_path}")
        model = self.model_config.build_model(use_lora=self.use_lora)
        ckpt_norm_stats_path = (
            pathlib.Path(self.model_config.model_name_or_path) / "norm_stats.json"
        )
        norm_stats_path = (
            ckpt_norm_stats_path
            if ckpt_norm_stats_path.exists()
            else self.data_config.norm_stats_path(self.model_config.chunk_size)
        )
        if ckpt_norm_stats_path.exists():
            logger.info(f"Using normalization stats from checkpoint: {norm_stats_path}")
        else:
            logger.warning(
                "Normalization stats not found in checkpoint at "
                f"{ckpt_norm_stats_path}; falling back to {norm_stats_path}"
            )
        dataset_info = self.data_config._dataset_info()
        default_robot_type = dataset_info.get("robot_type")
        if isinstance(default_robot_type, Enum):
            default_robot_type = default_robot_type.value
        self.inference_config.default_robot_type = default_robot_type
        self.inference_config.default_state_desc = dataset_info.get("state_desc")
        if dataset_info.get("speed") is not None:
            self.inference_config.default_speed = str(dataset_info["speed"])
        if "control_mode" in dataset_info:
            self.inference_config.default_control_mode = dataset_info["control_mode"]
        self.inference_config._initialize(
            model=model,
            model_name_or_path=self.model_config.model_name_or_path,
            norm_stats_path=str(norm_stats_path),
            n_bins=self.data_config.n_bins,
            model_max_length=self.trainer_config.model_max_length,
            use_absolute_action=(self.data_config.action_mode == ActionMode.RELATIVE),
            add_state=self.data_config.add_state,
            is_history=self.data_config.is_history,
        )

    def inference(self) -> None:
        self._initialize_inference_runtime()
        app = Flask(__name__)
        app.add_url_rule(
            "/process_frame",
            "process_frame",
            self.inference_config._infer_legacy,
            methods=["POST"],
        )
        app.add_url_rule(
            "/v1/infer",
            "v1_infer",
            self.inference_config._infer,
            methods=["POST"],
        )
        app.run(
            host="0.0.0.0",
            port=self.inference_config.port,
            debug=False,
            threaded=False,
        )


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")
