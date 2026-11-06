import base64
import binascii
import copy
import hashlib
import json
import os
import pathlib
import sys
import time
from contextlib import nullcontext
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor
from werkzeug.exceptions import BadRequest

import opendm.data.normalize as normalize
from opendm.constants.precision import (
    BF16_MIXED_PRECISION_POLICY,
    FP32_MIXED_PRECISION_POLICY,
    MODEL_DTYPE,
)
from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
from opendm.data.augmentations import NoAugmentationPipeline, TrainingTransformPipeline
from opendm.data.dataset import JsonlDataset
from opendm.data.transforms import (
    ActionAbsolute,
    ArrangeState,
    BuildAction,
    ChatTokenization,
    Denormalize,
    LoadHistory,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
    ToDevice,
    VLMChatTokenization,
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
from opendm.model.dm05.dm05_utils import (
    configure_fp32_precision_runtime,
    dm05_inference_context,
)


@dataclass
class Config:
    pass


@dataclass
class DM05ModelConfig(Config):
    model_name_or_path: str | None = field(default="./checkpoints/DM05")
    chunk_size: int = field(default=50)
    num_flow_samples: int = field(default=1)
    precision_policy: Literal["bf16_mixed", "fp32_mixed"] = field(
        default=BF16_MIXED_PRECISION_POLICY
    )
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
    align_action_rope_to_language: bool = field(default=False)
    sync_vlm_config_dtype_to_load_dtype: bool = field(default=False)
    lora_config: DM05LoraConfig = field(default_factory=DM05LoraConfig)

    def _config_overrides(self) -> dict:
        if type(self.num_flow_samples) is not int or self.num_flow_samples < 1:
            raise ValueError("num_flow_samples must be a positive integer")
        return {
            "chunk_size": self.chunk_size,
            "num_flow_samples": self.num_flow_samples,
        }

    def _torch_dtype(self) -> torch.dtype:
        if self.precision_policy == BF16_MIXED_PRECISION_POLICY:
            return torch.bfloat16 if self.bf16 else torch.float32
        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            return MODEL_DTYPE
        raise ValueError(f"Invalid precision_policy: {self.precision_policy!r}")

    def _apply_attention_config_before_load(self, config: DM05Config) -> None:
        vlm_attention = config.vlm_config._attn_implementation
        if isinstance(vlm_attention, dict):
            vlm_attention = dict(vlm_attention)
        elif vlm_attention is None:
            vlm_attention = {}
        else:
            vlm_attention = {"": vlm_attention}

        if self.llm_attn_implementation != "auto":
            vlm_attention[""] = self.llm_attn_implementation
            vlm_attention["text_config"] = self.llm_attn_implementation
        if self.vision_attn_implementation != "auto":
            vlm_attention["vision_config"] = self.vision_attn_implementation
        config.vlm_config._attn_implementation = vlm_attention

        if self.action_attn_implementation != "auto":
            config.action_config._attn_implementation = self.action_attn_implementation
        logger.info(
            "Configured attention before model construction: "
            f"LLM={self.llm_attn_implementation}, "
            f"Vision={self.vision_attn_implementation}, "
            f"Action={self.action_attn_implementation}"
        )

    def _apply_load_dtype_to_vlm_config(self, config: DM05Config) -> None:
        if not self.sync_vlm_config_dtype_to_load_dtype:
            return

        load_dtype = self._torch_dtype()
        config.vlm_config.dtype = load_dtype
        config.vlm_config.text_config.dtype = load_dtype
        config.vlm_config.vision_config.dtype = load_dtype
        logger.info(
            "Synchronized VLM config dtype to checkpoint load dtype: "
            f"VLM={load_dtype}, Text={load_dtype}, Vision={load_dtype}"
        )

    def _load_base_checkpoint_model(self) -> DM05ForConditionalGeneration:
        config = DM05Config.from_pretrained(self.model_name_or_path)
        self._apply_attention_config_before_load(config)
        self._apply_load_dtype_to_vlm_config(config)
        if self.align_action_rope_to_language:
            config.action_config.rope_parameters = copy.deepcopy(
                config.vlm_config.text_config.rope_parameters
            )
            logger.warning(
                "Aligning action expert RoPE parameters to the VLM language "
                "configuration for chat-only inference; the action expert does "
                "not participate in this inference path."
            )
        for attr, value in self._config_overrides().items():
            setattr(config, attr, value)
        return DM05ForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            config=config,
            dtype=self._torch_dtype(),
        )

    def _load_adapter_checkpoint_model(self) -> torch.nn.Module:
        return load_dm05_model_for_inference(
            self.model_name_or_path,
            config_overrides=self._config_overrides(),
            dtype=self._torch_dtype(),
            trust_remote_code=True,
        )

    def _set_precision_policy(self, model: torch.nn.Module) -> None:
        dm05_model = unwrap_dm05_model(model)
        dm05_model.set_precision_policy(self.precision_policy)

    def _apply_precision_policy(self, model: torch.nn.Module) -> None:
        if self.precision_policy == BF16_MIXED_PRECISION_POLICY:
            return

        if self.precision_policy == FP32_MIXED_PRECISION_POLICY:
            model.to(dtype=MODEL_DTYPE)
            configure_fp32_precision_runtime()
            return

        raise ValueError(f"Invalid precision_policy: {self.precision_policy!r}")

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
            self._set_precision_policy(model)
            self._apply_runtime_model_options(model)
            self._apply_precision_policy(model)
            return model

        model = self._load_base_checkpoint_model()
        self._set_precision_policy(model)
        self._apply_runtime_model_options(model)
        self._apply_full_model_runtime_optimizations(model)

        if not use_lora:
            self._apply_precision_policy(model)
            return model
        model = self._wrap_new_lora_for_training(model)
        self._set_precision_policy(model)
        self._apply_precision_policy(model)
        return model


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
        from opendm.optimizer.muon_adamw import is_default_muon_parameter

        return is_default_muon_parameter(name, param)

    def prepare_model(self, model: torch.nn.Module) -> None:
        if self.optim == "muon_adamw":
            from opendm.optimizer import mark_muon_parameters

            mark_muon_parameters(model, predicate=self.should_use_muon)

    def build_muon_adamw(self, model: torch.nn.Module) -> "MuonAdamW":
        from opendm.optimizer import MuonAdamW

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
    jsonl_dir: str | None = field(default=None)
    image_dir: str | None = field(default=None)
    n_bins: int = field(default=256)
    action_mode: ActionMode = field(default=ActionMode.RELATIVE)
    compute_norm_stats_max_batches: int | None = field(default=None)
    norm_stats_root: str = field(default="./norm_stats")
    norm_stats_default_robot_type: str | None = field(default=None)
    add_state: bool = field(default=True)
    is_history: bool = field(default=False)

    def _dataset_info(self) -> dict:
        assert self.dataset_name in CONVERSATION_DATA
        dataset_info = CONVERSATION_DATA[self.dataset_name]
        dataset_info = {
            k: val.value if isinstance(val, Enum) else val
            for k, val in dataset_info.items()
        }
        if self.jsonl_dir is not None:
            dataset_info["jsonl_dir"] = self.jsonl_dir
        if self.image_dir is not None:
            dataset_info["image_dir"] = self.image_dir
        return dataset_info

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
        explicit_path = pathlib.Path(self.norm_stats_root) / "norm_stats.json"
        if explicit_path.is_file():
            return explicit_path

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
        pipeline_steps = [
            self._action_transform(action_horizon),
            LoadImages(image_keys=image_keys, image_dir=dataset_info["image_dir"]),
        ]
        if self.is_history:
            pipeline_steps.append(
                LoadHistory(
                    image_key=image_keys[0],
                    image_dir=dataset_info["image_dir"],
                )
            )
        pipeline_steps.extend(
            [
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
                    is_history=self.is_history,
                ),
                PadAction(32),
            ]
        )
        pipeline = Pipeline(pipeline_steps)
        dataset = JsonlDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=pipeline,
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(dataset_info),
        )

        from opendm.data.collator import TrainingCollator

        collator = TrainingCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            max_length=tokenizer_max_length,
        )

        return dataset, collator

    def build_norm_stats_dataset(self, action_horizon: int) -> Dataset:
        """Build the experiment data stream used to compute normalization stats."""
        dataset_info = self._dataset_info()
        return JsonlDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=Pipeline([self._action_transform(action_horizon)]),
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(dataset_info),
        )

    def compute_norm_stats(
        self,
        action_horizon: int,
        batch_size: int = 128,
        num_workers: int = 32,
    ) -> None:
        from opendm.data.collator import NormStatsCollator

        dataset = self.build_norm_stats_dataset(action_horizon)
        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "collate_fn": NormStatsCollator(),
        }
        if num_workers > 0:
            dataloader_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 4,
                }
            )
        dataloader = DataLoader(
            dataset,
            **dataloader_kwargs,
        )
        stats_by_robot: dict[
            str | None,
            dict[str, normalize.RunningStats],
        ] = {}
        keys_by_robot: dict[str | None, set[str]] = {}
        max_batches = self.compute_norm_stats_max_batches or len(dataloader)
        total = min(len(dataloader), max_batches)
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            total=total,
            desc=f"Computing norm stats [{self.dataset_name}]",
        ):
            if batch_idx >= max_batches:
                break
            for robot_type, robot_batch in batch["robot_batches"].items():
                batch_keys = set(robot_batch)
                expected_keys = keys_by_robot.setdefault(robot_type, batch_keys)
                if batch_keys != expected_keys:
                    raise ValueError(
                        f"Inconsistent norm fields for robot_type {robot_type!r}: "
                        f"expected {sorted(expected_keys)}, got {sorted(batch_keys)}"
                    )
                robot_stats = stats_by_robot.setdefault(
                    robot_type,
                    {key: normalize.RunningStats() for key in sorted(expected_keys)},
                )
                for key, values in robot_batch.items():
                    robot_stats[key].update(values.reshape(-1, values.shape[-1]))

        if not stats_by_robot:
            raise ValueError("Cannot compute norm stats from an empty dataset")
        if None in stats_by_robot and len(stats_by_robot) > 1:
            raise ValueError(
                "Cannot compute norm stats from both typed and untyped robot samples"
            )

        computed_stats_by_robot: dict[
            str | None,
            dict[str, normalize.NormStats],
        ] = {}
        for robot_type, robot_stats in stats_by_robot.items():
            computed_stats = {}
            for key, running_stats in robot_stats.items():
                result = running_stats.get_statistics()
                computed_stats[key] = result.model_copy(
                    update={"min": None, "max": None}
                )
            computed_stats_by_robot[robot_type] = computed_stats

        if None in computed_stats_by_robot:
            output = normalize.serialize_json(
                computed_stats_by_robot[None],
                exclude_none=True,
            )
        else:
            typed_stats = {
                robot_type: stats
                for robot_type, stats in computed_stats_by_robot.items()
                if robot_type is not None
            }
            default_robot_type = self.norm_stats_default_robot_type
            if default_robot_type is None:
                if len(typed_stats) > 1:
                    raise ValueError(
                        "norm_stats_default_robot_type is required when an "
                        "experiment contains multiple robot types"
                    )
                default_robot_type = next(iter(typed_stats))
            output = normalize.serialize_norm_stats_file(
                typed_stats,
                default_robot_type=default_robot_type,
            )

        output_path = self.norm_stats_path(action_horizon)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
        logger.info(f"Saved norm stats for {self.dataset_name} to {output_path}")


@dataclass
class DM05InferenceConfig(Config):
    port: int = field(default=7891)
    diffusion_steps: int = field(default=10)
    max_history_images: int = field(default=5)
    output_action_dim: int = field(default=14)
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )
    backend: Literal["default", "fast"] = field(default="default")
    vision_trt_engine_path: str | None = field(default=None)
    force_rebuild: bool = field(default=False)
    prefix_seq_len_buckets: list[int] | None = field(default=None)
    chat_only: bool = field(default=False)
    enable_vlm_chat: bool = field(default=True)
    chat_max_new_tokens: int = field(default=2048)
    chat_max_input_tokens: int = field(default=8192)
    clip_to_bounds: bool = field(default=True)
    compose_eef_rot: bool = field(default=False)
    full_action_mask: bool = field(default=False)
    use_transformed_state: bool = field(default=False)
    state_layout: str = field(default="identity")
    denorm_state: bool = field(default=False)
    prefix_mlp_fp8_engine_dir: str | None = field(default=None)
    auto_prepared_fast_artifacts: bool = field(default=True)
    prepared_fast_artifacts_dir: str | None = field(default=None)

    @staticmethod
    def _norm_vector_dim(stats: normalize.NormStats, field_name: str) -> int:
        for key in ("q01", "q99", "mean", "std", "min", "max"):
            values = getattr(stats, key)
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

        if self.chat_only:
            model.eval()
            model.to(self.device)
            self.model = model
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            self.vlm_chat_tokenization = VLMChatTokenization(self.processor)
            logger.info("Model loaded successfully")
            return

        self.use_absolute_action = use_absolute_action
        self.last_model_latency_sec = None
        self.default_robot_type = self._request_default("default_robot_type")
        self.default_state_desc = self._request_default("default_state_desc")
        self.default_speed = str(self._request_default("default_speed", "0.5"))
        self.default_control_mode = self._request_default("default_control_mode")
        model.eval()
        self.model = model
        self.is_history = is_history
        self.fast_runtime = None
        if self.prefix_mlp_fp8_engine_dir is not None and self.backend != "fast":
            raise ValueError("prefix_mlp_fp8_engine_dir requires backend='fast'.")
        transform_max_length = model_max_length
        if self.backend == "fast":
            if self.enable_vlm_chat:
                logger.warning(
                    "VLM chat uses the standard PyTorch generation path; "
                    "the fast backend only accelerates action inference."
                )

            from opendm.infer.dm05_infer import DM05FastInferRuntime
            from opendm.infer.dm05_trt_utils import ensure_dm05_fast_inference_engine

            dm05_model = unwrap_dm05_model(model)
            max_history_images = int(self.max_history_images)
            fast_prefix_len = int(getattr(self, "fast_prefix_len", None) or 1024)
            num_vision_images = len(self.image_prompts)
            if is_history:
                num_vision_images += max_history_images
            if getattr(self, "_prepared_artifacts_selected", False):
                from opendm.infer.prefix_mlp_fp8_contract import (
                    prefix_mlp_model_contract,
                    prefix_source_model_sha256,
                    validate_bundle_manifest,
                )

                try:
                    manifest = json.loads(
                        (
                            pathlib.Path(self.prefix_mlp_fp8_engine_dir)
                            / "prefix_mlp_quant_manifest.json"
                        ).read_text()
                    )
                    vlm = dm05_model.model.vlm
                    validate_bundle_manifest(
                        manifest,
                        prefix_mlp_model_contract(vlm.model.language_model),
                        tuple(self.prefix_seq_len_buckets),
                        prefix_source_sha256=prefix_source_model_sha256(vlm),
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Prepared FP8 artifacts do not match the loaded checkpoint; "
                        "using the normal BF16 fast path: {}",
                        exc,
                    )
                    self._prepared_artifacts_selected = False
                    self.vision_trt_engine_path = None
                    self.prefix_mlp_fp8_engine_dir = None
            if getattr(self, "_prepared_artifacts_selected", False):
                engine_path = pathlib.Path(self.vision_trt_engine_path)
            else:
                engine_path = ensure_dm05_fast_inference_engine(
                    checkpoint=model_name_or_path,
                    engine_path=self.vision_trt_engine_path,
                    num_images=num_vision_images,
                    force_rebuild=self.force_rebuild,
                )
            self.vision_trt_engine_path = str(engine_path)
            model.to(self.device)
            self.fast_runtime = DM05FastInferRuntime(
                dm05_model,
                vision_trt_engine_path=engine_path,
                prefix_seq_len_buckets=self.prefix_seq_len_buckets,
                diffusion_steps=self.diffusion_steps,
                is_history=is_history,
                max_history_images=max_history_images,
                prefix_len=fast_prefix_len,
                num_current_images=len(self.image_prompts),
                prefix_mlp_fp8_engine_dir=self.prefix_mlp_fp8_engine_dir,
            )
            vlm_model = dm05_model.model.vlm.model
            self.fast_image_token_id = int(vlm_model.config.image_token_id)
            self.fast_mm_tokens_per_image = int(vlm_model.config.mm_tokens_per_image)
            transform_max_length = min(
                transform_max_length, self.fast_runtime.prefix_len
            )
        else:
            model.to(self.device)

        self._initialize_preprocessing(
            model_name_or_path=model_name_or_path,
            norm_stats_path=norm_stats_path,
            n_bins=n_bins,
            transform_max_length=transform_max_length,
            use_absolute_action=use_absolute_action,
            add_state=add_state,
            is_history=is_history,
        )

    def _initialize_preprocessing(
        self,
        *,
        model_name_or_path: str | None,
        norm_stats_path: str,
        n_bins: int,
        transform_max_length: int,
        use_absolute_action: bool,
        add_state: bool,
        is_history: bool,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the deployment transforms without loading a model."""
        if device is not None:
            self.device = torch.device(device)
        elif not hasattr(self, "device"):
            self.device = torch.device("cpu")
        self.use_absolute_action = use_absolute_action
        self.last_model_latency_sec = None
        self.is_history = is_history
        self.default_robot_type = self._request_default("default_robot_type")
        self.default_state_desc = self._request_default("default_state_desc")
        self.default_speed = str(self._request_default("default_speed", "0.5"))
        self.default_control_mode = self._request_default("default_control_mode")
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.vlm_chat_tokenization = VLMChatTokenization(self.processor)
        logger.info("Inference preprocessing initialized successfully")

        self.norm_stats_file = normalize.load_norm_stats_file(norm_stats_path)
        if self.default_robot_type is None and self.norm_stats_file.is_multi_robot:
            self.default_robot_type = self.norm_stats_file.default_robot_type

        input_normalize = Normalize(
            norm_stats_path=norm_stats_path,
            norm_keys=["state"],
            use_quantiles=True,
            norm_stats_file=self.norm_stats_file,
            clip_to_bounds=self.clip_to_bounds,
        )
        output_denormalize = Denormalize(
            norm_stats_path=norm_stats_path,
            norm_keys=["action", "state"] if self.denorm_state else ["action"],
            use_quantiles=True,
            norm_stats_file=self.norm_stats_file,
        )
        default_profile = self.norm_stats_file.select(self.default_robot_type)
        self.expected_action_dim = self._norm_vector_dim(
            default_profile["action"],
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
        if self.norm_stats_file.norm_stats_by_robot is not None:
            profile_action_dims = {
                robot_type: self._norm_vector_dim(profile["action"], "action")
                for robot_type, profile in self.norm_stats_file.norm_stats_by_robot.items()
            }
            incompatible_profiles = {
                robot_type: action_dim
                for robot_type, action_dim in profile_action_dims.items()
                if action_dim != self.output_action_dim
            }
            if incompatible_profiles:
                raise ValueError(
                    "All robot profiles served by one inference process must use "
                    f"output_action_dim={self.output_action_dim}; incompatible "
                    f"profiles: {incompatible_profiles}"
                )

        input_steps = []
        if self.compose_eef_rot:
            input_steps.append(ArrangeState())
        input_steps.extend(
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
                    max_history_images=int(self.max_history_images),
                    enable_logging=False,
                ),
                ToDevice(device=self.device),
            ]
        )
        self.input_transform = Pipeline(input_steps)
        self.output_transform = Pipeline(
            [
                output_denormalize,
                *(
                    [ActionAbsolute(compose_eef_rot=self.compose_eef_rot)]
                    if use_absolute_action
                    else []
                ),
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
        if history_images:
            max_history_images = int(self.max_history_images)
            if max_history_images < 0:
                raise ValueError(
                    "inference max_history_images must be non-negative, got "
                    f"{max_history_images}."
                )
            if len(history_images) > max_history_images:
                raise ValueError(
                    f"At most {max_history_images} history images are supported, "
                    f"got {len(history_images)}."
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
        original_state = data["state"]
        model_input = self.input_transform(data)
        state = data["state"] if self.use_transformed_state else original_state
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
        dtype = dm05_model.model.action_in_proj.weight.dtype
        if self.full_action_mask:
            action_mask = torch.ones(1, 1, action_dim, device=self.device, dtype=dtype)
        else:
            action_mask = torch.zeros(1, 1, action_dim, device=self.device, dtype=dtype)
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
            with dm05_inference_context(self.device, dm05_model.precision_policy):
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

    def _decode_chat_messages(self, messages: list) -> list[dict]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty JSON array")

        decoded_messages = []
        for message_index, message in enumerate(messages):
            field = f"messages[{message_index}]"
            if not isinstance(message, dict):
                raise ValueError(f"{field} must be a JSON object")

            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"{field}.role must be one of system, user, or assistant"
                )

            content = message.get("content")
            if isinstance(content, str):
                content_parts = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                content_parts = content
            else:
                raise ValueError(f"{field}.content must be a string or JSON array")

            decoded_content = []
            for part_index, part in enumerate(content_parts):
                part_field = f"{field}.content[{part_index}]"
                if not isinstance(part, dict):
                    raise ValueError(f"{part_field} must be a JSON object")

                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError(f"{part_field}.text must be a string")
                    decoded_content.append({"type": "text", "text": text})
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    if not isinstance(image_url, dict):
                        raise ValueError(
                            f"{part_field}.image_url must be a JSON object"
                        )
                    url = image_url.get("url")
                    if not isinstance(url, str):
                        raise ValueError(f"{part_field}.image_url.url must be a string")
                    if url.lower().startswith(("http://", "https://")):
                        raise ValueError(
                            f"{part_field}.image_url.url only supports a data URL "
                            "or raw base64; external HTTP URLs are not supported"
                        )
                    decoded_content.append(
                        {
                            "type": "image",
                            "image": self._decode_b64_image(
                                url,
                                f"{part_field}.image_url.url",
                            ),
                        }
                    )
                elif part_type in {"video_url", "audio_url"}:
                    raise ValueError(f"{part_type} content is not supported")
                else:
                    raise ValueError(
                        f"{part_field}.type must be text or image_url; "
                        f"got {part_type!r}"
                    )

            decoded_messages.append({"role": role, "content": decoded_content})
        return decoded_messages

    def _health(self):
        return jsonify({"status": "ok"})

    def _chat(self):
        if not self.enable_vlm_chat:
            return jsonify({"error": "VLM chat is disabled on this server"}), 501

        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            if body.get("stream", False):
                raise ValueError("streaming responses are not supported")

            request_t0 = time.monotonic()
            messages = self._decode_chat_messages(body.get("messages"))
            model_input = self.vlm_chat_tokenization({"messages": messages})
            prompt_tokens = int(model_input["input_ids"].shape[1])
            if prompt_tokens > self.chat_max_input_tokens:
                raise ValueError(
                    f"input prompt has {prompt_tokens} tokens, exceeding the "
                    f"limit of {self.chat_max_input_tokens}"
                )

            requested_max_tokens = body.get("max_tokens")
            if requested_max_tokens is None:
                requested_max_tokens = body.get("max_completion_tokens", 1024)
            if (
                isinstance(requested_max_tokens, bool)
                or not isinstance(requested_max_tokens, (int, float))
                or int(requested_max_tokens) != requested_max_tokens
                or requested_max_tokens <= 0
            ):
                raise ValueError("max_tokens must be a positive integer")
            max_new_tokens = int(requested_max_tokens)
            if max_new_tokens > self.chat_max_new_tokens:
                logger.warning(
                    "Clamping chat max tokens from {} to {}",
                    max_new_tokens,
                    self.chat_max_new_tokens,
                )
                max_new_tokens = self.chat_max_new_tokens

            temperature = body.get("temperature", 0)
            if temperature is None:
                temperature = 0
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or temperature < 0
            ):
                raise ValueError("temperature must be a non-negative number")
            temperature = float(temperature)
            do_sample = temperature > 0

            top_p = None
            if do_sample:
                top_p = body.get("top_p", 0.95)
                if (
                    isinstance(top_p, bool)
                    or not isinstance(top_p, (int, float))
                    or not 0 < top_p <= 1
                ):
                    raise ValueError("top_p must be greater than 0 and at most 1")
                top_p = float(top_p)

            stop = body.get("stop")
            if stop is None:
                stop_sequences = []
            elif isinstance(stop, str):
                stop_sequences = [stop]
            elif isinstance(stop, list) and all(isinstance(item, str) for item in stop):
                stop_sequences = stop
            else:
                raise ValueError("stop must be a string or an array of strings")
            if any(not item for item in stop_sequences):
                raise ValueError("stop strings must not be empty")

            model_name = body.get("model", "dm05")
            if not isinstance(model_name, str) or not model_name:
                raise ValueError("model must be a non-empty string")

            tokenizer = self.processor.tokenizer
            eos_token_ids = []
            tokenizer_eos = tokenizer.eos_token_id
            if isinstance(tokenizer_eos, (list, tuple)):
                eos_candidates = list(tokenizer_eos)
            else:
                eos_candidates = [tokenizer_eos]
            eos_candidates.append(tokenizer.convert_tokens_to_ids("<end_of_turn>"))
            for token_id in eos_candidates:
                if token_id is None or token_id == tokenizer.unk_token_id:
                    continue
                token_id = int(token_id)
                if token_id not in eos_token_ids:
                    eos_token_ids.append(token_id)

            generation_input = {
                key: value.to(self.device) if value is not None else None
                for key, value in model_input.items()
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_t0 = time.perf_counter()
            try:
                if do_sample:
                    rng_devices = (
                        [self.device.index] if self.device.type == "cuda" else []
                    )
                    rng_context = torch.random.fork_rng(devices=rng_devices)
                else:
                    rng_context = nullcontext()
                dm05_model = unwrap_dm05_model(self.model)
                with (
                    rng_context,
                    dm05_inference_context(self.device, dm05_model.precision_policy),
                ):
                    if do_sample:
                        torch.manual_seed(int.from_bytes(os.urandom(8), "big"))
                    output = dm05_model.generate_text(
                        input_ids=generation_input["input_ids"],
                        attention_mask=generation_input["attention_mask"],
                        pixel_values=generation_input["pixel_values"],
                        token_type_ids=generation_input["token_type_ids"],
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        top_p=top_p,
                        stop_strings=stop_sequences or None,
                        tokenizer=tokenizer,
                        eos_token_id=eos_token_ids or None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            except Exception as exc:
                message = (
                    f"VLM chat generation failed: {exc}. Retry with "
                    "--inference-config.backend default"
                )
                logger.exception(f"v1/chat/completions generation error: {exc}")
                return jsonify({"error": message}), 500

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_latency_sec = time.perf_counter() - model_t0

            text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
            finish_reason = output.finish_reason
            stop_index = None
            for stop_sequence in stop_sequences:
                candidate_index = text.find(stop_sequence)
                if candidate_index >= 0 and (
                    stop_index is None or candidate_index < stop_index
                ):
                    stop_index = candidate_index
            if stop_index is not None:
                text = text[:stop_index]
                finish_reason = "stop"

            latency_ms = round((time.monotonic() - request_t0) * 1000, 1)
            completion_tokens = int(len(output.token_ids))
            prompt_tokens = int(output.prompt_tokens)
            logger.info(
                "v1/chat/completions processing time: {:.3f}s, model_latency_ms={:.3f}",
                latency_ms / 1000.0,
                model_latency_sec * 1000.0,
            )
            return jsonify(
                {
                    "id": "chatcmpl-" + hashlib.sha256(os.urandom(16)).hexdigest()[:8],
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "metadata": {"latency_ms": latency_ms},
                }
            )
        except (BadRequest, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = exc.description if isinstance(exc, BadRequest) else str(exc)
            logger.warning(f"v1/chat/completions bad request: {message}")
            return jsonify({"error": message}), 400
        except Exception as exc:
            message = (
                f"VLM chat generation failed: {exc}. Retry with "
                "--inference-config.backend default"
            )
            logger.exception(f"v1/chat/completions generation error: {exc}")
            return jsonify({"error": message}), 500


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

    def _inference_preprocessing_settings(self) -> tuple[pathlib.Path, dict]:
        """Resolve the normalization file and playground request defaults."""
        checkpoint = pathlib.Path(self.model_config.model_name_or_path)
        checkpoint_norm = checkpoint / "norm_stats.json"
        norm_stats_path = (
            checkpoint_norm
            if checkpoint_norm.exists()
            else self.data_config.norm_stats_path(self.model_config.chunk_size)
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
        return norm_stats_path, dataset_info

    def initialize_inference_preprocessing(self, *, device: str = "cpu") -> None:
        """Initialize the exact playground deployment transform without a model."""
        norm_stats_path, _ = self._inference_preprocessing_settings()
        if not norm_stats_path.is_file():
            raise FileNotFoundError(
                f"Inference normalization stats do not exist: {norm_stats_path}"
            )
        self.inference_config._initialize_preprocessing(
            model_name_or_path=self.model_config.model_name_or_path,
            norm_stats_path=str(norm_stats_path),
            n_bins=self.data_config.n_bins,
            transform_max_length=(
                min(
                    self.trainer_config.model_max_length,
                    int(
                        getattr(self.inference_config, "fast_prefix_len", None) or 1024
                    ),
                )
                if self.inference_config.backend == "fast"
                else self.trainer_config.model_max_length
            ),
            use_absolute_action=(self.data_config.action_mode == ActionMode.RELATIVE),
            add_state=self.data_config.add_state,
            is_history=self.data_config.is_history,
            device=device,
        )
        for step in self.inference_config.input_transform.transforms:
            if isinstance(step, ChatTokenization):
                step.reject_prompt_truncation = True

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
        from opendm.trainer.trainer import DMTrainer

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
            from opendm.trainer.trainer import safe_save_model_for_hf_trainer

            safe_save_model_for_hf_trainer(
                trainer=self.trainer,
                output_dir=self.trainer_config.output_dir,
            )
        logger.info(
            f"Training completed and model saved to {self.trainer_config.output_dir}"
        )

    def _initialize_inference_runtime(self) -> None:
        """Build the model and transforms shared by inference entry points."""
        chat_only = self.inference_config.chat_only
        self.model_config.align_action_rope_to_language = chat_only
        use_fast_backend = self.inference_config.backend == "fast" and not chat_only
        self.model_config.llm_attn_implementation = (
            "flex_attention" if use_fast_backend else "eager"
        )
        if chat_only:
            logger.info(f"Loading model from {self.model_config.model_name_or_path}")
            model = self.model_config.build_model(use_lora=self.use_lora)
            self.inference_config._initialize(
                model=model,
                model_name_or_path=self.model_config.model_name_or_path,
                norm_stats_path="",
                n_bins=0,
                model_max_length=0,
                use_absolute_action=False,
            )
            return

        norm_stats_path, _ = self._inference_preprocessing_settings()
        if (
            use_fast_backend
            and self.model_config.model_name_or_path is not None
            and self.model_config.precision_policy == BF16_MIXED_PRECISION_POLICY
            and self.model_config.bf16
            and not self.inference_config.force_rebuild
            and self.inference_config.auto_prepared_fast_artifacts
            and self.inference_config.vision_trt_engine_path is None
            and self.inference_config.prefix_mlp_fp8_engine_dir is None
        ):
            from opendm.infer.fast_fp8_prepared import (
                preprocessing_contract,
                resolve_prepared_fast_artifacts,
            )

            num_images = len(self.inference_config.image_prompts)
            if self.data_config.is_history:
                num_images += int(self.inference_config.max_history_images)
            prepared = resolve_prepared_fast_artifacts(
                checkpoint=self.model_config.model_name_or_path,
                prepared_dir=self.inference_config.prepared_fast_artifacts_dir,
                num_images=num_images,
                chunk_size=int(self.model_config.chunk_size),
                diffusion_steps=int(self.inference_config.diffusion_steps),
                prefix_capacity=int(
                    getattr(self.inference_config, "fast_prefix_len", None) or 1024
                ),
                prefix_seq_len_buckets=self.inference_config.prefix_seq_len_buckets,
                is_history=bool(self.data_config.is_history),
                preprocessing=preprocessing_contract(
                    checkpoint=self.model_config.model_name_or_path,
                    norm_stats_path=norm_stats_path,
                    image_prompts=self.inference_config.image_prompts,
                    n_bins=self.data_config.n_bins,
                    add_state=self.data_config.add_state,
                    model_max_length=min(
                        self.trainer_config.model_max_length,
                        int(
                            getattr(self.inference_config, "fast_prefix_len", None)
                            or 1024
                        ),
                    ),
                    clip_to_bounds=self.inference_config.clip_to_bounds,
                    compose_eef_rot=self.inference_config.compose_eef_rot,
                    default_robot_type=self.inference_config._request_default(
                        "default_robot_type"
                    ),
                    default_state_desc=self.inference_config._request_default(
                        "default_state_desc"
                    ),
                    default_speed=self.inference_config._request_default(
                        "default_speed", "0.5"
                    ),
                    default_control_mode=self.inference_config._request_default(
                        "default_control_mode"
                    ),
                ),
                verify_build_environment=True,
            )
            if prepared is not None:
                self.inference_config._prepared_artifacts_selected = True
                self.inference_config.vision_trt_engine_path = str(
                    prepared.vision_engine
                )
                self.inference_config.prefix_mlp_fp8_engine_dir = str(
                    prepared.prefix_mlp_bundle
                )
                self.inference_config.prefix_seq_len_buckets = list(
                    prepared.prefix_seq_len_buckets
                )

        if use_fast_backend or self.inference_config.prefix_seq_len_buckets:
            from opendm.infer.dm05_infer import DM05FastInferRuntime

            DM05FastInferRuntime.resolve_prefix_seq_len_buckets(
                self.inference_config.prefix_seq_len_buckets,
                fast_backend=use_fast_backend,
                prefix_capacity=int(
                    getattr(self.inference_config, "fast_prefix_len", None) or 1024
                ),
            )
        if self.inference_config.prefix_mlp_fp8_engine_dir is not None:
            # The FP8 bundle contract fingerprints standard Gemma3 MLP weights.
            # Liger replaces those modules before runtime validation.
            self.model_config.liger_kernel = False
        logger.info(f"Loading model from {self.model_config.model_name_or_path}")
        model = self.model_config.build_model(use_lora=self.use_lora)
        ckpt_norm_stats_path = (
            pathlib.Path(self.model_config.model_name_or_path) / "norm_stats.json"
        )
        if ckpt_norm_stats_path.exists():
            logger.info(f"Using normalization stats from checkpoint: {norm_stats_path}")
        else:
            logger.warning(
                "Normalization stats not found in checkpoint at "
                f"{ckpt_norm_stats_path}; falling back to {norm_stats_path}"
            )
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
            "/health",
            "health",
            self.inference_config._health,
            methods=["GET"],
        )
        if not self.inference_config.chat_only:
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
        app.add_url_rule(
            "/v1/chat/completions",
            "v1_chat",
            self.inference_config._chat,
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
