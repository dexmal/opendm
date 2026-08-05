"""Unified RoboChallenge policy backed by opendm.exp.dm05_exp."""

from __future__ import annotations

import io
from collections import deque

import numpy as np
import torch
from loguru import logger
from PIL import Image
from utils.constants import IMAGE_MAPPING
from utils.transforms import euler_to_quat, quat_to_euler, unwrap_euler_sequence

from opendm.constants.robot import RobotStateDesc
from opendm.data.normalize import load_norm_stats_file
from opendm.data.transforms import ChatTokenization
from opendm.exp.dm05_exp import DM05InferenceConfig, DM05ModelConfig

from .logical_step_history import (
    LOGICAL_STEP_MODES,
    LogicalStepHistoryConfig,
    LogicalStepHistoryStore,
    build_history_placeholder_text,
)
from .output_tricks import apply_w1_gripper_trick, ur5_roll_pi_anchor


class _RCChatTokenization(ChatTokenization):
    """RC prompt layout for dexbotic-trained checkpoints.

    Stock OpenDM ChatTokenization is close; this keeps the RC training text
    format (bare task prompt) and logical-step ``history_placeholder_text``.
    """

    def __call__(self, data):
        meta = data.get("meta_data", {})
        # Gemma3 chat template strips whitespace at text-item boundaries;
        # keep Robot/speed/prompt in one text block (dexbotic _push_text).
        text_parts = []
        if meta.get("robot_type") is not None:
            text_parts.append(f"Robot: {meta['robot_type']}\n")
        if meta.get("control_mode") is not None:
            text_parts.append(f"Control mode: {meta['control_mode']}\n")
        if meta.get("speed") is not None:
            text_parts.append(f"Overall speed: {meta['speed']}\n")
        prompt = data.get("prompt") or ""
        if prompt:
            text_parts.append(f"{prompt}\n")
        user_content = [{"type": "text", "text": "".join(text_parts)}]

        history_pixel_values = None
        history_mask = None
        if self.is_history:
            history_images = data.get("history_images") or []
            user_content[-1]["text"] += "History images: "
            placeholder = data.get("history_placeholder_text")
            if placeholder is not None:
                user_content[-1]["text"] += placeholder
            else:
                user_content[-1]["text"] += "<unused0>" * (16 * len(history_images))
            normalized = [image.convert("RGB") for image in history_images]
            if normalized:
                history_pixel_values = self.processor.image_processor(
                    images=normalized,
                    return_tensors="pt",
                )["pixel_values"]

        for image_prompt, image in zip(self.image_prompts, data["images"], strict=True):
            # dexbotic get_camera_labels / append_camera_images
            label = f"{image_prompt} image: "
            if user_content[-1]["type"] == "text":
                user_content[-1]["text"] += label
            else:
                user_content.append({"type": "text", "text": label})
            user_content.append({"type": "image", "image": image})
        if self.add_state:
            # Keep full padded state for AbsoluteAction; only bin native dims
            # (dexbotic valid_dim_mask / States: length).
            state_for_bins = np.asarray(data["state"], dtype=np.float32).reshape(-1)
            meta = data.get("meta_data") or {}
            token_dim = meta.get("state_token_dim")
            if token_dim is not None:
                state_for_bins = state_for_bins[: int(token_dim)]
            state_text = " ".join(
                str(b)
                for b in self.action_to_bin_tokens(
                    state_for_bins,
                    self.n_bins,
                )
            )
            # Match tokenize_robot_infer: "States: " + bins + "\n"
            user_content.append(
                {"type": "text", "text": "States: " + state_text + "\n"}
            )
        messages = [{"role": "user", "content": user_content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] > self.max_length and prompt:
            prompt_token_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
            overflow = inputs["input_ids"].shape[1] - self.max_length
            keep_tokens = max(0, len(prompt_token_ids) - overflow - 16)
            if keep_tokens < len(prompt_token_ids):
                shortened_prompt = self.tokenizer.decode(
                    prompt_token_ids[:keep_tokens],
                    skip_special_tokens=False,
                ).strip()
                for content_item in messages[0]["content"]:
                    if content_item.get("type") != "text":
                        continue
                    text = content_item.get("text", "")
                    if prompt not in text:
                        continue
                    content_item["text"] = text.replace(prompt, shortened_prompt, 1)
                    break
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
        token_type_ids = inputs["token_type_ids"]
        if self.is_history:
            history_mask = inputs["input_ids"] == self.history_placeholder_token_id
            # Match OpenDM: mark ``<unused0>`` history slots as image tokens.
            token_type_ids = token_type_ids.clone()
            token_type_ids[history_mask] = 1
        return {
            "action": torch.from_numpy(data["action"]) if "action" in data else None,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "token_type_ids": token_type_ids,
            "history_pixel_values": history_pixel_values,
            "history_mask": history_mask,
        }


CAMERA_ALIASES = {
    "high": ["high", "cam_high"],
    "left_hand": ["left_hand", "cam_left_wrist", "cam_arm"],
    "right_hand": ["right_hand", "cam_right_wrist", "cam_global"],
    "cam_global": ["cam_global", "right_hand", "cam_high"],
    "cam_arm": ["cam_arm", "left_hand", "cam_right_wrist"],
    "cam_side": ["cam_side", "high", "left_hand"],
    "cam_high": ["cam_high", "high"],
    "cam_left_wrist": ["cam_left_wrist", "left_hand"],
    "cam_right_wrist": ["cam_right_wrist", "right_hand"],
}

# Match dexbotic RobotType string values used in training prompts
# (dexbotic/data/data_source/dm05_const.py: ARX5="ARX5", UR5="UR5", ...).
ROBOT_PROMPT_LABEL = {
    "arx5": "ARX5",
    "ur5": "UR5",
    "aloha": "Aloha",
    "w1": "DOS W1",
    "dm05_dosw1": "DOS W1",
}

DEFAULT_IMAGE_PROMPTS = {
    1: ["Head"],
    2: ["Head", "Left wrist"],
    3: ["Head", "Left wrist", "Right wrist"],
}

UR5_GRIPPER_WIDTH_OPEN = 0.085
UR5_GRIPPER_WIDTH_CLOSED = 0.0
UR5_ROBOTIQ_OPEN_VALUE = 3.0
UR5_ROBOTIQ_CLOSED_VALUE = 228.0


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_action_mode(mode: str | None) -> str:
    if mode is None or str(mode).strip().lower() in {"", "auto"}:
        return "relative"
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "relative": "relative",
        "delta": "relative",
        "rel": "relative",
        "absolute": "absolute",
        "abs": "absolute",
        "direct": "absolute",
    }
    if normalized not in aliases:
        raise ValueError(
            f"action_mode must be 'relative'/'delta' or 'absolute', got {mode!r}"
        )
    return aliases[normalized]


def _normalize_target(target: str | None, robot_type: str, action_type: str) -> str:
    if target is None or str(target).strip() == "":
        if robot_type in ("arx5", "ur5") and "pos" in action_type:
            return "eef"
        return "joint"
    normalized = str(target).strip().lower().replace("-", "_")
    aliases = {
        "joint": "joint",
        "joints": "joint",
        "qpos": "joint",
        "eef": "eef",
        "ee": "eef",
        "pose": "eef",
        "end_effector": "eef",
    }
    if normalized not in aliases:
        raise ValueError(f"single_arm_target must be 'joint' or 'eef', got {target!r}")
    return aliases[normalized]


def _build_state_desc(dim: int, target: str, native_dim: int) -> list[RobotStateDesc]:
    """Native robot layout, padded with JOINT to match action-norm dim."""
    if native_dim == 14:
        desc = (
            [RobotStateDesc.JOINT] * 6
            + [RobotStateDesc.GRIPPER]
            + [RobotStateDesc.JOINT] * 6
            + [RobotStateDesc.GRIPPER]
        )
    elif native_dim == 7 and target == "eef":
        desc = [RobotStateDesc.EEF] * 6 + [RobotStateDesc.GRIPPER]
    elif native_dim == 7:
        desc = [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
    else:
        raise ValueError(
            f"Unsupported native_dim={native_dim} for target={target!r}; "
            "expected 7 or 14."
        )
    if dim < native_dim:
        raise ValueError(
            f"state dim {dim} < native_dim {native_dim} for target={target!r}"
        )
    if dim > native_dim:
        desc = list(desc) + [RobotStateDesc.JOINT] * (dim - native_dim)
    return desc


def _read_action_norm_dim(
    norm_stats_path: str,
    robot_type: str | None = None,
) -> int:
    """Read action vector dim from an OpenDM ``norm_stats.json`` (q01/q99)."""
    profile = load_norm_stats_file(norm_stats_path).select(robot_type)
    action = profile["action"]
    missing = [key for key in ("q01", "q99") if getattr(action, key) is None]
    if missing:
        raise ValueError(f"action stats missing {missing} in {norm_stats_path}")
    action_dim = len(action.q01)
    if "state" in profile:
        state = profile["state"]
        for key in ("q01", "q99"):
            values = getattr(state, key)
            if values is not None and len(values) != action_dim:
                raise ValueError(
                    f"state.{key} dim {len(values)} != action dim "
                    f"{action_dim} in {norm_stats_path}"
                )
    return action_dim


class OpenDMPolicy:
    """Thin adapter around opendm DM05InferenceConfig._predict."""

    def __init__(
        self,
        ckpt_path: str,
        prompt: str,
        robot_type: str,
        action_type: str,
        action_horizon: int = 15,
        norm_stats: str | None = None,
        image_shape=(728, 728),
        task_name: str = "",
        speed: float | int | str | None = 0.5,
        add_state: bool = True,
        is_history: bool = False,
        history_max_frames: int = 5,
        action_mode: str = "relative",
        single_arm_target: str | None = None,
        image_prompts: list[str] | None = None,
        n_bins: int = 256,
        model_max_length: int = 1024,
        diffusion_steps: int = 10,
        action_playback_target_steps: int = 0,
        ur5_anchor_pitch_zero: bool = False,
        # When True: roll -> ±π (same side as runtime/pred), pitch -> 0.
        # Alias: ur5_anchor_roll_pi_pitch_zero (same meaning).
        ur5_anchor_roll_pitch_zero: bool = False,
        ur5_anchor_roll_pi_pitch_zero: bool | None = None,
        ur5_gripper_robotiq_to_width: bool = False,
        ur5_gripper_robotiq_mapping: str = "reverse",
        llm_attn_implementation: str | None = None,
        vision_attn_implementation: str | None = None,
        action_attn_implementation: str | None = None,
        # Pin diffusion / CPU / CUDA RNG inside run_policy when set.
        inference_seed: int | None = None,
        # ``default`` = eager VLM; ``fast`` = TensorRT vision + CUDA-graph path.
        backend: str = "default",
        vision_trt_engine_path: str | None = None,
        force_rebuild_trt: bool = False,
        single_arm_action_mode: str | None = None,
        add_discrete_state: bool | None = None,
        history_mode: str | None = None,
        history_action_step_increment: int = 0,
        history_raw_slots: int = 5,
        history_step_tolerance: int = 1,
        history_runtime_fps: float = 30.0,
        **_ignored,
    ):
        if not norm_stats:
            raise ValueError("norm_stats is required for OpenDMPolicy")

        self.ckpt_path = ckpt_path
        self.prompt = prompt
        self.robot_type = str(robot_type).lower()
        if self.robot_type == "dm05_dosw1":
            self.robot_type = "w1"
        self.action_type = action_type
        self.action_horizon = int(action_horizon)
        self.image_shape = image_shape
        self.task_name = task_name
        self.speed = speed
        if add_discrete_state is not None:
            add_state = add_discrete_state
        self.add_state = _as_bool(add_state, True)
        self.llm_attn_implementation = llm_attn_implementation
        self.vision_attn_implementation = vision_attn_implementation
        self.action_attn_implementation = action_attn_implementation
        self.inference_seed = None if inference_seed is None else int(inference_seed)
        self.backend = str(backend or "default").strip().lower()
        if self.backend not in {"default", "fast"}:
            raise ValueError(f"backend must be 'default' or 'fast', got {backend!r}")
        self.vision_trt_engine_path = vision_trt_engine_path
        self.force_rebuild_trt = bool(force_rebuild_trt)

        mode = str(history_mode).strip().lower() if history_mode is not None else ""
        self.history_mode = mode
        if mode in LOGICAL_STEP_MODES:
            is_history = True
        elif mode not in {"", "none", "null", "false", "0"} and not _as_bool(
            is_history, False
        ):
            is_history = True
            logger.warning(
                "history_mode={!r} mapped to is_history=True (deque mode)",
                history_mode,
            )
        self.is_history = _as_bool(is_history, False)
        self.use_logical_step_history = self.is_history and mode in LOGICAL_STEP_MODES
        self.history_max_frames = max(0, int(history_max_frames))
        if single_arm_action_mode is not None and (
            action_mode is None or str(action_mode).strip().lower() in {"", "auto"}
        ):
            action_mode = single_arm_action_mode
        self.action_mode = _normalize_action_mode(action_mode)
        self.single_arm_target = _normalize_target(
            single_arm_target, self.robot_type, action_type
        )
        self.n_bins = int(n_bins)
        self.model_max_length = int(model_max_length)
        self.diffusion_steps = int(diffusion_steps)
        self.action_playback_target_steps = int(action_playback_target_steps or 0)
        self.ur5_anchor_pitch_zero = _as_bool(ur5_anchor_pitch_zero, False)
        if ur5_anchor_roll_pi_pitch_zero is not None:
            ur5_anchor_roll_pitch_zero = ur5_anchor_roll_pi_pitch_zero
        self.ur5_anchor_roll_pitch_zero = _as_bool(ur5_anchor_roll_pitch_zero, False)
        self.ur5_gripper_robotiq_to_width = _as_bool(
            ur5_gripper_robotiq_to_width, False
        )
        mapping = str(ur5_gripper_robotiq_mapping or "reverse").strip().lower()
        self.ur5_gripper_robotiq_mapping = (
            "forward" if mapping in {"forward", "normal", "positive"} else "reverse"
        )

        self.image_mapping = dict(IMAGE_MAPPING.get(self.robot_type, {}))
        self.robot_prompt_label = ROBOT_PROMPT_LABEL.get(
            self.robot_type, self.robot_type
        )
        self._history_images: deque[Image.Image] = deque(maxlen=self.history_max_frames)
        self._last_runtime_quat_xyzw = None
        self._last_runtime_euler = None

        # Slot grid from playback (or horizon). Runtime may later sync hop via
        # ``sync_history_action_step_increment`` so hop and slot_step stay locked.
        slot_step = int(
            history_action_step_increment
            or self.action_playback_target_steps
            or action_horizon
            or 25
        )
        if slot_step <= 0:
            slot_step = 25
        self.history_action_step_increment = int(
            history_action_step_increment or slot_step
        )
        self._history_session_id = "realrobot_inference"
        self._history_action_step = 0
        self._history_chunk_idx = 0
        self._history_store: LogicalStepHistoryStore | None = None
        if self.use_logical_step_history:
            hist_cfg = LogicalStepHistoryConfig(
                raw_slots=max(1, int(history_raw_slots)),
                tokens_per_slot=16,
                slot_step=slot_step,
                runtime_fps=float(history_runtime_fps),
                step_tolerance=max(0, int(history_step_tolerance)),
            )
            self._history_store = LogicalStepHistoryStore(hist_cfg)
            self.history_max_frames = hist_cfg.raw_slots
            logger.info(
                "Configured logical-step history: raw_slots={}, uniform_fps={:.6f}, "
                "slot_step={}, step_increment={}, runtime_fps={}, tolerance={}",
                hist_cfg.raw_slots,
                hist_cfg.uniform_fps,
                hist_cfg.slot_step,
                self.history_action_step_increment,
                hist_cfg.runtime_fps,
                hist_cfg.step_tolerance,
            )

        if self.robot_type in ("aloha", "w1"):
            self.native_action_dim = 14
        else:
            self.native_action_dim = 7

        # rc_ckpt norm_stats use native dims (7 single-arm / 14 dual-arm) with
        # q01/q99. Missing state is fine: OpenDM Normalize skips absent keys.
        self.norm_stats = norm_stats
        self.shared_dim = _read_action_norm_dim(norm_stats, self.robot_prompt_label)
        if self.shared_dim != self.native_action_dim:
            raise ValueError(
                f"action norm dim={self.shared_dim} in {norm_stats} must equal "
                f"native_action_dim={self.native_action_dim} for robot="
                f"{self.robot_type}"
            )
        self.model_state_dim = self.native_action_dim
        self.output_action_dim = self.native_action_dim

        if image_prompts is not None:
            self.image_prompts = list(image_prompts)
        else:
            cam_count = {"arx5": 3, "ur5": 2, "aloha": 3, "w1": 3}.get(
                self.robot_type, 3
            )
            self.image_prompts = list(
                DEFAULT_IMAGE_PROMPTS.get(cam_count, DEFAULT_IMAGE_PROMPTS[3])
            )

        logger.info("Initializing OpenDMPolicy")
        logger.info(f"  checkpoint       = {ckpt_path}")
        logger.info(f"  robot_type       = {self.robot_type}")
        logger.info(f"  action_type      = {action_type}")
        logger.info(f"  action_mode      = {self.action_mode}")
        logger.info(f"  single_arm_target= {self.single_arm_target}")
        logger.info(f"  norm_stats       = {self.norm_stats}")
        logger.info(f"  shared_dim       = {self.shared_dim}")
        logger.info(f"  model_state_dim  = {self.model_state_dim}")
        logger.info(f"  native_action_dim= {self.native_action_dim}")
        logger.info(f"  output_action_dim= {self.output_action_dim}")
        logger.info(f"  is_history       = {self.is_history}")
        logger.info(
            f"  history_mode     = "
            f"{self.history_mode or ('deque' if self.is_history else 'none')}"
        )
        logger.info(f"  history_max      = {self.history_max_frames}")
        if self.use_logical_step_history:
            logger.info(f"  history_step_inc = {self.history_action_step_increment}")
        logger.info(f"  image_prompts    = {self.image_prompts}")

        # Keep the checkpoint's diffusion chunk length (typically 50). Per-task
        # action_horizon only selects how many steps to execute / return.
        ckpt_chunk_size = 50
        try:
            from opendm.model.dm05.dm05_arch import DM05Config as _DM05Config

            ckpt_chunk_size = int(
                getattr(
                    _DM05Config.from_pretrained(ckpt_path),
                    "chunk_size",
                    ckpt_chunk_size,
                )
                or ckpt_chunk_size
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to read chunk_size from checkpoint {}; fallback={}: {}",
                ckpt_path,
                ckpt_chunk_size,
                exc,
            )
        if self.action_horizon > ckpt_chunk_size:
            raise ValueError(
                f"action_horizon={self.action_horizon} exceeds model "
                f"chunk_size={ckpt_chunk_size}"
            )
        self.model_chunk_size = ckpt_chunk_size
        logger.info(f"  model_chunk_size = {self.model_chunk_size}")
        logger.info(f"  action_horizon   = {self.action_horizon}")

        llm_attn, vision_attn, action_attn = self._resolve_attn_implementations()
        logger.info(f"  llm_attn         = {llm_attn}")
        logger.info(f"  vision_attn      = {vision_attn}")
        logger.info(f"  action_attn      = {action_attn}")
        model_cfg = DM05ModelConfig(
            model_name_or_path=ckpt_path,
            chunk_size=self.model_chunk_size,
            llm_attn_implementation=llm_attn,
            vision_attn_implementation=vision_attn,
            action_attn_implementation=action_attn,
            # Liger fused kernels drift from dexbotic (SigLIP LN / RMSNorm / RoPE).
            liger_kernel=False,
        )
        model = model_cfg.build_model(use_lora=False)

        self.infer_cfg = DM05InferenceConfig(
            diffusion_steps=self.diffusion_steps,
            output_action_dim=self.output_action_dim,
            image_prompts=self.image_prompts,
            backend=self.backend,
            vision_trt_engine_path=self.vision_trt_engine_path,
            force_rebuild=self.force_rebuild_trt,
        )
        self.infer_cfg.default_robot_type = self.robot_prompt_label
        self.infer_cfg._initialize(
            model=model,
            model_name_or_path=ckpt_path,
            norm_stats_path=self.norm_stats,
            n_bins=self.n_bins,
            model_max_length=self.model_max_length,
            use_absolute_action=(self.action_mode == "relative"),
            add_state=self.add_state,
            is_history=self.is_history,
        )
        self._install_rc_chat_tokenization()

    def _resolve_attn_implementations(self) -> tuple[str, str, str]:
        """Resolve attn backends for model load.

        Matches ``Dexbotic-RoboChallengeInference`` policy defaults:

        - aloha/w1: ``auto`` / ``auto`` / ``auto``
        - arx5/ur5: ``sdpa`` / ``flash_attention_2|sdpa`` / ``sdpa``

        Explicit runtime kwargs always win when set.
        """
        if self.robot_type in ("aloha", "w1"):
            default_llm = "auto"
            default_vision = "auto"
            default_action = "auto"
        else:
            default_llm = "sdpa"
            default_vision = (
                "flash_attention_2" if torch.cuda.is_available() else "sdpa"
            )
            default_action = "sdpa"

        def _pick(value: str | None, default: str) -> str:
            if value is None or str(value).strip() == "":
                return default
            return str(value).strip()

        return (
            _pick(self.llm_attn_implementation, default_llm),
            _pick(self.vision_attn_implementation, default_vision),
            _pick(self.action_attn_implementation, default_action),
        )

    def _install_rc_chat_tokenization(self) -> None:
        """Swap in RC training prompt layout (bare task + logical-step history)."""
        if self.robot_type not in ("aloha", "w1", "arx5", "ur5"):
            return
        transforms = getattr(
            getattr(self.infer_cfg, "input_transform", None), "transforms", None
        )
        if not transforms:
            return
        for idx, transform in enumerate(transforms):
            if not isinstance(transform, ChatTokenization):
                continue
            transforms[idx] = _RCChatTokenization(
                processor=transform.processor,
                n_bins=transform.n_bins,
                max_length=transform.max_length,
                image_prompts=transform.image_prompts,
                add_state=transform.add_state,
                is_history=transform.is_history,
            )
            logger.info(
                "Installed RC ChatTokenization for robot={}",
                self.robot_type,
            )
            return

    def reset(self) -> None:
        self._history_images.clear()
        self._history_action_step = 0
        self._history_chunk_idx = 0
        if self._history_store is not None:
            self._history_store.clear(self._history_session_id)
        self._last_runtime_quat_xyzw = None
        self._last_runtime_euler = None

    def sync_history_action_step_increment(self, hop: int) -> None:
        """Keep append hop and logical-step slot grid locked together.

        If hop changes while ``slot_step`` / ``uniform_fps`` stay on an older
        grid, frames miss slot targets and valid history stalls. Rebuild the
        store config so both stay in lockstep.
        """
        hop = int(hop)
        if hop <= 0:
            return
        prev = int(getattr(self, "history_action_step_increment", 0) or 0)
        self.history_action_step_increment = hop
        if not self.use_logical_step_history or self._history_store is None:
            return
        cfg = self._history_store.config
        if int(cfg.slot_step) == hop:
            return
        new_cfg = LogicalStepHistoryConfig(
            raw_slots=int(cfg.raw_slots),
            tokens_per_slot=int(cfg.tokens_per_slot),
            slot_step=hop,
            runtime_fps=float(cfg.runtime_fps),
            step_tolerance=int(cfg.step_tolerance),
        )
        self._history_store = LogicalStepHistoryStore(new_cfg)
        self.history_max_frames = new_cfg.raw_slots
        logger.warning(
            "Resynced logical-step history grid: hop {}→{}, slot_step {}→{}, "
            "uniform_fps {:.6f}→{:.6f} (cleared in-flight history buffers)",
            prev,
            hop,
            cfg.slot_step,
            new_cfg.slot_step,
            cfg.uniform_fps,
            new_cfg.uniform_fps,
        )

    @staticmethod
    def _seed_inference_rng(seed: int) -> None:
        """Pin Python / NumPy / Torch (+CUDA) RNG for diffusion replay."""
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_policy(self, state, image_type):
        if self.inference_seed is not None:
            self._seed_inference_rng(self.inference_seed)
        images = self._parse_images(state, image_type)
        ordered_images = self._ordered_pil_images(images)
        robot_state = self._prepare_robot_state(state["action"])
        model_state = self._pad_state_to_shared(robot_state)
        if model_state.shape[-1] != self.model_state_dim:
            model_state = model_state[: self.model_state_dim]
        state_desc = _build_state_desc(
            model_state.shape[-1],
            self.single_arm_target,
            native_dim=self.native_action_dim,
        )

        # AbsoluteAction uses the pre-normalize physical state captured inside
        # OpenDM _predict; keep a contiguous float32 vector here.
        physical_state = np.asarray(model_state, dtype=np.float32).reshape(-1).copy()
        history_images: list[Image.Image] = []
        history_placeholder_text = None
        if self.use_logical_step_history and self._history_store is not None:
            raw_frames, valid_mask = self._history_store.build_slot_images(
                session_id=self._history_session_id,
                current_action_step=self._history_action_step,
            )
            history_images = [
                frame
                for frame, ok in zip(raw_frames, valid_mask, strict=True)
                if ok and frame is not None
            ]
            history_placeholder_text = build_history_placeholder_text(
                valid_mask,
                self._history_store.config.tokens_per_slot,
            )
            logger.debug(
                "Logical-step history: session={}, action_step={}, valid={}/{}, "
                "step_increment={}",
                self._history_session_id,
                self._history_action_step,
                int(sum(valid_mask)),
                len(valid_mask),
                self.history_action_step_increment,
            )
        elif self.is_history:
            history_images = list(self._history_images)

        data = {
            "images": ordered_images,
            "history_images": history_images,
            "prompt": self.prompt,
            "state": physical_state,
            "meta_data": {
                "robot_type": self.robot_prompt_label,
                "speed": None if self.speed is None else str(self.speed),
                "control_mode": None,
                "state_desc": state_desc,
                # AbsoluteAction uses full padded state; States: bins use native.
                "state_token_dim": int(self.native_action_dim),
            },
        }
        if history_placeholder_text is not None:
            data["history_placeholder_text"] = history_placeholder_text
        actions = self.infer_cfg._predict(data)[: self.action_horizon]
        actions = np.asarray(actions, dtype=np.float32)[:, : self.native_action_dim]
        actions = self._apply_action_playback(actions)
        actions = self._apply_output_tricks(actions)

        if (
            self.use_logical_step_history
            and self._history_store is not None
            and ordered_images
        ):
            self._history_store.append(
                session_id=self._history_session_id,
                image=ordered_images[0],
                action_step=self._history_action_step,
                chunk_idx=self._history_chunk_idx,
            )
            logger.debug(
                "Appended logical-step history frame: session={} action_step={} chunk_idx={}",
                self._history_session_id,
                self._history_action_step,
                self._history_chunk_idx,
            )
            self._history_action_step += int(self.history_action_step_increment)
            self._history_chunk_idx += 1
        elif self.is_history and ordered_images:
            self._history_images.append(ordered_images[0].copy())
        return actions.tolist()

    def _pad_state_to_shared(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] == self.shared_dim:
            return state
        if state.shape[0] > self.shared_dim:
            return state[: self.shared_dim]
        padded = np.zeros((self.shared_dim,), dtype=np.float32)
        padded[: state.shape[0]] = state
        return padded

    def _apply_action_playback(self, actions):
        target_steps = self.action_playback_target_steps
        if target_steps <= 0:
            return actions
        actions = np.asarray(actions)
        horizon = int(actions.shape[0])
        if target_steps > horizon:
            raise ValueError(
                "action_playback_target_steps cannot exceed available action "
                f"chunk length: {target_steps} > {horizon}"
            )
        if target_steps == horizon:
            return actions
        indices = np.linspace(0, horizon - 1, target_steps + 1)[1:].astype(np.int64)
        return actions[indices]

    def _prepare_robot_state(self, raw_state):
        robot_state = np.asarray(raw_state, dtype=np.float32)
        if robot_state.ndim > 1:
            robot_state = robot_state[0]

        if self.robot_type in ("arx5", "ur5") and self.single_arm_target == "eef":
            if robot_state.shape[-1] == 8:
                pos = robot_state[:3]
                quat = self._normalize_quat_xyzw(robot_state[3:7])
                self._last_runtime_quat_xyzw = quat.copy()
                gripper = robot_state[7:8]
                euler = quat_to_euler(quat, degrees=False)
                self._last_runtime_euler = np.asarray(euler, dtype=np.float32).copy()
                return np.concatenate([pos, euler, gripper]).astype(np.float32)
            if robot_state.shape[-1] in (7, 13):
                if robot_state.shape[-1] == 7:
                    self._last_runtime_euler = robot_state[3:6].copy()
                    return robot_state
                # 13-dim: [6 joint + xyz+rpy + gripper]; cache eef rpy, trim to 7
                self._last_runtime_euler = robot_state[9:12].copy()
                return robot_state[-7:].astype(np.float32)
            raise ValueError(
                f"{self.robot_type} EEF inference expects 8/7/13-dim state, "
                f"got {robot_state.shape[-1]}"
            )

        if self.robot_type in ("arx5", "ur5"):
            if robot_state.shape[-1] == 13:
                robot_state = np.concatenate(
                    [robot_state[:6], robot_state[-1:]]
                ).astype(np.float32)
            if self.ur5_gripper_robotiq_to_width:
                robot_state = robot_state.copy()
                width = robot_state[-1].copy()
                robot_state[-1] = self._ur5_width_to_robotiq(width)
            return robot_state

        return robot_state

    def _parse_images(self, state, image_type):
        raw_images = state.get("images", {})
        mapping = self.image_mapping
        parsed = {}
        for source in image_type:
            candidates = CAMERA_ALIASES.get(source, [source])
            selected = next((name for name in candidates if name in raw_images), None)
            if selected is None:
                continue
            image_data = raw_images[selected]
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # Slot assignment follows the requested semantic source
            # (cam_global/cam_side/cam_arm), matching dexbotic single-arm.
            slot = mapping.get(source)
            if slot:
                parsed[slot] = np.asarray(image)
        if not parsed:
            raise ValueError(f"No images parsed for image_type={image_type!r}")
        return parsed

    @staticmethod
    def _normalize_quat_xyzw(quat):
        quat = np.asarray(quat, dtype=np.float32)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        return np.divide(quat, norm, out=quat.copy(), where=norm > 1e-8)

    def _align_eef_quat_signs(self, quat):
        quat = self._normalize_quat_xyzw(quat)
        ref = self._last_runtime_quat_xyzw
        if ref is None or not np.all(np.isfinite(ref)):
            return quat
        ref = self._normalize_quat_xyzw(ref)
        aligned = quat.copy()
        for i in range(aligned.shape[0]):
            if float(np.dot(aligned[i], ref)) < 0.0:
                aligned[i] *= -1.0
            ref = aligned[i]
        self._last_runtime_quat_xyzw = ref.copy()
        return aligned

    @staticmethod
    def _ordered_pil_images(images):
        ordered = []
        for key in ("image_0", "image_1", "image_2"):
            if key in images:
                ordered.append(Image.fromarray(images[key]).convert("RGB"))
        if not ordered:
            raise ValueError("No model image slots found after parsing")
        return ordered

    def _ur5_robotiq_to_width(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        clipped = np.clip(raw, UR5_ROBOTIQ_OPEN_VALUE, UR5_ROBOTIQ_CLOSED_VALUE)
        raw_range = UR5_ROBOTIQ_CLOSED_VALUE - UR5_ROBOTIQ_OPEN_VALUE
        if self.ur5_gripper_robotiq_mapping == "forward":
            scale = (clipped - UR5_ROBOTIQ_OPEN_VALUE) / raw_range
            return scale * UR5_GRIPPER_WIDTH_OPEN
        scale = (UR5_ROBOTIQ_CLOSED_VALUE - clipped) / raw_range
        return scale * UR5_GRIPPER_WIDTH_OPEN

    def _ur5_width_to_robotiq(self, width):
        width = np.asarray(width, dtype=np.float32)
        clipped = np.clip(width, UR5_GRIPPER_WIDTH_CLOSED, UR5_GRIPPER_WIDTH_OPEN)
        scale = clipped / UR5_GRIPPER_WIDTH_OPEN
        if self.ur5_gripper_robotiq_mapping == "forward":
            return UR5_ROBOTIQ_OPEN_VALUE + scale * (
                UR5_ROBOTIQ_CLOSED_VALUE - UR5_ROBOTIQ_OPEN_VALUE
            )
        return UR5_ROBOTIQ_CLOSED_VALUE - scale * (
            UR5_ROBOTIQ_CLOSED_VALUE - UR5_ROBOTIQ_OPEN_VALUE
        )

    def _apply_output_tricks(self, actions):
        actions = np.asarray(actions, dtype=np.float32)

        if self.robot_type == "w1":
            actions = apply_w1_gripper_trick(actions, self.task_name)
        elif self.robot_type == "aloha":
            actions = actions.copy()
            for idx in (6, 13):
                if idx < actions.shape[1]:
                    actions[:, idx] = np.where(
                        actions[:, idx] < 0.01, 0.0, actions[:, idx]
                    )
        elif self.ur5_gripper_robotiq_to_width and self.robot_type == "ur5":
            actions = actions.copy()
            gripper_idx = 6 if actions.shape[-1] >= 7 else actions.shape[-1] - 1
            actions[:, gripper_idx] = self._ur5_robotiq_to_width(
                actions[:, gripper_idx]
            )
        elif self.robot_type in ("arx5", "ur5"):
            actions = actions.copy()
            gripper_idx = actions.shape[-1] - 1
            actions[:, gripper_idx] = np.where(
                actions[:, gripper_idx] < 0.01,
                0.0,
                actions[:, gripper_idx],
            )

        if self.single_arm_target == "eef" and "pos" in self.action_type:
            if actions.shape[-1] != 7:
                raise ValueError(
                    "EEF action conversion expects 7 dims [xyz + rpy + gripper], "
                    f"got shape={actions.shape}"
                )
            actions = actions.copy()
            actions[:, 3:6] = unwrap_euler_sequence(actions[:, 3:6])
            if self.robot_type == "ur5" and self.ur5_anchor_roll_pitch_zero:
                roll_anchor = self._ur5_roll_pi_anchor(actions[0, 3])
                actions[:, 3] = roll_anchor
                actions[:, 4] = 0.0
            elif self.robot_type == "ur5" and self.ur5_anchor_pitch_zero:
                actions[:, 4] = 0.0
            quat = self._align_eef_quat_signs(
                euler_to_quat(actions[:, 3:6], degrees=False)
            )
            return np.concatenate([actions[:, :3], quat, actions[:, 6:]], axis=1)

        if (
            self.robot_type in ("arx5", "ur5")
            and self.single_arm_target == "joint"
            and actions.shape[-1] > 7
        ):
            return actions[:, :7]
        return actions

    def _ur5_roll_pi_anchor(self, predicted_roll) -> float:
        """Anchor UR5 roll to ±π using runtime state roll when available."""
        ref_euler = self._last_runtime_euler
        if ref_euler is not None and np.all(np.isfinite(ref_euler)):
            reference = float(ref_euler[0])
        else:
            reference = float(predicted_roll)
        return ur5_roll_pi_anchor(reference)
