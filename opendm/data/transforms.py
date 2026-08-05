"""Robot imitation data transforms and pipeline."""

import io
import os
import re

import megfile
import numpy as np
import orjson
import torch
from loguru import logger
from PIL import Image

from opendm.constants.robot import (
    HISTORY_TOKENS_PER_IMAGE,
    ActionMode,
    RobotStateDesc,
)
from opendm.data.augmentations import TransformPipeline
from opendm.data.normalize import NormStats, NormStatsFile, load_norm_stats_file


def _state_desc_value(desc: RobotStateDesc | str) -> str:
    if isinstance(desc, RobotStateDesc):
        return desc.value
    return str(desc)


def _format_state_descs(descs) -> list[str]:
    return sorted(_state_desc_value(desc) for desc in descs)


class Pipeline:
    """Apply a sequence of transforms to a sample dictionary.

    Args:
        transforms: Ordered transforms. Each transform must accept the current
            sample dictionary and return the transformed dictionary.
    """

    def __init__(self, transforms: list):
        self.transforms = list(transforms)

    def __str__(self):
        return f"Pipeline([{', '.join(str(t) for t in self.transforms)}])"

    def __call__(self, data, **kw):
        for t in self.transforms:
            data = t(data, **kw)
        return data


class PixelTransform:
    """Apply an image augmentation pipeline to ``data["images"]`` in list order.

    Also transforms ``data["history_images"]`` when present, matching dexbotic's
    ``dm05_history`` inference path (PadToSquare + Resize before image_processor).

    Args:
        transform_pipeline: Albumentations-style pipeline that accepts an
            ``image`` keyword and returns a mapping containing ``"image"``.
    """

    def __init__(self, transform_pipeline: TransformPipeline):
        self.transform_pipeline = transform_pipeline

    def _transform_images(self, images):
        return [
            Image.fromarray(
                self.transform_pipeline(image=np.array(image))["image"],
                mode="RGB",
            )
            for image in images
        ]

    def __call__(self, data):
        data["images"] = self._transform_images(data["images"])
        history_images = data.get("history_images")
        if history_images:
            data["history_images"] = self._transform_images(history_images)
        return data


class Normalize:
    """Normalize numeric sample fields using precomputed dataset statistics.

    Args:
        norm_stats_path: Path to a JSON file containing a ``norm_stats`` object.
        norm_keys: Sample keys to normalize in place.
        use_quantiles: If ``True``, clip and scale values with q01/q99
            quantiles to the range ``[-1, 1]``. If ``False``, standardize with
            mean and standard deviation.
    """

    def __init__(
        self,
        norm_stats_path: str,
        norm_keys: list[str],
        use_quantiles: bool = True,
        norm_stats_file: NormStatsFile | None = None,
    ):
        self.norm_stats_path = norm_stats_path
        self.norm_stats_file = norm_stats_file or load_norm_stats_file(norm_stats_path)
        # Preserve the historical attribute for callers that inspect the default
        # profile directly.
        self.norm_stats = self.norm_stats_file.norm_stats
        self.norm_keys = norm_keys
        self.use_quantiles = use_quantiles

    def __call__(self, data, **kw):
        robot_type = data.get("meta_data", {}).get("robot_type")
        norm_stats = self.norm_stats_file.select(robot_type)
        for key in self.norm_keys:
            if key not in norm_stats:
                continue
            if key not in data:
                continue
            data[key] = self._normalize(data[key], norm_stats[key])
        return data

    def _normalize(self, arr, stats: NormStats):
        arr = np.asarray(arr, dtype=np.float32)
        if self.use_quantiles:
            if stats.q01 is None or stats.q99 is None:
                raise ValueError("q01 and q99 are required for quantile normalization")
            lo = np.asarray(stats.q01, dtype=np.float32)
            hi = np.asarray(stats.q99, dtype=np.float32)
            arr = np.clip(arr, lo, hi)
            out = ((arr - lo) / (hi - lo + 1e-6) * 2.0 - 1.0).astype(np.float32)
            return np.where((lo == 0) & (hi == 0), 0.0, out)

        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        return ((arr - mean) / (std + 1e-6)).astype(np.float32)


class Denormalize:
    """Invert normalization for numeric sample fields.

    Args:
        norm_stats_path: Path to a JSON file containing a ``norm_stats`` object.
        norm_keys: Sample keys to denormalize when both the key and statistics
            are present.
        use_quantiles: If ``True``, invert q01/q99 scaling from ``[-1, 1]``.
            If ``False``, invert mean/std standardization.
    """

    def __init__(
        self,
        norm_stats_path: str,
        norm_keys: list[str],
        use_quantiles: bool = True,
        norm_stats_file: NormStatsFile | None = None,
    ):
        self.norm_stats_file = norm_stats_file or load_norm_stats_file(norm_stats_path)
        self.norm_stats = self.norm_stats_file.norm_stats
        self.norm_keys = norm_keys
        self.use_quantiles = use_quantiles

    def __call__(self, data, **kw):
        robot_type = data.get("meta_data", {}).get("robot_type")
        norm_stats = self.norm_stats_file.select(robot_type)
        for key in self.norm_keys:
            if key in data and key in norm_stats:
                data[key] = self._denormalize(data[key], norm_stats[key])
        return data

    def _denormalize(self, arr, stats: NormStats):
        arr = np.asarray(arr, dtype=np.float32)
        if self.use_quantiles:
            if stats.q01 is None or stats.q99 is None:
                raise ValueError(
                    "q01 and q99 are required for quantile denormalization"
                )
            lo = np.asarray(stats.q01, dtype=np.float32)
            hi = np.asarray(stats.q99, dtype=np.float32)
            out = ((arr + 1.0) / 2.0 * (hi - lo + 1e-6) + lo).astype(np.float32)
            return np.where((lo == 0) & (hi == 0), 0.0, out)
        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        return (arr * (std + 1e-6) + mean).astype(np.float32)


class LoadImages:
    """Load image-like sample entries from image files or video frames.

    Reads JSONL fields named by ``image_keys`` in order and stores the loaded
    PIL images as ``data["images"]`` for downstream list-order processing.

    Args:
        image_keys: Keys whose values are metadata dictionaries with ``url``,
            ``type``, and optionally ``frame_idx`` fields.
        image_dir: Base directory prepended to relative image or video URLs.

    Raises:
        ValueError: If an entry has an unsupported ``type``.
    """

    def __init__(self, image_keys: list[str], image_dir: str = ""):
        self.image_keys = image_keys
        self.image_dir = image_dir

    def __call__(self, data):
        images = []
        for key in self.image_keys:
            image_url = os.path.join(self.image_dir, data[key]["url"].lstrip("./"))
            image_type = data[key]["type"]
            if image_type == "image":
                images.append(_load_image(image_url))
            elif image_type == "video":
                images.append(_load_video(image_url, data[key]["frame_idx"]))
            else:
                raise ValueError(f"Invalid image type: {image_type}")
        data["images"] = images
        return data


class BuildActionChunk:
    """Build a fixed-horizon action sequence from episode JSONL frames.

    The transform reads future values from ``raw_lines`` using the current
    ``meta_data["frame_index"]``. If the current sample already contains an
    ``action`` field, the chunk starts at the current frame. Otherwise, it uses
    future ``state`` values starting at the next frame as action targets. Values
    past the episode end repeat the last available value.

    Args:
        action_horizon: Number of timesteps to include in the action chunk.

    Raises:
        AssertionError: If required sample fields are missing or the horizon is
            not positive.
    """

    def __init__(self, action_horizon: int):
        assert action_horizon > 0, "action_horizon must be greater than 0"
        self.action_horizon = action_horizon

    def __str__(self):
        return f"BuildActionChunk(action_horizon={self.action_horizon})"

    def __call__(self, data, **kw):
        assert "state" in data, "BuildActionChunk requires 'state' in data"
        data["state"] = np.asarray(data["state"], dtype=np.float32)

        assert "raw_lines" in data and "meta_data" in data, (
            "BuildActionChunk requires 'raw_lines' and 'meta_data' in data"
        )
        lines = data["raw_lines"]
        meta = data["meta_data"]

        assert "frame_index" in meta, (
            "BuildActionChunk requires meta_data['frame_index']"
        )
        frame_index = meta["frame_index"]
        episode_term = len(lines) - 1

        if "action" in data:
            read_key = "action"
            start = frame_index
        else:
            assert "state" in data, "Either 'action' or 'state' must be present in data"
            read_key = "state"
            start = frame_index + 1

        values = []
        last_value = None
        for step in range(self.action_horizon):
            raw_idx = start + step
            if raw_idx <= episode_term:
                frame = orjson.loads(lines[raw_idx])
                last_value = np.asarray(frame[read_key], dtype=np.float32)
            assert last_value is not None, (
                "No valid action or state values found in episode"
            )
            values.append(last_value)
        assert len(values) == self.action_horizon, (
            f"Expected {self.action_horizon} values, got {len(values)}"
        )

        data["action"] = np.stack(values, axis=0)[None, ...]
        data["action_mask"] = np.ones_like(data["action"], dtype=bool)
        return data


class ActionRelative:
    """Convert absolute action targets to deltas relative to the state.

    Dimensions listed in ``non_delta_ids`` are copied from the original action
    instead of being converted to deltas. This is typically used for gripper
    commands whose target should remain absolute.

    Args:
        non_delta_ids: State descriptor names or enum values that should not be
            delta-encoded.

    Raises:
        AssertionError: If required state, action, or metadata fields are
            missing.
        ValueError: If the last dimension of ``action`` does not match
            ``state``.
    """

    def __init__(
        self,
        non_delta_ids: tuple[RobotStateDesc | str, ...] | list[RobotStateDesc | str] = (
            RobotStateDesc.GRIPPER,
        ),
    ):
        self.non_delta_ids = {_state_desc_value(desc) for desc in non_delta_ids}

    def __str__(self):
        non_delta_ids = _format_state_descs(self.non_delta_ids)
        return f"ActionRelative(non_delta_ids={non_delta_ids})"

    def __call__(self, data, **kw):
        assert "state" in data and "action" in data, (
            "Both state and action must be present to compute relative action"
        )
        state = data["state"]
        action = data["action"]
        assert isinstance(state, np.ndarray) and isinstance(
            action,
            np.ndarray,
        ), "State and action must be numpy arrays"

        if action.shape[-1] != state.shape[-1]:
            raise ValueError(
                f"action dim {action.shape[-1]} does not match "
                f"state dim {state.shape[-1]}"
            )
        relative = action - state

        assert "meta_data" in data, (
            "meta_data must be present to compute relative action"
        )
        assert "state_desc" in data["meta_data"], (
            "state_desc must be present in meta_data to compute relative action"
        )
        state_desc = data["meta_data"]["state_desc"]
        non_delta_indices = [
            i
            for i, sid in enumerate(state_desc)
            if _state_desc_value(sid) in self.non_delta_ids
        ]
        if non_delta_indices:
            relative[..., non_delta_indices] = action[..., non_delta_indices]

        data["action"] = relative
        return data


class BuildAction:
    """Build action targets in absolute or relative mode.

    This transform composes ``BuildActionChunk`` with ``ActionRelative`` when
    relative actions are requested, and only builds the chunk for absolute
    actions.

    Args:
        action_horizon: Number of future timesteps to include.
        action_mode: Action representation to produce.
        non_delta_ids: State descriptor names or enum values that should remain
            absolute when ``action_mode`` is ``ActionMode.RELATIVE``.

    Raises:
        AssertionError: If ``action_horizon`` is not positive or
            ``action_mode`` is not an ``ActionMode``.
        ValueError: If relative mode is configured without ``non_delta_ids`` or
            an unsupported action mode is provided.
    """

    def __init__(
        self,
        action_horizon: int,
        action_mode: ActionMode = ActionMode.RELATIVE,
        non_delta_ids: tuple[RobotStateDesc | str, ...] | list[RobotStateDesc | str] = (
            RobotStateDesc.GRIPPER,
        ),
    ):
        assert action_horizon > 0, "action_horizon must be greater than 0"
        assert isinstance(
            action_mode,
            ActionMode,
        ), "action_mode must be an instance of ActionMode"
        if action_mode == ActionMode.RELATIVE and not non_delta_ids:
            raise ValueError("non_delta_ids must be provided for relative action mode")

        self.action_horizon = action_horizon
        self.action_mode = action_mode
        self.non_delta_ids = {_state_desc_value(desc) for desc in non_delta_ids}

        if action_mode == ActionMode.RELATIVE:
            self.build_pipeline = self._build_relative_action_pipeline()
        elif action_mode == ActionMode.ABSOLUTE:
            self.build_pipeline = self._build_absolute_action_pipeline()
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

    def __str__(self):
        return f"BuildAction({self.build_pipeline})"

    def _build_relative_action_pipeline(self):
        return Pipeline(
            [
                BuildActionChunk(self.action_horizon),
                ActionRelative(non_delta_ids=list(self.non_delta_ids)),
            ]
        )

    def _build_absolute_action_pipeline(self):
        return Pipeline(
            [
                BuildActionChunk(self.action_horizon),
            ]
        )

    def __call__(self, data, **kw):
        return self.build_pipeline(data, **kw)


class ActionAbsolute:
    """Convert delta action targets back to absolute targets.

    The transform computes ``absolute_action = state + action`` and then
    restores dimensions listed in ``non_delta_ids`` from the original action.

    Args:
        non_delta_ids: State descriptor names or enum values that were not
            delta-encoded and should be preserved from ``action``.

    Raises:
        AssertionError: If required state, action, or metadata fields are
            missing.
    """

    def __init__(
        self,
        non_delta_ids: tuple[RobotStateDesc | str, ...] | list[RobotStateDesc | str] = (
            RobotStateDesc.GRIPPER,
        ),
    ):
        self.non_delta_ids = {_state_desc_value(desc) for desc in non_delta_ids}

    def __call__(self, data):
        assert "state" in data and "action" in data, (
            "Both state and action must be present to compute absolute action"
        )
        assert isinstance(data["state"], np.ndarray) and isinstance(
            data["action"],
            np.ndarray,
        ), "Both state and action must be numpy arrays"
        abs_action = data["state"] + data["action"]

        assert "meta_data" in data, (
            "meta_data must be present to compute absolute action"
        )
        assert "state_desc" in data["meta_data"], (
            "state_desc must be present in meta_data to compute absolute action"
        )
        state_desc = data["meta_data"]["state_desc"]
        non_delta_indices = [
            i
            for i, sid in enumerate(state_desc)
            if _state_desc_value(sid) in self.non_delta_ids
        ]
        if non_delta_indices:
            abs_action[..., non_delta_indices] = data["action"][..., non_delta_indices]

        data["action"] = abs_action
        return data


class ChatTokenization:
    """Tokenize a multimodal robot sample with a chat template.

    Current-view images are interleaved via the processor chat template.
    Optional history images use ``<unused0>`` placeholders
    (``HISTORY_TOKENS_PER_IMAGE`` per image) and are injected separately in the
    model prefix forward.

    Args:
        processor: Multimodal processor or tokenizer-compatible object with
            ``apply_chat_template``.
        n_bins: Number of bins used when discretizing normalized state values.
        max_length: Maximum token length. Long prompts are shortened to fit.
        image_prompts: Prompt labels zipped with ``data["images"]`` in order.
        add_state: Whether to append a discretized state text field.
        is_history: Whether to insert history-image placeholders and emit
            ``history_pixel_values`` / ``history_mask``.
        enable_logging: Whether to log the decoded tokenized prompt.
    """

    def __init__(
        self,
        processor,
        n_bins: int = 256,
        max_length: int | None = 1024,
        image_prompts: list[str] | None = None,
        add_state: bool = True,
        is_history: bool = False,
        enable_logging: bool = False,
    ):
        self.processor = processor
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.n_bins = n_bins
        self.max_length = max_length
        self.image_prompts = image_prompts or []
        self.add_state = add_state
        self.is_history = is_history
        self.enable_logging = enable_logging
        self.history_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            "<unused0>"
        )

    def action_to_bin_tokens(self, action: np.ndarray, n_bins: int = 256) -> list[int]:
        """Convert normalized continuous action values to integer bin IDs.

        Args:
            action: Action or state array expected to be normalized to
                ``[-1, 1]``.
            n_bins: Number of discrete bins.

        Returns:
            A Python list of integer bin IDs in ``[0, n_bins - 1]``.
        """
        clipped = np.clip(action, -1.0, 1.0)
        normalized = (clipped + 1.0) / 2.0
        bins = np.floor(normalized * (n_bins - 1)).astype(int)
        return np.clip(bins, 0, n_bins - 1).tolist()

    def __call__(self, data):
        meta = data.get("meta_data", {})
        text_parts = []
        if meta.get("robot_type") is not None:
            text_parts.append(f"Robot: {meta['robot_type']}\n")
        if meta.get("control_mode") is not None:
            text_parts.append(f"Control mode: {meta['control_mode']}\n")
        speed = meta.get("speed") or "0.5"
        assert isinstance(speed, str), (
            f"Expected speed to be a string, got {type(speed)}"
        )
        text_parts.append(f"Overall speed: {speed}\n")
        text_parts.append(f"Task: {data['prompt']}.\n")
        user_content = [{"type": "text", "text": "".join(text_parts)}]

        history_pixel_values = None
        history_mask = None
        if self.is_history:
            history_images = data.get("history_images") or []
            user_content[-1]["text"] += "History images: "
            user_content[-1]["text"] += "<unused0>" * (
                HISTORY_TOKENS_PER_IMAGE * len(history_images)
            )
            normalized = [image.convert("RGB") for image in history_images]
            if normalized:
                history_pixel_values = self.processor.image_processor(
                    images=normalized,
                    return_tensors="pt",
                )["pixel_values"]

        for prompt, image in zip(self.image_prompts, data["images"], strict=True):
            label = f"{prompt} image: "
            if user_content[-1]["type"] == "text":
                user_content[-1]["text"] += label
            else:
                user_content.append({"type": "text", "text": label})
            user_content.append({"type": "image", "image": image})
        if self.add_state:
            user_content.append(
                {
                    "type": "text",
                    "text": "States: "
                    + " ".join(
                        str(b)
                        for b in self.action_to_bin_tokens(
                            data["state"],
                            self.n_bins,
                        )
                    ),
                }
            )
        messages = [{"role": "user", "content": user_content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if self.enable_logging:
            decode_text = self.tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=False
            )
            decode_text_collapsed = re.sub(
                r"(<image_soft_token>)+",
                lambda m: (
                    f"<image_soft_token>x{m.group(0).count('<image_soft_token>')}"
                ),
                decode_text,
            )
            logger.info(f"Decoded input_ids: {decode_text_collapsed}")
        if inputs["input_ids"].shape[1] > self.max_length:
            prompt_for_truncation = data["prompt"]
            prompt_token_ids = self.tokenizer.encode(
                prompt_for_truncation,
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
                    if prompt_for_truncation not in text:
                        continue
                    content_item["text"] = text.replace(
                        prompt_for_truncation,
                        shortened_prompt,
                        1,
                    )
                    break
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
        token_type_ids = inputs["token_type_ids"]
        if self.is_history:
            history_mask = inputs["input_ids"] == self.history_placeholder_token_id
            # Match dexbotic: mark ``<unused0>`` history slots as image tokens (type=1).
            token_type_ids = token_type_ids.clone()
            token_type_ids[history_mask] = 1
        return {
            "action": torch.from_numpy(data["action"]) if "action" in data else None,
            "action_mask": (
                torch.from_numpy(data["action_mask"]) if "action_mask" in data else None
            ),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "token_type_ids": token_type_ids,
            "history_pixel_values": history_pixel_values,
            "history_mask": history_mask,
        }


class PadAction:
    """Pad or truncate the last action dimension to a shared size.

    Args:
        shared_dim: Target size for the last action dimension.
    """

    def __init__(self, shared_dim: int = 32):
        self.shared_dim = shared_dim

    def __call__(self, data, **kw):
        data["action"] = self._pad_last_dim(data["action"])
        data["action_mask"] = self._pad_last_dim(data["action_mask"])
        return data

    def _pad_last_dim(self, arr, pad_value=0):
        ndim = self.shared_dim
        if arr.shape[-1] >= ndim:
            return arr[..., :ndim]
        pad = ndim - arr.shape[-1]
        return torch.nn.functional.pad(arr, (0, pad), value=pad_value)


class ToDevice:
    """Move all tensor values in a sample dictionary to a target device.

    Args:
        device: Target PyTorch device.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)

    def __call__(self, data, **kw):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        return data


def _load_image(image_url: str) -> Image.Image:
    with megfile.smart_open(image_url, mode="rb") as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def _load_video(video_url: str, frame_idx: int) -> Image.Image:
    import av

    with megfile.smart_open(video_url, mode="rb") as f:
        with av.open(f) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if stream.average_rate is None or stream.time_base is None:
                raise RuntimeError(
                    "PyAV seek requires video average_rate and time_base"
                )
            fps = float(stream.average_rate)
            time_base = float(stream.time_base)
            container.seek(int(frame_idx / fps / time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                current_idx = int(frame.pts * time_base * fps + 0.5)
                if current_idx == frame_idx:
                    return Image.fromarray(frame.to_ndarray(format="rgb24")).convert(
                        "RGB"
                    )
                if current_idx > frame_idx:
                    break
    raise RuntimeError(f"Failed to seek/decode frame_idx={frame_idx}")
