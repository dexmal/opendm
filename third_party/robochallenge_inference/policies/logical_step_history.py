"""Logical-step history: action-step-indexed slots with unused0/unused1 pads.

Valid slots become real ``history_images`` (oldest→newest). Invalid slots are
emitted as ``<unused1>`` text pads with no vision features, so opendm can stay
on its ``history_images`` + ``<unused0>`` scatter path.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

HISTORY_PLACEHOLDER_TOKEN = "<unused0>"
HISTORY_PAD_TOKEN = "<unused1>"

LOGICAL_STEP_MODES = frozenset({"logical_step"})


@dataclass(frozen=True)
class HistoryFrame:
    action_step: int
    chunk_idx: int
    image: Image.Image


@dataclass
class LogicalStepHistoryConfig:
    raw_slots: int = 5
    tokens_per_slot: int = 16
    # Slot grid step (usually matches action playback target steps).
    slot_step: int = 25
    runtime_fps: float = 30.0
    step_tolerance: int = 1

    @property
    def uniform_fps(self) -> float:
        return float(self.runtime_fps) / float(self.slot_step)

    @property
    def offset_seconds(self) -> list[float]:
        fps = self.uniform_fps
        return [idx / fps for idx in range(self.raw_slots, 0, -1)]

    @property
    def max_history_steps(self) -> int:
        return int(round(self.raw_slots / self.uniform_fps * self.runtime_fps))


def build_history_placeholder_text(
    active_mask: list[bool], tokens_per_slot: int
) -> str:
    parts: list[str] = []
    for is_valid in active_mask:
        token = HISTORY_PLACEHOLDER_TOKEN if is_valid else HISTORY_PAD_TOKEN
        parts.append(token * int(tokens_per_slot))
        if is_valid:
            parts.append("\n")
    return "".join(parts)


class LogicalStepHistoryStore:
    """Per-session main-view history indexed by executed action steps."""

    def __init__(self, config: LogicalStepHistoryConfig) -> None:
        self.config = config
        self._buffers: dict[str, list[HistoryFrame]] = {}

    def clear(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)

    def append(
        self,
        session_id: str,
        image: Image.Image,
        action_step: int,
        chunk_idx: int,
    ) -> int:
        action_step = int(action_step)
        frame = HistoryFrame(
            action_step=action_step,
            chunk_idx=int(chunk_idx),
            image=image.convert("RGB").copy(),
        )
        frames = self._buffers.setdefault(session_id, [])
        for idx, existing in enumerate(frames):
            if existing.action_step == action_step:
                frames[idx] = frame
                self._prune(frames, latest_step=action_step)
                return len(frames)
        frames.append(frame)
        frames.sort(key=lambda item: item.action_step)
        self._prune(frames, latest_step=action_step)
        return len(frames)

    def build_slot_images(
        self,
        session_id: str,
        current_action_step: int,
    ) -> tuple[list[Image.Image | None], list[bool]]:
        raw_frames: list[Image.Image | None] = []
        valid_mask: list[bool] = []
        for offset_seconds in self.config.offset_seconds:
            info = self.fetch_frame_info(
                session_id=session_id,
                current_action_step=current_action_step,
                offset_seconds=offset_seconds,
            )
            frame = info["image"]
            raw_frames.append(frame)
            valid_mask.append(frame is not None)
        return raw_frames, valid_mask

    def fetch_frame_info(
        self,
        session_id: str,
        current_action_step: int,
        offset_seconds: float,
    ) -> dict:
        offset_steps = int(
            round(float(offset_seconds) * float(self.config.runtime_fps))
        )
        target_step = int(current_action_step) - offset_steps
        frames = self._buffers.get(session_id, [])
        candidate = None
        for frame in frames:
            if frame.action_step <= target_step:
                candidate = frame
            else:
                break

        info = {
            "offset_seconds": float(offset_seconds),
            "offset_steps": offset_steps,
            "target_step": target_step,
            "valid": False,
            "source_action_step": None,
            "source_chunk_idx": None,
            "age_steps": None,
            "image": None,
        }
        if candidate is None:
            return info

        age_steps = target_step - candidate.action_step
        if age_steps > self.config.step_tolerance:
            return info

        info.update(
            {
                "valid": True,
                "source_action_step": candidate.action_step,
                "source_chunk_idx": candidate.chunk_idx,
                "age_steps": age_steps,
                "image": candidate.image.copy(),
            }
        )
        return info

    def _prune(self, frames: list[HistoryFrame], latest_step: int) -> None:
        keep_from = (
            latest_step - self.config.max_history_steps - self.config.step_tolerance
        )
        keep = [frame for frame in frames if frame.action_step >= keep_from]
        frames.clear()
        frames.extend(keep)
