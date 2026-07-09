from typing import Any, Literal

import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.type_definitions import Targets
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    BasicTransform,
)
import albumentations as A


class PadToSquare(A.DualTransform):
    """Pad the input image to make it square."""

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

    def __init__(
        self,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.fill = fill
        self.fill_mask = fill_mask
        self.border_mode = border_mode

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        height, width = img.shape[:2]
        size = max(height, width)
        pad_top = (size - height) // 2
        pad_bottom = size - height - pad_top
        pad_left = (size - width) // 2
        pad_right = size - width - pad_left
        return fgeometric.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill,
        )


class TransformPipeline:
    """Albumentations-style image augmentation pipeline."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.steps = self.build_steps()
        self.transforms = self.steps
        self.compose = A.Compose(self.steps)

    def build_steps(self) -> list[BasicTransform]:
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return self.compose(**kwargs)


class TrainingTransformPipeline(TransformPipeline):
    """Training-time augmentation pipeline."""

    def build_steps(self) -> list[BasicTransform]:
        return [
            PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
            A.Resize(448, 448, p=1.0),
            A.RandomResizedCrop(
                size=(448, 448), scale=(0.95, 0.95), ratio=(1.0, 1.0), p=self.p
            ),
            A.Rotate(limit=(-2, 2), p=self.p),
            A.ColorJitter(
                brightness=0.3, contrast=0.4, saturation=0.3, hue=0.0, p=self.p
            ),
        ]


class NoAugmentationPipeline(TransformPipeline):
    """No augmentation pipeline."""

    def build_steps(self) -> list[BasicTransform]:
        return [
            PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
            A.Resize(448, 448, p=1.0),
        ]
