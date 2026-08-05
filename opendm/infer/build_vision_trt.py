#!/usr/bin/env python3
"""Export and build the DM05 vision TensorRT engine for fast inference.

This script loads a DM05 checkpoint, exports the vision tower plus multimodal
projector to ONNX, and builds the static-shape TensorRT engine consumed by the
fast inference wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from opendm.infer.dm05_trt_utils import (
    build_fp16_engine_from_onnx,
    engine_shape,
    load_tensorrt,
    resolve_io_names,
)
from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration
from opendm.model.dm05.dm05_lora import load_dm05_model_for_inference


class DM05VisionFeatureModule(nn.Module):
    """Expose Gemma3 vision tower + projector as one tensor module."""

    def __init__(self, vlm_model: nn.Module) -> None:
        super().__init__()
        self.vlm_model = vlm_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_dtype = next(self.vlm_model.vision_tower.parameters()).dtype
        projector_dtype = (
            self.vlm_model.multi_modal_projector.mm_input_projection_weight.dtype
        )
        vision_outputs = self.vlm_model.vision_tower(
            pixel_values=pixel_values.to(dtype=vision_dtype),
            return_dict=True,
        )
        last_hidden_state = vision_outputs.last_hidden_state.to(dtype=projector_dtype)
        return self.vlm_model.multi_modal_projector(last_hidden_state)


def _load_model(checkpoint: Path) -> DM05ForConditionalGeneration:
    model = load_dm05_model_for_inference(
        str(checkpoint),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if not isinstance(model, DM05ForConditionalGeneration):
        raise TypeError(
            "build_vision_trt.py expects a DM05ForConditionalGeneration-compatible "
            f"checkpoint, got {type(model)!r}."
        )
    model.model.vlm.set_attn_implementation({"vision_config": "sdpa"})
    return model.eval().to(device="cuda", dtype=torch.float16)


def export_vision_onnx(
    *,
    model: DM05ForConditionalGeneration,
    onnx_path: Path,
    num_images: int,
    opset: int,
) -> tuple[int, tuple[int, ...]]:
    vlm_model = model.model.vlm.model
    vlm_model.vision_tower.to(device="cuda", dtype=torch.float16)
    vlm_model.multi_modal_projector.to(device="cuda", dtype=torch.float16)
    image_size = int(vlm_model.config.vision_config.image_size)
    wrapper = DM05VisionFeatureModule(vlm_model).eval().to("cuda")
    dummy = torch.randn(
        num_images,
        3,
        image_size,
        image_size,
        dtype=torch.float16,
        device="cuda",
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        output = wrapper(dummy)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    return image_size, tuple(int(dim) for dim in output.shape)


def _vision_engine_num_images(engine_path: Path) -> int | None:
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        return None
    trt = load_tensorrt("Inspecting DM05 vision TensorRT engine")
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        return None
    input_names, _ = resolve_io_names(engine, trt)
    if len(input_names) != 1:
        return None
    shape = engine_shape(engine, input_names[0])
    if len(shape) != 4 or shape[0] <= 0:
        return None
    return int(shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="DM05 checkpoint used to export/build the vision TensorRT engine.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Output ONNX path for the vision feature extractor.",
    )
    parser.add_argument(
        "--engine-path",
        type=Path,
        required=True,
        help="Output TensorRT engine path for the vision feature extractor.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2,
        help="Number of images supported by the vision TensorRT engine.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=8.0,
        help="TensorRT workspace size in GiB.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if a matching engine already exists.",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep the intermediate ONNX file after building.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Building the DM05 vision TensorRT engine requires CUDA.")
    if args.num_images <= 0:
        raise ValueError(f"--num-images must be positive, got {args.num_images}.")

    existing_num_images = _vision_engine_num_images(args.engine_path)
    if (
        not args.force_rebuild
        and args.engine_path.is_file()
        and args.engine_path.stat().st_size > 0
        and existing_num_images == args.num_images
    ):
        print(
            "vision engine exists with matching image count, skipping "
            f"(use --force-rebuild to override): {args.engine_path}"
        )
        return

    if args.onnx_path.exists() and not args.keep_onnx:
        args.onnx_path.unlink()
    model = _load_model(args.checkpoint)
    image_size, output_shape = export_vision_onnx(
        model=model,
        onnx_path=args.onnx_path,
        num_images=args.num_images,
        opset=args.opset,
    )
    build_fp16_engine_from_onnx(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        workspace_gb=args.workspace_gb,
        context="DM05 vision",
    )
    if args.onnx_path.exists() and not args.keep_onnx:
        args.onnx_path.unlink()
    del model
    torch.cuda.empty_cache()
    print(
        "Built DM05 vision TensorRT engine: "
        f"input_shape=({args.num_images}, 3, {image_size}, {image_size}), "
        f"output_shape={output_shape}, engine={args.engine_path}"
    )


if __name__ == "__main__":
    main()
