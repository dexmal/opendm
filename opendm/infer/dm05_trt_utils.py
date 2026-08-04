#!/usr/bin/env python3
"""TensorRT build and runtime helpers for the DM05 fast path.

This module centralizes engine loading, ONNX-to-engine export, execution
context setup, tensor/binding utilities, and the DM05 vision TensorRT runner
used during fast inference.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

from opendm.constants.robot import HISTORY_POOL_SIZE

DEFAULT_DM05_VISION_TRT_ENGINE_PATH = "checkpoints/trt_engines/dm05_vision.engine"

MAX_HISTORY_IMAGES = 5


def pool_image_features_to_history(
    image_features: torch.Tensor,
    *,
    pool_size: int = HISTORY_POOL_SIZE,
) -> torch.Tensor:
    """Pool projected soft tokens ``(N, T, H)`` to ``(N, pool_size**2, H)``.

    Default ``pool_size`` is ``sqrt(HISTORY_TOKENS_PER_IMAGE)``.
    """
    tokens = int(image_features.shape[1])
    spatial = int(tokens**0.5)
    if spatial * spatial != tokens:
        raise ValueError(
            "image_features token count must be a perfect square for 2D pooling, "
            f"got tokens={tokens}."
        )
    hidden = int(image_features.shape[-1])
    grid = image_features.view(-1, spatial, spatial, hidden).permute(0, 3, 1, 2)
    grid = F.adaptive_avg_pool2d(grid, output_size=(pool_size, pool_size))
    return grid.permute(0, 2, 3, 1).reshape(-1, pool_size * pool_size, hidden)


def pack_current_and_history_pixels(
    current_pixel_values: torch.Tensor,
    history_pixel_values: torch.Tensor | None = None,
    *,
    max_history: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Pack current views + zero-padded history slots into a fixed TRT batch.

    Returns ``(packed_pixels, num_real_history)``. Empty history slots are zero
    images; callers must ignore the corresponding TRT feature rows.
    """
    if max_history is None:
        max_history = MAX_HISTORY_IMAGES
    if current_pixel_values.ndim != 4:
        raise ValueError(
            "current_pixel_values must be (N, C, H, W), got "
            f"shape={tuple(current_pixel_values.shape)}."
        )
    num_current = int(current_pixel_values.shape[0])
    if num_current <= 0:
        raise ValueError("current_pixel_values must contain at least one image.")
    if max_history < 0:
        raise ValueError(f"max_history must be >= 0, got {max_history}.")

    num_history = 0
    if history_pixel_values is not None and int(history_pixel_values.shape[0]) > 0:
        if history_pixel_values.ndim != 4:
            raise ValueError(
                "history_pixel_values must be (N, C, H, W), got "
                f"shape={tuple(history_pixel_values.shape)}."
            )
        if tuple(history_pixel_values.shape[1:]) != tuple(
            current_pixel_values.shape[1:]
        ):
            raise ValueError(
                "history_pixel_values spatial/channel shape must match current "
                f"views: current={tuple(current_pixel_values.shape[1:])}, "
                f"history={tuple(history_pixel_values.shape[1:])}."
            )
        num_history = int(history_pixel_values.shape[0])
        if num_history > max_history:
            raise ValueError(
                f"At most {max_history} history images are supported, got "
                f"{num_history}."
            )

    packed = current_pixel_values.new_zeros(
        (num_current + max_history, *current_pixel_values.shape[1:])
    )
    packed[:num_current].copy_(current_pixel_values)
    if num_history > 0:
        packed[num_current : num_current + num_history].copy_(
            history_pixel_values.to(
                device=current_pixel_values.device,
                dtype=current_pixel_values.dtype,
            )
        )
    return packed, num_history


def load_tensorrt(context: str):
    try:
        import tensorrt as trt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(f"{context} requires the tensorrt Python package.") from exc
    return trt


def resolve_dm05_vision_trt_engine_path(
    engine_path: str | Path | None,
) -> Path:
    """Resolve the DM05 fast-path vision TensorRT engine path."""
    if engine_path:
        return Path(engine_path).expanduser()

    resolved_path = Path(DEFAULT_DM05_VISION_TRT_ENGINE_PATH).expanduser()
    logger.info(
        "Auto-selected fast vision TensorRT engine path: {}",
        resolved_path,
    )
    return resolved_path


def ensure_dm05_fast_inference_engine(
    *,
    checkpoint: str | Path | None,
    engine_path: str | Path | None,
    num_images: int,
    force_rebuild: bool = False,
) -> Path:
    """Ensure the DM05 fast backend vision TensorRT engine is ready."""
    if not checkpoint:
        raise ValueError("Fast inference requires model_config.model_name_or_path.")
    if num_images <= 0:
        raise ValueError("Fast inference requires at least one inference image key.")

    resolved_engine_path = resolve_dm05_vision_trt_engine_path(engine_path)
    return ensure_dm05_vision_trt_engine(
        checkpoint=checkpoint,
        engine_path=resolved_engine_path,
        num_images=num_images,
        force_rebuild=force_rebuild,
    )


def ensure_dm05_vision_trt_engine(
    *,
    checkpoint: str | Path,
    engine_path: str | Path,
    num_images: int,
    force_rebuild: bool = False,
) -> Path:
    """Ensure the static-shape DM05 vision TensorRT engine is ready."""
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}.")

    checkpoint_path = Path(checkpoint).expanduser()
    engine_path = Path(engine_path).expanduser()
    onnx_path = engine_path.with_suffix(".onnx")
    command = [
        sys.executable,
        "-m",
        "opendm.infer.build_vision_trt",
        "--checkpoint",
        str(checkpoint_path),
        "--onnx-path",
        str(onnx_path),
        "--engine-path",
        str(engine_path),
        "--num-images",
        str(num_images),
    ]
    if force_rebuild:
        command.append("--force-rebuild")

    logger.info(
        "Ensuring DM05 fast vision TensorRT engine via: {}",
        shlex.join(command),
    )
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to prepare DM05 vision TensorRT engine: {engine_path}"
        ) from exc

    return engine_path


def resolve_cuda_device(device: torch.device | str, context: str) -> torch.device:
    requested_device = torch.device(device)
    if requested_device.type != "cuda":
        raise RuntimeError(f"{context} requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{context} requires CUDA.")
    if requested_device.index is None:
        requested_device = torch.device("cuda", torch.cuda.current_device())
    return requested_device


def deserialize_engine(
    trt: Any,
    engine_path: str | Path,
    context: str,
) -> tuple[Any, Any]:
    engine_path = Path(engine_path)
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize {context} TensorRT engine: {engine_path}"
        )
    return logger, engine


def build_fp16_engine_from_onnx(
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    workspace_gb: float,
    context: str,
) -> None:
    trt = load_tensorrt(f"{context} TensorRT export")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
    else:
        network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    onnx_path = Path(onnx_path)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(idx)) for idx in range(parser.num_errors)]
        raise RuntimeError(
            f"Failed to parse {context} ONNX for TensorRT:\n" + "\n".join(errors)
        )

    config = builder.create_builder_config()
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    workspace_bytes = int(workspace_gb * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized = None if engine is None else engine.serialize()
    if serialized is None:
        raise RuntimeError(f"Failed to build {context} TensorRT engine.")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def create_execution_context(
    engine: Any,
    context: str,
    *,
    use_external_device_memory: bool = False,
) -> Any:
    if use_external_device_memory:
        execution_context = engine.create_execution_context_without_device_memory()
    else:
        execution_context = engine.create_execution_context()
    if execution_context is None:
        raise RuntimeError(f"Failed to create {context} TensorRT context.")
    return execution_context


def has_tensor_api(engine: Any) -> bool:
    return hasattr(engine, "num_io_tensors")


def resolve_io_names(engine: Any, trt: Any) -> tuple[list[str], list[str]]:
    inputs: list[str] = []
    outputs: list[str] = []
    if has_tensor_api(engine):
        for idx in range(int(engine.num_io_tensors)):
            name = engine.get_tensor_name(idx)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
    else:
        for idx in range(int(engine.num_bindings)):
            name = engine.get_binding_name(idx)
            if engine.binding_is_input(idx):
                inputs.append(name)
            else:
                outputs.append(name)
    return inputs, outputs


def binding_index(engine: Any, name: str) -> int:
    if hasattr(engine, "get_binding_index"):
        return int(engine.get_binding_index(name))
    raise RuntimeError("TensorRT binding-index API is unavailable.")


def engine_shape(engine: Any, name: str) -> tuple[int, ...]:
    if has_tensor_api(engine):
        return tuple(int(dim) for dim in engine.get_tensor_shape(name))
    return tuple(
        int(dim) for dim in engine.get_binding_shape(binding_index(engine, name))
    )


def runtime_shape(engine: Any, context: Any, name: str) -> tuple[int, ...]:
    if has_tensor_api(engine):
        return tuple(int(dim) for dim in context.get_tensor_shape(name))
    return tuple(
        int(dim) for dim in context.get_binding_shape(binding_index(engine, name))
    )


def trt_dtype_to_torch(trt: Any, trt_dtype: Any) -> torch.dtype:
    mapping = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "int64"):
        mapping[trt.int64] = torch.int64
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    return mapping[trt_dtype]


def tensor_dtype(engine: Any, trt: Any, name: str) -> torch.dtype:
    if has_tensor_api(engine):
        trt_dtype = engine.get_tensor_dtype(name)
    else:
        trt_dtype = engine.get_binding_dtype(binding_index(engine, name))
    return trt_dtype_to_torch(trt, trt_dtype)


def set_input_shape(
    engine: Any,
    context: Any,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if has_tensor_api(engine):
        if any(dim < 0 for dim in engine_shape(engine, name)):
            context.set_input_shape(name, shape)
    else:
        idx = binding_index(engine, name)
        engine_shape_value = tuple(int(dim) for dim in engine.get_binding_shape(idx))
        if any(dim < 0 for dim in engine_shape_value):
            context.set_binding_shape(idx, shape)


def device_memory_size(engine: Any) -> int:
    return int(
        engine.get_device_memory_size_v2()
        if hasattr(engine, "get_device_memory_size_v2")
        else engine.device_memory_size
    )


def set_context_device_memory(context: Any, device_memory: torch.Tensor) -> None:
    if hasattr(context, "set_device_memory"):
        try:
            context.set_device_memory(int(device_memory.data_ptr()))
        except TypeError:
            context.device_memory = int(device_memory.data_ptr())
    else:
        context.device_memory = int(device_memory.data_ptr())


def assign_device_memory(
    *,
    context: Any,
    device_memory: torch.Tensor,
    device: torch.device,
    required_size: int,
    label: str,
) -> torch.Tensor:
    if not device_memory.is_cuda:
        raise RuntimeError(f"{label} device memory must be CUDA.")
    if device_memory.device != device:
        device_memory = device_memory.to(device)
    if not device_memory.is_contiguous():
        device_memory = device_memory.contiguous()
    device_memory_bytes = int(device_memory.numel() * device_memory.element_size())
    if device_memory_bytes < int(required_size):
        raise RuntimeError(
            f"Shared {label} device memory is too small: "
            f"need {required_size} bytes, got {device_memory_bytes}."
        )
    set_context_device_memory(context, device_memory)
    return device_memory


def prepare_engine_input_tensor(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    label: str,
    require_cuda: bool = True,
) -> torch.Tensor:
    if tensor.device != device:
        tensor = tensor.to(device)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def execute_async(
    *,
    engine: Any,
    context: Any,
    device: torch.device,
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
) -> bool:
    stream = torch.cuda.current_stream(device)
    if has_tensor_api(engine):
        for name, tensor in inputs.items():
            context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            context.set_tensor_address(name, int(tensor.data_ptr()))
        return bool(context.execute_async_v3(stream_handle=stream.cuda_stream))

    bindings = [0] * int(engine.num_bindings)
    for name, tensor in inputs.items():
        bindings[binding_index(engine, name)] = int(tensor.data_ptr())
    for name, tensor in outputs.items():
        bindings[binding_index(engine, name)] = int(tensor.data_ptr())
    return bool(
        context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream.cuda_stream,
        )
    )


class _TensorRTRunnerBase:
    """Shared TensorRT engine/context setup for fixed DM05 runners."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | str,
        runtime_context: str,
        engine_context: str,
        device_memory: torch.Tensor | None = None,
        use_external_device_memory: bool = False,
    ) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(
                f"{engine_context} TensorRT engine not found: {self.engine_path}"
            )
        self.device = resolve_cuda_device(device, runtime_context)
        self.trt = load_tensorrt(runtime_context)
        self.logger, self.engine = deserialize_engine(
            self.trt,
            self.engine_path,
            engine_context,
        )
        self.device_memory_size = device_memory_size(self.engine)
        self.uses_external_device_memory = bool(
            use_external_device_memory or device_memory is not None
        )
        self.context = create_execution_context(
            self.engine,
            engine_context,
            use_external_device_memory=self.uses_external_device_memory,
        )
        self.device_memory = None
        if device_memory is not None:
            self.set_device_memory(device_memory)
        self.input_names, self.output_names = resolve_io_names(self.engine, self.trt)
        self.input_dtypes = {
            name: tensor_dtype(self.engine, self.trt, name) for name in self.input_names
        }
        self.output_dtypes = {
            name: tensor_dtype(self.engine, self.trt, name)
            for name in self.output_names
        }

    def set_device_memory(self, device_memory: torch.Tensor) -> None:
        self.device_memory = assign_device_memory(
            context=self.context,
            device_memory=device_memory,
            device=self.device,
            required_size=self.device_memory_size,
            label=f"{self.engine_path.name} TensorRT",
        )


class DM05VisionTensorRTRunner(_TensorRTRunnerBase):
    """Run a static-shape TensorRT engine that returns DM05 image features."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(
            engine_path,
            device=device,
            runtime_context="DM05 vision TensorRT inference",
            engine_context="DM05 vision",
        )
        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise RuntimeError(
                "DM05 vision TensorRT engine must have exactly one input and one "
                f"output, got inputs={self.input_names}, outputs={self.output_names}."
            )
        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        self.input_dtype = self.input_dtypes[self.input_name]
        self.output_dtype = self.output_dtypes[self.output_name]
        self.input_shape = engine_shape(self.engine, self.input_name)
        self.output_shape = engine_shape(self.engine, self.output_name)
        if len(self.input_shape) != 4 or any(dim <= 0 for dim in self.input_shape):
            raise RuntimeError(
                "DM05 vision TensorRT engine must have a static 4D input "
                f"shape, got {self.input_shape}."
            )
        self.num_images = int(self.input_shape[0])

    def _set_dynamic_input_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        set_input_shape(self.engine, self.context, self.input_name, shape)
        output_shape = runtime_shape(self.engine, self.context, self.output_name)
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(
                "DM05 vision TensorRT output shape is still dynamic after setting "
                f"input shape {shape}: {output_shape}."
            )
        return output_shape

    @torch.inference_mode()
    def __call__(
        self,
        pixel_values: torch.Tensor,
        *,
        output_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_tensor = prepare_engine_input_tensor(
            pixel_values,
            device=self.device,
            dtype=self.input_dtype,
            label="DM05 vision TensorRT input",
        )
        input_shape = tuple(int(dim) for dim in input_tensor.shape)
        if len(input_shape) != 4:
            raise ValueError(
                "DM05 vision TensorRT input must be a 4D image tensor, "
                f"got shape {input_shape}."
            )
        if int(input_shape[0]) != self.num_images:
            raise ValueError(
                "DM05 vision TensorRT image count mismatch: engine was "
                f"built for {self.num_images} images, got {input_shape[0]}."
            )
        expected_shape = tuple(int(dim) for dim in self.input_shape)
        if not any(dim < 0 for dim in expected_shape) and input_shape != expected_shape:
            raise ValueError(
                "DM05 vision TensorRT input shape mismatch: "
                f"expected {expected_shape}, got {input_shape}."
            )
        output_shape = self._set_dynamic_input_shape(input_shape)
        if output_tensor is None:
            output_tensor = torch.empty(
                output_shape,
                device=self.device,
                dtype=self.output_dtype,
            )
        elif (
            tuple(output_tensor.shape) != tuple(output_shape)
            or output_tensor.device != self.device
            or output_tensor.dtype != self.output_dtype
            or not output_tensor.is_contiguous()
        ):
            raise ValueError(
                "DM05 vision TensorRT output tensor does not match the engine: "
                f"expected shape={output_shape}, dtype={self.output_dtype}, "
                f"device={self.device}, contiguous=True; got "
                f"shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, "
                f"device={output_tensor.device}, contiguous={output_tensor.is_contiguous()}."
            )
        ok = execute_async(
            engine=self.engine,
            context=self.context,
            device=self.device,
            inputs={self.input_name: input_tensor},
            outputs={self.output_name: output_tensor},
        )
        if not ok:
            raise RuntimeError("DM05 vision TensorRT execution failed.")
        return output_tensor
