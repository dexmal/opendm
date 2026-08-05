"""Service wrapper for bucketed DM05 fast inference.

The runtime executes the TensorRT vision encoder outside CUDA Graphs, then
replays startup-captured prefix-prefill and suffix-decode graphs selected by
the processed multimodal prefix length.
"""

import bisect
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers.cache_utils import Cache, CacheLayerMixin

from opendm.infer.dm05_infer_arch import DM05FastForCausalLM
from opendm.infer.dm05_trt_utils import DM05VisionTensorRTRunner
from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration


class StaticPrefixCacheLayer(CacheLayerMixin):
    """Address-stable overwrite cache layer used by suffix graph replay."""

    is_sliding = False

    def __init__(self) -> None:
        super().__init__()
        self.keys = None
        self.values = None
        self.seq_len = 0

    def reset_for_prefill(self) -> None:
        self.seq_len = 0

    def reset(self) -> None:
        self.reset_for_prefill()

    def _allocate_like(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self.keys = torch.empty(
            tuple(key_states.shape),
            device=key_states.device,
            dtype=key_states.dtype,
        )
        self.values = torch.empty(
            tuple(value_states.shape),
            device=value_states.device,
            dtype=value_states.dtype,
        )
        if self.keys.is_cuda:
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self._allocate_like(key_states, value_states)
        self.seq_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needs_realloc = (
            self.keys is None
            or self.values is None
            or self.keys.shape != key_states.shape
            or self.values.shape != value_states.shape
            or self.keys.dtype != key_states.dtype
            or self.values.dtype != value_states.dtype
            or self.keys.device != key_states.device
            or self.values.device != value_states.device
        )
        if needs_realloc:
            self._allocate_like(key_states, value_states)

        self.keys.copy_(key_states)
        self.values.copy_(value_states)
        self.seq_len = int(key_states.shape[-2])
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = int(cache_position.shape[0])
        return self.seq_len + query_length, 0

    def get_seq_length(self) -> int:
        return int(self.seq_len)

    def get_max_cache_shape(self) -> int:
        if self.keys is None:
            return -1
        return int(self.keys.shape[-2])


@dataclass
class DM05StaticGraphProfile:
    bucket_len: int
    prefix_cache: Cache
    prefix_buffers: Any
    suffix_buffers: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    image_features: torch.Tensor
    noise: torch.Tensor
    action_mask: torch.Tensor
    output: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    capture_count: int = 0


class DM05FastInferRuntime:
    """Service-facing fast backend with startup-captured prefix buckets."""

    DEFAULT_PREFIX_SEQ_LEN_BUCKETS = (576, 704, 768, 896, 1024)

    @classmethod
    def resolve_prefix_seq_len_buckets(
        cls,
        prefix_seq_len_buckets: list[int] | tuple[int, ...] | None,
        *,
        fast_backend: bool = True,
        prefix_capacity: int = 1024,
        minimum_prefix_len: int = 1,
    ) -> tuple[int, ...]:
        if not fast_backend:
            if prefix_seq_len_buckets:
                raise ValueError(
                    "Static prefix buckets require --inference-config.backend fast."
                )
            return ()

        if prefix_seq_len_buckets is not None and not prefix_seq_len_buckets:
            raise ValueError(
                "prefix_seq_len_buckets must contain at least one bucket for the "
                "fast backend."
            )

        use_defaults = prefix_seq_len_buckets is None
        buckets = (
            cls.DEFAULT_PREFIX_SEQ_LEN_BUCKETS
            if use_defaults
            else tuple(int(value) for value in prefix_seq_len_buckets)
        )
        if buckets != tuple(sorted(set(buckets))):
            raise ValueError(
                "prefix_seq_len_buckets must be strictly increasing and unique, "
                f"got {list(buckets)}."
            )
        invalid_buckets = [
            value for value in buckets if not 0 < value <= int(prefix_capacity)
        ]
        if invalid_buckets:
            raise ValueError(
                "prefix_seq_len_buckets must be within the fast backend prefix "
                f"capacity 1..{prefix_capacity}, got {invalid_buckets}."
            )

        too_small = [value for value in buckets if value < int(minimum_prefix_len)]
        if too_small and not use_defaults:
            raise ValueError(
                "Static prefix buckets are too small for the configured image tokens: "
                f"minimum={minimum_prefix_len}, got={too_small}."
            )
        if use_defaults:
            buckets = tuple(
                value for value in buckets if value >= int(minimum_prefix_len)
            )
            if not buckets:
                raise ValueError(
                    "No default prefix bucket can fit the configured image tokens: "
                    f"minimum={minimum_prefix_len}, capacity={prefix_capacity}."
                )
        return buckets

    def __init__(
        self,
        model: DM05ForConditionalGeneration,
        vision_trt_engine_path: str | Path,
        *,
        prefix_seq_len_buckets: list[int] | tuple[int, ...] | None = None,
        diffusion_steps: int = 10,
    ) -> None:
        self.model = DM05FastForCausalLM(model)
        self.device = next(model.parameters()).device
        self.dm05 = self.model.dm05
        self.action_expert = self.model.action_expert
        self.padding_idx = self.model.padding_idx
        self.prefix_len = self.model.prefix_len
        self.suffix_len = self.model.suffix_len
        self.action_dim = self.model.action_dim
        self.noise_dtype = self.model.noise_dtype
        self.graph_diffusion_steps = int(diffusion_steps)
        if self.graph_diffusion_steps <= 0:
            raise ValueError(
                f"diffusion_steps must be positive, got {self.graph_diffusion_steps}."
            )
        self._inference_lock = threading.Lock()
        self.graph_profiles: dict[int, DM05StaticGraphProfile] = {}
        self.startup_capture_count = 0
        self.dynamic_fallback_count = 0
        self.prefix_cache = self._make_static_prefix_cache()
        self.vision_trt_engine_path = vision_trt_engine_path
        self.vision_trt_runner = DM05VisionTensorRTRunner(
            self.vision_trt_engine_path,
            device=self.device,
        )
        self.vision_trt_num_images = int(self.vision_trt_runner.num_images)
        tokens_per_image = int(self.dm05.vlm.model.config.mm_tokens_per_image)
        minimum_prefix_len = self.vision_trt_num_images * (tokens_per_image + 1)
        self.prefix_seq_len_buckets = self.resolve_prefix_seq_len_buckets(
            prefix_seq_len_buckets,
            prefix_capacity=self.prefix_len,
            minimum_prefix_len=minimum_prefix_len,
        )
        self._initialize_graph_profiles()

    def _make_static_prefix_cache(self) -> Cache:
        cache_config = self.dm05.language_model.config
        layer_types = getattr(cache_config, "layer_types", None)
        if layer_types is not None:
            n_layers = len(layer_types)
        else:
            n_layers = int(
                getattr(
                    cache_config, "num_hidden_layers", len(self.action_expert.layers)
                )
            )
        if hasattr(cache_config, "num_kv_shared_layers"):
            n_layers -= int(cache_config.num_kv_shared_layers)
        return Cache(layers=[StaticPrefixCacheLayer() for _ in range(n_layers)])

    def _initialize_prefix_fastpath(self) -> None:
        self.model.setup_fast_prefix_decoder()

    @staticmethod
    def _mark_static_tensors(*tensors: torch.Tensor) -> None:
        for tensor in tensors:
            if tensor.is_cuda:
                torch._dynamo.mark_static_address(tensor)

    @torch.inference_mode()
    def _initialize_graph_profiles(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Static prefix bucket profiles require CUDA.")
        self._initialize_prefix_fastpath()

        vlm_model = self.dm05.vlm.model
        image_token_id = int(vlm_model.config.image_token_id)
        tokens_per_image = int(vlm_model.config.mm_tokens_per_image)

        dummy_pixels = torch.zeros(
            self.vision_trt_runner.input_shape,
            device=self.device,
            dtype=self.vision_trt_runner.input_dtype,
        )
        dummy_image_features = self.vision_trt_runner(dummy_pixels)
        self.model._get_suffix_time_modulations(
            batch_size=1,
            diffusion_steps=self.graph_diffusion_steps,
            dtype=self.noise_dtype,
            device=self.device,
        )

        text_config = vlm_model.config.get_text_config()
        regular_token_id = int(getattr(text_config, "bos_token_id", 2))
        for bucket_len in self.prefix_seq_len_buckets:
            profile = self._make_graph_profile(
                bucket_len=bucket_len,
                image_token_id=image_token_id,
                tokens_per_image=tokens_per_image,
                regular_token_id=regular_token_id,
                dummy_image_features=dummy_image_features,
            )
            self._capture_graph_profile(profile)
            self.graph_profiles[bucket_len] = profile

        logger.info(
            "Pre-captured {} DM05 prefix+suffix graph profiles for buckets {}",
            len(self.graph_profiles),
            list(self.prefix_seq_len_buckets),
        )

    def _make_graph_profile(
        self,
        *,
        bucket_len: int,
        image_token_id: int,
        tokens_per_image: int,
        regular_token_id: int,
        dummy_image_features: torch.Tensor,
    ) -> DM05StaticGraphProfile:
        input_ids = torch.full(
            (1, bucket_len),
            self.padding_idx,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        cursor = 0
        input_ids[0, cursor] = regular_token_id
        attention_mask[0, cursor] = 1
        cursor += 1
        for image_idx in range(self.vision_trt_num_images):
            image_end = cursor + tokens_per_image
            input_ids[0, cursor:image_end] = image_token_id
            attention_mask[0, cursor:image_end] = 1
            token_type_ids[0, cursor:image_end] = 1
            cursor = image_end
            if image_idx + 1 < self.vision_trt_num_images:
                input_ids[0, cursor] = regular_token_id
                attention_mask[0, cursor] = 1
                cursor += 1

        image_features = torch.empty_like(dummy_image_features)
        image_features.copy_(dummy_image_features)
        noise = torch.zeros(
            (1, self.suffix_len, self.action_dim),
            dtype=self.noise_dtype,
            device=self.device,
        )
        action_mask = torch.ones(
            (1, 1, self.action_dim),
            dtype=self.noise_dtype,
            device=self.device,
        )
        output = torch.empty_like(noise)
        prefix_buffers = self.action_expert.make_fast_prefix_buffers(bucket_len)
        suffix_buffers = self.action_expert.make_fast_suffix_buffers(
            batch_size=1,
            seq_len=self.suffix_len,
            max_kv_len=bucket_len + self.suffix_len,
            dtype=self.noise_dtype,
            device=self.device,
        )
        self._mark_static_tensors(
            input_ids,
            attention_mask,
            token_type_ids,
            image_features,
            noise,
            action_mask,
            output,
        )
        return DM05StaticGraphProfile(
            bucket_len=bucket_len,
            prefix_cache=self._make_static_prefix_cache(),
            prefix_buffers=prefix_buffers,
            suffix_buffers=suffix_buffers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_features=image_features,
            noise=noise,
            action_mask=action_mask,
            output=output,
        )

    def _run_graph_profile_ops(self, profile: DM05StaticGraphProfile) -> None:
        valid = profile.attention_mask.to(dtype=torch.bool)
        position_ids = (valid.to(torch.long).cumsum(dim=-1) - 1).clamp_min_(0)
        prefix_cache, prefix_len = self.model.prefill_fast_from_image_features(
            input_ids=profile.input_ids,
            attention_mask=profile.attention_mask,
            position_ids=position_ids,
            image_features=profile.image_features,
            token_type_ids=profile.token_type_ids,
            prefix_cache=profile.prefix_cache,
            prefix_buffers=profile.prefix_buffers,
        )
        prefix_visible_mask = self.model.build_prefix_visible_mask(
            input_ids=profile.input_ids,
            attention_mask=profile.attention_mask,
            prefix_len=prefix_len,
        )
        effective_prefix_len = prefix_visible_mask.to(torch.long).sum(dim=1)
        suffix_position_ids = (
            effective_prefix_len[:, None]
            + torch.arange(
                self.suffix_len,
                device=self.device,
            )[None, :]
        )
        context = self.model.prepare_fast_suffix_context(
            prefix_cache=prefix_cache,
            prefix_visible_mask=prefix_visible_mask,
            suffix_position_ids=suffix_position_ids,
            initial_noise=profile.noise,
            diffusion_steps=self.graph_diffusion_steps,
            suffix_buffers=profile.suffix_buffers,
        )
        output = self.model.decode_fast_prepared(
            initial_noise=profile.noise,
            diffusion_steps=self.graph_diffusion_steps,
            action_mask=profile.action_mask,
            **context,
        )
        profile.output.copy_(output)

    def _capture_graph_profile(self, profile: DM05StaticGraphProfile) -> None:
        torch.cuda.synchronize(self.device)
        for _ in range(2):
            self._run_graph_profile_ops(profile)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_graph_profile_ops(profile)
        profile.graph = graph
        profile.capture_count += 1
        self.startup_capture_count += 1
        graph.replay()
        torch.cuda.synchronize(self.device)
        if not bool(torch.isfinite(profile.output).all().item()):
            raise RuntimeError(
                f"Static prefix bucket {profile.bucket_len} produced non-finite output "
                "during startup capture."
            )

    @torch.inference_mode()
    def infer(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        diffusion_input_noise: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        with self._inference_lock:
            return self._forward_locked(
                images=images,
                input_ids=input_ids,
                diffusion_input_noise=diffusion_input_noise,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                diffusion_steps=diffusion_steps,
                action_mask=action_mask,
            )

    def _forward_locked(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        diffusion_input_noise: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device, dtype=torch.long)
        images = images.to(self.device, dtype=self.vision_trt_runner.input_dtype)
        diffusion_input_noise = diffusion_input_noise.to(
            self.device,
            dtype=self.noise_dtype,
        )
        action_mask = action_mask.to(self.device, dtype=self.noise_dtype)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
        input_ids, attention_mask, token_type_ids = self.model._validate_prefix_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if int(input_ids.shape[0]) != 1:
            raise ValueError(
                "Static prefix graph profiles require batch size 1, got "
                f"{int(input_ids.shape[0])}."
            )
        if int(diffusion_steps) != self.graph_diffusion_steps:
            raise ValueError(
                "Static graph profiles were pre-captured for "
                f"diffusion_steps={self.graph_diffusion_steps}, got {diffusion_steps}."
            )
        profile = self._select_graph_profile(int(input_ids.shape[1]))
        if profile is not None:
            return self._replay_graph_profile(
                profile,
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                diffusion_input_noise=diffusion_input_noise,
                action_mask=action_mask,
            )
        self.dynamic_fallback_count += 1
        if self.dynamic_fallback_count == 1:
            logger.warning(
                "Prefix length {} exceeds the largest static bucket {}; "
                "using uncaptured eager fallback",
                int(input_ids.shape[1]),
                self.prefix_seq_len_buckets[-1],
            )
        return self._run_dynamic_without_capture(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            diffusion_input_noise=diffusion_input_noise,
            diffusion_steps=diffusion_steps,
            action_mask=action_mask,
        )

    def _select_graph_profile(self, request_len: int) -> DM05StaticGraphProfile | None:
        index = bisect.bisect_left(self.prefix_seq_len_buckets, int(request_len))
        if index >= len(self.prefix_seq_len_buckets):
            return None
        return self.graph_profiles[self.prefix_seq_len_buckets[index]]

    @contextmanager
    def _prefix_attention_backend(self, implementation: str):
        vlm_model = self.dm05.vlm.model
        configs = [vlm_model.config.get_text_config(), vlm_model.language_model.config]
        unique_configs = {id(config): config for config in configs}.values()
        missing = object()
        saved = []
        for config in unique_configs:
            values = {
                name: getattr(config, name, missing)
                for name in ("_attn_implementation", "attn_implementation")
            }
            saved.append((config, values))
            config._attn_implementation = implementation
            config.attn_implementation = implementation
        try:
            yield
        finally:
            for config, values in saved:
                for name, value in values.items():
                    if value is missing:
                        delattr(config, name)
                    else:
                        setattr(config, name, value)

    def _replay_graph_profile(
        self,
        profile: DM05StaticGraphProfile,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        diffusion_input_noise: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if profile.graph is None:
            raise RuntimeError(
                f"Static prefix bucket {profile.bucket_len} was not captured at startup."
            )
        request_len = int(input_ids.shape[1])
        if request_len > profile.bucket_len:
            raise ValueError(
                f"Request length {request_len} exceeds bucket {profile.bucket_len}."
            )
        expected_noise_shape = (1, self.suffix_len, self.action_dim)
        if tuple(diffusion_input_noise.shape) != expected_noise_shape:
            raise ValueError(
                "Static graph diffusion noise has the wrong shape: expected "
                f"{expected_noise_shape}, got {tuple(diffusion_input_noise.shape)}."
            )
        if tuple(action_mask.shape) != tuple(profile.action_mask.shape):
            raise ValueError(
                "Static graph action mask has the wrong shape: expected "
                f"{tuple(profile.action_mask.shape)}, got {tuple(action_mask.shape)}."
            )

        expected_image_tokens = int(
            profile.image_features.numel() // profile.image_features.shape[-1]
        )
        image_token_id = int(self.dm05.vlm.model.config.image_token_id)
        image_token_count = int((input_ids == image_token_id).sum().item())
        if image_token_count != expected_image_tokens:
            raise ValueError(
                "Image features and image tokens do not match for bucketed prefix "
                f"inference: image_tokens={image_token_count}, "
                f"features={expected_image_tokens}."
            )

        profile.input_ids.fill_(self.padding_idx)
        profile.input_ids[:, :request_len].copy_(input_ids)
        profile.attention_mask.zero_()
        profile.attention_mask[:, :request_len].copy_(attention_mask)
        profile.token_type_ids.zero_()
        profile.token_type_ids[:, :request_len].copy_(token_type_ids)
        profile.noise.copy_(diffusion_input_noise)
        profile.action_mask.copy_(action_mask)
        self.vision_trt_runner(images, output_tensor=profile.image_features)
        profile.graph.replay()
        return profile.output

    def _run_dynamic_without_capture(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        diffusion_input_noise: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Dynamic fallback must not trigger shape-specific Flex compilation in
        # the request path. The lock around forward makes this scoped config
        # switch safe for the shared model.
        with self._prefix_attention_backend("eager"):
            image_features = self.vision_trt_runner(images)
            prefix_cache, prefix_len = self.model.prefill_from_image_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_features=image_features,
                token_type_ids=token_type_ids,
                prefix_cache=self.prefix_cache,
            )
        return self.model.decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_cache=prefix_cache,
            prefix_len=prefix_len,
            diffusion_input_noise=diffusion_input_noise,
            diffusion_steps=diffusion_steps,
            action_mask=action_mask,
        )

    @torch.inference_mode()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        diffusion_steps: int = 10,
        past_key_values: Cache | None = None,
        action_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del states, image_masks, past_key_values, kwargs
        if action_mask is None:
            raise ValueError("action_mask is required for fast inference.")

        diffusion_input_noise = torch.randn(
            int(input_ids.shape[0]),
            int(self.suffix_len),
            int(self.action_dim),
            dtype=self.noise_dtype,
            device=self.device,
        )
        return self.infer(
            images=pixel_values,
            input_ids=input_ids,
            diffusion_input_noise=diffusion_input_noise,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            diffusion_steps=diffusion_steps,
            action_mask=action_mask,
        )
