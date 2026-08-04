"""Fast DM05 model-side prefill and suffix decode helpers.

This module adapts ``DM05ForConditionalGeneration`` into an inference-only
schedule that reuses prefix KV cache state, precomputes suffix decode context,
and dispatches the Triton big-kernel suffix path on CUDA.
"""

import copy
import inspect
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

try:
    from transformers.integrations.flex_attention import make_flex_block_causal_mask
except ImportError:
    make_flex_block_causal_mask = None

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.modeling_gemma3 import (
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
    token_type_ids_mask_function,
)

from opendm.constants.robot import HISTORY_POOL_SIZE
from opendm.model.dm05.dm05_arch import (
    DM05ActionExpert,
    DM05ForConditionalGeneration,
)
from opendm.model.dm05.dm05_utils import (
    HISTORY_PAD_TOKEN_ID,
    fused_linear_euler_update,
    make_suffix_attn_mask,
    mask_history_pad_tokens_in_attention,
    posemb_sincos,
)

try:
    import opendm.infer.dm05_bigkernel as bigkernel
except ImportError as exc:
    bigkernel = None
    _BIGKERNEL_IMPORT_ERROR = exc
else:
    _BIGKERNEL_IMPORT_ERROR = None


_MASK_INPUT_EMBEDS_KWARG = (
    "input_embeds"
    if "input_embeds" in inspect.signature(create_causal_mask).parameters
    else "inputs_embeds"
)
_TOKEN_TYPE_MASK_NEEDS_TOKENS_PER_IMAGE = (
    "tokens_per_image" in inspect.signature(token_type_ids_mask_function).parameters
)
_FAST_PREFIX_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 16,
    "BLOCK_N": 32,
    "num_stages": 1,
    "num_warps": 4,
    # Bucketed profiles contain padded query rows with no visible keys.
    "ROWS_GUARANTEED_SAFE": False,
}


def _convert_sdpa_mask_to_eager_bias(
    attention_mask: torch.Tensor | None,
    *,
    dtype: torch.dtype,
):
    if attention_mask is None:
        return None
    if attention_mask.dtype != torch.bool:
        return attention_mask
    # transformers==5.3 eager mask materialization creates a CPU scalar tensor,
    # which breaks CUDA graph capture. Build the additive bias directly on device.
    attention_bias = attention_mask.logical_not().to(dtype=dtype)
    attention_bias.mul_(torch.finfo(dtype).min)
    return attention_bias


def _make_prefix_causal_mask_mapping(
    *,
    vlm_model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    position_ids: torch.LongTensor,
    token_type_ids: torch.LongTensor,
    mask_attn_implementation: str | None = None,
):
    text_config = vlm_model.config.get_text_config()

    if mask_attn_implementation is None:
        mask_attn_implementation = getattr(text_config, "_attn_implementation", None)

    effective_mask_backend = (
        "sdpa" if mask_attn_implementation == "eager" else mask_attn_implementation
    )
    if effective_mask_backend is not None:
        text_config = copy.copy(text_config)
        text_config._attn_implementation = effective_mask_backend

    mask_kwargs = {
        "config": text_config,
        _MASK_INPUT_EMBEDS_KWARG: inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    is_image = (token_type_ids == 1).to(cache_position.device)
    prev_is_image = F.pad(is_image, (1, 0), value=0)[:, :-1]
    new_image_start = is_image & ~prev_is_image
    image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
    image_group_ids_local = torch.where(
        is_image,
        image_group_ids,
        torch.full_like(token_type_ids, -1),
    )
    token_type_args = (
        token_type_ids.to(cache_position.device),
        image_group_ids_local.to(cache_position.device),
    )
    if _TOKEN_TYPE_MASK_NEEDS_TOKENS_PER_IMAGE:
        token_type_args = token_type_args + (int(vlm_model.config.mm_tokens_per_image),)
    mask_kwargs["or_mask_function"] = token_type_ids_mask_function(*token_type_args)
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    if mask_attn_implementation == "eager":
        causal_mask_mapping = {
            key: _convert_sdpa_mask_to_eager_bias(
                value,
                dtype=inputs_embeds.dtype,
            )
            for key, value in causal_mask_mapping.items()
        }
    return causal_mask_mapping


def _is_block_mask(attention_mask: Any) -> bool:
    return attention_mask.__class__.__name__ == "BlockMask"


def _prepare_prefix_flex_mask(attention_mask: Any) -> Any:
    if _is_block_mask(attention_mask):
        return attention_mask
    if getattr(attention_mask, "ndim", None) != 2:
        return attention_mask
    if make_flex_block_causal_mask is None:
        raise RuntimeError(
            "Static prefix inference requires a Transformers build that exposes "
            "make_flex_block_causal_mask."
        )
    return make_flex_block_causal_mask(attention_mask)


def _mark_static_address(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch._dynamo.mark_static_address(tensor)


def _pack_qkv_weight(layer) -> torch.Tensor:
    """Pack Gemma3 q/k/v projection weights into one [hidden, qkv] matrix."""
    attn = layer.self_attn
    return torch.cat(
        [
            attn.q_proj.weight.detach().transpose(0, 1),
            attn.k_proj.weight.detach().transpose(0, 1),
            attn.v_proj.weight.detach().transpose(0, 1),
        ],
        dim=1,
    ).contiguous()


def _prefix_decoder_shape(layer) -> "bigkernel.DM05TransformerShape":
    attn = layer.self_attn
    head_dim = int(attn.head_dim)
    return bigkernel.DM05TransformerShape(
        hidden_size=int(attn.q_proj.in_features),
        intermediate_size=int(layer.mlp.gate_proj.out_features),
        head_dim=head_dim,
        q_heads=int(attn.q_proj.out_features // head_dim),
        kv_heads=int(attn.k_proj.out_features // head_dim),
    )


def _supported_prefix_decoder_shape() -> "bigkernel.DM05TransformerShape":
    return bigkernel.DM05TransformerShape(
        hidden_size=bigkernel.SUPPORTED_PREFIX_HIDDEN_SIZE,
        intermediate_size=bigkernel.SUPPORTED_PREFIX_INTERMEDIATE_SIZE,
        head_dim=bigkernel.SUPPORTED_PREFIX_HEAD_DIM,
        q_heads=bigkernel.SUPPORTED_PREFIX_Q_HEADS,
        kv_heads=bigkernel.SUPPORTED_PREFIX_KV_HEADS,
    )


def _pack_suffix_mlp_weights(layer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mlp = layer.mlp
    return (
        mlp.gate_proj.weight.detach().transpose(0, 1).contiguous(),
        mlp.up_proj.weight.detach().transpose(0, 1).contiguous(),
        mlp.down_proj.weight.detach().transpose(0, 1).contiguous(),
    )


def update_fast_rope_cache(
    rope_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    computed_rope_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Update static-address RoPE tensors used by CUDA graph replay."""
    if rope_cache is None:
        rope_cache = {}

    for layer_type, (cos, sin) in computed_rope_embeddings.items():
        cached = rope_cache.get(layer_type)
        needs_realloc = (
            cached is None
            or cached[0].shape != cos.shape
            or cached[0].dtype != cos.dtype
            or cached[0].device != cos.device
            or cached[1].shape != sin.shape
            or cached[1].dtype != sin.dtype
            or cached[1].device != sin.device
        )
        if needs_realloc:
            cached_cos = cos.contiguous()
            cached_sin = sin.contiguous()
            _mark_static_address(cached_cos)
            _mark_static_address(cached_sin)
            rope_cache[layer_type] = (cached_cos, cached_sin)
        else:
            cached_cos, cached_sin = cached
            cached_cos.copy_(cos)
            cached_sin.copy_(sin)
    return rope_cache


class DM05FastActionExpert(DM05ActionExpert):
    """Inference-only action expert extension with Triton suffix big-kernel."""

    @classmethod
    def from_action_expert(
        cls,
        action_expert: DM05ActionExpert,
    ) -> "DM05FastActionExpert":
        if not isinstance(action_expert, cls):
            action_expert.__class__ = cls
        action_expert._initialize_fast_state()
        return action_expert

    def _initialize_fast_state(self) -> None:
        self.set_action_attention_backend("sdpa")
        self._fast_qkv_weights = None
        self._fast_mlp_weights = None
        self._fast_buffers = None
        self._fast_prefix_qkv_weights = None
        self._fast_prefix_shape = None

    def setup_fast_bigkernel_suffix(self) -> None:
        if bigkernel is None:
            raise ImportError(
                "Triton suffix big-kernel backend requires triton to be installed."
            ) from _BIGKERNEL_IMPORT_ERROR

        qkv_weights = []
        mlp_weights = []
        for layer in self.layers:
            qkv_weights.append(_pack_qkv_weight(layer))
            mlp_weights.append(_pack_suffix_mlp_weights(layer))

        self._fast_qkv_weights = tuple(qkv_weights)
        self._fast_mlp_weights = tuple(mlp_weights)
        self._fast_buffers = None

    def setup_fast_bigkernel_prefix(
        self,
        prefix_decoder_layers,
    ) -> None:
        if bigkernel is None:
            raise ImportError(
                "Triton prefix decoder fastpath requires triton to be installed."
            ) from _BIGKERNEL_IMPORT_ERROR

        prefix_decoder_layers = tuple(prefix_decoder_layers)
        if not prefix_decoder_layers:
            raise ValueError("Prefix decoder layer list must not be empty.")
        expected_shape = _prefix_decoder_shape(prefix_decoder_layers[0])
        supported_shape = _supported_prefix_decoder_shape()
        if expected_shape != supported_shape:
            raise ValueError(
                "Prefix decoder fastpath does not support this Gemma3 shape: "
                f"expected {supported_shape}, got {expected_shape}."
            )

        qkv_weights = []
        for layer_idx, layer in enumerate(prefix_decoder_layers):
            layer_shape = _prefix_decoder_shape(layer)
            if layer_shape != expected_shape:
                raise ValueError(
                    "Prefix decoder layers must use one transformer shape; "
                    f"layer 0 uses {expected_shape}, layer {layer_idx} uses "
                    f"{layer_shape}."
                )
            attention_backend = getattr(
                layer.self_attn.config,
                "_attn_implementation",
                getattr(layer.self_attn.config, "attn_implementation", "eager"),
            )
            if attention_backend != "flex_attention":
                raise ValueError(
                    "Static prefix inference requires flex_attention for every "
                    f"decoder layer, but layer {layer_idx} uses {attention_backend!r}."
                )
            qkv_weights.append(_pack_qkv_weight(layer))

        self._fast_prefix_qkv_weights = tuple(qkv_weights)
        self._fast_prefix_shape = expected_shape

    def make_fast_prefix_buffers(self, prefix_seq_len: int):
        prefix_seq_len = int(prefix_seq_len)
        if prefix_seq_len <= 0:
            raise ValueError(f"prefix_seq_len must be positive, got {prefix_seq_len}.")
        if self._fast_prefix_qkv_weights is None or self._fast_prefix_shape is None:
            raise RuntimeError(
                "Prefix decoder weights must be initialized before allocating buffers."
            )
        reference_weight = self._fast_prefix_qkv_weights[0]
        return bigkernel.make_prefix_decoder_buffers(
            batch_size=1,
            seq_len=prefix_seq_len,
            dtype=reference_weight.dtype,
            device=reference_weight.device,
            shape=self._fast_prefix_shape,
        )

    @staticmethod
    def make_fast_suffix_buffers(
        *,
        batch_size: int,
        seq_len: int,
        max_kv_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        return bigkernel.make_suffix_buffers(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            max_kv_len=int(max_kv_len),
            dtype=dtype,
            device=device,
        )

    def initialize_fast_suffix_buffers(
        self,
        *,
        batch_size: int,
        seq_len: int,
        max_kv_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        buffers = self._fast_buffers
        needs_realloc = (
            buffers is None
            or int(buffers.batch_size) != int(batch_size)
            or int(buffers.seq_len) != int(seq_len)
            or int(buffers.max_kv_len) != int(max_kv_len)
            or buffers.dtype != dtype
            or buffers.device != device
        )
        if needs_realloc:
            self._fast_buffers = self.make_fast_suffix_buffers(
                batch_size=int(batch_size),
                seq_len=int(seq_len),
                max_kv_len=int(max_kv_len),
                dtype=dtype,
                device=device,
            )

    def _suffix_forward_layers_fast_bigkernel(
        self,
        suffix_embeds: torch.Tensor,
        prefix_visible_mask: torch.Tensor,
        prefix_cache_keys: tuple[torch.Tensor, ...],
        prefix_cache_values: tuple[torch.Tensor, ...],
        rope_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        input_modulations: tuple[torch.Tensor, ...],
        mlp_modulations: tuple[torch.Tensor, ...],
        final_modulation: torch.Tensor,
        buffers=None,
    ) -> torch.Tensor:
        if buffers is None:
            buffers = self._fast_buffers
        if buffers is None:
            raise RuntimeError(
                "Low-latency suffix buffers must be initialized before suffix forward."
            )
        hidden_states = suffix_embeds
        for layer_idx, (layer_cache_keys, layer_cache_values) in enumerate(
            zip(prefix_cache_keys, prefix_cache_values, strict=True)
        ):
            hidden_states = self._compute_suffix_layer_fast_bigkernel(
                layer_idx,
                hidden_states,
                prefix_visible_mask,
                layer_cache_keys,
                layer_cache_values,
                rope_embeddings=rope_embeddings,
                input_modulation=input_modulations[layer_idx],
                mlp_modulation=mlp_modulations[layer_idx],
                buffers=buffers,
            )

        dtype = hidden_states.dtype
        var = torch.mean(torch.square(hidden_states.float()), dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(var + self.norm.eps)
        scale, shift, _ = torch.chunk(final_modulation, 3, dim=-1)
        hidden_states = hidden_states * (1.0 + scale.float()) + shift.float()
        return hidden_states.to(dtype)

    def _compute_suffix_layer_fast_bigkernel(
        self,
        layer_idx: int,
        suffix_embeds: torch.Tensor,
        prefix_visible_mask: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        *,
        rope_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        input_modulation: torch.Tensor,
        mlp_modulation: torch.Tensor,
        buffers,
    ) -> torch.Tensor:
        ae_layer = self.layers[layer_idx]
        layer_type = ae_layer.attention_type
        batch_size = buffers.batch_size
        seq_len = buffers.seq_len

        cos, sin = rope_embeddings[layer_type]
        qkv_weights = self._fast_qkv_weights
        bigkernel.launch_input_adarmsnorm(
            suffix_embeds,
            input_modulation,
            buffers,
            eps=float(ae_layer.input_layernorm.eps),
        )
        bigkernel.project_qkv_and_launch_qknorm_rope_split(
            buffers.mlp_input,
            qkv_weights[layer_idx],
            buffers,
            ae_layer.self_attn.q_norm.weight,
            ae_layer.self_attn.k_norm.weight,
            cos,
            sin,
            eps=float(ae_layer.input_layernorm.eps),
        )
        bigkernel.launch_suffix_attention(
            buffers.query,
            buffers.key,
            buffers.value,
            cache_keys,
            cache_values,
            prefix_visible_mask,
            buffers,
            scale=float(ae_layer.self_attn.scaling),
        )
        attn_output = buffers.attn_out.view(batch_size, seq_len, -1)
        attn_embeds = torch.mm(
            attn_output.view(buffers.rows, -1),
            ae_layer.self_attn.o_proj.weight.transpose(0, 1),
            out=buffers.residual.view(buffers.rows, -1),
        )
        hidden = buffers.shape.hidden_size
        attn_gate = input_modulation[..., 2 * hidden :]
        bigkernel.launch_suffix_attn_postnorm_residual_prenorm(
            attn_embeds,
            suffix_embeds,
            attn_gate,
            ae_layer.post_attention_layernorm.weight,
            mlp_modulation,
            buffers,
            post_eps=float(ae_layer.post_attention_layernorm.eps),
            pre_eps=float(ae_layer.pre_feedforward_layernorm.eps),
        )
        gate_weight, up_weight, down_weight = self._fast_mlp_weights[layer_idx]
        bigkernel.launch_geglu_gate_up(
            buffers.mlp_input,
            gate_weight,
            up_weight,
            buffers.mlp_hidden,
            buffers,
        )
        bigkernel.launch_mlp_down(
            buffers.mlp_hidden,
            down_weight,
            buffers.mlp_out,
            buffers,
        )
        mlp_gate = mlp_modulation[..., 2 * hidden :]
        return bigkernel.launch_suffix_post_ffn_norm_residual(
            mlp_gate,
            ae_layer.post_feedforward_layernorm.weight,
            buffers,
            eps=float(ae_layer.post_feedforward_layernorm.eps),
        )

    def precompute_fast_rope(
        self,
        *,
        position_ids: torch.LongTensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        suffix_len = int(position_ids.shape[-1])
        batch_size = int(position_ids.shape[0])
        reference = torch.empty(
            batch_size,
            suffix_len,
            int(self.config.hidden_size),
            device=device,
            dtype=dtype,
        )
        layer_types = tuple(
            dict.fromkeys(layer.attention_type for layer in self.layers)
        )
        return {
            layer_type: self.rotary_emb(
                reference,
                position_ids,
                layer_type=layer_type,
            )
            for layer_type in layer_types
        }

    def precompute_fast_time_modulations(
        self,
        adarms_cond_steps: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        def _stack_modulator_outputs(modulator: nn.Linear) -> torch.Tensor:
            outputs = []
            for adarms_cond in adarms_cond_steps:
                modulation = modulator(adarms_cond.to(dtype=modulator.weight.dtype))
                modulation = modulation[:, None, :]
                outputs.append(modulation)
            return torch.stack(outputs, dim=0).contiguous()

        return (
            tuple(
                _stack_modulator_outputs(modulator)
                for modulator in self.input_time_modulators
            ),
            tuple(
                _stack_modulator_outputs(modulator)
                for modulator in self.mlp_time_modulators
            ),
            _stack_modulator_outputs(self.final_time_modulator),
        )


class DM05FastForCausalLM:
    """Model-side prefill/decode schedule used by the ablation fast backend."""

    def __init__(
        self,
        model: DM05ForConditionalGeneration,
    ) -> None:
        self.model = model
        self.dm05 = model.model
        self.action_expert = DM05FastActionExpert.from_action_expert(
            self.dm05.action_expert
        )
        self.padding_idx = int(self.dm05.vlm.model.language_model.padding_idx)
        # Maximum processed multimodal prefix length supported by FastInfer.
        self.prefix_len = 1024
        self.suffix_len = int(self.dm05.config.chunk_size)
        self.action_dim = int(self.dm05.config.action_dim)
        self.noise_dtype = self.dm05.action_in_proj.weight.dtype
        self.prefix_decoder_initialized = False
        self.suffix_rope_embeddings = None
        self.suffix_time_modulations = None
        self.suffix_time_modulation_signature = None
        self.action_expert.setup_fast_bigkernel_suffix()

    def setup_fast_prefix_decoder(self) -> None:
        language_model = self.dm05.vlm.model.language_model
        layer_count = int(language_model.config.num_hidden_layers)
        layers = tuple(language_model.layers[:layer_count])
        self.action_expert.setup_fast_bigkernel_prefix(
            layers,
        )
        self.prefix_decoder_initialized = True

    def initialize_fast_suffix_buffers(
        self,
        *,
        batch_size: int,
        seq_len: int,
        max_kv_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.action_expert.initialize_fast_suffix_buffers(
            batch_size=batch_size,
            seq_len=seq_len,
            max_kv_len=max_kv_len,
            dtype=dtype,
            device=device,
        )

    def _validate_prefix_inputs(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        expected_shape = tuple(input_ids.shape)
        if input_ids.ndim != 2:
            raise ValueError(
                "Prefix input_ids must be a 2D [batch, sequence] tensor, "
                f"got shape {expected_shape}."
            )
        mismatched_shapes = {
            name: tuple(tensor.shape)
            for name, tensor in (
                ("attention_mask", attention_mask),
                ("token_type_ids", token_type_ids),
            )
            if tensor.ndim != 2 or tuple(tensor.shape) != expected_shape
        }
        if mismatched_shapes:
            raise ValueError(
                "Prefix token tensors must all have the same [batch, sequence] "
                f"shape as input_ids={expected_shape}, got {mismatched_shapes}."
            )
        input_len = int(input_ids.shape[1])
        if input_len > self.prefix_len:
            raise ValueError(
                f"Prefix length {input_len} exceeds the fast "
                f"backend limit of {self.prefix_len} tokens. Shorten the "
                f"prompt or reduce model_max_length."
            )
        return input_ids, attention_mask, token_type_ids

    def _merge_image_features_into_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        image_token_id = int(self.dm05.vlm.model.config.image_token_id)
        special_image_mask = input_ids.eq(image_token_id)
        flat_image_features = image_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).reshape(-1, int(inputs_embeds.shape[-1]))
        if int(flat_image_features.shape[0]) == 0:
            return inputs_embeds

        flat_image_token_indices = (
            special_image_mask.reshape(-1).to(torch.long).cumsum(dim=0) - 1
        ).clamp_min_(0)
        gathered_image_features = flat_image_features.index_select(
            0,
            flat_image_token_indices,
        ).view(*input_ids.shape, int(inputs_embeds.shape[-1]))
        return torch.where(
            special_image_mask.unsqueeze(-1),
            gathered_image_features,
            inputs_embeds,
        )

    def _zero_history_pad_embeds(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Zero ``<unused1>`` positions (dexbotic 0525 invalid history slots).

        Avoid host sync (``.item()`` / boolean indexing) so this stays valid
        inside CUDA graph capture used by the fast prefix path.
        """
        mask = (input_ids == HISTORY_PAD_TOKEN_ID).unsqueeze(-1)
        return inputs_embeds.masked_fill(mask, 0)

    def _inject_history_into_embeds(
        self,
        *,
        inputs_embeds: torch.Tensor,
        history_mask: torch.BoolTensor | None,
        history_features: torch.Tensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_mask is None or not bool(history_mask.any().item()):
            return inputs_embeds
        if history_features is None:
            if history_pixel_values is None or int(history_pixel_values.shape[0]) == 0:
                return inputs_embeds
            vlm_model = self.dm05.vlm.model
            pixels = history_pixel_values.to(
                device=inputs_embeds.device,
                dtype=next(vlm_model.vision_tower.parameters()).dtype,
            )
            image_features = vlm_model.get_image_features(
                pixels, return_dict=True
            ).pooler_output
            spatial = int(image_features.shape[1] ** 0.5)
            hidden = image_features.shape[-1]
            grid = image_features.view(-1, spatial, spatial, hidden).permute(0, 3, 1, 2)
            grid = torch.nn.functional.adaptive_avg_pool2d(
                grid, output_size=(HISTORY_POOL_SIZE, HISTORY_POOL_SIZE)
            )
            history_features = grid.permute(0, 2, 3, 1).reshape(
                -1, HISTORY_POOL_SIZE * HISTORY_POOL_SIZE, hidden
            )
        history_features = history_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        history_mask_expanded = history_mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(history_mask_expanded, history_features)

    def _prefix_fastpath_compute_cache_tensors_from_image_features(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        image_features: torch.Tensor,
        token_type_ids: torch.LongTensor,
        kv_cache: Cache,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        history_features: torch.Tensor | None = None,
    ) -> None:
        vlm_model = self.dm05.vlm.model
        image_token_id = int(vlm_model.config.image_token_id)
        if image_token_id >= vlm_model.vocab_size:
            llm_input_ids = torch.where(
                input_ids == image_token_id,
                torch.zeros_like(input_ids),
                input_ids,
            )
        else:
            llm_input_ids = input_ids

        inputs_embeds = vlm_model.get_input_embeddings()(llm_input_ids)
        inputs_embeds = self._zero_history_pad_embeds(input_ids, inputs_embeds)
        inputs_embeds = self._inject_history_into_embeds(
            inputs_embeds=inputs_embeds,
            history_mask=history_mask,
            history_features=history_features,
            history_pixel_values=history_pixel_values,
        )
        inputs_embeds = self._merge_image_features_into_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        cache_position = torch.arange(
            0,
            int(inputs_embeds.shape[1]),
            device=inputs_embeds.device,
        )
        causal_mask_mapping = _make_prefix_causal_mask_mapping(
            vlm_model=vlm_model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=kv_cache,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        vlm_model.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=kv_cache,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )

    def prefill_from_image_features(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_features: torch.Tensor,
        token_type_ids: torch.LongTensor,
        prefix_cache: Cache,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        history_features: torch.Tensor | None = None,
    ) -> tuple[Cache, int]:
        for layer in prefix_cache.layers:
            reset_for_prefill = getattr(layer, "reset_for_prefill", None)
            if reset_for_prefill is not None:
                reset_for_prefill()
        attention_mask = mask_history_pad_tokens_in_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        prefix_position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min_(0)
        self._prefix_fastpath_compute_cache_tensors_from_image_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            image_features=image_features,
            token_type_ids=token_type_ids,
            kv_cache=prefix_cache,
            history_pixel_values=history_pixel_values,
            history_mask=history_mask,
            history_features=history_features,
        )
        return prefix_cache, prefix_cache.get_seq_length()

    def prefill_fast_from_image_features(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        image_features: torch.Tensor,
        token_type_ids: torch.LongTensor,
        prefix_cache: Cache,
        prefix_buffers,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        history_features: torch.Tensor | None = None,
    ) -> tuple[Cache, int]:
        """Fill the prefix KV cache with the static decoder fastpath."""
        if not self.prefix_decoder_initialized:
            raise RuntimeError(
                "Prefix decoder fastpath is not initialized. Call "
                "setup_fast_prefix_decoder() before fast prefix prefill."
            )
        if prefix_cache is None:
            raise ValueError("prefix_cache is required for prefix decoder fastpath.")

        vlm_model = self.dm05.vlm.model
        language_model = vlm_model.language_model
        layer_count = int(language_model.config.num_hidden_layers)
        if len(prefix_cache.layers) < layer_count:
            raise RuntimeError(
                "Prefix cache has fewer layers than the Gemma3 decoder: "
                f"cache={len(prefix_cache.layers)}, decoder={layer_count}."
            )
        for cache_layer in prefix_cache.layers[:layer_count]:
            cache_layer.reset_for_prefill()

        image_token_id = int(vlm_model.config.image_token_id)
        if image_token_id >= vlm_model.vocab_size:
            special_image_mask = input_ids == image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        inputs_embeds = vlm_model.get_input_embeddings()(llm_input_ids).contiguous()
        inputs_embeds = self._zero_history_pad_embeds(input_ids, inputs_embeds)
        inputs_embeds = self._inject_history_into_embeds(
            inputs_embeds=inputs_embeds,
            history_mask=history_mask,
            history_features=history_features,
            history_pixel_values=history_pixel_values,
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        bigkernel.launch_prefix_image_merge(
            input_ids,
            image_features,
            inputs_embeds,
            image_token_id=image_token_id,
        )
        cache_position = torch.arange(
            int(prefix_buffers.seq_len),
            device=inputs_embeds.device,
        )
        causal_mask_mapping = _make_prefix_causal_mask_mapping(
            vlm_model=vlm_model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=prefix_cache,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        prefix_qkv_weights = self.action_expert._fast_prefix_qkv_weights
        if prefix_qkv_weights is None:
            raise RuntimeError("Prefix decoder fastpath weights are not initialized.")

        hidden_states = inputs_embeds
        layer_types = tuple(
            dict.fromkeys(
                layer.attention_type for layer in language_model.layers[:layer_count]
            )
        )
        position_embeddings = {
            layer_type: language_model.rotary_emb(
                hidden_states,
                position_ids,
                layer_type=layer_type,
            )
            for layer_type in layer_types
        }
        for layer_idx, decoder_layer in enumerate(language_model.layers[:layer_count]):
            attention = decoder_layer.self_attn
            layer_attention_mask = _prepare_prefix_flex_mask(
                causal_mask_mapping[decoder_layer.attention_type]
            )
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]
            cos, sin = position_embeddings[decoder_layer.attention_type]
            bigkernel.project_qkv_and_launch_qknorm_rope_split(
                hidden_states,
                prefix_qkv_weights[layer_idx],
                prefix_buffers,
                attention.q_norm.weight,
                attention.k_norm.weight,
                cos,
                sin,
                eps=float(decoder_layer.input_layernorm.eps),
            )
            query_states = prefix_buffers.query
            key_states = prefix_buffers.key
            value_states = prefix_buffers.value
            prefix_cache.layers[layer_idx].update(
                key_states,
                value_states,
                {"cache_position": cache_position},
            )

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                "flex_attention",
                eager_attention_forward,
            )
            attn_output, _ = attention_interface(
                attention,
                query_states,
                key_states,
                value_states,
                layer_attention_mask,
                dropout=attention.attention_dropout if attention.training else 0.0,
                scaling=attention.scaling,
                sliding_window=attention.sliding_window,
                position_ids=position_ids,
                kernel_options=_FAST_PREFIX_FLEX_KERNEL_OPTIONS,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            hidden_states = attention.o_proj(attn_output)

            bigkernel.launch_prefix_attn_postnorm_residual_prenorm(
                hidden_states,
                residual,
                prefix_buffers,
                decoder_layer.post_attention_layernorm.weight,
                decoder_layer.pre_feedforward_layernorm.weight,
                post_eps=float(decoder_layer.post_attention_layernorm.eps),
                pre_eps=float(decoder_layer.pre_feedforward_layernorm.eps),
            )
            mlp_input = prefix_buffers.mlp_input
            prefix_buffers.mlp_out.copy_(decoder_layer.mlp(mlp_input))
            hidden_states = bigkernel.launch_prefix_post_ffn_norm_residual(
                prefix_buffers,
                decoder_layer.post_feedforward_layernorm.weight,
                eps=float(decoder_layer.post_feedforward_layernorm.eps),
            )
        return prefix_cache, int(prefix_buffers.seq_len)

    def decode(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        prefix_cache: Cache,
        prefix_len: int,
        diffusion_input_noise: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.device.type == "cuda":
            prepared_noise = diffusion_input_noise.to(
                device=input_ids.device,
                dtype=self.noise_dtype,
            )
            prefix_visible_mask = self.build_prefix_visible_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prefix_len=prefix_len,
            )
            suffix_len = int(prepared_noise.shape[1])
            suffix_position_ids = self._build_suffix_position_ids(
                prefix_len,
                suffix_len,
                input_ids.device,
                input_ids=input_ids,
                pad_token_id=self.padding_idx,
                invisible_prefix_token_ids=(HISTORY_PAD_TOKEN_ID,),
            )
            context = self.prepare_fast_suffix_context(
                prefix_cache=prefix_cache,
                prefix_visible_mask=prefix_visible_mask,
                suffix_position_ids=suffix_position_ids,
                initial_noise=prepared_noise,
                diffusion_steps=diffusion_steps,
            )
            return self.decode_fast_prepared(
                initial_noise=prepared_noise,
                diffusion_steps=diffusion_steps,
                action_mask=action_mask,
                **context,
            )

        return self._decode_default(
            input_ids=input_ids,
            prefix_cache=prefix_cache,
            prefix_len=prefix_len,
            diffusion_input_noise=diffusion_input_noise,
            diffusion_steps=diffusion_steps,
            action_mask=action_mask,
        )

    def _decode_default(
        self,
        *,
        input_ids: torch.LongTensor,
        prefix_cache: Cache,
        prefix_len: int,
        diffusion_input_noise: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if diffusion_input_noise is None:
            raise ValueError("diffusion_input_noise is required for fast decode.")
        x_t = diffusion_input_noise.to(
            device=input_ids.device,
            dtype=self.noise_dtype,
        )
        x_t = x_t * action_mask
        batch_size = int(input_ids.shape[0])
        device = input_ids.device
        dt = -1.0 / diffusion_steps
        time_tensor = torch.full(
            (batch_size,),
            1.0,
            device=device,
            dtype=self.noise_dtype,
        )

        for _ in range(diffusion_steps):
            suffix_embeds = self.dm05.action_in_proj(x_t)
            adarms_cond = self._build_adarms_cond(time_tensor, suffix_embeds.dtype)
            suffix_len = int(suffix_embeds.shape[1])
            suffix_attn_mask = make_suffix_attn_mask(
                input_ids=input_ids,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                batch_size=batch_size,
                device=suffix_embeds.device,
                dtype=suffix_embeds.dtype,
                pad_token_id=self.padding_idx,
                invisible_prefix_token_ids=(HISTORY_PAD_TOKEN_ID,),
            )
            suffix_position_ids = self._build_suffix_position_ids(
                prefix_len,
                suffix_len,
                device,
                input_ids=input_ids,
                pad_token_id=self.padding_idx,
                invisible_prefix_token_ids=(HISTORY_PAD_TOKEN_ID,),
            )

            suffix_out = self._suffix_forward(
                suffix_embeds=suffix_embeds,
                attention_mask=suffix_attn_mask,
                position_ids=suffix_position_ids,
                prefix_cache=prefix_cache,
                adarms_cond=adarms_cond,
            )

            v_t = self.dm05.action_out_proj(suffix_out)
            x_t = x_t + v_t * dt
            x_t = x_t * action_mask
            time_tensor = time_tensor + dt

        return x_t

    def _extract_prefix_cache_tensors(
        self,
        prefix_cache: Cache,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        keys = []
        values = []
        for layer_idx in range(len(self.action_expert.layers)):
            layer = prefix_cache.layers[layer_idx]
            key_states = layer.keys
            value_states = layer.values
            # Prefix cache tensors can be non-contiguous after cache reuse.
            if key_states is not None and not key_states.is_contiguous():
                key_states = key_states.contiguous()
            if value_states is not None and not value_states.is_contiguous():
                value_states = value_states.contiguous()
            keys.append(key_states)
            values.append(value_states)
        return tuple(keys), tuple(values)

    def build_prefix_visible_mask(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        prefix_ids = input_ids[:, :prefix_len]
        valid_prefix = attention_mask[:, :prefix_len].to(torch.bool)
        valid_prefix = valid_prefix & (prefix_ids != self.padding_idx)
        valid_prefix = valid_prefix & (prefix_ids != HISTORY_PAD_TOKEN_ID)
        return valid_prefix.to(torch.int8)

    def prepare_fast_suffix_context(
        self,
        *,
        prefix_cache: Cache | None,
        prefix_visible_mask: torch.Tensor,
        prefix_cache_keys: tuple[torch.Tensor, ...] | None = None,
        prefix_cache_values: tuple[torch.Tensor, ...] | None = None,
        suffix_position_ids: torch.LongTensor,
        initial_noise: torch.Tensor,
        diffusion_steps: int,
        suffix_buffers=None,
    ) -> dict[str, object]:
        if initial_noise is None:
            raise ValueError("diffusion_input_noise is required for fast decode.")
        if initial_noise.device.type != "cuda":
            raise RuntimeError("Suffix big-kernel decode requires CUDA tensors.")

        if prefix_cache_keys is None or prefix_cache_values is None:
            if prefix_cache is None:
                raise RuntimeError(
                    "Either prefix_cache or explicit prefix_cache_keys/values is required."
                )
            prefix_cache_keys, prefix_cache_values = self._extract_prefix_cache_tensors(
                prefix_cache,
            )
        if any(tensor is None for tensor in prefix_cache_keys + prefix_cache_values):
            raise RuntimeError("Prefix cache must be populated before suffix decode.")

        initial_noise = initial_noise.to(dtype=self.noise_dtype)
        batch_size = int(initial_noise.shape[0])
        suffix_len = int(initial_noise.shape[1])
        if suffix_buffers is None:
            max_kv_len = self.prefix_len + suffix_len
            self.initialize_fast_suffix_buffers(
                batch_size=batch_size,
                seq_len=suffix_len,
                max_kv_len=max_kv_len,
                dtype=initial_noise.dtype,
                device=initial_noise.device,
            )
            suffix_buffers = self.action_expert._fast_buffers
        elif (
            int(suffix_buffers.batch_size) != batch_size
            or int(suffix_buffers.seq_len) != suffix_len
            or int(suffix_buffers.max_kv_len)
            != int(prefix_cache_keys[0].shape[2]) + suffix_len
            or suffix_buffers.dtype != initial_noise.dtype
            or suffix_buffers.device != initial_noise.device
        ):
            raise ValueError(
                "Suffix profile buffers do not match the decode input: "
                f"buffers={(suffix_buffers.batch_size, suffix_buffers.seq_len, suffix_buffers.dtype, suffix_buffers.device)}, "
                f"input={(batch_size, suffix_len, initial_noise.dtype, initial_noise.device)}."
            )

        rope_embeddings = self._get_suffix_rope_embeddings(
            position_ids=suffix_position_ids,
            dtype=initial_noise.dtype,
            device=initial_noise.device,
        )
        time_modulations = self._get_suffix_time_modulations(
            batch_size=batch_size,
            diffusion_steps=diffusion_steps,
            dtype=initial_noise.dtype,
            device=initial_noise.device,
        )
        return {
            "prefix_visible_mask": prefix_visible_mask,
            "prefix_cache_keys": prefix_cache_keys,
            "prefix_cache_values": prefix_cache_values,
            "rope_embeddings": rope_embeddings,
            "time_modulations": time_modulations,
            "suffix_buffers": suffix_buffers,
        }

    def _build_suffix_time_modulations(
        self,
        *,
        batch_size: int,
        diffusion_steps: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        dt = -1.0 / int(diffusion_steps)
        time_values = torch.tensor(
            [1.0 + step_idx * dt for step_idx in range(int(diffusion_steps))],
            device=device,
            dtype=dtype,
        )
        adarms_cond_steps = tuple(
            self._build_adarms_cond(
                time_value.expand(int(batch_size)),
                dtype,
            )
            for time_value in time_values
        )
        return self.action_expert.precompute_fast_time_modulations(
            adarms_cond_steps,
        )

    @staticmethod
    def _iter_suffix_time_modulation_tensors(time_modulations: tuple):
        input_modulations, mlp_modulations, final_modulation = time_modulations
        yield from input_modulations
        yield from mlp_modulations
        yield final_modulation

    @staticmethod
    def _slice_suffix_time_modulations(
        time_modulations: tuple,
        step_idx: int,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        input_modulations, mlp_modulations, final_modulation = time_modulations
        return (
            tuple(tensor[step_idx] for tensor in input_modulations),
            tuple(tensor[step_idx] for tensor in mlp_modulations),
            final_modulation[step_idx],
        )

    def _get_suffix_time_modulations(
        self,
        *,
        batch_size: int,
        diffusion_steps: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple:
        signature = (int(batch_size), int(diffusion_steps), dtype, device)
        time_modulations = self.suffix_time_modulations
        if (
            time_modulations is None
            or self.suffix_time_modulation_signature != signature
        ):
            time_modulations = self._build_suffix_time_modulations(
                batch_size=batch_size,
                diffusion_steps=diffusion_steps,
                dtype=dtype,
                device=device,
            )
            for tensor in self._iter_suffix_time_modulation_tensors(time_modulations):
                _mark_static_address(tensor)
            self.suffix_time_modulations = time_modulations
            self.suffix_time_modulation_signature = signature
        return time_modulations

    def _get_suffix_rope_embeddings(
        self,
        *,
        position_ids: torch.LongTensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        computed_rope_embeddings = self.action_expert.precompute_fast_rope(
            position_ids=position_ids,
            dtype=dtype,
            device=device,
        )
        self.suffix_rope_embeddings = update_fast_rope_cache(
            self.suffix_rope_embeddings,
            computed_rope_embeddings,
        )
        return self.suffix_rope_embeddings

    def decode_fast_prepared(
        self,
        *,
        initial_noise: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor,
        prefix_visible_mask: torch.Tensor,
        prefix_cache_keys: tuple[torch.Tensor, ...],
        prefix_cache_values: tuple[torch.Tensor, ...],
        rope_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        time_modulations: tuple,
        suffix_buffers=None,
    ) -> torch.Tensor:
        x_t = initial_noise.to(dtype=self.noise_dtype)
        x_t = x_t * action_mask
        dt = -1.0 / int(diffusion_steps)
        for step_idx in range(int(diffusion_steps)):
            input_modulations, mlp_modulations, final_modulation = (
                self._slice_suffix_time_modulations(time_modulations, step_idx)
            )
            suffix_embeds = self.dm05.action_in_proj(x_t)
            suffix_out = self.action_expert._suffix_forward_layers_fast_bigkernel(
                suffix_embeds=suffix_embeds,
                prefix_visible_mask=prefix_visible_mask,
                prefix_cache_keys=prefix_cache_keys,
                prefix_cache_values=prefix_cache_values,
                rope_embeddings=rope_embeddings,
                input_modulations=input_modulations,
                mlp_modulations=mlp_modulations,
                final_modulation=final_modulation,
                buffers=suffix_buffers,
            )
            x_t = fused_linear_euler_update(
                hidden_states=suffix_out,
                current=x_t,
                linear=self.dm05.action_out_proj,
                dt=dt,
            )
            x_t = x_t * action_mask
        return x_t

    def _build_adarms_cond(
        self,
        time: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ae_hidden = self.dm05.action_in_proj.out_features
        time_emb = posemb_sincos(time, ae_hidden, max_period=4.0).to(dtype)
        cond = self.dm05.time_mlp_in(time_emb)
        cond = F.silu(cond)
        cond = self.dm05.time_mlp_out(cond)
        return F.silu(cond)

    def _build_suffix_position_ids(
        self,
        prefix_len,
        suffix_len,
        device,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pad_token_id=0,
        invisible_prefix_token_ids: tuple[int, ...] = (),
    ):
        if attention_mask is not None:
            valid_prefix = attention_mask[:, :prefix_len].to(torch.bool)
        else:
            valid_prefix = input_ids[:, :prefix_len] != pad_token_id
        for token_id in invisible_prefix_token_ids:
            valid_prefix = valid_prefix & (input_ids[:, :prefix_len] != token_id)
        effective_prefix_len = valid_prefix.sum(dim=1)
        suffix_offsets = torch.arange(suffix_len, device=device).unsqueeze(0)
        return effective_prefix_len.unsqueeze(1) + suffix_offsets
