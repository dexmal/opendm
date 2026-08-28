"""DM05 model architecture built on a Gemma3 VLM and Action Expert."""

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3Config,
    Gemma3TextConfig,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3CausalLMOutputWithPast,
    Gemma3DecoderLayer,
    Gemma3ForConditionalGeneration,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3TextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from opendm.constants.robot import HISTORY_POOL_SIZE
from opendm.model.base import (
    DMBaseConfig,
    DMPreTrainedModel,
)
from opendm.model.dm05.dm05_utils import (
    HISTORY_PAD_TOKEN_ID,
    VLADynamicCache,
    is_flash_attention_2_available,
    is_flex_attention_available,
    make_suffix_attn_mask,
    mask_history_pad_tokens_in_attention,
    patch_decoder_layers,
    posemb_sincos,
    validate_action_config_compatible,
)

logger = logging.getLogger(__name__)

_SUFFIX_GRAPH_PROFILE_CACHE_SIZE = 8
_SUFFIX_GRAPH_PREFIX_BUCKET_ALIGNMENT = 16

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DM05Config(DMBaseConfig):
    """DM05 model configuration."""

    model_type = "dm05"
    tie_word_embeddings: bool = True

    def __init__(
        self,
        vlm_config=None,
        action_config=None,
        action_dim: int = 32,
        chunk_size: int = 50,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if isinstance(vlm_config, dict):
            vlm_config = Gemma3Config(**vlm_config)
        if isinstance(action_config, dict):
            action_config = Gemma3TextConfig(**action_config)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vlm_config = vlm_config
        self.action_config = action_config
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.tie_word_embeddings = tie_word_embeddings


class DM05ActionExpert(Gemma3TextModel):
    """Gemma3 action expert with DM05 suffix-only forward.

    The module takes ownership of the original text model's layers/norm so
    parameter names stay under ``model.action_expert.layers.*`` after wrapping.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
    ):
        super(Gemma3TextModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        hidden_size = config.hidden_size
        self.input_time_modulators = nn.ModuleList(
            nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers
        )
        self.mlp_time_modulators = nn.ModuleList(
            nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers
        )
        self.final_time_modulator = nn.Linear(hidden_size, 3 * hidden_size)
        self.post_init()
        self._init_time_modulators()
        self.set_action_attention_backend("eager")

    def _init_time_modulators(self) -> None:
        std = float(getattr(self.config, "initializer_range", 0.02))
        modulators = (
            list(self.input_time_modulators)
            + list(self.mlp_time_modulators)
            + [self.final_time_modulator]
        )
        for modulator in modulators:
            nn.init.normal_(modulator.weight, mean=0.0, std=std)
            if modulator.bias is not None:
                nn.init.zeros_(modulator.bias)

    def set_action_attention_backend(self, action_attn_implementation: str) -> None:
        if action_attn_implementation == "eager":
            self._suffix_attn_backend = "eager"
            self._attn_fn = self._eager_attention
        elif action_attn_implementation == "sdpa":
            self._suffix_attn_backend = "sdpa"
            self._attn_fn = self._sdpa_attention
        elif action_attn_implementation == "flex_attention":
            self._suffix_attn_backend = "flex_attention"
            self._attn_fn = self._flex_attention
        else:
            raise ValueError(
                "Unsupported suffix attention implementation: "
                f"{action_attn_implementation!r}"
            )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    @staticmethod
    def _sdpa_attention(query_states, key_states, value_states, attention_mask, layer):
        kv_states_k = repeat_kv(key_states, layer.self_attn.num_key_value_groups)
        kv_states_v = repeat_kv(value_states, layer.self_attn.num_key_value_groups)
        bool_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_output = F.scaled_dot_product_attention(
            query_states,
            kv_states_k,
            kv_states_v,
            attn_mask=bool_mask,
            scale=layer.self_attn.scaling,
        )
        return attn_output.transpose(1, 2).contiguous()

    @staticmethod
    def _eager_attention(query_states, key_states, value_states, attention_mask, layer):
        attn_output, _ = eager_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=layer.self_attn.scaling,
        )
        return attn_output

    @staticmethod
    def _flex_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer,
    ):
        from transformers.integrations.flex_attention import flex_attention_forward

        attn_output, _ = flex_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=layer.self_attn.scaling,
        )
        return attn_output

    def forward(
        self,
        *,
        suffix_embeds: torch.Tensor | None = None,
        attention_mask: Any = None,
        position_ids: torch.LongTensor | None = None,
        prefix_cache_keys: tuple[torch.Tensor, ...] | None = None,
        prefix_cache_values: tuple[torch.Tensor, ...] | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if suffix_embeds is None:
            raise RuntimeError(
                "DM05ActionExpert only supports suffix forward. "
                "Pass suffix_embeds, position_ids, prefix_cache_keys, and "
                "prefix_cache_values."
            )
        if position_ids is None:
            raise ValueError(
                "position_ids is required for DM05ActionExpert suffix forward."
            )
        if prefix_cache_keys is None or prefix_cache_values is None:
            raise ValueError(
                "prefix_cache_keys and prefix_cache_values are required for "
                "DM05ActionExpert suffix forward."
            )
        if adarms_cond is None:
            raise ValueError(
                "adarms_cond is required for DM05ActionExpert suffix forward."
            )
        return self._suffix_forward_layers(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefix_cache_keys=prefix_cache_keys,
            prefix_cache_values=prefix_cache_values,
            adarms_cond=adarms_cond,
        )

    def _suffix_forward_layers(
        self,
        suffix_embeds: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.LongTensor,
        prefix_cache_keys: tuple[torch.Tensor, ...],
        prefix_cache_values: tuple[torch.Tensor, ...],
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = suffix_embeds
        num_layers = len(self.layers)

        for layer_idx in range(num_layers):
            layer_cache_keys = prefix_cache_keys[layer_idx]
            layer_cache_values = prefix_cache_values[layer_idx]
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._compute_suffix_layer,
                    layer_idx,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    adarms_cond,
                    layer_cache_keys,
                    layer_cache_values,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._compute_suffix_layer(
                    layer_idx,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    adarms_cond,
                    layer_cache_keys,
                    layer_cache_values,
                )

        hidden_states, _ = self._adaptive_rmsnorm(
            self.norm,
            hidden_states,
            adarms_cond,
            self.final_time_modulator,
        )
        return hidden_states

    def _adaptive_rmsnorm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
        adarms_cond: torch.Tensor,
        modulator: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        eps = getattr(norm, "eps", None)
        if eps is None:
            eps = norm.variance_epsilon
        normed = x * torch.rsqrt(var + eps)

        modulation = modulator(adarms_cond.to(dtype=modulator.weight.dtype))
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def _compute_suffix_layer(
        self,
        layer_idx: int,
        suffix_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Any,
        adarms_cond: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
    ) -> torch.Tensor:
        ae_layer = self.layers[layer_idx]
        layer_type = ae_layer.attention_type

        prenorm, attn_gate = self._adaptive_rmsnorm(
            ae_layer.input_layernorm,
            suffix_embeds,
            adarms_cond,
            self.input_time_modulators[layer_idx],
        )
        batch_size, seq_len, _ = prenorm.shape
        hidden_shape = (batch_size, seq_len, -1, ae_layer.self_attn.head_dim)

        query_states = (
            ae_layer.self_attn.q_proj(prenorm).view(hidden_shape).transpose(1, 2)
        )
        key_states = (
            ae_layer.self_attn.k_proj(prenorm).view(hidden_shape).transpose(1, 2)
        )
        value_states = (
            ae_layer.self_attn.v_proj(prenorm).view(hidden_shape).transpose(1, 2)
        )

        query_states = ae_layer.self_attn.q_norm(query_states)
        key_states = ae_layer.self_attn.k_norm(key_states)

        cos, sin = self.rotary_emb(query_states, position_ids, layer_type=layer_type)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = torch.cat([cache_keys, key_states], dim=2)
        value_states = torch.cat([cache_values, value_states], dim=2)
        if (
            attention_mask is not None
            and attention_mask.shape[-1] != key_states.shape[-2]
        ):
            attention_mask = attention_mask[..., -key_states.shape[-2] :]

        attn_output = self._attn_fn(
            query_states, key_states, value_states, attention_mask, ae_layer
        )
        attn_output = attn_output.view(batch_size, seq_len, -1)

        attn_embeds = ae_layer.self_attn.o_proj(attn_output)
        attn_embeds = ae_layer.post_attention_layernorm(attn_embeds)
        residual = suffix_embeds + attn_embeds * attn_gate

        mlp_input, mlp_gate = self._adaptive_rmsnorm(
            ae_layer.pre_feedforward_layernorm,
            residual,
            adarms_cond,
            self.mlp_time_modulators[layer_idx],
        )
        mlp_out = ae_layer.mlp(mlp_input)
        mlp_out = ae_layer.post_feedforward_layernorm(mlp_out)
        return residual + mlp_out * mlp_gate


# ---------------------------------------------------------------------------
# Model Outputs
# ---------------------------------------------------------------------------


@dataclass
class DM05OutputWithPast(Gemma3CausalLMOutputWithPast):
    fm_loss: torch.FloatTensor | None = None


@dataclass
class DM05SuffixGraphProfile:
    """Static buffers for replaying one action-expert diffusion step."""

    prefix_len: int
    diffusion_steps: int
    prefix_cache_keys: tuple[torch.Tensor, ...]
    prefix_cache_values: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    state: torch.Tensor
    time: torch.Tensor
    action_mask: torch.Tensor | None
    graph: torch.cuda.CUDAGraph | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DM05Model(DMPreTrainedModel):
    """Core DM05 model.

    The model owns the Gemma3 VLM, the action expert, and the projection layers
    used to bridge action vectors to the action expert hidden size.
    - ``action_in_proj`` / ``action_out_proj`` for action-dimension and action
      expert hidden-size projection.
    - ``time_mlp_in`` / ``time_mlp_out`` for action expert AdaRMSNorm time
      conditioning.
    """

    def __init__(self, config: DM05Config):
        super().__init__(config)

        vlm_config_ref = config.vlm_config
        action_config_ref = config.action_config

        logger.info(f"Final DM05Config: {config}")

        # Initialize the VLM.
        self.vlm = Gemma3ForConditionalGeneration(vlm_config_ref)
        validate_action_config_compatible(
            action_config_ref,
            self.language_model.config,
        )

        # Initialize the action expert (Gemma3TextModel without lm_head), preserving suffix-only semantics.
        self.action_expert = DM05ActionExpert(action_config_ref)

        ae_hidden_size = action_config_ref.hidden_size

        # Action projection layers.
        self.action_in_proj = nn.Linear(config.action_dim, ae_hidden_size)
        self.action_out_proj = nn.Linear(ae_hidden_size, config.action_dim)

        # Time-conditioning layers.
        self.time_mlp_in = nn.Linear(ae_hidden_size, ae_hidden_size)
        self.time_mlp_out = nn.Linear(ae_hidden_size, ae_hidden_size)

    @property
    def visual(self):
        """Return the VLM vision module; Gemma3 uses ``vision_tower``."""
        return self.vlm.model.vision_tower

    @property
    def language_model(self):
        return self.vlm.model.language_model

    @property
    def lm_head(self):
        return self.vlm.lm_head

    def embed_language_tokens(self, tokens):
        return self.language_model.embed_tokens(tokens)

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    def parameter_summary(self, only_trainable: bool = False) -> dict[str, int]:
        result = {}
        for name, module in self.named_children():
            if only_trainable:
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            else:
                count = sum(p.numel() for p in module.parameters())
            result[name] = count

        if only_trainable:
            total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            total = sum(p.numel() for p in self.parameters())
        result["total"] = total
        return result


# ---------------------------------------------------------------------------
# ForCausalLM
# ---------------------------------------------------------------------------


class DM05ForConditionalGeneration(DMPreTrainedModel):
    """Full DM05 model with training and action inference support.

    Methods:
    - ``forward()``: training with flow matching.
    - ``inference_action()``: action inference with Euler sampling.
    """

    config_class = DM05Config
    _tied_weights_keys = {
        "model.vlm.lm_head.weight": "model.vlm.model.language_model.embed_tokens.weight",
    }

    def __init__(self, config: DM05Config):
        super().__init__(config)
        config.model_type = self.config_class.model_type
        self.all_tied_weights_keys = dict(getattr(self, "_tied_weights_keys", {}))
        self._real_init(config)
        self._init_suffix_graph_runtime_state()

    def _init_suffix_graph_runtime_state(self) -> None:
        self._suffix_graph_profiles = OrderedDict()
        self._suffix_graph_candidates = OrderedDict()
        self._suffix_graph_disabled_profiles = OrderedDict()
        self._suffix_graph_lock = threading.RLock()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_suffix_graph_profiles", None)
        state.pop("_suffix_graph_candidates", None)
        state.pop("_suffix_graph_disabled_profiles", None)
        state.pop("_suffix_graph_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._init_suffix_graph_runtime_state()

    def _real_init(self, config: DM05Config):
        self.model = DM05Model(config)
        self.post_init()

        # Patch decoder layers to safely handle KV cache during checkpointing.
        patch_decoder_layers(self.model.vlm.model)

    def freeze_vlm_embedding(self):
        """Freeze the Gemma VLM token embedding layer.

        Because ``lm_head.weight`` and ``embed_tokens.weight`` are tied through
        ``_tied_weights_keys`` as the same tensor, ``lm_head`` is frozen as
        well.
        """
        embed = self.model.vlm.model.language_model.embed_tokens
        for p in embed.parameters():
            p.requires_grad = False
        logger.info("Frozen VLM token embedding (and tied lm_head).")

    def _resolve_llm_attn_implementation(
        self,
        requested: Literal["auto", "eager", "sdpa", "flex_attention"],
    ) -> str:
        if requested == "auto":
            return "flex_attention" if is_flex_attention_available() else "sdpa"

        if requested == "flex_attention":
            if not is_flex_attention_available():
                raise ImportError(
                    "flex_attention requires PyTorch FlexAttention support "
                    "(for example torch>=2.5 with torch.nn.attention.flex_attention)"
                )
            return requested

        return requested

    def _resolve_vision_attn_implementation(
        self,
        requested: Literal["auto", "eager", "sdpa", "flash_attention_2"],
        *,
        bf16: bool,
    ) -> str:
        if requested == "auto":
            if bf16 and is_flash_attention_2_available():
                return "flash_attention_2"
            return "sdpa"

        if requested == "flash_attention_2":
            if not bf16:
                raise ValueError("flash_attention_2 requires bf16=True")
            if not torch.cuda.is_available():
                raise RuntimeError("flash_attention_2 requires CUDA")
            if not is_flash_attention_2_available():
                raise ImportError("flash_attention_2 requires the flash_attn package")
            return requested

        return requested

    def _resolve_action_attn_implementation(
        self,
        requested: Literal["auto", "eager", "sdpa", "flex_attention"],
    ) -> str:
        if requested == "auto":
            return "flex_attention" if is_flex_attention_available() else "sdpa"

        if requested == "flex_attention":
            if not is_flex_attention_available():
                raise ImportError(
                    "flex_attention requires PyTorch FlexAttention support "
                    "(for example torch>=2.5 with torch.nn.attention.flex_attention)"
                )
            return requested

        return requested

    def set_attention_implementation(
        self,
        *,
        llm_attn_implementation: Literal[
            "auto", "eager", "sdpa", "flex_attention"
        ] = "auto",
        vision_attn_implementation: Literal[
            "auto", "eager", "sdpa", "flash_attention_2"
        ] = "auto",
        action_attn_implementation: Literal[
            "auto", "eager", "sdpa", "flex_attention"
        ] = "auto",
        bf16: bool = True,
    ) -> None:
        """Configure runtime VLM and action expert attention backends."""
        llm_attn_valid = llm_attn_implementation in {
            "auto",
            "eager",
            "sdpa",
            "flex_attention",
        }
        vision_attn_valid = vision_attn_implementation in {
            "auto",
            "eager",
            "sdpa",
            "flash_attention_2",
        }
        action_attn_valid = action_attn_implementation in {
            "auto",
            "eager",
            "sdpa",
            "flex_attention",
        }
        assert llm_attn_valid and vision_attn_valid and action_attn_valid, (
            "Invalid DM05 attention implementation: "
            f"llm={llm_attn_implementation!r}, "
            f"vision={vision_attn_implementation!r}, "
            f"action={action_attn_implementation!r}"
        )
        llm_attn = self._resolve_llm_attn_implementation(llm_attn_implementation)
        vision_attn = self._resolve_vision_attn_implementation(
            vision_attn_implementation,
            bf16=bf16,
        )
        action_attn = self._resolve_action_attn_implementation(
            action_attn_implementation
        )
        self.model.vlm.set_attn_implementation(
            {
                "": llm_attn,
                "text_config": llm_attn,
                "vision_config": vision_attn,
            }
        )
        self.model.action_expert.set_action_attention_backend(action_attn)
        logger.info(
            "Configured runtime attention implementations: "
            "LLM=%s, Vision=%s, Action=%s",
            llm_attn,
            vision_attn,
            action_attn,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        token_type_ids=None,
        **kwargs,
    ):
        return self.model.vlm.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            **kwargs,
        )

    def _apply_liger_kernel(self):
        """Apply Liger Kernel fused optimizations to the VLM and Action Expert."""
        try:
            from liger_kernel.transformers import (
                apply_liger_kernel_to_gemma3,
                apply_liger_kernel_to_gemma3_text,
            )
        except ImportError:
            logger.warning(
                "liger-kernel is not installed; skipping Liger optimizations. "
                "Install it with: pip install liger-kernel"
            )
            return

        # Patch the VLM (Gemma3ForConditionalGeneration):
        # RMSNorm / GeGLU / RoPE / SigLIP LayerNorm.
        apply_liger_kernel_to_gemma3(
            model=self.model.vlm,
            rope=True,
            rms_norm=True,
            geglu=True,
            layer_norm=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

        # Patch the Action Expert (Gemma3TextModel):
        # RMSNorm / GeGLU. RoPE and loss are handled by suffix forward.
        apply_liger_kernel_to_gemma3_text(
            model=self.model.action_expert,
            rope=False,
            rms_norm=True,
            geglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

        logger.info(
            "Liger Kernel applied: VLM (RMSNorm+GeGLU+RoPE+LayerNorm), "
            "AE (RMSNorm+GeGLU)"
        )

    def enable_gradient_checkpointing(
        self,
        *,
        vlm_gradient_checkpointing: bool = True,
        ae_gradient_checkpointing: bool = True,
    ):
        """Configure runtime VLM prefix and AE suffix gradient checkpointing."""
        vlm_gradient_checkpointing = bool(vlm_gradient_checkpointing)
        ae_gradient_checkpointing = bool(ae_gradient_checkpointing)

        if vlm_gradient_checkpointing:
            self.model.vlm.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            vision_tower = getattr(self.model.vlm.model, "vision_tower", None)
            if vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing_enable"
            ):
                vision_tower.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            elif vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing"
            ):
                vision_tower.gradient_checkpointing = True
        else:
            self.model.vlm.model.language_model.gradient_checkpointing_disable()
            vision_tower = getattr(self.model.vlm.model, "vision_tower", None)
            if vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing_disable"
            ):
                vision_tower.gradient_checkpointing_disable()
            elif vision_tower is not None and hasattr(
                vision_tower, "gradient_checkpointing"
            ):
                vision_tower.gradient_checkpointing = False

        if ae_gradient_checkpointing:
            self.model.action_expert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.model.action_expert.gradient_checkpointing_disable()

    # -----------------------------------------------------------------------
    # Training forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        action: torch.FloatTensor | None = None,
        action_mask: torch.BoolTensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        **kwargs,
    ) -> Gemma3CausalLMOutputWithPast:
        """Run the training forward pass and compute flow-matching loss."""
        batch_size = input_ids.shape[0]

        # Step 1: prefix forward — history via unused0 scatter, current views via VLM.
        kv_cache, prefix_len = self._compute_prefix_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            history_pixel_values=history_pixel_values,
            history_mask=history_mask,
            cache_cls=VLADynamicCache,
        )

        # Step 2: run suffix forward for flow matching.
        noise = torch.randn_like(action)
        time = (
            torch.distributions.Beta(1.5, 1.0)
            .sample((batch_size,))
            .to(action.device, dtype=action.dtype)
            * 0.999
            + 0.001
        )
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action
        x_t = x_t * action_mask
        u_t = noise - action

        # Build suffix embeddings and per-layer time conditioning.
        suffix_embeds = self.model.action_in_proj(x_t)
        adarms_cond = self._build_adarms_cond(time, suffix_embeds.dtype)

        # Run suffix forward with fused attention.
        suffix_len = suffix_embeds.shape[1]

        invisible_prefix_token_ids = (HISTORY_PAD_TOKEN_ID,)
        suffix_attn_mask = make_suffix_attn_mask(
            input_ids=input_ids,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            batch_size=batch_size,
            device=suffix_embeds.device,
            dtype=suffix_embeds.dtype,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )

        suffix_position_ids = self._build_suffix_position_ids(
            prefix_len,
            suffix_len,
            suffix_embeds.device,
            input_ids=input_ids,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )

        suffix_out = self._suffix_forward(
            suffix_embeds=suffix_embeds,
            attention_mask=suffix_attn_mask,
            position_ids=suffix_position_ids,
            past_key_values=kv_cache,
            adarms_cond=adarms_cond,
        )

        # Compute the flow-matching loss.
        v_t = self.model.action_out_proj(suffix_out).to(torch.float32)
        u_t = u_t.to(dtype=v_t.dtype)

        elem_mse = F.mse_loss(v_t, u_t, reduction="none")  # [B, T, D]

        per_sample_fm = (elem_mse * action_mask).sum(dim=(1, 2)) / action_mask.sum(
            dim=(1, 2)
        )
        fm_loss = per_sample_fm.mean()

        return DM05OutputWithPast(
            loss=fm_loss,
            fm_loss=fm_loss,
            logits=None,
            past_key_values=None,
        )

    # -----------------------------------------------------------------------
    # Action inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        diffusion_steps: int = 10,
        past_key_values: DynamicCache | None = None,
        action_mask: torch.BoolTensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self._can_use_suffix_graph(input_ids):
            with self._suffix_graph_lock:
                return self._inference_action_impl(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    states=states,
                    image_masks=image_masks,
                    diffusion_steps=diffusion_steps,
                    past_key_values=past_key_values,
                    action_mask=action_mask,
                    history_pixel_values=history_pixel_values,
                    history_mask=history_mask,
                    **kwargs,
                )
        return self._inference_action_impl(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            states=states,
            image_masks=image_masks,
            diffusion_steps=diffusion_steps,
            past_key_values=past_key_values,
            action_mask=action_mask,
            history_pixel_values=history_pixel_values,
            history_mask=history_mask,
            **kwargs,
        )

    def _inference_action_impl(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        diffusion_steps: int = 10,
        past_key_values: DynamicCache | None = None,
        action_mask: torch.BoolTensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kv_cache, prefix_len = self._compute_prefix_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            history_pixel_values=history_pixel_values,
            history_mask=history_mask,
            cache_cls=DynamicCache,
        )

        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.model.action_in_proj.weight.dtype

        x_t = torch.randn(
            batch_size,
            self.model.config.chunk_size,
            self.model.config.action_dim,
            device=device,
            dtype=dtype,
        )
        if self._can_use_suffix_graph(x_t):
            graph_result = self._run_suffix_graph(
                input_ids=input_ids,
                kv_cache=kv_cache,
                prefix_len=prefix_len,
                initial_noise=x_t,
                diffusion_steps=diffusion_steps,
                action_mask=action_mask,
            )
            if graph_result is not None:
                return graph_result

        time_val = 1.0
        dt = -1.0 / diffusion_steps
        for _ in range(diffusion_steps):
            time_tensor = torch.full(
                (batch_size,), time_val, device=device, dtype=dtype
            )
            if action_mask is not None:
                x_t = x_t * action_mask
            suffix_embeds = self.model.action_in_proj(x_t)
            adarms_cond = self._build_adarms_cond(time_tensor, suffix_embeds.dtype)
            suffix_len = int(suffix_embeds.shape[1])
            invisible_prefix_token_ids = (HISTORY_PAD_TOKEN_ID,)
            suffix_attn_mask = make_suffix_attn_mask(
                input_ids=input_ids,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                batch_size=batch_size,
                device=suffix_embeds.device,
                dtype=suffix_embeds.dtype,
                pad_token_id=self.model.vlm.model.language_model.padding_idx,
                invisible_prefix_token_ids=invisible_prefix_token_ids,
            )
            suffix_position_ids = self._build_suffix_position_ids(
                prefix_len,
                suffix_len,
                device,
                input_ids=input_ids,
                pad_token_id=self.model.vlm.model.language_model.padding_idx,
                invisible_prefix_token_ids=invisible_prefix_token_ids,
            )

            suffix_out = self._suffix_forward(
                suffix_embeds=suffix_embeds,
                attention_mask=suffix_attn_mask,
                position_ids=suffix_position_ids,
                past_key_values=kv_cache,
                adarms_cond=adarms_cond,
            )

            v_t = self.model.action_out_proj(suffix_out)
            x_t = x_t + v_t * dt
            time_val += dt
        return x_t

    def _can_use_suffix_graph(self, initial_noise: torch.Tensor) -> bool:
        return (
            not self.training
            and initial_noise.is_cuda
            and self.model.action_expert._suffix_attn_backend == "sdpa"
        )

    def _suffix_graph_key(
        self,
        *,
        prefix_len: int,
        initial_noise: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor | None,
    ) -> tuple:
        graph_modules = (
            self.model.action_in_proj,
            self.model.action_expert,
            self.model.action_out_proj,
            self.model.time_mlp_in,
            self.model.time_mlp_out,
        )
        return (
            tuple(initial_noise.shape),
            int(prefix_len),
            int(diffusion_steps),
            initial_noise.device,
            initial_noise.dtype,
            (
                None
                if action_mask is None
                else (tuple(action_mask.shape), action_mask.device, action_mask.dtype)
            ),
            tuple(
                (id(parameter), parameter._version)
                for module in graph_modules
                for parameter in module.parameters()
            ),
        )

    def _suffix_graph_prefix_bucket_len(self, prefix_len: int) -> int:
        alignment = int(_SUFFIX_GRAPH_PREFIX_BUCKET_ALIGNMENT)
        return int(math.ceil(int(prefix_len) / alignment) * alignment)

    def _pad_suffix_graph_input_ids(
        self,
        input_ids: torch.LongTensor,
        target_len: int,
    ) -> torch.LongTensor:
        input_len = int(input_ids.shape[1])
        if input_len == target_len:
            return input_ids
        if input_len > target_len:
            raise ValueError(
                f"suffix graph prefix bucket {target_len} is shorter than input "
                f"length {input_len}."
            )
        pad_token_id = int(self.model.vlm.model.language_model.padding_idx)
        return F.pad(input_ids, (0, target_len - input_len), value=pad_token_id)

    @staticmethod
    def _copy_prefix_cache_tensor(
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> None:
        target.zero_()
        source_len = int(source.shape[2])
        target_len = int(target.shape[2])
        if source_len > target_len:
            raise ValueError(
                f"suffix graph prefix cache bucket {target_len} is shorter than "
                f"source prefix cache {source_len}."
            )
        target[:, :, :source_len, :].copy_(source)

    def _make_suffix_graph_cache_tensor(
        self,
        source: torch.Tensor,
        target_prefix_len: int,
    ) -> torch.Tensor:
        target = source.detach().new_zeros(
            *source.shape[:2],
            int(target_prefix_len),
            source.shape[3],
        )
        self._copy_prefix_cache_tensor(target, source)
        return target

    def _cache_suffix_graph_profile(
        self,
        profile_key: tuple,
        profile: DM05SuffixGraphProfile,
    ) -> None:
        self._suffix_graph_disabled_profiles.pop(profile_key, None)
        self._suffix_graph_profiles[profile_key] = profile
        self._suffix_graph_profiles.move_to_end(profile_key)
        if len(self._suffix_graph_profiles) > _SUFFIX_GRAPH_PROFILE_CACHE_SIZE:
            _, retired = self._suffix_graph_profiles.popitem(last=False)
            if retired.graph is not None:
                retired.graph.reset()

    def _retire_suffix_graph_profile(self, profile_key: tuple) -> None:
        profile = self._suffix_graph_profiles.pop(profile_key, None)
        self._suffix_graph_candidates.pop(profile_key, None)
        if profile is not None and profile.graph is not None:
            profile.graph.reset()

    def _disable_suffix_graph_profile(
        self,
        profile_key: tuple,
        reason: Exception,
    ) -> None:
        self._retire_suffix_graph_profile(profile_key)
        self._suffix_graph_disabled_profiles[profile_key] = None
        self._suffix_graph_disabled_profiles.move_to_end(profile_key)
        if len(self._suffix_graph_disabled_profiles) > _SUFFIX_GRAPH_PROFILE_CACHE_SIZE:
            self._suffix_graph_disabled_profiles.popitem(last=False)
        logger.warning(
            "Disabling DM05 suffix CUDA Graph profile after failure; "
            "falling back to eager suffix decode: %s",
            reason,
            exc_info=True,
        )

    def _should_capture_suffix_graph_profile(self, profile_key: tuple) -> bool:
        if profile_key in self._suffix_graph_candidates:
            del self._suffix_graph_candidates[profile_key]
            return True
        self._suffix_graph_candidates[profile_key] = None
        if len(self._suffix_graph_candidates) > _SUFFIX_GRAPH_PROFILE_CACHE_SIZE:
            self._suffix_graph_candidates.popitem(last=False)
        return False

    def _build_suffix_metadata(
        self,
        *,
        input_ids: torch.LongTensor,
        prefix_len: int,
        suffix_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        invisible_prefix_token_ids = (HISTORY_PAD_TOKEN_ID,)
        attention_mask = make_suffix_attn_mask(
            input_ids=input_ids,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            batch_size=int(input_ids.shape[0]),
            device=device,
            dtype=dtype,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        position_ids = self._build_suffix_position_ids(
            prefix_len,
            suffix_len,
            device,
            input_ids=input_ids,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        return attention_mask, position_ids

    @torch.no_grad()
    def _run_suffix_graph(
        self,
        *,
        input_ids: torch.LongTensor,
        kv_cache: Cache,
        prefix_len: int,
        initial_noise: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        graph_prefix_len = self._suffix_graph_prefix_bucket_len(prefix_len)
        profile_key = self._suffix_graph_key(
            prefix_len=graph_prefix_len,
            initial_noise=initial_noise,
            diffusion_steps=diffusion_steps,
            action_mask=action_mask,
        )
        with self._suffix_graph_lock:
            if profile_key in self._suffix_graph_disabled_profiles:
                return None
            try:
                profile = self._suffix_graph_profiles.get(profile_key)
                if profile is None:
                    if not self._should_capture_suffix_graph_profile(profile_key):
                        return None
                    profile = self._capture_suffix_graph_profile(
                        input_ids=input_ids,
                        kv_cache=kv_cache,
                        prefix_len=graph_prefix_len,
                        initial_noise=initial_noise,
                        diffusion_steps=diffusion_steps,
                        action_mask=action_mask,
                    )
                self._cache_suffix_graph_profile(profile_key, profile)

                if profile is None or profile.graph is None:
                    raise RuntimeError("Suffix CUDA Graph profile was not captured.")
                self._copy_suffix_graph_inputs(
                    profile,
                    input_ids=input_ids,
                    kv_cache=kv_cache,
                    initial_noise=initial_noise,
                    action_mask=action_mask,
                )

                time_value = 1.0
                dt = -1.0 / diffusion_steps
                for _ in range(diffusion_steps):
                    profile.time.fill_(time_value)
                    profile.graph.replay()
                    time_value += dt
                return profile.state.clone()
            except Exception as exc:
                self._disable_suffix_graph_profile(profile_key, exc)
                return None

    @torch.no_grad()
    def _capture_suffix_graph_profile(
        self,
        *,
        input_ids: torch.LongTensor,
        kv_cache: Cache,
        prefix_len: int,
        initial_noise: torch.Tensor,
        diffusion_steps: int,
        action_mask: torch.Tensor | None,
    ) -> DM05SuffixGraphProfile:
        padded_input_ids = self._pad_suffix_graph_input_ids(input_ids, prefix_len)
        attention_mask, position_ids = self._build_suffix_metadata(
            input_ids=padded_input_ids,
            prefix_len=prefix_len,
            suffix_len=int(initial_noise.shape[1]),
            device=initial_noise.device,
            dtype=initial_noise.dtype,
        )
        source_keys, source_values = self._extract_prefix_cache_tensors(kv_cache)
        profile = DM05SuffixGraphProfile(
            prefix_len=int(prefix_len),
            diffusion_steps=int(diffusion_steps),
            prefix_cache_keys=tuple(
                self._make_suffix_graph_cache_tensor(tensor, prefix_len)
                for tensor in source_keys
            ),
            prefix_cache_values=tuple(
                self._make_suffix_graph_cache_tensor(tensor, prefix_len)
                for tensor in source_values
            ),
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=initial_noise.detach().clone(),
            time=torch.ones(
                int(initial_noise.shape[0]),
                device=initial_noise.device,
                dtype=initial_noise.dtype,
            ),
            action_mask=(
                None
                if action_mask is None
                else action_mask.new_empty(initial_noise.shape).copy_(action_mask)
            ),
        )

        warmup_stream = torch.cuda.Stream(device=initial_noise.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(initial_noise.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                self._run_suffix_graph_step(profile)
        torch.cuda.current_stream(initial_noise.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(initial_noise.device)

        profile.state.copy_(initial_noise)
        profile.time.fill_(1.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_suffix_graph_step(profile)
        profile.graph = graph
        return profile

    def _copy_suffix_graph_inputs(
        self,
        profile: DM05SuffixGraphProfile,
        *,
        input_ids: torch.LongTensor,
        kv_cache: Cache,
        initial_noise: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> None:
        profile.state.copy_(initial_noise)
        if profile.action_mask is not None:
            if action_mask is None:
                raise ValueError(
                    "action_mask does not match the captured graph profile"
                )
            profile.action_mask.copy_(action_mask)

        padded_input_ids = self._pad_suffix_graph_input_ids(
            input_ids,
            profile.prefix_len,
        )
        attention_mask, position_ids = self._build_suffix_metadata(
            input_ids=padded_input_ids,
            prefix_len=profile.prefix_len,
            suffix_len=int(initial_noise.shape[1]),
            device=initial_noise.device,
            dtype=initial_noise.dtype,
        )
        profile.attention_mask.copy_(attention_mask)
        profile.position_ids.copy_(position_ids)

        source_keys, source_values = self._extract_prefix_cache_tensors(kv_cache)
        for target, source in zip(profile.prefix_cache_keys, source_keys, strict=True):
            self._copy_prefix_cache_tensor(target, source)
        for target, source in zip(
            profile.prefix_cache_values, source_values, strict=True
        ):
            self._copy_prefix_cache_tensor(target, source)

    def _run_suffix_graph_step(self, profile: DM05SuffixGraphProfile) -> None:
        x_t = profile.state
        if profile.action_mask is not None:
            x_t = x_t * profile.action_mask
        suffix_embeds = self.model.action_in_proj(x_t)
        adarms_cond = self._build_adarms_cond(profile.time, suffix_embeds.dtype)
        suffix_out = self.model.action_expert(
            suffix_embeds=suffix_embeds,
            attention_mask=profile.attention_mask,
            position_ids=profile.position_ids,
            prefix_cache_keys=profile.prefix_cache_keys,
            prefix_cache_values=profile.prefix_cache_values,
            adarms_cond=adarms_cond,
        )
        updated = x_t + self.model.action_out_proj(suffix_out) * (
            -1.0 / profile.diffusion_steps
        )
        profile.state.copy_(updated)

    def _clear_suffix_graph_profile(self) -> None:
        for profile in self._suffix_graph_profiles.values():
            if profile.graph is not None:
                profile.graph.reset()
        self._suffix_graph_profiles.clear()
        self._suffix_graph_candidates.clear()
        self._suffix_graph_disabled_profiles.clear()

    def _compute_prefix_cache(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        token_type_ids: torch.LongTensor | None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.BoolTensor | None = None,
        cache_cls: type[Cache] = DynamicCache,
    ) -> tuple[Cache, int]:
        """Fill VLM KV cache; history images are injected separately from current views.

        History uses ``<unused0>`` placeholders replaced via vision features +
        ``masked_scatter``. Invalid 0525 slots use ``<unused1>`` pads; those
        positions are zeroed in embeds and masked in attention/position ids
        (dexbotic-open parity). Current camera images still go through the
        standard Gemma3 multimodal path (``pixel_values`` + ``token_type_ids``).
        """
        kv_cache = cache_cls(config=self.model.language_model.config)
        prefix_inputs_embeds = None
        vlm_model = self.model.vlm.model
        embed_tokens = vlm_model.get_input_embeddings()
        has_history_pixels = (
            history_pixel_values is not None
            and history_mask is not None
            and bool(history_mask.any().item())
        )
        has_history_pads = bool((input_ids == HISTORY_PAD_TOKEN_ID).any().item())

        if has_history_pixels or has_history_pads:
            inputs_embeds = embed_tokens(input_ids)
            if has_history_pads:
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[input_ids == HISTORY_PAD_TOKEN_ID] = 0

            if has_history_pixels:
                pixels = history_pixel_values.to(
                    device=inputs_embeds.device,
                    dtype=next(vlm_model.vision_tower.parameters()).dtype,
                )
                image_features = vlm_model.get_image_features(
                    pixels, return_dict=True
                ).pooler_output

                spatial = int(image_features.shape[1] ** 0.5)
                hidden = image_features.shape[-1]
                grid = image_features.view(-1, spatial, spatial, hidden).permute(
                    0, 3, 1, 2
                )
                grid = F.adaptive_avg_pool2d(
                    grid, output_size=(HISTORY_POOL_SIZE, HISTORY_POOL_SIZE)
                )
                image_features = grid.permute(0, 2, 3, 1).reshape(
                    -1, HISTORY_POOL_SIZE * HISTORY_POOL_SIZE, hidden
                )

                history_mask_expanded = history_mask.unsqueeze(-1).expand_as(
                    inputs_embeds
                )
                prefix_inputs_embeds = inputs_embeds.masked_scatter(
                    history_mask_expanded, image_features
                )
            else:
                prefix_inputs_embeds = inputs_embeds

        prefix_attention_mask = mask_history_pad_tokens_in_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        prefix_position_ids = (prefix_attention_mask.cumsum(dim=-1) - 1).clamp_min_(0)
        model_kwargs = {
            "input_ids": None if prefix_inputs_embeds is not None else input_ids,
            "attention_mask": prefix_attention_mask,
            "position_ids": prefix_position_ids,
            "past_key_values": kv_cache,
            "inputs_embeds": prefix_inputs_embeds,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "use_cache": True,
        }

        self.model.vlm.model(**model_kwargs)
        return kv_cache, kv_cache.get_seq_length()

    def _extract_prefix_cache_tensors(
        self,
        kv_cache: Cache,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        n_layers = len(self.model.action_expert.layers)
        keys = []
        values = []
        for layer_idx in range(n_layers):
            layer = kv_cache.layers[layer_idx]
            keys.append(layer.keys)
            values.append(layer.values)
        return tuple(keys), tuple(values)

    def _build_adarms_cond(
        self,
        time: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ae_hidden = self.model.action_in_proj.out_features
        time_emb = posemb_sincos(time, ae_hidden, max_period=4.0).to(dtype)
        cond = self.model.time_mlp_in(time_emb)
        cond = F.silu(cond)
        cond = self.model.time_mlp_out(cond)
        return F.silu(cond)

    def _build_suffix_position_ids(
        self,
        prefix_len,
        suffix_len,
        device,
        input_ids: torch.Tensor,
        pad_token_id=0,
        invisible_prefix_token_ids: tuple[int, ...] = (),
    ):
        valid_prefix = input_ids[:, :prefix_len] != pad_token_id  # [B, prefix_len]
        for token_id in invisible_prefix_token_ids:
            valid_prefix = valid_prefix & (input_ids[:, :prefix_len] != token_id)
        effective_prefix_len = valid_prefix.sum(dim=1)  # [B]

        # suffix position ids = [effective_prefix_len, effective_prefix_len+1, ...]
        suffix_offsets = torch.arange(suffix_len, device=device).unsqueeze(
            0
        )  # [1, suffix_len]
        pos = effective_prefix_len.unsqueeze(1) + suffix_offsets  #  [B, suffix_len]
        return pos

    def _suffix_forward(
        self,
        suffix_embeds: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Extract cache tensors and run the action expert suffix forward."""
        prefix_cache_keys, prefix_cache_values = self._extract_prefix_cache_tensors(
            past_key_values,
        )

        return self.model.action_expert(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefix_cache_keys=prefix_cache_keys,
            prefix_cache_values=prefix_cache_values,
            adarms_cond=adarms_cond,
        )


AutoConfig.register("dm05", DM05Config)
AutoModelForCausalLM.register(DM05Config, DM05ForConditionalGeneration)
