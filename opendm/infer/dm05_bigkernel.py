"""Shape-specialized Triton kernels and buffers for DM05 fast inference.

This module implements the fixed-shape GPU kernels used by the DM05 fast path,
covering prefix image-feature merge, prefix decoder fused blocks, and suffix
action-expert attention/MLP kernels together with their reusable work buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh as tl_tanh

SUPPORTED_HIDDEN_SIZE = 1024
SUPPORTED_INTERMEDIATE_SIZE = 4096
SUPPORTED_HEAD_DIM = 256
SUPPORTED_Q_HEADS = 8
SUPPORTED_KV_HEADS = 4

# Fixed Gemma3-4B prefix decoder shape used by the deployed fast path.
SUPPORTED_PREFIX_HIDDEN_SIZE = 2560
SUPPORTED_PREFIX_INTERMEDIATE_SIZE = 10240
SUPPORTED_PREFIX_HEAD_DIM = 256
SUPPORTED_PREFIX_Q_HEADS = 8
SUPPORTED_PREFIX_KV_HEADS = 4


@dataclass(frozen=True)
class DM05TransformerShape:
    hidden_size: int
    intermediate_size: int
    head_dim: int
    q_heads: int
    kv_heads: int

    @property
    def total_qkv(self) -> int:
        return (self.q_heads + 2 * self.kv_heads) * self.head_dim


SUPPORTED_SUFFIX_SHAPE = DM05TransformerShape(
    hidden_size=SUPPORTED_HIDDEN_SIZE,
    intermediate_size=SUPPORTED_INTERMEDIATE_SIZE,
    head_dim=SUPPORTED_HEAD_DIM,
    q_heads=SUPPORTED_Q_HEADS,
    kv_heads=SUPPORTED_KV_HEADS,
)


@dataclass
class DM05BigKernelBuffers:
    rows: int
    batch_size: int
    seq_len: int
    max_kv_len: int
    dtype: torch.dtype
    device: torch.device
    shape: DM05TransformerShape
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    residual: torch.Tensor
    mlp_input: torch.Tensor
    mlp_hidden: torch.Tensor
    mlp_out: torch.Tensor
    layer_out: torch.Tensor
    attn_logits: torch.Tensor
    attn_probs: torch.Tensor
    attn_out: torch.Tensor


@dataclass
class DM05PrefixDecoderBuffers:
    rows: int
    batch_size: int
    seq_len: int
    dtype: torch.dtype
    device: torch.device
    shape: DM05TransformerShape
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    residual: torch.Tensor
    mlp_input: torch.Tensor
    mlp_gate_up: torch.Tensor
    mlp_hidden: torch.Tensor
    mlp_out: torch.Tensor
    layer_out: torch.Tensor


@dataclass(frozen=True)
class KernelLaunchConfig:
    max_row_programs: int = 1024
    norm: tuple[int, int] = (1024, 8)
    qkv_warps: int = 8
    prefix_image: tuple[int, int, int] = (1024, 1024, 8)
    qk: tuple[int, int, int, int, int] = (16, 32, 64, 4, 4)
    softmax: tuple[int, int, int] = (4, 1024, 8)
    attn_value: tuple[int, int, int, int, int] = (16, 64, 32, 4, 4)
    matmul: tuple[int, int, int, int] = (16, 64, 4, 4)
    geglu_block_k: int = 32
    mlp_down_block_k: int = 64


DEFAULT_LAUNCH_CONFIG = KernelLaunchConfig()
_SM89_SUFFIX_10_LAUNCH_CONFIG = KernelLaunchConfig(
    softmax=(2, 1024, 8),
    attn_value=(16, 64, 64, 4, 4),
    mlp_down_block_k=128,
)
_L = DEFAULT_LAUNCH_CONFIG


@lru_cache
def _cuda_device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _suffix_launch_config(buffers: DM05BigKernelBuffers) -> KernelLaunchConfig:
    if buffers.seq_len == 10 and buffers.device.type == "cuda":
        device_index = buffers.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if _cuda_device_capability(device_index) == (8, 9):
            return _SM89_SUFFIX_10_LAUNCH_CONFIG
    return _L


def make_suffix_buffers(
    *,
    batch_size: int,
    seq_len: int,
    max_kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
    shape: DM05TransformerShape = SUPPORTED_SUFFIX_SHAPE,
) -> DM05BigKernelBuffers:
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    max_kv_len = int(max_kv_len)
    rows = batch_size * seq_len

    def empty(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype)

    return DM05BigKernelBuffers(
        rows=rows,
        batch_size=batch_size,
        seq_len=seq_len,
        max_kv_len=max_kv_len,
        dtype=dtype,
        device=device,
        shape=shape,
        qkv=empty((rows, shape.total_qkv)),
        query=empty((batch_size, shape.q_heads, seq_len, shape.head_dim)),
        key=empty((batch_size, shape.kv_heads, seq_len, shape.head_dim)),
        value=empty((batch_size, shape.kv_heads, seq_len, shape.head_dim)),
        residual=empty((batch_size, seq_len, shape.hidden_size)),
        mlp_input=empty((batch_size, seq_len, shape.hidden_size)),
        mlp_hidden=empty((rows, shape.intermediate_size)),
        mlp_out=empty((batch_size, seq_len, shape.hidden_size)),
        layer_out=empty((batch_size, seq_len, shape.hidden_size)),
        attn_logits=torch.empty(
            (batch_size, shape.q_heads, seq_len, max_kv_len),
            device=device,
            dtype=torch.float32,
        ),
        attn_probs=empty((batch_size, shape.q_heads, seq_len, max_kv_len)),
        attn_out=empty((batch_size, seq_len, shape.q_heads, shape.head_dim)),
    )


def make_prefix_decoder_buffers(
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    shape: DM05TransformerShape,
) -> DM05PrefixDecoderBuffers:
    rows = int(batch_size) * int(seq_len)

    def empty(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype)

    return DM05PrefixDecoderBuffers(
        rows=rows,
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        dtype=dtype,
        device=device,
        shape=shape,
        qkv=empty((rows, shape.total_qkv)),
        query=empty((batch_size, shape.q_heads, seq_len, shape.head_dim)),
        key=empty((batch_size, shape.kv_heads, seq_len, shape.head_dim)),
        value=empty((batch_size, shape.kv_heads, seq_len, shape.head_dim)),
        residual=empty((batch_size, seq_len, shape.hidden_size)),
        mlp_input=empty((batch_size, seq_len, shape.hidden_size)),
        mlp_gate_up=empty((rows, 2 * shape.intermediate_size)),
        mlp_hidden=empty((rows, shape.intermediate_size)),
        mlp_out=empty((batch_size, seq_len, shape.hidden_size)),
        layer_out=empty((batch_size, seq_len, shape.hidden_size)),
    )


if tl is not None:

    @triton.jit
    def _geglu_tanh_approx(gate, up):
        gate_cubed = gate * gate * gate
        tanh_arg = 0.7978845608028654 * (gate + 0.044715 * gate_cubed)
        gate_act = 0.5 * gate * (1.0 + tl_tanh(tanh_arg))
        return gate_act.to(tl.bfloat16) * up

    @triton.jit
    def _prefix_image_merge_kernel(
        input_ids_ptr,
        image_features_ptr,
        inputs_embeds_ptr,
        image_token_id: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_SCAN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        token = tl.load(input_ids_ptr + seq_idx)
        is_image = token == image_token_id

        scan_offsets = tl.arange(0, BLOCK_SCAN)
        image_count = tl.full((), 0, dtype=tl.int32)
        for start in range(0, seq_len, BLOCK_SCAN):
            offsets = start + scan_offsets
            mask = (offsets <= seq_idx) & (offsets < seq_len)
            ids = tl.load(input_ids_ptr + offsets, mask=mask, other=-1)
            image_count += tl.sum(tl.where(ids == image_token_id, 1, 0), axis=0)

        image_idx = image_count - 1
        hidden_offsets = tl.arange(0, BLOCK_H)
        for hidden_start in range(0, hidden, BLOCK_H):
            h = hidden_start + hidden_offsets
            copy_mask = (h < hidden) & is_image
            values = tl.load(
                image_features_ptr + image_idx * hidden + h,
                mask=copy_mask,
                other=0.0,
            )
            tl.store(
                inputs_embeds_ptr + seq_idx * hidden + h,
                values,
                mask=copy_mask,
            )

    @triton.jit
    def _qkv_qknorm_rope_split_kernel(
        qkv_ptr,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        rows: tl.constexpr,
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        q_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        total_qkv: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        total_heads: tl.constexpr = q_heads + kv_heads + kv_heads
        half_dim: tl.constexpr = head_dim // 2
        dim_offsets = tl.arange(0, BLOCK_D)
        for idx in range(pid, rows * total_heads, tl.num_programs(0)):
            row = idx // total_heads
            head_idx = idx - row * total_heads
            batch = row // seq_len
            token = row - batch * seq_len
            head_offset = head_idx * head_dim

            raw = tl.load(
                qkv_ptr + row * total_qkv + head_offset + dim_offsets,
                mask=dim_offsets < head_dim,
                other=0.0,
            ).to(tl.float32)

            if head_idx < q_heads + kv_heads:
                is_q = head_idx < q_heads
                rms = tl.rsqrt(tl.sum(raw * raw) / head_dim + eps)
                rot_offsets = tl.where(
                    dim_offsets < half_dim,
                    dim_offsets + half_dim,
                    dim_offsets - half_dim,
                )
                rot_raw = tl.load(
                    qkv_ptr + row * total_qkv + head_offset + rot_offsets,
                    mask=rot_offsets < head_dim,
                    other=0.0,
                ).to(tl.float32)
                if is_q:
                    norm_w = tl.load(
                        q_norm_weight_ptr + dim_offsets,
                        mask=dim_offsets < head_dim,
                        other=0.0,
                    ).to(tl.float32)
                    rot_w = tl.load(
                        q_norm_weight_ptr + rot_offsets,
                        mask=rot_offsets < head_dim,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    kv_head = head_idx - q_heads
                    norm_w = tl.load(
                        k_norm_weight_ptr + dim_offsets,
                        mask=dim_offsets < head_dim,
                        other=0.0,
                    ).to(tl.float32)
                    rot_w = tl.load(
                        k_norm_weight_ptr + rot_offsets,
                        mask=rot_offsets < head_dim,
                        other=0.0,
                    ).to(tl.float32)
                val = raw * rms * (1.0 + norm_w)
                rot = rot_raw * rms * (1.0 + rot_w)
                rot = tl.where(dim_offsets < half_dim, -rot, rot)
                cos = tl.load(
                    cos_ptr + row * head_dim + dim_offsets,
                    mask=dim_offsets < head_dim,
                    other=1.0,
                ).to(tl.float32)
                sin = tl.load(
                    sin_ptr + row * head_dim + dim_offsets,
                    mask=dim_offsets < head_dim,
                    other=0.0,
                ).to(tl.float32)
                out = val * cos + rot * sin
                if is_q:
                    tl.store(
                        query_ptr
                        + ((batch * q_heads + head_idx) * seq_len + token) * head_dim
                        + dim_offsets,
                        out.to(tl.bfloat16),
                        mask=dim_offsets < head_dim,
                    )
                else:
                    kv_head = head_idx - q_heads
                    tl.store(
                        key_ptr
                        + ((batch * kv_heads + kv_head) * seq_len + token) * head_dim
                        + dim_offsets,
                        out.to(tl.bfloat16),
                        mask=dim_offsets < head_dim,
                    )
            else:
                kv_head = head_idx - q_heads - kv_heads
                tl.store(
                    value_ptr
                    + ((batch * kv_heads + kv_head) * seq_len + token) * head_dim
                    + dim_offsets,
                    raw.to(tl.bfloat16),
                    mask=dim_offsets < head_dim,
                )

    @triton.jit
    def _input_adarmsnorm_kernel(
        x_ptr,
        modulation_ptr,
        prenorm_ptr,
        mod_stride_b: tl.constexpr,
        mod_stride_h: tl.constexpr,
        rows: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        for row in range(pid, rows, tl.num_programs(0)):
            x = tl.load(
                x_ptr + row * hidden + offs,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            rms = tl.rsqrt(tl.sum(x * x) / hidden + eps)
            batch = row // seq_len
            scale = tl.load(
                modulation_ptr + batch * mod_stride_b + offs * mod_stride_h,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            shift = tl.load(
                modulation_ptr + batch * mod_stride_b + (hidden + offs) * mod_stride_h,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            out = x * rms * (1.0 + scale) + shift
            tl.store(
                prenorm_ptr + row * hidden + offs,
                out.to(tl.bfloat16),
                mask=offs < hidden,
            )

    @triton.jit
    def _attn_postnorm_residual_prenorm_kernel(
        attn_ptr,
        suffix_ptr,
        attn_gate_ptr,
        post_norm_weight_ptr,
        mlp_modulation_ptr,
        residual_ptr,
        mlp_input_ptr,
        attn_gate_stride_b: tl.constexpr,
        attn_gate_stride_h: tl.constexpr,
        mlp_mod_stride_b: tl.constexpr,
        mlp_mod_stride_h: tl.constexpr,
        rows: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        post_eps: tl.constexpr,
        pre_eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        for row in range(pid, rows, tl.num_programs(0)):
            attn = tl.load(
                attn_ptr + row * hidden + offs,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            post_rms = tl.rsqrt(tl.sum(attn * attn) / hidden + post_eps)
            post_w = tl.load(
                post_norm_weight_ptr + offs,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            attn_normed = (attn * post_rms * (1.0 + post_w)).to(tl.bfloat16)
            suffix = tl.load(
                suffix_ptr + row * hidden + offs,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            batch = row // seq_len
            attn_gate = tl.load(
                attn_gate_ptr + batch * attn_gate_stride_b + offs * attn_gate_stride_h,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            residual = (suffix + attn_normed.to(tl.float32) * attn_gate).to(tl.bfloat16)
            tl.store(
                residual_ptr + row * hidden + offs,
                residual,
                mask=offs < hidden,
            )

            residual_fp32 = residual.to(tl.float32)
            pre_rms = tl.rsqrt(tl.sum(residual_fp32 * residual_fp32) / hidden + pre_eps)
            mlp_scale = tl.load(
                mlp_modulation_ptr + batch * mlp_mod_stride_b + offs * mlp_mod_stride_h,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            mlp_shift = tl.load(
                mlp_modulation_ptr
                + batch * mlp_mod_stride_b
                + (hidden + offs) * mlp_mod_stride_h,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            mlp_input = residual_fp32 * pre_rms * (1.0 + mlp_scale) + mlp_shift
            tl.store(
                mlp_input_ptr + row * hidden + offs,
                mlp_input.to(tl.bfloat16),
                mask=offs < hidden,
            )

    @triton.jit
    def _post_ffn_norm_residual_kernel(
        mlp_out_ptr,
        residual_ptr,
        mlp_gate_ptr,
        post_ffn_norm_weight_ptr,
        layer_out_ptr,
        mlp_gate_stride_b: tl.constexpr,
        mlp_gate_stride_h: tl.constexpr,
        rows: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        for row in range(pid, rows, tl.num_programs(0)):
            mlp_out = tl.load(
                mlp_out_ptr + row * hidden + offs,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            rms = tl.rsqrt(tl.sum(mlp_out * mlp_out) / hidden + eps)
            post_w = tl.load(
                post_ffn_norm_weight_ptr + offs,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            mlp_normed = (mlp_out * rms * (1.0 + post_w)).to(tl.bfloat16)
            residual = tl.load(
                residual_ptr + row * hidden + offs,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            batch = row // seq_len
            mlp_gate = tl.load(
                mlp_gate_ptr + batch * mlp_gate_stride_b + offs * mlp_gate_stride_h,
                mask=offs < hidden,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                layer_out_ptr + row * hidden + offs,
                (residual + mlp_normed.to(tl.float32) * mlp_gate).to(tl.bfloat16),
                mask=offs < hidden,
            )

    @triton.jit
    def _prefix_attn_postnorm_residual_prenorm_kernel(
        attn_embeds_ptr,
        hidden_states_ptr,
        post_attention_weight_ptr,
        pre_feedforward_weight_ptr,
        residual_ptr,
        mlp_input_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        post_eps: tl.constexpr,
        pre_eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        for row in range(pid, rows, tl.num_programs(0)):
            attn_sq_sum = tl.zeros((), dtype=tl.float32)
            for h_start in range(0, hidden, BLOCK_H):
                h_offsets = h_start + offs
                attn = tl.load(
                    attn_embeds_ptr + row * hidden + h_offsets,
                    mask=h_offsets < hidden,
                    other=0.0,
                ).to(tl.float32)
                attn_sq_sum += tl.sum(attn * attn, axis=0)
            post_rms = tl.rsqrt(attn_sq_sum / hidden + post_eps)

            residual_sq_sum = tl.zeros((), dtype=tl.float32)
            for h_start in range(0, hidden, BLOCK_H):
                h_offsets = h_start + offs
                h_mask = h_offsets < hidden
                attn = tl.load(
                    attn_embeds_ptr + row * hidden + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                post_w = tl.load(
                    post_attention_weight_ptr + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                attn_normed = attn * post_rms * (1.0 + post_w)
                hidden_states = tl.load(
                    hidden_states_ptr + row * hidden + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                residual = hidden_states + attn_normed
                tl.store(
                    residual_ptr + row * hidden + h_offsets,
                    residual.to(tl.bfloat16),
                    mask=h_mask,
                )
                residual_sq_sum += tl.sum(residual * residual, axis=0)

            pre_rms = tl.rsqrt(residual_sq_sum / hidden + pre_eps)
            for h_start in range(0, hidden, BLOCK_H):
                h_offsets = h_start + offs
                h_mask = h_offsets < hidden
                residual = tl.load(
                    residual_ptr + row * hidden + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                pre_w = tl.load(
                    pre_feedforward_weight_ptr + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    mlp_input_ptr + row * hidden + h_offsets,
                    (residual * pre_rms * (1.0 + pre_w)).to(tl.bfloat16),
                    mask=h_mask,
                )

    @triton.jit
    def _prefix_geglu_from_gate_up_kernel(
        gate_up_ptr,
        hidden_out_ptr,
        rows: tl.constexpr,
        intermediate: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(intermediate, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid - pid_m * grid_n

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offsets < rows
        n_mask = n_offsets < intermediate
        pair_offsets = 2 * n_offsets

        gate = tl.load(
            gate_up_ptr
            + m_offsets[:, None] * (2 * intermediate)
            + pair_offsets[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up_ptr
            + m_offsets[:, None] * (2 * intermediate)
            + (pair_offsets + 1)[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out = _geglu_tanh_approx(gate, up)
        tl.store(
            hidden_out_ptr + m_offsets[:, None] * intermediate + n_offsets[None, :],
            out.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _prefix_post_ffn_norm_residual_kernel(
        mlp_out_ptr,
        residual_ptr,
        post_ffn_norm_weight_ptr,
        layer_out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        for row in range(pid, rows, tl.num_programs(0)):
            mlp_sq_sum = tl.zeros((), dtype=tl.float32)
            for h_start in range(0, hidden, BLOCK_H):
                h_offsets = h_start + offs
                mlp_out = tl.load(
                    mlp_out_ptr + row * hidden + h_offsets,
                    mask=h_offsets < hidden,
                    other=0.0,
                ).to(tl.float32)
                mlp_sq_sum += tl.sum(mlp_out * mlp_out, axis=0)
            rms = tl.rsqrt(mlp_sq_sum / hidden + eps)

            for h_start in range(0, hidden, BLOCK_H):
                h_offsets = h_start + offs
                h_mask = h_offsets < hidden
                mlp_out = tl.load(
                    mlp_out_ptr + row * hidden + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                post_w = tl.load(
                    post_ffn_norm_weight_ptr + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                mlp_normed = mlp_out * rms * (1.0 + post_w)
                residual = tl.load(
                    residual_ptr + row * hidden + h_offsets,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    layer_out_ptr + row * hidden + h_offsets,
                    (residual + mlp_normed).to(tl.bfloat16),
                    mask=h_mask,
                )

    @triton.jit
    def _geglu_gate_up_kernel(
        x_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        hidden_out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        intermediate: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(intermediate, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid - pid_m * grid_n

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        m_mask = m_offsets < rows
        n_mask = n_offsets < intermediate

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, hidden, BLOCK_K):
            k = k_start + k_offsets
            k_mask = k < hidden
            x = tl.load(
                x_ptr + m_offsets[:, None] * hidden + k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            gate_w = tl.load(
                gate_weight_ptr + k[:, None] * intermediate + n_offsets[None, :],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            up_w = tl.load(
                up_weight_ptr + k[:, None] * intermediate + n_offsets[None, :],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            gate_acc = tl.dot(x, gate_w, gate_acc)
            up_acc = tl.dot(x, up_w, up_acc)

        out = _geglu_tanh_approx(gate_acc, up_acc)
        tl.store(
            hidden_out_ptr + m_offsets[:, None] * intermediate + n_offsets[None, :],
            out.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _mlp_down_kernel(
        hidden_ptr,
        down_weight_ptr,
        out_ptr,
        rows: tl.constexpr,
        intermediate: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(hidden, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid - pid_m * grid_n

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        m_mask = m_offsets < rows
        n_mask = n_offsets < hidden

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, intermediate, BLOCK_K):
            k = k_start + k_offsets
            k_mask = k < intermediate
            x = tl.load(
                hidden_ptr + m_offsets[:, None] * intermediate + k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                down_weight_ptr + k[:, None] * hidden + n_offsets[None, :],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + m_offsets[:, None] * hidden + n_offsets[None, :],
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _suffix_qk_logits_kernel(
        query_ptr,
        prefix_key_ptr,
        suffix_key_ptr,
        prefix_visible_mask_ptr,
        logits_ptr,
        batch_size: tl.constexpr,
        prefix_len: tl.constexpr,
        seq_len: tl.constexpr,
        max_kv_len: tl.constexpr,
        total_kv_len: tl.constexpr,
        q_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        kv_blocks = tl.cdiv(total_kv_len, BLOCK_N)
        q_blocks = tl.cdiv(seq_len, BLOCK_M)
        pid_k = pid % kv_blocks
        pid_q = (pid // kv_blocks) % q_blocks
        q_head = (pid // (kv_blocks * q_blocks)) % q_heads
        batch = pid // (kv_blocks * q_blocks * q_heads)
        kv_head = q_head // (q_heads // kv_heads)

        q_offsets = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offsets = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offsets = tl.arange(0, BLOCK_D)

        q_mask = q_offsets < seq_len
        k_valid = k_offsets < total_kv_len
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d_start in range(0, head_dim, BLOCK_D):
            d = d_start + d_offsets
            d_mask = d < head_dim
            q = tl.load(
                query_ptr
                + (
                    ((batch * q_heads + q_head) * seq_len + q_offsets[:, None])
                    * head_dim
                    + d[None, :]
                ),
                mask=q_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            prefix_k = tl.load(
                prefix_key_ptr
                + (
                    ((batch * kv_heads + kv_head) * prefix_len + k_offsets[:, None])
                    * head_dim
                    + d[None, :]
                ),
                mask=(k_offsets[:, None] < prefix_len) & d_mask[None, :],
                other=0.0,
            )
            suffix_offsets = k_offsets - prefix_len
            suffix_k = tl.load(
                suffix_key_ptr
                + (
                    ((batch * kv_heads + kv_head) * seq_len + suffix_offsets[:, None])
                    * head_dim
                    + d[None, :]
                ),
                mask=(k_offsets[:, None] >= prefix_len)
                & (k_offsets[:, None] < total_kv_len)
                & d_mask[None, :],
                other=0.0,
            )
            k = prefix_k + suffix_k
            acc = tl.dot(q, tl.trans(k), acc)

        acc *= scale
        prefix_visible = tl.load(
            prefix_visible_mask_ptr + batch * prefix_len + k_offsets,
            mask=k_offsets < prefix_len,
            other=0,
        )
        k_visible = (
            tl.where(k_offsets < prefix_len, prefix_visible != 0, True) & k_valid
        )
        acc = tl.where(k_visible[None, :], acc, -3.4028234663852886e38)
        tl.store(
            logits_ptr
            + (
                ((batch * q_heads + q_head) * seq_len + q_offsets[:, None]) * max_kv_len
                + k_offsets[None, :]
            ),
            acc,
            mask=q_mask[:, None] & k_valid[None, :],
        )

    @triton.jit
    def _suffix_softmax_kernel(
        logits_ptr,
        probs_ptr,
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        max_kv_len: tl.constexpr,
        total_kv_len: tl.constexpr,
        q_heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_blocks = tl.cdiv(seq_len, BLOCK_M)
        pid_q = pid % q_blocks
        q_head = (pid // q_blocks) % q_heads
        batch = pid // (q_blocks * q_heads)

        q_offsets = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offsets = tl.arange(0, BLOCK_K)
        q_mask = q_offsets < seq_len
        k_mask = k_offsets < total_kv_len
        vals = tl.load(
            logits_ptr
            + (
                ((batch * q_heads + q_head) * seq_len + q_offsets[:, None]) * max_kv_len
                + k_offsets[None, :]
            ),
            mask=q_mask[:, None] & k_mask[None, :],
            other=-3.4028234663852886e38,
        ).to(tl.float32)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        denom = tl.sum(vals, axis=1, keep_dims=True)
        probs = vals / denom
        tl.store(
            probs_ptr
            + (
                ((batch * q_heads + q_head) * seq_len + q_offsets[:, None]) * max_kv_len
                + k_offsets[None, :]
            ),
            probs.to(tl.bfloat16),
            mask=q_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _suffix_attn_value_kernel(
        probs_ptr,
        prefix_value_ptr,
        suffix_value_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        prefix_len: tl.constexpr,
        seq_len: tl.constexpr,
        max_kv_len: tl.constexpr,
        total_kv_len: tl.constexpr,
        q_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        d_blocks = tl.cdiv(head_dim, BLOCK_N)
        q_blocks = tl.cdiv(seq_len, BLOCK_M)
        pid_d = pid % d_blocks
        pid_q = (pid // d_blocks) % q_blocks
        q_head = (pid // (d_blocks * q_blocks)) % q_heads
        batch = pid // (d_blocks * q_blocks * q_heads)
        kv_head = q_head // (q_heads // kv_heads)

        q_offsets = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = pid_d * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        q_mask = q_offsets < seq_len
        d_mask = d_offsets < head_dim
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, total_kv_len, BLOCK_K):
            k = k_start + k_offsets
            k_valid = k < total_kv_len
            probs = tl.load(
                probs_ptr
                + (
                    ((batch * q_heads + q_head) * seq_len + q_offsets[:, None])
                    * max_kv_len
                    + k[None, :]
                ),
                mask=q_mask[:, None] & k_valid[None, :],
                other=0.0,
            )
            prefix_v = tl.load(
                prefix_value_ptr
                + (
                    ((batch * kv_heads + kv_head) * prefix_len + k[:, None]) * head_dim
                    + d_offsets[None, :]
                ),
                mask=(k[:, None] < prefix_len) & d_mask[None, :],
                other=0.0,
            )
            suffix_offsets = k - prefix_len
            suffix_v = tl.load(
                suffix_value_ptr
                + (
                    ((batch * kv_heads + kv_head) * seq_len + suffix_offsets[:, None])
                    * head_dim
                    + d_offsets[None, :]
                ),
                mask=(k[:, None] >= prefix_len)
                & (k[:, None] < total_kv_len)
                & d_mask[None, :],
                other=0.0,
            )
            value = prefix_v + suffix_v
            acc = tl.dot(probs, value, acc)

        tl.store(
            out_ptr
            + (
                ((batch * seq_len + q_offsets[:, None]) * q_heads + q_head) * head_dim
                + d_offsets[None, :]
            ),
            acc.to(tl.bfloat16),
            mask=q_mask[:, None] & d_mask[None, :],
        )


def _row_grid(rows: int) -> tuple[int]:
    return (min(int(rows), _L.max_row_programs),)


def _matmul_grid(rows: int, cols: int) -> tuple[int]:
    block_m, block_n, _, _ = _L.matmul
    return (triton.cdiv(int(rows), block_m) * triton.cdiv(int(cols), block_n),)


def launch_prefix_image_merge(
    input_ids: torch.Tensor,
    image_features: torch.Tensor,
    inputs_embeds: torch.Tensor,
    *,
    image_token_id: int,
) -> None:
    _, seq_len = input_ids.shape
    hidden = int(inputs_embeds.shape[-1])
    block_scan, block_h, num_warps = _L.prefix_image
    _prefix_image_merge_kernel[(int(seq_len),)](
        input_ids.contiguous().view(-1),
        image_features.contiguous().view(-1, hidden),
        inputs_embeds.view(-1, hidden),
        image_token_id=int(image_token_id),
        seq_len=int(seq_len),
        hidden=hidden,
        BLOCK_SCAN=block_scan,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


def project_qkv_and_launch_qknorm_rope_split(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    buffers: DM05BigKernelBuffers | DM05PrefixDecoderBuffers,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float,
) -> None:
    shape = buffers.shape
    rows = buffers.rows
    torch.mm(
        hidden_states.contiguous().view(rows, shape.hidden_size),
        qkv_weight,
        out=buffers.qkv,
    )
    _qkv_qknorm_rope_split_kernel[
        (min(rows * (shape.q_heads + 2 * shape.kv_heads), _L.max_row_programs),)
    ](
        buffers.qkv,
        q_norm_weight,
        k_norm_weight,
        cos.contiguous().view(rows, shape.head_dim),
        sin.contiguous().view(rows, shape.head_dim),
        buffers.query,
        buffers.key,
        buffers.value,
        rows=rows,
        batch_size=buffers.batch_size,
        seq_len=buffers.seq_len,
        head_dim=shape.head_dim,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        total_qkv=shape.total_qkv,
        eps=float(eps),
        BLOCK_D=shape.head_dim,
        num_warps=_L.qkv_warps,
    )


def launch_input_adarmsnorm(
    suffix_embeds: torch.Tensor,
    modulation: torch.Tensor,
    buffers: DM05BigKernelBuffers,
    *,
    eps: float,
) -> None:
    shape = buffers.shape
    block_h, num_warps = _L.norm
    _input_adarmsnorm_kernel[_row_grid(buffers.rows)](
        suffix_embeds.contiguous().view(buffers.rows, shape.hidden_size),
        modulation,
        buffers.mlp_input.view(buffers.rows, shape.hidden_size),
        mod_stride_b=modulation.stride(0),
        mod_stride_h=modulation.stride(-1),
        rows=buffers.rows,
        seq_len=buffers.seq_len,
        hidden=shape.hidden_size,
        eps=float(eps),
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


def launch_suffix_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    prefix_visible_mask: torch.Tensor,
    buffers: DM05BigKernelBuffers,
    *,
    scale: float,
) -> None:
    shape = buffers.shape
    batch_size = buffers.batch_size
    seq_len = buffers.seq_len
    prefix_len = int(cache_keys.shape[2])
    total_kv_len = prefix_len + int(key_states.shape[2])
    if tuple(prefix_visible_mask.shape) != (batch_size, prefix_len):
        raise ValueError(
            "prefix_visible_mask must have shape "
            f"{(batch_size, prefix_len)}, got {tuple(prefix_visible_mask.shape)}."
        )
    launch_config = _suffix_launch_config(buffers)
    qk_m, qk_n, qk_d, qk_warps, qk_stages = launch_config.qk
    softmax_m, base_softmax_k, softmax_warps = launch_config.softmax
    softmax_k = max(base_softmax_k, triton.next_power_of_2(total_kv_len))
    value_m, value_n, value_k, value_warps, value_stages = launch_config.attn_value
    _suffix_qk_logits_kernel[
        (
            batch_size
            * shape.q_heads
            * triton.cdiv(seq_len, qk_m)
            * triton.cdiv(total_kv_len, qk_n),
        )
    ](
        query_states,
        cache_keys,
        key_states,
        prefix_visible_mask,
        buffers.attn_logits,
        batch_size=batch_size,
        prefix_len=prefix_len,
        seq_len=seq_len,
        max_kv_len=buffers.max_kv_len,
        total_kv_len=total_kv_len,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        scale=float(scale),
        BLOCK_M=qk_m,
        BLOCK_N=qk_n,
        BLOCK_D=qk_d,
        num_warps=qk_warps,
        num_stages=qk_stages,
    )
    _suffix_softmax_kernel[
        (batch_size * shape.q_heads * triton.cdiv(seq_len, softmax_m),)
    ](
        buffers.attn_logits,
        buffers.attn_probs,
        batch_size=batch_size,
        seq_len=seq_len,
        max_kv_len=buffers.max_kv_len,
        total_kv_len=total_kv_len,
        q_heads=shape.q_heads,
        BLOCK_M=softmax_m,
        BLOCK_K=softmax_k,
        num_warps=softmax_warps,
    )
    _suffix_attn_value_kernel[
        (
            batch_size
            * shape.q_heads
            * triton.cdiv(seq_len, value_m)
            * triton.cdiv(shape.head_dim, value_n),
        )
    ](
        buffers.attn_probs,
        cache_values,
        value_states,
        buffers.attn_out,
        batch_size=batch_size,
        prefix_len=prefix_len,
        seq_len=seq_len,
        max_kv_len=buffers.max_kv_len,
        total_kv_len=total_kv_len,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        BLOCK_M=value_m,
        BLOCK_N=value_n,
        BLOCK_K=value_k,
        num_warps=value_warps,
        num_stages=value_stages,
    )


def launch_suffix_attn_postnorm_residual_prenorm(
    attn_embeds: torch.Tensor,
    suffix_embeds: torch.Tensor,
    attn_gate: torch.Tensor,
    post_norm_weight: torch.Tensor,
    mlp_modulation: torch.Tensor,
    buffers: DM05BigKernelBuffers,
    *,
    post_eps: float,
    pre_eps: float,
) -> None:
    shape = buffers.shape
    rows = buffers.rows
    hidden = shape.hidden_size
    norm_block_h, norm_warps = _L.norm
    _attn_postnorm_residual_prenorm_kernel[_row_grid(rows)](
        attn_embeds.contiguous().view(rows, hidden),
        suffix_embeds.contiguous().view(rows, hidden),
        attn_gate,
        post_norm_weight,
        mlp_modulation,
        buffers.residual.view(rows, hidden),
        buffers.mlp_input.view(rows, hidden),
        attn_gate_stride_b=attn_gate.stride(0),
        attn_gate_stride_h=attn_gate.stride(-1),
        mlp_mod_stride_b=mlp_modulation.stride(0),
        mlp_mod_stride_h=mlp_modulation.stride(-1),
        rows=rows,
        seq_len=buffers.seq_len,
        hidden=hidden,
        post_eps=float(post_eps),
        pre_eps=float(pre_eps),
        BLOCK_H=norm_block_h,
        num_warps=norm_warps,
    )


def launch_geglu_gate_up(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hidden_out: torch.Tensor,
    buffers: DM05BigKernelBuffers,
) -> None:
    shape = buffers.shape
    rows = buffers.rows
    hidden = shape.hidden_size
    mm_block_m, mm_block_n, mm_warps, mm_stages = _L.matmul
    _geglu_gate_up_kernel[_matmul_grid(rows, shape.intermediate_size)](
        x.contiguous().view(rows, hidden),
        gate_weight,
        up_weight,
        hidden_out,
        rows=rows,
        hidden=hidden,
        intermediate=shape.intermediate_size,
        BLOCK_M=mm_block_m,
        BLOCK_N=mm_block_n,
        BLOCK_K=_L.geglu_block_k,
        num_warps=mm_warps,
        num_stages=mm_stages,
    )


def launch_mlp_down(
    hidden_states: torch.Tensor,
    down_weight: torch.Tensor,
    out: torch.Tensor,
    buffers: DM05BigKernelBuffers,
) -> None:
    shape = buffers.shape
    rows = buffers.rows
    hidden = shape.hidden_size
    launch_config = _suffix_launch_config(buffers)
    mm_block_m, mm_block_n, mm_warps, mm_stages = launch_config.matmul
    _mlp_down_kernel[_matmul_grid(rows, hidden)](
        hidden_states,
        down_weight,
        out.view(rows, hidden),
        rows=rows,
        intermediate=shape.intermediate_size,
        hidden=hidden,
        BLOCK_M=mm_block_m,
        BLOCK_N=mm_block_n,
        BLOCK_K=launch_config.mlp_down_block_k,
        num_warps=mm_warps,
        num_stages=mm_stages,
    )


def launch_suffix_post_ffn_norm_residual(
    mlp_gate: torch.Tensor,
    post_ffn_norm_weight: torch.Tensor,
    buffers: DM05BigKernelBuffers,
    *,
    eps: float,
) -> torch.Tensor:
    shape = buffers.shape
    rows = buffers.rows
    hidden = shape.hidden_size
    norm_block_h, norm_warps = _L.norm
    _post_ffn_norm_residual_kernel[_row_grid(rows)](
        buffers.mlp_out.contiguous().view(rows, hidden),
        buffers.residual.contiguous().view(rows, hidden),
        mlp_gate,
        post_ffn_norm_weight,
        buffers.layer_out.view(rows, hidden),
        mlp_gate_stride_b=mlp_gate.stride(0),
        mlp_gate_stride_h=mlp_gate.stride(-1),
        rows=rows,
        seq_len=buffers.seq_len,
        hidden=hidden,
        eps=float(eps),
        BLOCK_H=norm_block_h,
        num_warps=norm_warps,
    )
    return buffers.layer_out


def launch_prefix_attn_postnorm_residual_prenorm(
    attn_embeds: torch.Tensor,
    residual: torch.Tensor,
    buffers: DM05PrefixDecoderBuffers,
    post_attention_weight: torch.Tensor,
    pre_feedforward_weight: torch.Tensor,
    *,
    post_eps: float,
    pre_eps: float,
) -> None:
    shape = buffers.shape
    rows = buffers.rows
    hidden = shape.hidden_size
    norm_block_h, norm_warps = _L.norm
    _prefix_attn_postnorm_residual_prenorm_kernel[_row_grid(rows)](
        attn_embeds.view(rows, hidden),
        residual.view(rows, hidden),
        post_attention_weight,
        pre_feedforward_weight,
        buffers.residual.view(rows, hidden),
        buffers.mlp_input.view(rows, hidden),
        rows=rows,
        hidden=hidden,
        post_eps=float(post_eps),
        pre_eps=float(pre_eps),
        BLOCK_H=norm_block_h,
        num_warps=norm_warps,
    )


def launch_prefix_geglu_from_gate_up(
    gate_up: torch.Tensor,
    hidden_out: torch.Tensor,
    buffers: DM05PrefixDecoderBuffers,
) -> None:
    block_m, block_n, mm_warps, mm_stages = _L.matmul
    _prefix_geglu_from_gate_up_kernel[
        _matmul_grid(buffers.rows, buffers.shape.intermediate_size)
    ](
        gate_up,
        hidden_out,
        rows=buffers.rows,
        intermediate=buffers.shape.intermediate_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=mm_warps,
        num_stages=mm_stages,
    )


def launch_prefix_post_ffn_norm_residual(
    buffers: DM05PrefixDecoderBuffers,
    post_feedforward_weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    shape = buffers.shape
    rows = buffers.rows
    hidden = shape.hidden_size
    norm_block_h, norm_warps = _L.norm
    _prefix_post_ffn_norm_residual_kernel[_row_grid(rows)](
        buffers.mlp_out.view(rows, hidden),
        buffers.residual.view(rows, hidden),
        post_feedforward_weight,
        buffers.layer_out.view(rows, hidden),
        rows=rows,
        hidden=hidden,
        eps=float(eps),
        BLOCK_H=norm_block_h,
        num_warps=norm_warps,
    )
    return buffers.layer_out
