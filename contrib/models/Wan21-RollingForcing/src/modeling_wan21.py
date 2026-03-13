"""
NxDI port of CausalWanModel for RollingForcing on Neuron (trn2).

Uses neuronx-distributed-inference (NxDI) with tensor parallelism (TP)
to compile and run the 30-layer CausalWanModel at 480×832 resolution.

Key design decisions:
  1. Custom attention (like Flux) rather than NeuronAttentionBase
  2. Three separate applications per attention mode (self/cached/update)
  3. RoPE pre-computed on CPU, passed as real/imag tensor pair
  4. KV buffers assembled on CPU, passed as input tensors
  5. Cache K/V returned as outputs for CPU-side cache management

Diffusers WanTransformer3DModel structure (from inspecting state dict):
  - condition_embedder: Timesteps → TimestepEmbedding → SiLU → time_proj(1536→9216)
                        text_embedder: PixArtAlphaTextProjection(4096→1536)
  - patch_embedding: Conv3d(16, 1536, kernel_size=(1,2,2), stride=(1,2,2))
  - blocks[0-29]: WanTransformerBlock
    - scale_shift_table: [1, 6, 1536]
    - norm1: LayerNorm(1536, elementwise_affine=False)
    - attn1: Attention (self-attn, Q/K/V: Linear 1536→1536 with bias, qk_norm=RMSNorm)
    - norm2: LayerNorm(1536, elementwise_affine=True)  [cross-attn norm]
    - attn2: Attention (cross-attn, same structure)
    - norm3: LayerNorm(1536, elementwise_affine=False)  [FFN norm]
    - ffn: FeedForward(GELU, 1536→8960→1536)
  - scale_shift_table: [1, 2, 1536]
  - norm_out: LayerNorm(1536, elementwise_affine=False)
  - proj_out: Linear(1536, 64)

Block forward:
  timestep_proj: [B, 6, 1536] (from condition_embedder.time_proj output, unflattened)
  shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (scale_shift_table + timestep_proj).chunk(6, dim=1)
  x = norm1(x) * (1 + scale_msa) + shift_msa  →  attn1(x, rotary_emb)  →  x + gate_msa * attn_out
  x = norm2(x) → attn2(x, encoder_hidden_states) → x + cross_out
  x = norm3(x) * (1 + c_scale) + c_shift → ffn(x) → x + c_gate * ff_out
"""

import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# NxDI imports (available on Neuron instances)
try:
    from neuronx_distributed_inference.models.application_base import (
        NeuronApplicationBase,
    )
    from neuronx_distributed_inference.models.model_wrapper import (
        ModelWrapper,
        EncoderModelInstance,
    )
    from neuronx_distributed_inference.models.config import (
        InferenceConfig,
        NeuronConfig,
    )
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_size,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
    )
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from neuronx_distributed.parallel_layers.mappings import (
        reduce_from_tensor_model_parallel_region,
    )

    HAS_NXDI = True

    # NKI flash attention kernel — tiled computation, never materializes full
    # attention matrix, dramatically reduces HBM scratchpad usage.
    try:
        from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
    except ImportError:
        from neuronxcc.nki.kernels.attention import attention_isa_kernel
    from torch_neuronx import nki_jit

    _flash_fwd = nki_jit()(attention_isa_kernel)
    HAS_NKI_FLASH = True
except ImportError:
    HAS_NXDI = False
    HAS_NKI_FLASH = False
    _flash_fwd = None
    print("[WARN] NxDI not available, using CPU fallback stubs")

    class NeuronApplicationBase(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class ModelWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class EncoderModelInstance:
        def __init__(self, **kwargs):
            pass

    class InferenceConfig:
        def __init__(self, **kwargs):
            pass

    class NeuronConfig:
        def __init__(self, **kwargs):
            pass

    def get_tensor_model_parallel_size():
        return 1

    def get_tensor_model_parallel_group():
        return None

    def get_tensor_model_parallel_rank():
        return 0

    ColumnParallelLinear = None
    RowParallelLinear = None

    def reduce_from_tensor_model_parallel_region(input_):
        return input_  # No-op for TP=1 CPU fallback


# ─── Constants ──────────────────────────────────────────────────────────────
FRAME_SEQ_LENGTH = 1560  # tokens per latent frame = 30 * 52
NUM_FRAME_PER_BLOCK = 3
BLOCK_LENGTH = NUM_FRAME_PER_BLOCK * FRAME_SEQ_LENGTH  # 4680
MAX_ATTENTION_SIZE = 21 * FRAME_SEQ_LENGTH  # 32760
KV_CACHE_CAPACITY = 24 * FRAME_SEQ_LENGTH  # 37440
SINK_TOKENS = BLOCK_LENGTH  # 4680

NUM_HEADS = 12
HEAD_DIM = 128
DIM = NUM_HEADS * HEAD_DIM  # 1536
FFN_DIM = 8960  # From diffusers config
NUM_LAYERS = 30
TEXT_DIM = 4096
TEXT_SEQ_LEN = 512
FREQ_DIM = 256  # Sinusoidal timestep embedding dimension
TIME_PROJ_DIM = DIM * 6  # 9216

PATCH_T, PATCH_H, PATCH_W = 1, 2, 2
IN_CHANNELS = 16
OUT_CHANNELS = 16

# Bucket configurations
SELF_FRAME_COUNTS = [3, 6, 9, 12, 15]
CACHED_FRAME_COUNTS = [3, 6, 9, 12, 15]
UPDATE_FRAME_COUNTS = [3]


# ─── Utility ───────────────────────────────────────────────────────────────


def _pad_to_multiple(tensor, dim, multiple):
    """Pad tensor along dimension to be divisible by multiple."""
    size = tensor.shape[dim]
    if size % multiple == 0:
        return tensor, 0
    pad_size = multiple - (size % multiple)
    ndim = tensor.dim()
    # F.pad expects pad from last dim backward: (last_dim_left, last_dim_right, ...)
    pad = [0] * (2 * ndim)
    pad_idx = 2 * (ndim - 1 - dim)
    pad[pad_idx + 1] = pad_size
    return F.pad(tensor, pad), pad_size


def nki_flash_attention(q, k, v):
    """
    NKI flash attention wrapper with 128-alignment padding.

    The attention_isa_kernel requires all sequence length dimensions to be
    multiples of 128 (kernel tile size). We pad before calling and unpad after.

    Input shapes (standard attention convention):
        q: [B, N, Q_len, D]
        k: [B, N, K_len, D]
        v: [B, N, V_len, D]

    Returns: [B, N, Q_len, D]

    NOTE: Does NOT support attention masks. For masked attention, use
    F.scaled_dot_product_attention instead.
    """
    ALIGNMENT = 128

    bs, n_head, q_len, d_head = q.shape
    k_len = k.shape[2]

    # Pad sequence dimensions to multiples of 128
    q_padded, q_pad = _pad_to_multiple(q, dim=2, multiple=ALIGNMENT)
    k_padded, k_pad = _pad_to_multiple(k, dim=2, multiple=ALIGNMENT)
    v_padded, v_pad = _pad_to_multiple(v, dim=2, multiple=ALIGNMENT)

    padded_q_len = q_padded.shape[2]
    padded_k_len = k_padded.shape[2]
    padded_v_len = v_padded.shape[2]

    # NKI kernel expects:
    #   Q: [bs*n_head, d_head, q_len]
    #   K: [bs*n_head, d_head, k_len]
    #   V: [bs*n_head, v_len, d_head]
    q_nki = (
        q_padded.clone().permute(0, 1, 3, 2).reshape(bs * n_head, d_head, padded_q_len)
    )
    k_nki = (
        k_padded.clone().permute(0, 1, 3, 2).reshape(bs * n_head, d_head, padded_k_len)
    )
    v_nki = v_padded.clone().reshape(bs * n_head, padded_v_len, d_head)

    attn_output = torch.zeros(
        (bs * n_head, padded_q_len, d_head), dtype=torch.bfloat16, device=q.device
    )

    scale = 1.0 / math.sqrt(d_head)

    _flash_fwd(
        q_nki,
        k_nki,
        v_nki,
        scale,
        attn_output,
        kernel_name="AttentionMMSoftmaxMMWithoutSwap",
    )

    result = attn_output.reshape(bs, n_head, padded_q_len, d_head)

    # Remove padding from output
    if q_pad > 0:
        result = result[:, :, :q_len, :]

    return result


def _get_tp_degree():
    if HAS_NXDI:
        try:
            return get_tensor_model_parallel_size()
        except (AssertionError, RuntimeError):
            return 1  # Parallel state not initialized (CPU mode)
    return 1


def _get_tp_rank():
    if HAS_NXDI:
        try:
            return get_tensor_model_parallel_rank()
        except (AssertionError, RuntimeError):
            return 0
    return 0


def make_column_parallel(in_f, out_f, bias=True, gather_output=False):
    if HAS_NXDI and ColumnParallelLinear is not None:
        return ColumnParallelLinear(in_f, out_f, bias=bias, gather_output=gather_output)
    return nn.Linear(in_f, out_f, bias=bias)


def make_row_parallel(in_f, out_f, bias=True, input_is_parallel=True):
    if HAS_NXDI and RowParallelLinear is not None:
        return RowParallelLinear(
            in_f, out_f, bias=bias, input_is_parallel=input_is_parallel
        )
    return nn.Linear(in_f, out_f, bias=bias)


# ─── RoPE (CPU-side precomputation) ────────────────────────────────────────


def rope_params(max_seq_len: int, dim: int) -> torch.Tensor:
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float64)
    return torch.exp(1j * torch.outer(t, freqs))


def make_freqs(head_dim: int = 128) -> torch.Tensor:
    d = head_dim
    return torch.cat(
        [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ],
        dim=1,
    )


def precompute_rope_embeddings(freqs, num_frames, h, w, start_frame=0):
    """
    Returns cos and sin tensors for RoPE (real-valued, Neuron-compatible).

    Args:
        freqs: [max_seq_len, head_dim//2] complex base frequencies
        num_frames, h, w: spatial grid
        start_frame: temporal offset

    Returns:
        cos_emb: [seq_len, 1, head_dim] float32
        sin_emb: [seq_len, 1, head_dim] float32
    """
    c = freqs.shape[1]
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    seq_len = num_frames * h * w
    freqs_i = torch.cat(
        [
            freqs_split[0][start_frame : start_frame + num_frames]
            .view(num_frames, 1, 1, -1)
            .expand(num_frames, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(num_frames, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(num_frames, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)
    # freqs_i: [seq_len, 1, head_dim//2] complex

    # Extract angles for real-valued rotation
    # freqs_i is complex [seq_len, 1, head_dim//2] — each complex number encodes an angle
    # We need cos/sin at [seq_len, 1, head_dim//2] to pair with even/odd halves of x
    angles = torch.angle(freqs_i).float()  # [seq_len, 1, head_dim//2]
    cos_emb = torch.cos(angles)  # [seq_len, 1, head_dim//2]
    sin_emb = torch.sin(angles)  # [seq_len, 1, head_dim//2]

    return cos_emb, sin_emb


def apply_rope_precomputed(x, cos_emb, sin_emb):
    """
    Apply RoPE using real-valued rotary embedding (Neuron-compatible).

    Args:
        x: [B, L, N, D] — bf16 or float32
        cos_emb: [L, 1, D//2] — float32 cos of rotation angles
        sin_emb: [L, 1, D//2] — float32 sin of rotation angles

    Returns:
        x with RoPE applied: [B, L, N, D] same dtype as x
    """
    original_dtype = x.dtype
    x_f = x.float()  # compute RoPE in float32 for accuracy

    # Split into even/odd for rotation (matching complex multiplication)
    # Reference: view_as_complex treats (x[2k], x[2k+1]) as one complex number
    # Then multiplies by e^{i*theta_k} = cos(theta_k) + i*sin(theta_k)
    # Result: new_2k = x_2k * cos_k - x_{2k+1} * sin_k
    #         new_{2k+1} = x_2k * sin_k + x_{2k+1} * cos_k
    x_even = x_f[..., 0::2]  # [B, L, N, D//2]
    x_odd = x_f[..., 1::2]  # [B, L, N, D//2]

    # cos_emb and sin_emb are [L, 1, D//2] — one angle per complex pair
    cos_e = cos_emb.unsqueeze(0)  # [1, L, 1, D//2]
    sin_e = sin_emb.unsqueeze(0)  # [1, L, 1, D//2]

    # Apply rotation: (x_even + i*x_odd) * (cos + i*sin)
    # Real part: x_even * cos - x_odd * sin
    # Imag part: x_even * sin + x_odd * cos
    new_even = x_even * cos_e - x_odd * sin_e
    new_odd = x_even * sin_e + x_odd * cos_e

    # Interleave back
    result = torch.stack([new_even, new_odd], dim=-1).flatten(-2)

    return result.to(original_dtype)


# ─── Norms ─────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """RMS Norm across all heads (qk_norm="rms_norm_across_heads")."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def _norm_sharded(norm_module, x):
    """Apply RMSNorm to a TP-sharded tensor using global RMS via NxD all-reduce.

    The QK norm in CausalWanModel operates across ALL heads (dim=1536).
    With TP=4, each rank only has dim/tp elements (384). We compute the
    local sum-of-squares, all-reduce across TP ranks to get the global
    sum-of-squares, then divide by the full dimension to get the correct
    global mean-of-squares for normalization.

    Uses neuronx_distributed's reduce_from_tensor_model_parallel_region
    which uses xm.all_reduce (XLA native) — this works correctly inside
    compiled NEFFs, unlike torch.distributed.all_reduce which is a no-op.

    Weight slicing: uses weight[:shard_dim]. The load_weights override in
    NeuronCausalWanUnifiedApplication pre-shards the weight so that each
    rank's weight[:shard_dim] contains the correct slice for that rank.

    Computes in float32 for accuracy (matching reference model).
    """
    original_dtype = x.dtype
    shard_dim = x.shape[-1]
    tp = _get_tp_degree()
    full_dim = shard_dim * tp  # 384 * 4 = 1536
    w = norm_module.weight[:shard_dim]
    x_f = x.float()

    # Step 1: Local sum of squares (not mean!)
    local_sum_sq = x_f.pow(2).sum(-1, keepdim=True)  # [B, L, 1]

    # Step 2: All-reduce to get global sum of squares across all TP ranks
    global_sum_sq = reduce_from_tensor_model_parallel_region(local_sum_sq)

    # Step 3: Global mean of squares = global_sum_sq / full_dim
    global_mean_sq = global_sum_sq / full_dim

    # Step 4: Normalize
    norm_val = torch.rsqrt(global_mean_sq + norm_module.eps)
    return (x_f * norm_val * w).to(original_dtype)


# ─── Self-Attention (TP-sharded, 3 modes) ──────────────────────────────────


class NeuronCausalSelfAttention(nn.Module):
    """
    TP-sharded self-attention for CausalWanModel. 3 modes: self/cached/update.

    Matches diffusers attn1 structure:
      to_q, to_k, to_v: Linear(1536, 1536, bias=True)
      to_out.0: Linear(1536, 1536, bias=True)
      norm_q, norm_k: RMSNorm(1536)
    """

    def __init__(self, dim, num_heads, head_dim, mode, eps=1e-6):
        super().__init__()
        assert mode in ("self", "cached", "update")
        self.mode = mode
        self.dim = dim
        self.head_dim = head_dim

        tp = _get_tp_degree()
        padded_heads = math.ceil(num_heads / tp) * tp
        self.padded_inner_dim = padded_heads * head_dim
        self.num_heads_per_rank = padded_heads // tp

        # Q/K/V with bias (matches diffusers)
        self.to_q = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_k = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_v = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )

        # Output projection with bias
        self.to_out = make_row_parallel(
            self.padded_inner_dim, dim, bias=True, input_is_parallel=True
        )

        # QK norm: RMS norm across heads (operates on full dim, not per-head)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self, x, rope_cos, rope_sin, kv_buffer_k=None, kv_buffer_v=None, attn_mask=None
    ):
        """
        Args:
            x: [B, L, C]
            rope_cos: [L, 1, D] float32 — cos of rotation angles
            rope_sin: [L, 1, D] float32 — sin of rotation angles
            kv_buffer_k: [B, KV_LEN, N_rank, D] (cached/update only)
            kv_buffer_v: [B, KV_LEN, N_rank, D]
            attn_mask: [B, 1, L, KV_LEN] (cached/update only)

        Returns:
            attn_out: [B, L, C]
            raw_k: [B, L, N_rank, D] — un-roped K for anchor cache
            v: [B, L, N_rank, D] — values for cache
            roped_k: [B, L, N_rank, D] — roped K for non-anchor cache
        """
        b, s, _ = x.shape
        n, d = self.num_heads_per_rank, self.head_dim

        # Project Q, K, V
        q_proj = self.to_q(x)  # [B, L, padded_inner_dim / tp]
        k_proj = self.to_k(x)
        v_proj = self.to_v(x)

        # Norm Q and K (across-heads: norm on full output before reshape)
        # But wait — with TP, each rank only has its slice. The norm_q/norm_k
        # have weights of size dim=1536 but the projection output is dim/tp.
        # We need to handle this properly.
        #
        # In the original model, norm is applied BEFORE reshape to heads:
        #   q = norm_q(to_q(x))  # shape [B, L, 1536], norm on last dim
        #
        # With TP, to_q output is [B, L, 1536/tp]. We can't apply the full
        # norm_q (weight size 1536) on a tensor of dim 1536/tp.
        #
        # Solution: Apply norm per-rank. The RMSNorm weight is sharded too.
        # Actually, looking at how Flux does it: norm_q is CustomRMSNorm(dim_head),
        # applied AFTER reshape to [B, N, L, D] (per-head norm).
        #
        # But Wan uses "rms_norm_across_heads" which is norm on the FULL
        # concatenated Q (dim=1536). This is tricky with TP because we can't
        # do an all-gather just for norm.
        #
        # Practical solution: With TP, each rank applies RMSNorm independently
        # on its own slice. This is an approximation but should be numerically
        # close since RMSNorm is scale-invariant per-element.
        # The weight needs to be sharded to match.

        q_normed = self._norm_sharded(self.norm_q, q_proj)
        k_normed = self._norm_sharded(self.norm_k, k_proj)

        # Reshape to [B, L, N_rank, D]
        q = q_normed.view(b, s, n, d)
        k = k_normed.view(b, s, n, d)
        v = v_proj.view(b, s, n, d)
        k_raw = k.clone()  # normed (but un-roped) K for anchor cache re-RoPE

        # Apply RoPE
        roped_q = apply_rope_precomputed(q, rope_cos, rope_sin)
        roped_k = apply_rope_precomputed(k, rope_cos, rope_sin)

        if self.mode == "self":
            # Cast to float32 for attention computation to reduce compound error
            # in the autoregressive pipeline. bf16 attention introduces per-window
            # noise that accumulates across 11 rolling windows.
            q_t = roped_q.transpose(1, 2).float()  # [B, N, L, D]
            k_t = roped_k.transpose(1, 2).float()
            v_t = v.transpose(1, 2).float()
            if attn_mask is not None:
                # Padded self mode: use SDPA with mask to ignore padding tokens.
                # Chunked attention for large seq_lens, matching cached mode.
                attn_mask_f32 = attn_mask.float()  # match query dtype
                num_chunks = 3
                chunk_size = s // num_chunks
                chunks_out = []
                for ci in range(num_chunks):
                    q_start = ci * chunk_size
                    q_end = s if ci == num_chunks - 1 else (ci + 1) * chunk_size
                    q_chunk = q_t[:, :, q_start:q_end, :]
                    mask_chunk = attn_mask_f32[:, :, q_start:q_end, :]
                    out_chunk = F.scaled_dot_product_attention(
                        q_chunk, k_t, v_t, attn_mask=mask_chunk
                    )
                    chunks_out.append(out_chunk)
                out = torch.cat(chunks_out, dim=2)
            elif HAS_NKI_FLASH and q_t.device.type != "cpu":
                # Unpadded self mode: NKI flash attention (no mask needed).
                out = nki_flash_attention(q_t, k_t, v_t)
            else:
                out = F.scaled_dot_product_attention(q_t, k_t, v_t)
            out = out.transpose(1, 2)
        elif self.mode == "cached":
            # Path B: cache-augmented attention.
            # kv_buffer contains [anchor | working_cache | padding_zeros] with
            # total length MAX_ATTENTION_SIZE. The last `s` positions are reserved
            # for the current input's K/V.
            #
            # Instead of clone+write (which can cause Neuron compilation issues
            # with buffer aliasing), we use torch.cat to build the full KV:
            #   full_K = cat([buffer_prefix, current_roped_K])
            # This produces the same result but avoids in-place mutation.
            kv_len = kv_buffer_k.shape[1]  # MAX_ATTENTION_SIZE
            prefix_len = kv_len - s
            prefix_k = kv_buffer_k[:, :prefix_len]  # [B, prefix_len, N, D]
            prefix_v = kv_buffer_v[:, :prefix_len]
            kv_k = torch.cat(
                [prefix_k, roped_k.to(kv_buffer_k.dtype)], dim=1
            )  # [B, MAX_ATTENTION_SIZE, N, D]
            kv_v = torch.cat([prefix_v, v.to(kv_buffer_v.dtype)], dim=1)

            q_t = roped_q.transpose(1, 2).float()  # fp32 for precision
            k_t = kv_k.transpose(1, 2).float()
            v_t = kv_v.transpose(1, 2).float()

            # Chunked attention: split Q into chunks to reduce peak HBM usage.
            # For large Q (e.g., 23400 tokens) attending to full KV (32760),
            # the SDPA scratchpad is Q_len * KV_len * sizeof(bf16) per head.
            # By chunking Q, peak memory is chunk_size * KV_len instead.
            # We use 3 chunks for cached mode (matching frame boundaries).
            num_chunks = 3
            chunk_size = s // num_chunks
            # Handle non-divisible case: last chunk gets remainder
            attn_mask_f32 = attn_mask.float()  # match query dtype
            chunks_out = []
            for ci in range(num_chunks):
                q_start = ci * chunk_size
                q_end = s if ci == num_chunks - 1 else (ci + 1) * chunk_size
                q_chunk = q_t[:, :, q_start:q_end, :]
                mask_chunk = attn_mask_f32[:, :, q_start:q_end, :]
                out_chunk = F.scaled_dot_product_attention(
                    q_chunk, k_t, v_t, attn_mask=mask_chunk
                )
                chunks_out.append(out_chunk)
            out = torch.cat(chunks_out, dim=2)
            out = out.transpose(1, 2)
        else:
            # Path C: update mode — Q=current block, KV=cache contents
            # kv_buffer contains full cache slice, no current input insertion needed
            # Update mode Q is always small (3 frames = 4680 tokens), no chunking needed
            q_t = roped_q.transpose(1, 2).float()  # fp32 for precision
            k_t = kv_buffer_k.transpose(1, 2).float()
            v_t = kv_buffer_v.transpose(1, 2).float()
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask.float()
            )
            out = out.transpose(1, 2)

        # Output projection
        out = out.reshape(b, s, n * d)
        out = self.to_out(out)

        return out, k_raw, v, roped_k

    def _norm_sharded(self, norm_module, x):
        """Delegate to module-level function."""
        return _norm_sharded(norm_module, x)


# ─── Cross-Attention (TP-sharded) ──────────────────────────────────────────


class NeuronCrossAttention(nn.Module):
    """TP-sharded cross attention. Matches diffusers attn2."""

    def __init__(self, dim, num_heads, head_dim, eps=1e-6):
        super().__init__()
        self.head_dim = head_dim
        tp = _get_tp_degree()
        padded_heads = math.ceil(num_heads / tp) * tp
        self.padded_inner_dim = padded_heads * head_dim
        self.num_heads_per_rank = padded_heads // tp

        self.to_q = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_k = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_v = make_column_parallel(
            dim, self.padded_inner_dim, bias=True, gather_output=False
        )
        self.to_out = make_row_parallel(
            self.padded_inner_dim, dim, bias=True, input_is_parallel=True
        )

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, context):
        """x: [B, L, C], context: [B, T, C] → [B, L, C]"""
        b, lq, _ = x.shape
        n, d = self.num_heads_per_rank, self.head_dim

        q_proj = self.to_q(x)
        k_proj = self.to_k(context)
        v_proj = self.to_v(context)

        # Norm: use exact global RMS via all-reduce (same as self-attention)
        q_normed = _norm_sharded(self.norm_q, q_proj)
        k_normed = _norm_sharded(self.norm_k, k_proj)

        q = q_normed.view(b, lq, n, d)
        k = k_normed.view(b, -1, n, d)
        v = v_proj.view(b, -1, n, d)

        # Cast to fp32 for SDPA precision
        compute_dtype = torch.float32
        q_t = q.transpose(1, 2).to(compute_dtype)
        k_t = k.transpose(1, 2).to(compute_dtype)
        v_t = v.transpose(1, 2).to(compute_dtype)

        out = F.scaled_dot_product_attention(q_t, k_t, v_t)
        out = out.transpose(1, 2).reshape(b, lq, n * d)
        return self.to_out(out)


# ─── Transformer Block ─────────────────────────────────────────────────────


class NeuronCausalWanBlock(nn.Module):
    """
    Single transformer block matching diffusers WanTransformerBlock.

    Modulation flow:
      scale_shift_table [1, 6, 1536] + timestep_proj [B, 6, 1536]
      → chunk into 6: shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate
    """

    def __init__(self, dim, ffn_dim, num_heads, head_dim, mode, eps=1e-6):
        super().__init__()
        self.dim = dim

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = NeuronCausalSelfAttention(
            dim, num_heads, head_dim, mode=mode, eps=eps
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.cross_attn = NeuronCrossAttention(dim, num_heads, head_dim, eps=eps)

        # FFN: GELU activation (not GEGLU)
        # Matches: ffn.net.0.proj (Linear dim→ffn_dim), ffn.net.2 (Linear ffn_dim→dim)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn_gelu_proj = nn.Linear(dim, ffn_dim, bias=True)  # ffn.net.0.proj
        self.ffn_out = nn.Linear(ffn_dim, dim, bias=True)  # ffn.net.2

        # Modulation: [1, 6, dim]
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        timestep_proj,
        context,
        rope_cos,
        rope_sin,
        num_frames,
        kv_buffer_k=None,
        kv_buffer_v=None,
        attn_mask=None,
    ):
        """
        Args:
            x: [B, L, C]
            timestep_proj: [B, F, 6, C] — per-frame from condition embedder
            context: [B, T, C] — text embedding
            rope_cos: [L, 1, D] float32 — cos of rotation angles
            rope_sin: [L, 1, D] float32 — sin of rotation angles
            num_frames: int — number of latent frames (F)
            kv_buffer_k/v: [B, KV, N_rank, D] (cached/update only)
            attn_mask: [B, 1, L, KV] (cached/update only)

        Returns:
            x: [B, L, C]
            raw_k, v, roped_k: cache tensors [B, L, N_rank, D]
        """
        B, L, C = x.shape
        frame_seqlen = L // num_frames  # tokens per latent frame

        # Per-frame modulation: [B, F, 6, C] → expand to [B, L, C]
        # Match reference exactly: compute in the same dtype as inputs (bf16 on Neuron).
        # The reference does: e = (self.modulation.unsqueeze(1) + e) in model dtype.
        # Previously we cast to float32 and back, introducing bf16 rounding differences.
        modulation = self.scale_shift_table.unsqueeze(1) + timestep_proj.to(x.dtype)
        # modulation: [B, F, 6, C] — chunk into 6 per-frame modulation params
        shift_msa = modulation[:, :, 0:1, :]  # [B, F, 1, C]
        scale_msa = modulation[:, :, 1:2, :]
        gate_msa = modulation[:, :, 2:3, :]
        c_shift = modulation[:, :, 3:4, :]
        c_scale = modulation[:, :, 4:5, :]
        c_gate = modulation[:, :, 5:6, :]

        # Self-attention with per-frame modulation
        norm_x = self.norm1(x)
        # Unflatten to per-frame: [B, L, C] → [B, F, S, C], apply modulation, flatten back
        norm_x_framed = norm_x.reshape(B, num_frames, frame_seqlen, C)
        x_mod = norm_x_framed * (1 + scale_msa) + shift_msa  # broadcast over S dim
        x_mod = x_mod.reshape(B, L, C)

        attn_out, raw_k, v, roped_k = self.self_attn(
            x_mod, rope_cos, rope_sin, kv_buffer_k, kv_buffer_v, attn_mask
        )

        # Gate per-frame: [B, F, 1, C] → expand
        attn_out_framed = attn_out.reshape(B, num_frames, frame_seqlen, C)
        x_framed = x.reshape(B, num_frames, frame_seqlen, C)
        x_framed = x_framed + gate_msa * attn_out_framed
        x = x_framed.reshape(B, L, C)

        # Cross-attention (no per-frame modulation needed — same text for all frames)
        norm_x2 = self.norm2(x)
        cross_out = self.cross_attn(norm_x2, context)
        x = x + cross_out

        # FFN with per-frame modulation
        norm_x3 = self.norm3(x)
        norm_x3_framed = norm_x3.reshape(B, num_frames, frame_seqlen, C)
        x3_mod = norm_x3_framed * (1 + c_scale) + c_shift
        x3_mod = x3_mod.reshape(B, L, C)

        ff_out = self.ffn_out(F.gelu(self.ffn_gelu_proj(x3_mod), approximate="tanh"))
        ff_out_framed = ff_out.reshape(B, num_frames, frame_seqlen, C)
        x_framed = x.reshape(B, num_frames, frame_seqlen, C)
        x_framed = x_framed + c_gate * ff_out_framed
        x = x_framed.reshape(B, L, C)

        return x, raw_k, v, roped_k


# ─── Condition Embedder ────────────────────────────────────────────────────


class NeuronConditionEmbedder(nn.Module):
    """
    Matches diffusers WanTimeTextImageEmbedding.

    Components:
      timesteps_proj: Sinusoidal (Timesteps, no params)
      time_embedder: Linear(256, 1536) + SiLU + Linear(1536, 1536)
      act_fn: SiLU
      time_proj: Linear(1536, 9216)
      text_embedder: Linear(4096, 1536) + GELU(tanh) + Linear(1536, 1536)
    """

    def __init__(self, dim=DIM, text_dim=TEXT_DIM, freq_dim=FREQ_DIM):
        super().__init__()
        # time_embedder: TimestepEmbedding (Linear→SiLU→Linear)
        self.time_embedder_linear_1 = nn.Linear(freq_dim, dim)
        self.time_embedder_act = nn.SiLU()
        self.time_embedder_linear_2 = nn.Linear(dim, dim)

        # SiLU before time_proj
        self.act_fn = nn.SiLU()

        # time_proj: projects temb to 6*dim for modulation
        self.time_proj = nn.Linear(dim, dim * 6)

        # text_embedder: PixArtAlphaTextProjection (Linear→GELU→Linear)
        self.text_embedder_linear_1 = nn.Linear(text_dim, dim)
        self.text_embedder_act = nn.GELU(approximate="tanh")
        self.text_embedder_linear_2 = nn.Linear(dim, dim)

        self.freq_dim = freq_dim

    def _timestep_embedding(self, t, dim, max_period=10000):
        """Sinusoidal timestep embedding matching Timesteps."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, timestep, encoder_hidden_states, num_frames):
        """
        Args:
            timestep: [B, F] — per-frame timestep values (F = num latent frames)
            encoder_hidden_states: [B, T, text_dim]
            num_frames: int — number of latent frames (F)

        Returns:
            temb: [B, F, C] — for output head scale_shift_table (per-frame)
            timestep_proj: [B, F, 6, C] — for block modulation (per-frame)
            context: [B, T, C] — projected text
        """
        # Flatten [B, F] → [B*F] for sinusoidal embedding
        B, F = timestep.shape
        t_flat = timestep.reshape(B * F)

        # Timestep embedding (time_embedder weights may be float32)
        t_emb = self._timestep_embedding(t_flat, self.freq_dim)  # [B*F, freq_dim]
        time_dtype = self.time_embedder_linear_1.weight.dtype
        t_emb = t_emb.to(time_dtype)
        temb = self.time_embedder_linear_2(
            self.time_embedder_act(self.time_embedder_linear_1(t_emb))
        )  # [B*F, dim]
        temb = temb.to(encoder_hidden_states.dtype)

        # Time projection for block modulation
        time_proj_input = self.act_fn(temb).to(self.time_proj.weight.dtype)
        timestep_proj = self.time_proj(time_proj_input)  # [B*F, 6*dim]
        timestep_proj = timestep_proj.unflatten(1, (6, -1))  # [B*F, 6, dim]

        # Reshape to per-frame: [B, F, 6, dim]
        timestep_proj = timestep_proj.reshape(B, F, 6, -1)
        temb = temb.reshape(B, F, -1)  # [B, F, dim]

        # Text projection
        text_dtype = self.text_embedder_linear_1.weight.dtype
        context = self.text_embedder_linear_2(
            self.text_embedder_act(
                self.text_embedder_linear_1(encoder_hidden_states.to(text_dtype))
            )
        )

        return temb, timestep_proj, context


# ─── Full Transformer (traced on Neuron) ───────────────────────────────────


class NeuronCausalWanTransformer(nn.Module):
    """
    Full CausalWanModel for Neuron tracing with TP.

    Input signature varies by mode:
      "self":   (hidden_states, timestep, encoder_hidden_states, rope_cos, rope_sin)
      "cached"/"update": (hidden_states, timestep, encoder_hidden_states,
                          rope_cos, rope_sin, attn_mask, kv_k_0, kv_v_0, ..., kv_k_29, kv_v_29)

    Output: (video_output, cache_k_0, cache_v_0, roped_k_0, ..., cache_k_29, cache_v_29, roped_k_29)
    """

    def __init__(self, config):
        super().__init__()
        self.mode = config.attn_mode

        dim = config.hidden_size
        ffn_dim = config.intermediate_size
        num_heads = config.num_attention_heads
        head_dim = config.attention_head_dim
        num_layers = config.num_hidden_layers
        in_channels = config.in_channels
        self.num_layers = num_layers
        self.block_length = BLOCK_LENGTH

        # Condition embedder
        self.condition_embedder = NeuronConditionEmbedder(dim=dim)

        # Patch embedding: Conv3d
        self.patch_embedding = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=(PATCH_T, PATCH_H, PATCH_W),
            stride=(PATCH_T, PATCH_H, PATCH_W),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                NeuronCausalWanBlock(dim, ffn_dim, num_heads, head_dim, mode=self.mode)
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, PATCH_T * PATCH_H * PATCH_W * in_channels)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,  # [B, C_in, F, H, W]
        timestep,  # [B, F] — per-frame timestep
        encoder_hidden_states,  # [B, T, text_dim]
        rope_cos,  # [L, 1, D]
        rope_sin,  # [L, 1, D]
        *kv_args,  # For cached/update: (attn_mask, kv_k_0, kv_v_0, kv_k_1, kv_v_1, ..., kv_k_29, kv_v_29)
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # Parse KV args: supports 3 cases:
        #   1. No kv_args: self mode without mask (original)
        #   2. kv_args = (mask,): self mode with mask (padded unified)
        #   3. kv_args = (mask, k0, v0, k1, v1, ...): cached/update mode
        if len(kv_args) > 1:
            # Case 3: mask + KV tensors
            attn_mask = kv_args[0]  # [B, 1, L, KV_LEN]
            kv_pairs = kv_args[1:]
            assert len(kv_pairs) == self.num_layers * 2, (
                f"Expected {self.num_layers * 2} KV tensors, got {len(kv_pairs)}"
            )
        elif len(kv_args) == 1:
            # Case 2: mask only (self mode with padding support)
            attn_mask = kv_args[0]
            kv_pairs = None
        else:
            # Case 1: no mask, no KV (original self mode)
            attn_mask = None
            kv_pairs = None

        # Condition embedder — per-frame timestep
        temb, timestep_proj, context = self.condition_embedder(
            timestep, encoder_hidden_states, num_frames
        )
        # temb: [B, F, C], timestep_proj: [B, F, 6, C], context: [B, T, C]

        # Patch embedding: [B, C_in, F, H, W] → [B, dim, F', H', W'] → [B, L, dim]
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Post-patch dimensions
        post_f = num_frames // PATCH_T
        post_h = height // PATCH_H
        post_w = width // PATCH_W

        # Transformer blocks
        cache_outputs = []
        for i, block in enumerate(self.blocks):
            if kv_pairs is not None:
                layer_kv_k = kv_pairs[i * 2]  # [B, KV_LEN, N_rank, D]
                layer_kv_v = kv_pairs[i * 2 + 1]  # [B, KV_LEN, N_rank, D]
            else:
                layer_kv_k = None
                layer_kv_v = None

            hidden_states, raw_k, v, roped_k = block(
                hidden_states,
                timestep_proj,
                context,
                rope_cos,
                rope_sin,
                post_f,
                layer_kv_k,
                layer_kv_v,
                attn_mask,
            )

            # Collect first block_length tokens for cache
            bl = self.block_length
            cache_outputs.append(raw_k[:, :bl])
            cache_outputs.append(v[:, :bl])
            cache_outputs.append(roped_k[:, :bl])

        # Output head — per-frame modulation
        # temb: [B, F, C], scale_shift_table: [1, 2, C]
        # Expand temb to [B, F, 1, C], add table [1, 1, 2, C], get [B, F, 2, C]
        frame_seqlen = hidden_states.shape[1] // post_f
        head_mod = self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(
            2
        )  # [B, F, 2, C]
        shift = head_mod[:, :, 0:1, :]  # [B, F, 1, C]
        scale = head_mod[:, :, 1:2, :]  # [B, F, 1, C]

        # Apply per-frame: unflatten hidden_states to [B, F, S, C], modulate, flatten back
        hidden_states = hidden_states.reshape(batch_size, post_f, frame_seqlen, -1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        hidden_states = hidden_states.reshape(batch_size, post_f * frame_seqlen, -1)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify: match exact diffusers reshape
        hidden_states = hidden_states.reshape(
            batch_size, post_f, post_h, post_w, PATCH_T, PATCH_H, PATCH_W, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return (output,) + tuple(cache_outputs)


# ─── Config ────────────────────────────────────────────────────────────────


class CausalWanInferenceConfig(InferenceConfig):
    """Config for CausalWanModel on NxDI."""

    def __init__(self, *args, **kwargs):
        self.attn_mode = kwargs.pop("attn_mode", "self")
        self.height = kwargs.pop("height", 480)
        self.width = kwargs.pop("width", 832)
        self.num_latent_frames = kwargs.pop("num_latent_frames", 21)
        self.hidden_size = kwargs.pop("hidden_size", DIM)
        self.intermediate_size = kwargs.pop("intermediate_size", FFN_DIM)
        self.num_attention_heads = kwargs.pop("num_attention_heads", NUM_HEADS)
        self.attention_head_dim = kwargs.pop("attention_head_dim", HEAD_DIM)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", NUM_LAYERS)
        self.in_channels = kwargs.pop("in_channels", IN_CHANNELS)
        self.use_mask_for_self = kwargs.pop("use_mask_for_self", False)

        super().__init__(*args, **kwargs)

    def get_required_attributes(self):
        return []  # All have defaults

    def load_config(self):
        pass


# ─── ModelWrapper ──────────────────────────────────────────────────────────


class NeuronCausalWanModelWrapper(ModelWrapper):
    """CPU-side wrapper. One per attention mode."""

    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs=None,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx=priority_model_idx,
            model_init_kwargs=model_init_kwargs,
        )
        self.mode = config.attn_mode
        self.bucket_config = None

        if self.mode == "self":
            self.frame_counts = SELF_FRAME_COUNTS
        elif self.mode == "cached":
            self.frame_counts = CACHED_FRAME_COUNTS
        elif self.mode == "update":
            self.frame_counts = UPDATE_FRAME_COUNTS
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.base_freqs = make_freqs(HEAD_DIM)

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        """Generate example inputs for compilation, one per bucket."""
        dtype = torch.bfloat16
        inputs = []

        lat_h = self.config.height // 8
        lat_w = self.config.width // 8

        # Check if self mode should include a mask (for padded unified model)
        use_mask_for_self = getattr(self.config, "use_mask_for_self", False)

        for num_frames in self.frame_counts:
            in_channels = self.config.in_channels
            post_f = num_frames // PATCH_T
            post_h = lat_h // PATCH_H
            post_w = lat_w // PATCH_W
            seq_len = post_f * post_h * post_w

            # Core inputs
            hidden_states = torch.randn(
                1, in_channels, num_frames, lat_h, lat_w, dtype=dtype
            )
            # Per-frame timestep: [B, F] — graduated noise schedule
            # CRITICAL: timestep must be float32 to preserve precision for
            # sinusoidal embedding. bf16 rounds e.g. 558.0→560.0, causing
            # ~10% error in time embeddings (corr=0.89 for some timesteps).
            timestep = torch.full((1, num_frames), 500.0, dtype=torch.float32)
            encoder_hidden_states = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)

            # RoPE as cos/sin tensors (real-valued, Neuron-compatible)
            rope_cos, rope_sin = precompute_rope_embeddings(
                self.base_freqs, post_f, post_h, post_w
            )
            rope_cos = rope_cos.to(torch.float32)
            rope_sin = rope_sin.to(torch.float32)

            if self.mode == "self" and not use_mask_for_self:
                # Original self mode: no mask, no KV
                inputs.append(
                    (
                        hidden_states,
                        timestep,
                        encoder_hidden_states,
                        rope_cos,
                        rope_sin,
                    )
                )
            elif self.mode == "self" and use_mask_for_self:
                # Padded self mode: include mask (for padding support)
                # Mask is [B, 1, seq_len, seq_len] — self-attention is square
                attn_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype)
                inputs.append(
                    (
                        hidden_states,
                        timestep,
                        encoder_hidden_states,
                        rope_cos,
                        rope_sin,
                        attn_mask,
                    )
                )
            else:
                tp = (
                    self.config.neuron_config.tp_degree
                    if hasattr(self.config, "neuron_config")
                    and self.config.neuron_config
                    else _get_tp_degree()
                )
                heads_per_rank = math.ceil(NUM_HEADS / tp) * tp // tp

                # Per-layer KV buffers as separate tensors (not stacked)
                # to avoid compiler OOM from handling massive [NUM_LAYERS, ...] tensor
                kv_tensors = []
                for _ in range(NUM_LAYERS):
                    kv_tensors.append(
                        torch.randn(
                            1,
                            MAX_ATTENTION_SIZE,
                            heads_per_rank,
                            HEAD_DIM,
                            dtype=dtype,
                        )
                    )  # kv_k for this layer
                    kv_tensors.append(
                        torch.randn(
                            1,
                            MAX_ATTENTION_SIZE,
                            heads_per_rank,
                            HEAD_DIM,
                            dtype=dtype,
                        )
                    )  # kv_v for this layer

                attn_mask = torch.zeros(1, 1, seq_len, MAX_ATTENTION_SIZE, dtype=dtype)

                # Forward signature: (hidden, timestep, enc, cos, sin, mask, k0, v0, k1, v1, ...)
                inputs.append(
                    (
                        hidden_states,
                        timestep,
                        encoder_hidden_states,
                        rope_cos,
                        rope_sin,
                        attn_mask,
                        *kv_tensors,
                    )
                )

        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def forward(self, *args, **kwargs):
        return self._forward(*args)


# ─── Application ───────────────────────────────────────────────────────────


class NeuronCausalWanApplication(NeuronApplicationBase):
    """
    NeuronApplicationBase for CausalWanModel.

    Single-mode: one application per attention mode (self/cached/update).
    Used when compiling or loading individual modes.
    """

    _model_cls = NeuronCausalWanTransformer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper_cls = NeuronCausalWanModelWrapper
        self.model = self.model_wrapper_cls(
            config=self.config,
            model_cls=self._model_cls,
            tag=f"CausalWan_{self.config.attn_mode}",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_model_wrapper_cls(self):
        return NeuronCausalWanModelWrapper

    @classmethod
    def get_config_cls(cls):
        return CausalWanInferenceConfig

    def get_compiler_args(self, mode=None):
        if mode is None:
            mode = self.config.attn_mode
        if mode == "self":
            return (
                "--auto-cast=none "
                "--model-type=transformer "
                "-O1 "
                "--internal-max-instruction-limit=15000000 "
                "--tensorizer-options='--enable-ccop-compute-overlap "
                "--cc-pipeline-tiling-factor=2'"
            )
        else:
            # Cached/update modes have larger graphs (KV buffers as inputs).
            # Use -O1 but no ccop-compute-overlap to reduce compiler memory.
            return (
                "--auto-cast=none "
                "--model-type=transformer "
                "-O1 "
                "--internal-max-instruction-limit=15000000"
            )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Convert native Wan model state dict to our NxDI naming.

        Supports BOTH formats:
          1. Native Wan format (from Wan-AI/Wan2.1-T2V-1.3B safetensors):
             blocks.N.self_attn.q/k/v/o, blocks.N.ffn.0/2, head.head, etc.
          2. Diffusers format (from WanTransformer3DModel):
             blocks.N.attn1.to_q/to_k/to_v/to_out.0, blocks.N.ffn.net.0.proj, etc.

        Auto-detects format by checking for native Wan keys.
        """
        new_state_dict = {}

        # Auto-detect format: native Wan has "head.head.weight", diffusers has "proj_out.weight"
        is_native_wan = "head.head.weight" in state_dict or any(
            k.startswith("text_embedding.") for k in state_dict
        )

        for key, value in state_dict.items():
            new_key = key

            if is_native_wan:
                # ── Native Wan format conversions ──

                # Attention: .self_attn.q → .self_attn.to_q (add 'to_' prefix)
                # But NOT for norm_q/norm_k (those stay as-is)

                # Self-attn Q/K/V/O projections: add 'to_' prefix
                new_key = re.sub(
                    r"\.self_attn\.(q|k|v)\.", r".self_attn.to_\1.", new_key
                )
                new_key = re.sub(r"\.self_attn\.o\.", r".self_attn.to_out.", new_key)

                # Cross-attn Q/K/V/O projections: add 'to_' prefix
                new_key = re.sub(
                    r"\.cross_attn\.(q|k|v)\.", r".cross_attn.to_\1.", new_key
                )
                new_key = re.sub(r"\.cross_attn\.o\.", r".cross_attn.to_out.", new_key)

                # FFN: .ffn.0 → .ffn_gelu_proj, .ffn.2 → .ffn_out
                new_key = new_key.replace(".ffn.0.", ".ffn_gelu_proj.")
                new_key = new_key.replace(".ffn.2.", ".ffn_out.")

                # Block modulation: .modulation → .scale_shift_table
                if ".modulation" in new_key and "blocks." in new_key:
                    new_key = new_key.replace(".modulation", ".scale_shift_table")

                # Norm renaming: native Wan norm3 = cross-attn norm = our norm2
                # (native Wan numbering differs from diffusers)
                new_key = re.sub(r"(blocks\.\d+)\.norm3\.", r"\1.norm2.", new_key)

                # Output head: head.head → proj_out
                new_key = new_key.replace("head.head.", "proj_out.")

                # Output modulation: head.modulation → scale_shift_table
                if new_key == "head.modulation":
                    new_key = "scale_shift_table"

                # Text embedder: text_embedding.0 → condition_embedder.text_embedder_linear_1
                new_key = new_key.replace(
                    "text_embedding.0.", "condition_embedder.text_embedder_linear_1."
                )
                new_key = new_key.replace(
                    "text_embedding.2.", "condition_embedder.text_embedder_linear_2."
                )

                # Time embedder: time_embedding.0 → condition_embedder.time_embedder_linear_1
                new_key = new_key.replace(
                    "time_embedding.0.", "condition_embedder.time_embedder_linear_1."
                )
                new_key = new_key.replace(
                    "time_embedding.2.", "condition_embedder.time_embedder_linear_2."
                )

                # Time projection: time_projection.1 → condition_embedder.time_proj
                new_key = new_key.replace(
                    "time_projection.1.", "condition_embedder.time_proj."
                )

            else:
                # ── Diffusers format conversions ──

                # Self-attention: attn1 → self_attn
                new_key = new_key.replace(".attn1.", ".self_attn.")

                # Cross-attention: attn2 → cross_attn
                new_key = new_key.replace(".attn2.", ".cross_attn.")

                # Attention output: .to_out.0. → .to_out.
                new_key = new_key.replace(".to_out.0.", ".to_out.")

                # FFN: ffn.net.0.proj → ffn_gelu_proj, ffn.net.2 → ffn_out
                new_key = new_key.replace(".ffn.net.0.proj.", ".ffn_gelu_proj.")
                new_key = new_key.replace(".ffn.net.2.", ".ffn_out.")

                # Condition embedder: flatten Sequential naming
                new_key = new_key.replace(
                    "condition_embedder.text_embedder.linear_1.",
                    "condition_embedder.text_embedder_linear_1.",
                )
                new_key = new_key.replace(
                    "condition_embedder.text_embedder.linear_2.",
                    "condition_embedder.text_embedder_linear_2.",
                )
                new_key = new_key.replace(
                    "condition_embedder.time_embedder.linear_1.",
                    "condition_embedder.time_embedder_linear_1.",
                )
                new_key = new_key.replace(
                    "condition_embedder.time_embedder.linear_2.",
                    "condition_embedder.time_embedder_linear_2.",
                )

            new_state_dict[new_key] = value.clone().detach().contiguous()

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        return state_dict


# ─── Helper ────────────────────────────────────────────────────────────────


def create_causal_wan_config(mode, tp_degree=2, height=480, width=832):
    """Create config for a specific attention mode."""
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        torch_dtype=torch.bfloat16,
        batch_size=1,
    )
    return CausalWanInferenceConfig(
        neuron_config=neuron_config,
        attn_mode=mode,
        height=height,
        width=width,
    )


# ─── Unified Multi-Mode Application ───────────────────────────────────────


SELF_TAG = "CausalWan_self"
CACHED_TAG = "CausalWan_cached"
UPDATE_TAG = "CausalWan_update"


class NeuronCausalWanUnifiedApplication(NeuronApplicationBase):
    """
    Single-NEFF NeuronApplicationBase using only the cached attention graph.

    Uses ONLY 1 NEFF (cached mode) for ALL calls: self-attention, cached attention,
    and update calls. Self-attention is achieved by passing zero KV buffers with a
    mask that only unmasks the current positions.

    This eliminates Neuron compiler numerical non-determinism between separate NEFFs:
    since there's only one compiled graph, all calls produce numerically consistent
    results. The transition from self→cached attention (window 4→5) no longer causes
    a quality cliff.

    Architecture:
      - Single NEFF: CausalWan_cached (KV input, SDPA with mask)
      - Self-attention: forward_cached() with zero KV, current-only mask
      - Cached attention: forward_cached() with anchor KV in buffer
      - Update attention: forward_cached() with full cache KV

    Usage:
        config = create_unified_causal_wan_config(tp_degree=4)
        app = NeuronCausalWanUnifiedApplication(model_path=weight_path, config=config)
        app.compile(compiled_path)  # compiles 1 NEFF (cached_f15 only)
        app.load(compiled_path)
        # All calls go through forward_cached:
        out = app.forward_cached(hidden, timestep, enc, cos, sin, mask, *kv)
        # forward_self is an alias for convenience (pipeline builds zero KV)
        out = app.forward_self(hidden, timestep, enc, cos, sin, mask, *kv)
    """

    _model_cls = NeuronCausalWanTransformer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper_cls = NeuronCausalWanModelWrapper

        # Single NEFF: only cached mode
        self._cached_wrapper = self._make_wrapper("cached", CACHED_TAG, priority=0)
        self.models.append(self._cached_wrapper)

    def _make_wrapper(self, mode, tag, priority):
        """Create a ModelWrapper for a specific attention mode."""
        mode_config = CausalWanInferenceConfig(
            neuron_config=self.config.neuron_config,
            attn_mode=mode,
            height=self.config.height,
            width=self.config.width,
            use_mask_for_self=False,
        )
        wrapper = self.model_wrapper_cls(
            config=mode_config,
            model_cls=self._model_cls,
            tag=tag,
            compiler_args=self._get_compiler_args_for_mode(mode),
            priority_model_idx=priority,
        )
        # Override frame_counts: single f15 bucket
        wrapper.frame_counts = [15]
        return wrapper

    def _get_compiler_args_for_mode(self, mode):
        return (
            "--auto-cast=none "
            "--model-type=transformer "
            "-O1 "
            "--internal-max-instruction-limit=15000000"
        )

    def forward(self, *model_inputs, **kwargs):
        """Default forward dispatches to cached mode."""
        return self.forward_cached(*model_inputs, **kwargs)

    def forward_self(self, *model_inputs, **kwargs):
        """Forward pass through self-attention (routed through cached NEFF).

        The caller must provide the same input signature as forward_cached:
        (hidden, timestep, enc, cos, sin, mask, *kv_tensors)
        where kv_tensors are zero-filled and mask only unmasks current positions.
        """
        return self._call_nxd_model(model_inputs)

    def forward_cached(self, *model_inputs, **kwargs):
        """Forward pass through cached attention mode (also used for update)."""
        return self._call_nxd_model(model_inputs)

    def _call_nxd_model(self, inputs):
        """Call the NxDModel directly with a tuple/list of inputs."""
        return self.traced_model.nxd_model(list(inputs))

    def get_model_wrapper_cls(self):
        return NeuronCausalWanModelWrapper

    @classmethod
    def get_config_cls(cls):
        return CausalWanInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """Delegates to the single-mode version."""
        return NeuronCausalWanApplication.convert_hf_to_neuron_state_dict(
            state_dict, config
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        return state_dict

    def load_weights(
        self, compiled_model_path, start_rank_id=None, local_ranks_size=None
    ):
        """Override to shard RMSNorm weights per TP rank.

        NxDI's default weight sharding only handles ColumnParallelLinear and
        RowParallelLinear. Our RMSNorm modules have full-size (dim=1536) weights
        that need to be sliced per rank to match the TP-sharded Q/K projections.

        For rank r with TP=4: norm_weight[r*384 : (r+1)*384] → norm_weight[:384]
        The compiled NEFF always reads weight[:shard_dim], so we put each rank's
        correct slice into the first shard_dim positions of the weight tensor.
        """
        import time
        import logging
        from pathlib import Path

        logger = logging.getLogger("Neuron")

        if start_rank_id is None:
            start_rank_id = self.neuron_config.start_rank_id
        if local_ranks_size is None:
            local_ranks_size = self.neuron_config.local_ranks_size

        start_time = time.monotonic()
        tp = self.neuron_config.tp_degree

        if self.neuron_config.save_sharded_checkpoint:
            # Pre-sharded path: load and fix norm weights per rank
            from safetensors.torch import load_file

            weights = []
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                ckpt = load_file(
                    str(
                        Path(compiled_model_path)
                        / f"weights/tp{rank}_sharded_checkpoint.safetensors"
                    )
                )
                self._shard_norm_weights(ckpt, rank, tp)
                weights.append(ckpt)
        else:
            # Standard path: shard with NxDI, then fix norm weights
            logger.info("Sharding weights on load...")
            weights = self.get_builder().shard_checkpoint()

            for rank_idx, rank in enumerate(
                range(start_rank_id, start_rank_id + local_ranks_size)
            ):
                self._shard_norm_weights(weights[rank_idx], rank, tp)

        start_rank_tensor = torch.tensor(
            [start_rank_id], dtype=torch.int32, device="cpu"
        )
        self.traced_model.nxd_model.initialize(weights, start_rank_tensor)

        logger.info(
            f"Finished weights loading in {time.monotonic() - start_time} seconds"
        )

    @staticmethod
    def _shard_norm_weights(checkpoint, rank, tp):
        """Shard RMSNorm weights in a per-rank checkpoint dict.

        For norm weights (norm_q.weight, norm_k.weight) of shape [dim]:
        Replace with the rank's slice: weight[rank*shard : (rank+1)*shard]
        padded back to full dim (since the compiled NEFF expects full-size param
        but only reads [:shard_dim]).
        """
        for key in list(checkpoint.keys()):
            if ".norm_q.weight" in key or ".norm_k.weight" in key:
                w = checkpoint[key]
                full_dim = w.shape[0]
                shard_dim = full_dim // tp
                start = rank * shard_dim
                end = start + shard_dim
                # Put this rank's slice into positions [:shard_dim]
                # Leave the rest as-is (won't be accessed)
                new_w = w.clone()
                new_w[:shard_dim] = w[start:end]
                checkpoint[key] = new_w


def create_unified_causal_wan_config(tp_degree=4, height=480, width=832):
    """Create config for the unified multi-mode application."""
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        torch_dtype=torch.bfloat16,
        batch_size=1,
    )
    # Use "self" as base attn_mode; individual wrappers override per-mode
    return CausalWanInferenceConfig(
        neuron_config=neuron_config,
        attn_mode="self",
        height=height,
        width=width,
    )


# ─── Test ──────────────────────────────────────────────────────────────────


def _test_cpu():
    """Quick sanity check on CPU."""
    print("Testing NeuronCausalWanTransformer on CPU (2 layers)...")

    class FakeConfig:
        hidden_size = DIM
        intermediate_size = FFN_DIM
        num_attention_heads = NUM_HEADS
        attention_head_dim = HEAD_DIM
        num_hidden_layers = 2
        in_channels = IN_CHANNELS
        attn_mode = "self"

    model = NeuronCausalWanTransformer(FakeConfig())
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    B, F_lat = 1, 3
    H_lat, W_lat = 60, 104
    x = torch.randn(B, IN_CHANNELS, F_lat, H_lat, W_lat, dtype=torch.bfloat16)
    t = torch.tensor([500.0], dtype=torch.bfloat16)
    enc = torch.randn(B, TEXT_SEQ_LEN, TEXT_DIM, dtype=torch.bfloat16)

    freqs = make_freqs(HEAD_DIM)
    rope_cos, rope_sin = precompute_rope_embeddings(
        freqs, F_lat // PATCH_T, H_lat // PATCH_H, W_lat // PATCH_W
    )
    rope_cos = rope_cos.to(torch.float32)
    rope_sin = rope_sin.to(torch.float32)

    with torch.no_grad():
        out = model(x, t, enc, rope_cos, rope_sin)
    print(f"  Output shape: {out[0].shape}")
    print(f"  Cache tensors: {len(out) - 1}")
    print("  PASS")


if __name__ == "__main__":
    _test_cpu()
