"""
NKI Flash Attention kernel for MiMo-V2-Flash CTE (Context Encoding).

Handles MiMo's unique architecture:
  - Asymmetric head dims: Q/K use d_qk (192), V uses d_v (128)
  - d_qk not a multiple of 128: wrapper zero-pads Q/K to next multiple of 128
    so the kernel always sees uniform D_TILE=128 chunks
  - d_v <= 128: PV matmul fits on free axis, no d-tiling needed
  - Optional sliding window attention (SWA layers)
  - Causal masking

Note: Attention sink bias is NOT handled by this kernel. Layers with sink bias
should fall back to the PyTorch attention path.

Input layout (after wrapper padding):
    Q: (B*H, seqlen_q, d_qk_padded) -- bfloat16, d_qk_padded % 128 == 0
    K: (B*H_kv, seqlen_k, d_qk_padded) -- bfloat16
    V: (B*H_kv, seqlen_k, d_v) -- bfloat16

Output layout:
    O: (B*H, d_v, seqlen_q) -- bfloat16, transposed

Based on proven kernels:
  - Gemma4 nki_flash_attn_large_d.py (d-tiling pattern)
  - Qwen3.5 nki_flash_attn_d256.py (d=256 tiled QK accumulation)
"""

import math
import numpy as np
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc import nki
from neuronxcc.nki.language import par_dim

B_P = 128  # partition dim max (nl.tile_size.pmax)
B_F = 512  # free dim max for matmul moving operand
D_TILE = 128  # head_dim tile size for QK contraction
NEG_INF = -9984.0  # bfloat16-safe negative infinity


@nki.jit
def flash_attn_mimo(
    q,
    k,
    v,
    scale: float = 1.0,
    use_causal_mask: bool = True,
    sliding_window: int = 0,
):
    """
    Flash attention for MiMo's asymmetric head dims.

    Q/K must be pre-padded so d_qk is a multiple of D_TILE (128).
    For MiMo's d_qk=192, the wrapper pads to 256 with zeros.

    The QK matmul is tiled: QK = sum_dc( Q_dc^T @ K_dc ) over d-chunks.
    The PV matmul places d_v on the free axis (max 512), no tiling needed.

    Args:
        q: (bs, seqlen_q, d_qk) -- bfloat16, d_qk must be multiple of 128
        k: (bs_kv, seqlen_k, d_qk) -- bfloat16
        v: (bs_kv, seqlen_k, d_v) -- bfloat16
        scale: float, 1/sqrt(d_qk_original) -- NOT 1/sqrt(d_qk_padded)
        use_causal_mask: bool
        sliding_window: int, 0 = no sliding window

    Returns:
        o: (bs, d_v, seqlen_q) -- bfloat16, transposed layout
    """
    bs, seqlen_q, d_qk = q.shape
    bs_kv, seqlen_k, _ = k.shape
    _, _, d_v = v.shape

    assert d_qk % D_TILE == 0, f"d_qk must be multiple of {D_TILE}, got {d_qk}"
    assert d_v <= B_P, f"d_v must be <= {B_P}, got {d_v}"
    assert seqlen_q % B_P == 0, f"seqlen_q must be divisible by {B_P}"
    assert seqlen_k % B_P == 0, f"seqlen_k must be divisible by {B_P}"

    num_d_chunks = d_qk // D_TILE  # e.g., 2 for d_qk=256 (padded from 192)
    q_h_per_k_h = bs // bs_kv  # GQA ratio

    # Output: (bs, d_v, seqlen_q) -- transposed layout
    o = nl.ndarray((bs, d_v, seqlen_q), dtype=q.dtype, buffer=nl.shared_hbm)

    batch_id = nl.program_id(axis=0)

    n_q_tiles = seqlen_q // B_P
    # K/V tiles: use B_F if seqlen_k is divisible, else B_P
    kv_tile_size = B_F if seqlen_k % B_F == 0 else B_P
    n_kv_tiles = seqlen_k // kv_tile_size

    for i_q_h in nl.affine_range(q_h_per_k_h):
        q_batch = batch_id * q_h_per_k_h + i_q_h
        k_batch = batch_id

        for qi in nl.sequential_range(n_q_tiles):
            # Accumulators for online softmax
            o_acc = nl.zeros((par_dim(B_P), d_v), dtype=np.float32, buffer=nl.sbuf)
            m_acc = nl.full((par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32)
            l_acc = nl.full((par_dim(B_P), 1), fill_value=NEG_INF, dtype=np.float32)

            # Load Q tile: uniform D_TILE chunks, all same size
            q_chunks = nl.ndarray(
                (num_d_chunks, par_dim(D_TILE), B_P), dtype=nl.bfloat16
            )
            for dc in nl.affine_range(num_d_chunks):
                # Load Q slice: (B_P, D_TILE) then transpose to (D_TILE, B_P)
                q_tile_raw = nl.ndarray((par_dim(B_P), D_TILE), dtype=nl.bfloat16)
                q_tile_raw[:, :] = nl.load(
                    q[q_batch, nl.ds(qi * B_P, B_P), nl.ds(dc * D_TILE, D_TILE)]
                )
                q_t_psum = nl.ndarray(
                    (par_dim(D_TILE), B_P), dtype=np.float32, buffer=nl.psum
                )
                q_t_psum[:, :] = nisa.nc_transpose(q_tile_raw)
                q_chunks[dc, :, :] = nl.copy(q_t_psum, dtype=nl.bfloat16)

            # Apply scale to Q chunks (scale = 1/sqrt(d_qk_original))
            if scale != 1.0:
                for dc in nl.affine_range(num_d_chunks):
                    q_chunks[dc, :, :] = nisa.tensor_scalar(
                        q_chunks[dc], nl.multiply, scale, dtype=nl.bfloat16
                    )

            for kvi in nl.sequential_range(n_kv_tiles):
                kv_start = kvi * kv_tile_size

                # Causal skip
                if use_causal_mask:
                    q_end = (qi + 1) * B_P - 1
                    skip_condition = q_end < kv_start
                else:
                    skip_condition = False

                # Sliding window skip
                if sliding_window > 0 and use_causal_mask:
                    q_start_pos = qi * B_P
                    kv_end = kv_start + kv_tile_size - 1
                    skip_sw = kv_end < (q_start_pos - sliding_window + 1)
                    skip_condition = skip_condition or skip_sw

                if not skip_condition:
                    # Load K tile: uniform D_TILE chunks, transposed
                    k_chunks = nl.ndarray(
                        (num_d_chunks, par_dim(D_TILE), kv_tile_size),
                        dtype=nl.bfloat16,
                    )
                    for dc in nl.affine_range(num_d_chunks):
                        if kv_tile_size <= B_P:
                            k_raw = nl.ndarray(
                                (par_dim(kv_tile_size), D_TILE), dtype=nl.bfloat16
                            )
                            k_raw[:, :] = nl.load(
                                k[
                                    k_batch,
                                    nl.ds(kv_start, kv_tile_size),
                                    nl.ds(dc * D_TILE, D_TILE),
                                ]
                            )
                            k_t_psum = nl.ndarray(
                                (par_dim(D_TILE), kv_tile_size),
                                dtype=np.float32,
                                buffer=nl.psum,
                            )
                            k_t_psum[:, :] = nisa.nc_transpose(k_raw)
                            k_chunks[dc, :, :] = nl.copy(k_t_psum, dtype=nl.bfloat16)
                        else:
                            # Large tile (B_F=512): load in B_P sub-tiles
                            n_sub = kv_tile_size // B_P
                            for si in nl.affine_range(n_sub):
                                k_raw = nl.ndarray(
                                    (par_dim(B_P), D_TILE), dtype=nl.bfloat16
                                )
                                k_raw[:, :] = nl.load(
                                    k[
                                        k_batch,
                                        nl.ds(kv_start + si * B_P, B_P),
                                        nl.ds(dc * D_TILE, D_TILE),
                                    ]
                                )
                                k_t_psum = nl.ndarray(
                                    (par_dim(D_TILE), B_P),
                                    dtype=np.float32,
                                    buffer=nl.psum,
                                )
                                k_t_psum[:, :] = nisa.nc_transpose(k_raw)
                                k_chunks[dc, :, nl.ds(si * B_P, B_P)] = nl.copy(
                                    k_t_psum, dtype=nl.bfloat16
                                )

                    # ---- Tiled QK matmul ----
                    qk = nl.ndarray(
                        (par_dim(B_P), kv_tile_size),
                        dtype=np.float32,
                        buffer=nl.psum,
                    )
                    qk[:, :] = nl.matmul(q_chunks[0], k_chunks[0], transpose_x=True)
                    for dc in nl.affine_range(num_d_chunks - 1):
                        qk[:, :] += nl.matmul(
                            q_chunks[dc + 1], k_chunks[dc + 1], transpose_x=True
                        )

                    # Move to SBUF for masking/softmax
                    qk_sbuf = nl.ndarray(
                        (par_dim(B_P), kv_tile_size),
                        dtype=np.float32,
                        buffer=nl.sbuf,
                    )

                    # Apply causal mask
                    if use_causal_mask:
                        i_q, i_k = nl.mgrid[0:B_P, 0:kv_tile_size]
                        q_pos = qi * B_P + i_q
                        k_pos = kv_start + i_k
                        pred_causal = q_pos >= k_pos

                        qk_sbuf[:, :] = nisa.affine_select(
                            pred=pred_causal,
                            on_true_tile=qk,
                            on_false_value=NEG_INF,
                            dtype=np.float32,
                        )

                        # Sliding window mask
                        if sliding_window > 0:
                            pred_sw = (q_pos - k_pos) < sliding_window
                            qk_sw = nl.ndarray(
                                (par_dim(B_P), kv_tile_size),
                                dtype=np.float32,
                                buffer=nl.sbuf,
                            )
                            qk_sw[:, :] = nisa.affine_select(
                                pred=pred_sw,
                                on_true_tile=qk_sbuf,
                                on_false_value=NEG_INF,
                                dtype=np.float32,
                            )
                            qk_sbuf = qk_sw
                    else:
                        qk_sbuf[:, :] = nl.copy(qk, dtype=np.float32)

                    # ---- Online softmax ----
                    new_max = nisa.tensor_reduce(
                        np.max, qk_sbuf, axis=(1,), dtype=np.float32, negate=False
                    )

                    m_prev = nl.copy(m_acc[:, 0])
                    m_acc[:, 0] = nl.maximum(m_prev, new_max)
                    m_cur = m_acc[:, 0]

                    # Rescale previous output
                    alpha = nisa.activation(np.exp, m_cur, bias=m_prev, scale=-1.0)
                    o_acc[...] = nl.multiply(o_acc, alpha)

                    # exp(qk - max) and row sum
                    p = nl.ndarray((par_dim(B_P), kv_tile_size), dtype=nl.bfloat16)
                    p_sum = nl.ndarray((par_dim(B_P), 1), dtype=np.float32)
                    p[:, :] = nisa.activation_reduce(
                        np.exp,
                        qk_sbuf,
                        bias=-1 * m_cur,
                        scale=1.0,
                        reduce_op=nl.add,
                        reduce_res=p_sum[:, 0],
                        dtype=nl.bfloat16,
                    )

                    # ---- Load V tile ----
                    n_v_sub = kv_tile_size // B_P
                    v_tile = nl.ndarray((n_v_sub, par_dim(B_P), d_v), dtype=nl.bfloat16)
                    for vi in nl.affine_range(n_v_sub):
                        v_tile[vi, :, :] = nl.load(
                            v[k_batch, nl.ds(kv_start + vi * B_P, B_P), nl.ds(0, d_v)],
                            dtype=nl.bfloat16,
                        )

                    # ---- PV matmul ----
                    # Transpose P for matmul contraction
                    p_t = nl.ndarray((par_dim(B_P), kv_tile_size), dtype=nl.bfloat16)
                    for ti in nl.affine_range(kv_tile_size // B_P):
                        p_t_psum = nl.ndarray(
                            (par_dim(B_P), B_P),
                            dtype=np.float32,
                            buffer=nl.psum,
                        )
                        p_t_psum[:, :] = nisa.nc_transpose(p[:, nl.ds(ti * B_P, B_P)])
                        p_t[:, nl.ds(ti * B_P, B_P)] = nl.copy(
                            p_t_psum, dtype=nl.bfloat16
                        )

                    # Accumulate PV
                    pv = nl.zeros(
                        (par_dim(B_P), d_v),
                        dtype=np.float32,
                        buffer=nl.psum,
                        lazy_initialization=True,
                    )
                    for vi in nl.affine_range(n_v_sub):
                        pv[:, :] += nl.matmul(
                            p_t[:, nl.ds(vi * B_P, B_P)],
                            v_tile[vi, :, :],
                            transpose_x=True,
                        )

                    o_acc[:, :] = nl.add(o_acc, pv)

                    # Update log-sum-exp
                    exp_l = nisa.activation(nl.exp, m_cur, bias=l_acc[:, 0], scale=-1.0)
                    l_acc[:, 0] = nl.add(
                        m_cur,
                        nisa.activation(nl.log, exp_l, bias=p_sum[:, 0]),
                    )

            # ---- Final rescale and store ----
            final_exp = nisa.activation(
                np.exp, l_acc[:, 0], bias=m_acc[:, 0], scale=-1.0
            )
            out = nl.multiply(o_acc, final_exp, dtype=nl.bfloat16)

            # Store: (B_P, d_v) -> o[batch, d_v, seqlen_q] transposed
            out_t_psum = nl.ndarray(
                (par_dim(d_v), B_P), dtype=np.float32, buffer=nl.psum
            )
            out_t_psum[:, :] = nisa.nc_transpose(out)
            out_t = nl.ndarray((par_dim(d_v), B_P), dtype=nl.bfloat16)
            out_t[:, :] = nl.copy(out_t_psum, dtype=nl.bfloat16)
            nl.store(
                o[q_batch, nl.ds(0, d_v), nl.ds(qi * B_P, B_P)],
                out_t,
            )

    return o


def flash_attn_mimo_wrapper(
    q, k, v, scale=None, use_causal_mask=True, sliding_window=0
):
    """
    PyTorch wrapper for the NKI flash attention kernel.

    Handles:
      1. Layout conversion: [B, H, S, D] -> [B*H, S, D]
      2. Padding d_qk to next multiple of 128 (zero-padding is lossless for QK matmul)
      3. Grid launch and output reshape

    Args:
        q: (bsz, num_heads, seqlen_q, d_qk)  -- bfloat16
        k: (bsz, num_kv_heads, seqlen_k, d_qk)  -- bfloat16
        v: (bsz, num_kv_heads, seqlen_k, d_v)  -- bfloat16
        scale: float, 1/sqrt(d_qk). If None, computed from ORIGINAL d_qk.
        use_causal_mask: bool
        sliding_window: int, 0 = disabled

    Returns:
        attn_output: (bsz, num_heads, seqlen_q, d_v)  -- bfloat16
    """
    import torch

    bsz, num_heads, seqlen_q, d_qk = q.shape
    _, num_kv_heads, seqlen_k, d_v_dim = v.shape

    if scale is None:
        scale = 1.0 / math.sqrt(d_qk)

    # Pad d_qk to next multiple of D_TILE (128) if needed
    d_qk_padded = ((d_qk + D_TILE - 1) // D_TILE) * D_TILE
    if d_qk_padded != d_qk:
        pad_size = d_qk_padded - d_qk
        q = torch.nn.functional.pad(q, (0, pad_size), value=0.0)
        k = torch.nn.functional.pad(k, (0, pad_size), value=0.0)

    # Merge batch and head dims: (bsz, H, S, D) -> (bsz*H, S, D)
    q_merged = q.reshape(bsz * num_heads, seqlen_q, d_qk_padded).contiguous()
    k_merged = k.reshape(bsz * num_kv_heads, seqlen_k, d_qk_padded).contiguous()
    v_merged = v.reshape(bsz * num_kv_heads, seqlen_k, d_v_dim).contiguous()

    # Launch kernel: grid = bs_kv (one program per KV head in the batch)
    grid_size = bsz * num_kv_heads
    o_merged = flash_attn_mimo[grid_size](
        q_merged,
        k_merged,
        v_merged,
        scale=scale,
        use_causal_mask=use_causal_mask,
        sliding_window=sliding_window,
    )

    # Output: (bsz*H, d_v, S) -> (bsz, H, d_v, S) -> (bsz, H, S, d_v)
    attn_output = o_merged.reshape(bsz, num_heads, d_v_dim, seqlen_q)
    attn_output = attn_output.transpose(2, 3).contiguous()

    return attn_output
