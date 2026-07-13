# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVIDIA Nemotron-3-Nano-4B-BF16 (nemotron_h dense) model for NxD Inference.

NVIDIA Nemotron-3-Nano-4B is a dense hybrid Mamba2/Attention/MLP architecture
(same `nemotron_h` model_type as the 30B, but with dense MLP instead of MoE):
- 42 layers: 21 Mamba-2, 17 dense MLP, 4 GQA Attention
- Pattern: M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-
  where M = Mamba-2, `-` = dense MLP, `*` = attention
- Each layer has ONE block (Mamba OR MLP OR Attention), single-block-per-layer
- Single pre-norm per layer, no post-attention layernorm
- Dense MLP with relu2 activation, intermediate_size=12544
- 40 Q heads / 8 KV heads (5:1 GQA), explicit head_dim=128
- tie_word_embeddings=False
- Context window: 262144 (declared; actual Neuron ceiling determined by SBUF budget)

Ported from the 30B-A3B port at `../../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/src/`.
Removes MoE (Router, MoELayer, expert weight remapping) and adds a
NeuronNemotronMLP (up_proj -> relu2 -> down_proj, ColumnParallel/RowParallel).
Carries over verbatim: RMSNorm gate-order fix (2026-07-12), USE_NATIVE_CONV1D
toggle, chunked NKI SSD kernel (with head_dim=80 audit still pending).

Key implementation details:
- Mamba state persistence via nn.ParameterList + input_output_aliases
- Manual depthwise conv1d (default) or F.conv1d groups=conv_dim (USE_NATIVE_CONV1D=1)
- Full-sequence parallel scan for prefill (head-grouped quadratic G=8 default,
  optional O(L) NKI or chunked NKI SSD via USE_CHUNKED_NKI_SCAN=1)
- O(1) recurrence for decode
- Dense MLP replaces MoE: no expert routing, no top-k, no sparse dispatch
- NeuronAttentionBase for GQA with RoPE
"""

import gc
import logging
import os
import warnings
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    ModelWrapper,
    DecoderModelInstance,
)
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
    BaseParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.utils import cpu_mode

logger = logging.getLogger(__name__)


# ==============================================================================
# NKI Selective Scan Kernel for Mamba2 prefill
# ==============================================================================
# When enabled, replaces the O(L^2) quadratic parallel scan with O(L) hardware-
# accelerated scan using nisa.tensor_tensor_scan on Trainium2.
# Set USE_NKI_SCAN = False to fall back to the quadratic implementation.
# Ported from Granite4 contrib (working/Granite4/contrib/src/modeling_granite.py).

USE_NKI_SCAN = (
    False  # O(L^2) quadratic scan: NKI scan produces more instructions at long context
)

# SSD NKI kernel: chunk-based O(L) scan using TensorE matmuls for intra-chunk
# structured attention and sequential state propagation between chunks.
# Memory bounded by O(chunk^2) per chunk, not O(L^2).
# Adapted from nki-lib scan-kernels branch ssd.py with n_groups=8 support.
USE_SSD_SCAN = False  # NKI SSD kernel: DGE OOB in full model (compiler issue)

# Pure PyTorch SSD: same chunk-based O(chunk^2) algorithm as SSD NKI kernel,
# but implemented entirely in PyTorch ops. Avoids NKI compiler DGE OOB issue.
# Memory bounded by O(chunk^2 * nheads) per chunk, not O(L^2 * nheads).
USE_PYTORCH_SSD = False  # DISABLED: WLT DramToDramTranspose hang (754 transposes)

# Chunked scan: processes seq_len in chunk_size=128 blocks, avoiding O(L^2) memory.
# Within each chunk: O(chunk^2) quadratic scan (128x128 matrix).
# Between chunks: O(1) state propagation from previous chunk's final state.
# Memory: O(chunk^2) instead of O(L^2) — enables ctx=512+ on trn2.3xlarge.
USE_CHUNKED_SCAN = False  # Disabled: multi-hour compile times + 36x TTFT regression

# Head-grouped quadratic scan: process SCAN_HEAD_GROUP heads at a time instead of
# all 64 heads simultaneously. Reduces peak intermediate memory by num_heads/SCAN_HEAD_GROUP.
# At ctx=512 with 64 heads: weight matrix [1,512,512,64] = 67MB → [1,512,512,8] = 8.4MB.
# The 1GB transpose scratchpad that caused OOM should scale proportionally.
SCAN_HEAD_GROUP = 8  # Process 8 heads at a time (64/8 = 8 groups)

# Chunked NKI SSD scan (DGE-OOB survivor pattern from Qwen3-Coder-Next DeltaNet template).
# Uses cheap-ops-only kernel with intra-chunk quadratic + inter-chunk recurrent state.
# Chunk size = 128 = P_MAX. Both linear-scaling in seq_len AND HBM-ceiling-defeating.
#
# Full-model validated on SDK 2.31 DLAMI 20260708 (Task 016 Phase 3):
#   - Bit-identical 50-token generation vs head-grouped quadratic at ctx=128 (Task 016)
#   - Bit-identical 30-token generation vs head-grouped quadratic at ctx=1024 (8 chunks)
#   - Extends ctx ceiling from 2048 to 8192 on trn2.3xlarge (4x improvement)
#   - Prefill 6-41% faster on matched contexts
#   - Decode throughput unchanged (~100-104 tok/s)
#
# When True, prefill uses `mamba2_ssd_chunked_fwd`. Decode still uses the O(1) recurrent
# path (chunked kernel is prefill-only; state handoff CE->TG is via the final_state output).
# Kernel: contrib/src/nki_kernels/nki_mamba2_ssd_chunked.py
# Wrapper: contrib/src/nki_kernels/chunked_ssd_wrapper.py
# Standalone tests: tests/test_mamba2_ssd_chunked.py, tests/test_chunked_d64.py
#
# Set via environment variable USE_CHUNKED_NKI_SCAN=1 (or =0 to explicitly disable).
USE_CHUNKED_NKI_SCAN = os.environ.get("USE_CHUNKED_NKI_SCAN", "0") == "1"

# USE_NATIVE_CONV1D: Use F.conv1d(groups=conv_dim) instead of the manual weight
# loop for the depthwise conv1d in Mamba-2 prefill. The manual loop was inherited
# from Granite4 to work around a compiler crash (TEN404) on SDK 2.28. If newer
# SDK versions no longer crash, F.conv1d should be equivalent but may compile
# to more efficient code. Off by default while the manual loop is still needed
# for the decode (TKG) path.
#
# Set via environment variable USE_NATIVE_CONV1D=1.
USE_NATIVE_CONV1D = os.environ.get("USE_NATIVE_CONV1D", "0") == "1"

# The 4B variant is dense (no MoE) so the FORCE_SPARSE_MOE_PREFILL, USE_NKI_MOE_CTE,
# USE_NKI_ROUTER flags from the 30B port are not applicable.

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    HAS_NKI = True
except ImportError:
    HAS_NKI = False
    if USE_NKI_SCAN:
        logger.warning("NKI not available, falling back to quadratic scan")
        USE_NKI_SCAN = False
    if USE_SSD_SCAN:
        logger.warning("NKI not available, falling back to quadratic scan")
        USE_SSD_SCAN = False

if HAS_NKI and USE_NKI_SCAN:
    P_MAX = 128

    @nki.jit
    def nki_scan_kernel(
        dA_exp_t,  # (NH, SL) -- pre-transposed decay coefficients
        dBx_t,  # (NH * HD * SS, SL) -- flattened+transposed
        C_t,  # (NH * SS, SL) -- flattened+transposed
        Dx_t,  # (NH * HD, SL) -- pre-computed D*x, flattened+transposed
        x_t,  # (NH * HD, SL) -- flattened+transposed (unused, for shape)
        hd_range,  # (HD,) -- dummy tensor for head_dim
        ss_range,  # (SS,) -- dummy tensor for ssm_state_size
    ):
        """NKI O(L) selective scan using tensor_tensor_scan.

        Computes: state[h, t] = dA[h, t] * state[h, t-1] + dBx[h, t]
                  y[h, d, t] = sum_s C[h, s, t] * state[h, d, s, t] + D[h] * x[h, d, t]
        """
        NH = dA_exp_t.shape[0]
        SL = dA_exp_t.shape[1]
        HD = hd_range.shape[0]
        SS = ss_range.shape[0]

        y_out = nl.ndarray((NH * HD, SL), dtype=nl.float32, buffer=nl.shared_hbm)
        final_state_out = nl.ndarray(
            (NH * HD * SS, 1), dtype=nl.float32, buffer=nl.shared_hbm
        )

        dA_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=dA_sb, value=0.0)
        nisa.dma_copy(dst=dA_sb[0:NH, 0:SL], src=dA_exp_t[0:NH, 0:SL])

        for d in nl.affine_range(HD):
            y_acc_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=y_acc_sb, value=0.0)
            Dx_row = d * NH
            nisa.dma_copy(
                dst=y_acc_sb[0:NH, 0:SL],
                src=Dx_t[Dx_row : Dx_row + NH, 0:SL],
            )

            for s in nl.affine_range(SS):
                dBx_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=dBx_sb, value=0.0)
                dBx_row = (d * SS + s) * NH
                nisa.dma_copy(
                    dst=dBx_sb[0:NH, 0:SL],
                    src=dBx_t[dBx_row : dBx_row + NH, 0:SL],
                )

                init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=init_sb, value=0.0)

                state_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(
                    dst=state_sb[0:NH, 0:SL],
                    data0=dA_sb[0:NH, 0:SL],
                    data1=dBx_sb[0:NH, 0:SL],
                    initial=init_sb[0:NH, 0:1],
                    op0=nl.multiply,
                    op1=nl.add,
                )

                final_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=final_sb[0:NH, 0:1],
                    src=state_sb[0:NH, SL - 1 : SL],
                )
                fs_row = (d * SS + s) * NH
                nisa.dma_copy(
                    dst=final_state_out[fs_row : fs_row + NH, 0:1],
                    src=final_sb[0:NH, 0:1],
                )

                C_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=C_sb, value=0.0)
                C_row = s * NH
                nisa.dma_copy(
                    dst=C_sb[0:NH, 0:SL],
                    src=C_t[C_row : C_row + NH, 0:SL],
                )

                Cs_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=Cs_sb[0:NH, 0:SL],
                    data1=C_sb[0:NH, 0:SL],
                    data2=state_sb[0:NH, 0:SL],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=y_acc_sb[0:NH, 0:SL],
                    data1=y_acc_sb[0:NH, 0:SL],
                    data2=Cs_sb[0:NH, 0:SL],
                    op=nl.add,
                )

            y_row = d * NH
            nisa.dma_copy(
                dst=y_out[y_row : y_row + NH, 0:SL],
                src=y_acc_sb[0:NH, 0:SL],
            )

        return y_out, final_state_out


def _nki_selective_scan(
    hidden_states_ssm, dt_processed, A, B, C, D, num_heads, head_dim, ssm_state_size
):
    """NKI-accelerated selective scan for Mamba2 prefill.

    Replaces O(L^2) quadratic scan with O(L) hardware scan.
    Handles data layout transformation (PyTorch -> NKI partition-first).
    Ported from Granite4 contrib.

    Args:
        hidden_states_ssm: (batch, seq_len, num_heads, head_dim) float32
        dt_processed: (batch, seq_len, num_heads) float32
        A: (num_heads,) float32 -- negative
        B: (batch, seq_len, num_heads, ssm_state_size) float32
        C: (batch, seq_len, num_heads, ssm_state_size) float32
        D: (num_heads,) float32
        num_heads, head_dim, ssm_state_size: ints

    Returns:
        y: (batch, seq_len, num_heads, head_dim)
        final_state: (batch, num_heads, head_dim, ssm_state_size)
    """
    batch, seq_len = hidden_states_ssm.shape[:2]

    # Pre-compute on PyTorch side (traced as XLA ops)
    dA_exp = torch.exp(dt_processed * A.view(1, 1, -1))  # (B, L, H)
    dB = dt_processed.unsqueeze(-1) * B  # (B, L, H, S)
    dBx = dB.unsqueeze(3) * hidden_states_ssm.unsqueeze(-1)  # (B, L, H, D, S)

    # Transpose to NKI partition-first layout (squeeze batch=1)
    dA_exp_t = dA_exp[0].transpose(0, 1).contiguous()  # (H, L)

    dBx_0 = dBx[0]  # (L, H, D, S)
    dBx_t = (
        dBx_0.permute(0, 2, 3, 1)
        .reshape(seq_len, head_dim * ssm_state_size * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (D*S*H, L)

    C_t = (
        C[0]
        .permute(0, 2, 1)
        .reshape(seq_len, ssm_state_size * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (S*H, L)

    x_t = (
        hidden_states_ssm[0]
        .permute(0, 2, 1)
        .reshape(seq_len, head_dim * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (D*H, L)

    Dx_t = (
        (D.view(1, -1, 1) * hidden_states_ssm[0])
        .permute(0, 2, 1)
        .reshape(seq_len, head_dim * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (D*H, L)

    hd_range = torch.zeros(head_dim, dtype=torch.float32, device=dA_exp_t.device)
    ss_range = torch.zeros(ssm_state_size, dtype=torch.float32, device=dA_exp_t.device)

    # Call NKI kernel
    y_flat, state_flat = nki_scan_kernel(
        dA_exp_t,
        dBx_t,
        C_t,
        Dx_t,
        x_t,
        hd_range,
        ss_range,
    )

    # Unpack: y_flat (D*H, L) -> (1, L, H, D)
    y = (
        y_flat.reshape(head_dim, num_heads, seq_len)
        .permute(2, 1, 0)
        .unsqueeze(0)
        .contiguous()
    )

    # state_flat (D*S*H, 1) -> (1, H, D, S)
    final_state = (
        state_flat.reshape(head_dim, ssm_state_size, num_heads)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    return y, final_state


if HAS_NKI and USE_SSD_SCAN:
    import numpy as np_nki

    @nki.jit
    def ssd_scan_kernel(
        x,  # (batch, nheads, seqlen, headdim) float32
        dt_2d,  # (batch*nheads, seqlen) float32 — pre-reshaped
        dt_flat,  # (batch*nheads*seqlen, 1) float32 — pre-reshaped
        A_2d,  # (nheads, 1) float32 — pre-reshaped, negative
        B,  # (batch, n_groups, seqlen, dstate) float32
        C,  # (batch, n_groups, seqlen, dstate) float32
        D_2d,  # (nheads, 1) float32 — pre-reshaped
        chunk_size_tensor,  # (chunk_size,) dummy — encodes chunk_size as shape
        n_groups_tensor,  # (n_groups,) dummy — encodes n_groups as shape
    ):
        """Chunk-based SSD kernel for Mamba-2 with n_groups support.

        Adapted from nki-lib scan-kernels/ssd.py. Key changes for Nemotron:
        - B/C are (batch, n_groups, seqlen, dstate) instead of (batch, seqlen, dstate)
        - Single affine_range(nheads) loop (like upstream), compute i_group for B/C
        - Causal mask built in-SBUF via affine_select (avoids DMA OOB issue)
        - A, D, dt pre-reshaped OUTSIDE kernel to avoid WLT DramToDramTranspose hang
        """
        batch = x.shape[0]
        nheads = x.shape[1]
        seqlen = x.shape[2]
        headdim = x.shape[3]
        n_groups = n_groups_tensor.shape[0]
        dstate = B.shape[3]
        Q = chunk_size_tensor.shape[0]
        num_chunks = seqlen // Q
        heads_per_group = nheads // n_groups

        # Allocate outputs — NKI 0.3.0 requires nl.shared_hbm for kernel outputs
        y = nl.ndarray(
            (batch, nheads, seqlen, headdim), dtype=x.dtype, buffer=nl.shared_hbm
        )
        final_state_out = nl.ndarray(
            (batch, nheads, dstate, headdim), dtype=nl.float32, buffer=nl.shared_hbm
        )

        shuffle_mask = [0] * 32

        for i_batch in nl.affine_range(batch):
            for i_head in nl.affine_range(nheads):
                # Compute group index for B/C access
                i_group = i_head // heads_per_group

                # --- Per-head constants ---
                A_h = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=A_h[0:1, 0:1], src=A_2d[i_head : i_head + 1, 0:1])

                # Causal mask: construct in SBUF using affine_select
                # (avoids NCC_IBIR243 DMA access pattern OOB with (128,128) DMA copy)
                # affine_value = row * 1 + col * (-1) = row - col
                # predicate: row - col >= 0 → lower triangular
                ones_mask = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=ones_mask, value=1.0)
                causal_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.affine_select(
                    dst=causal_sb,
                    pattern=[[-1, Q]],
                    channel_multiplier=1,
                    on_true_tile=ones_mask,
                    on_false_value=0.0,
                    cmp_op=nl.greater_equal,
                    offset=0,
                )

                ones_sb = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=ones_sb, value=1.0)

                zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=zero_11, value=0.0)

                # Init hidden state (dstate, headdim)
                state_sb = nl.ndarray(
                    (dstate, headdim), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.memset(dst=state_sb, value=0.0)

                # D broadcast — D_2d already (nheads, 1), pre-reshaped outside kernel
                D_val = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=D_val[0:1, 0:1], src=D_2d[i_head : i_head + 1, 0:1])
                D_Q = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                for i_shuf in nl.static_range((Q + 31) // 32):
                    cur_npar = min(32, Q - i_shuf * 32)
                    nisa.nc_stream_shuffle(
                        src=D_val[0:1, 0:1],
                        dst=D_Q[i_shuf * 32 : i_shuf * 32 + cur_npar, 0:1],
                        shuffle_mask=shuffle_mask,
                    )

                # --- Sequential chunk processing ---
                for i_chunk in nl.sequential_range(num_chunks):
                    chunk_start = i_chunk * Q

                    # Load x: (Q, headdim)
                    x_sb = nl.ndarray((Q, headdim), dtype=x.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=x_sb[0:Q, 0:headdim],
                        src=x[
                            i_batch,
                            i_head,
                            chunk_start : chunk_start + Q,
                            0:headdim,
                        ],
                    )

                    # Load dt: (1, Q)
                    dt_row = i_batch * nheads + i_head
                    dt_f = nl.ndarray((1, Q), dtype=dt_2d.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=dt_f[0:1, 0:Q],
                        src=dt_2d[dt_row : dt_row + 1, chunk_start : chunk_start + Q],
                    )

                    # Load B for this GROUP: (Q, dstate)
                    B_sb = nl.ndarray((Q, dstate), dtype=B.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=B_sb[0:Q, 0:dstate],
                        src=B[
                            i_batch,
                            i_group,
                            chunk_start : chunk_start + Q,
                            0:dstate,
                        ],
                    )

                    # Load C for this GROUP: (Q, dstate)
                    C_sb = nl.ndarray((Q, dstate), dtype=C.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=C_sb[0:Q, 0:dstate],
                        src=C[
                            i_batch,
                            i_group,
                            chunk_start : chunk_start + Q,
                            0:dstate,
                        ],
                    )

                    # ====== Step 1: Cumulative decay ======
                    log_decay_f = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=log_decay_f[0:1, 0:Q],
                        data=dt_f[0:1, 0:Q],
                        op0=nl.multiply,
                        operand0=A_h[0:1, 0:1],
                    )

                    cs_f = nl.ndarray((1, Q), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor_scan(
                        dst=cs_f[0:1, 0:Q],
                        data0=ones_sb[0:1, 0:Q],
                        data1=log_decay_f[0:1, 0:Q],
                        initial=zero_11[0:1, 0:1],
                        op0=nl.multiply,
                        op1=nl.add,
                    )

                    # Transpose cs: (1, Q) -> (Q, 1)
                    cs_padded = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=cs_padded[0:1, 0:Q], src=cs_f[0:1, 0:Q])
                    cs_tp_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=cs_tp_psum[0:Q, 0:Q], data=cs_padded[0:Q, 0:Q]
                    )
                    cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=cs_p[0:Q, 0:1], src=cs_tp_psum[0:Q, 0:1])

                    exp_cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(
                        op=nl.exp, data=cs_p[0:Q, 0:1], dst=exp_cs_p[0:Q, 0:1]
                    )

                    neg_cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=neg_cs_p[0:Q, 0:1],
                        data=cs_p[0:Q, 0:1],
                        op0=nl.multiply,
                        operand0=-1.0,
                    )
                    exp_neg_cs_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(
                        op=nl.exp,
                        data=neg_cs_p[0:Q, 0:1],
                        dst=exp_neg_cs_p[0:Q, 0:1],
                    )

                    # dt as (Q, 1) from flat layout
                    dt_flat_start = dt_row * seqlen + chunk_start
                    dt_p = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=dt_p[0:Q, 0:1],
                        src=dt_flat[dt_flat_start : dt_flat_start + Q, 0:1],
                    )

                    # dtx = dt * x: (Q, headdim)
                    dtx = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=dtx[0:Q, 0:headdim],
                        data=x_sb[0:Q, 0:headdim],
                        op0=nl.multiply,
                        operand0=dt_p[0:Q, 0:1],
                    )

                    # ====== Step 2: Intra-chunk structured attention ======
                    B_f32 = nl.ndarray((Q, dstate), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=B_f32[0:Q, 0:dstate], src=B_sb[0:Q, 0:dstate])

                    C_f32 = nl.ndarray((Q, dstate), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=C_f32[0:Q, 0:dstate], src=C_sb[0:Q, 0:dstate])

                    C_T_psum = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=C_T_psum[0:dstate, 0:Q], data=C_f32[0:Q, 0:dstate]
                    )
                    C_T = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=C_T[0:dstate, 0:Q], src=C_T_psum[0:dstate, 0:Q]
                    )

                    B_T_psum = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=B_T_psum[0:dstate, 0:Q], data=B_f32[0:Q, 0:dstate]
                    )
                    B_T = nl.ndarray((dstate, Q), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=B_T[0:dstate, 0:Q], src=B_T_psum[0:dstate, 0:Q]
                    )

                    # CB = C @ B^T: (Q, Q)
                    CB_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=CB_psum[0:Q, 0:Q],
                        stationary=C_T[0:dstate, 0:Q],
                        moving=B_T[0:dstate, 0:Q],
                    )
                    CB_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=CB_sb[0:Q, 0:Q], src=CB_psum[0:Q, 0:Q])

                    # Apply causal mask
                    nisa.tensor_tensor(
                        dst=CB_sb[0:Q, 0:Q],
                        data1=CB_sb[0:Q, 0:Q],
                        data2=causal_sb[0:Q, 0:Q],
                        op=nl.multiply,
                    )

                    # Scale input: X_scaled = dtx * exp(-cs)
                    X_scaled = nl.ndarray(
                        (Q, headdim), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=X_scaled[0:Q, 0:headdim],
                        data=dtx[0:Q, 0:headdim],
                        op0=nl.multiply,
                        operand0=exp_neg_cs_p[0:Q, 0:1],
                    )

                    # Transpose CB for matmul
                    CB_T_psum = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=CB_T_psum[0:Q, 0:Q], data=CB_sb[0:Q, 0:Q])
                    CB_T = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=CB_T[0:Q, 0:Q], src=CB_T_psum[0:Q, 0:Q])

                    # Y_intra = CB_causal @ X_scaled: (Q, headdim)
                    Y_psum = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=Y_psum[0:Q, 0:headdim],
                        stationary=CB_T[0:Q, 0:Q],
                        moving=X_scaled[0:Q, 0:headdim],
                    )
                    Y_intra = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=Y_intra[0:Q, 0:headdim], src=Y_psum[0:Q, 0:headdim]
                    )

                    # Scale by exp(cs)
                    nisa.tensor_scalar(
                        dst=Y_intra[0:Q, 0:headdim],
                        data=Y_intra[0:Q, 0:headdim],
                        op0=nl.multiply,
                        operand0=exp_cs_p[0:Q, 0:1],
                    )

                    # ====== Step 3: State-to-output ======
                    Y_off_psum = nl.ndarray(
                        (Q, headdim), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(
                        dst=Y_off_psum[0:Q, 0:headdim],
                        stationary=C_T[0:dstate, 0:Q],
                        moving=state_sb[0:dstate, 0:headdim],
                    )
                    Y_off = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=Y_off[0:Q, 0:headdim], src=Y_off_psum[0:Q, 0:headdim]
                    )
                    nisa.tensor_scalar(
                        dst=Y_off[0:Q, 0:headdim],
                        data=Y_off[0:Q, 0:headdim],
                        op0=nl.multiply,
                        operand0=exp_cs_p[0:Q, 0:1],
                    )

                    # ====== Step 4: Update hidden state ======
                    exp_cs_last = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(
                        op=nl.exp,
                        data=cs_f[0:1, Q - 1 : Q],
                        dst=exp_cs_last[0:1, 0:1],
                    )

                    # Broadcast exp(cs_last) to (Q, 1)
                    exp_cs_last_Q = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                    for i_shuf in nl.static_range((Q + 31) // 32):
                        cur_npar = min(32, Q - i_shuf * 32)
                        nisa.nc_stream_shuffle(
                            src=exp_cs_last[0:1, 0:1],
                            dst=exp_cs_last_Q[
                                i_shuf * 32 : i_shuf * 32 + cur_npar, 0:1
                            ],
                            shuffle_mask=shuffle_mask,
                        )

                    # decay_to_end = exp(cs_last - cs)
                    decay_to_end = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=decay_to_end[0:Q, 0:1],
                        data1=exp_neg_cs_p[0:Q, 0:1],
                        data2=exp_cs_last_Q[0:Q, 0:1],
                        op=nl.multiply,
                    )

                    # dtx_state = dtx * decay_to_end
                    dtx_state = nl.ndarray(
                        (Q, headdim), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_scalar(
                        dst=dtx_state[0:Q, 0:headdim],
                        data=dtx[0:Q, 0:headdim],
                        op0=nl.multiply,
                        operand0=decay_to_end[0:Q, 0:1],
                    )

                    # chunk_state = B^T @ dtx_state: (dstate, headdim)
                    chunk_state_psum = nl.ndarray(
                        (dstate, headdim), dtype=nl.float32, buffer=nl.psum
                    )
                    nisa.nc_matmul(
                        dst=chunk_state_psum[0:dstate, 0:headdim],
                        stationary=B_f32[0:Q, 0:dstate],
                        moving=dtx_state[0:Q, 0:headdim],
                    )
                    chunk_state_sb = nl.ndarray(
                        (dstate, headdim), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=chunk_state_sb[0:dstate, 0:headdim],
                        src=chunk_state_psum[0:dstate, 0:headdim],
                    )

                    # Broadcast exp(cs_last) to (dstate, 1)
                    exp_cs_last_N = nl.ndarray(
                        (dstate, 1), dtype=nl.float32, buffer=nl.sbuf
                    )
                    for i_shuf in nl.static_range((dstate + 31) // 32):
                        cur_npar = min(32, dstate - i_shuf * 32)
                        nisa.nc_stream_shuffle(
                            src=exp_cs_last[0:1, 0:1],
                            dst=exp_cs_last_N[
                                i_shuf * 32 : i_shuf * 32 + cur_npar, 0:1
                            ],
                            shuffle_mask=shuffle_mask,
                        )

                    # state = exp(cs_last) * state + chunk_state
                    nisa.tensor_scalar(
                        dst=state_sb[0:dstate, 0:headdim],
                        data=state_sb[0:dstate, 0:headdim],
                        op0=nl.multiply,
                        operand0=exp_cs_last_N[0:dstate, 0:1],
                    )
                    nisa.tensor_tensor(
                        dst=state_sb[0:dstate, 0:headdim],
                        data1=state_sb[0:dstate, 0:headdim],
                        data2=chunk_state_sb[0:dstate, 0:headdim],
                        op=nl.add,
                    )

                    # ====== Step 5: Combine and store output ======
                    y_chunk = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=y_chunk[0:Q, 0:headdim],
                        data1=Y_intra[0:Q, 0:headdim],
                        data2=Y_off[0:Q, 0:headdim],
                        op=nl.add,
                    )

                    # D skip connection
                    Dx = nl.ndarray((Q, headdim), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=Dx[0:Q, 0:headdim],
                        data=x_sb[0:Q, 0:headdim],
                        op0=nl.multiply,
                        operand0=D_Q[0:Q, 0:1],
                    )
                    nisa.tensor_tensor(
                        dst=y_chunk[0:Q, 0:headdim],
                        data1=y_chunk[0:Q, 0:headdim],
                        data2=Dx[0:Q, 0:headdim],
                        op=nl.add,
                    )

                    # Store output
                    nisa.dma_copy(
                        dst=y[
                            i_batch,
                            i_head,
                            chunk_start : chunk_start + Q,
                            0:headdim,
                        ],
                        src=y_chunk[0:Q, 0:headdim],
                    )

                # Store final state
                nisa.dma_copy(
                    dst=final_state_out[i_batch, i_head, 0:dstate, 0:headdim],
                    src=state_sb[0:dstate, 0:headdim],
                )

        return y, final_state_out


def _ssd_prefill_scan(
    hidden_states_ssm,
    dt_processed,
    A,
    B,
    C,
    D,
    num_heads,
    head_dim,
    ssm_state_size,
    n_groups,
    chunk_size=128,
):
    """Wrapper to call SSD NKI kernel from PyTorch.

    Handles tensor layout transformation:
    - PyTorch: x (B, L, H, D), B/C (B, L, n_groups, S)
    - NKI kernel: x (B, H, L, D), B/C (B, n_groups, L, S)

    Args:
        hidden_states_ssm: (batch, seq_len, num_heads, head_dim)
        dt_processed: (batch, seq_len, num_heads)
        A: (num_heads,) negative
        B: (batch, seq_len, n_groups, ssm_state_size) -- NOT expanded
        C: (batch, seq_len, n_groups, ssm_state_size) -- NOT expanded
        D: (num_heads,)
        n_groups: int (8 for Nemotron)
        chunk_size: int (128)

    Returns:
        y: (batch, seq_len, num_heads, head_dim)
        final_state: (batch, num_heads, head_dim, ssm_state_size)
    """
    batch, seq_len = hidden_states_ssm.shape[:2]

    # Transpose to kernel layout
    x_nki = hidden_states_ssm.permute(0, 2, 1, 3).contiguous()  # (B, H, L, D)
    dt_nki = dt_processed.permute(0, 2, 1).contiguous()  # (B, H, L)
    B_nki = B.permute(0, 2, 1, 3).contiguous()  # (B, n_groups, L, S)
    C_nki = C.permute(0, 2, 1, 3).contiguous()  # (B, n_groups, L, S)

    # Pre-reshape dt, A, D OUTSIDE the NKI kernel to avoid WLT DramToDramTranspose
    # hang. When reshapes happen inside the kernel on model parameters, the WLT
    # extracts them as weight transpose ops, causing the compiler to hang.
    dt_2d = dt_nki.reshape(batch * num_heads, seq_len).contiguous()  # (B*H, L)
    dt_flat = dt_nki.reshape(batch * num_heads * seq_len, 1).contiguous()  # (B*H*L, 1)
    A_2d = A.reshape(num_heads, 1).contiguous()  # (H, 1)
    D_2d = D.reshape(num_heads, 1).contiguous()  # (H, 1)

    # Dummy tensors encoding compile-time dimensions
    device = hidden_states_ssm.device
    chunk_size_tensor = torch.zeros(chunk_size, dtype=torch.float32, device=device)
    n_groups_tensor = torch.zeros(n_groups, dtype=torch.float32, device=device)

    y_nki, state_nki = ssd_scan_kernel(
        x_nki,
        dt_2d,
        dt_flat,
        A_2d,
        B_nki,
        C_nki,
        D_2d,
        chunk_size_tensor,
        n_groups_tensor,
    )

    # Transpose back: (B, H, L, D) -> (B, L, H, D)
    y = y_nki.permute(0, 2, 1, 3).contiguous()

    # state: (B, H, S, D) -> (B, H, D, S)
    final_state = state_nki.permute(0, 1, 3, 2).contiguous()

    return y, final_state


def _pytorch_ssd_scan(
    hidden_states_ssm,
    dt_processed,
    A,
    B,
    C,
    D,
    num_heads,
    head_dim,
    ssm_state_size,
    n_groups,
    chunk_size=128,
    padding_mask=None,
):
    """Pure PyTorch SSD (State Space Duality) scan for Mamba-2 prefill.

    Implements the same chunk-based algorithm as the NKI SSD kernel but using
    standard PyTorch ops. This avoids compiler DGE OOB issues while achieving
    O(chunk^2) memory per chunk instead of O(L^2).

    SSD algorithm per chunk (size Q):
      1. Cumulative decay: cs = cumsum(dt * A)
      2. Intra-chunk: Y_intra = exp(cs) * ((CB * causal) @ (exp(-cs) * dt * x))
         where CB = C @ B^T is the structured attention matrix (Q, Q)
      3. State-to-output: Y_off = exp(cs) * (C @ state)
      4. State update: state = exp(cs_last) * state + B^T @ (dt * x * exp(cs_last - cs))
      5. Output: y = Y_intra + Y_off + D * x

    Vectorized: within each B/C group, all heads share the same CB matrix.
    CB @ X_scaled is done as a single batched matmul across heads.

    Args:
        hidden_states_ssm: (batch, seq_len, num_heads, head_dim) float32
        dt_processed: (batch, seq_len, num_heads) float32
        A: (num_heads,) float32 (negative)
        B: (batch, seq_len, n_groups, ssm_state_size) float32 — NOT expanded
        C: (batch, seq_len, n_groups, ssm_state_size) float32 — NOT expanded
        D: (num_heads,) float32
        n_groups: int (8 for Nemotron)
        chunk_size: int (128)
        padding_mask: (batch, max_seq_len) or None

    Returns:
        y: (batch, seq_len, num_heads, head_dim) float32
        final_state: (batch, num_heads, head_dim, ssm_state_size) float32
    """
    batch, seq_len = hidden_states_ssm.shape[:2]
    device = hidden_states_ssm.device
    Q = chunk_size
    n_chunks = seq_len // Q
    heads_per_group = num_heads // n_groups

    # Reshape to (batch, n_chunks, Q, ...) for chunk processing
    x = hidden_states_ssm.reshape(batch, n_chunks, Q, num_heads, head_dim)
    dt_r = dt_processed.reshape(batch, n_chunks, Q, num_heads)
    B_chunked = B.reshape(batch, n_chunks, Q, n_groups, ssm_state_size)
    C_chunked = C.reshape(batch, n_chunks, Q, n_groups, ssm_state_size)

    # Causal mask (Q, Q) — lower triangular, reused for all chunks
    causal_mask = torch.tril(
        torch.ones(Q, Q, device=device, dtype=hidden_states_ssm.dtype)
    )

    # Allocate output
    y = torch.zeros_like(hidden_states_ssm)  # (B, L, H, D)

    # Hidden state: (B, H, S, D) — following SSD convention (dstate, headdim)
    state = torch.zeros(
        batch,
        num_heads,
        ssm_state_size,
        head_dim,
        device=device,
        dtype=hidden_states_ssm.dtype,
    )

    for chunk_idx in range(n_chunks):
        x_c = x[:, chunk_idx]  # (B, Q, H, D)
        dt_c = dt_r[:, chunk_idx]  # (B, Q, H)
        B_c = B_chunked[:, chunk_idx]  # (B, Q, n_groups, S)
        C_c = C_chunked[:, chunk_idx]  # (B, Q, n_groups, S)

        t0 = chunk_idx * Q
        y_parts = []
        state_new_parts = []

        # Process one B/C group at a time.
        # Each group has `heads_per_group` heads sharing the same B/C.
        for gg in range(n_groups):
            h_start = gg * heads_per_group
            h_end = h_start + heads_per_group
            G = heads_per_group

            x_g = x_c[:, :, h_start:h_end, :]  # (B, Q, G, D)
            dt_g = dt_c[:, :, h_start:h_end]  # (B, Q, G)
            A_g = A[h_start:h_end]  # (G,)

            # Step 1: Cumulative decay
            log_decay = dt_g * A_g.view(1, 1, -1)  # (B, Q, G)
            cs = torch.cumsum(log_decay, dim=1)  # (B, Q, G)

            exp_cs = torch.exp(cs)  # (B, Q, G)
            exp_neg_cs = torch.exp(-cs)  # (B, Q, G)

            # dt * x: (B, Q, G, D)
            dtx = dt_g.unsqueeze(-1) * x_g

            # exp(-cs) * dt * x: (B, Q, G, D)
            X_scaled = exp_neg_cs.unsqueeze(-1) * dtx

            B_g = B_c[:, :, gg, :]  # (B, Q, S)
            C_g = C_c[:, :, gg, :]  # (B, Q, S)

            # Step 2: Intra-chunk structured attention
            # CB = C @ B^T: (B, Q, Q) — shared across all heads in group
            CB = torch.bmm(C_g, B_g.transpose(1, 2))  # (B, Q, Q)
            CB = CB * causal_mask.unsqueeze(0)  # (B, Q, Q)

            # Y_intra for all G heads at once using einsum (avoids expand/reshape):
            # CB: (B, Q, Q), X_scaled: (B, Q, G, D)
            # Y_intra = CB @ X_scaled along the Q dimension
            # einsum("bti,bigd->btgd", CB, X_scaled) — broadcast CB across G heads
            Y_intra = torch.einsum("bti,bigd->btgd", CB, X_scaled)  # (B, Q, G, D)

            # Scale by exp(cs)
            Y_intra = exp_cs.unsqueeze(-1) * Y_intra  # (B, Q, G, D)

            # Step 3: State-to-output
            # Y_off = exp(cs) * (C @ state)
            # C_g: (B, Q, S), state_g: (B, G, S, D) -> Y_off: (B, Q, G, D)
            # For each head h: Y_off_h = C_g @ state_g[:, h]
            # einsum("bqs,bgsd->bqgd", C_g, state_g)
            state_g = state[:, h_start:h_end, :, :]  # (B, G, S, D)
            Y_off = torch.einsum("bqs,bgsd->bqgd", C_g, state_g)  # (B, Q, G, D)

            Y_off = exp_cs.unsqueeze(-1) * Y_off  # (B, Q, G, D)

            # D skip connection
            D_g = D[h_start:h_end].view(1, 1, -1, 1)
            y_g = Y_intra + Y_off + D_g * x_g  # (B, Q, G, D)

            y_parts.append(y_g)

            # Step 4: Compute new state for this group
            # state_new = exp(cs_last) * state + B^T @ (dtx * decay_to_end)
            cs_last = cs[:, -1:, :]  # (B, 1, G)
            decay_to_end = torch.exp(cs_last - cs)  # (B, Q, G)

            # dtx_state = dtx * decay_to_end: (B, Q, G, D)
            dtx_state = decay_to_end.unsqueeze(-1) * dtx  # (B, Q, G, D)

            # chunk_state: B^T @ dtx_state for each head
            # B_g: (B, Q, S), dtx_state: (B, Q, G, D)
            # For each head h: chunk_state_h = B_g^T @ dtx_state[:,:,h,:] -> (B, S, D)
            # einsum("bqs,bqgd->bgsd", B_g, dtx_state)
            chunk_state = torch.einsum("bqs,bqgd->bgsd", B_g, dtx_state)
            # (B, G, S, D)

            # Decay existing state and add chunk contribution
            exp_cs_last_g = torch.exp(cs_last[:, 0, :])  # (B, G)
            state_new_g = (
                exp_cs_last_g.unsqueeze(-1).unsqueeze(-1) * state_g + chunk_state
            )  # (B, G, S, D)

            state_new_parts.append(state_new_g)

        # Concatenate groups and store output
        y_chunk = torch.cat(y_parts, dim=2)  # (B, Q, H, D)
        y[:, t0 : t0 + Q, :, :] = y_chunk

        # Update full state (no in-place slice assignment — full tensor reassignment)
        state = torch.cat(state_new_parts, dim=1)  # (B, H, S, D)

    # Final state: (B, H, S, D) -> (B, H, D, S)
    final_state = state.permute(0, 1, 3, 2).contiguous()

    return y, final_state


def _chunked_quadratic_scan(
    x,
    dt_processed,
    A,
    B,
    C,
    D_param,
    num_heads,
    head_dim,
    ssm_state_size,
    chunk_size=128,
    padding_mask=None,
):
    """Chunked quadratic scan for Mamba-2 prefill.

    Processes seq_len in chunk_size blocks, avoiding O(L^2) memory.
    Within each chunk: O(chunk^2) quadratic scan (same math as full scan).
    Between chunks: propagates SSM state from previous chunk's final state.

    This is mathematically equivalent to the full O(L^2) scan but uses
    O(chunk^2) memory instead of O(L^2), enabling ctx=512+ on trn2.3xlarge.

    The key insight (Mamba-2 SSD, Section 3.3): the selective scan can be
    decomposed into intra-chunk computation (structured attention within a
    chunk) and inter-chunk state propagation (carrying the hidden state
    forward). Within each chunk, the causal weight matrix is only
    (chunk_size, chunk_size). Between chunks, the state carry is O(1).

    Args:
        x: (batch, seq_len, num_heads, head_dim) float32
        dt_processed: (batch, seq_len, num_heads) float32
        A: (num_heads,) float32 (negative)
        B: (batch, seq_len, num_heads, ssm_state_size) float32
        C: (batch, seq_len, num_heads, ssm_state_size) float32
        D_param: (num_heads,) float32
        num_heads, head_dim, ssm_state_size: ints
        chunk_size: int (default 128)
        padding_mask: (batch, max_seq_len) or None

    Returns:
        y: (batch, seq_len, num_heads, head_dim)
        final_state: (batch, num_heads, head_dim, ssm_state_size)
    """
    batch_size, seq_len = x.shape[:2]
    device = x.device

    # Number of chunks (seq_len must be divisible by chunk_size — NxDI pads)
    n_chunks = seq_len // chunk_size

    # Pre-compute dA_log for the full sequence, then chunk it
    dA_log = dt_processed * A.view(1, 1, -1)  # (B, L, H)

    # Pre-compute dB and dBx for the full sequence
    # NOTE: dBx is (B, L, H, D, S) = ~2 GB at L=2048, H=64, D=64, S=128 in float32.
    # This is the dominant scratchpad contributor. Moving inside the loop does NOT
    # help because XLA unrolls the chunk loop, creating n_chunks separate allocations
    # that are all live simultaneously. A single allocation with slicing is cheaper.
    dB = dt_processed.unsqueeze(-1) * B  # (B, L, H, S)
    dBx = dB.unsqueeze(3) * x.unsqueeze(-1)  # (B, L, H, D, S)

    # Causal mask for intra-chunk scan (reused for all chunks)
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=x.dtype)
    )

    # Allocate output
    y = torch.zeros_like(x)  # (B, L, H, D)

    # Initial state: zeros
    state = torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        ssm_state_size,
        device=device,
        dtype=x.dtype,
    )

    for chunk_idx in range(n_chunks):
        t0 = chunk_idx * chunk_size
        t1 = t0 + chunk_size

        # Slice chunk data
        dA_log_chunk = dA_log[:, t0:t1, :]  # (B, Q, H)
        dBx_chunk = dBx[:, t0:t1, :, :, :]  # (B, Q, H, D, S)
        C_chunk = C[:, t0:t1, :, :]  # (B, Q, H, S)
        x_chunk = x[:, t0:t1, :, :]  # (B, Q, H, D)

        # --- Intra-chunk quadratic scan ---
        # Same as the full quadratic scan but only chunk_size x chunk_size
        log_dA_cumsum_chunk = torch.cumsum(dA_log_chunk, dim=1)  # (B, Q, H)

        log_diff = log_dA_cumsum_chunk.unsqueeze(2) - log_dA_cumsum_chunk.unsqueeze(
            1
        )  # (B, Q, Q, H)
        log_diff = log_diff.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(-1) == 0, -1e9
        )
        weights = torch.exp(log_diff)  # (B, Q, Q, H)

        # states_intra: (B, Q, H, D, S) — intra-chunk states at each position
        states_intra = torch.einsum("btih,bihds->bthds", weights, dBx_chunk)

        # --- Inter-chunk state contribution ---
        # Decay the incoming state by cumulative dA within this chunk.
        # At position t within chunk: decay = exp(sum of dA_log from 0..t)
        # = exp(log_dA_cumsum_chunk[:, t, :])
        decay = torch.exp(log_dA_cumsum_chunk)  # (B, Q, H)

        # Inter-chunk contribution to output:
        # y_inter[b, t, h, d] = sum_s C[b,t,h,s] * decay[b,t,h] * state[b,h,d,s]
        # = decay[b,t,h] * (C[b,t,h,:] @ state[b,h,d,:].T)
        # state: (B, H, D, S), C_chunk: (B, Q, H, S)
        y_inter = torch.einsum("bths,bhds->bthd", C_chunk, state)  # (B, Q, H, D)
        y_inter = y_inter * decay.unsqueeze(-1)  # scale by cumulative decay

        # Combine intra + inter + D*x
        y_intra = torch.einsum("blhs,blhds->blhd", C_chunk, states_intra)
        y_chunk = y_intra + y_inter + D_param.view(1, 1, -1, 1) * x_chunk

        y[:, t0:t1, :, :] = y_chunk

        # --- Update state for next chunk ---
        # The state after this chunk is:
        # state_new = decay_total * state + states_intra[:, -1, :, :, :]
        # where decay_total = exp(sum of all dA_log in this chunk)
        decay_total = torch.exp(
            dA_log_chunk.sum(dim=1)
        )  # (B, H) — total decay over chunk
        state = (
            decay_total.unsqueeze(-1).unsqueeze(-1) * state
            + states_intra[:, -1, :, :, :]
        )

    # Final state: for padding_mask handling, we'd need to track which chunk
    # contains the last real token. Since NxDI pads all sequences to
    # max_context_length and padding tokens have dt=0 (zeroed out earlier),
    # the state propagation naturally handles this — padding tokens contribute
    # nothing to the state update.
    final_state = state.contiguous()

    return y, final_state


# ==============================================================================
# Pattern decoder
# ==============================================================================

PATTERN_MAP = {"M": "mamba", "-": "mlp", "*": "attention"}


def decode_block_pattern(pattern: str) -> list:
    """Convert hybrid_override_pattern string to list of block types.

    4B pattern grammar: M=Mamba-2, `-`=dense MLP, `*`=attention.
    Example: "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-" -> 42 layers.
    """
    return [PATTERN_MAP[c] for c in pattern if c in PATTERN_MAP]


# ==============================================================================
# Activation
# ==============================================================================


def relu2(x: torch.Tensor) -> torch.Tensor:
    """ReLU squared: relu(x)^2. Nemotron's MLP activation (mlp_hidden_act=relu2)."""
    return torch.relu(x).square()


# ==============================================================================
# RMSNorm helpers
# ==============================================================================


def get_rmsnorm_cls():
    """Return appropriate RMSNorm implementation (CPU or Neuron)."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def _gated_rmsnorm_with_weight(hidden_states, gate, weight, group_size, eps):
    """Gated RMSNorm with explicit weight tensor (for per-rank TP slicing).

    Matches HF MambaRMSNormGated with `norm_before_gate=False`, which in
    mamba-ssm's rmsnorm_fn means: y = RMSNorm(x * SiLU(z)) * w.

    The gate is applied to x BEFORE variance is computed, NOT after normalization.
    Previous versions of this file applied SiLU(gate) *after* normalization, which
    was a bug that produced ~82% argmax match with the HF PyTorch fallback
    (which had the same bug), but produced fundamentally wrong output vs
    the true CUDA reference. See https://github.com/state-spaces/mamba,
    layernorm_gated.py: `if z is not None and not norm_before_gate: x = x * F.silu(z)`.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    # Apply gate BEFORE normalization (norm_before_gate=False semantics)
    if gate is not None:
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

    orig_shape = hidden_states.shape
    num_groups = orig_shape[-1] // group_size

    hidden_states = hidden_states.view(*orig_shape[:-1], num_groups, group_size)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = hidden_states.view(*orig_shape)

    hidden_states = (weight * hidden_states).to(input_dtype)
    return hidden_states


class NemotronRMSNormGated(nn.Module):
    """Gated RMSNorm: y = RMSNorm(x * SiLU(gate)) * weight.

    Matches HF MambaRMSNormGated with `norm_before_gate=False`, which in
    mamba-ssm's `rmsnorm_fn` means: **the gate is applied to x BEFORE variance
    is computed**, NOT `RMSNorm(x) * SiLU(gate)` as the name might suggest.

    Reference (state-spaces/mamba, layernorm_gated.py):
        if z is not None and not norm_before_gate:
            x = x * F.silu(z)
        rstd = 1 / sqrt(x.square().mean(-1) + eps)
        out = x * rstd * weight

    CRITICAL: Uses group_size for per-group normalization. The CUDA rmsnorm_fn
    normalizes per-group (groups of group_size elements), NOT over the full dim.
    For Nemotron: intermediate_size=4096, n_groups=8, group_size=512.
    This means 8 independent RMSNorms, each over 512 elements.
    """

    def __init__(self, hidden_size, group_size=None, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size if group_size is not None else hidden_size

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Apply gate BEFORE normalization (norm_before_gate=False semantics)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        # Group-wise RMSNorm: reshape into groups, normalize each independently
        orig_shape = hidden_states.shape
        group_size = self.group_size
        num_groups = orig_shape[-1] // group_size

        hidden_states = hidden_states.view(*orig_shape[:-1], num_groups, group_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.view(*orig_shape)

        hidden_states = (self.weight * hidden_states).to(input_dtype)
        return hidden_states


# ==============================================================================
# Config
# ==============================================================================


class NemotronHInferenceConfig(InferenceConfig):
    """Configuration class for Nemotron-3-Nano-4B inference (dense variant)."""

    output_attentions = False
    output_hidden_states = False
    use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        return [
            # Core model
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "tie_word_embeddings",
            # Mamba-2
            "mamba_num_heads",
            "mamba_head_dim",
            "ssm_state_size",
            "n_groups",
            "conv_kernel",
            # Dense MLP
            "intermediate_size",
            # GQA Attention
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            # Layer pattern
            "hybrid_override_pattern",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronConfig

    def add_derived_config(self):
        """Add derived attributes after HF config is loaded."""
        super().add_derived_config()

        # Decode block pattern
        if hasattr(self, "hybrid_override_pattern"):
            self.layer_types = decode_block_pattern(self.hybrid_override_pattern)
            assert len(self.layer_types) == self.num_hidden_layers, (
                f"Pattern gives {len(self.layer_types)} layers but config says "
                f"{self.num_hidden_layers}"
            )

        # Defaults for optional attributes
        if not hasattr(self, "expand"):
            self.expand = 2
        if not hasattr(self, "chunk_size"):
            self.chunk_size = 256  # 4B uses 256 (30B uses 128)
        if not hasattr(self, "use_conv_bias"):
            self.use_conv_bias = True
        if not hasattr(self, "mamba_proj_bias"):
            self.mamba_proj_bias = False
        if not hasattr(self, "mlp_bias"):
            self.mlp_bias = False
        if not hasattr(self, "attention_bias"):
            self.attention_bias = False
        # 4B config uses `rms_norm_eps` (30B uses `norm_eps`). Support both:
        # copy whichever is present into both attribute names so downstream code
        # can read either.
        if not hasattr(self, "norm_eps"):
            self.norm_eps = getattr(
                self, "rms_norm_eps",
                getattr(self, "layer_norm_epsilon", 1e-5),
            )
        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = self.norm_eps
        # 4B config does not set `rope_theta` explicitly; HF default is 10000.
        if not hasattr(self, "rope_theta"):
            self.rope_theta = 10000.0


# ==============================================================================
# Mamba-2 Layer
# ==============================================================================


class NeuronNemotronMamba2Layer(nn.Module):
    """
    Mamba-2 layer for Nemotron, with NxDI TP support.

    TP sharding strategy (HEAD-SHARDED SSM via split projections):
    - gate_proj, xBC_proj, dt_proj: three separate ColumnParallelLinear
      (gather_output=False) modules. preshard_hook splits the checkpoint's
      in_proj.weight into these three, with xBC columns reordered for TP
      alignment. Each rank's ColumnParallelLinear produces per-rank output
      directly without any post-projection slicing.
    - SSM computation runs on num_heads/TP heads per rank.
    - SSM params (dt_bias, A_log, D, conv weights, norm) sharded at load time
      via preshard_hook — no runtime rank-based indexing needed.
    - out_proj: RowParallelLinear(input_is_parallel=True) — takes per-rank
      intermediate_size/TP input, produces hidden_size output with all-reduce.

    This reduces the dominant dBx scratchpad from 2 GB/layer (64 heads) to
    ~250 MB/layer (16 heads at TP=4), enabling CTE>2048.

    NKI O(L) SSD kernel or O(L^2) quadratic fallback. O(1) decode.
    Manual depthwise conv1d (TEN404 workaround).
    External state buffers for input_output_aliases persistence.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.mamba_head_dim
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.chunk_size = getattr(config, "chunk_size", 128)

        # FULL dimensions (before TP sharding)
        self.num_heads_full = config.mamba_num_heads  # 64
        self.n_groups_full = config.n_groups  # 8

        # TP degree
        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1

        # Per-rank dimensions (sharded along heads)
        assert self.num_heads_full % self.tp_degree == 0, (
            f"num_heads={self.num_heads_full} not divisible by tp_degree={self.tp_degree}"
        )
        assert self.n_groups_full % self.tp_degree == 0, (
            f"n_groups={self.n_groups_full} not divisible by tp_degree={self.tp_degree}"
        )
        self.num_heads = self.num_heads_full // self.tp_degree  # 64/4=16
        self.n_groups = self.n_groups_full // self.tp_degree  # 8/4=2
        self.intermediate_size = self.num_heads * self.head_dim  # 16*64=1024
        self.groups_time_state_size = self.n_groups * self.ssm_state_size  # 2*128=256
        self.conv_dim = self.intermediate_size + 2 * self.groups_time_state_size  # 1536

        # Full dimensions for projections
        self.intermediate_size_full = self.num_heads_full * self.head_dim  # 4096
        self.groups_time_state_size_full = (
            self.n_groups_full * self.ssm_state_size
        )  # 1024
        self.conv_dim_full = (
            self.intermediate_size_full + 2 * self.groups_time_state_size_full
        )  # 6144
        projection_size_full = (
            self.intermediate_size_full + self.conv_dim_full + self.num_heads_full
        )  # 10304

        # Split in_proj into 3 separate ColumnParallelLinear to avoid
        # neuronx-cc ICE (NCC_IBIR182) triggered by slicing a single
        # ColumnParallelLinear(gather_output=False) output of size 2576.
        # preshard_hook splits the checkpoint's in_proj.weight into these 3.
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size_full,  # 4096
                bias=config.mamba_proj_bias,
                gather_output=False,
            )
            self.xBC_proj = ColumnParallelLinear(
                self.hidden_size,
                self.conv_dim_full,  # 6144
                bias=config.mamba_proj_bias,
                gather_output=False,
            )
            self.dt_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads_full,  # 64
                bias=config.mamba_proj_bias,
                gather_output=False,
            )
            # out_proj: RowParallelLinear — input is per-rank intermediate_size,
            # output is hidden_size, with all-reduce across TP ranks.
            self.out_proj = RowParallelLinear(
                self.intermediate_size_full,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
            )
        else:
            self.gate_proj = nn.Linear(
                self.hidden_size,
                self.intermediate_size_full,
                bias=config.mamba_proj_bias,
            )
            self.xBC_proj = nn.Linear(
                self.hidden_size, self.conv_dim_full, bias=config.mamba_proj_bias
            )
            self.dt_proj = nn.Linear(
                self.hidden_size, self.num_heads_full, bias=config.mamba_proj_bias
            )
            self.out_proj = nn.Linear(
                self.intermediate_size_full, self.hidden_size, bias=False
            )

        # Manual depthwise conv1d (TEN404 workaround) — per-rank conv_dim.
        # Checkpoint has full-size tensors; shard_children shards per-rank.
        self.conv_weight = nn.Parameter(
            torch.randn(self.conv_dim, self.conv_kernel_size)
        )
        if getattr(config, "use_conv_bias", True):
            self.conv_bias = nn.Parameter(torch.zeros(self.conv_dim))
        else:
            self.conv_bias = None

        # Mark conv params for TP sharding by shard_children (dim=0 split).
        # NeuronNemotronMamba2Layer is registered as a supported sharded module
        # so shard_children will process these parameters per-rank.
        for param in [self.conv_weight] + (
            [self.conv_bias] if self.conv_bias is not None else []
        ):
            param.tensor_model_parallel = True
            param.partition_dim = 0
            param.num_partitions = self.tp_degree
            param.partition_stride = 1

        # SSM parameters — per-rank num_heads. shard_children shards per-rank.
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.num_heads))

        for param in [self.dt_bias, self.A_log, self.D]:
            param.tensor_model_parallel = True
            param.partition_dim = 0
            param.num_partitions = self.tp_degree
            param.partition_stride = 1

        # Gated RMSNorm — per-rank intermediate_size. shard_children shards weight.
        group_size = self.intermediate_size // self.n_groups  # 512 (same as full)
        self.norm = NemotronRMSNormGated(
            self.intermediate_size,
            group_size=group_size if self.n_groups > 0 else self.intermediate_size,
            eps=getattr(config, "norm_eps", 1e-5),
        )
        # Mark norm weight for TP sharding
        self.norm.weight.tensor_model_parallel = True
        self.norm.weight.partition_dim = 0
        self.norm.weight.num_partitions = self.tp_degree
        self.norm.weight.partition_stride = 1

        self.time_step_limit = (0.0, float("inf"))

        # Per-rank projection output size: gate + xBC + dt per rank
        # = intermediate_size + conv_dim + num_heads = 1024 + 1536 + 16 = 2576
        self.projection_size_per_rank = (
            self.intermediate_size + self.conv_dim + self.num_heads
        )

    def preshard_hook(self, model_state_dict, prefix):
        """Transform checkpoint layout for TP-sharded Mamba (rank-independent).

        This hook is called ONCE by invoke_preshard_hook (not per-rank).
        It performs rank-independent transformations only:
        1. Splits in_proj.weight/bias into gate_proj, xBC_proj, dt_proj
        2. Reorders xBC to TP-interleaved layout

        Per-rank sharding of conv_weight, conv_bias, dt_bias, A_log, D,
        and norm.weight is handled by shard_children via tensor_model_parallel
        annotations on the parameters.

        invoke_preshard_hook stops recursing into children when it finds
        a preshard_hook, so we manually forward to children's hooks.
        """
        if self.tp_degree <= 1:
            return True

        # Strip trailing "weight"/"bias" that invoke_preshard_hook appends
        if prefix.endswith("weight"):
            module_prefix = prefix[: -len("weight")]
        elif prefix.endswith("bias"):
            module_prefix = prefix[: -len("bias")]
        else:
            module_prefix = prefix

        tp = self.tp_degree

        # ---------------------------------------------------------------
        # Split checkpoint in_proj.weight/bias into gate_proj, xBC_proj, dt_proj.
        # HF layout: [gate(4096) | x(4096) | B(1024) | C(1024) | dt(64)] = 10304
        # gate_proj gets gate(4096), xBC_proj gets [x|B|C](6144), dt_proj gets dt(64).
        # ColumnParallelLinear's shard_children will then shard each per-rank.
        # ---------------------------------------------------------------
        for suffix in ["weight", "bias"]:
            in_proj_key = module_prefix + "in_proj." + suffix
            if in_proj_key not in model_state_dict:
                continue
            tensor = model_state_dict[in_proj_key]
            # ColumnParallelLinear weight: (output_size, input_size) — split along dim 0
            dim = 0

            sz = tensor.shape[dim]
            assert (
                sz
                == self.intermediate_size_full
                + self.conv_dim_full
                + self.num_heads_full
            ), (
                f"in_proj size mismatch: {sz} != {self.intermediate_size_full + self.conv_dim_full + self.num_heads_full}"
            )

            # Extract gate(4096), xBC(6144 = x+B+C), dt(64) from HF layout
            gate_w = tensor.narrow(dim, 0, self.intermediate_size_full)  # 4096
            x_w = tensor.narrow(
                dim, self.intermediate_size_full, self.intermediate_size_full
            )  # 4096
            B_w = tensor.narrow(
                dim, 2 * self.intermediate_size_full, self.groups_time_state_size_full
            )  # 1024
            C_w = tensor.narrow(
                dim,
                2 * self.intermediate_size_full + self.groups_time_state_size_full,
                self.groups_time_state_size_full,
            )  # 1024
            dt_w = tensor.narrow(
                dim, sz - self.num_heads_full, self.num_heads_full
            )  # 64

            # xBC needs TP-interleaved layout: for each rank, [x_r | B_r | C_r]
            # so that ColumnParallelLinear's column split gives each rank correct data.
            x_chunks = x_w.chunk(tp, dim=dim)
            B_chunks = B_w.chunk(tp, dim=dim)
            C_chunks = C_w.chunk(tp, dim=dim)
            xBC_reordered = []
            for r in range(tp):
                xBC_reordered.extend([x_chunks[r], B_chunks[r], C_chunks[r]])
            xBC_w = torch.cat(xBC_reordered, dim=dim)

            # Store as gate_proj, xBC_proj, dt_proj weights
            model_state_dict[module_prefix + "gate_proj." + suffix] = gate_w
            model_state_dict[module_prefix + "xBC_proj." + suffix] = xBC_w
            model_state_dict[module_prefix + "dt_proj." + suffix] = dt_w

            # Remove original in_proj key
            del model_state_dict[in_proj_key]

        # NOTE: conv_weight, conv_bias, dt_bias, A_log, D, and norm.weight
        # are NOT sharded here. They remain full-size in the checkpoint.
        # shard_children handles per-rank sharding via tensor_model_parallel
        # annotations set in __init__.

        # Forward to children's preshard_hooks (invoke_preshard_hook stops
        # recursing when it finds our preshard_hook, so we must do it manually).
        for name, child in self._modules.items():
            if child is not None and hasattr(child, "preshard_hook"):
                child_prefix = module_prefix + name + "."
                child.preshard_hook(model_state_dict, child_prefix + "weight")
                if getattr(child, "add_bias", False):
                    child.preshard_hook(model_state_dict, child_prefix + "bias")

        return True

    @staticmethod
    def get_state_shapes(config, batch_size=1):
        """Return (conv_state_shape, ssm_state_shape) for buffer allocation.

        Returns PER-RANK sizes (head-sharded TP pattern).
        Each rank holds state only for its shard of heads/groups.
        """
        if parallel_state.model_parallel_is_initialized():
            tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            tp_degree = 1

        num_heads = config.mamba_num_heads // tp_degree  # Per-rank: 64/4=16
        n_groups = config.n_groups // tp_degree  # Per-rank: 8/4=2
        intermediate_size = num_heads * config.mamba_head_dim  # 16*64=1024
        groups_time_state_size = n_groups * config.ssm_state_size  # 2*128=256
        conv_dim = intermediate_size + 2 * groups_time_state_size  # 1536
        conv_shape = (batch_size, conv_dim, config.conv_kernel - 1)
        ssm_shape = (
            batch_size,
            num_heads,
            config.mamba_head_dim,
            config.ssm_state_size,
        )
        return conv_shape, ssm_shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mamba_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Forward pass with external state and TP-sharded SSM computation.

        Three separate ColumnParallelLinear projections (gate_proj, xBC_proj,
        dt_proj) each produce per-rank output directly. SSM runs on per-rank
        heads, and out_proj (RowParallelLinear) does the all-reduce.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mamba_state: (conv_state, ssm_state) from persistence buffers

        Returns:
            output: (batch, seq_len, hidden_size)
            present_key_value: dummy (K, V) tuple for KV cache compatibility
            updated_mamba_state: (conv_state, ssm_state) for persistence
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        if mamba_state is not None:
            conv_state, ssm_state = mamba_state
        else:
            conv_state = torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                device=hidden_states.device,
                dtype=dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )

        # Use the padding mask passed from NeuronNemotronModel.forward().
        padding_mask = kwargs.get("padding_mask", None)

        # Mask hidden_states BEFORE projection (Falcon-H1 pattern)
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :seq_len, None].to(
                hidden_states.dtype
            )

        # Three separate projections — each ColumnParallelLinear(gather_output=False)
        # produces per-rank output directly, avoiding slice/deconcat ICE.
        gate = self.gate_proj(hidden_states)  # (B,L,1024) per rank
        hidden_states_B_C = self.xBC_proj(hidden_states)  # (B,L,1536) per rank
        dt = self.dt_proj(hidden_states)  # (B,L,16) per rank

        if seq_len > 1:
            output, conv_state_new, ssm_state_new = self._forward_prefill(
                hidden_states_B_C, gate, dt, batch_size, seq_len, dtype, padding_mask
            )
            # Dummy dependency for XLA aliasing preservation
            conv_state_new = conv_state_new + conv_state * 0
            ssm_state_new = ssm_state_new + ssm_state * 0
        else:
            output, conv_state_new, ssm_state_new = self._forward_decode(
                hidden_states_B_C, gate, dt, batch_size, dtype, conv_state, ssm_state
            )

        # Dummy KV cache for compatibility with attention-based generation loop
        dummy_k = torch.zeros(
            batch_size, 1, 1, 1, dtype=output.dtype, device=output.device
        )
        dummy_v = torch.zeros(
            batch_size, 1, 1, 1, dtype=output.dtype, device=output.device
        )

        return (output, (dummy_k, dummy_v), (conv_state_new, ssm_state_new))

    def _forward_prefill(
        self, hidden_states_B_C, gate, dt, batch_size, seq_len, dtype, padding_mask=None
    ):
        """Selective scan for prefill with TP-sharded SSM computation.

        All inputs (hidden_states_B_C, gate, dt) are already per-rank slices.
        Conv weights and SSM params are pre-sharded by preshard_hook.

        Uses O(L) NKI SSD kernel or O(L^2) quadratic parallel scan as fallback.
        """
        # Parameters are already per-rank from preshard_hook
        conv_weight = self.conv_weight
        conv_bias = self.conv_bias
        dt_bias = self.dt_bias
        A = -torch.exp(self.A_log.float())
        D = self.D

        # Depthwise conv1d with per-rank conv weights.
        # Two code paths (toggled by USE_NATIVE_CONV1D env var):
        #   - Manual weight loop (default): originally added as a Granite4-era
        #     workaround for compiler crash TEN404 on SDK 2.28.
        #   - F.conv1d(groups=conv_dim): native depthwise conv. If SDK 2.31 no
        #     longer crashes, this is mathematically equivalent and may compile
        #     to more efficient code.
        if USE_NATIVE_CONV1D:
            # F.conv1d expects (B, C, L) with weight (C_out, C_in/groups, K).
            # For depthwise: groups=conv_dim, C_in/groups=1, weight shape (conv_dim, 1, K).
            x_bcl = hidden_states_B_C.transpose(1, 2)  # (B, conv_dim, seq_len)
            w = conv_weight.unsqueeze(1)  # (conv_dim, 1, K)
            conv_out = F.conv1d(
                x_bcl,
                w,
                bias=conv_bias,
                padding=self.conv_kernel_size - 1,
                groups=self.conv_dim,
            )
            # Output has length seq_len + K - 1; trim to seq_len from the LEFT
            # (keeping outputs 0..seq_len-1) to match causal-conv1d semantics.
            hidden_states_conv = conv_out[..., :seq_len].transpose(1, 2).contiguous()
        else:
            # Manual depthwise conv1d loop (Granite4 workaround for TEN404 on SDK 2.28).
            padded = F.pad(
                hidden_states_B_C, (0, 0, self.conv_kernel_size - 1, 0), value=0.0
            )
            hidden_states_conv = torch.zeros_like(hidden_states_B_C)
            for k in range(self.conv_kernel_size):
                hidden_states_conv = hidden_states_conv + (
                    padded[:, k : k + seq_len, :]
                    * conv_weight[:, k].unsqueeze(0).unsqueeze(0)
                )
            if conv_bias is not None:
                hidden_states_conv = hidden_states_conv + conv_bias.unsqueeze(0).unsqueeze(
                    0
                )

        # Save conv_state from last K-1 REAL token positions (Granite4 pattern)
        if padding_mask is not None and seq_len >= self.conv_kernel_size - 1:
            real_len = padding_mask[:, :seq_len].sum(dim=1, keepdim=True).long()
            K_minus_1 = self.conv_kernel_size - 1
            offsets = torch.arange(
                K_minus_1, device=hidden_states_B_C.device
            ).unsqueeze(0)
            gather_idx = (real_len - K_minus_1 + offsets).clamp(min=0)
            gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, self.conv_dim)
            conv_state_seq = torch.gather(hidden_states_B_C, 1, gather_idx_expanded)
            conv_state_new = conv_state_seq.transpose(1, 2).contiguous()
        elif seq_len >= self.conv_kernel_size - 1:
            conv_state_new = (
                hidden_states_B_C[:, -(self.conv_kernel_size - 1) :, :]
                .transpose(1, 2)
                .contiguous()
            )
        else:
            pad_len = self.conv_kernel_size - 1 - seq_len
            conv_state_new = F.pad(
                hidden_states_B_C.transpose(1, 2), (pad_len, 0), value=0.0
            ).contiguous()

        hidden_states_conv = F.silu(hidden_states_conv)

        # Zero out conv output at padding positions (Granite4 pattern)
        if padding_mask is not None:
            hidden_states_conv = hidden_states_conv * padding_mask[
                :, :seq_len, None
            ].to(hidden_states_conv.dtype)

        # Split into x, B, C — using per-rank sizes
        x = hidden_states_conv[..., : self.intermediate_size]
        B = hidden_states_conv[
            ...,
            self.intermediate_size : self.intermediate_size
            + self.groups_time_state_size,
        ]
        C = hidden_states_conv[..., -self.groups_time_state_size :]

        # SSM computation in float32 — using per-rank params
        dt_processed = F.softplus(dt + dt_bias)
        dt_processed = torch.clamp(dt_processed, self.time_step_limit[0], 1e6)

        # Zero dt at padding positions (prevents SSM from processing padding)
        if padding_mask is not None:
            dt_processed = dt_processed * padding_mask[:, :seq_len, None].to(
                dt_processed.dtype
            )

        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()

        if USE_SSD_SCAN:
            # SSD chunk-based O(L) scan using NKI TensorE matmuls.
            # B/C stay as (B, L, n_groups, S) — NOT expanded via repeat_interleave.
            # The SSD kernel handles group→head mapping internally with nested loops.
            # Memory is O(chunk^2) per chunk, enabling ctx=768+.
            y, ssm_state_new = _ssd_prefill_scan(
                x,
                dt_processed,
                A,
                B,
                C,
                D.float(),
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
                self.n_groups,
                chunk_size=self.chunk_size,
            )
            if padding_mask is not None:
                pass
        elif USE_CHUNKED_NKI_SCAN:
            # Chunked NKI SSD scan (DGE-OOB survivor, Phase 3).
            # Intra-chunk O(Q^2) quadratic + inter-chunk O(1) recurrent state propagation.
            # Q=128=P_MAX. Uses only cheap NKI ops (dma_copy, nc_matmul, nc_transpose,
            # tensor_copy, tensor_scalar, tensor_tensor, activation, memset) -- proven to
            # survive the DGE budget on SDK 2.31 with 92+ invocations per NEFF.
            # Linear O(S) prefill, and state is only 4MB/layer -- no HBM ceiling.
            try:
                from .nki_kernels.chunked_ssd_wrapper import chunked_ssd_prefill_scan
            except ImportError:
                from nki_kernels.chunked_ssd_wrapper import chunked_ssd_prefill_scan
            # The chunked NKI SSD kernel requires chunk_size=128 (P_MAX / SBUF
            # partition width). The 4B config sets chunk_size=256 for the
            # reference SSD scan; we override to 128 for the NKI kernel path.
            # This is safe because chunk_size only affects the block structure
            # of the scan, not the mathematical output.
            y, ssm_state_new = chunked_ssd_prefill_scan(
                x,
                dt_processed,
                A,
                B,
                C,
                D.float(),
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
                self.n_groups,
                chunk_size=128,  # kernel constraint (see comment above)
                padding_mask=padding_mask,
            )
        elif USE_PYTORCH_SSD:
            # Pure PyTorch SSD: same chunk-based algorithm as NKI SSD but using
            # standard PyTorch ops. Avoids compiler DGE OOB issue.
            # B/C stay as (B, L, n_groups, S) — NOT expanded.
            # Memory is O(chunk^2 * G) per chunk iteration.
            y, ssm_state_new = _pytorch_ssd_scan(
                x,
                dt_processed,
                A,
                B,
                C,
                D.float(),
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
                self.n_groups,
                chunk_size=self.chunk_size,
                padding_mask=padding_mask,
            )
        else:
            # Non-SSD paths need B/C expanded to per-head
            B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
            C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)

        if (
            not USE_SSD_SCAN
            and not USE_PYTORCH_SSD
            and not USE_CHUNKED_NKI_SCAN
            and USE_CHUNKED_SCAN
            and seq_len >= self.chunk_size
        ):
            y, ssm_state_new = _chunked_quadratic_scan(
                x,
                dt_processed,
                A,
                B,
                C,
                D.float(),
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
                chunk_size=self.chunk_size,
                padding_mask=padding_mask,
            )
        elif not USE_SSD_SCAN and not USE_PYTORCH_SSD and not USE_CHUNKED_NKI_SCAN and USE_NKI_SCAN:
            y, ssm_state_new = _nki_selective_scan(
                x,
                dt_processed,
                A,
                B,
                C,
                D.float(),
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
            )
        elif not USE_SSD_SCAN and not USE_PYTORCH_SSD and not USE_CHUNKED_NKI_SCAN:
            # O(L^2) quadratic parallel scan — head-grouped
            G = SCAN_HEAD_GROUP
            n_groups_scan = self.num_heads // G
            assert self.num_heads % G == 0, (
                f"num_heads={self.num_heads} not divisible by SCAN_HEAD_GROUP={G}"
            )

            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype)
            )

            y_parts = []
            ssm_state_parts = []

            for g in range(n_groups_scan):
                h_start = g * G
                h_end = h_start + G

                x_g = x[:, :, h_start:h_end, :]
                B_g = B[:, :, h_start:h_end, :]
                C_g = C[:, :, h_start:h_end, :]
                dt_g = dt_processed[:, :, h_start:h_end]
                A_g = A[h_start:h_end]

                dA_log_g = dt_g * A_g.view(1, 1, -1)
                dB_g = dt_g.unsqueeze(-1) * B_g
                dBx_g = dB_g.unsqueeze(3) * x_g.unsqueeze(-1)

                log_dA_cumsum_g = torch.cumsum(dA_log_g, dim=1)

                log_diff_g = log_dA_cumsum_g.unsqueeze(2) - log_dA_cumsum_g.unsqueeze(1)
                log_diff_g = log_diff_g.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(-1) == 0, -1e9
                )
                weights_g = torch.exp(log_diff_g)

                states_g = torch.einsum("btih,bihds->bthds", weights_g, dBx_g)

                if padding_mask is not None:
                    real_len = padding_mask[:, :seq_len].sum(dim=1, keepdim=True).long()
                    last_real_idx = (real_len - 1).clamp(min=0)
                    gather_idx = last_real_idx.view(batch_size, 1, 1, 1, 1).expand(
                        -1, -1, G, self.head_dim, self.ssm_state_size
                    )
                    ssm_state_g = torch.gather(states_g, 1, gather_idx).squeeze(1)
                else:
                    ssm_state_g = states_g[:, -1, :, :, :]

                y_g = torch.einsum("blhs,blhds->blhd", C_g, states_g)
                D_g = D[h_start:h_end].view(1, 1, -1, 1)
                y_g = y_g + D_g * x_g

                y_parts.append(y_g)
                ssm_state_parts.append(ssm_state_g)

            y = torch.cat(y_parts, dim=2)
            ssm_state_new = torch.cat(ssm_state_parts, dim=1).contiguous()

        y = y.reshape(batch_size, seq_len, -1)

        # Gated RMSNorm — norm.weight pre-sharded by preshard_hook
        scan_output = self.norm(y, gate)

        # out_proj: RowParallelLinear(input_is_parallel=True) handles TP all-reduce
        output = self.out_proj(scan_output.to(dtype))

        return output, conv_state_new, ssm_state_new.to(dtype)

    def _forward_decode(
        self, hidden_states_B_C, gate, dt, batch_size, dtype, conv_state, ssm_state
    ):
        """O(1) recurrence for single-token decode with TP-sharded SSM."""
        # Parameters are already per-rank from preshard_hook
        conv_weight = self.conv_weight
        conv_bias = self.conv_bias
        dt_bias = self.dt_bias
        A = -torch.exp(self.A_log.float())
        D = self.D

        xBC_new = hidden_states_B_C.squeeze(1)  # (B, conv_dim)
        xBC_new_t = xBC_new.unsqueeze(2)  # (B, conv_dim, 1)
        conv_input = torch.cat(
            [conv_state, xBC_new_t], dim=2
        )  # (B, conv_dim, conv_kernel)

        conv_out = (conv_input * conv_weight.unsqueeze(0)).sum(dim=2)
        if conv_bias is not None:
            conv_out = conv_out + conv_bias

        conv_state_new = conv_input[:, :, 1:].contiguous()
        conv_out = F.silu(conv_out)

        x = conv_out[..., : self.intermediate_size]
        B = conv_out[
            ...,
            self.intermediate_size : self.intermediate_size
            + self.groups_time_state_size,
        ]
        C = conv_out[..., -self.groups_time_state_size :]

        dt_processed = F.softplus(dt.squeeze(1) + dt_bias)
        dt_processed = torch.clamp(dt_processed, self.time_step_limit[0], 1e6)

        x = x.reshape(batch_size, self.num_heads, self.head_dim).float()
        B = B.reshape(batch_size, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, self.n_groups, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=1)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=1)

        dA = torch.exp(dt_processed * A.view(1, -1))
        dB = dt_processed.unsqueeze(-1) * B
        dBx = dB.unsqueeze(2) * x.unsqueeze(-1)
        ssm_state_new = dA.unsqueeze(-1).unsqueeze(-1) * ssm_state.float() + dBx

        y = torch.einsum("bhds,bhs->bhd", ssm_state_new, C)
        y = y + D.view(1, -1, 1) * x
        y = y.reshape(batch_size, -1)

        gate_squeezed = gate.squeeze(1)

        # Gated RMSNorm — norm.weight pre-sharded by preshard_hook
        scan_output = self.norm(y, gate_squeezed)

        if len(scan_output.shape) == 2:
            scan_output = scan_output.unsqueeze(1)
        # out_proj: RowParallelLinear(input_is_parallel=True) handles TP all-reduce
        output = self.out_proj(scan_output.to(dtype))

        return output, conv_state_new, ssm_state_new.to(dtype)


# ==============================================================================
# Register custom modules for shard_children TP sharding
# ==============================================================================
# shard_children only processes modules in __SUPPORTED_SHARDED_MODULES.
# NeuronNemotronMamba2Layer and NemotronRMSNormGated have parameters
# marked with tensor_model_parallel=True that need per-rank sharding.
# Monkeypatch the supported modules tuple to include them.
try:
    import neuronx_distributed.trace.trace as _trace_module

    _trace_module.__SUPPORTED_SHARDED_MODULES = (
        _trace_module.__SUPPORTED_SHARDED_MODULES
        + (NeuronNemotronMamba2Layer, NemotronRMSNormGated)
    )
    logger.info(
        "Registered NeuronNemotronMamba2Layer and NemotronRMSNormGated "
        "as supported sharded modules for shard_children"
    )
except Exception as e:
    logger.warning(f"Failed to register custom sharded modules: {e}")


# ==============================================================================
# Dense MLP Layer (4B variant)
# ==============================================================================


class NeuronNemotronMLP(nn.Module):
    """
    Dense MLP for Nemotron-3-Nano-4B: up_proj -> relu2 -> down_proj.

    HF layout:
        mlp.up_proj:   Linear(hidden_size, intermediate_size, bias=False)
        mlp.down_proj: Linear(intermediate_size, hidden_size, bias=False)
        activation:    relu2 (relu(x)^2)

    NxDI TP layout (matches the standard Llama-style MLP):
        up_proj:   ColumnParallelLinear(H -> I/TP)   -- weight shard along output dim
        down_proj: RowParallelLinear(I/TP -> H)      -- weight shard along input dim
        activation is applied on the (per-rank) intermediate, then RowParallel
        performs an all-reduce inside its forward.

    Note: the 4B variant does NOT use a SwiGLU-style gate. It is a plain
    two-layer MLP with relu2 activation (per config `mlp_hidden_act="relu2"`).

    Returns (hidden_states, dummy_kv, None) to match the interface expected by
    NeuronNemotronDecoderLayer (which unpacks a 3-tuple like MoE/Mamba do).
    """

    def __init__(self, config: NemotronHInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # dtype / init defaults inherited from the surrounding module setup
        dtype = getattr(config.neuron_config, "torch_dtype", torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
            gather_output=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=getattr(config, "mlp_bias", False),
            input_is_parallel=True,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        # up_proj is ColumnParallel(gather_output=False), so its output is
        # already sharded along the intermediate dim on this rank.
        h = self.up_proj(hidden_states)
        h = relu2(h)
        # down_proj is RowParallel(input_is_parallel=True), which performs the
        # all-reduce internally.
        result = self.down_proj(h)

        # Dummy KV to satisfy the decoder layer's uniform unpack.
        batch_size = hidden_states.shape[0]
        dummy_k = torch.zeros(
            batch_size, 1, 1, 1, dtype=result.dtype, device=result.device
        )
        dummy_v = torch.zeros(
            batch_size, 1, 1, 1, dtype=result.dtype, device=result.device
        )
        return (result, (dummy_k, dummy_v), None)


# ==============================================================================
# GQA Attention Layer
# ==============================================================================


class NeuronNemotronAttention(NeuronAttentionBase):
    """
    GQA attention for Nemotron using NxDI's NeuronAttentionBase.

    32 Q heads, 2 KV heads (16:1 ratio), explicit head_dim=128.
    RoPE with theta=10000.
    """

    def __init__(self, config: NemotronHInferenceConfig, layer_idx: int):
        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            base=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,  # MUST be explicit 128, not hidden_size//num_heads=84
            rotary_emb=rotary_emb,
            rms_norm_eps=getattr(
                config, "rms_norm_eps", getattr(config, "norm_eps", 1e-5)
            ),
            use_qk_norm=False,
            qkv_bias=getattr(config, "attention_bias", False),
        )

        self.layer_idx = layer_idx

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronNemotronAttention must be initialized in a distributed env."
            )


# ==============================================================================
# Decoder Layer (unified wrapper)
# ==============================================================================


class NeuronNemotronDecoderLayer(nn.Module):
    """
    Single decoder layer. Contains ONE of: Mamba-2, dense MLP, or GQA Attention.
    Each layer has a pre-norm (RMSNorm) and residual connection.

    Returns the standard 6-tuple:
        (hidden_states, present_key_value, cos_cache, sin_cache, None, updated_mamba_state)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=getattr(config, "norm_eps", 1e-5),
        )

        if self.layer_type == "mamba":
            self.mixer = NeuronNemotronMamba2Layer(config, layer_idx)
        elif self.layer_type == "mlp":
            self.mixer = NeuronNemotronMLP(config, layer_idx)
        elif self.layer_type == "attention":
            self.self_attn = NeuronNemotronAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown block type: {self.layer_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mamba_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Returns 6-tuple matching Granite4 decoder layer interface:
            (hidden_states, present_key_value, cos_cache, sin_cache, None, updated_mamba_state)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        updated_mamba_state = None
        cos_cache, sin_cache = None, None

        if self.layer_type == "attention":
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
        elif self.layer_type == "mamba":
            hidden_states, present_key_value, updated_mamba_state = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                mamba_state=mamba_state,
                **kwargs,
            )
        elif self.layer_type == "mlp":
            hidden_states, present_key_value, _ = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        return (
            hidden_states,
            present_key_value,
            cos_cache,
            sin_cache,
            None,
            updated_mamba_state,
        )


# ==============================================================================
# Model Wrapper and DecoderModelInstance
# ==============================================================================


class NemotronDecoderModelInstance(DecoderModelInstance):
    """
    Extends DecoderModelInstance to alias Mamba state parameters.

    After calling super().get() which aliases the KV cache parameters,
    we add Mamba state parameters (conv_state, ssm_state for each Mamba layer)
    to the input_output_aliases dict.

    Output indices must match NeuronNemotronModel.forward() output order:
        [res, K0, V0, K1, V1, ..., conv_state_0, ssm_state_0, ...]
    """

    def get(self, bucket_rank, **kwargs):
        self.module, self.input_output_aliases = super().get(bucket_rank, **kwargs)

        past_key_values = self.module.kv_mgr.past_key_values
        mamba_states = self.module.mamba_states

        # Count where Mamba state outputs start
        num_output_from_trace = 1  # logits/tokens
        if getattr(self.module, "neuron_config", None) and getattr(
            self.module.neuron_config, "output_logits", False
        ):
            num_output_from_trace = 2
        num_output_from_trace += len(past_key_values)

        for i in range(len(mamba_states)):
            self.input_output_aliases[mamba_states[i]] = num_output_from_trace + i

        logger.info(
            f"NemotronDecoderModelInstance: aliased {len(past_key_values)} KV cache "
            f"entries and {len(mamba_states)} Mamba state entries "
            f"(Mamba starts at output index {num_output_from_trace})"
        )

        return self.module, self.input_output_aliases


class NemotronModelWrapper(ModelWrapper):
    """Custom ModelWrapper that returns NemotronDecoderModelInstance.

    Also post-processes compiler args to raise the HLO verifier instruction
    limit and disable strict verification.  The Mamba-2 quadratic scan
    generates ~6.86M HLO instructions at max_context_length>=4224, exceeding
    the default 5M verifier check (NCC_EVRF007).  Setting verify-hlo=false
    bypasses this verifier-only check; the backend's own limit is already
    raised to 15M via --internal-max-instruction-limit=15000000.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Raise the instruction count limit everywhere to handle the Mamba-2
        # quadratic scan's ~6.86M instructions at max_context_length>=4224.
        # Three checks must be bypassed:
        #   1. HLO verifier (NCC_EVRF007) -- via --tiled-inst-limit in hlo2tensorizer
        #   2. Backend limit (NCC_EBVF030) -- via --internal-max-instruction-limit
        #   3. Backend verifier -- via --internal-backend-options --max-instruction-limit
        if hasattr(self, "compiler_args") and self.compiler_args:
            self.compiler_args = self.compiler_args.replace(
                "--verify-hlo=true", "--verify-hlo=false --tiled-inst-limit=15000000"
            )

    def get_model_instance(self):
        return NemotronDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


# ==============================================================================
# Model Body
# ==============================================================================


class NeuronNemotronModel(NeuronBaseModel):
    """
    NeuronNemotronModel -- traced model body for Nemotron-3-Nano-4B.

    Overrides forward() and get_model_output() to handle:
    - Heterogeneous layers (Mamba/dense-MLP/Attention in single-block-per-layer)
    - Mamba state persistence alongside KV cache
    - Per-layer routing of inputs (only attention layers need KV cache)
    """

    def setup_attr_for_model(self, config: NemotronHInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: NemotronHInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        self.layers = nn.ModuleList(
            [
                NeuronNemotronDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = get_rmsnorm_cls()(
            self.hidden_size,
            eps=getattr(config, "norm_eps", 1e-5),
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )

        # Mamba state persistence buffers
        self._mamba_layer_indices = [
            i
            for i in range(config.num_hidden_layers)
            if config.layer_types[i] == "mamba"
        ]
        batch_size = config.neuron_config.batch_size
        conv_shape, ssm_shape = NeuronNemotronMamba2Layer.get_state_shapes(
            config, batch_size
        )
        dtype = config.neuron_config.torch_dtype

        self.mamba_states = nn.ParameterList()
        for _ in self._mamba_layer_indices:
            self.mamba_states.append(
                nn.Parameter(torch.zeros(conv_shape, dtype=dtype), requires_grad=False)
            )
            self.mamba_states.append(
                nn.Parameter(torch.zeros(ssm_shape, dtype=dtype), requires_grad=False)
            )

        logger.info(
            f"Initialized Mamba state persistence: {len(self._mamba_layer_indices)} layers, "
            f"{len(self.mamba_states)} buffers"
        )

    def _get_mamba_states(self):
        """Get Mamba states as list of (conv_state, ssm_state) tuples."""
        states = []
        for i in range(0, len(self.mamba_states), 2):
            states.append((self.mamba_states[i], self.mamba_states[i + 1]))
        return states

    def _build_mamba_state_map(self):
        """Map layer_idx -> mamba_idx for Mamba layers."""
        return {
            layer_idx: mamba_idx
            for mamba_idx, layer_idx in enumerate(self._mamba_layer_indices)
        }

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Thread Mamba state through decoder layers alongside KV cache.
        Returns: (hidden_states, next_decoder_cache, updated_mamba_state_list)
        """
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = inputs_embeds

        if self.sequence_parallel_enabled:
            self.validate_sequence_parallel(seq_length)
        hidden_states = self.process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, kwargs.get("active_block_table", None)
        )

        next_decoder_cache = ()
        cos_cache = None
        sin_cache = None
        cache_size = self.n_positions

        if not is_for_context_encoding or windowed_context_encoding_window_idx >= 1:
            past_key_values = self.kv_mgr.get_cache(
                seq_ids=seq_ids,
                seq_len=cache_size,
                is_for_context_encoding=is_for_context_encoding,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        mamba_states = self._get_mamba_states()
        mamba_state_map = self._build_mamba_state_map()
        updated_mamba_states = [None] * len(self._mamba_layer_indices)

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            mamba_state = None
            if idx in mamba_state_map:
                mamba_state = mamba_states[mamba_state_map[idx]]

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                kv_mgr=self.kv_mgr,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                local_mask=local_attn_mask,
                padding_mask=padding_mask,
                mamba_state=mamba_state,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:4]
            layer_mamba_state = layer_outputs[5]

            if idx in mamba_state_map and layer_mamba_state is not None:
                updated_mamba_states[mamba_state_map[idx]] = layer_mamba_state

        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return (hidden_states, next_decoder_cache, updated_mamba_states)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        """
        Traced forward -- appends Mamba state tensors to output list.
        Output: [res, K0, V0, ..., conv_state_0, ssm_state_0, ...]
        """
        prev_hidden = self.set_none_if_empty(prev_hidden)
        adapter_ids = self.set_none_if_empty(adapter_ids)
        accepted_indices = self.set_none_if_empty(accepted_indices)
        current_length = self.set_none_if_empty(current_length)
        medusa_mask = self.set_none_if_empty(medusa_mask)
        scatter_index = self.set_none_if_empty(scatter_index)
        slot_mapping = self.set_none_if_empty(slot_mapping)
        active_block_table = self.set_none_if_empty(active_block_table)
        num_queries = self.set_none_if_empty(num_queries)
        computed_context_lens = self.set_none_if_empty(computed_context_lens)
        tile_q_indices = self.set_none_if_empty(tile_q_indices)
        tile_block_tables = self.set_none_if_empty(tile_block_tables)
        tile_masks = self.set_none_if_empty(tile_masks)
        inputs_embeds = self.set_none_if_empty(inputs_embeds)
        kv_cache = self.set_none_if_empty(kv_cache)
        active_mask = self.set_none_if_empty(active_mask)
        rotary_position_id = self.set_none_if_empty(rotary_position_id)
        vision_embeddings = self.set_none_if_empty(vision_embeddings)
        vision_mask = self.set_none_if_empty(vision_mask)

        is_for_context_encoding = self._is_context_encoding(input_ids)

        if attention_mask is not None and is_for_context_encoding:
            padding_mask = attention_mask.float()
        else:
            padding_mask = None

        attn_mask = self.create_attn_mask(
            attention_mask,
            is_for_context_encoding,
            False,
            position_ids=position_ids,
        )

        hidden_states, updated_kv_cache, updated_mamba_states = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            is_for_context_encoding=is_for_context_encoding,
            scatter_index=slot_mapping
            if getattr(self, "is_block_kv_layout", False)
            else scatter_index,
            kvcache_buffer=kv_cache,
            update_cache=True,
            padding_mask=padding_mask,
        )

        batch_size = input_ids.shape[0]
        if not self.sliced_hidden:
            if not (
                position_ids.shape[-1] == getattr(self, "speculation_length", 0)
                or position_ids.shape[-1] == 1
            ):
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            from neuronx_distributed_inference.models.model_base import (
                mask_padded_logits,
            )

            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, False, is_for_context_encoding
            )
        else:
            res = logits

        outputs = [res]
        if getattr(self.neuron_config, "output_logits", False):
            from neuronx_distributed_inference.models.model_base import (
                _gather_along_dim,
                get_tp_group,
            )

            gathered_logits = _gather_along_dim(
                logits,
                partition_dim=2,
                process_group=get_tp_group(self.config),
            )
            outputs += [gathered_logits]
        outputs += updated_kv_cache

        # Append Mamba states for aliasing
        for conv_state, ssm_state in updated_mamba_states:
            outputs.append(conv_state)
            outputs.append(ssm_state)

        return outputs


# ==============================================================================
# CausalLM Wrapper
# ==============================================================================


class NeuronNemotronForCausalLM(NeuronBaseForCausalLM):
    """Top-level causal LM class for Nemotron-3-Nano-4B."""

    _model_cls = NeuronNemotronModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @classmethod
    def get_config_cls(cls):
        return NemotronHInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: NemotronHInferenceConfig
    ) -> dict:
        return _convert_nemotron_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        args = (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--model-type transformer -O1 "
            "--auto-cast=none "
            "--internal-max-instruction-limit=15000000 "
            "--internal-backend-options='--max-instruction-limit=15000000'"
        )

        # Scratchpad page size: controls HBM scratchpad allocation granularity.
        # Must be paired with NEURON_SCRATCHPAD_PAGE_SIZE env var at runtime.
        if getattr(self.neuron_config, "scratchpad_page_size", None):
            args += (
                f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"
            )

        # Spill-reload DGE: reduces DMA ring memory for long context.
        if getattr(self.neuron_config, "enable_spill_reload_dge", False):
            args += " --internal-enable-dge-levels spill_reload"

        return args

    def get_model_wrapper_cls(self):
        return NemotronModelWrapper

    def shard_weights(
        self, compiled_model_path, debug=False, pre_shard_weights_hook=None
    ):
        """
        Override to guard against multi-process weight loading OOM.

        The full Nemotron model is ~59 GB BF16. If accidentally run with
        torchrun (multi-process), each process would independently load
        the full model, totaling ~240 GB which exceeds 128 GB RAM.

        This override gates on RANK==0 so only one process loads and shards.
        For normal single-process usage (the intended pattern), this is a no-op.
        """
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            import time

            logger.info(f"Rank {rank}: skipping weight sharding (rank 0 only)")
            barrier_file = os.path.join(compiled_model_path, ".shard_complete")
            timeout = 3600
            elapsed = 0
            while not os.path.exists(barrier_file):
                time.sleep(5)
                elapsed += 5
                if elapsed >= timeout:
                    raise RuntimeError(f"Rank {rank}: timed out waiting for sharding")
            logger.info(f"Rank {rank}: sharding complete (waited {elapsed}s)")
            return

        logger.info("Rank 0: starting weight sharding")
        super().shard_weights(compiled_model_path, debug, pre_shard_weights_hook)
        barrier_file = os.path.join(compiled_model_path, ".shard_complete")
        with open(barrier_file, "w") as f:
            f.write("done")
        logger.info("Rank 0: weight sharding complete")

    def _copy_past_key_values(self, outputs):
        """Also copy Mamba states for CPU debugging path."""
        n_mamba_entries = len(self.context_encoding_model.model.mamba_states)

        if n_mamba_entries > 0:
            super()._copy_past_key_values(outputs[:-n_mamba_entries])

            mamba_outputs = outputs[-n_mamba_entries:]
            for i, state_tensor in enumerate(mamba_outputs):
                self.token_generation_model.model.mamba_states[i].data = state_tensor
                self.context_encoding_model.model.mamba_states[i].data = state_tensor
        else:
            super()._copy_past_key_values(outputs)


# ==============================================================================
# State Dict Conversion
# ==============================================================================


def _convert_mamba_conv_weights(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert HF conv1d.weight (3D) to our conv_weight (2D) parameter names."""
    converted = {}
    for key, tensor in state_dict.items():
        if "conv1d.weight" in key:
            new_key = key.replace("conv1d.weight", "conv_weight")
            converted[new_key] = tensor.squeeze(1)
        elif "conv1d.bias" in key:
            new_key = key.replace("conv1d.bias", "conv_bias")
            converted[new_key] = tensor
        else:
            converted[key] = tensor
    return converted


def _split_mamba_projections(
    state_dict: Dict[str, Any], config: NemotronHInferenceConfig
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """No-op: ColumnParallelLinear/RowParallelLinear handle TP sharding via preshard_hook.

    The in_proj uses ColumnParallelLinear(gather_output=True) and out_proj uses
    RowParallelLinear(input_is_parallel=True). Their built-in preshard_hooks handle
    weight sharding. No manual splitting or transposing needed.

    Returns:
        (unmodified state_dict, empty dict)
    """
    return state_dict, {}


def _remap_hf_key(key: str, config) -> Optional[str]:
    """Remap a single HF weight key to NxDI format. Returns None to skip."""
    # 4B is dense -- no MoE expert keys to skip.

    # Mamba in_proj and out_proj: pass through.
    # ColumnParallelLinear/RowParallelLinear handle TP sharding via their
    # own preshard_hook -- no manual splitting or transposing needed.

    # Embeddings
    if key == "backbone.embeddings.weight":
        return "embed_tokens.weight"

    # Final norm (backbone.norm_f -> norm)
    if key == "backbone.norm_f.weight":
        return "norm.weight"

    # LM head (already correct prefix)
    if key == "lm_head.weight":
        return "lm_head.weight"

    # Layer keys
    if key.startswith("backbone.layers."):
        # Strip "backbone." prefix -> "layers.{i}.norm.weight" etc.
        layer_key = key[len("backbone.") :]

        # Determine layer index to route correctly
        parts = layer_key.split(".")
        layer_idx = int(parts[1])
        layer_type = config.layer_types[layer_idx]

        # Norm: layers.{i}.norm.weight -> layers.{i}.input_layernorm.weight
        if f"layers.{layer_idx}.norm.weight" == layer_key:
            return f"layers.{layer_idx}.input_layernorm.weight"

        if layer_type == "attention":
            # Attention: NxDI's preshard_hook in GroupQueryAttention_QKV/O
            # handles the remapping from flat keys to nested module paths.
            # We just need to replace .mixer. with .self_attn.:
            #   HF: backbone.layers.{i}.mixer.q_proj.weight
            #   Our state_dict: layers.{i}.self_attn.q_proj.weight
            #   preshard_hook: -> layers.{i}.self_attn.qkv_proj.q_proj.weight
            if ".mixer." in layer_key:
                return layer_key.replace(".mixer.", ".self_attn.")

        # Mamba and dense MLP: keep .mixer. prefix as-is
        return layer_key

    return None


def _convert_nemotron_hf_to_neuron_state_dict(
    state_dict: Dict[str, Any], config: NemotronHInferenceConfig
) -> Dict[str, Any]:
    """
    Convert HF Nemotron-H 4B weights to NxDI model format.

    HF format (prefix: backbone.):
        backbone.embeddings.weight
        backbone.layers.{i}.norm.weight
        backbone.layers.{i}.mixer.{...}   # Mamba, MLP, or Attention
        backbone.norm_f.weight
        lm_head.weight

    NxDI format:
        embed_tokens.weight
        layers.{i}.input_layernorm.weight
        layers.{i}.mixer.{...}         # Mamba or MLP
        layers.{i}.self_attn.{...}     # Attention (preshard_hook adds nesting)
        norm.weight
        lm_head.weight

    The 4B variant has no MoE, so no expert stacking / shared-expert transpose is
    needed. This is a straightforward key remap + Mamba conv weight reshape.
    """
    new_state_dict = {}

    # First pass: convert Mamba conv1d weight shapes (HF Conv1d -> flat tensor)
    state_dict = _convert_mamba_conv_weights(state_dict)

    # Reorder Mamba conv_weight/conv_bias from HF layout [x|B|C] to
    # TP-interleaved [x_0|B_0|C_0 | x_1|B_1|C_1 | ...] so that
    # preshard_hook's contiguous chunking gives correct per-rank weights.
    tp_degree = (
        config.neuron_config.tp_degree
        if hasattr(config, "neuron_config") and config.neuron_config
        else 1
    )
    if tp_degree > 1:
        intermediate_size_full = config.mamba_num_heads * config.mamba_head_dim
        groups_time_state_size_full = config.n_groups * config.ssm_state_size
        for key in list(state_dict.keys()):
            if ".conv_weight" in key or ".conv_bias" in key:
                tensor = state_dict[key]
                dim = 0
                sz = tensor.shape[dim]
                expected = intermediate_size_full + 2 * groups_time_state_size_full
                if sz == expected:
                    x_part = tensor.narrow(dim, 0, intermediate_size_full)
                    B_part = tensor.narrow(
                        dim, intermediate_size_full, groups_time_state_size_full
                    )
                    C_part = tensor.narrow(
                        dim,
                        intermediate_size_full + groups_time_state_size_full,
                        groups_time_state_size_full,
                    )
                    x_chunks = x_part.chunk(tp_degree, dim=dim)
                    B_chunks = B_part.chunk(tp_degree, dim=dim)
                    C_chunks = C_part.chunk(tp_degree, dim=dim)
                    reordered = []
                    for r in range(tp_degree):
                        reordered.extend([x_chunks[r], B_chunks[r], C_chunks[r]])
                    state_dict[key] = torch.cat(reordered, dim=dim)

    # Split Mamba in_proj (no-op with Falcon-H1 pattern - returns empty dict).
    # Kept for code structure compatibility with the 30B port.
    state_dict, mamba_proj_entries = _split_mamba_projections(state_dict, config)
    new_state_dict.update(mamba_proj_entries)

    # Remap all remaining keys. For the 4B dense variant this is simply:
    #   backbone.embeddings.weight        -> embed_tokens.weight
    #   backbone.layers.{i}.norm.weight   -> layers.{i}.input_layernorm.weight
    #   backbone.layers.{i}.mixer.{k}     -> layers.{i}.mixer.{k}          (Mamba, MLP)
    #                                       layers.{i}.self_attn.{k}       (Attention)
    #   backbone.norm_f.weight            -> norm.weight
    #   lm_head.weight                    -> lm_head.weight
    for key, value in state_dict.items():
        new_key = _remap_hf_key(key, config)
        if new_key is not None:
            new_state_dict[new_key] = value

    # Add rank utility tensor (model-level, used for lm_head logits masking).
    new_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    logger.info(f"State dict conversion complete: {len(new_state_dict)} keys")
    gc.collect()
    return new_state_dict
