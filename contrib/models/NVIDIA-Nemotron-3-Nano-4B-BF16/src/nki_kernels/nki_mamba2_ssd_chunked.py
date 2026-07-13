"""NKI Mamba-2 SSD chunked scan -- DGE-OOB survivor variant.

Adapted from the recurrent kernel (nki_mamba2_ssd_recurrent.py), which is
itself adapted from Qwen3-Coder-Next's nki_deltanet.py.

Validation (SDK 2.31, DLAMI 20260708, trn2.3xlarge, Nemotron-3-Nano-30B):
    Standalone (fp32 vs PyTorch reference):
        - cos_sim = 1.000000 at seq_len in {128, 256, 512, 1024, 2048, 4096, 8192}
        - max_diff <= 3e-5 (fp32 accumulation noise only)
        - Compiles + runs cleanly, no DGE OOB, no NCC_IBIR243
    Full-model (Nemotron-3-Nano-30B, USE_CHUNKED_NKI_SCAN=True):
        - 50/50 tokens bit-identical vs head-grouped quadratic at ctx=128
        - 30/30 tokens bit-identical vs head-grouped quadratic at ctx=1024 (8 chunks)
        - Extends ctx ceiling from 2048 to 8192 on trn2.3xlarge (4x)
        - Prefill 6-41% faster on matched contexts

The recurrence is (per token, per (batch, head)):
    S_{t+1} = a_t * S_t + outer(x_t, dB_t)
    y_t     = S_{t+1} . C_t + D * x_t

Chunk math (per chunk of size Q=128, initial state S_c):
    L[i] = cum_log_a[i] = sum_{k<=i} log(a_{cQ+k})
    A_vec[i] = exp(L[i])                                    # decay applied to S_c for token i
    A_Q      = exp(L[Q-1])                                  # chunk-end decay
    A_Qj[j]  = exp(L[Q-1] - L[j])                           # decay for outer(j) -> state end
    A_ij[i, j] = exp(L[i] - L[j])   for j<=i, 0 otherwise   # causal within-chunk decay

Within-chunk output (uses S_{i+1}, which is state AFTER token i's own update):
    y_i = A_vec[i] * (S_c . C_i)                            # state-carry
          + sum_{j<=i} A_ij[i,j] * (dB_j . C_i) * x_j       # intra-chunk quadratic
          + D * x_i                                         # skip

Chunk-end state:
    S_{c+1} = A_Q * S_c + sum_{j<Q} A_Qj[j] * outer(x_j, dB_j)

Layout:
    state (N, D) in SBUF: partition=N=P_MAX=128 (dstate), free=D (head_dim)
    Both dims can differ; N is fixed to 128 (P_MAX), D can be any value <= 128
    (Nemotron uses D=64). Chunk size Q = P_MAX = 128 (fixed).

Cheap ops only: dma_copy, nc_matmul, nc_transpose, tensor_copy, tensor_scalar,
tensor_tensor, activation, memset. Preserves DGE budget safety at Nemotron-scale
(23 Mamba layers x 64 heads x TP=4 = 368 invocations per NEFF).

Reference: nki_mamba2_ssd_recurrent.py (validated bit-exact vs PyTorch fp32).
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
CHUNK_SIZE = P_MAX  # Q


@nki.jit
def mamba2_ssd_chunked_fwd(
    x_head: nl.ndarray,       # (S, D) float32 -- D=head_dim (any value <= 128)
    dA_head: nl.ndarray,      # (S, N=128) float32 -- unused now (kept for API parity)
    dB_head: nl.ndarray,      # (S, N=128) float32
    C_head: nl.ndarray,       # (S, N=128) float32
    D_x_head: nl.ndarray,     # (S, D) float32
    cum_log_a_head: nl.ndarray,  # (S,) float32 -- precomputed cumsum(log(a)) from t=0
    log_a_head: nl.ndarray,   # (S,) float32 -- precomputed log(a)
    log_mask: nl.ndarray,     # (128, 128) float32 -- precomputed: 0 where j<=i, -1e30 elsewhere
):
    """NKI Mamba-2 SSD chunked forward -- single (batch, head).

    Layout:
        state (N, D) in SBUF: partition=N=P_MAX=128 (dstate), free=D (head_dim, e.g. 64 or 128)
        Both dims can differ; N is fixed to 128 (P_MAX), D can be any small size.
        Chunk size Q = P_MAX = 128 (fixed).

    Caller-precomputed inputs:
        cum_log_a_head: cumsum(log(a)) from t=0 (per-token scalar decay logs, cumulative)
        log_a_head: log(a) (per-token scalar decay logs)
        log_mask: (128,128) with 0 where j<=i, -1e30 where j>i (added to log_A_ij pre-exp)

    Args:
        x_head: (S, D) float32 -- S must be divisible by 128; D <= 128
        dA_head: (S, N=128) float32 -- currently unused, kept for API parity with recurrent kernel
        dB_head: (S, N=128) float32
        C_head: (S, N=128) float32
        D_x_head: (S, D) float32

    Returns:
        output: (S, D) float32
        final_state: (N=128, D) float32 -- ssm state after last token
                     ATTENTION: partition axis is N (dstate), free axis is D (head_dim).
                     Caller may need to transpose to (D, N) or store as-is.
    """
    seq_len, dim_D = x_head.shape  # dim_D = head_dim
    _, dim_N = dB_head.shape       # dim_N = dstate = 128
    Q = CHUNK_SIZE
    N_CHUNKS = seq_len // Q

    output = nl.ndarray((seq_len, dim_D), dtype=x_head.dtype, buffer=nl.shared_hbm)
    # Also emit final state (N=partition, D=free) to HBM for decode handoff
    final_state = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.shared_hbm)

    # State (N=P_MAX partition, D=free)
    state = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # Load static log_mask once
    log_mask_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=log_mask_sb, src=log_mask)

    for c in nl.sequential_range(N_CHUNKS):
        chunk_offset = c * Q

        # ---- Load chunk tiles ----
        x_c = nl.ndarray((P_MAX, dim_D), dtype=x_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_c,
            src=x_head.ap(pattern=[[dim_D, P_MAX], [1, dim_D]], offset=chunk_offset * dim_D),
        )
        dB_c = nl.ndarray((P_MAX, dim_N), dtype=dB_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=dB_c,
            src=dB_head.ap(pattern=[[dim_N, P_MAX], [1, dim_N]], offset=chunk_offset * dim_N),
        )
        C_c = nl.ndarray((P_MAX, dim_N), dtype=C_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=C_c,
            src=C_head.ap(pattern=[[dim_N, P_MAX], [1, dim_N]], offset=chunk_offset * dim_N),
        )
        Dx_c = nl.ndarray((P_MAX, dim_D), dtype=D_x_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=Dx_c,
            src=D_x_head.ap(pattern=[[dim_D, P_MAX], [1, dim_D]], offset=chunk_offset * dim_D),
        )
        # cum_log_a chunk: (Q,) 1D -> load as (P_MAX, 1) partition=Q
        cla_c = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=cla_c,
            src=cum_log_a_head.ap(pattern=[[1, P_MAX]], offset=chunk_offset),
        )
        la_c = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=la_c,
            src=log_a_head.ap(pattern=[[1, P_MAX]], offset=chunk_offset),
        )

        # ---- Compute per-token L_chunk[i] = cla_c[i] - L_prev_chunk_end ----
        # where L_prev_chunk_end = cum_log_a_head[chunk_offset - 1] (or 0 for c=0).
        # From the identity: L[k] - L[k-1] = log_a[k], so L_prev = cla_c[0] - la_c[0].
        # For c=0, cla_c[0] = log_a[0] = la_c[0], so L_prev = 0. ✓

        # Compute L_prev as (1, 1) scalar, then broadcast to (P_MAX, 1) via nc_matmul.
        L0 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        L0_la = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L0, src=cla_c[0:1, :])
        nisa.tensor_copy(dst=L0_la, src=la_c[0:1, :])
        L_prev_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=L_prev_scalar, data1=L0, data2=L0_la, op=nl.subtract)

        # Broadcast L_prev_scalar (1, 1) to L_prev (P_MAX, 1) partition-wise:
        # nc_matmul(stationary=ones (1, P_MAX), moving=L_prev_scalar (1, 1)) -> (P_MAX, 1)
        ones_1P = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_1P, value=1.0)
        L_prev_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=L_prev_psum, stationary=ones_1P, moving=L_prev_scalar)
        L_prev = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L_prev, src=L_prev_psum)

        # L_local[i] = cla_c[i] - L_prev[i]  (in-chunk cumulative log-decay)
        L_local = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=L_local, data1=cla_c, data2=L_prev, op=nl.subtract)

        # A_vec[i] = exp(L_local[i])  -- state-carry decay per token
        A_vec = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=A_vec, op=nl.exp, data=L_local, bias=None, scale=1.0)

        # For A_Q and A_Qj, we need L_local[P_MAX-1] broadcast to all P_MAX partitions.
        # Rather than reading L_local[P_MAX-1, :] into partition-0 (cross-partition SBUF
        # access, not free), we use nc_transpose to get (1, P_MAX) partition-0 form,
        # then extract the single element at free-dim P_MAX-1.
        # L_local (P_MAX, 1) -- transpose to (1, P_MAX) in PSUM.
        L_local_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=L_local_row_psum, data=L_local)
        L_local_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L_local_row_sb, src=L_local_row_psum)
        # L_last_scalar (1, 1) = L_local_row_sb[0:1, P_MAX-1:P_MAX] -- partition-0 slice
        L_last_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L_last_scalar, src=L_local_row_sb[:, P_MAX - 1:P_MAX])
        A_Q_scalar = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=A_Q_scalar, op=nl.exp, data=L_last_scalar, bias=None, scale=1.0)

        # Broadcast A_Q_scalar to (P_MAX, 1) for later use with state.
        A_Q_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=A_Q_psum, stationary=ones_1P, moving=A_Q_scalar)
        A_Q = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=A_Q, src=A_Q_psum)

        # Also broadcast L_last_scalar to (P_MAX, 1) partition-wise for A_Qj.
        L_last_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=L_last_psum, stationary=ones_1P, moving=L_last_scalar)
        L_last = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L_last, src=L_last_psum)

        # A_Qj_log[j] = L_last[j] - L_local[j]  -- (P_MAX, 1) all partitions match
        A_Qj_log = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=A_Qj_log, data1=L_last, data2=L_local, op=nl.subtract)
        A_Qj = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=A_Qj, op=nl.exp, data=A_Qj_log, bias=None, scale=1.0)

        # ---- Compute A_ij[i, j] = exp(L_local[i] - L_local[j])  for j <= i ----
        # Build log_A_ij[i, j] = L_local[i] - L_local[j]. Both i and j are in [0, Q=P_MAX).
        # Result tensor shape: (P_MAX, P_MAX). Free axis = j (also P_MAX).
        # Broadcast L_local along free axis: L_i_bcast[i, j] = L_local[i].
        zeros_QQ = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zeros_QQ, value=0.0)
        L_i_bcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=L_i_bcast, data=zeros_QQ, op0=nl.add,
            operand0=L_local, engine=nisa.vector_engine,
        )
        # Broadcast L_local along partition axis: L_j_bcast[i, j] = L_local[j].
        # Transpose L_local (P, 1) -> (1, P) then partition-broadcast via nc_matmul with ones.
        L_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=L_row_psum, data=L_local)
        L_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L_row_sb, src=L_row_psum)
        ones_col = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)
        L_j_bcast_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=L_j_bcast_psum, stationary=ones_col, moving=L_row_sb)
        L_j_bcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L_j_bcast, src=L_j_bcast_psum)

        # log_A_ij[i, j] = L_i_bcast[i, j] - L_j_bcast[i, j] = L_local[i] - L_local[j]
        log_A_ij_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=log_A_ij_raw, data1=L_i_bcast, data2=L_j_bcast, op=nl.subtract)

        # Add log_mask to zero out non-causal region (log_mask is -1e30 where j>i, 0 elsewhere)
        log_A_ij_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=log_A_ij_masked, data1=log_A_ij_raw, data2=log_mask_sb, op=nl.add)

        A_ij_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=A_ij_masked, op=nl.exp, data=log_A_ij_masked, bias=None, scale=1.0)
        # A_ij_masked[i, j]: partition=i, free=j.

        # ---- Compute dB_C[j, i] = sum_n dB_c[j, n] * C_c[i, n] via nc_matmul ----
        # dB_c: partition=Q free=N
        # C_c: partition=Q free=N
        # We need sum over n. Transpose both to partition=N free=Q.
        dB_c_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=dB_c_T_psum, data=dB_c)
        dB_c_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dB_c_T, src=dB_c_T_psum)
        C_c_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=C_c_T_psum, data=C_c)
        C_c_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=C_c_T, src=C_c_T_psum)
        # nc_matmul(stationary=dB_c_T (N, Q), moving=C_c_T (N, Q)):
        #   result[k, f] = sum_n dB_c_T[n, k] * C_c_T[n, f] = sum_n dB[k, n] * C[f, n]
        # So result[j, i] = dB_C[j, i]. partition=j, free=i.
        dB_C_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=dB_C_psum, stationary=dB_c_T, moving=C_c_T)
        dB_C = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dB_C, src=dB_C_psum)

        # We want M[i, j] = A_ij_masked[i, j] * dB_C[j, i].
        # A_ij_masked: partition=i free=j.
        # dB_C: partition=j free=i => transpose.
        dB_C_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=dB_C_T_psum, data=dB_C)
        dB_C_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dB_C_T, src=dB_C_T_psum)
        # dB_C_T[i, j] = dB_C[j, i] = sum_n dB[j, n] * C[i, n].

        M = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=M, data1=A_ij_masked, data2=dB_C_T, op=nl.multiply)

        # ---- y_intra[i, d] = sum_j M[i, j] * x_c[j, d] ----
        # Need to reduce over j = partition of x_c. Transpose M so partition=j.
        M_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=M_T_psum, data=M)
        M_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=M_T, src=M_T_psum)
        # M_T[j, i] = M[i, j]. partition=j.
        # nc_matmul(stationary=M_T (Q, Q), moving=x_c (Q, D)):
        #   result[k=i, f=d] = sum_p M_T[p, k] * x_c[p, d] = sum_j M[i, j] * x[j, d] ✓
        y_intra_psum = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=y_intra_psum, stationary=M_T, moving=x_c)
        y_intra = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y_intra, src=y_intra_psum)
        # y_intra[i, d]: partition=i, free=d.

        # ---- state_carry[i, d] = A_vec[i] * (state @ C_c[i]) ----
        # state @ C_c[i] where state (N, D), C_c[i] (N,) -> (D,)
        # state: partition=N free=D. C_c_T: partition=N free=Q.
        # nc_matmul(stationary=state (N, D), moving=C_c_T (N, Q)):
        #   result[k=D, f=Q] = sum_n state[n, k] * C_c_T[n, f]
        #                    = sum_n state[n, k] * C_c[f, n]
        # So result[d, i] = state_C[i, d]. partition=d, free=i.
        state_C_pre_psum = nl.ndarray((dim_D, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=state_C_pre_psum, stationary=state, moving=C_c_T)
        state_C_pre = nl.ndarray((dim_D, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=state_C_pre, src=state_C_pre_psum)
        # Transpose (d, i) -> (i, d).
        state_C_T_psum = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=state_C_T_psum, data=state_C_pre)
        state_C = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=state_C, src=state_C_T_psum)
        # state_C[i, d]: partition=i, free=d.

        # state_carry[i, d] = A_vec[i, 0] * state_C[i, d]
        state_carry = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_carry, data=state_C, op0=nl.multiply,
            operand0=A_vec, engine=nisa.vector_engine,
        )

        # ---- y_c = state_carry + y_intra + Dx_c ----
        y_sum1 = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_sum1, data1=state_carry, data2=y_intra, op=nl.add)
        y_c = nl.ndarray((P_MAX, dim_D), dtype=y_sum1.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_c, data1=y_sum1, data2=Dx_c, op=nl.add)

        nisa.dma_copy(
            dst=output.ap(pattern=[[dim_D, P_MAX], [1, dim_D]], offset=chunk_offset * dim_D),
            src=y_c,
        )

        # ---- Update state: S_new = A_Q * state + sum_j A_Qj[j] * outer(x_j, dB_j) ----
        # dS[n, d] = sum_j A_Qj[j] * x_c[j, d] * dB_c[j, n]
        # Let W_x[j, d] = A_Qj[j] * x_c[j, d]. Broadcast A_Qj across free.
        W_x = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=W_x, data=x_c, op0=nl.multiply,
            operand0=A_Qj, engine=nisa.vector_engine,
        )
        # dS[n, d] = sum_j dB_c[j, n] * W_x[j, d]
        # nc_matmul(stationary=dB_c (Q, N), moving=W_x (Q, D)):
        #   result[k=N, f=D] = sum_j dB_c[j, k] * W_x[j, f] = dS[k, f] ✓ partition=N
        dS_psum = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=dS_psum, stationary=dB_c, moving=W_x)
        dS = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dS, src=dS_psum)

        state_decayed = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed, data=state, op0=nl.multiply,
            operand0=A_Q, engine=nisa.vector_engine,
        )
        state_new = nl.ndarray((P_MAX, dim_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=state_new, data1=state_decayed, data2=dS, op=nl.add)
        nisa.tensor_copy(dst=state, src=state_new)

    # Write final state to HBM (state after last chunk)
    nisa.dma_copy(dst=final_state, src=state)

    return output, final_state
