"""NKI kernels for Mamba-2 SSD recurrent forward -- DGE-OOB survivor pattern.

Adapted from Qwen3-Coder-Next's `nki_deltanet.py` (SDK 2.30, NKI 0.4.0).
Uses ONLY the "cheap ops" that pass the DGE budget in full-model NEFFs:
  dma_copy, nc_matmul, nc_transpose, tensor_copy, tensor_scalar, tensor_tensor,
  activation, memset.

The DeltaNet template is documented in
  /Users/jburtoft/Documents/opencode/working/Solar2/tickets/TBD-dge-oob-nki-kernel-complexity-consolidated/README.md
as the ONE kernel structure that survives full-graph compilation in the
Qwen3-Coder-Next 80B production deployment.

This adapts the pattern to Mamba-2 SSD math:
    S_{t+1} = dA_t * S_t + outer(x_t, dB_t)
    y_t = state @ C_t + D * x_t

Layout: partition dim = ssm_state_size (N=128), free dim = head_dim (D=128).
Both are 128 = P_MAX = SBUF partition width, matching the DeltaNet template.

Called per (batch, head) -- outer loop over B*H in PyTorch, matching the
DeltaNet integration pattern. For Nemotron's num_heads=64, this means 64
kernel calls per Mamba-2 layer at batch=1.

Group handling (Nemotron n_groups=8): B and C are shared across heads within
a group. The PyTorch caller must broadcast group B/C to per-head B/C before
invocation (add head axis, repeat n_groups_per_head times).
"""

import nki
import nki.isa as nisa
import nki.language as nl

# SBUF partition dim = 128 = NeuronCore tile width.
# Both head_dim (D) and dstate (N) are 128 for Nemotron and Zamba2/Falcon-H1.
P_MAX = 128


@nki.jit
def mamba2_ssd_recurrent_fwd(
    x_head: nl.ndarray,       # (S, D=128) float32   -- value per token for this head
    dA_head: nl.ndarray,      # (S, N=128) float32   -- scalar decay, broadcast to N
    dB_head: nl.ndarray,      # (S, N=128) float32   -- input coefficient per token
    C_head: nl.ndarray,       # (S, N=128) float32   -- output coefficient per token
    D_x_head: nl.ndarray,     # (S, D=128) float32   -- pre-computed D * x for skip
) -> nl.ndarray:
    """NKI Mamba-2 SSD recurrent forward -- single (batch, head) pair.

    Sequentially processes S tokens with a 128x128 SBUF state matrix.
    State layout: partition dim = N (dstate), free dim = D (head_dim).

    Args:
        x_head: (S, D=128) float32 -- x[t] is the value token for this head
        dA_head: (S, N=128) float32 -- dA[t] is exp(dt * A_head), broadcast to N
        dB_head: (S, N=128) float32 -- dB[t] = dt * B[t], for this head/group
        C_head: (S, N=128) float32 -- C[t] for this head/group
        D_x_head: (S, D=128) float32 -- (D_scalar * x)[t] pre-computed on caller side

    Returns:
        output: (S, D=128) float32 -- y[t] = C^T @ state + D * x  for each t
    """
    seq_len, dim = x_head.shape  # dim=D=128

    # Output tensor in HBM
    output = nl.ndarray((seq_len, dim), dtype=x_head.dtype, buffer=nl.shared_hbm)

    # Stride: for 2D (S, D), row stride = D
    seq_stride = dim

    # Initialize recurrent state in SBUF: (P_MAX=N, D=128)
    # Layout: partition=N (dstate index), free=D (head_dim index)
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # Sequential loop over tokens (state-dependent)
    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        # ---- Load per-token inputs for this head ----
        # x_t: (D=P_MAX, 1)  -- for output skip step and for outer product
        x_t = nl.ndarray((P_MAX, 1), dtype=x_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_t,
            src=x_head.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # dA_t: (N=P_MAX, 1)  -- scalar decay, already broadcast to N on caller side
        dA_t = nl.ndarray((P_MAX, 1), dtype=dA_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=dA_t,
            src=dA_head.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # dB_t: (N=P_MAX, 1)
        dB_t = nl.ndarray((P_MAX, 1), dtype=dB_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=dB_t,
            src=dB_head.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # C_t: (N=P_MAX, 1)
        C_t = nl.ndarray((P_MAX, 1), dtype=C_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=C_t,
            src=C_head.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # D_x_t: (D=P_MAX, 1)  -- for the skip connection
        D_x_t = nl.ndarray((P_MAX, 1), dtype=D_x_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=D_x_t,
            src=D_x_head.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # ---- Step 1: Decay state -- state = state * dA_t ----
        # state (N, D) *= dA_t (N, 1)  broadcast along free axis
        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=dA_t,
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(dst=state, src=state_decayed)

        # ---- Step 2: Build outer(x_t, dB_t) -- shape (N=P_MAX, D=dim) ----
        # We need outer[n, d] = dB_t[n] * x_t[d]
        # dB_t is (N, 1) on partition; x_t is (D, 1) on partition.
        # To multiply: broadcast x_t along partition (from D=partition to N=partition,
        # keeping D on free axis), then multiply by dB_t (N, 1) with tensor_scalar.

        # Transpose x_t: (D=partition, 1=free) -> (1=partition, D=free) in PSUM
        x_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=x_row_psum, data=x_t)

        # PSUM -> SBUF (matching partition dim)
        x_row_sbuf = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_row_sbuf, src=x_row_psum)

        # Partition-broadcast: ones (1, P_MAX)^T @ x_row (1, D) = (P_MAX, D)
        # via nc_matmul with all-ones stationary of shape (1, P_MAX).
        # nc_matmul(stationary=(P, K), moving=(P, F)) -> (K, F) in PSUM.
        # With P=1, K=P_MAX, F=dim: ones^T @ x_row = (P_MAX, dim) ✓
        ones_col = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)

        x_broadcast_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=x_broadcast_psum, stationary=ones_col, moving=x_row_sbuf
        )
        x_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_broadcast, src=x_broadcast_psum)

        # Now element-wise multiply: outer[n, d] = x_broadcast[n, d] * dB_t[n, 0]
        # tensor_scalar broadcasts (P, 1) across free axis F.
        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=outer_prod,
            data=x_broadcast,
            op0=nl.multiply,
            operand0=dB_t,
            engine=nisa.vector_engine,
        )

        # ---- Step 3: state += outer(x_t, dB_t) ----
        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=state_new, data1=state, data2=outer_prod, op=nl.add)
        nisa.tensor_copy(dst=state, src=state_new)

        # ---- Step 4: y_pre = state @ C_t   -- output projection ----
        # state (P_MAX=N, D) is stationary; C_t (P_MAX=N, 1) is moving.
        # nc_matmul reduces over partition N: (K=D, F=1) in PSUM.
        y_pre_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=y_pre_psum, stationary=state, moving=C_t)
        # Note: y_pre now has D on the partition axis (matching x_t layout).
        y_pre = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y_pre, src=y_pre_psum)

        # ---- Step 5: y_t = y_pre + D_x_t   -- add skip connection ----
        y_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_t, data1=y_pre, data2=D_x_t, op=nl.add)

        # ---- Store output for token t ----
        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=y_t,
        )

    return output


@nki.jit
def mamba2_ssd_recurrent_fwd_state(
    x_head: nl.ndarray,       # (S, D=128) float32
    dA_head: nl.ndarray,      # (S, N=128) float32
    dB_head: nl.ndarray,      # (S, N=128) float32
    C_head: nl.ndarray,       # (S, N=128) float32
    D_x_head: nl.ndarray,     # (S, D=128) float32
):
    """Same as mamba2_ssd_recurrent_fwd but also returns the final (D, N) state.

    Enables CTE -> TKG state carry-over: after prefill, the final state per
    (batch, head) can be persisted and passed as an initial state for decode.
    This matches the pattern in Qwen3-Coder-Next's deltanet_recurrent_fwd_state.

    Note: the state's on-disk layout is (N=P_MAX, D=free). The caller may
    need to transpose to (D, N) or (B, H, D, N) depending on how the SSM state
    is stored in the model.

    Returns:
        output:      (S, D) float32
        final_state: (N, D) float32  -- ATTENTION: partition axis is N, not D.
                                        Caller may need to transpose for storage.
    """
    seq_len, dim = x_head.shape

    output = nl.ndarray((seq_len, dim), dtype=x_head.dtype, buffer=nl.shared_hbm)
    final_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    seq_stride = dim
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        x_t = nl.ndarray((P_MAX, 1), dtype=x_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_t, src=x_head.ap(pattern=[[1, P_MAX]], offset=tok_offset)
        )
        dA_t = nl.ndarray((P_MAX, 1), dtype=dA_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=dA_t, src=dA_head.ap(pattern=[[1, P_MAX]], offset=tok_offset)
        )
        dB_t = nl.ndarray((P_MAX, 1), dtype=dB_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=dB_t, src=dB_head.ap(pattern=[[1, P_MAX]], offset=tok_offset)
        )
        C_t = nl.ndarray((P_MAX, 1), dtype=C_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=C_t, src=C_head.ap(pattern=[[1, P_MAX]], offset=tok_offset)
        )
        D_x_t = nl.ndarray((P_MAX, 1), dtype=D_x_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=D_x_t, src=D_x_head.ap(pattern=[[1, P_MAX]], offset=tok_offset)
        )

        # Decay
        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed, data=state, op0=nl.multiply,
            operand0=dA_t, engine=nisa.vector_engine,
        )
        nisa.tensor_copy(dst=state, src=state_decayed)

        # outer(x_t, dB_t)
        x_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=x_row_psum, data=x_t)
        x_row_sbuf = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_row_sbuf, src=x_row_psum)

        ones_col = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)
        x_broadcast_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=x_broadcast_psum, stationary=ones_col, moving=x_row_sbuf)
        x_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_broadcast, src=x_broadcast_psum)

        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=outer_prod, data=x_broadcast, op0=nl.multiply,
            operand0=dB_t, engine=nisa.vector_engine,
        )

        # State update
        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=state_new, data1=state, data2=outer_prod, op=nl.add)
        nisa.tensor_copy(dst=state, src=state_new)

        # y = state @ C + D*x
        y_pre_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=y_pre_psum, stationary=state, moving=C_t)
        y_pre = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y_pre, src=y_pre_psum)

        y_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_t, data1=y_pre, data2=D_x_t, op=nl.add)

        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=y_t,
        )

    # Write final state to HBM
    nisa.dma_copy(dst=final_state, src=state)

    return output, final_state
