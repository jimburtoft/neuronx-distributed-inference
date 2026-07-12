"""PyTorch chunked-scan reference for Mamba-2 SSD.

Standalone reference to develop against. Computes y_t and final state
using the chunk-parallel formulation, then compares to the naive recurrent
scan (which we've already validated).

Chunk math (per chunk of size Q, initial state S_c):
    A_i = prod_{k<i} a_{cQ+k}                              # (Q,)
    A_{Q,j+1} = prod_{k in [j+1,Q)} a_{cQ+k}                # for chunk-end state

    y_i = A_i * (S_c · C_i)                                 # state-carry
          + sum_{j<i} A_{i,j+1} * (dB_j·C_i) * x_j          # intra-chunk
          + D * x_i                                         # skip

    dS = sum_{j<Q} A_{Q,j+1} * outer(x_j, dB_j)            # intra-chunk state contrib
    S_new = A_Q * S_c + dS

Verified against naive recurrent scan below.
"""

import math

import torch


def naive_recurrent_scan(x, a, dB, C, D_x):
    """Naive per-token recurrence -- ground truth."""
    S, D = x.shape
    _, N = dB.shape
    state = torch.zeros(D, N, dtype=torch.float32)
    y = torch.zeros(S, D, dtype=torch.float32)
    for t in range(S):
        state = a[t] * state + torch.outer(x[t], dB[t])
        y[t] = state @ C[t] + D_x[t]
    return y, state


def chunked_scan(x, a, dB, C, D_x, chunk_size=128):
    """Chunk-parallel form. Same result as naive_recurrent_scan.

    Args:
        x: (S, D)
        a: (S,) -- per-token scalar decay (exp(dt*A_head))
        dB: (S, N)
        C: (S, N)
        D_x: (S, D)
        chunk_size: Q

    Returns:
        y: (S, D)
        final_state: (D, N)
    """
    S, D = x.shape
    _, N = dB.shape
    Q = chunk_size
    assert S % Q == 0, f"seq_len {S} must be divisible by chunk_size {Q}"
    num_chunks = S // Q

    log_a = torch.log(a)  # (S,)

    state = torch.zeros(D, N, dtype=torch.float32)
    y = torch.zeros(S, D, dtype=torch.float32)

    for c in range(num_chunks):
        t0 = c * Q
        t1 = t0 + Q
        # Chunk-local slices
        x_c = x[t0:t1]           # (Q, D)
        dB_c = dB[t0:t1]         # (Q, N)
        C_c = C[t0:t1]           # (Q, N)
        Dx_c = D_x[t0:t1]        # (Q, D)
        log_a_c = log_a[t0:t1]   # (Q,)

        # Cumulative log-decay from chunk start.
        # Let L[i] = sum_{k<=i} log_a[k], i.e., cum_log_a[i].
        cum_log_a = torch.cumsum(log_a_c, dim=0)  # (Q,) -- cum_log_a[i] = sum_{k<=i} log_a[k]

        # y_i uses S_{i+1} (post-update state), where S_{i+1} = a_i * S_i + outer(x_i, dB_i).
        # Contributions:
        #   S_{i+1} = Α_{i, 0} * S_c + sum_{j=0..i} Α_{i, j+1} * outer(x_j, dB_j)
        # where Α_{a, b} = a_a * a_{a-1} * ... * a_b for a>=b, else 1 (empty product).
        #
        # State-carry decay for token i: Α_{i, 0} = a_0 * a_1 * ... * a_i = exp(cum_log_a[i]).
        A_vec = torch.exp(cum_log_a)  # (Q,) -- state-carry decay per token

        # For chunk-end state: state after all Q tokens = S_{cQ+Q}. Contributions:
        #   S_{cQ+Q} = Α_{Q-1, 0} * S_c + sum_{j=0..Q-1} Α_{Q-1, j+1} * outer(x_j, dB_j)
        A_Q = torch.exp(cum_log_a[Q - 1])  # scalar = a_0 * ... * a_{Q-1}

        # Α_{Q-1, j+1} = a_{j+1} * a_{j+2} * ... * a_{Q-1}
        #             = exp(cum_log_a[Q-1] - cum_log_a[j])  for j < Q-1
        #             = 1  for j = Q-1  (empty product; matches exp(cum_log_a[Q-1] - cum_log_a[Q-1]))
        A_Qj1 = torch.exp(cum_log_a[Q - 1] - cum_log_a)  # (Q,)

        # Intra-chunk: y_i = Α_{i, 0} · (S_c · C_i) + sum_{j<=i} Α_{i, j+1} · (dB_j · C_i) · x_j + D_x_i
        # Α_{i, j+1} = a_{j+1} · a_{j+2} · ... · a_i = exp(cum_log_a[i] - cum_log_a[j])  for j <= i.
        # For j = i (diagonal), this is exp(cum_log_a[i] - cum_log_a[i]) = 1.
        # For j > i (non-causal), we mask to 0.
        # log(Α_{i, j+1}) = cum_log_a[i] - cum_log_a[j] for j<=i.
        decay_matrix = cum_log_a.unsqueeze(1) - cum_log_a.unsqueeze(0)  # (Q, Q): row i, col j
        # Causal mask: j <= i
        i_idx = torch.arange(Q).unsqueeze(1)
        j_idx = torch.arange(Q).unsqueeze(0)
        causal = (j_idx <= i_idx)
        # Set non-causal region to -inf (via -1e30) so exp gives 0
        decay_matrix = torch.where(causal, decay_matrix, torch.tensor(-1e30, dtype=torch.float32))
        A_ij_masked = torch.exp(decay_matrix)  # (Q, Q): 0 for j>i, decay for j<=i

        # dB_C[j, i] = dB_c[j] · C_c[i] = sum_n dB_c[j, n] * C_c[i, n]
        dB_C = dB_c @ C_c.T  # (Q, Q)

        # M[i, j] = A_ij_masked[i, j] * dB_C[j, i]
        M = A_ij_masked * dB_C.T  # (Q, Q)

        # Intra-chunk output: y_intra[i] = sum_j M[i, j] * x_c[j]
        y_intra = M @ x_c  # (Q, D)

        # State-carry contribution: state_carry[i] = A_vec[i] * (state @ C_c[i])
        state_C = C_c @ state.T  # (Q, D)
        state_carry = A_vec.unsqueeze(-1) * state_C  # (Q, D)

        # Output for chunk
        y_c = state_carry + y_intra + Dx_c
        y[t0:t1] = y_c

        # Update state: S_new = A_Q * S + sum_{j<Q} A_Qj1[j] * outer(x_c[j], dB_c[j])
        weighted_x = (A_Qj1.unsqueeze(-1) * x_c)  # (Q, D)
        dS = weighted_x.T @ dB_c  # (D, N)
        state = A_Q * state + dS

    return y, state


def main():
    torch.manual_seed(42)
    S, D, N, Q = 512, 128, 128, 128
    x = torch.randn(S, D) * 0.1
    a = torch.rand(S) * 0.9 + 0.05  # (0.05, 0.95)
    dB = torch.randn(S, N) * 0.1
    C = torch.randn(S, N) * 0.1
    D_x = 0.5 * x

    print("Running naive recurrent...")
    y_ref, state_ref = naive_recurrent_scan(x, a, dB, C, D_x)
    print(f"  y_ref: mean={y_ref.mean().item():.4f}, std={y_ref.std().item():.4f}, absmax={y_ref.abs().max().item():.4f}")

    print("Running chunked...")
    y_chunked, state_chunked = chunked_scan(x, a, dB, C, D_x, chunk_size=Q)

    diff_y = (y_chunked - y_ref).abs()
    diff_state = (state_chunked - state_ref).abs()
    cos_y = torch.nn.functional.cosine_similarity(y_chunked.flatten(), y_ref.flatten(), dim=0).item()
    cos_state = torch.nn.functional.cosine_similarity(state_chunked.flatten(), state_ref.flatten(), dim=0).item()
    print(f"  y cos_sim: {cos_y:.9f}, max_diff: {diff_y.max().item():.9f}")
    print(f"  state cos_sim: {cos_state:.9f}, max_diff: {diff_state.max().item():.9f}")

    # Also test with multiple chunks
    for S_test in [256, 512, 1024, 2048]:
        x = torch.randn(S_test, D) * 0.1
        a = torch.rand(S_test) * 0.9 + 0.05
        dB = torch.randn(S_test, N) * 0.1
        C = torch.randn(S_test, N) * 0.1
        D_x = 0.5 * x
        y_ref, _ = naive_recurrent_scan(x, a, dB, C, D_x)
        y_chunked, _ = chunked_scan(x, a, dB, C, D_x, chunk_size=Q)
        cos = torch.nn.functional.cosine_similarity(y_chunked.flatten(), y_ref.flatten(), dim=0).item()
        max_d = (y_chunked - y_ref).abs().max().item()
        print(f"  S={S_test}: cos={cos:.9f}, max_diff={max_d:.9f}")


if __name__ == "__main__":
    main()
