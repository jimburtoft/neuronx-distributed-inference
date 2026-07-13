"""PyTorch wrapper that adapts Nemotron Mamba-2 inputs to the chunked NKI kernel.

Handles:
  - Group -> per-head B/C broadcasting (e.g., Nemotron n_groups=8, num_heads=64)
  - Precomputing log_a = dt * A, cum_log_a = cumsum(log_a) per (batch, head)
  - Precomputing D_x = D * x skip term
  - Static log_mask (chunk_size, chunk_size), cached across calls
  - Batch and head loops in PyTorch (kernel is per-(batch, head))

Called from `NeuronNemotronMamba2Layer._forward_prefill` in modeling_nemotron_h.py
when `USE_CHUNKED_NKI_SCAN=True`. Correctness verified bit-for-bit vs the head-grouped
quadratic scan (50/50 tokens at ctx=128, 30/30 at ctx=1024) on Nemotron-3-Nano-30B.

Returns:
  - y: (batch, seq_len, num_heads, head_dim) float32
  - ssm_state: (batch, num_heads, head_dim, ssm_state_size) float32
    -- matches the signatures of _pytorch_ssd_scan / _chunked_quadratic_scan.

Constraints:
  - head_dim <= 128 (Nemotron uses 64; Falcon-H1 uses 128)
  - ssm_state_size == 128 (chunk math assumes N=P_MAX=128)
  - chunk_size == 128 (P_MAX; the kernel's SBUF partition width)
  - seq_len can be arbitrary; wrapper pads up to next multiple of chunk_size
    and crops output back to original seq_len

Padding-mask handling: the caller in modeling_nemotron_h.py zeroes dt at padded
positions BEFORE calling this wrapper. When dt=0, log_a=0 and dB=0, so state
update at padded positions is identity and outer product is zero -- padded
positions don't affect y or the final state. No explicit padding_mask handling
needed here.
"""

from typing import Optional, Tuple

import torch


# Module-level cache for the static log_mask (shape (128, 128), always the same
# for a given chunk_size). Reduces per-call overhead.
_LOG_MASK_CACHE = {}


def _get_log_mask(Q: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(Q, Q) mask with 0 where j<=i, -1e30 where j>i. Cached per (device, dtype)."""
    key = (Q, str(device), dtype)
    if key not in _LOG_MASK_CACHE:
        m = torch.full((Q, Q), -1e30, dtype=dtype, device=device)
        i_idx = torch.arange(Q, device=device).unsqueeze(1)
        j_idx = torch.arange(Q, device=device).unsqueeze(0)
        m = torch.where(j_idx <= i_idx, torch.zeros_like(m), m)
        _LOG_MASK_CACHE[key] = m
    return _LOG_MASK_CACHE[key]


def chunked_ssd_prefill_scan(
    x: torch.Tensor,          # (B, L, num_heads, head_dim=128)
    dt: torch.Tensor,          # (B, L, num_heads)  -- softplused per-head dt
    A: torch.Tensor,           # (num_heads,)       -- per-head A parameter (negative)
    B_tensor: torch.Tensor,   # (B, L, n_groups, ssm_state_size=128)
    C_tensor: torch.Tensor,   # (B, L, n_groups, ssm_state_size=128)
    D: torch.Tensor,           # (num_heads,)       -- skip scalar per head
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
    n_groups: int,
    chunk_size: int = 128,
    padding_mask: Optional[torch.Tensor] = None,  # accepted for signature parity; not directly used
                                                    # (see docstring: dt zeroed by caller handles padding implicitly)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute prefill Mamba-2 SSD scan using the NKI chunked kernel.

    Returns:
        y: (batch, seq_len, num_heads, head_dim) float32
        ssm_state: (batch, num_heads, head_dim, ssm_state_size) float32
    """
    try:
        from .nki_mamba2_ssd_chunked import mamba2_ssd_chunked_fwd
    except ImportError:
        from nki_mamba2_ssd_chunked import mamba2_ssd_chunked_fwd

    assert head_dim <= 128, f"chunked kernel requires head_dim<=128, got {head_dim}"
    assert ssm_state_size == 128, f"chunked kernel requires ssm_state_size=128, got {ssm_state_size}"
    assert chunk_size == 128, f"chunked kernel requires chunk_size=128, got {chunk_size}"

    batch_size, seq_len, _, _ = x.shape

    # Pad seq_len up to a multiple of chunk_size if needed
    pad_len = 0
    if seq_len % chunk_size != 0:
        pad_len = chunk_size - (seq_len % chunk_size)
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
        dt = torch.nn.functional.pad(dt, (0, 0, 0, pad_len))
        B_tensor = torch.nn.functional.pad(B_tensor, (0, 0, 0, 0, 0, pad_len))
        C_tensor = torch.nn.functional.pad(C_tensor, (0, 0, 0, 0, 0, pad_len))
    S_padded = seq_len + pad_len

    # Broadcast group B/C to per-head B/C
    # B, C shape (batch, S, n_groups, ssm_state_size) -> (batch, S, num_heads, ssm_state_size)
    heads_per_group = num_heads // n_groups
    B_h = B_tensor.repeat_interleave(heads_per_group, dim=2)  # (batch, S, num_heads, N)
    C_h = C_tensor.repeat_interleave(heads_per_group, dim=2)

    # Per-token scalar decay: a_t_head = exp(dt_t_head * A_head)
    # dt (batch, S, num_heads), A (num_heads,)
    # log_a (batch, S, num_heads)
    log_a = (dt * A.view(1, 1, -1)).to(torch.float32)  # A is negative so log_a is negative -> a_t in (0, 1)
    cum_log_a = torch.cumsum(log_a, dim=1).to(torch.float32)  # (batch, S, num_heads)

    # D * x skip: (num_heads,) * (batch, S, num_heads, head_dim)
    D_x = (D.view(1, 1, -1, 1) * x).to(torch.float32)  # (batch, S, num_heads, head_dim)

    # dB per-head: (batch, S, num_heads, N) = dt.unsqueeze(-1) * B_h
    dB_all = (dt.unsqueeze(-1) * B_h).to(torch.float32)

    # Cast x and C to fp32 once
    x_f32 = x.to(torch.float32)
    C_f32 = C_h.to(torch.float32)

    device = x.device
    dtype_f32 = torch.float32

    log_mask = _get_log_mask(chunk_size, device, dtype_f32)

    # Kernel is invoked per (batch, head). Loop in PyTorch.
    output_all = torch.zeros(batch_size, S_padded, num_heads, head_dim, dtype=dtype_f32, device=device)
    # Track final state per (batch, head) for KV-cache style handoff to decode.
    # ssm_state layout: (batch, num_heads, head_dim, ssm_state_size) matches other scan paths.
    ssm_state_final = torch.zeros(
        batch_size, num_heads, head_dim, ssm_state_size,
        dtype=dtype_f32, device=device,
    )

    # dA placeholder (unused by kernel; kept for API parity with recurrent kernel)
    dA_placeholder = torch.zeros(S_padded, ssm_state_size, dtype=dtype_f32, device=device)

    for b in range(batch_size):
        for h in range(num_heads):
            x_head = x_f32[b, :, h, :].contiguous()             # (S, head_dim)
            dB_head = dB_all[b, :, h, :].contiguous()           # (S, N)
            C_head = C_f32[b, :, h, :].contiguous()             # (S, N)
            D_x_head = D_x[b, :, h, :].contiguous()             # (S, D)
            cum_log_a_head = cum_log_a[b, :, h].contiguous()    # (S,)
            log_a_head = log_a[b, :, h].contiguous()            # (S,)

            y_head, state_head = mamba2_ssd_chunked_fwd(
                x_head,
                dA_placeholder,     # unused
                dB_head,
                C_head,
                D_x_head,
                cum_log_a_head,
                log_a_head,
                log_mask,
            )
            # y_head: (S, head_dim); state_head: (N, D)

            output_all[b, :, h, :] = y_head
            # state_head is (N, D); other scan paths use (num_heads, head_dim, ssm_state_size) = (H, D, N).
            # Transpose (N, D) -> (D, N) to match.
            ssm_state_final[b, h] = state_head.T

    # Crop output back to original seq_len (drop padding)
    output = output_all[:, :seq_len, :, :].contiguous()

    return output, ssm_state_final
