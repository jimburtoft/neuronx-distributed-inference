"""Standalone test: mamba2_ssd_recurrent NKI kernel vs PyTorch reference.

Verifies:
  1. Kernel compiles under SDK 2.31 (torch_neuronx.trace succeeds)
  2. Kernel output matches PyTorch reference (cos >= 0.999, max abs diff <= 1e-3 fp32)

If both pass, we have a working DGE-OOB survivor pattern for Nemotron Mamba-2 SSD.

Usage on trn2.x instance:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  python test_mamba2_ssd_recurrent.py
"""

import sys
import time
import torch
import torch.nn.functional as F

import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "nki_kernels"))


def reference_mamba2_ssd_scan(x, dA, dB, C, D_x):
    """PyTorch reference for the Mamba-2 SSD recurrence.

    Args:
        x: (S, D) float32
        dA: (S, N) float32 -- scalar per token broadcast to N
        dB: (S, N) float32
        C: (S, N) float32
        D_x: (S, D) float32 -- D_scalar * x, precomputed

    Returns:
        y: (S, D) float32
    """
    S, D = x.shape
    _, N = dB.shape

    # Actual scalar decay per token (dA broadcast to N is just for kernel layout;
    # semantically it's a scalar per token, so we take dA[:, 0]).
    dA_scalar = dA[:, 0]  # (S,)

    state = torch.zeros(D, N, dtype=torch.float32)
    y = torch.zeros(S, D, dtype=torch.float32)

    for t in range(S):
        # state = dA[t] * state + outer(x[t], dB[t])
        state = dA_scalar[t] * state + torch.outer(x[t], dB[t])
        # y[t] = state @ C[t] + D_x[t]
        y[t] = state @ C[t] + D_x[t]

    return y


class SsdWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dA, dB, C, D_x):
        from nki_mamba2_ssd_recurrent import mamba2_ssd_recurrent_fwd
        return mamba2_ssd_recurrent_fwd(x, dA, dB, C, D_x)


def main():
    import torch_neuronx

    # Realistic dimensions for Nemotron Mamba-2 SSD
    S = 128  # sequence length (test at ctx=128; larger seq is just more iterations)
    D = 128  # head_dim
    N = 128  # dstate

    torch.manual_seed(42)
    # Use fp32 for numerical fidelity in this test.
    x = torch.randn(S, D, dtype=torch.float32) * 0.1
    # dA is exp(dt * A_head) -- values in (0, 1] for stability
    dA_scalar = torch.rand(S, dtype=torch.float32) * 0.9 + 0.05  # (0.05, 0.95)
    dA = dA_scalar.unsqueeze(-1).expand(S, N).contiguous()  # broadcast to N
    dB = torch.randn(S, N, dtype=torch.float32) * 0.1
    C = torch.randn(S, N, dtype=torch.float32) * 0.1
    D_scalar = 0.5  # skip connection scalar (like Mamba-2's D parameter)
    D_x = D_scalar * x  # precomputed

    # Reference
    print("Computing PyTorch reference...")
    t0 = time.perf_counter()
    ref = reference_mamba2_ssd_scan(x, dA, dB, C, D_x)
    ref_s = time.perf_counter() - t0
    print(f"Reference: {ref_s*1000:.1f}ms")
    print(f"Reference stats: mean={ref.mean().item():.4f}, std={ref.std().item():.4f}, "
          f"absmax={ref.abs().max().item():.4f}")

    # Wrap + trace
    wrapper = SsdWrapper()
    print("\nTracing NKI kernel via torch_neuronx.trace...")
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        wrapper,
        example_inputs=(x, dA, dB, C, D_x),
        compiler_args=["--auto-cast=none"],
    )
    trace_s = time.perf_counter() - t0
    print(f"Trace + compile: {trace_s:.1f}s")

    # Run on device
    print("\nRunning on Neuron device...")
    t0 = time.perf_counter()
    out = traced(x, dA, dB, C, D_x)
    run_s = time.perf_counter() - t0
    print(f"Device run: {run_s*1000:.2f}ms")
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"Output stats: mean={out.mean().item():.4f}, std={out.std().item():.4f}, "
          f"absmax={out.abs().max().item():.4f}")

    # Compare
    out_cpu = out.cpu().float()
    diff = (out_cpu - ref).abs()
    max_diff = diff.max().item()
    cos_sim = F.cosine_similarity(out_cpu.flatten(), ref.flatten(), dim=0).item()
    print(f"\n=== NUMERICAL COMPARISON ===")
    print(f"cosine similarity: {cos_sim:.6f}")
    print(f"max abs diff: {max_diff:.6f}")

    passed = cos_sim >= 0.999 and max_diff <= 1e-3
    print(f"\n{'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"Sample diffs at t=0,S//2,S-1 (first 4 dims):")
        for tt in [0, S // 2, S - 1]:
            print(f"  t={tt}: ref={ref[tt, :4].tolist()}, out={out_cpu[tt, :4].tolist()}, "
                  f"diff={(out_cpu[tt, :4] - ref[tt, :4]).abs().tolist()}")
    return passed


if __name__ == "__main__":
    main()
