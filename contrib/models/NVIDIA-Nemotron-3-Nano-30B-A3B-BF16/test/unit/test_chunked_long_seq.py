"""Longer-sequence tests for the chunked kernel. Verify:
1. Correctness at S=2048, 4096, 8192 (increasing chunks)
2. Performance scales O(S/Q) as expected
3. No HBM ceiling (state is 4MB/layer)
"""

import sys
import time
import torch
import torch.nn.functional as F

import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "nki_kernels"))


def naive_recurrent_scan(x, a, dB, C, D_x):
    S, D = x.shape; _, N = dB.shape
    state = torch.zeros(D, N, dtype=torch.float32)
    y = torch.zeros(S, D, dtype=torch.float32)
    for t in range(S):
        state = a[t] * state + torch.outer(x[t], dB[t])
        y[t] = state @ C[t] + D_x[t]
    return y, state


def precompute_scan_aux(a):
    log_a = torch.log(a); cum_log_a = torch.cumsum(log_a, dim=0)
    return log_a, cum_log_a


def build_log_mask(Q=128):
    m = torch.full((Q, Q), -1e30, dtype=torch.float32)
    for i in range(Q):
        for j in range(i + 1):
            m[i, j] = 0.0
    return m


class ChunkedSsdWrapper(torch.nn.Module):
    def __init__(self, log_mask):
        super().__init__()
        self.register_buffer("log_mask", log_mask)

    def forward(self, x, dA, dB, C, D_x, cum_log_a, log_a):
        from nki_mamba2_ssd_chunked import mamba2_ssd_chunked_fwd
        y, _ = mamba2_ssd_chunked_fwd(
            x, dA, dB, C, D_x, cum_log_a, log_a, self.log_mask
        )
        return y


def test_seq_len(S):
    import torch_neuronx
    D = 128; N = 128; Q = 128
    torch.manual_seed(S)
    x = torch.randn(S, D) * 0.1
    a = torch.rand(S) * 0.9 + 0.05
    dA = a.unsqueeze(-1).expand(S, N).contiguous()
    dB = torch.randn(S, N) * 0.1
    C = torch.randn(S, N) * 0.1
    D_x = 0.5 * x
    log_a, cum_log_a = precompute_scan_aux(a)
    log_mask = build_log_mask(Q)

    print(f"\n===== S={S} ({S//Q} chunks) =====")
    t0 = time.perf_counter()
    y_ref, _ = naive_recurrent_scan(x, a, dB, C, D_x)
    ref_ms = (time.perf_counter() - t0) * 1000

    wrapper = ChunkedSsdWrapper(log_mask)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        wrapper, example_inputs=(x, dA, dB, C, D_x, cum_log_a, log_a),
        compiler_args=["--auto-cast=none"],
    )
    compile_s = time.perf_counter() - t0

    # Warm up
    _ = traced(x, dA, dB, C, D_x, cum_log_a, log_a)

    # Timing (5 runs)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = traced(x, dA, dB, C, D_x, cum_log_a, log_a)
        times.append((time.perf_counter() - t0) * 1000)
    run_ms = min(times)

    out_cpu = out.cpu().float()
    cos_sim = F.cosine_similarity(out_cpu.flatten(), y_ref.flatten(), dim=0).item()
    max_diff = (out_cpu - y_ref).abs().max().item()
    print(f"  ref: {ref_ms:.1f}ms   compile: {compile_s:.1f}s   device: {run_ms:.2f}ms  ({run_ms/S*1000:.2f}us/tok)")
    print(f"  cos_sim: {cos_sim:.6f}, max_diff: {max_diff:.6f}")
    print(f"  {'PASS' if cos_sim >= 0.999 else 'FAIL'}")
    return cos_sim >= 0.999


if __name__ == "__main__":
    results = []
    for S in [512, 1024, 2048, 4096, 8192]:
        try:
            results.append((f"S={S}", test_seq_len(S)))
        except Exception as e:
            print(f"  FAILED with {type(e).__name__}: {str(e)[:200]}")
            results.append((f"S={S}", False))
    print("\nSUMMARY")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
