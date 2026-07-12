"""Test the chunked Mamba-2 SSD NKI kernel.

Tests:
  A. Correctness: chunked kernel vs naive recurrent (both PyTorch), then vs NKI recurrent.
     Multiple seq lengths.
  B. Multi-invocation compile: stress the DGE budget with realistic layer counts.
  C. Performance vs recurrent-only kernel.
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
    """Return (log_a, cum_log_a). a is (S,) per-token decay."""
    log_a = torch.log(a)
    cum_log_a = torch.cumsum(log_a, dim=0)
    return log_a, cum_log_a


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


def build_log_mask(Q=128):
    """(Q, Q) mask: 0 where j<=i, -1e30 elsewhere. Added to log_A_ij before exp()."""
    m = torch.full((Q, Q), -1e30, dtype=torch.float32)
    for i in range(Q):
        for j in range(i + 1):  # j <= i
            m[i, j] = 0.0
    return m


def test_chunked_correctness(seq_len):
    """Test chunked kernel numerical equivalence to naive recurrent."""
    import torch_neuronx
    D = 128; N = 128; Q = 128
    torch.manual_seed(seq_len + 7)

    x = torch.randn(seq_len, D) * 0.1
    a_scalar = torch.rand(seq_len) * 0.9 + 0.05
    dA = a_scalar.unsqueeze(-1).expand(seq_len, N).contiguous()
    dB = torch.randn(seq_len, N) * 0.1
    C = torch.randn(seq_len, N) * 0.1
    D_x = 0.5 * x

    log_a, cum_log_a = precompute_scan_aux(a_scalar)
    log_mask = build_log_mask(Q)

    print(f"\n===== Chunked correctness, seq_len={seq_len} =====")
    t0 = time.perf_counter()
    y_ref, _ = naive_recurrent_scan(x, a_scalar, dB, C, D_x)
    ref_ms = (time.perf_counter() - t0) * 1000

    wrapper = ChunkedSsdWrapper(log_mask)
    t0 = time.perf_counter()
    try:
        traced = torch_neuronx.trace(
            wrapper,
            example_inputs=(x, dA, dB, C, D_x, cum_log_a, log_a),
            compiler_args=["--auto-cast=none"],
        )
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {str(e)[:300]}")
        return False
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = traced(x, dA, dB, C, D_x, cum_log_a, log_a)
    run_ms = (time.perf_counter() - t0) * 1000

    out_cpu = out.cpu().float()
    diff = (out_cpu - y_ref).abs()
    max_diff = diff.max().item()
    cos_sim = F.cosine_similarity(out_cpu.flatten(), y_ref.flatten(), dim=0).item()

    print(f"  ref: {ref_ms:.1f}ms   compile: {compile_s:.1f}s   device: {run_ms:.1f}ms")
    print(f"  cos_sim: {cos_sim:.6f}, max_diff: {max_diff:.6f}")
    if cos_sim < 0.99:
        # Show first few diffs
        print(f"  First 3 rows (ref vs out):")
        for i in [0, 1, 2, seq_len // 2, seq_len - 1]:
            r = y_ref[i, :4].tolist()
            o = out_cpu[i, :4].tolist()
            print(f"    t={i}: ref={[f'{v:.4f}' for v in r]},  out={[f'{v:.4f}' for v in o]}")
    passed = cos_sim >= 0.999 and max_diff <= 1e-2
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


class MultiInvChunkedWrapper(torch.nn.Module):
    def __init__(self, log_mask, num_layers, num_heads):
        super().__init__()
        self.register_buffer("log_mask", log_mask)
        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(self, x_all, dA_all, dB_all, C_all, D_x_all, cla_all, la_all):
        from nki_mamba2_ssd_chunked import mamba2_ssd_chunked_fwd
        outs = []
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                y, _ = mamba2_ssd_chunked_fwd(
                    x_all[l, h], dA_all[l, h], dB_all[l, h],
                    C_all[l, h], D_x_all[l, h],
                    cla_all[l, h], la_all[l, h], self.log_mask,
                )
                outs.append(y)
        return torch.stack(outs)


def test_chunked_multi_inv(num_layers, num_heads, seq_len=256):
    """Stress: N kernel invocations in one traced NEFF (DGE budget test)."""
    import torch_neuronx
    D = 128; N = 128; Q = 128; S = seq_len
    torch.manual_seed(num_heads * 100 + num_layers + 33)

    x_all = torch.randn(num_layers, num_heads, S, D) * 0.1
    a_scalar_all = torch.rand(num_layers, num_heads, S) * 0.9 + 0.05
    dA_all = a_scalar_all.unsqueeze(-1).expand(num_layers, num_heads, S, N).contiguous()
    dB_all = torch.randn(num_layers, num_heads, S, N) * 0.1
    C_all = torch.randn(num_layers, num_heads, S, N) * 0.1
    D_x_all = 0.5 * x_all

    # precompute cum_log_a per (l, h)
    la_all = torch.log(a_scalar_all)
    cla_all = torch.cumsum(la_all, dim=-1)

    log_mask = build_log_mask(Q)

    inv = num_layers * num_heads
    print(f"\n===== Chunked multi-inv: {num_layers}L x {num_heads}H = {inv} invocations, S={S} =====")

    ref = torch.zeros(num_layers, num_heads, S, D)
    for l in range(num_layers):
        for h in range(num_heads):
            ref[l, h], _ = naive_recurrent_scan(
                x_all[l, h], a_scalar_all[l, h], dB_all[l, h],
                C_all[l, h], D_x_all[l, h],
            )
    ref_flat = ref.reshape(-1, S, D)

    wrapper = MultiInvChunkedWrapper(log_mask, num_layers, num_heads)
    t0 = time.perf_counter()
    try:
        traced = torch_neuronx.trace(
            wrapper,
            example_inputs=(x_all, dA_all, dB_all, C_all, D_x_all, cla_all, la_all),
            compiler_args=["--auto-cast=none"],
        )
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {str(e)[:300]}")
        return False
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = traced(x_all, dA_all, dB_all, C_all, D_x_all, cla_all, la_all)
    run_ms = (time.perf_counter() - t0) * 1000

    out_cpu = out.cpu().float()
    cos_sim = F.cosine_similarity(out_cpu.flatten(), ref_flat.flatten(), dim=0).item()
    max_diff = (out_cpu - ref_flat).abs().max().item()

    print(f"  Compile: {compile_s:.1f}s")
    print(f"  Device: {run_ms:.1f}ms  ({run_ms/inv:.2f}ms/inv, {run_ms/(inv*S):.4f}ms/tok/inv)")
    print(f"  cos_sim: {cos_sim:.6f}, max_diff: {max_diff:.6f}")
    print(f"  {'PASS' if cos_sim >= 0.999 else 'FAIL'}")
    return cos_sim >= 0.999


if __name__ == "__main__":
    results = []

    # A: correctness at multiple seq lengths (all multiples of 128)
    print("=" * 70)
    print("A: chunked kernel correctness")
    print("=" * 70)
    for S in [128, 256, 512, 1024]:
        results.append((f"chunked_S={S}", test_chunked_correctness(S)))

    # B: DGE stress
    print("\n" + "=" * 70)
    print("B: chunked kernel DGE stress")
    print("=" * 70)
    results.append(("chunked_4Lx4H_S=256", test_chunked_multi_inv(4, 4, 256)))
    results.append(("chunked_23Lx4H_S=128", test_chunked_multi_inv(23, 4, 128)))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nOverall: {'ALL PASS' if all(p for _, p in results) else 'SOME FAILED'}")
