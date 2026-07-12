"""Test the chunked kernel at Nemotron shape: D=64, N=128, Q=128."""
import sys, time, torch
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


def build_log_mask(Q=128):
    m = torch.full((Q, Q), -1e30, dtype=torch.float32)
    for i in range(Q):
        for j in range(i + 1):
            m[i, j] = 0.0
    return m


class W(torch.nn.Module):
    def __init__(self, log_mask):
        super().__init__()
        self.register_buffer("log_mask", log_mask)
    def forward(self, x, dA, dB, C, D_x, cla, la):
        from nki_mamba2_ssd_chunked import mamba2_ssd_chunked_fwd
        y, s = mamba2_ssd_chunked_fwd(x, dA, dB, C, D_x, cla, la, self.log_mask)
        return y, s


def test(S, D, N):
    import torch_neuronx
    torch.manual_seed(S * 13 + D)
    x = torch.randn(S, D) * 0.1
    a = torch.rand(S) * 0.9 + 0.05
    dA = torch.zeros(S, N)  # unused
    dB = torch.randn(S, N) * 0.1
    C = torch.randn(S, N) * 0.1
    D_x = 0.5 * x
    log_a = torch.log(a); cum_log_a = torch.cumsum(log_a, dim=0)
    log_mask = build_log_mask(128)

    print(f"\n===== S={S}, D={D}, N={N} =====")
    y_ref, s_ref = naive_recurrent_scan(x, a, dB, C, D_x)

    wrapper = W(log_mask)
    t0 = time.perf_counter()
    try:
        traced = torch_neuronx.trace(
            wrapper, example_inputs=(x, dA, dB, C, D_x, cum_log_a, log_a),
            compiler_args=["--auto-cast=none"],
        )
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {str(e)[:300]}")
        return False
    compile_s = time.perf_counter() - t0

    y_out, s_out = traced(x, dA, dB, C, D_x, cum_log_a, log_a)
    y_cpu = y_out.cpu().float(); s_cpu = s_out.cpu().float()

    y_cos = F.cosine_similarity(y_cpu.flatten(), y_ref.flatten(), dim=0).item()
    y_maxd = (y_cpu - y_ref).abs().max().item()
    # s_ref is (D, N); s_cpu (from kernel) is (N, D) -- transpose s_ref to (N, D)
    s_ref_NxD = s_ref.T
    s_cos = F.cosine_similarity(s_cpu.flatten(), s_ref_NxD.flatten(), dim=0).item()
    s_maxd = (s_cpu - s_ref_NxD).abs().max().item()
    print(f"  compile: {compile_s:.1f}s")
    print(f"  y: cos={y_cos:.6f}, max_diff={y_maxd:.6f}")
    print(f"  state: cos={s_cos:.6f}, max_diff={s_maxd:.6f}")
    passed = y_cos >= 0.999 and s_cos >= 0.999
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = []
    # Nemotron shape
    results.append((f"S=128 D=64 N=128 (Nemotron)", test(128, 64, 128)))
    results.append((f"S=512 D=64 N=128", test(512, 64, 128)))
    results.append((f"S=1024 D=64 N=128", test(1024, 64, 128)))
    # Also test D=128 to be sure not broken
    results.append((f"S=128 D=128 N=128 (regression)", test(128, 128, 128)))
    print("\nSUMMARY")
    for n, p in results:
        print(f"  {'PASS' if p else 'FAIL'}  {n}")
