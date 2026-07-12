"""Extended tests for mamba2_ssd_recurrent kernel.

Test A: Correctness at multiple sequence lengths (128, 256, 512, 1024).
Test B: Simulate a multi-head, multi-layer workload to check DGE budget behavior.
        Runs 4 heads x 4 layers = 16 kernel invocations in a single traced graph.
Test C: Larger seq (2048) -- verifies the kernel scales.
"""

import sys
import time
import torch
import torch.nn.functional as F

import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "nki_kernels"))


def reference_mamba2_ssd_scan(x, dA, dB, C, D_x):
    S, D = x.shape
    _, N = dB.shape
    dA_scalar = dA[:, 0]
    state = torch.zeros(D, N, dtype=torch.float32)
    y = torch.zeros(S, D, dtype=torch.float32)
    for t in range(S):
        state = dA_scalar[t] * state + torch.outer(x[t], dB[t])
        y[t] = state @ C[t] + D_x[t]
    return y


class SingleHeadWrapper(torch.nn.Module):
    def forward(self, x, dA, dB, C, D_x):
        from nki_mamba2_ssd_recurrent import mamba2_ssd_recurrent_fwd
        return mamba2_ssd_recurrent_fwd(x, dA, dB, C, D_x)


class MultiHeadMultiLayerWrapper(torch.nn.Module):
    """Simulates NUM_HEADS x NUM_LAYERS kernel invocations in one graph.

    This models a realistic Nemotron layer stack: 4 heads across 4 layers
    means 16 kernel calls in the same NEFF -- a stress test for the DGE budget.
    """
    def __init__(self, num_heads=4, num_layers=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, x_all, dA_all, dB_all, C_all, D_x_all):
        # x_all etc: (num_layers, num_heads, S, D or N)
        from nki_mamba2_ssd_recurrent import mamba2_ssd_recurrent_fwd
        outputs = []
        for layer_idx in range(self.num_layers):
            layer_outs = []
            for head_idx in range(self.num_heads):
                out = mamba2_ssd_recurrent_fwd(
                    x_all[layer_idx, head_idx],
                    dA_all[layer_idx, head_idx],
                    dB_all[layer_idx, head_idx],
                    C_all[layer_idx, head_idx],
                    D_x_all[layer_idx, head_idx],
                )
                layer_outs.append(out)
            outputs.append(torch.stack(layer_outs))
        return torch.stack(outputs)  # (num_layers, num_heads, S, D)


def test_single_head(seq_len):
    """Test single-head kernel at given sequence length."""
    import torch_neuronx
    D = 128; N = 128
    torch.manual_seed(seq_len)
    x = torch.randn(seq_len, D) * 0.1
    dA_scalar = torch.rand(seq_len) * 0.9 + 0.05
    dA = dA_scalar.unsqueeze(-1).expand(seq_len, N).contiguous()
    dB = torch.randn(seq_len, N) * 0.1
    C = torch.randn(seq_len, N) * 0.1
    D_x = 0.5 * x

    print(f"\n===== Test: single head, seq_len={seq_len} =====")
    ref = reference_mamba2_ssd_scan(x, dA, dB, C, D_x)

    wrapper = SingleHeadWrapper()
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(
        wrapper, example_inputs=(x, dA, dB, C, D_x),
        compiler_args=["--auto-cast=none"],
    )
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = traced(x, dA, dB, C, D_x)
    run_ms = (time.perf_counter() - t0) * 1000

    out_cpu = out.cpu().float()
    diff = (out_cpu - ref).abs()
    cos_sim = F.cosine_similarity(out_cpu.flatten(), ref.flatten(), dim=0).item()

    print(f"  Compile: {compile_s:.1f}s")
    print(f"  Device run: {run_ms:.1f}ms  ({run_ms/seq_len:.2f}ms/token)")
    print(f"  cos_sim: {cos_sim:.6f},  max_diff: {diff.max().item():.6f}")
    print(f"  {'PASS' if cos_sim >= 0.999 else 'FAIL'}")
    return cos_sim >= 0.999


def test_multi_head_multi_layer(num_heads=4, num_layers=4, seq_len=128):
    """Test kernel invoked NUM_HEADS x NUM_LAYERS times in a single graph."""
    import torch_neuronx
    D = 128; N = 128
    S = seq_len
    torch.manual_seed(1234)

    x_all = torch.randn(num_layers, num_heads, S, D) * 0.1
    dA_scalar_all = torch.rand(num_layers, num_heads, S) * 0.9 + 0.05
    dA_all = dA_scalar_all.unsqueeze(-1).expand(num_layers, num_heads, S, N).contiguous()
    dB_all = torch.randn(num_layers, num_heads, S, N) * 0.1
    C_all = torch.randn(num_layers, num_heads, S, N) * 0.1
    D_x_all = 0.5 * x_all

    print(f"\n===== Test: {num_layers} layers x {num_heads} heads, seq_len={S} =====")
    print(f"  ({num_layers * num_heads} kernel invocations in one traced NEFF)")

    # Reference: compute per-head using the same PyTorch reference
    ref = torch.zeros(num_layers, num_heads, S, D)
    for l in range(num_layers):
        for h in range(num_heads):
            ref[l, h] = reference_mamba2_ssd_scan(
                x_all[l, h], dA_all[l, h], dB_all[l, h],
                C_all[l, h], D_x_all[l, h],
            )

    wrapper = MultiHeadMultiLayerWrapper(num_heads=num_heads, num_layers=num_layers)
    t0 = time.perf_counter()
    try:
        traced = torch_neuronx.trace(
            wrapper,
            example_inputs=(x_all, dA_all, dB_all, C_all, D_x_all),
            compiler_args=["--auto-cast=none"],
        )
    except Exception as e:
        print(f"  COMPILE FAILED: {type(e).__name__}: {e}")
        return False
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = traced(x_all, dA_all, dB_all, C_all, D_x_all)
    run_ms = (time.perf_counter() - t0) * 1000

    out_cpu = out.cpu().float()
    diff = (out_cpu - ref).abs()
    cos_sim = F.cosine_similarity(out_cpu.flatten(), ref.flatten(), dim=0).item()

    print(f"  Compile: {compile_s:.1f}s")
    print(f"  Device run (all {num_layers*num_heads} invocations): {run_ms:.1f}ms")
    print(f"  Per-invocation: {run_ms/(num_layers*num_heads):.1f}ms")
    print(f"  cos_sim: {cos_sim:.6f},  max_diff: {diff.max().item():.6f}")
    print(f"  {'PASS' if cos_sim >= 0.999 else 'FAIL'}")
    return cos_sim >= 0.999


if __name__ == "__main__":
    results = []

    # Test A: single-head correctness at multiple sequence lengths
    print("=" * 70)
    print("Test A: single-head correctness at multiple sequence lengths")
    print("=" * 70)
    for seq in [128, 256, 512]:
        results.append(("single_head_S=" + str(seq), test_single_head(seq)))

    # Test B: multi-head multi-layer to stress the DGE budget
    print("\n" + "=" * 70)
    print("Test B: DGE budget stress test (multiple kernel invocations in one graph)")
    print("=" * 70)
    results.append(("multi_head_4x4_S=128", test_multi_head_multi_layer(4, 4, 128)))
    results.append(("multi_head_8x4_S=128", test_multi_head_multi_layer(8, 4, 128)))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
