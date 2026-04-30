"""
Unit test for nki_flash_attn_mimo kernel.

Validates:
  1. Basic d_qk=192, d_v=128 with causal masking
  2. GQA (multiple Q heads per KV head)
  3. Sliding window attention
  4. Standard d_qk=128 (regression check)
  5. Batch > 1

Run on trn2:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    python3 test_nki_flash_attn_mimo.py
"""

import os

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import math
import time
import torch
import torch.nn.functional as F


def reference_attention(q, k, v, scale, use_causal_mask=True, sliding_window=0):
    """
    CPU reference attention for MiMo's asymmetric heads.

    Args:
        q: (bsz, num_heads, seqlen_q, d_qk) -- float32
        k: (bsz, num_kv_heads, seqlen_k, d_qk) -- float32
        v: (bsz, num_kv_heads, seqlen_k, d_v) -- float32
        scale: float
        use_causal_mask: bool
        sliding_window: int (0 = disabled)

    Returns:
        output: (bsz, num_heads, seqlen_q, d_v) -- float32
    """
    bsz, num_heads, seqlen_q, d_qk = q.shape
    _, num_kv_heads, seqlen_k, d_v = v.shape
    gqa_ratio = num_heads // num_kv_heads

    outputs = []
    for h in range(num_heads):
        kv_h = h // gqa_ratio
        q_h = q[:, h : h + 1]
        k_h = k[:, kv_h : kv_h + 1]
        v_h = v[:, kv_h : kv_h + 1]

        attn = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale

        if use_causal_mask:
            mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))

        if sliding_window > 0 and use_causal_mask:
            row_idx = torch.arange(seqlen_q).unsqueeze(1)
            col_idx = torch.arange(seqlen_k).unsqueeze(0)
            sw_mask = (row_idx - col_idx) >= sliding_window
            attn = attn.masked_fill(sw_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        out = torch.matmul(attn, v_h)
        outputs.append(out)

    return torch.cat(outputs, dim=1)


def run_test(
    label,
    bsz,
    num_heads,
    num_kv_heads,
    seqlen,
    d_qk,
    d_v,
    use_causal_mask=True,
    sliding_window=0,
):
    """Run a single test case."""
    print(f"\n=== {label} ===")
    print(
        f"  bsz={bsz}, heads={num_heads}, kv_heads={num_kv_heads}, "
        f"seq={seqlen}, d_qk={d_qk}, d_v={d_v}, "
        f"causal={use_causal_mask}, sw={sliding_window}"
    )

    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(d_qk)

    q = torch.randn(bsz, num_heads, seqlen, d_qk, dtype=torch.bfloat16)
    k = torch.randn(bsz, num_kv_heads, seqlen, d_qk, dtype=torch.bfloat16)
    v = torch.randn(bsz, num_kv_heads, seqlen, d_v, dtype=torch.bfloat16)

    # CPU reference
    ref = reference_attention(
        q.float(),
        k.float(),
        v.float(),
        scale,
        use_causal_mask=use_causal_mask,
        sliding_window=sliding_window,
    )

    # NKI kernel
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    q_dev = q.to(device)
    k_dev = k.to(device)
    v_dev = v.to(device)

    from nki_flash_attn_mimo import flash_attn_mimo_wrapper

    t0 = time.time()
    out = flash_attn_mimo_wrapper(
        q_dev,
        k_dev,
        v_dev,
        scale=scale,
        use_causal_mask=use_causal_mask,
        sliding_window=sliding_window,
    )
    xm.mark_step()
    out_cpu = out.cpu().float()
    t1 = time.time()

    # Metrics
    cos = F.cosine_similarity(
        ref.reshape(-1).unsqueeze(0),
        out_cpu.reshape(-1).unsqueeze(0),
    ).item()
    maxd = (ref - out_cpu).abs().max().item()
    meand = (ref - out_cpu).abs().mean().item()

    passed = cos > 0.999
    print(f"  Time: {t1 - t0:.1f}s (includes compile)")
    print(f"  Cosine sim: {cos:.6f}")
    print(f"  Max diff:   {maxd:.6f}")
    print(f"  Mean diff:  {meand:.6f}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = []

    # Test 1: Basic d_qk=192, d_v=128 (MiMo's actual config)
    results.append(
        run_test(
            "MiMo basic: d_qk=192, d_v=128, causal",
            bsz=1,
            num_heads=1,
            num_kv_heads=1,
            seqlen=512,
            d_qk=192,
            d_v=128,
        )
    )

    # Test 2: GQA 4:1 (num_heads=4, kv_heads=1)
    results.append(
        run_test(
            "MiMo GQA 4:1",
            bsz=1,
            num_heads=4,
            num_kv_heads=1,
            seqlen=512,
            d_qk=192,
            d_v=128,
        )
    )

    # Test 3: Longer sequence
    results.append(
        run_test(
            "MiMo seq=1024",
            bsz=1,
            num_heads=1,
            num_kv_heads=1,
            seqlen=1024,
            d_qk=192,
            d_v=128,
        )
    )

    # Test 4: d_qk=128 (standard, should still work)
    results.append(
        run_test(
            "Standard d_qk=d_v=128",
            bsz=1,
            num_heads=1,
            num_kv_heads=1,
            seqlen=512,
            d_qk=128,
            d_v=128,
        )
    )

    # Test 5: Sliding window
    results.append(
        run_test(
            "MiMo SWA (window=256)",
            bsz=1,
            num_heads=1,
            num_kv_heads=1,
            seqlen=512,
            d_qk=192,
            d_v=128,
            sliding_window=256,
        )
    )

    # Test 6: Batch > 1 with GQA
    results.append(
        run_test(
            "MiMo batch=2, GQA 2:1",
            bsz=2,
            num_heads=2,
            num_kv_heads=1,
            seqlen=512,
            d_qk=192,
            d_v=128,
        )
    )

    # Test 7: Non-causal (should work but unusual for MiMo)
    results.append(
        run_test(
            "MiMo non-causal",
            bsz=1,
            num_heads=1,
            num_kv_heads=1,
            seqlen=512,
            d_qk=192,
            d_v=128,
            use_causal_mask=False,
        )
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        for i, r in enumerate(results):
            if not r:
                print(f"  Test {i + 1} FAILED")
