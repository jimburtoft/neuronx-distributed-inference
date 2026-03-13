#!/usr/bin/env python3
"""
Compile the unified CausalWan model (single-NEFF architecture, single model.pt).

V6 Single-NEFF: Only the cached attention NEFF is compiled. Self-attention and
update calls are routed through the cached NEFF with zero KV buffers and
appropriate masks. This eliminates Neuron compiler numerical non-determinism
between separate NEFFs.

This uses the "cache then bundle" strategy:
  Phase 1: (Optional) Compile cached_f15 bucket to warm neuronx-cc cache
  Phase 2: Run unified compile() → single model.pt with 1 NEFF
  Phase 3: Test loading + inference (self, cached, update-via-cached)

Usage:
    # Phase 2: compile unified model.pt
    python compile_transformer.py --weight-path /path/to/wan_diffusers_weights \\
        --compiled-path /path/to/output/compiled_model --phase 2

    # Phase 3: test compiled model
    python compile_transformer.py --weight-path /path/to/wan_diffusers_weights \\
        --compiled-path /path/to/output/compiled_model --phase 3

    # All phases:
    python compile_transformer.py --weight-path /path/to/wan_diffusers_weights \\
        --compiled-path /path/to/output/compiled_model --phase all
"""

import os
import sys
import time
import argparse
import subprocess
import concurrent.futures

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"


def phase1_compile_individual(tp_degree):
    """Phase 1 (Optional): Compile cached_f15 bucket via subprocess.

    This phase is optional and exists to warm the neuronx-cc compiler cache
    before the main compilation in Phase 2. It may reference external scripts
    (e.g., compile_nxdi.py) that are not included in this contrib package.
    Skip this phase if the compiler cache is already populated or if the
    referenced scripts are not available.

    Single-NEFF architecture: only the cached attention NEFF is compiled.
    Self-attention and update calls route through this same NEFF.
    """
    print("=" * 70)
    print("PHASE 1: Individual bucket compilation (populate neuronx-cc cache)")
    print("  Single-NEFF: cached_f15 only")
    print("=" * 70)

    # Only 1 bucket needed for the single-NEFF architecture
    buckets = [
        ("cached", 15),
    ]

    for mode, frames in buckets:
        print(f"\n--- Compiling {mode}_f{frames} ---")
        t0 = time.time()

        cmd = [
            sys.executable,
            "compile_nxdi.py",
            "--mode",
            mode,
            "--frames",
            str(frames),
            "--tp",
            str(tp_degree),
            "--compile",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  PASS ({elapsed:.1f}s)")
        else:
            print(f"  FAIL ({elapsed:.1f}s)")
            print(f"  stderr: {result.stderr[-500:]}")

    print(f"\nPhase 1 complete")


def _patch_threadpool_max_workers(max_workers=1):
    """Monkey-patch ThreadPoolExecutor to limit parallel workers."""
    original_init = concurrent.futures.ThreadPoolExecutor.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["max_workers"] = max_workers
        original_init(self, *args, **kwargs)

    concurrent.futures.ThreadPoolExecutor.__init__ = patched_init
    return original_init


def _unpatch_threadpool(original_init):
    """Restore original ThreadPoolExecutor."""
    concurrent.futures.ThreadPoolExecutor.__init__ = original_init


def phase2_bundle(weight_path, compiled_path, tp_degree):
    """Phase 2: Compile unified model.pt with all 11 buckets."""
    print("=" * 70)
    print("PHASE 2: Unified model.pt compilation (all modes, shared weights)")
    print("=" * 70)

    from modeling_wan21 import (
        NeuronCausalWanUnifiedApplication,
        create_unified_causal_wan_config,
    )

    config = create_unified_causal_wan_config(tp_degree=tp_degree)
    app = NeuronCausalWanUnifiedApplication(
        model_path=weight_path,
        config=config,
    )

    # Monkey-patch to prevent OOM from parallel compilation
    original_init = _patch_threadpool_max_workers(max_workers=1)

    try:
        print(f"\nCompiling to {compiled_path}...")
        t0 = time.time()
        app.compile(compiled_path)
        elapsed = time.time() - t0
        print(f"\nPhase 2 complete in {elapsed:.1f}s")
    finally:
        _unpatch_threadpool(original_init)

    # Check output
    model_pt = os.path.join(compiled_path, "model.pt")
    if os.path.exists(model_pt):
        size_mb = os.path.getsize(model_pt) / 1e6
        print(f"  model.pt: {size_mb:.0f} MB")
    else:
        print("  ERROR: model.pt not created!")


def phase3_test(weight_path, compiled_path, tp_degree):
    """Phase 3: Load unified model and test all modes through single NEFF.

    Single-NEFF: all calls route through the cached NEFF.
    - Self: zero KV buffers + current-only mask
    - Cached: anchor KV in buffer + appropriate mask
    - Update-via-cached: padded f15 with full cache KV
    """
    print("=" * 70)
    print("PHASE 3: Test unified model loading + inference (single-NEFF)")
    print("=" * 70)

    import torch
    import math
    from modeling_wan21 import (
        NeuronCausalWanUnifiedApplication,
        create_unified_causal_wan_config,
        make_freqs,
        precompute_rope_embeddings,
        FRAME_SEQ_LENGTH,
        NUM_LAYERS,
        NUM_HEADS,
        HEAD_DIM,
        MAX_ATTENTION_SIZE,
        PATCH_T,
        PATCH_H,
        PATCH_W,
        TEXT_SEQ_LEN,
        TEXT_DIM,
    )

    config = create_unified_causal_wan_config(tp_degree=tp_degree)
    app = NeuronCausalWanUnifiedApplication(
        model_path=weight_path,
        config=config,
    )

    print(f"\nLoading unified model from {compiled_path}...")
    t0 = time.time()
    app.load(compiled_path)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    dtype = torch.bfloat16
    lat_h, lat_w = 60, 104
    heads_per_rank = math.ceil(NUM_HEADS / tp_degree) * tp_degree // tp_degree
    base_freqs = make_freqs(HEAD_DIM)

    # --- Test self mode (routed through cached NEFF with zero KV) ---
    print(f"\n--- Self mode test (f15, zero KV + current-only mask) ---")
    num_frames = 15
    hidden = torch.randn(1, 16, num_frames, lat_h, lat_w, dtype=dtype)
    timestep = torch.full(
        (1, num_frames), 500.0, dtype=torch.float32
    )  # V11: float32 timestep
    enc = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)

    post_f = num_frames // PATCH_T
    post_h = lat_h // PATCH_H
    post_w = lat_w // PATCH_W
    seq_len = post_f * post_h * post_w
    rope_cos, rope_sin = precompute_rope_embeddings(base_freqs, post_f, post_h, post_w)
    rope_cos = rope_cos.to(torch.float32)
    rope_sin = rope_sin.to(torch.float32)

    # Self-mode mask for cached NEFF: [1, 1, seq_len, MAX_ATTENTION_SIZE]
    # Only unmask current positions at prefix offset
    prefix_len = MAX_ATTENTION_SIZE - seq_len
    attn_mask_self = torch.full(
        (1, 1, seq_len, MAX_ATTENTION_SIZE),
        torch.finfo(dtype).min,
        dtype=dtype,
    )
    attn_mask_self[:, :, :seq_len, prefix_len : prefix_len + seq_len] = 0.0

    # Zero KV buffers
    kv_tensors_zero = []
    for _ in range(NUM_LAYERS):
        kv_tensors_zero.append(
            torch.zeros(1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype)
        )
        kv_tensors_zero.append(
            torch.zeros(1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype)
        )

    t1 = time.time()
    with torch.no_grad():
        outputs = app.forward_self(
            hidden,
            timestep,
            enc,
            rope_cos,
            rope_sin,
            attn_mask_self,
            *kv_tensors_zero,
        )
    t_self = time.time() - t1

    output = outputs[0]
    print(
        f"  self_f{num_frames}: output {output.shape}, "
        f"range [{output.min():.2f}, {output.max():.2f}], "
        f"time {t_self:.2f}s PASS"
    )

    # --- Test cached mode (f15 with KV buffers) ---
    print(f"\n--- Cached mode test (f15) ---")
    num_frames = 15
    hidden = torch.randn(1, 16, num_frames, lat_h, lat_w, dtype=dtype)
    timestep = torch.full(
        (1, num_frames), 500.0, dtype=torch.float32
    )  # V11: float32 timestep
    enc = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)

    post_f = num_frames // PATCH_T
    post_h = lat_h // PATCH_H
    post_w = lat_w // PATCH_W
    seq_len = post_f * post_h * post_w
    rope_cos, rope_sin = precompute_rope_embeddings(base_freqs, post_f, post_h, post_w)
    rope_cos = rope_cos.to(torch.float32)
    rope_sin = rope_sin.to(torch.float32)

    attn_mask = torch.zeros(1, 1, seq_len, MAX_ATTENTION_SIZE, dtype=dtype)
    kv_tensors = []
    for _ in range(NUM_LAYERS):
        kv_tensors.append(
            torch.randn(1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype)
        )
        kv_tensors.append(
            torch.randn(1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype)
        )

    t1 = time.time()
    with torch.no_grad():
        outputs = app.forward_cached(
            hidden, timestep, enc, rope_cos, rope_sin, attn_mask, *kv_tensors
        )
    t_cached = time.time() - t1

    output = outputs[0]
    print(
        f"  cached_f{num_frames}: output {output.shape}, "
        f"range [{output.min():.2f}, {output.max():.2f}], "
        f"time {t_cached:.2f}s PASS"
    )

    # --- Test "update via cached" (f3 padded to f15) ---
    print(f"\n--- Update-via-cached test (f3 padded to f15) ---")
    num_frames_padded = 15
    hidden_padded = torch.zeros(1, 16, num_frames_padded, lat_h, lat_w, dtype=dtype)
    hidden_padded[:, :, :3] = torch.randn(1, 16, 3, lat_h, lat_w, dtype=dtype)
    timestep = torch.full(
        (1, num_frames_padded), 0.0, dtype=torch.float32
    )  # V11: float32 timestep
    enc = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)

    post_f = num_frames_padded // PATCH_T
    seq_len = post_f * (lat_h // PATCH_H) * (lat_w // PATCH_W)
    real_seq_len = (3 // PATCH_T) * (lat_h // PATCH_H) * (lat_w // PATCH_W)  # 4680
    rope_cos, rope_sin = precompute_rope_embeddings(
        base_freqs, post_f, lat_h // PATCH_H, lat_w // PATCH_W
    )
    rope_cos = rope_cos.to(torch.float32)
    rope_sin = rope_sin.to(torch.float32)

    # Mask: real Q rows attend to valid KV, padding Q rows fully masked
    valid_kv_len = 9360  # example: anchor(4680) + working(4680)
    attn_mask = torch.full(
        (1, 1, seq_len, MAX_ATTENTION_SIZE),
        torch.finfo(dtype).min,
        dtype=dtype,
    )
    attn_mask[:, :, :real_seq_len, :valid_kv_len] = 0.0

    kv_tensors = []
    for _ in range(NUM_LAYERS):
        kv_tensors.append(
            torch.randn(1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype)
        )
        kv_tensors.append(
            torch.randn(1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype)
        )

    t1 = time.time()
    with torch.no_grad():
        outputs = app.forward_cached(
            hidden_padded, timestep, enc, rope_cos, rope_sin, attn_mask, *kv_tensors
        )
    t_update = time.time() - t1

    output = outputs[0]
    print(
        f"  update-via-cached: output {output.shape}, "
        f"range [{output.min():.2f}, {output.max():.2f}], "
        f"time {t_update:.2f}s PASS"
    )

    print(f"\n{'=' * 70}")
    print(f"ALL TESTS PASS — Single-NEFF unified model loaded")
    print(f"  Load time: {load_time:.1f}s (shared weights, 1 NEFF)")
    print(f"  Self-via-cached: {t_self:.2f}s")
    print(f"  Cached inference: {t_cached:.2f}s")
    print(f"  Update-via-cached: {t_update:.2f}s")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Compile unified CausalWan model")
    parser.add_argument(
        "--weight-path",
        required=True,
        help="Path to Wan diffusers weights directory",
    )
    parser.add_argument(
        "--compiled-path",
        required=True,
        help="Output path for compiled model",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=4,
        help="Tensor parallelism degree (default: 4)",
    )
    parser.add_argument("--phase", default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        phase1_compile_individual(args.tp_degree)
    if args.phase in ("2", "all"):
        phase2_bundle(args.weight_path, args.compiled_path, args.tp_degree)
    if args.phase in ("3", "all"):
        phase3_test(args.weight_path, args.compiled_path, args.tp_degree)


if __name__ == "__main__":
    main()
