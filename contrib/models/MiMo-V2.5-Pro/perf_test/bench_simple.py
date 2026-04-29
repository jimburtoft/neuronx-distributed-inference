#!/usr/bin/env python3
"""Simple benchmark for MiMo-V2-Pro FP8 on Trn2.

Uses the same generation approach as smoke_generate (which produced coherent
output at 1.30 tok/s), but adds proper timing measurement.

Runs each test independently with explicit flush after every print.
Falls back gracefully if any test fails.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python3 -u bench_simple.py 2>&1 | tee /home/ubuntu/bench_simple.log
"""

import os
import sys
import time
import traceback
import json

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_PATH = os.environ.get(
    "MIMO_V2_PRO_MODEL_PATH",
    "/home/ubuntu/models/MiMo-V2.5-Pro-Neuron-FP8",
)
COMPILED_PATH = os.environ.get(
    "MIMO_V2_PRO_COMPILED_PATH",
    "/mnt/models/compiled/mimo_v2_pro_tp64_moetp1_ep64_fp8/",
)

TP_DEGREE = int(os.environ.get("TP_DEGREE", "64"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "48"))
CTX_BATCH_SIZE = int(os.environ.get("CTX_BATCH_SIZE", "1"))
MOE_TP = int(os.environ.get("MOE_TP", "1"))
MOE_EP = int(os.environ.get("MOE_EP", "64"))

os.environ.setdefault(
    "BASE_COMPILE_WORK_DIR",
    os.path.join("/tmp/nxd_model", os.path.basename(COMPILED_PATH.rstrip("/"))),
)


def log(msg):
    print(msg, flush=True)


def build_model():
    from transformers import AutoConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    contrib_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "src",
    )
    sys.path.insert(0, os.path.abspath(contrib_src))

    from modeling_mimo_v2 import MiMoV2InferenceConfig, NeuronMiMoV2ForCausalLM

    bwmm_config = {
        "use_shard_on_intermediate_dynamic_while": True,
        "skip_dma_token": True,
    }

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        ep_degree=1,
        logical_nc_config=2,
        batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        ctx_batch_size=CTX_BATCH_SIZE,
        tkg_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        n_active_tokens=128,
        torch_dtype="bfloat16",
        capacity_factor=1.0,
        glu_mlp=True,
        is_continuous_batching=True,
        moe_ep_degree=MOE_EP,
        moe_tp_degree=MOE_TP,
        context_encoding_buckets=[SEQ_LEN],
        router_config={"act_fn": "sigmoid", "dtype": "float32"},
        blockwise_matmul_config=bwmm_config,
        save_sharded_checkpoint=True,
        quantized=True,
        quantized_checkpoints_path=MODEL_PATH,
        quantization_dtype="f8e4m3",
        quantization_type="blockwise_symmetric",
        quantization_block_axis=[1, 2],
        quantization_block_size=[128, 128],
        modules_to_not_convert=[
            "embed_tokens",
            "lm_head",
            "norm",
            "router",
            "o_proj",
        ],
    )

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = MiMoV2InferenceConfig(
        neuron_config, load_config=load_pretrained_config(hf_config=hf_config)
    )

    model = NeuronMiMoV2ForCausalLM(MODEL_PATH, config)
    return model


def run_test(adapter, tokenizer, prompt_text, max_new_tokens, label, num_runs=3):
    """Run a single benchmark test with explicit error handling."""
    from transformers import GenerationConfig

    log(f"\n--- Test: {label} ---")
    log(f"  Prompt: {prompt_text[:80]!r}")
    log(f"  BS={BATCH_SIZE}, max_new_tokens={max_new_tokens}")

    # Tokenize: replicate single prompt to fill batch
    inputs = tokenizer(
        [prompt_text] * BATCH_SIZE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN - max_new_tokens,
    )
    prompt_len = inputs["input_ids"].shape[1]
    log(f"  prompt_len={prompt_len} tokens")

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
    )

    # Warmup
    log(f"  Warmup...")
    t0 = time.time()
    warmup_ids = adapter.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=gen_config,
    )
    warmup_time = time.time() - t0
    actual_new = warmup_ids.shape[1] - prompt_len
    log(f"  Warmup done: {warmup_time:.2f}s, generated {actual_new} tokens")

    # Decode warmup output for quality check
    new_tokens = warmup_ids[0, prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    log(f"  Output: {decoded[:200]!r}")

    # Timed runs
    times = []
    for i in range(num_runs):
        log(f"  Run {i + 1}/{num_runs}...")
        t0 = time.time()
        output_ids = adapter.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_config,
        )
        elapsed = time.time() - t0
        actual_new = output_ids.shape[1] - prompt_len
        times.append(elapsed)
        log(f"    {elapsed:.2f}s, {actual_new} tokens generated")

    best = min(times)
    # Use actual_new from last run for tok/s calculation
    total_toks = actual_new * BATCH_SIZE
    tps = total_toks / best
    tpot_ms = best / actual_new * 1000 if actual_new > 0 else 0

    result = {
        "label": label,
        "batch_size": BATCH_SIZE,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "actual_new_tokens": actual_new,
        "best_time_s": round(best, 2),
        "all_times_s": [round(t, 2) for t in times],
        "total_tok_s": round(tps, 2),
        "per_stream_tok_s": round(tps / BATCH_SIZE, 2),
        "tpot_ms": round(tpot_ms, 2),
        "sample_output": decoded[:200],
    }

    log(
        f"  RESULT: {tps:.1f} tok/s total | {tps / BATCH_SIZE:.1f} tok/s/stream | TPOT={tpot_ms:.1f}ms"
    )
    log(f"  Times: {[f'{t:.2f}' for t in times]}")

    return result


def main():
    from transformers import AutoTokenizer
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )

    log("=" * 60)
    log("MiMo-V2-Pro FP8 NxDI Simple Benchmark")
    log("=" * 60)
    log(f"MODEL_PATH={MODEL_PATH}")
    log(f"COMPILED_PATH={COMPILED_PATH}")
    log(f"TP={TP_DEGREE}, BS={BATCH_SIZE}, SEQ={SEQ_LEN}")
    log(f"MOE_TP={MOE_TP}, MOE_EP={MOE_EP}")
    log(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    # Build model
    log("[bench] Building model...")
    model = build_model()

    log(f"[bench] Loading from {COMPILED_PATH}")
    t0 = time.time()
    model.load(COMPILED_PATH, skip_warmup=False)
    load_time = time.time() - t0
    log(f"[bench] Loaded in {load_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    adapter = HuggingFaceGenerationAdapter(model)

    results = [
        {
            "load_time_s": round(load_time, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]

    # Define tests: (prompt, max_new_tokens, label)
    tests = [
        ("Hello! Please introduce yourself in one sentence.", 50, "short_50tok"),
        ("Hello! Please introduce yourself in one sentence.", 128, "short_128tok"),
        (
            "Explain in detail what a transformer neural network is, how attention "
            "works, and why it has been so successful in natural language processing tasks.",
            128,
            "medium_128tok",
        ),
        (
            "Write a short story about a robot learning to paint.",
            256,
            "creative_256tok",
        ),
        ("请用中文介绍一下你自己。", 128, "chinese_128tok"),
        (
            "Write a Python function to compute the nth Fibonacci number efficiently "
            "using memoization.",
            128,
            "code_128tok",
        ),
    ]

    # Also try chat template
    try:
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2+2? Think step by step."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tests.append((chat_prompt, 128, "chat_template_128tok"))
    except Exception as e:
        log(f"[bench] Chat template not available: {e}")

    # Run each test independently
    for prompt, max_new, label in tests:
        try:
            r = run_test(adapter, tokenizer, prompt, max_new, label)
            results.append(r)
        except Exception as e:
            log(f"\n  [{label}] FAILED: {e}")
            traceback.print_exc()
            sys.stdout.flush()

    # Summary table
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(
        f"{'Test':<25} {'BS':>4} {'Gen':>5} {'Actual':>7} {'Total tok/s':>12} {'Per-stream':>12} {'TPOT ms':>10}"
    )
    log("-" * 80)
    for r in results:
        if "label" in r:
            log(
                f"{r['label']:<25} {r['batch_size']:>4} {r['max_new_tokens']:>5} "
                f"{r['actual_new_tokens']:>7} {r['total_tok_s']:>12.1f} "
                f"{r['per_stream_tok_s']:>12.1f} {r['tpot_ms']:>10.1f}"
            )

    # Save JSON results
    out_path = "/home/ubuntu/bench_results_simple.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log("[bench] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
