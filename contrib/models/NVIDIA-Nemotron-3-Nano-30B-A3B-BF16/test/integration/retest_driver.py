#!/usr/bin/env python3
"""Unified compile + benchmark + quality driver for the 3 known-issue retests.

Usage examples:
    # Test #1: native F.conv1d
    USE_NATIVE_CONV1D=1 python retest_driver.py --ctx 128 --seq 2048 \
        --out-dir /home/ubuntu/artifacts --tag conv1d_native

    # Test #7: sparse MoE prefill
    FORCE_SPARSE_MOE_PREFILL=1 python retest_driver.py --ctx 128 --seq 2048 \
        --out-dir /home/ubuntu/artifacts --tag sparse_prefill

    # Test #8: batch_size=2
    python retest_driver.py --ctx 128 --seq 2048 --batch 2 \
        --out-dir /home/ubuntu/artifacts --tag bs2
"""

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from modeling_nemotron_h import (  # noqa: E402
    NeuronNemotronForCausalLM,
    NemotronHInferenceConfig,
    USE_NATIVE_CONV1D,
    FORCE_SPARSE_MOE_PREFILL,
)


MODEL_PATH = os.environ.get("NEMOTRON_MODEL_PATH", "/mnt/models/nemotron-30b")


def build_config(ctx, seq, batch=1):
    neuron_config = MoENeuronConfig(
        tp_degree=4,
        batch_size=batch,
        max_context_length=ctx,
        seq_len=seq,
        on_device_sampling_config=None,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        torch_dtype="bfloat16",
        save_sharded_checkpoint=True,
    )
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return NemotronHInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )


def compile_or_load(ctx, seq, batch, compiled_path):
    config = build_config(ctx, seq, batch)

    if not (compiled_path / "model.pt").exists():
        print(f"[compile] target {compiled_path}", flush=True)
        t0 = time.perf_counter()
        model = NeuronNemotronForCausalLM(MODEL_PATH, config)
        model.compile(str(compiled_path))
        compile_s = time.perf_counter() - t0
        print(f"[compile] done in {compile_s:.1f}s", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tokenizer.save_pretrained(str(compiled_path))
        del model
        gc.collect()
    else:
        print(f"[compile] cache hit at {compiled_path}", flush=True)
        compile_s = 0.0

    print("[load] loading compiled model...", flush=True)
    t0 = time.perf_counter()
    model = NeuronNemotronForCausalLM(MODEL_PATH, config)
    model.load(str(compiled_path))
    load_s = time.perf_counter() - t0
    print(f"[load] done in {load_s:.1f}s", flush=True)
    return model, compile_s, load_s


def bench_raw(model, tokenizer, batch, num_gen=50):
    """Benchmark on the standard raw-completion prompt(s) for TTFT/decode."""
    prompt = "The capital of France is"
    inputs = tokenizer([prompt] * batch, return_tensors="pt", padding=False)
    adapter = HuggingFaceGenerationAdapter(model)

    # Warmup
    print("[bench] warmup...", flush=True)
    _ = adapter.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=2,
        do_sample=False, temperature=1.0, top_k=1,
    )

    # First-token latency (TTFT)
    t0 = time.perf_counter()
    _ = adapter.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=1,
        do_sample=False, temperature=1.0, top_k=1,
    )
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Full decode (num_gen tokens)
    t0 = time.perf_counter()
    output = adapter.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=num_gen,
        do_sample=False, temperature=1.0, top_k=1,
    )
    total_s = time.perf_counter() - t0
    prompt_len = inputs.input_ids.shape[1]
    total_new = output.shape[1] - prompt_len
    tokens_per_sec = (total_new * batch) / total_s
    tpot_ms = (total_s - ttft_ms/1000) * 1000 / max(total_new - 1, 1)

    outputs_text = []
    for b in range(batch):
        gen_ids = output[b][prompt_len:]
        outputs_text.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return {
        "TTFT_ms": ttft_ms,
        "decode_tok_s": tokens_per_sec,
        "TPOT_ms": tpot_ms,
        "num_gen": total_new,
        "outputs": outputs_text,
    }


def bench_chat(model, tokenizer, batch, num_gen=50):
    """Chat-templated coherent-output test."""
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    # Repeat for batch dimension
    input_ids_batch = input_ids.repeat(batch, 1)
    adapter = HuggingFaceGenerationAdapter(model)

    output = adapter.generate(
        input_ids_batch,
        attention_mask=torch.ones_like(input_ids_batch),
        max_new_tokens=num_gen,
        do_sample=False, temperature=1.0, top_k=1,
    )

    prompt_len = input_ids_batch.shape[1]
    results = []
    for b in range(batch):
        gen_ids = output[b][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        results.append(text)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--num-gen", type=int, default=50)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    tag = f"{args.tag}_ctx{args.ctx}_seq{args.seq}_bs{args.batch}"
    compiled_path = Path(args.out_dir) / tag

    print(f"[test] tag={tag}", flush=True)
    print(f"[test] USE_NATIVE_CONV1D={USE_NATIVE_CONV1D}", flush=True)
    print(f"[test] FORCE_SPARSE_MOE_PREFILL={FORCE_SPARSE_MOE_PREFILL}", flush=True)
    print(f"[test] batch={args.batch} ctx={args.ctx} seq={args.seq}", flush=True)

    try:
        model, compile_s, load_s = compile_or_load(
            args.ctx, args.seq, args.batch, compiled_path
        )
    except Exception as e:
        print(f"[test] COMPILE FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return

    tokenizer = AutoTokenizer.from_pretrained(compiled_path, padding_side="right")

    try:
        bench = bench_raw(model, tokenizer, args.batch, num_gen=args.num_gen)
    except Exception as e:
        print(f"[test] RAW BENCH FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        bench = None

    try:
        chat_results = bench_chat(model, tokenizer, args.batch, num_gen=args.num_gen)
    except Exception as e:
        print(f"[test] CHAT BENCH FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        chat_results = None

    print("\n" + "=" * 70)
    print(f"===== RESULTS for {tag} =====")
    print("=" * 70)
    print(f"compile_s={compile_s:.1f}")
    print(f"load_s={load_s:.1f}")
    if bench:
        print(f"TTFT_ms={bench['TTFT_ms']:.2f}")
        print(f"decode_tok_s={bench['decode_tok_s']:.2f}")
        print(f"TPOT_ms={bench['TPOT_ms']:.2f}")
        print(f"num_gen={bench['num_gen']}")
        for b, text in enumerate(bench["outputs"]):
            print(f"  raw[b={b}]: {text!r}")
    if chat_results:
        for b, text in enumerate(chat_results):
            print(f"  chat[b={b}]: {text!r}")

    print("[test] Done.")


if __name__ == "__main__":
    main()
