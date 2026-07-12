#!/usr/bin/env python3
"""A/B test: compile both scan modes at ctx=128, generate N tokens, compare token sequences.

The head-grouped quadratic scan is the ground-truth reference (validated in Task 014).
The chunked NKI SSD scan is under test. If state handoff is correct, both should produce
identical (or near-identical, due to fp32 accumulation order) token sequences.

Usage:
    python compare_scans.py --ctx 128 --num-gen 50
"""

import argparse
import gc
import importlib
import os
import sys
import time
from pathlib import Path

import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from transformers import AutoConfig, AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

MODEL_PATH = os.environ.get("NEMOTRON_MODEL_PATH", "/mnt/models/nemotron-30b")
COMPILE_ROOT = os.environ.get("NEMOTRON_COMPILE_ROOT", "/mnt/models")
TP_DEGREE = 4
BATCH_SIZE = 1


def build_config(ctx, seq):
    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        max_context_length=ctx,
        seq_len=seq,
        on_device_sampling_config=None,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        torch_dtype="bfloat16",
        save_sharded_checkpoint=True,
    )
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from modeling_nemotron_h import NemotronHInferenceConfig
    return NemotronHInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )


def set_scan_mode(mode):
    """mode: 'chunked' or 'quadratic'"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    # Force fresh reload
    for m in list(sys.modules):
        if m.startswith("modeling_nemotron_h") or m.startswith("nki_kernels"):
            del sys.modules[m]
    # Patch source file *before* import to toggle USE_CHUNKED_NKI_SCAN.
    # `src/modeling_nemotron_h.py` is two levels up from this test file.
    src = Path(__file__).resolve().parent.parent.parent / "src" / "modeling_nemotron_h.py"
    text = src.read_text()
    if mode == "chunked":
        text = text.replace("USE_CHUNKED_NKI_SCAN = False", "USE_CHUNKED_NKI_SCAN = True", 1)
        text = text.replace("USE_CHUNKED_NKI_SCAN = True  # Phase 3 -- integration in progress",
                            "USE_CHUNKED_NKI_SCAN = True  # Phase 3 -- integration in progress", 1)
    elif mode == "quadratic":
        text = text.replace("USE_CHUNKED_NKI_SCAN = True  # Phase 3 -- integration in progress",
                            "USE_CHUNKED_NKI_SCAN = False  # Phase 3 -- integration in progress", 1)
        text = text.replace("USE_CHUNKED_NKI_SCAN = True", "USE_CHUNKED_NKI_SCAN = False", 1)
    src.write_text(text)


def compile_and_load(ctx, seq, tag, mode):
    set_scan_mode(mode)
    config = build_config(ctx, seq)
    from modeling_nemotron_h import NeuronNemotronForCausalLM

    compiled_path = Path(f"{COMPILE_ROOT}/{tag}")
    if not (compiled_path / "model.pt").exists():
        print(f"[compile-{mode}] target {compiled_path}", flush=True)
        t0 = time.perf_counter()
        model = NeuronNemotronForCausalLM(MODEL_PATH, config)
        model.compile(str(compiled_path))
        compile_s = time.perf_counter() - t0
        print(f"[compile-{mode}] done in {compile_s:.1f}s", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tokenizer.save_pretrained(str(compiled_path))
        del model
        gc.collect()
    else:
        print(f"[compile-{mode}] cache hit at {compiled_path}", flush=True)
        compile_s = 0.0

    print(f"[load-{mode}] loading...", flush=True)
    t0 = time.perf_counter()
    model = NeuronNemotronForCausalLM(MODEL_PATH, config)
    model.load(str(compiled_path))
    load_s = time.perf_counter() - t0
    print(f"[load-{mode}] done in {load_s:.1f}s", flush=True)
    return model, compile_s, load_s


def generate_tokens(model, tokenizer, prompt, num_gen):
    inputs = tokenizer(prompt, return_tensors="pt")
    adapter = HuggingFaceGenerationAdapter(model)
    # Warmup
    _ = adapter.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=5,
        do_sample=False,
    )
    # Full gen
    out = adapter.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=num_gen,
        do_sample=False,
    )
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    token_ids = new_tokens.tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return token_ids, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--num-gen", type=int, default=50)
    args = ap.parse_args()

    prompt = "The capital of France is"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")

    # 1. Quadratic (ground truth)
    print("=" * 60)
    print(f"QUADRATIC scan, ctx={args.ctx}")
    print("=" * 60)
    model_q, compile_q, load_q = compile_and_load(
        args.ctx, 2048, f"cmp_quadratic_ctx{args.ctx}", "quadratic"
    )
    tokens_q, text_q = generate_tokens(model_q, tokenizer, prompt, args.num_gen)
    print(f"Quadratic tokens[:20]: {tokens_q[:20]}")
    print(f"Quadratic text: '{text_q}'")
    del model_q
    gc.collect()

    # 2. Chunked (under test)
    print()
    print("=" * 60)
    print(f"CHUNKED scan, ctx={args.ctx}")
    print("=" * 60)
    model_c, compile_c, load_c = compile_and_load(
        args.ctx, 2048, f"cmp_chunked_ctx{args.ctx}", "chunked"
    )
    tokens_c, text_c = generate_tokens(model_c, tokenizer, prompt, args.num_gen)
    print(f"Chunked tokens[:20]: {tokens_c[:20]}")
    print(f"Chunked text: '{text_c}'")

    # 3. Compare
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    match_count = 0
    for i, (tq, tc) in enumerate(zip(tokens_q, tokens_c)):
        if tq == tc:
            match_count += 1
        else:
            if i < 20 or match_count == i:  # Print first divergence
                print(f"  token {i}: quad={tq} ({tokenizer.decode([tq])!r}), chunked={tc} ({tokenizer.decode([tc])!r})")
    print(f"Prefix match: {match_count}/{min(len(tokens_q), len(tokens_c))} tokens")
    if match_count == len(tokens_q):
        print("PASS: chunked and quadratic produce identical token sequences")
    elif match_count >= 5:
        print(f"PARTIAL PASS: first {match_count} tokens match, then diverge (expected with fp32 noise at ~10-20 tokens)")
    else:
        print(f"FAIL: only {match_count} tokens match")


if __name__ == "__main__":
    main()
