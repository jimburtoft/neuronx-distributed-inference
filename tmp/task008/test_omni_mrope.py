#!/usr/bin/env python3
"""
Task 008: Validate Qwen2.5-Omni-7B text-only on Neuron (SDK 2.29, TP=4).

Compares before (1D RoPE) vs after (M-RoPE) by measuring:
  - Token match vs HF CPU BF16 reference
  - Throughput (tok/s)
  - TPOT (ms)

Usage:
  python3 test_omni_mrope.py --phase before   # Test original 1D RoPE
  python3 test_omni_mrope.py --phase after    # Test after M-RoPE fix
  python3 test_omni_mrope.py --phase hf-only  # Generate HF reference only
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

MODEL_PATH = os.environ.get(
    "QWEN25_OMNI_MODEL_PATH", "/opt/dlami/nvme/models/Qwen2.5-Omni-7B"
)
COMPILED_PATH_TEMPLATE = "/opt/dlami/nvme/compiled/qwen25_omni_tp4_{phase}"
TP_DEGREE = 4

TEST_PROMPTS = [
    "The capital of France is",
    "In the year 2024, artificial intelligence",
    "The Pythagorean theorem states that",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
]

MAX_NEW_TOKENS = 30


def generate_hf_reference(model_path, tokenizer, prompts, max_new_tokens):
    """Generate reference outputs using HF thinker model (BF16, CPU)."""
    print("=" * 60)
    print("Generating HF reference outputs (thinker, BF16, CPU)...")
    print("=" * 60)

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    hf_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    hf_model.eval()

    results = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        with torch.no_grad():
            output = hf_model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False
            )

        gen_tokens = output[0][input_ids.shape[1] :].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        results[prompt] = {"tokens": gen_tokens, "text": gen_text}
        print(f"  [{prompt[:40]}] -> {gen_text[:60]}...")

    del hf_model
    gc.collect()
    return results


def compile_and_test_neuron(
    model_path, compiled_path, tokenizer, prompts, max_new_tokens
):
    """Compile and test the Neuron model, return results dict."""
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config,
        HuggingFaceGenerationAdapter,
    )
    from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni import (
        NeuronQwen25OmniForCausalLM,
        Qwen25OmniInferenceConfig,
    )

    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        seq_len=128,
        max_context_length=128,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
    )

    config = Qwen25OmniInferenceConfig(
        neuron_config, load_config=load_pretrained_config(model_path)
    )

    # Check if rope_scaling is present (M-RoPE fix indicator)
    has_mrope = hasattr(config, "rope_scaling") and config.rope_scaling is not None
    print(f"Config has rope_scaling: {has_mrope}")
    if has_mrope:
        mrope_section = config.rope_scaling.get("mrope_section", None)
        print(f"  mrope_section: {mrope_section}")

    model = NeuronQwen25OmniForCausalLM(model_path, config)

    # Compile
    print(f"\nCompiling to {compiled_path}...")
    compile_start = time.time()
    model.compile(compiled_path)
    compile_time = time.time() - compile_start
    print(f"Compilation: {compile_time:.1f}s")

    # Load
    print("Loading compiled model...")
    model.load(compiled_path)

    # Create adapter for generation
    adapter = HuggingFaceGenerationAdapter(model)

    # Generate
    print("\nRunning inference...")
    results = {}
    times = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        start = time.time()
        output_ids = adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        elapsed = time.time() - start

        # Extract generated tokens (strip the input prefix)
        gen_tokens = output_ids[0][input_ids.shape[1] :].tolist()
        output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        results[prompt] = {"tokens": gen_tokens, "text": output_text}
        times.append(elapsed)

        n_gen = len(gen_tokens)
        tok_s = n_gen / elapsed if elapsed > 0 else 0
        tpot = (elapsed / n_gen * 1000) if n_gen > 0 else 0
        print(
            f"  [{prompt[:40]}] {n_gen} tok, {elapsed:.2f}s, {tok_s:.1f} tok/s, {tpot:.1f}ms TPOT"
        )
        print(f"    -> {output_text[:80]}...")

    avg_time = sum(times) / len(times) if times else 0
    avg_toks = MAX_NEW_TOKENS / avg_time if avg_time > 0 else 0
    avg_tpot = (avg_time / MAX_NEW_TOKENS * 1000) if MAX_NEW_TOKENS > 0 else 0

    del model
    gc.collect()

    return results, {
        "compile_time_s": compile_time,
        "avg_throughput_toks": avg_toks,
        "avg_tpot_ms": avg_tpot,
        "has_mrope": has_mrope,
    }


def compare_tokens(neuron_results, hf_results):
    """Compare token match between Neuron and HF."""
    print("\n" + "=" * 60)
    print("TOKEN MATCH COMPARISON")
    print("=" * 60)

    total = 0
    matching = 0

    for prompt in neuron_results:
        if prompt not in hf_results:
            continue

        n_tok = neuron_results[prompt]["tokens"]
        h_tok = hf_results[prompt]["tokens"]
        compare_len = min(len(n_tok), len(h_tok))
        matches = sum(1 for i in range(compare_len) if n_tok[i] == h_tok[i])

        total += compare_len
        matching += matches
        pct = (matches / compare_len * 100) if compare_len > 0 else 0

        print(f"\n  Prompt: {prompt[:50]}...")
        print(f"  Neuron: {neuron_results[prompt]['text'][:80]}...")
        print(f"  HF:     {hf_results[prompt]['text'][:80]}...")
        print(f"  Match:  {matches}/{compare_len} ({pct:.1f}%)")

        if pct < 100:
            for i in range(compare_len):
                if n_tok[i] != h_tok[i]:
                    print(
                        f"  First divergence at pos {i}: neuron={n_tok[i]} vs hf={h_tok[i]}"
                    )
                    break

    overall = (matching / total * 100) if total > 0 else 0
    print(f"\nOVERALL: {matching}/{total} ({overall:.1f}%)")
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["before", "after", "hf-only"], required=True
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_ref_path = "/opt/dlami/nvme/compiled/hf_reference.json"

    if args.phase == "hf-only":
        hf_results = generate_hf_reference(
            args.model_path, tokenizer, TEST_PROMPTS, args.max_new_tokens
        )
        os.makedirs(os.path.dirname(hf_ref_path), exist_ok=True)
        with open(hf_ref_path, "w") as f:
            json.dump(hf_results, f, indent=2)
        print(f"\nHF reference saved to {hf_ref_path}")
        return

    compiled_path = COMPILED_PATH_TEMPLATE.format(phase=args.phase)

    # Run Neuron test
    neuron_results, metrics = compile_and_test_neuron(
        args.model_path, compiled_path, tokenizer, TEST_PROMPTS, args.max_new_tokens
    )

    # Load or generate HF reference
    if os.path.exists(hf_ref_path):
        print(f"\nLoading HF reference from {hf_ref_path}")
        with open(hf_ref_path) as f:
            hf_results = json.load(f)
    else:
        hf_results = generate_hf_reference(
            args.model_path, tokenizer, TEST_PROMPTS, args.max_new_tokens
        )
        os.makedirs(os.path.dirname(hf_ref_path), exist_ok=True)
        with open(hf_ref_path, "w") as f:
            json.dump(hf_results, f, indent=2)

    # Compare
    token_match = compare_tokens(neuron_results, hf_results)
    metrics["token_match_pct"] = token_match

    # Save results
    results_path = f"/opt/dlami/nvme/compiled/results_{args.phase}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "phase": args.phase,
                "metrics": metrics,
                "prompts": {
                    k: {
                        "neuron": neuron_results[k]["text"],
                        "hf": hf_results.get(k, {}).get("text", ""),
                    }
                    for k in neuron_results
                },
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({args.phase})")
    print(f"{'=' * 60}")
    print(f"  Phase:        {args.phase}")
    print(f"  M-RoPE:       {'YES' if metrics['has_mrope'] else 'NO (1D RoPE)'}")
    print(f"  Compile:      {metrics['compile_time_s']:.1f}s")
    print(f"  Throughput:   {metrics['avg_throughput_toks']:.1f} tok/s")
    print(f"  TPOT:         {metrics['avg_tpot_ms']:.1f} ms")
    print(f"  Token match:  {token_match:.1f}%")
    print(f"  Results:      {results_path}")


if __name__ == "__main__":
    main()
