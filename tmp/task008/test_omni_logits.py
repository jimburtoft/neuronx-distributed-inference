#!/usr/bin/env python3
"""
Task 008: Cascade-free logit comparison for Qwen2.5-Omni-7B text-only on Neuron.

Uses teacher-forcing via HuggingFaceGenerationAdapter.generate() with output_scores=True.
At each token position, checks whether the HF reference token is in the Neuron model's
top-1, top-5, or top-10 predictions. Reports per-position accuracy.

This eliminates autoregressive cascade divergence: even if Neuron's greedy token differs
from HF at position i, position i+1 still uses HF's token as input (via the generate
loop's teacher-forcing behavior from the logit_validation framework).

Simpler approach used here: Run generate() with output_scores=True, then compare
argmax at each position. Also report top-k containment for the HF reference token.

Usage:
  python3 test_omni_logits.py --phase before   # Test original 1D RoPE
  python3 test_omni_logits.py --phase after     # Test after M-RoPE fix
"""

import argparse
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
    """Generate reference outputs with per-token logits using HF thinker model (BF16, CPU)."""
    print("=" * 60)
    print("Generating HF reference outputs with logits (thinker, BF16, CPU)...")
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
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # output.scores is a tuple of (num_generated_tokens,) tensors, each [batch, vocab]
        gen_tokens = output.sequences[0][input_ids.shape[1] :].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Store per-position top-10 token IDs from HF logits
        hf_top10_per_pos = []
        for score in output.scores:
            top10 = torch.topk(score[0], k=10)
            hf_top10_per_pos.append(
                {
                    "top10_ids": top10.indices.tolist(),
                    "top10_logits": top10.values.float().tolist(),
                }
            )

        results[prompt] = {
            "tokens": gen_tokens,
            "text": gen_text,
            "top10_per_pos": hf_top10_per_pos,
        }
        print(f"  [{prompt[:40]}] -> {gen_text[:60]}...")

    import gc

    del hf_model
    gc.collect()
    return results


def test_neuron_logits(
    model_path, compiled_path, tokenizer, prompts, max_new_tokens, hf_results
):
    """Compile/load the Neuron model, generate with output_scores, compare logits."""
    from neuronx_distributed_inference.models.config import NeuronConfig
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

    # Create adapter
    adapter = HuggingFaceGenerationAdapter(model)

    # Generate with output_scores
    print("\nRunning inference with logit capture...")
    all_results = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        start = time.time()
        output = adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        elapsed = time.time() - start

        gen_tokens = output.sequences[0][input_ids.shape[1] :].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Per-position comparison against HF reference
        hf_ref = hf_results[prompt]
        hf_tokens = hf_ref["tokens"]
        n_compare = min(len(gen_tokens), len(hf_tokens), len(output.scores))

        top1_match = 0
        top5_match = 0
        top10_match = 0
        pos_details = []

        for pos in range(n_compare):
            neuron_score = output.scores[pos][0]  # [vocab_size]
            neuron_top10 = torch.topk(neuron_score, k=10)
            neuron_top10_ids = neuron_top10.indices.tolist()
            neuron_argmax = neuron_top10_ids[0]
            hf_token = hf_tokens[pos]

            in_top1 = neuron_argmax == hf_token
            in_top5 = hf_token in neuron_top10_ids[:5]
            in_top10 = hf_token in neuron_top10_ids[:10]

            if in_top1:
                top1_match += 1
            if in_top5:
                top5_match += 1
            if in_top10:
                top10_match += 1

            # Find rank of HF token in Neuron logits
            rank = -1
            if in_top10:
                rank = neuron_top10_ids.index(hf_token)
            else:
                # Search full vocab
                sorted_indices = torch.argsort(neuron_score, descending=True)
                for r in range(len(sorted_indices)):
                    if sorted_indices[r].item() == hf_token:
                        rank = r
                        break

            pos_details.append(
                {
                    "pos": pos,
                    "hf_token": hf_token,
                    "neuron_argmax": neuron_argmax,
                    "hf_rank_in_neuron": rank,
                    "match": in_top1,
                }
            )

        top1_pct = top1_match / n_compare * 100 if n_compare > 0 else 0
        top5_pct = top5_match / n_compare * 100 if n_compare > 0 else 0
        top10_pct = top10_match / n_compare * 100 if n_compare > 0 else 0

        tok_s = len(gen_tokens) / elapsed if elapsed > 0 else 0

        print(f"\n  [{prompt[:40]}]")
        print(f"    Neuron: {gen_text[:70]}...")
        print(f"    HF:     {hf_ref['text'][:70]}...")
        print(f"    Top-1 match: {top1_match}/{n_compare} ({top1_pct:.1f}%)")
        print(f"    Top-5 match: {top5_match}/{n_compare} ({top5_pct:.1f}%)")
        print(f"    Top-10 match: {top10_match}/{n_compare} ({top10_pct:.1f}%)")
        print(f"    {tok_s:.1f} tok/s, {elapsed:.2f}s")

        # Show first few divergences
        divergences = [d for d in pos_details if not d["match"]]
        if divergences:
            print(f"    First divergences:")
            for d in divergences[:5]:
                hf_tok_str = tokenizer.decode([d["hf_token"]])
                n_tok_str = tokenizer.decode([d["neuron_argmax"]])
                print(
                    f"      pos {d['pos']}: neuron='{n_tok_str}' vs hf='{hf_tok_str}' (hf at rank {d['hf_rank_in_neuron']})"
                )

        all_results[prompt] = {
            "gen_text": gen_text,
            "top1_pct": top1_pct,
            "top5_pct": top5_pct,
            "top10_pct": top10_pct,
            "n_compare": n_compare,
            "tok_s": tok_s,
            "pos_details": pos_details,
        }

    # Overall summary
    total_pos = sum(r["n_compare"] for r in all_results.values())
    total_top1 = sum(
        sum(1 for d in r["pos_details"] if d["match"]) for r in all_results.values()
    )
    total_top5 = sum(
        sum(1 for d in r["pos_details"] if d["hf_rank_in_neuron"] < 5)
        for r in all_results.values()
    )
    total_top10 = sum(
        sum(1 for d in r["pos_details"] if 0 <= d["hf_rank_in_neuron"] < 10)
        for r in all_results.values()
    )

    print(f"\n{'=' * 60}")
    print(f"OVERALL LOGIT COMPARISON")
    print(f"{'=' * 60}")
    print(f"  M-RoPE:      {'YES' if has_mrope else 'NO (1D RoPE)'}")
    print(
        f"  Top-1 match: {total_top1}/{total_pos} ({total_top1 / total_pos * 100:.1f}%)"
    )
    print(
        f"  Top-5 match: {total_top5}/{total_pos} ({total_top5 / total_pos * 100:.1f}%)"
    )
    print(
        f"  Top-10 match: {total_top10}/{total_pos} ({total_top10 / total_pos * 100:.1f}%)"
    )
    print(f"  Compile:     {compile_time:.1f}s")

    return {
        "has_mrope": has_mrope,
        "compile_time_s": compile_time,
        "total_positions": total_pos,
        "top1_match": total_top1,
        "top1_pct": total_top1 / total_pos * 100 if total_pos > 0 else 0,
        "top5_match": total_top5,
        "top5_pct": total_top5 / total_pos * 100 if total_pos > 0 else 0,
        "top10_match": total_top10,
        "top10_pct": total_top10 / total_pos * 100 if total_pos > 0 else 0,
        "per_prompt": all_results,
    }


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

    hf_ref_path = "/opt/dlami/nvme/compiled/hf_reference_logits.json"

    if args.phase == "hf-only":
        hf_results = generate_hf_reference(
            args.model_path, tokenizer, TEST_PROMPTS, args.max_new_tokens
        )
        os.makedirs(os.path.dirname(hf_ref_path), exist_ok=True)
        with open(hf_ref_path, "w") as f:
            json.dump(hf_results, f, indent=2)
        print(f"\nHF reference saved to {hf_ref_path}")
        return

    # Load or generate HF reference
    if os.path.exists(hf_ref_path):
        print(f"Loading HF reference from {hf_ref_path}")
        with open(hf_ref_path) as f:
            hf_results = json.load(f)
    else:
        hf_results = generate_hf_reference(
            args.model_path, tokenizer, TEST_PROMPTS, args.max_new_tokens
        )
        os.makedirs(os.path.dirname(hf_ref_path), exist_ok=True)
        with open(hf_ref_path, "w") as f:
            json.dump(hf_results, f, indent=2)

    compiled_path = COMPILED_PATH_TEMPLATE.format(phase=args.phase)

    metrics = test_neuron_logits(
        args.model_path,
        compiled_path,
        tokenizer,
        TEST_PROMPTS,
        args.max_new_tokens,
        hf_results,
    )

    # Save
    results_path = f"/opt/dlami/nvme/compiled/logit_results_{args.phase}.json"
    with open(results_path, "w") as f:
        # Strip pos_details for cleaner output
        save_metrics = {k: v for k, v in metrics.items() if k != "per_prompt"}
        save_metrics["per_prompt_summary"] = {
            p: {k: v for k, v in r.items() if k != "pos_details"}
            for p, r in metrics["per_prompt"].items()
        }
        json.dump(save_metrics, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
