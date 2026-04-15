#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example inference script for MiniMax-M2 on AWS Trainium (trn2).

Known limitation: At TP=32 on trn2.48xlarge, the 256-expert MoE weights
consume ~22 GB of the ~24 GB HBM per core. This limits max_context_length
to ~128 with the current NxDI runtime (CE+TKG NEFFs duplicate MoE weights).
Larger contexts require INT8 quantization or EP=2.

Usage:
    # Compile + run on trn2.48xlarge (TP=32)
    python minimax_m2_inference.py \
        --model-path /mnt/models/MiniMax-M2 \
        --compiled-model-path /mnt/models/MiniMax-M2-compiled \
        --tp-degree 32 \
        --batch-size 1 \
        --max-context-length 128 \
        --max-new-tokens 128

    # With NKI attention kernel (experimental, ~30 tok/s vs ~51 tok/s baseline)
    python minimax_m2_inference.py \
        --model-path /mnt/models/MiniMax-M2 \
        --compiled-model-path /mnt/models/MiniMax-M2-compiled-nki \
        --tp-degree 32 \
        --batch-size 1 \
        --max-context-length 128 \
        --max-new-tokens 128 \
        --enable-nki-attention
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

# Add src to path for contrib import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMax-M2 inference on Trainium")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to HF checkpoint"
    )
    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default=None,
        help="Path to pre-compiled model (or where to save compiled model)",
    )
    parser.add_argument(
        "--tp-degree", type=int, default=32, help="Tensor parallelism degree"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=128,
        help="Max context length (limited by HBM at TP=32)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max tokens to generate"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Input prompt",
    )
    parser.add_argument(
        "--on-cpu", action="store_true", help="Run on CPU (for testing weight loading)"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile, don't run inference",
    )
    parser.add_argument(
        "--enable-nki-attention",
        action="store_true",
        help="Enable NKI attention block kernel (experimental, slower due to "
        "framework KV cache update workaround — see technical notes)",
    )
    return parser.parse_args()


def create_config(args) -> MiniMaxM2InferenceConfig:
    """Create MiniMax-M2 inference config from HF checkpoint and CLI args."""

    # NKI attention kernel configuration:
    #
    # The NKI attention_block_tkg kernel requires fused_qkv=True, cascaded
    # attention, and the QKV NKI kernel. However, the in-kernel KV cache update
    # (_update_flat_cache in nkilib) has a known issue at B=1, S_tkg=1 that
    # corrupts the KV cache, producing garbage output during token generation.
    # Both K_cache_transposed=True and K_cache_transposed=False paths are
    # affected.
    #
    # Workaround: cache_update=False lets the NxDI framework handle KV cache
    # updates outside the kernel. This produces correct output but at ~30 tok/s
    # vs ~54 tok/s baseline (the framework cache update path is slower).
    #
    # Recommendation: Use fused_qkv=True without the NKI attention kernel for
    # best throughput (51.7 tok/s with correct output). The NKI attention kernel
    # provides fused QKV + RoPE + attention in a single NEFF with only ~5% overhead.
    #
    # Performance summary (SDK 2.29, trn2.48xlarge, TP=32, B=1, ctx=128):
    #   Baseline (no NKI, fused_qkv=True):  51.7 tok/s, correct  <-- default
    #   NKI attention + cache_update=False:  49.0 tok/s, correct
    #   NKI attention + cache_update=True:   ~49-55 tok/s, GARBAGE (nkilib issue)
    #
    # The NKI in-kernel KV cache update (cache_update=True) corrupts output at B=1
    # due to an issue in nkilib's _update_flat_cache DMA pattern. Using framework
    # KV cache update (cache_update=False) avoids this with minimal throughput cost.

    use_nki_attn = args.enable_nki_attention
    if use_nki_attn:
        print(
            "INFO: NKI attention kernel enabled with framework KV cache update "
            "(cache_update=False). Expected ~49 tok/s vs ~52 tok/s baseline."
        )

    neuron_config = MoENeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        seq_len=args.max_context_length + args.max_new_tokens,
        torch_dtype=torch.bfloat16,
        on_cpu=args.on_cpu,
        # MoE settings: MiniMax-M2 uses SiLU activation (silu(gate) * up).
        # This maps to glu_mlp=True with the default glu_type="glu".
        glu_mlp=True,
        # On-device sampling for greedy decoding
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=False, global_topk=1
        ),
        # Bucketing disabled — single context length for HBM-constrained config
        enable_bucketing=False,
        # NKI MoE kernels: disabled at TP=32 because intermediate_size/TP=48
        # which causes overhead from internal padding to 128 in the NKI kernel.
        router_topk_nki_kernel_enabled=False,
        expert_mlp_nki_kernel_enabled=False,
        # NKI attention block kernel (fused QKV + RoPE + attention + KV cache)
        attn_block_tkg_nki_kernel_enabled=use_nki_attn,
        attn_block_tkg_nki_kernel_cascaded_attention=use_nki_attn,
        # cache_update must be False — in-kernel DMA update corrupts output at B=1
        attn_block_tkg_nki_kernel_cache_update=False,
        qkv_kernel_enabled=use_nki_attn,
        # fused_qkv=True always — uses fused QKV weight for cleaner loading
        fused_qkv=True,
    )

    # MiniMax-M2 HF repo has auto_map, requiring trust_remote_code for AutoConfig.
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config = MiniMaxM2InferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    return config


def main():
    args = parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Creating config (TP={args.tp_degree}, batch={args.batch_size})...")
    config = create_config(args)
    print(
        f"  Model: {config.num_hidden_layers} layers, {config.num_local_experts} experts"
    )
    print(f"  Hidden: {config.hidden_size}, Intermediate: {config.intermediate_size}")
    print(
        f"  Heads: {config.num_attention_heads}Q / {config.num_key_value_heads}KV, head_dim={config.head_dim}"
    )
    print(
        f"  Routing: sigmoid + e_score_correction_bias, top-{config.num_experts_per_tok}"
    )

    print(f"Initializing model...")
    model = NeuronMiniMaxM2ForCausalLM(args.model_path, config)

    compiled_path = args.compiled_model_path or f"{args.model_path}_compiled"

    if (
        args.compiled_model_path
        and Path(args.compiled_model_path).exists()
        and (Path(args.compiled_model_path) / "model.pt").exists()
    ):
        print(f"Loading pre-compiled model from {args.compiled_model_path}...")
        model.load(args.compiled_model_path)
    else:
        print(f"Compiling model to {compiled_path} (this may take a while)...")
        model.compile(compiled_path)
        print(f"Loading compiled model from {compiled_path}...")
        model.load(compiled_path)

    if args.compile_only:
        print("Compile-only mode, exiting.")
        return

    # --- Generation using HuggingFace generate() via NxDI adapter ---
    print(f"\nPrompt: {args.prompt}")

    inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Generating up to {args.max_new_tokens} new tokens...\n")

    generation_model = HuggingFaceGenerationAdapter(model)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = generation_model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    elapsed = time.perf_counter() - start_time

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_new_tokens = outputs.shape[1] - input_ids.shape[1]

    print(f"Output: {output_text}")
    print(
        f"\nGenerated {num_new_tokens} tokens in {elapsed:.2f}s "
        f"({num_new_tokens / elapsed:.1f} tok/s)"
    )

    model.reset()


if __name__ == "__main__":
    main()
