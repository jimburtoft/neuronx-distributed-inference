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

    # With NKI attention kernel (fastest: ~54 tok/s with in-kernel cache update)
    python minimax_m2_inference.py \
        --model-path /mnt/models/MiniMax-M2 \
        --compiled-model-path /mnt/models/MiniMax-M2-compiled-nki \
        --tp-degree 32 \
        --batch-size 1 \
        --max-context-length 128 \
        --max-new-tokens 128 \
        --enable-nki-attention

    # With FP8 checkpoint, dequanted to BF16 (recommended — avoids CTE DGE issue):
    # Step 1: Preprocess the FP8 checkpoint
    python conversion_script/preprocess_minimax_m2_fp8.py \
        --hf_model_path /mnt/models/MiniMax-M2 \
        --save_path /mnt/models/MiniMax-M2-fp8-neuron
    # Step 2: Run with --fp8-dequant (loads FP8, dequants to BF16 during conversion)
    python minimax_m2_inference.py \
        --model-path /mnt/models/MiniMax-M2-fp8-neuron \
        --compiled-model-path /mnt/models/MiniMax-M2-compiled-fp8dq \
        --quantized-checkpoints-path /mnt/models/MiniMax-M2-fp8-neuron \
        --fp8-dequant \
        --tp-degree 32 \
        --batch-size 1 \
        --max-context-length 128 \
        --max-new-tokens 128

    # With FP8 + fused MoE TKG kernel (best FP8 throughput):
    # Requires: nkilib fork (branch feature/selection-bias-routing)
    UNSAFE_FP8FNCAST=1 XLA_HANDLE_SPECIAL_SCALAR=1 python minimax_m2_inference.py \
        --model-path /mnt/models/MiniMax-M2 \
        --compiled-model-path /mnt/models/MiniMax-M2-compiled-fused-tkg \
        --quantized-checkpoints-path /mnt/models/MiniMax-M2-fp8-neuron \
        --fused-moe-tkg \
        --tp-degree 32 \
        --batch-size 1 \
        --max-context-length 128 \
        --max-new-tokens 128

    # With native FP8 ROW quantization (target: best FP8 performance):
    # Step 1: Preprocess with per-row scales
    python conversion_script/preprocess_minimax_m2_fp8.py \
        --hf_model_path /mnt/models/MiniMax-M2 \
        --save_path /mnt/models/MiniMax-M2-fp8-perrow \
        --scale-mode per_row
    # Step 2: Run with --fp8-native-row
    UNSAFE_FP8FNCAST=1 XLA_HANDLE_SPECIAL_SCALAR=1 python minimax_m2_inference.py \
        --model-path /mnt/models/MiniMax-M2 \
        --compiled-model-path /mnt/models/MiniMax-M2-compiled-fp8-row \
        --quantized-checkpoints-path /mnt/models/MiniMax-M2-fp8-perrow \
        --fp8-native-row \
        --tp-degree 32 \
        --batch-size 1 \
        --max-context-length 128 \
        --max-new-tokens 128
"""

import argparse
import os
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
        help="Enable NKI attention block kernel (fused QKV + RoPE + attention "
        "+ in-kernel KV cache update — ~54 tok/s, fastest configuration)",
    )
    parser.add_argument(
        "--quantized-checkpoints-path",
        type=str,
        default=None,
        help="Path to preprocessed FP8 checkpoint (from preprocess_minimax_m2_fp8.py). "
        "When used with --fp8-dequant (recommended): loads FP8 checkpoint, dequantizes "
        "expert weights to BF16 during conversion, model runs in BF16. "
        "Without --fp8-dequant: attempts native FP8 inference (blocked by compiler DGE issue).",
    )
    parser.add_argument(
        "--fp8-dequant",
        action="store_true",
        help="Dequantize FP8 expert weights to BF16 during checkpoint loading. "
        "Model runs entirely in BF16 — avoids CTE DGE compiler issue in SDK 2.29. "
        "Requires --quantized-checkpoints-path or FP8 checkpoint at --model-path.",
    )
    parser.add_argument(
        "--fused-moe-tkg",
        action="store_true",
        help="Enable fused MoE TKG NKI kernel with selection_bias support. "
        "Requires nkilib fork (branch feature/selection-bias-routing) installed. "
        "The fused kernel handles FP8 natively inside the kernel — no external dequant. "
        "Recommended for best FP8 throughput.",
    )
    parser.add_argument(
        "--fp8-native-row",
        action="store_true",
        help="Enable native FP8 ROW quantization in the nkilib TKG kernel. "
        "Requires a per-row preprocessed checkpoint (--scale-mode per_row). "
        "FP8 weights are kept in HBM and per-row scales are applied post-matmul "
        "inside the kernel. Implies --fused-moe-tkg. "
        "This is the target mode for best FP8 performance.",
    )
    return parser.parse_args()


def create_config(args) -> MiniMaxM2InferenceConfig:
    """Create MiniMax-M2 inference config from HF checkpoint and CLI args."""

    # NKI attention kernel configuration:
    #
    # The NKI attention_block_tkg kernel requires fused_qkv=True, cascaded
    # attention, and the QKV NKI kernel. The in-kernel KV cache update uses
    # a fixed _update_flat_cache that flattens cache tensors to 2D and uses
    # indirect_dim=0 with absolute indices (nki-library fork, commit daad362).
    #
    # Performance summary (SDK 2.29, trn2.48xlarge, TP=32, B=1, ctx=128):
    #   Baseline (no NKI, fused_qkv=True):        44.0 tok/s, correct
    #   NKI attention + cache_update=False:        48.9 tok/s, correct
    #   NKI attention + cache_update=True (fixed): 54.1 tok/s, correct  <-- best
    #
    # The NKI attention kernel with in-kernel cache update is now the fastest
    # configuration, ~23% faster than baseline.

    use_nki_attn = args.enable_nki_attention
    use_fp8 = args.quantized_checkpoints_path is not None
    fp8_dequant = getattr(args, "fp8_dequant", False)
    use_fused_moe_tkg = getattr(args, "fused_moe_tkg", False)
    fp8_native_row = getattr(args, "fp8_native_row", False)
    # --fp8-native-row implies --fused-moe-tkg
    if fp8_native_row:
        use_fused_moe_tkg = True
    # Native FP8 inference (quantized=True) — only when FP8 checkpoint provided
    # AND --fp8-dequant is NOT set. With --fp8-dequant, weights are dequanted
    # to BF16 during conversion and the model runs in BF16.
    use_fp8_native = use_fp8 and not fp8_dequant
    if use_fused_moe_tkg and not use_fp8_native:
        print(
            "WARNING: --fused-moe-tkg is most effective with native FP8 "
            "(--quantized-checkpoints-path without --fp8-dequant). "
            "Proceeding anyway for BF16 fused TKG."
        )
    if use_nki_attn:
        print(
            "INFO: NKI attention kernel enabled with in-kernel KV cache update "
            "(cache_update=True). Expected ~54 tok/s vs ~44 tok/s baseline."
        )
    if use_fp8 and fp8_dequant:
        print(
            f"INFO: FP8 dequant mode. Loading FP8 checkpoint from "
            f"{args.quantized_checkpoints_path or args.model_path}, dequantizing "
            f"expert weights to BF16 during conversion. Model runs in BF16."
        )
    elif use_fp8:
        print(
            f"INFO: FP8 MoE inference enabled. Loading preprocessed checkpoint from "
            f"{args.quantized_checkpoints_path}. Expert weights stay in FP8 (1 byte/param "
            f"vs 2 for BF16), halving MoE HBM usage."
        )
    if use_fused_moe_tkg:
        print(
            "INFO: Fused MoE TKG kernel enabled with selection_bias support. "
            "Requires nkilib fork (branch feature/selection-bias-routing). "
            "The fused kernel handles FP8 natively — no external dequant for TKG path."
        )
    if fp8_native_row:
        print(
            "INFO: Native FP8 ROW mode enabled. FP8 expert weights stay in HBM, "
            "per-row scales applied post-matmul by nkilib TKG kernel. "
            "Requires per-row preprocessed checkpoint (--scale-mode per_row)."
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
        # NKI MoE kernels: Enable when native FP8 or when fused MoE TKG is requested.
        # The fused TKG kernel requires router_topk and expert_mlp NKI kernels.
        # For fp8-dequant or BF16 without --fused-moe-tkg: disabled at TP=32 because
        # intermediate_size/TP=48 causes overhead from internal padding to 128.
        router_topk_nki_kernel_enabled=(use_fp8_native or use_fused_moe_tkg)
        and not os.environ.get("DISABLE_NKI_MOE"),
        expert_mlp_nki_kernel_enabled=(use_fp8_native or use_fused_moe_tkg)
        and not os.environ.get("DISABLE_NKI_MOE"),
        # Fused MoE TKG mega-kernel: single kernel for router + expert MLP + shared MLP.
        # Handles FP8 natively inside the kernel — no external dequant needed for TKG.
        # Requires nkilib fork with selection_bias support.
        moe_fused_nki_kernel_enabled=use_fused_moe_tkg
        and not os.environ.get("DISABLE_NKI_MOE"),
        # NKI attention block kernel (fused QKV + RoPE + attention + KV cache)
        attn_block_tkg_nki_kernel_enabled=use_nki_attn,
        attn_block_tkg_nki_kernel_cascaded_attention=use_nki_attn,
        # cache_update=True — in-kernel DMA update (fixed in nki-library fork)
        attn_block_tkg_nki_kernel_cache_update=use_nki_attn,
        qkv_kernel_enabled=use_nki_attn,
        # fused_qkv=True always — uses fused QKV weight for cleaner loading
        fused_qkv=True,
        # FP8 MoE quantization: only enable when using native FP8 (not dequant mode).
        # In dequant mode, quantized=False and weights are BF16 at runtime.
        quantized=use_fp8_native,
        quantized_checkpoints_path=args.quantized_checkpoints_path
        if use_fp8_native
        else None,
        quantization_type="per_channel_symmetric"
        if use_fp8_native
        else "per_tensor_symmetric",
        quantization_dtype="f8e4m3" if use_fp8_native else "int8",
        # Blockwise scales from preprocessing; NxDI does not shard these
        quantization_block_size=None,
        quantization_block_axis=None,
        # Exclude ALL modules from NxD convert() for native FP8.
        # Block-wise expert scales have block dimensions that don't divide
        # by TP=32, so NxDI's shard_children would crash. Instead, compat.py
        # handles FP8 dequant manually (CTE path) and the fused TKG kernel
        # reads scales from raw checkpoint tensors attached to the modules.
        modules_to_not_convert=(
            [
                "self_attn",
                "lm_head",
                "embed_tokens",
                "norm",
                "block_sparse_moe",
                "expert_mlps",
                "router",
            ]
            if use_fp8_native
            else None
        ),
    )

    # Native FP8 ROW mode flag — signals modeling_minimax_m2.py to register
    # .scale parameters and skip BF16 dequant for expert MLP weights.
    neuron_config.fp8_native_row_mode = fp8_native_row

    # MiniMax-M2 HF repo has auto_map, requiring trust_remote_code for AutoConfig.
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config = MiniMaxM2InferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    return config


def main():
    args = parse_args()

    # FP8 native inference requires UNSAFE_FP8FNCAST=1 for torch_neuronx XLA tracing to
    # accept float8_e4m3fn tensors. Must be set before model trace/compile.
    # XLA_HANDLE_SPECIAL_SCALAR=1 is also required for correct FP8 scalar handling.
    # NOT needed for --fp8-dequant mode (model is BF16, no FP8 in XLA graph).
    if args.quantized_checkpoints_path is not None and not getattr(
        args, "fp8_dequant", False
    ):
        os.environ["UNSAFE_FP8FNCAST"] = "1"
        os.environ["XLA_HANDLE_SPECIAL_SCALAR"] = "1"

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

    # --- Weight diagnostic: dump parameter info after load ---
    if os.environ.get("DUMP_WEIGHTS"):
        import sys

        print(
            "\n[WEIGHT-DIAG] Dumping model parameter info after load...",
            file=sys.stderr,
            flush=True,
        )
        fp8_count = 0
        bf16_count = 0
        fp32_count = 0
        scale_count = 0
        for name, param in model.named_parameters():
            if "layers.0.block_sparse_moe.expert_mlps" in name:
                print(
                    f"  [WEIGHT-DIAG] {name}: shape={param.shape}, dtype={param.dtype}",
                    file=sys.stderr,
                    flush=True,
                )
            if param.dtype == torch.float8_e4m3fn:
                fp8_count += 1
            elif param.dtype == torch.bfloat16:
                bf16_count += 1
            elif param.dtype == torch.float32:
                fp32_count += 1
            if "scale" in name:
                scale_count += 1
        print(
            f"  [WEIGHT-DIAG] Total params: fp8={fp8_count}, bf16={bf16_count}, fp32={fp32_count}, scales={scale_count}",
            file=sys.stderr,
            flush=True,
        )

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
