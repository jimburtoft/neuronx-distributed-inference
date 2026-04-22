#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Preprocess MiniMax-M2 FP8 checkpoint for Neuron FP8 inference.

The HuggingFace MiniMax-M2 checkpoint (MiniMaxAI/MiniMax-M2) stores expert
weights in OCP FP8 E4M3 (float8_e4m3fn, range +-448) with block-wise
quantization (weight_block_size=[128,128]). Neuron hardware uses IEEE-754
FP8 E4M3 (range +-240). This script:

1. Rescales FP8 expert weights from OCP range (448) to Neuron range (240)
2. Converts block-wise weight_scale_inv to .scale format
3. Dequantizes attention weights (q/k/v/o_proj) to BF16
4. Preserves HF key naming so that convert_hf_to_neuron_state_dict
   can process the output identically to the original HF checkpoint
5. Saves preprocessed checkpoint with tokenizer/config files

After preprocessing, use the checkpoint path as --quantized-checkpoints-path
in the inference script. The model's convert_state_dict will handle expert
stacking, scale fusing, router renaming, QK norm padding, QKV fusion, etc.

Usage:
    python preprocess_minimax_m2_fp8.py \\
        --hf_model_path /path/to/MiniMaxAI/MiniMax-M2 \\
        --save_path /path/to/MiniMax-M2-fp8-neuron

Memory note: MiniMax-M2 has 256 experts x 62 layers. Each layer's expert
weights are ~1.5 GB in FP8. The script processes one layer at a time and
deletes consumed keys to limit peak memory to ~10-15 GB.
"""

import argparse
import gc
import json
import os
import shutil
from typing import Dict, List, Tuple

import torch

from neuronx_distributed_inference.modules.checkpoint import (
    load_state_dict,
    save_state_dict_safetensors,
)


# FP8 range: OCP E4M3/e4m3fn (HuggingFace) = +-448, Neuron E4M3 (IEEE-754) = +-240
FP8_SCALING_FACTOR = 448.0 / 240.0


def rescale_fp8_weight_blockwise(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: list = [128, 128]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rescale FP8 weight from OCP range to Neuron range, converting blockwise
    scales to per-channel (per-row) scales.

    The NxDI MoE kernels (MoEFusedTKG and CTE blockwise matmul) expect
    per-channel scales, not blockwise grids. This function:
    1. Dequantizes the FP8 weight using blockwise scales
    2. Computes per-row max abs for a per-channel scale
    3. Requantizes to FP8 in Neuron range with per-channel scale

    Args:
        weight: FP8 weight (float8_e4m3fn), shape [out_features, in_features]
        scale_inv: Block-wise inverse scale (weight_scale_inv), shape [sh, sw]
        block_size: Blockwise quantization block size [block_h, block_w]

    Returns:
        (rescaled_weight_fp8, per_channel_scale):
        - rescaled_weight_fp8: FP8 weight in Neuron range, per-channel quantized
        - per_channel_scale: Per-row scale [out_features], FP32
    """
    # Neuron FP8 E4M3 max value (IEEE-754 range)
    NEURON_FP8_MAX = 240.0

    # Step 1: Dequantize blockwise: original_bf16 = fp8_weight * scale_inv (per block)
    h, w = weight.shape
    block_h, block_w = block_size
    weight_float = weight.float()
    dequantized = torch.zeros(h, w, dtype=torch.float32)

    sh, sw = scale_inv.shape
    for i in range(sh):
        for j in range(sw):
            h_start = i * block_h
            h_end = min((i + 1) * block_h, h)
            w_start = j * block_w
            w_end = min((j + 1) * block_w, w)
            dequantized[h_start:h_end, w_start:w_end] = (
                weight_float[h_start:h_end, w_start:w_end] * scale_inv[i, j].item()
            )

    # Step 2: Compute per-row (per-channel) scale
    row_max_abs = dequantized.abs().amax(dim=1)  # [out_features]
    per_channel_scale = row_max_abs / NEURON_FP8_MAX
    per_channel_scale = torch.clamp(per_channel_scale, min=1e-10)

    # Step 3: Requantize to FP8 in Neuron range
    scale_expanded = per_channel_scale.unsqueeze(1)  # [out_features, 1]
    requantized = (dequantized / scale_expanded).to(torch.float8_e4m3fn)

    return requantized, per_channel_scale.to(torch.float32)


def dequantize_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: List[int],
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize FP8 weight using block-wise scales.

    Args:
        weight: FP8 weight, shape [out_features, in_features]
        scale_inv: Block-wise inverse scale, shape [sh, sw]
        block_size: [block_h, block_w], typically [128, 128]
        target_dtype: Output dtype (default bfloat16)

    Returns:
        Dequantized weight in target_dtype
    """
    h, w = weight.shape
    block_h, block_w = block_size
    scale_h, scale_w = scale_inv.shape

    weight_float = weight.float()
    dequantized = torch.zeros(h, w, dtype=torch.float32)

    for i in range(scale_h):
        for j in range(scale_w):
            h_start = i * block_h
            h_end = min((i + 1) * block_h, h)
            w_start = j * block_w
            w_end = min((j + 1) * block_w, w)

            block_scale = scale_inv[i, j].item()
            dequantized[h_start:h_end, w_start:w_end] = (
                weight_float[h_start:h_end, w_start:w_end] * block_scale
            )

    return dequantized.to(target_dtype)


def process_minimax_m2_checkpoint(
    hf_model_path: str,
    save_path: str,
):
    """
    Process MiniMax-M2 checkpoint for Neuron FP8 inference.

    Strategy:
    - MoE expert weights: Keep in FP8, rescale to Neuron range, output .scale
    - Attention weights: Dequantize to BF16 (small, not worth FP8 complexity)
    - Router, embeddings, norms, bias: Pass through as-is
    - Output preserves HF key naming (with model. prefix stripped) so that
      convert_hf_to_neuron_state_dict can process it normally
    """
    print(f"Loading checkpoint from: {hf_model_path}", flush=True)
    state_dict = load_state_dict(hf_model_path)

    # Load config
    config_path = os.path.join(hf_model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_experts = config["num_local_experts"]

    quant_config = config.get("quantization_config", {})
    block_size = quant_config.get("weight_block_size", [128, 128])

    print(f"\nModel configuration:", flush=True)
    print(f"  num_layers: {num_layers}", flush=True)
    print(f"  hidden_size: {hidden_size}", flush=True)
    print(f"  intermediate_size: {intermediate_size}", flush=True)
    print(f"  num_experts: {num_experts}", flush=True)
    print(f"  block_size: {block_size}", flush=True)

    state_dict_keys = list(state_dict.keys())
    new_state_dict: Dict[str, torch.Tensor] = {}

    # Strip "model." prefix from all keys (matching get_state_dict behavior)
    # The base class get_state_dict strips "model." before calling convert_state_dict,
    # but since we're saving a preprocessed checkpoint that will be re-loaded by
    # get_state_dict, we save WITHOUT the model. prefix so it doesn't get double-stripped.
    # Actually, we need to check: does get_state_dict strip "model." from our saved
    # checkpoint too? Yes it does. So we should keep "model." prefix so it gets stripped
    # once during loading.
    #
    # Wait -- the MiniMax get_state_dict at line 1360 strips "model." prefix from loaded
    # keys. If our preprocessed checkpoint has "model." prefix, it gets stripped correctly.
    # If it doesn't have "model." prefix, the strip is a no-op.
    # Let's keep the "model." prefix for consistency with HF format.

    for layer_idx in range(num_layers):
        print(f"\nProcessing layer {layer_idx}/{num_layers - 1}...", end="", flush=True)

        prefix = f"model.layers.{layer_idx}."

        # --- Attention weights: dequantize FP8 -> BF16 ---
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_key = f"{prefix}self_attn.{proj}.weight"
            scale_key = f"{prefix}self_attn.{proj}.weight_scale_inv"

            if weight_key not in state_dict:
                continue

            weight = state_dict[weight_key]
            scale_inv = state_dict.get(scale_key)

            if weight.dtype == torch.float8_e4m3fn and scale_inv is not None:
                weight = dequantize_blockwise(weight, scale_inv, block_size)
                if scale_key in state_dict:
                    del state_dict[scale_key]
            elif weight.dtype != torch.bfloat16:
                weight = weight.to(torch.bfloat16)

            new_state_dict[weight_key] = weight
            del state_dict[weight_key]
            # No .scale for attention (dequantized to BF16)

        # --- QK norm, layer norms: pass through ---
        for norm_key_suffix in [
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]:
            key = f"{prefix}{norm_key_suffix}"
            if key in state_dict:
                new_state_dict[key] = state_dict.pop(key)

        # --- Router gate weight + e_score_correction_bias: pass through ---
        for suffix in [
            "block_sparse_moe.gate.weight",
            "block_sparse_moe.e_score_correction_bias",
        ]:
            key = f"{prefix}{suffix}"
            if key in state_dict:
                new_state_dict[key] = state_dict.pop(key)

        # --- MoE expert weights: rescale FP8 to Neuron range ---
        for expert_idx in range(num_experts):
            exp_prefix = f"{prefix}block_sparse_moe.experts.{expert_idx}."

            for w_name in ["w1", "w2", "w3"]:
                w_key = f"{exp_prefix}{w_name}.weight"
                s_key = f"{exp_prefix}{w_name}.weight_scale_inv"

                if w_key not in state_dict:
                    continue

                weight = state_dict[w_key]
                scale_inv = state_dict.get(s_key)

                if weight.dtype == torch.float8_e4m3fn and scale_inv is not None:
                    weight, scale = rescale_fp8_weight_blockwise(
                        weight, scale_inv, block_size
                    )
                    new_state_dict[w_key] = weight
                    # Save as .scale (NOT .weight_scale, since MiniMax-M2's
                    # get_state_dict doesn't do the .weight_scale -> .scale rename
                    # that the base class does)
                    new_state_dict[f"{exp_prefix}{w_name}.scale"] = scale
                    del state_dict[w_key]
                    if s_key in state_dict:
                        del state_dict[s_key]
                else:
                    # BF16 weight, just pass through
                    new_state_dict[w_key] = state_dict.pop(w_key)

            if (expert_idx + 1) % 64 == 0:
                gc.collect()

        gc.collect()
        print(" done", flush=True)

    # ---- Embeddings and final norm ----
    print("\nProcessing embeddings and final norm...", flush=True)

    for key in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        if key in state_dict:
            new_state_dict[key] = state_dict[key]

    # ---- Save preprocessed checkpoint ----
    print(f"\nSaving preprocessed checkpoint to: {save_path}", flush=True)
    os.makedirs(save_path, exist_ok=True)

    save_state_dict_safetensors(new_state_dict, save_path)

    # Copy config.json (update to note Neuron preprocessing)
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config_data["neuron_preprocessed"] = True
    config_data["neuron_preprocessing_note"] = (
        "FP8 weights rescaled from OCP range (448) to Neuron range (240). "
        "Attention weights dequantized to BF16. Expert weight_scale_inv "
        "converted to .scale (direct scale)."
    )
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)

    # Copy tokenizer files
    for tok_file in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ]:
        src = os.path.join(hf_model_path, tok_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(save_path, tok_file))

    # Copy model code files (needed for trust_remote_code)
    for code_file in ["configuration_minimax_m2.py", "modeling_minimax_m2.py"]:
        src = os.path.join(hf_model_path, code_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(save_path, code_file))

    print(f"\nPreprocessing complete!", flush=True)
    print(f"  Total parameters: {len(new_state_dict)}", flush=True)

    fp8_count = sum(
        1 for v in new_state_dict.values() if v.dtype == torch.float8_e4m3fn
    )
    scale_count = sum(1 for k in new_state_dict.keys() if k.endswith(".scale"))
    bf16_count = sum(1 for v in new_state_dict.values() if v.dtype == torch.bfloat16)
    print(f"  FP8 weights: {fp8_count}", flush=True)
    print(f"  Scale parameters: {scale_count}", flush=True)
    print(f"  BF16 weights: {bf16_count}", flush=True)

    total_bytes = sum(v.numel() * v.element_size() for v in new_state_dict.values())
    print(f"  Total size: {total_bytes / 1e9:.1f} GB", flush=True)

    # Verify expected counts
    expected_fp8 = num_layers * num_experts * 3  # w1, w2, w3 per expert per layer
    expected_scales = expected_fp8
    print(f"  Expected FP8 weights: {expected_fp8} (got {fp8_count})", flush=True)
    print(f"  Expected scales: {expected_scales} (got {scale_count})", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MiniMax-M2 FP8 checkpoint for Neuron inference"
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to HuggingFace MiniMax-M2 checkpoint (MiniMaxAI/MiniMax-M2)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save preprocessed checkpoint",
    )

    args = parser.parse_args()

    process_minimax_m2_checkpoint(
        hf_model_path=args.hf_model_path,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
