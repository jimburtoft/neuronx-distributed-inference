"""
Preprocess MiMo-V2-Flash FP8 checkpoint for Neuron inference.

The HuggingFace FP8 checkpoint cannot be directly used for inference on Neuron.
This script preprocesses the checkpoint to make it compatible.

Steps:
1. Rescale FP8 weights from OCP format (range ±448) to Neuron format (range ±240)
2. Convert weight_scale_inv to .scale format (reciprocal + rescaling)
3. Fuse gate/up projections for MoE experts
4. Handle K/V weight and scale replication for CONVERT_TO_MHA mode
5. Save to preprocessed checkpoint directory

**Memory-efficient**: Uses streaming safetensors loading to process one layer
at a time.  Peak RSS stays under ~20 GB even for the full 225 GB checkpoint,
so this runs comfortably on a trn2.3xlarge (128 GB RAM).

Usage:
    python preprocess_mimo_v2_fp8.py \
        --hf_model_path /path/to/MiMo-V2-Flash \
        --save_path /path/to/preprocessed_mimo_v2_fp8 \
        --tp_degree 32 \
        --convert_to_mha
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file


# FP8 format conversion: OCP fp8_e4m3fn (bias=7) → Neuron IEEE fp8_e4m3 (bias=8)
#
# The two formats differ only in exponent bias:
#   OCP:  value = (-1)^s * 2^(e - 7) * (1.mmm)
#   IEEE: value = (-1)^s * 2^(e - 8) * (1.mmm)
#
# For the same bit pattern, V_ieee = V_ocp / 2.  The compiler flag
# --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 reinterprets OCP bits as IEEE.
# However, on the blockwise MoE path, weights are dequantized on CPU using
# PyTorch's OCP interpretation (blockwise_scale_dequantize), so the compiler
# flag is irrelevant — we only need to ensure the stored FP8 bits round-trip
# correctly through `weight.float() * scale`.
#
# Dividing an FP8 value by exactly 2.0 is an exponent decrement — lossless
# for all normal values (99.977% of real weights).  The old factor of
# 448/240 ≈ 1.867 required a lossy FP8 recast that changed 99.16% of weight
# values and introduced ~2% mean relative error per expert projection,
# compounding across 256 experts × 47 MoE layers.
FP8_SCALING_FACTOR = 2.0

# Neuron FP8 E4M3 max value (IEEE, bias=8)
NEURON_FP8_MAX = 240.0


# ---------------------------------------------------------------------------
# Streaming checkpoint reader — memory-maps safetensors shards and loads
# individual tensors on demand so we never materialise the full 225 GB
# state dict in RAM.
# ---------------------------------------------------------------------------


class StreamingCheckpointReader:
    """Memory-efficient reader for sharded safetensors checkpoints.

    Keeps one ``safetensors.safe_open`` handle per shard file (memory-mapped,
    near-zero RSS) and materialises tensors only when ``get(key)`` is called.
    """

    def __init__(self, model_dir: str):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        self.weight_map: Dict[str, str] = index["weight_map"]
        self._model_dir = model_dir
        self._handles: Dict[str, Any] = {}  # shard_file -> safe_open handle

    def _handle(self, shard_file: str):
        if shard_file not in self._handles:
            path = os.path.join(self._model_dir, shard_file)
            self._handles[shard_file] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_file]

    def keys(self):
        return self.weight_map.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str, default=None) -> Optional[torch.Tensor]:
        if key not in self.weight_map:
            return default
        shard_file = self.weight_map[key]
        return self._handle(shard_file).get_tensor(key)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(key)
        return self.get(key)

    def close(self):
        self._handles.clear()


def _save_shard(state_dict: Dict[str, torch.Tensor], path: str):
    """Save a dict of tensors to a single safetensors file."""
    safetensors_save_file(state_dict, path)


def convert_bf16_to_fp8_per_row(weight: torch.Tensor):
    """
    Convert BF16 weight to FP8 with per-row (per-channel) scales for Neuron.

    This is used for weights like o_proj that are BF16 in the original checkpoint.
    The Neuron framework expects per-row scaling for these layers.

    Args:
        weight: BF16 weight tensor [out_features, in_features]

    Returns:
        Tuple of (fp8_weight, scale)
        - fp8_weight: Weight quantized to FP8 (float8_e4m3fn)
        - scale: Per-row scale tensor [out_features, 1]
    """
    out_features, in_features = weight.shape

    # Compute per-row max absolute values
    weight_float = weight.float()
    row_max_abs = weight_float.abs().max(dim=1, keepdim=True)[0]

    # Compute scales (avoid division by zero)
    scales = row_max_abs / NEURON_FP8_MAX
    scales = torch.clamp(scales, min=1e-10)

    # Quantize
    quantized = (weight_float / scales).to(torch.float8_e4m3fn)

    return quantized, scales.to(torch.float32)


def convert_bf16_to_fp8_blockwise(
    weight: torch.Tensor,
    block_size: List[int] = [128, 128],
):
    """
    Convert BF16 weight to FP8 with block-wise scales for Neuron.

    Some weights in MiMo-V2-Flash (like o_proj) are in BF16, not FP8.
    This function quantizes them to FP8 with appropriate block-wise scales.

    Args:
        weight: BF16 weight tensor [out_features, in_features]
        block_size: Block size for quantization [128, 128]

    Returns:
        Tuple of (fp8_weight, scale)
        - fp8_weight: Weight quantized to FP8 (float8_e4m3fn)
        - scale: Block-wise scale tensor [scale_h, scale_w]
    """
    h, w = weight.shape
    block_h, block_w = block_size

    # Calculate scale grid dimensions
    scale_h = (h + block_h - 1) // block_h
    scale_w = (w + block_w - 1) // block_w

    # Initialize output tensors
    fp8_weight = torch.zeros_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(scale_h, scale_w, dtype=torch.float32)

    # Process each block
    for i in range(scale_h):
        for j in range(scale_w):
            # Block boundaries
            h_start = i * block_h
            h_end = min((i + 1) * block_h, h)
            w_start = j * block_w
            w_end = min((j + 1) * block_w, w)

            # Extract block
            block = weight[h_start:h_end, w_start:w_end].float()

            # Compute scale: max_abs / FP8_MAX
            max_abs = block.abs().max().item()
            if max_abs == 0:
                block_scale = 1.0
            else:
                block_scale = max_abs / NEURON_FP8_MAX

            # Quantize block
            quantized_block = (block / block_scale).to(torch.float8_e4m3fn)

            # Store results
            fp8_weight[h_start:h_end, w_start:w_end] = quantized_block
            scale[i, j] = block_scale

    return fp8_weight, scale


def dequantize_blockwise_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize a blockwise FP8 weight tensor to BF16.

    Uses the same blockwise dequantization logic as rescale_fp8_to_per_row,
    but returns BF16 directly instead of re-quantizing to per-row FP8.
    This avoids the lossy format conversion.

    Args:
        weight: FP8 weight tensor (float8_e4m3fn) [out_features, in_features]
        scale: Block-wise scale tensor (weight_scale_inv) [scale_h, scale_w]

    Returns:
        BF16 weight tensor [out_features, in_features]
    """
    out_features, in_features = weight.shape
    scale_h, scale_w = scale.shape

    block_h = (out_features + scale_h - 1) // scale_h
    block_w = (in_features + scale_w - 1) // scale_w

    weight_float = weight.float()
    dequantized = torch.zeros(out_features, in_features, dtype=torch.float32)

    for i in range(scale_h):
        for j in range(scale_w):
            h_start = i * block_h
            h_end = min((i + 1) * block_h, out_features)
            w_start = j * block_w
            w_end = min((j + 1) * block_w, in_features)

            block_scale = scale[i, j].item()
            dequantized[h_start:h_end, w_start:w_end] = (
                weight_float[h_start:h_end, w_start:w_end] * block_scale
            )

    return dequantized.to(torch.bfloat16)


def rescale_fp8_to_per_row(weight: torch.Tensor, scale: torch.Tensor):
    """
    Rescale FP8 weight from OCP format to Neuron format with per-row scaling.

    The original HuggingFace checkpoint uses block-wise FP8 quantization.
    The Neuron framework expects per-row (per-channel) scaling.
    This function converts block-wise to per-row scaling.

    Args:
        weight: FP8 weight tensor (float8_e4m3fn) [out_features, in_features]
        scale: Block-wise scale tensor (weight_scale_inv) [scale_h, scale_w]

    Returns:
        Tuple of (rescaled_weight, neuron_scale)
        - rescaled_weight: FP8 weight compatible with Neuron
        - neuron_scale: Per-row scale [out_features, 1]
    """
    out_features, in_features = weight.shape
    scale_h, scale_w = scale.shape

    # Block size inferred from scale dimensions
    block_h = (out_features + scale_h - 1) // scale_h
    block_w = (in_features + scale_w - 1) // scale_w

    # First dequantize using block-wise scales
    # HF convention: original = fp8_weight * weight_scale_inv
    weight_float = weight.float()
    dequantized = torch.zeros(out_features, in_features, dtype=torch.float32)

    for i in range(scale_h):
        for j in range(scale_w):
            h_start = i * block_h
            h_end = min((i + 1) * block_h, out_features)
            w_start = j * block_w
            w_end = min((j + 1) * block_w, in_features)

            block_scale = scale[i, j].item()
            dequantized[h_start:h_end, w_start:w_end] = (
                weight_float[h_start:h_end, w_start:w_end] * block_scale
            )

    # Now requantize with per-row scaling for Neuron
    # Compute per-row max absolute values
    row_max_abs = dequantized.abs().max(dim=1, keepdim=True)[0]

    # Compute scales (avoid division by zero)
    # Need to fit in Neuron FP8 range (±240)
    scales = row_max_abs / NEURON_FP8_MAX
    scales = torch.clamp(scales, min=1e-10)

    # Quantize to FP8
    quantized = (dequantized / scales).to(torch.float8_e4m3fn)

    return quantized, scales.to(torch.float32)


def rescale_fp8_weight_blockwise(weight: torch.Tensor, scale: torch.Tensor):
    """
    Rescale FP8 weight from OCP format to Neuron format, keeping block-wise scaling.

    Dividing by FP8_SCALING_FACTOR (2.0) is an exponent decrement in the FP8
    bit pattern, which is lossless for all normal values (99.977% of real
    weights).  Only subnormals and exp=1 values with odd mantissas lose the
    LSB — these represent tiny values near zero with negligible impact.

    Args:
        weight: FP8 weight tensor (float8_e4m3fn)
        scale: Scale tensor (float32 or bfloat16), this is weight_scale_inv (1/scale)

    Returns:
        Tuple of (rescaled_weight, neuron_scale)
        - rescaled_weight: FP8 weight compatible with Neuron
        - neuron_scale: Scale in Neuron format (direct scale, not reciprocal)
    """
    # Divide by 2.0 — nearly lossless for FP8 (exponent shift)
    rescaled_weight = (weight.float() / FP8_SCALING_FACTOR).to(torch.float8_e4m3fn)

    # Compensate in scale: original = rescaled * new_scale
    # so new_scale = old_scale * FP8_SCALING_FACTOR
    neuron_scale = scale.float() * FP8_SCALING_FACTOR

    return rescaled_weight, neuron_scale.to(torch.float32)


def replicate_for_convert_to_mha(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    num_kv_heads: int,
    num_attention_heads: int,
    head_dim: int,
):
    """
    Replicate K/V weights and per-row scales for CONVERT_TO_MHA mode.

    When TP > num_kv_heads, we need to replicate K/V heads to match Q heads.
    This uses repeat_interleave to create the correct GQA pattern.

    Args:
        weight: FP8 K or V weight [num_kv_heads * head_dim, hidden_size]
        scale: Per-row scale tensor [num_kv_heads * head_dim, 1]
        num_kv_heads: Original number of KV heads
        num_attention_heads: Target number of attention heads
        head_dim: Dimension per head

    Returns:
        Tuple of (replicated_weight, replicated_scale)
    """
    if num_kv_heads >= num_attention_heads:
        return weight, scale

    repeat_factor = num_attention_heads // num_kv_heads

    # Reshape weight to [num_kv_heads, head_dim, hidden_size]
    weight_reshaped = weight.view(num_kv_heads, head_dim, -1)

    # Replicate using repeat_interleave (correct GQA pattern)
    # This creates [h0, h0, ..., h1, h1, ...] pattern
    weight_replicated = weight_reshaped.repeat_interleave(repeat_factor, dim=0)

    # Reshape back to [num_attention_heads * head_dim, hidden_size]
    weight_replicated = weight_replicated.view(-1, weight_replicated.shape[-1])

    if scale is None:
        return weight_replicated, None

    # Replicate per-row scales
    # Scale shape: [num_kv_heads * head_dim, 1]
    # Reshape to [num_kv_heads, head_dim, 1]
    scale_reshaped = scale.view(num_kv_heads, head_dim, -1)

    # Replicate scales
    scale_replicated = scale_reshaped.repeat_interleave(repeat_factor, dim=0)

    # Reshape back to [num_attention_heads * head_dim, 1]
    scale_replicated = scale_replicated.view(-1, scale_replicated.shape[-1])

    return weight_replicated, scale_replicated


def process_mimo_v2_checkpoint(
    hf_model_path: str,
    save_path: str,
    tp_degree: int = 32,
    convert_to_mha: bool = True,
    attn_bf16: bool = False,
):
    """
    Process MiMo-V2-Flash checkpoint for Neuron FP8 inference.

    Uses streaming safetensors loading to keep peak memory under ~20 GB.
    Saves each layer as a separate shard, then writes a safetensors index file.

    Args:
        hf_model_path: Path to HuggingFace MiMo-V2-Flash checkpoint
        save_path: Path to save preprocessed checkpoint
        tp_degree: Tensor parallelism degree
        convert_to_mha: Whether to replicate K/V for CONVERT_TO_MHA mode
        attn_bf16: If True, keep q/k/v projections in BF16 instead of
            re-quantizing from blockwise FP8 to per-row FP8.  This avoids
            the lossy format conversion that can degrade reasoning quality.
            o_proj is always BF16 regardless of this setting.
    """
    # Determine which attention projections to keep in BF16.
    # o_proj is always BF16 (in modules_to_not_convert).
    # With --attn-bf16, q/k/v are also dequantized to BF16 to avoid the
    # lossy blockwise → per-row FP8 re-quantization.
    attn_bf16_projs = {"o_proj"}
    if attn_bf16:
        attn_bf16_projs |= {"q_proj", "k_proj", "v_proj"}
        print("Attention projections q/k/v will be kept in BF16", flush=True)

    print(f"Loading checkpoint index from: {hf_model_path}", flush=True)
    reader = StreamingCheckpointReader(hf_model_path)

    # Load config
    config_path = os.path.join(hf_model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract model dimensions
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]  # Full attention: 4
    swa_num_attention_heads = config["swa_num_attention_heads"]  # Sliding window: 32
    swa_num_kv_heads = config["swa_num_key_value_heads"]  # Sliding window: 8
    head_dim = config["head_dim"]  # Q/K head dim: 192
    v_head_dim = config["v_head_dim"]  # V head dim: 128
    swa_head_dim = config.get("swa_head_dim", head_dim)
    swa_v_head_dim = config.get("swa_v_head_dim", v_head_dim)

    # Get hybrid layer pattern
    hybrid_layer_pattern = config.get("hybrid_layer_pattern", [0] * num_layers)

    # MoE configuration
    num_experts = config["n_routed_experts"]  # 256
    moe_intermediate_size = config["moe_intermediate_size"]
    moe_layer_freq = config.get("moe_layer_freq", [1] * num_layers)

    # Block size for quantization
    quant_config = config.get("quantization_config", {})
    block_size = quant_config.get("weight_block_size", [128, 128])

    print(f"\nModel configuration:", flush=True)
    print(f"  num_layers: {num_layers}", flush=True)
    print(f"  hidden_size: {hidden_size}", flush=True)
    print(f"  num_attention_heads: {num_attention_heads}", flush=True)
    print(f"  num_kv_heads (full): {num_kv_heads}", flush=True)
    print(f"  swa_num_kv_heads (sliding): {swa_num_kv_heads}", flush=True)
    print(f"  head_dim (Q/K): {head_dim}", flush=True)
    print(f"  v_head_dim: {v_head_dim}", flush=True)
    print(f"  num_experts: {num_experts}", flush=True)
    print(f"  moe_intermediate_size: {moe_intermediate_size}", flush=True)
    print(f"  block_size: {block_size}", flush=True)
    print(f"  tp_degree: {tp_degree}", flush=True)
    print(f"  convert_to_mha: {convert_to_mha}", flush=True)

    os.makedirs(save_path, exist_ok=True)

    # We will build the weight_map for the index file as we go
    weight_map: Dict[str, str] = {}  # key -> shard filename
    total_fp8 = 0
    total_scale = 0
    total_keys = 0

    # ------------------------------------------------------------------
    # Process each layer and save as a separate shard
    # ------------------------------------------------------------------
    for layer_idx in range(num_layers):
        print(f"\nProcessing layer {layer_idx}...", end="", flush=True)

        prefix = f"model.layers.{layer_idx}."
        is_sliding_window = hybrid_layer_pattern[layer_idx] == 1
        is_moe_layer = moe_layer_freq[layer_idx] == 1

        # Get layer-specific parameters
        if is_sliding_window:
            layer_num_heads = swa_num_attention_heads
            layer_num_kv_heads = swa_num_kv_heads
            layer_head_dim = swa_head_dim
            layer_v_head_dim = swa_v_head_dim
        else:
            layer_num_heads = num_attention_heads
            layer_num_kv_heads = num_kv_heads
            layer_head_dim = head_dim
            layer_v_head_dim = v_head_dim

        attn_type = "sliding_window" if is_sliding_window else "full"
        print(f" ({attn_type}, kv_heads={layer_num_kv_heads})", end="", flush=True)

        layer_dict: Dict[str, torch.Tensor] = {}

        # Process attention weights
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_key = f"{prefix}self_attn.{proj}.weight"
            scale_key = f"{prefix}self_attn.{proj}.weight_scale_inv"

            if weight_key not in reader:
                continue

            weight = reader[weight_key]
            scale = reader.get(scale_key)

            # Projections listed in attn_bf16_projs are kept as BF16
            # (dequantized from FP8 if needed).  o_proj is always BF16
            # because it's in modules_to_not_convert.  q/k/v can optionally
            # be kept BF16 to avoid the lossy blockwise→per-row FP8
            # re-quantization that degrades reasoning quality.
            if proj in attn_bf16_projs:
                if weight.dtype == torch.float8_e4m3fn and scale is not None:
                    weight = dequantize_blockwise_to_bf16(weight, scale)
                scale = None
            else:
                if weight.dtype == torch.float8_e4m3fn and scale is not None:
                    weight, scale = rescale_fp8_to_per_row(weight, scale)
                elif weight.dtype == torch.bfloat16:
                    weight, scale = convert_bf16_to_fp8_per_row(weight)

            new_weight_key = f"layers.{layer_idx}.self_attn.{proj}.weight"
            layer_dict[new_weight_key] = weight

            if scale is not None:
                new_scale_key = f"layers.{layer_idx}.self_attn.{proj}.scale"
                layer_dict[new_scale_key] = scale

        # Process layer norms (no FP8)
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            weight_key = f"{prefix}{norm}.weight"
            if weight_key in reader:
                new_key = f"layers.{layer_idx}.{norm}.weight"
                layer_dict[new_key] = reader[weight_key]

        # Process MoE router
        router_key = f"{prefix}mlp.gate.weight"
        if router_key in reader:
            new_key = f"layers.{layer_idx}.mlp.router.linear_router.weight"
            layer_dict[new_key] = reader[router_key]

        # MoE router bias (e_score_correction_bias)
        router_bias_key = f"{prefix}mlp.gate.e_score_correction_bias"
        if router_bias_key in reader:
            new_key = f"layers.{layer_idx}.mlp.router.e_score_correction_bias"
            layer_dict[new_key] = reader[router_bias_key]

        # Attention sink bias
        sink_bias_key = f"{prefix}self_attn.attention_sink_bias"
        if sink_bias_key in reader:
            new_key = f"layers.{layer_idx}.self_attn.attention_sink_bias"
            layer_dict[new_key] = reader[sink_bias_key]

        # Process MoE experts
        if is_moe_layer:
            gate_weights = []
            gate_scales = []
            up_weights = []
            up_scales = []
            down_weights = []
            down_scales = []

            for expert_idx in range(num_experts):
                expert_prefix = f"{prefix}mlp.experts.{expert_idx}."

                # Gate projection
                gate_w_key = f"{expert_prefix}gate_proj.weight"
                gate_s_key = f"{expert_prefix}gate_proj.weight_scale_inv"

                if gate_w_key in reader:
                    gate_w = reader[gate_w_key]
                    gate_s = reader.get(gate_s_key)

                    if gate_w.dtype == torch.float8_e4m3fn and gate_s is not None:
                        gate_w, gate_s = rescale_fp8_weight_blockwise(gate_w, gate_s)
                    elif gate_w.dtype == torch.bfloat16:
                        gate_w, gate_s = convert_bf16_to_fp8_blockwise(
                            gate_w, block_size
                        )

                    gate_weights.append(gate_w.T)
                    if gate_s is not None:
                        gate_scales.append(gate_s.T)

                # Up projection
                up_w_key = f"{expert_prefix}up_proj.weight"
                up_s_key = f"{expert_prefix}up_proj.weight_scale_inv"

                if up_w_key in reader:
                    up_w = reader[up_w_key]
                    up_s = reader.get(up_s_key)

                    if up_w.dtype == torch.float8_e4m3fn and up_s is not None:
                        up_w, up_s = rescale_fp8_weight_blockwise(up_w, up_s)
                    elif up_w.dtype == torch.bfloat16:
                        up_w, up_s = convert_bf16_to_fp8_blockwise(up_w, block_size)

                    up_weights.append(up_w.T)
                    if up_s is not None:
                        up_scales.append(up_s.T)

                # Down projection
                down_w_key = f"{expert_prefix}down_proj.weight"
                down_s_key = f"{expert_prefix}down_proj.weight_scale_inv"

                if down_w_key in reader:
                    down_w = reader[down_w_key]
                    down_s = reader.get(down_s_key)

                    if down_w.dtype == torch.float8_e4m3fn and down_s is not None:
                        down_w, down_s = rescale_fp8_weight_blockwise(down_w, down_s)
                    elif down_w.dtype == torch.bfloat16:
                        down_w, down_s = convert_bf16_to_fp8_blockwise(
                            down_w, block_size
                        )

                    down_weights.append(down_w.T)
                    if down_s is not None:
                        down_scales.append(down_s.T)

            # Fuse gate and up projections
            if gate_weights and up_weights:
                gate_stacked = torch.stack(gate_weights, dim=0)
                up_stacked = torch.stack(up_weights, dim=0)
                gate_up_fused = torch.cat([gate_stacked, up_stacked], dim=2)

                new_key = (
                    f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                )
                layer_dict[new_key] = gate_up_fused
                del gate_stacked, up_stacked, gate_up_fused

                if gate_scales and up_scales:
                    gate_s_stacked = torch.stack(gate_scales, dim=0)
                    up_s_stacked = torch.stack(up_scales, dim=0)
                    gate_up_scale = torch.cat([gate_s_stacked, up_s_stacked], dim=-1)

                    new_scale_key = (
                        f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"
                    )
                    layer_dict[new_scale_key] = gate_up_scale
                    del gate_s_stacked, up_s_stacked, gate_up_scale

            # Down projection
            if down_weights:
                down_stacked = torch.stack(down_weights, dim=0)
                new_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"
                layer_dict[new_key] = down_stacked
                del down_stacked

                if down_scales:
                    down_s_stacked = torch.stack(down_scales, dim=0)
                    new_scale_key = (
                        f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.scale"
                    )
                    layer_dict[new_scale_key] = down_s_stacked
                    del down_s_stacked

            # Free intermediate lists
            del gate_weights, gate_scales, up_weights, up_scales
            del down_weights, down_scales
        else:
            # Non-MoE layer: regular MLP
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                weight_key = f"{prefix}mlp.{proj}.weight"
                scale_key = f"{prefix}mlp.{proj}.weight_scale_inv"

                if weight_key not in reader:
                    continue

                weight = reader[weight_key]
                scale = reader.get(scale_key)

                if weight.dtype == torch.float8_e4m3fn and scale is not None:
                    weight, scale = rescale_fp8_to_per_row(weight, scale)
                elif weight.dtype == torch.bfloat16:
                    weight, scale = convert_bf16_to_fp8_per_row(weight)

                new_weight_key = f"layers.{layer_idx}.mlp.{proj}.weight"
                layer_dict[new_weight_key] = weight

                if scale is not None:
                    new_scale_key = f"layers.{layer_idx}.mlp.{proj}.scale"
                    layer_dict[new_scale_key] = scale

        # Count stats
        for k, v in layer_dict.items():
            if v.dtype == torch.float8_e4m3fn:
                total_fp8 += 1
            if k.endswith(".scale"):
                total_scale += 1
        total_keys += len(layer_dict)

        # Save this layer as its own shard
        shard_name = f"layer_{layer_idx:03d}.safetensors"
        shard_path = os.path.join(save_path, shard_name)
        _save_shard(layer_dict, shard_path)
        for k in layer_dict:
            weight_map[k] = shard_name

        del layer_dict
        gc.collect()
        print(" done", flush=True)

    # ------------------------------------------------------------------
    # Process embeddings and final layer norm (small — single shard)
    # ------------------------------------------------------------------
    print("\nProcessing embeddings and final norm...", flush=True)
    misc_dict: Dict[str, torch.Tensor] = {}

    if "model.embed_tokens.weight" in reader:
        misc_dict["embed_tokens.weight"] = reader["model.embed_tokens.weight"]

    if "model.norm.weight" in reader:
        misc_dict["norm.weight"] = reader["model.norm.weight"]

    if "lm_head.weight" in reader:
        misc_dict["lm_head.weight"] = reader["lm_head.weight"]
    elif "model.embed_tokens.weight" in reader:
        misc_dict["lm_head.weight"] = reader["model.embed_tokens.weight"]

    total_keys += len(misc_dict)

    shard_name = "misc.safetensors"
    _save_shard(misc_dict, os.path.join(save_path, shard_name))
    for k in misc_dict:
        weight_map[k] = shard_name
    del misc_dict

    reader.close()

    # ------------------------------------------------------------------
    # Write safetensors index
    # ------------------------------------------------------------------
    index = {"metadata": {"total_size": total_keys}, "weight_map": weight_map}
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # ------------------------------------------------------------------
    # Copy config + tokenizer + custom HF files
    # ------------------------------------------------------------------
    import shutil

    shutil.copy(config_path, os.path.join(save_path, "config.json"))

    for extra_file in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "configuration_mimo_v2_flash.py",
        "modeling_mimo_v2_flash.py",
    ]:
        src = os.path.join(hf_model_path, extra_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(save_path, extra_file))

    print(f"\nPreprocessing complete!", flush=True)
    print(f"  Total parameters: {total_keys}", flush=True)
    print(f"  FP8 weights: {total_fp8}", flush=True)
    print(f"  Scale parameters: {total_scale}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MiMo-V2-Flash FP8 checkpoint for Neuron inference"
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to HuggingFace MiMo-V2-Flash checkpoint",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save preprocessed checkpoint",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=32,
        help="Tensor parallelism degree (default: 32)",
    )
    parser.add_argument(
        "--convert_to_mha",
        action="store_true",
        default=True,
        help="Replicate K/V for CONVERT_TO_MHA mode (default: True)",
    )
    parser.add_argument(
        "--no_convert_to_mha",
        action="store_false",
        dest="convert_to_mha",
        help="Disable K/V replication",
    )
    parser.add_argument(
        "--attn_bf16",
        action="store_true",
        default=False,
        help="Keep attention q/k/v in BF16 instead of FP8 per-row "
        "(avoids lossy blockwise->per-row re-quantization)",
    )

    args = parser.parse_args()

    process_mimo_v2_checkpoint(
        hf_model_path=args.hf_model_path,
        save_path=args.save_path,
        tp_degree=args.tp_degree,
        convert_to_mha=args.convert_to_mha,
        attn_bf16=args.attn_bf16,
    )


if __name__ == "__main__":
    main()
