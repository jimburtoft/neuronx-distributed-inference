"""
Preprocess MiMo-V2-Pro FP8 checkpoint for Neuron inference.

Based on preprocess_mimo_v2_flash_fp8.py with adaptations for Pro:
  - Handles fused_qkv projection layout (Pro stores a single qkv_proj;
    Flash stores separate q/k/v projections). The fused QKV is split
    into separate q_proj, k_proj, v_proj during preprocessing.
  - CRITICAL: Pro's qkv_proj uses an interleaved num_kv_heads group
    layout, NOT simple [all_Q | all_K | all_V] concatenation. Each of
    num_kv_heads groups contains (heads_per_group Q heads, 1 K head,
    1 V head) packed contiguously. The FP8 blockwise scales also follow
    this group layout with phantom padding rows between K and V blocks.
    See split_qkv_fused() for full documentation.
  - Router bias mean-subtraction: e_score_correction_bias has mean ~71,
    and BF16 step size at that magnitude is ~0.5, which destroys the
    per-expert variation (std ~3e-4). We subtract the mean during
    preprocessing to preserve routing information in BF16.
  - Handles EP-sharded checkpoint format (model_pp0_ep*_shard*.safetensors).
    The LazyWeightMap reads model.safetensors.index.json which maps tensor
    names to shard files, so EP-sharding is transparent.
  - 384 experts (vs Flash's 256) -- handled dynamically from config.
  - 70 layers (vs Flash's 48) -- handled dynamically from config.

MiMo-V2.5-Pro checkpoint layout:
  - qkv_proj is a single fused FP8 weight per layer with shape
    [q_size + k_size + v_size, hidden_size] in interleaved group order.
  - o_proj is BF16 (listed in quantization_config.ignored_layers).
  - Layer 0 is a dense MLP (moe_layer_freq[0] == 0) with intermediate_size
    16384; layers 1..69 are MoE with 384 experts each.
  - Hybrid attention: 10 "full" layers + 60 "sliding window" layers.
  - EP-sharded safetensors (32 EP shards + MTP shard).

Output layout:
  save_path/
    config.json, tokenizer.*, etc.
    configuration_mimo_v2.py, modeling_mimo_v2.py (trust_remote_code)
    model.safetensors.index.json  (regenerated)
    model_extras.safetensors       (embed_tokens, norm, lm_head)
    model_layer{N}.safetensors     (one per decoder layer, N=0..69)

Usage:
    python preprocess_mimo_v2_pro_fp8.py \\
        --hf_model_path /mnt/models/MiMo-V2.5-Pro \\
        --save_path /mnt/models/MiMo-V2.5-Pro-Neuron-FP8 \\
        --tp_degree 64
"""

import argparse
import gc
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP8_SCALING_FACTOR = 448.0 / 240.0
NEURON_FP8_MAX = 240.0


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def convert_bf16_to_fp8_per_row(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 [out, in] -> Neuron FP8 per-row (scales shape [out, 1])."""
    weight_float = weight.float()
    row_max_abs = weight_float.abs().max(dim=1, keepdim=True)[0]
    scales = torch.clamp(row_max_abs / NEURON_FP8_MAX, min=1e-10)
    quantized = (weight_float / scales).to(torch.float8_e4m3fn)
    return quantized, scales.to(torch.float32)


def rescale_fp8_to_per_row(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 + blockwise scale -> Neuron per-row FP8.

    Dequantize to float32 using block broadcast, then per-row requantize.
    """
    out_features, in_features = weight.shape
    scale_h, scale_w = scale.shape

    block_h = (out_features + scale_h - 1) // scale_h
    block_w = (in_features + scale_w - 1) // scale_w

    weight_float = weight.float()
    dequantized = torch.zeros(out_features, in_features, dtype=torch.float32)
    for i in range(scale_h):
        for j in range(scale_w):
            h0, h1 = i * block_h, min((i + 1) * block_h, out_features)
            w0, w1 = j * block_w, min((j + 1) * block_w, in_features)
            dequantized[h0:h1, w0:w1] = weight_float[h0:h1, w0:w1] * scale[i, j].item()

    row_max_abs = dequantized.abs().max(dim=1, keepdim=True)[0]
    scales = torch.clamp(row_max_abs / NEURON_FP8_MAX, min=1e-10)
    quantized = (dequantized / scales).to(torch.float8_e4m3fn)
    return quantized, scales.to(torch.float32)


def rescale_fp8_weight_blockwise(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep blockwise scales, just rescale into Neuron FP8 range.

    MoE expert weights stay block-quantized; only the dtype range changes.
    """
    weight_bf16 = weight.bfloat16()
    rescaled = (weight_bf16 / FP8_SCALING_FACTOR).to(torch.float8_e4m3fn)
    neuron_scale = scale.float() * FP8_SCALING_FACTOR
    return rescaled, neuron_scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Streaming weight access (one open safetensors handle at a time)
# ---------------------------------------------------------------------------


class LazyWeightMap:
    """Lazily fetch tensors from sharded safetensors, keeping one handle live."""

    def __init__(self, model_dir: str, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._cur_filename: Optional[str] = None
        self._cur_handle = None

    def _open(self, filename: str):
        if self._cur_filename == filename:
            return self._cur_handle
        if self._cur_handle is not None:
            self._cur_handle.__exit__(None, None, None)
            self._cur_handle = None
        path = os.path.join(self.model_dir, filename)
        self._cur_handle = safe_open(path, framework="pt", device="cpu")
        self._cur_handle.__enter__()
        self._cur_filename = filename
        return self._cur_handle

    def get(self, key: str) -> Optional[torch.Tensor]:
        filename = self.weight_map.get(key)
        if filename is None:
            return None
        return self._open(filename).get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def close(self):
        if self._cur_handle is not None:
            self._cur_handle.__exit__(None, None, None)
            self._cur_handle = None
            self._cur_filename = None


# ---------------------------------------------------------------------------
# Per-tensor helpers
# ---------------------------------------------------------------------------


def _requantize_per_row(dequant: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Float -> Neuron FP8 per-row."""
    row_max_abs = dequant.abs().max(dim=1, keepdim=True)[0]
    scales = torch.clamp(row_max_abs / NEURON_FP8_MAX, min=1e-10)
    quantized = (dequant / scales).to(torch.float8_e4m3fn)
    return quantized, scales.to(torch.float32)


def split_qkv_fused(
    qkv_weight: torch.Tensor,
    qkv_scale: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Split Pro's pre-fused qkv_proj into q/k/v. MiMo-V2.5-Pro specific.

    HF layout -- cross-validated against sglang on H200:
    ``qkv_proj.weight`` is NOT ``[all_Q | all_K | all_V]``. It is
    num_kv_heads interleaved groups, each holding (heads_per_group Q
    heads, 1 K head, 1 V head) packed contiguously:

        group g (g = 0 .. num_kv_heads-1):
            rows [g*R     : g*R + qg]        = Q heads [g*hpg : (g+1)*hpg]
            rows [g*R+qg  : g*R + qg + kg]   = K head g
            rows [g*R+qg+kg : g*R + R]        = V head g
          where
            hpg = num_q_heads / num_kv_heads         (e.g. 128/8 = 16)
            qg  = hpg * head_dim                     (e.g. 16 * 192 = 3072)
            kg  = 1 * head_dim                       (e.g. 192)
            vg  = 1 * v_head_dim                     (e.g. 128)
            R   = qg + kg + vg                       (e.g. 3392)

    Scale: per-group 27 scale rows covering 27*128 = 3456 "padded" rows:
        24 rows for Q (24 * 128 = 3072 real Q rows)
         2 rows for K (1 full block + 1 half-real/half-phantom block)
         1 row for V (128 rows)
    Total: 8 * 27 = 216 scale rows, 8 * 3392 = 27136 weight rows.

    The "phantom" 64 rows sit between each group's K tail and V start in
    *scale block coordinates* only; in the physical weight tensor, group g's
    V is immediately followed by group (g+1)'s Q. We recover the correct
    dequant by padding each group up to 3456 rows before applying the scale,
    then stripping the phantom rows.
    """
    in_features = qkv_weight.shape[1]
    hpg = num_q_heads // num_kv_heads
    qg_rows = hpg * head_dim
    kg_rows = 1 * head_dim
    vg_rows = 1 * v_head_dim
    real_rows_per_group = qg_rows + kg_rows + vg_rows
    total_real_rows = num_kv_heads * real_rows_per_group

    BLOCK = 128
    q_scale_rows_per_group = qg_rows // BLOCK
    k_scale_rows_per_group = (kg_rows + BLOCK - 1) // BLOCK
    v_scale_rows_per_group = (vg_rows + BLOCK - 1) // BLOCK
    scale_rows_per_group = (
        q_scale_rows_per_group + k_scale_rows_per_group + v_scale_rows_per_group
    )
    padded_rows_per_group = scale_rows_per_group * BLOCK

    assert qkv_weight.shape[0] == total_real_rows, (
        f"qkv_proj.weight row count {qkv_weight.shape[0]} != "
        f"expected {total_real_rows} "
        f"(num_kv_heads={num_kv_heads}, R={real_rows_per_group})"
    )

    if qkv_weight.dtype != torch.float8_e4m3fn or qkv_scale is None:
        # BF16 path: simple reshape and split per group
        w = qkv_weight.view(num_kv_heads, real_rows_per_group, in_features)
        q_w = (
            w[:, :qg_rows, :].reshape(num_kv_heads * qg_rows, in_features).contiguous()
        )
        k_w = (
            w[:, qg_rows : qg_rows + kg_rows, :]
            .reshape(num_kv_heads * kg_rows, in_features)
            .contiguous()
        )
        v_w = (
            w[:, qg_rows + kg_rows :, :]
            .reshape(num_kv_heads * vg_rows, in_features)
            .contiguous()
        )
        q_w2, q_s2 = convert_bf16_to_fp8_per_row(q_w)
        k_w2, k_s2 = convert_bf16_to_fp8_per_row(k_w)
        v_w2, v_s2 = convert_bf16_to_fp8_per_row(v_w)
        return {"q_proj": (q_w2, q_s2), "k_proj": (k_w2, k_s2), "v_proj": (v_w2, v_s2)}

    # FP8 + blockwise scale path.
    expected_scale_rows = num_kv_heads * scale_rows_per_group
    expected_scale_cols = (in_features + BLOCK - 1) // BLOCK
    assert qkv_scale.shape == (expected_scale_rows, expected_scale_cols), (
        f"qkv scale shape {tuple(qkv_scale.shape)} != expected "
        f"({expected_scale_rows}, {expected_scale_cols})"
    )

    # Reshape weight and scale into per-group views
    w = qkv_weight.to(torch.float32).view(
        num_kv_heads, real_rows_per_group, in_features
    )
    # Pad each group to padded_rows_per_group (phantom rows for scale alignment)
    w_padded = torch.zeros(
        num_kv_heads, padded_rows_per_group, in_features, dtype=torch.float32
    )
    w_padded[:, :real_rows_per_group, :] = w

    s = qkv_scale.to(torch.float32).view(
        num_kv_heads, scale_rows_per_group, expected_scale_cols
    )
    # Expand scale to per-element by repeating each scale block 128x
    s_exp = s.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
    s_exp = s_exp[:, :padded_rows_per_group, :in_features]

    # Dequantize: multiply padded weight by expanded scale, then strip phantom rows
    deq_padded = w_padded * s_exp
    deq = deq_padded[:, :real_rows_per_group, :]

    # Extract Q, K, V per group and reassemble
    q_deq = (
        deq[:, :qg_rows, :].reshape(num_kv_heads * qg_rows, in_features).contiguous()
    )
    k_deq = (
        deq[:, qg_rows : qg_rows + kg_rows, :]
        .reshape(num_kv_heads * kg_rows, in_features)
        .contiguous()
    )
    v_deq = (
        deq[:, qg_rows + kg_rows :, :]
        .reshape(num_kv_heads * vg_rows, in_features)
        .contiguous()
    )

    # Re-quantize to Neuron per-row FP8
    q_w2, q_s2 = _requantize_per_row(q_deq)
    k_w2, k_s2 = _requantize_per_row(k_deq)
    v_w2, v_s2 = _requantize_per_row(v_deq)

    return {"q_proj": (q_w2, q_s2), "k_proj": (k_w2, k_s2), "v_proj": (v_w2, v_s2)}


def _maybe_fp8_to_neuron_per_row(
    weight: torch.Tensor, scale: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """FP8 blockwise -> per-row, or BF16 -> FP8 per-row. Pass-through otherwise."""
    if weight.dtype == torch.float8_e4m3fn and scale is not None:
        return rescale_fp8_to_per_row(weight, scale)
    if weight.dtype == torch.bfloat16:
        return convert_bf16_to_fp8_per_row(weight)
    return weight, scale


# ---------------------------------------------------------------------------
# Per-layer processing
# ---------------------------------------------------------------------------


def process_layer(
    layer_idx: int,
    lazy: LazyWeightMap,
    config: dict,
    is_dense: bool,
    is_swa: bool,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    prefix = f"model.layers.{layer_idx}."
    out_prefix = f"layers.{layer_idx}."

    # --- Layer norms (BF16, untouched) ---
    for name in ("input_layernorm", "post_attention_layernorm"):
        t = lazy.get(f"{prefix}{name}.weight")
        if t is not None:
            out[f"{out_prefix}{name}.weight"] = t.detach().clone()

    # --- Attention projections ---
    # Pro uses fused_qkv layout with interleaved num_kv_heads group order.
    # Flash uses separate q/k/v projections. Handle both.
    qkv_key = f"{prefix}self_attn.qkv_proj.weight"
    qkv_scale_key = f"{prefix}self_attn.qkv_proj.weight_scale_inv"

    if lazy.has(qkv_key):
        # Pro: fused_qkv -- split via interleaved group layout
        qkv_w = lazy.get(qkv_key)
        qkv_s = lazy.get(qkv_scale_key)

        # Compute attention dimensions from config (SWA vs full)
        num_q_heads = (
            config.get("swa_num_attention_heads", config["num_attention_heads"])
            if is_swa
            else config["num_attention_heads"]
        )
        num_kv_heads = (
            config.get("swa_num_key_value_heads", config["num_key_value_heads"])
            if is_swa
            else config["num_key_value_heads"]
        )
        head_dim = (
            config.get(
                "swa_head_dim",
                config.get(
                    "head_dim", config["hidden_size"] // config["num_attention_heads"]
                ),
            )
            if is_swa
            else config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            )
        )
        v_head_dim = (
            config.get("swa_v_head_dim", config.get("v_head_dim", head_dim))
            if is_swa
            else config.get("v_head_dim", head_dim)
        )

        split = split_qkv_fused(
            qkv_w, qkv_s, num_q_heads, num_kv_heads, head_dim, v_head_dim
        )
        for proj_name, (w2, s2) in split.items():
            out[f"{out_prefix}self_attn.{proj_name}.weight"] = w2
            if s2 is not None:
                out[f"{out_prefix}self_attn.{proj_name}.scale"] = s2

        del qkv_w, qkv_s
    else:
        # Flash: separate q/k/v projections
        for proj in ("q_proj", "k_proj", "v_proj"):
            w = lazy.get(f"{prefix}self_attn.{proj}.weight")
            if w is None:
                continue
            s = lazy.get(f"{prefix}self_attn.{proj}.weight_scale_inv")
            w2, s2 = _maybe_fp8_to_neuron_per_row(w, s)
            out[f"{out_prefix}self_attn.{proj}.weight"] = w2
            if s2 is not None:
                out[f"{out_prefix}self_attn.{proj}.scale"] = s2

    # o_proj is listed in HF quantization_config.ignored_layers and ships as
    # BF16; on Neuron it binds to a plain RowParallelLinear (see
    # modeling_mimo_v2.py: self.o_proj = RowParallelLinear(...)), NOT a
    # QuantizedRowParallel. Writing FP8 + .scale here would silently be
    # reinterpreted as BF16 bytes at load time and produce garbage outputs.
    # Keep BF16, never emit .scale.
    o_w = lazy.get(f"{prefix}self_attn.o_proj.weight")
    o_s = lazy.get(f"{prefix}self_attn.o_proj.weight_scale_inv")
    if o_w is not None:
        if o_w.dtype == torch.float8_e4m3fn:
            # Defensive: if a future checkpoint FP8-quantizes o_proj, dequant
            # blockwise back to BF16 (no per-row requant; RowParallelLinear has
            # no .scale parameter).
            assert o_s is not None, "FP8 o_proj requires weight_scale_inv"
            out_features, in_features = o_w.shape
            scale_h, scale_w = o_s.shape
            block_h = (out_features + scale_h - 1) // scale_h
            block_w = (in_features + scale_w - 1) // scale_w
            wf = o_w.float()
            tmp = torch.zeros(out_features, in_features, dtype=torch.float32)
            for i in range(scale_h):
                for j in range(scale_w):
                    h0, h1 = i * block_h, min((i + 1) * block_h, out_features)
                    w0, w1 = j * block_w, min((j + 1) * block_w, in_features)
                    tmp[h0:h1, w0:w1] = wf[h0:h1, w0:w1] * o_s[i, j].item()
            o_bf16 = tmp.to(torch.bfloat16)
        else:
            o_bf16 = o_w.to(torch.bfloat16)
        out[f"{out_prefix}self_attn.o_proj.weight"] = o_bf16.detach().clone()

    # --- attention_sink_bias: present only on SWA layers in MiMo-V2-Flash.
    # config.add_swa_attention_sink_bias=True, add_full_attention_sink_bias=False.
    if is_swa and config.get("add_swa_attention_sink_bias", False):
        sink = lazy.get(f"{prefix}self_attn.attention_sink_bias")
        if sink is not None:
            out[f"{out_prefix}self_attn.attention_sink_bias"] = sink.detach().clone()
    elif not is_swa and config.get("add_full_attention_sink_bias", False):
        sink = lazy.get(f"{prefix}self_attn.attention_sink_bias")
        if sink is not None:
            out[f"{out_prefix}self_attn.attention_sink_bias"] = sink.detach().clone()

    # --- MLP: dense vs MoE ---
    if is_dense:
        # Dense MLP: gate_proj, up_proj, down_proj (FP8 blockwise in Flash layer 0).
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = lazy.get(f"{prefix}mlp.{proj}.weight")
            if w is None:
                continue
            s = lazy.get(f"{prefix}mlp.{proj}.weight_scale_inv")
            w2, s2 = _maybe_fp8_to_neuron_per_row(w, s)
            out[f"{out_prefix}mlp.{proj}.weight"] = w2
            if s2 is not None:
                out[f"{out_prefix}mlp.{proj}.scale"] = s2
        return out

    # --- MoE ---
    # Router: mlp.gate -> mlp.router.linear_router
    router_w = lazy.get(f"{prefix}mlp.gate.weight")
    if router_w is not None:
        out[f"{out_prefix}mlp.router.linear_router.weight"] = router_w.detach().clone()
    router_bias = lazy.get(f"{prefix}mlp.gate.e_score_correction_bias")
    if router_bias is not None:
        # MiMo-V2.5-Pro: HF bias has mean ~71 with per-expert std ~3e-4.
        # NxDI casts router parameters to BF16 at load time, and BF16 step
        # size at magnitude 71 is ~0.5 — which completely wipes out the
        # per-expert std=3e-4 variation, collapsing all 384 experts to a
        # single bias value (~71.0) and reducing noaux_tc topk to plain
        # sigmoid topk. Subtracting the mean puts the bias near 0, where
        # BF16 step is ~2.4e-4 (small enough to preserve the variation).
        # topk is invariant to additive constants across all experts.
        bias_f32 = router_bias.detach().float().clone()
        bias_f32 = bias_f32 - bias_f32.mean()
        out[f"{out_prefix}mlp.router.e_score_correction_bias"] = bias_f32

    num_experts = config["n_routed_experts"]

    # Peek expert 0 to learn shapes/dtypes.
    e0_gw = lazy.get(f"{prefix}mlp.experts.0.gate_proj.weight")
    if e0_gw is None:
        return out  # no experts (shouldn't happen for MoE layers, but be safe)
    e0_gs = lazy.get(f"{prefix}mlp.experts.0.gate_proj.weight_scale_inv")

    if e0_gw.dtype == torch.float8_e4m3fn and e0_gs is not None:
        sample_w, sample_s = rescale_fp8_weight_blockwise(e0_gw, e0_gs)
    elif e0_gw.dtype == torch.bfloat16:
        # Should not happen for Flash (experts ship in FP8); flag loudly.
        raise NotImplementedError(
            f"Layer {layer_idx} expert 0 gate_proj is BF16; Flash expects FP8."
        )
    else:
        sample_w, sample_s = e0_gw, e0_gs

    intermediate_size, hidden_size = sample_w.shape  # [IM, H]
    # Packed transpose layout: [num_experts, H, 2*IM] for gate_up.
    gate_up_proj = torch.empty(
        num_experts, hidden_size, 2 * intermediate_size, dtype=sample_w.dtype
    )
    i_blocks, h_blocks = sample_s.shape  # [IM_blocks, H_blocks]
    gate_up_scale = torch.empty(
        num_experts, h_blocks, 2 * i_blocks, dtype=sample_s.dtype
    )

    e0_dw = lazy.get(f"{prefix}mlp.experts.0.down_proj.weight")
    e0_ds = lazy.get(f"{prefix}mlp.experts.0.down_proj.weight_scale_inv")
    if e0_dw.dtype == torch.float8_e4m3fn and e0_ds is not None:
        sample_dw, sample_ds = rescale_fp8_weight_blockwise(e0_dw, e0_ds)
    else:
        raise NotImplementedError(
            f"Layer {layer_idx} expert 0 down_proj dtype {e0_dw.dtype} not handled."
        )
    d_h_blocks, d_i_blocks = sample_ds.shape  # [H_blocks, IM_blocks]
    down_proj = torch.empty(
        num_experts, intermediate_size, hidden_size, dtype=sample_dw.dtype
    )
    down_scale = torch.empty(num_experts, d_i_blocks, d_h_blocks, dtype=sample_ds.dtype)

    # Slot expert 0 (already rescaled above).
    gate_up_proj[0, :, :intermediate_size] = sample_w.T
    gate_up_scale[0, :, :i_blocks] = sample_s.T
    e0_uw = lazy.get(f"{prefix}mlp.experts.0.up_proj.weight")
    e0_us = lazy.get(f"{prefix}mlp.experts.0.up_proj.weight_scale_inv")
    up_w0, up_s0 = rescale_fp8_weight_blockwise(e0_uw, e0_us)
    gate_up_proj[0, :, intermediate_size:] = up_w0.T
    gate_up_scale[0, :, i_blocks:] = up_s0.T
    down_proj[0] = sample_dw.T
    down_scale[0] = sample_ds.T
    del e0_gw, e0_gs, e0_uw, e0_us, e0_dw, e0_ds
    del sample_w, sample_s, sample_dw, sample_ds, up_w0, up_s0

    for e in range(1, num_experts):
        gw = lazy.get(f"{prefix}mlp.experts.{e}.gate_proj.weight")
        gs = lazy.get(f"{prefix}mlp.experts.{e}.gate_proj.weight_scale_inv")
        uw = lazy.get(f"{prefix}mlp.experts.{e}.up_proj.weight")
        us = lazy.get(f"{prefix}mlp.experts.{e}.up_proj.weight_scale_inv")
        dw = lazy.get(f"{prefix}mlp.experts.{e}.down_proj.weight")
        ds = lazy.get(f"{prefix}mlp.experts.{e}.down_proj.weight_scale_inv")
        g_w, g_s = rescale_fp8_weight_blockwise(gw, gs)
        u_w, u_s = rescale_fp8_weight_blockwise(uw, us)
        d_w, d_s = rescale_fp8_weight_blockwise(dw, ds)
        gate_up_proj[e, :, :intermediate_size] = g_w.T
        gate_up_proj[e, :, intermediate_size:] = u_w.T
        gate_up_scale[e, :, :i_blocks] = g_s.T
        gate_up_scale[e, :, i_blocks:] = u_s.T
        down_proj[e] = d_w.T
        down_scale[e] = d_s.T
        del gw, gs, uw, us, dw, ds, g_w, g_s, u_w, u_s, d_w, d_s

    out[f"{out_prefix}mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
    out[f"{out_prefix}mlp.expert_mlps.mlp_op.gate_up_proj.scale"] = gate_up_scale
    out[f"{out_prefix}mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj
    out[f"{out_prefix}mlp.expert_mlps.mlp_op.down_proj.scale"] = down_scale
    return out


# ---------------------------------------------------------------------------
# Shard saving / index
# ---------------------------------------------------------------------------


def save_shard(
    tensors: Dict[str, torch.Tensor],
    save_path: str,
    filename: str,
    weight_map: Dict[str, str],
) -> int:
    """Save a sub-state-dict; clone tensors so safetensors doesn't complain
    about views of mmapped storage. Returns bytes written."""
    path = os.path.join(save_path, filename)
    materialized: Dict[str, torch.Tensor] = {}
    total_bytes = 0
    for k, v in tensors.items():
        if not v.is_contiguous():
            v = v.contiguous()
        v = v.detach().clone()
        materialized[k] = v
        total_bytes += v.numel() * v.element_size()
    save_file(materialized, path)
    for k in materialized.keys():
        weight_map[k] = filename
    del materialized
    return total_bytes


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def process_pro_checkpoint(hf_model_path: str, save_path: str, tp_degree: int):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(hf_model_path, "model.safetensors.index.json")) as f:
        weight_map_in = json.load(f)["weight_map"]

    with open(os.path.join(hf_model_path, "config.json")) as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    hybrid = config.get("hybrid_layer_pattern", [0] * num_layers)
    moe_freq = config.get("moe_layer_freq", [1] * num_layers)

    print(
        f"Processing {num_layers} decoder layers"
        f" (full={sum(1 for v in hybrid if v == 0)},"
        f" swa={sum(1 for v in hybrid if v == 1)},"
        f" dense={sum(1 for v in moe_freq if v == 0)},"
        f" moe={sum(1 for v in moe_freq if v == 1)})",
        flush=True,
    )

    lazy = LazyWeightMap(hf_model_path, weight_map_in)
    weight_map_out: Dict[str, str] = {}

    try:
        for li in range(num_layers):
            t0 = time.time()
            is_dense = moe_freq[li] == 0
            is_swa = hybrid[li] == 1
            layer_sd = process_layer(li, lazy, config, is_dense=is_dense, is_swa=is_swa)
            filename = f"model_layer{li}.safetensors"
            size = save_shard(layer_sd, save_path, filename, weight_map_out)
            del layer_sd
            gc.collect()
            tag = "dense" if is_dense else "moe  "
            attn = "swa " if is_swa else "full"
            print(
                f"  layer {li:2d} [{tag} {attn}] {size / 1e9:6.2f} GB in {time.time() - t0:5.1f}s",
                flush=True,
            )

        print("Processing embed_tokens, norm, lm_head ...", flush=True)
        extras: Dict[str, torch.Tensor] = {}
        for src, dst in (
            ("model.embed_tokens.weight", "embed_tokens.weight"),
            ("model.norm.weight", "norm.weight"),
            ("lm_head.weight", "lm_head.weight"),
        ):
            t = lazy.get(src)
            if t is not None:
                extras[dst] = t.detach().clone()
            else:
                print(f"  WARNING: missing {src}", flush=True)
        if "lm_head.weight" not in extras and "embed_tokens.weight" in extras:
            # Tied embeddings
            extras["lm_head.weight"] = extras["embed_tokens.weight"].detach().clone()
        save_shard(extras, save_path, "model_extras.safetensors", weight_map_out)
        del extras
    finally:
        lazy.close()

    # --- Index file ---
    total_size = 0
    for f in set(weight_map_out.values()):
        total_size += os.path.getsize(os.path.join(save_path, f))
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # --- Copy auxiliary files (config.json, tokenizer, chat template,
    # and crucially the trust_remote_code modules the HF config references).
    for name in sorted(os.listdir(hf_model_path)):
        if name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        src = os.path.join(hf_model_path, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(save_path, name))

    print(f"\nPreprocess complete. total_size={total_size / 1e9:.2f} GB", flush=True)
    print(f"  tensors written: {len(weight_map_out)}", flush=True)
    print(f"  output dir: {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MiMo-V2-Pro FP8 checkpoint for Neuron inference"
    )
    parser.add_argument("--hf_model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=64,
        help="Tensor parallelism (currently informational only; "
        "the framework does the TP sharding at load time).",
    )
    args = parser.parse_args()
    process_pro_checkpoint(args.hf_model_path, args.save_path, args.tp_degree)


if __name__ == "__main__":
    main()
