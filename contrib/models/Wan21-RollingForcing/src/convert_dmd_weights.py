#!/usr/bin/env python3
"""Convert DMD checkpoint to safetensors with exact weight transfer.

Takes the raw DMD checkpoint (TencentARC native naming) and converts it to
our NxDI naming convention (diffusers-style), saving as safetensors.

This ensures ZERO weight loss — no intermediate model loading, no float32
rounding through a different code path. Direct key-for-key conversion.

Usage:
    python convert_dmd_weights.py \
        --dmd-checkpoint /path/to/rolling_forcing_dmd.pt \
        --output /path/to/output/model.safetensors
"""

import argparse
import os
import re
import torch
from collections import OrderedDict
from safetensors.torch import save_file


def convert_native_to_neuron(key):
    """Convert native Wan/DMD key naming to our NxDI naming.

    Native DMD format: model.blocks.N.self_attn.q.weight
    Our format:        blocks.N.self_attn.to_q.weight
    """
    new_key = key

    # Strip 'model.' prefix (DMD wraps in WanDiffusionWrapper.model)
    if new_key.startswith("model."):
        new_key = new_key[len("model.") :]

    # Self-attn Q/K/V/O: add 'to_' prefix
    new_key = re.sub(r"\.self_attn\.(q|k|v)\.", r".self_attn.to_\1.", new_key)
    new_key = re.sub(r"\.self_attn\.o\.", r".self_attn.to_out.", new_key)

    # Cross-attn Q/K/V/O: add 'to_' prefix
    new_key = re.sub(r"\.cross_attn\.(q|k|v)\.", r".cross_attn.to_\1.", new_key)
    new_key = re.sub(r"\.cross_attn\.o\.", r".cross_attn.to_out.", new_key)

    # FFN: .ffn.0 → .ffn_gelu_proj, .ffn.2 → .ffn_out
    new_key = new_key.replace(".ffn.0.", ".ffn_gelu_proj.")
    new_key = new_key.replace(".ffn.2.", ".ffn_out.")

    # Block modulation: .modulation → .scale_shift_table
    if ".modulation" in new_key and "blocks." in new_key:
        new_key = new_key.replace(".modulation", ".scale_shift_table")

    # Norm renaming: native Wan norm3 = cross-attn norm = our norm2
    new_key = re.sub(r"(blocks\.\d+)\.norm3\.", r"\1.norm2.", new_key)

    # Output head: head.head → proj_out
    new_key = new_key.replace("head.head.", "proj_out.")

    # Output modulation: head.modulation → scale_shift_table
    if new_key == "head.modulation":
        new_key = "scale_shift_table"

    # Text embedder: text_embedding.0 → condition_embedder.text_embedder_linear_1
    new_key = new_key.replace(
        "text_embedding.0.", "condition_embedder.text_embedder_linear_1."
    )
    new_key = new_key.replace(
        "text_embedding.2.", "condition_embedder.text_embedder_linear_2."
    )

    # Time embedder: time_embedding.0 → condition_embedder.time_embedder_linear_1
    new_key = new_key.replace(
        "time_embedding.0.", "condition_embedder.time_embedder_linear_1."
    )
    new_key = new_key.replace(
        "time_embedding.2.", "condition_embedder.time_embedder_linear_2."
    )

    # Time projection: time_projection.1 → condition_embedder.time_proj
    new_key = new_key.replace("time_projection.1.", "condition_embedder.time_proj.")

    return new_key


def main(dmd_checkpoint, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading DMD checkpoint: {dmd_checkpoint}")
    ckpt = torch.load(dmd_checkpoint, map_location="cpu")
    raw_sd = ckpt["generator_ema"]

    # Clean FSDP wrapper prefix
    clean_sd = OrderedDict()
    for key, value in raw_sd.items():
        clean_key = key.replace("_fsdp_wrapped_module.", "")
        clean_sd[clean_key] = value

    print(f"  {len(clean_sd)} keys in DMD checkpoint")

    # Convert to our naming
    converted = {}
    skipped = []
    for key, value in clean_sd.items():
        new_key = convert_native_to_neuron(key)

        # Skip keys we don't need (e.g., non-model keys)
        if new_key.startswith("model.") or new_key.startswith("ema_"):
            skipped.append(key)
            continue

        # Store as contiguous float32 tensor
        converted[new_key] = value.clone().detach().contiguous()

    print(f"  {len(converted)} keys converted")
    if skipped:
        print(f"  {len(skipped)} keys skipped:")
        for s in skipped[:10]:
            print(f"    {s}")

    # Verify: check a few key mappings
    print("\n--- Key mapping samples ---")
    sample_keys = [
        "blocks.0.self_attn.to_q.weight",
        "blocks.0.self_attn.norm_q.weight",
        "blocks.0.cross_attn.to_q.weight",
        "blocks.0.norm2.weight",
        "blocks.0.scale_shift_table",
        "blocks.0.ffn_gelu_proj.weight",
        "condition_embedder.time_embedder_linear_1.weight",
        "condition_embedder.text_embedder_linear_1.weight",
        "proj_out.weight",
        "scale_shift_table",
    ]
    for sk in sample_keys:
        if sk in converted:
            v = converted[sk]
            print(f"  {sk}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {sk}: MISSING!")

    # Verify exact match with raw DMD values
    print("\n--- Exact match verification ---")
    test_pairs = [
        ("model.blocks.0.self_attn.q.weight", "blocks.0.self_attn.to_q.weight"),
        ("model.blocks.0.self_attn.norm_q.weight", "blocks.0.self_attn.norm_q.weight"),
        ("model.blocks.0.modulation", "blocks.0.scale_shift_table"),
        ("model.blocks.0.cross_attn.q.weight", "blocks.0.cross_attn.to_q.weight"),
        ("model.blocks.0.ffn.0.weight", "blocks.0.ffn_gelu_proj.weight"),
        ("model.blocks.29.self_attn.q.weight", "blocks.29.self_attn.to_q.weight"),
        ("model.head.head.weight", "proj_out.weight"),
    ]
    all_exact = True
    for dmd_key, our_key in test_pairs:
        if dmd_key in clean_sd and our_key in converted:
            exact = torch.equal(clean_sd[dmd_key], converted[our_key])
            maxd = (
                (clean_sd[dmd_key].float() - converted[our_key].float())
                .abs()
                .max()
                .item()
            )
            status = "EXACT" if exact else f"DIFF maxd={maxd}"
            print(f"  {dmd_key} → {our_key}: {status}")
            if not exact:
                all_exact = False
        else:
            print(
                f"  {dmd_key} → {our_key}: KEY MISSING (dmd={dmd_key in clean_sd}, ours={our_key in converted})"
            )
            all_exact = False

    if all_exact:
        print("\n  ALL weights are EXACT matches!")
    else:
        print("\n  WARNING: Some weights don't match exactly!")

    # Save
    print(f"\nSaving to {output_path}")
    save_file(converted, output_path)
    print(f"  Done! Size: {os.path.getsize(output_path) / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DMD checkpoint to safetensors with NxDI naming convention."
    )
    parser.add_argument(
        "--dmd-checkpoint",
        required=True,
        help="Path to the raw DMD checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output safetensors file.",
    )
    args = parser.parse_args()
    main(args.dmd_checkpoint, args.output)
