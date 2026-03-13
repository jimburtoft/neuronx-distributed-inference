"""
Wan 2.1 VAE Decoder Compilation for Neuron — ModelBuilder V3 approach.

Architecture: Wan 2.1 VAE (dim=96, z_dim=16, 3D causal, ~73M params)
Strategy:
  - conv2 (post_quant_conv, CausalConv3d 16→16, kernel=1): Runs on CPU (cheap, one call)
  - Decoder chunks 0+1 (first 2 latent frames): Run on CPU
    * Chunk 0 populates feat_cache, writes "Rep" strings to 2 Resample slots
    * Chunk 1 reads "Rep", branches, writes real tensors — all 32 slots become tensors
  - Decoder chunks 2-N (steady-state): Compiled for Neuron (all 32 cache slots are tensors)

The "Rep" string issue: On chunk 0, two Resample.upsample3d time_conv slots (11, 18) get
"Rep" strings instead of tensors. On chunk 1, the code branches on "Rep" vs tensor.
Since Neuron can't handle dynamic string-vs-tensor branching, we run chunks 0+1 on CPU
and compile the steady-state path (chunks 2+) where all 32 cache slots are real tensors.

For 480x832 @ 81 frames (21 latent frames):
  - conv2 on CPU: cheap (1x1x1 conv on 21 frames)
  - Chunks 0+1 on CPU: ~50s (produces 5 pixel frames: 1+4)
  - Chunks 2-20 on Neuron: ~19-38s (19 frames × 4 = 76 pixel frames)
  - Total: ~50-88s (vs 483s all-CPU baseline, ~5-10x speedup)

Adapted from Henan's compile_decoder_v3.py (Wan 2.2, 34 feat_cache slots)
to Wan 2.1 (32 feat_cache slots, smaller channels: dim=96 vs 256).

Usage:
    python compile_vae_decoder.py [--height 480] [--width 832] [--compiled_models_dir ...]
"""

import os
import sys
import json
import argparse

# Environment setup for trn2 with LNC=2 (4 logical cores, TP=4)
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

# Compiler flags for trn2
compiler_flags = " --target=trn2 --lnc=2 --enable-fast-loading-neuron-binaries "
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn

# Patch torch.cuda before importing wan (T5 module calls torch.cuda.current_device())
import torch.cuda as _tc

_orig_current_device = _tc.current_device
_orig_is_available = _tc.is_available
try:
    _tc.current_device()
except RuntimeError:
    _tc.current_device = lambda: 0
    _tc.is_available = lambda: False

from neuronx_distributed import ModelBuilder, NxDParallelState
from safetensors.torch import save_file

# Wan 2.1 VAE source path — resolved after argparse (see __main__ block).
# Set via --wan-repo argument or WAN_REPO_PATH environment variable.
WAN_REPO = None  # Will be set in main()


# ============================================================================
# Wan 2.1 VAE dimensions (from vae.py: _video_vae with dim=96, z_dim=16)
# ============================================================================
# Decoder3d(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
#           temperal_upsample=[False, True, True])
#
# Channel progression: [384, 384, 192, 96] (dims = dim * [4, 4, 2, 1])
# Spatial upsamples: 3 stages of 2x (total 8x)
# Temporal upsamples: [False, True, True] → 2 stages of 2x (total 4x)
#
# feat_cache has 32 used slots (33 CausalConv3d in decoder, but one Resample
# writes "Rep" string on chunk 0 instead of using cache, so 32 real slots).
# On chunk 1+, all 32 slots contain tensors.
# ============================================================================

NUM_FEAT_CACHE = 32  # Number of feat_cache tensor slots for steady-state


class SteadyStateDecoderWrapper(nn.Module):
    """
    Wrapper for Wan 2.1 VAE Decoder3d that takes all feat_cache entries as
    individual tensor arguments (ModelBuilder requires tensor-only inputs).

    This is for chunks 2+ (steady-state) where all 32 cache slots are tensors.
    The decoder internally mutates feat_cache list entries via assignment
    (feat_cache[idx] = new_tensor). However, NxDModel does NOT propagate
    these mutations back to the host — input tensors are unchanged after the
    NEFF call. Therefore, we explicitly return all 32 updated cache tensors
    as additional outputs alongside the decoded pixel frames.

    At trace time, all feat_cache[idx] comparisons with 'Rep' will resolve to
    False (they're tensors), so the compiled graph always uses the cache path.

    Returns:
        tuple of (output, cache_0, cache_1, ..., cache_31) — 33 tensors total.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        x,
        feat_cache_0,
        feat_cache_1,
        feat_cache_2,
        feat_cache_3,
        feat_cache_4,
        feat_cache_5,
        feat_cache_6,
        feat_cache_7,
        feat_cache_8,
        feat_cache_9,
        feat_cache_10,
        feat_cache_11,
        feat_cache_12,
        feat_cache_13,
        feat_cache_14,
        feat_cache_15,
        feat_cache_16,
        feat_cache_17,
        feat_cache_18,
        feat_cache_19,
        feat_cache_20,
        feat_cache_21,
        feat_cache_22,
        feat_cache_23,
        feat_cache_24,
        feat_cache_25,
        feat_cache_26,
        feat_cache_27,
        feat_cache_28,
        feat_cache_29,
        feat_cache_30,
        feat_cache_31,
    ):
        feat_cache = [
            feat_cache_0,
            feat_cache_1,
            feat_cache_2,
            feat_cache_3,
            feat_cache_4,
            feat_cache_5,
            feat_cache_6,
            feat_cache_7,
            feat_cache_8,
            feat_cache_9,
            feat_cache_10,
            feat_cache_11,
            feat_cache_12,
            feat_cache_13,
            feat_cache_14,
            feat_cache_15,
            feat_cache_16,
            feat_cache_17,
            feat_cache_18,
            feat_cache_19,
            feat_cache_20,
            feat_cache_21,
            feat_cache_22,
            feat_cache_23,
            feat_cache_24,
            feat_cache_25,
            feat_cache_26,
            feat_cache_27,
            feat_cache_28,
            feat_cache_29,
            feat_cache_30,
            feat_cache_31,
        ]

        # The decoder mutates feat_cache list entries (feat_cache[idx] = new_tensor).
        # Must pass fresh feat_idx=[0] each call (mutable default footgun).
        output = self.decoder(x, feat_cache=feat_cache, feat_idx=[0])

        # Explicitly return updated cache tensors so the HLO graph includes them
        # as outputs. Without this, cache mutations are lost after NEFF execution.
        return (
            output,
            feat_cache[0],
            feat_cache[1],
            feat_cache[2],
            feat_cache[3],
            feat_cache[4],
            feat_cache[5],
            feat_cache[6],
            feat_cache[7],
            feat_cache[8],
            feat_cache[9],
            feat_cache[10],
            feat_cache[11],
            feat_cache[12],
            feat_cache[13],
            feat_cache[14],
            feat_cache[15],
            feat_cache[16],
            feat_cache[17],
            feat_cache[18],
            feat_cache[19],
            feat_cache[20],
            feat_cache[21],
            feat_cache[22],
            feat_cache[23],
            feat_cache[24],
            feat_cache[25],
            feat_cache[26],
            feat_cache[27],
            feat_cache[28],
            feat_cache[29],
            feat_cache[30],
            feat_cache[31],
        )


def get_steady_state_cache_shapes(height, width, dtype=torch.bfloat16):
    """
    Return the feat_cache tensor shapes for steady-state (chunks 2+).

    Empirically verified by running probe_vae_cache.py on the instance at
    480x832 resolution. After chunks 0+1 on CPU, all 32 used slots contain
    tensors with temporal dim=2 (CACHE_T=2).

    Slot 32 (the 33rd CausalConv3d) is never used (stays None).

    Architecture layout:
    - [0]: conv_in input cache (z_dim=16, latent resolution)
    - [1-10]: mid_block (2 ResBlocks × 2 CausalConv3d + shortcut if needed)
              + up_block_0 residual blocks (384 channels, latent resolution)
    - [11]: Resample upsample3d time_conv cache (384 channels, latent resolution)
    - [12-17]: up_block_1 residual blocks (192→384 channels, 2x spatial)
    - [18]: Resample upsample3d time_conv cache (384 channels, 2x spatial)
    - [19-24]: up_block_2 residual blocks (192 channels, 4x spatial)
    - [25-31]: up_block_3 residual blocks + head conv (96 channels, 8x = full resolution)
    """
    lat_h = height // 8
    lat_w = width // 8

    # Empirically verified shapes from probe_vae_cache.py
    shapes = [
        (1, 16, 2, lat_h, lat_w),  # [0]  conv_in
        (1, 384, 2, lat_h, lat_w),  # [1]  mid/up_block_0
        (1, 384, 2, lat_h, lat_w),  # [2]
        (1, 384, 2, lat_h, lat_w),  # [3]
        (1, 384, 2, lat_h, lat_w),  # [4]
        (1, 384, 2, lat_h, lat_w),  # [5]
        (1, 384, 2, lat_h, lat_w),  # [6]
        (1, 384, 2, lat_h, lat_w),  # [7]
        (1, 384, 2, lat_h, lat_w),  # [8]
        (1, 384, 2, lat_h, lat_w),  # [9]
        (1, 384, 2, lat_h, lat_w),  # [10]
        (1, 384, 2, lat_h, lat_w),  # [11] Resample upsample3d time_conv
        (1, 192, 2, lat_h * 2, lat_w * 2),  # [12] up_block_1
        (1, 384, 2, lat_h * 2, lat_w * 2),  # [13]
        (1, 384, 2, lat_h * 2, lat_w * 2),  # [14]
        (1, 384, 2, lat_h * 2, lat_w * 2),  # [15]
        (1, 384, 2, lat_h * 2, lat_w * 2),  # [16]
        (1, 384, 2, lat_h * 2, lat_w * 2),  # [17]
        (1, 384, 2, lat_h * 2, lat_w * 2),  # [18] Resample upsample3d time_conv
        (1, 192, 2, lat_h * 4, lat_w * 4),  # [19] up_block_2
        (1, 192, 2, lat_h * 4, lat_w * 4),  # [20]
        (1, 192, 2, lat_h * 4, lat_w * 4),  # [21]
        (1, 192, 2, lat_h * 4, lat_w * 4),  # [22]
        (1, 192, 2, lat_h * 4, lat_w * 4),  # [23]
        (1, 192, 2, lat_h * 4, lat_w * 4),  # [24]
        (1, 96, 2, lat_h * 8, lat_w * 8),  # [25] up_block_3 + head
        (1, 96, 2, lat_h * 8, lat_w * 8),  # [26]
        (1, 96, 2, lat_h * 8, lat_w * 8),  # [27]
        (1, 96, 2, lat_h * 8, lat_w * 8),  # [28]
        (1, 96, 2, lat_h * 8, lat_w * 8),  # [29]
        (1, 96, 2, lat_h * 8, lat_w * 8),  # [30]
        (1, 96, 2, lat_h * 8, lat_w * 8),  # [31]
    ]

    assert len(shapes) == NUM_FEAT_CACHE, (
        f"Expected {NUM_FEAT_CACHE} shapes, got {len(shapes)}"
    )
    return shapes


def save_model_config(output_path, config):
    """Save model configuration."""
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def compile_decoder(args):
    """Compile Wan 2.1 VAE decoder (steady-state path) using ModelBuilder V3."""

    # Wan 2.1 latent space: spatial 8x compression (not 16x like Wan 2.2)
    lat_h = args.height // 8
    lat_w = args.width // 8
    num_latent_frames = (args.num_frames - 1) // 4 + 1  # 81 frames → 21 latent frames

    batch_size = 1
    z_dim = 16  # Wan 2.1 z_dim
    dtype = torch.bfloat16
    world_size = args.world_size
    tp_degree = args.tp_degree

    print("=" * 70)
    print("Wan 2.1 VAE Decoder Compilation V5 (1-frame input, ModelBuilder V3)")
    print("=" * 70)
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Latent resolution: {lat_h}x{lat_w}")
    print(f"Frames: {args.num_frames} pixel → {num_latent_frames} latent")
    print(f"World size: {world_size}, TP: {tp_degree}")
    print(f"Decoder dtype: {dtype}")
    print(f"Feat cache slots: {NUM_FEAT_CACHE}")
    print(f"Input frames: 1 (matches CPU path exactly — no padding)")
    print(f"Strategy: chunks 0+1 on CPU, chunks 2-{num_latent_frames - 1} on Neuron")
    print(f"Compiler args: --model-type=unet-inference -O1 --auto-cast=matmult")
    print("=" * 70)

    # Load VAE
    print("\nLoading Wan 2.1 VAE...")
    from wan.modules.vae import WanVAE_

    vae_path = args.vae_path
    model = WanVAE_(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    # Load weights
    state_dict = torch.load(vae_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        # ========== Compile Steady-State Decoder (bfloat16) ==========
        print("\nPreparing decoder (bfloat16, steady-state)...")
        decoder = model.decoder
        decoder = decoder.to(dtype)
        decoder.eval()

        # Decoder input: single latent frame after conv2.
        # V5: Use 1-frame input (matches CPU decode path exactly).
        # Each CausalConv3d uses feat_cache for temporal context.
        # With 1-frame input, the cache update logic correctly does:
        #   cache_x = x[:,:,-2:] → shape [1,C,1,H,W]
        #   cache_x.shape[2] < 2 → True → concatenates old cache last frame
        #   cache_x = cat([old_last, current]) → [1,C,2,H,W]
        # This is the correct cache propagation.
        # Shape: [1, z_dim=16, 1, lat_h, lat_w]
        decoder_frames = 1  # Single frame (no padding)
        decoder_input = torch.rand(
            (batch_size, z_dim, decoder_frames, lat_h, lat_w), dtype=dtype
        )

        # Create feat_cache tensors (steady-state shapes)
        cache_shapes = get_steady_state_cache_shapes(args.height, args.width, dtype)
        feat_cache_tensors = [torch.rand(s, dtype=dtype) for s in cache_shapes]

        # Wrap decoder
        decoder_wrapper = SteadyStateDecoderWrapper(decoder)

        # Build trace kwargs (all tensor arguments for ModelBuilder)
        trace_kwargs = {"x": decoder_input}
        for i, fc in enumerate(feat_cache_tensors):
            trace_kwargs[f"feat_cache_{i}"] = fc

        # Initialize ModelBuilder
        print("\nInitializing ModelBuilder for decoder...")
        decoder_builder = ModelBuilder(model=decoder_wrapper)

        print("Tracing decoder (this may take a few minutes)...")
        decoder_builder.trace(
            kwargs=trace_kwargs,
            tag="decode",
        )

        # Compile with unet-inference model type (optimized for Conv3D, not attention)
        print("Compiling decoder (this may take 15-30 minutes)...")
        compile_args = "--model-type=unet-inference -O1 --auto-cast=matmult"
        traced_decoder = decoder_builder.compile(
            compiler_args=compile_args,
            compiler_workdir=args.compiler_workdir,
        )

        # Save compiled decoder
        decoder_output_path = os.path.join(args.compiled_models_dir, "vae_decoder_v5")
        os.makedirs(decoder_output_path, exist_ok=True)
        print(f"\nSaving compiled decoder to {decoder_output_path}...")
        traced_decoder.save(os.path.join(decoder_output_path, "nxd_model.pt"))

        # Save weights (duplicated across ranks, not sharded)
        print("Saving decoder weights...")
        weights_path = os.path.join(decoder_output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)
        decoder_checkpoint = decoder_wrapper.state_dict()
        save_file(
            decoder_checkpoint,
            os.path.join(weights_path, "tp0_sharded_checkpoint.safetensors"),
        )

        # Save config
        config = {
            "model": "wan2.1_vae_decoder_steady_state_v5",
            "batch_size": batch_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_latent_frames": num_latent_frames,
            "decoder_frames": decoder_frames,
            "z_dim": z_dim,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "num_feat_cache": NUM_FEAT_CACHE,
            "tp_degree": tp_degree,
            "world_size": world_size,
            "dtype": "bfloat16",
            "strategy": "chunk0_cpu_chunkN_neuron",
            "compile_args": compile_args,
            "cache_shapes": [list(s) for s in cache_shapes],
        }
        save_model_config(decoder_output_path, config)

    print("\n" + "=" * 70)
    print("Compilation Complete!")
    print("=" * 70)
    print(f"Decoder saved to: {decoder_output_path}")
    print(f"\nStrategy:")
    print(
        f"  - conv2 (post_quant_conv): CPU (1x1x1 conv, runs once on all {num_latent_frames} frames)"
    )
    print(
        f"  - Chunks 0+1 (first 2 frames): CPU (populates feat_cache, handles 'Rep' string)"
    )
    print(
        f"  - Chunks 2-{num_latent_frames - 1} ({num_latent_frames - 2} frames): Neuron (compiled)"
    )
    print(f"\nKey settings:")
    print(f"  - compiler_args: --model-type=unet-inference (Conv3D optimized)")
    print(f"  - Decoder dtype: bfloat16")
    print(f"  - Feat cache: {NUM_FEAT_CACHE} tensor slots")
    print(f"  - Weights: duplicated across {tp_degree} ranks (not sharded)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile Wan 2.1 VAE Decoder for Neuron (steady-state path)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Pixel height (default: 480)"
    )
    parser.add_argument(
        "--width", type=int, default=832, help="Pixel width (default: 832)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of pixel frames (default: 81)",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to Wan 2.1 VAE weights (.pth file)",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=4,
        help="Tensor parallelism degree (default: 4, matches DiT)",
    )
    parser.add_argument(
        "--world_size", type=int, default=4, help="World size (default: 4, matches DiT)"
    )
    parser.add_argument(
        "--compiled_models_dir",
        type=str,
        required=True,
        help="Output directory for compiled model",
    )
    parser.add_argument(
        "--compiler_workdir",
        type=str,
        default="./compiler_workdir_vae",
        help="Compiler work directory (default: ./compiler_workdir_vae)",
    )
    parser.add_argument(
        "--wan_repo",
        type=str,
        default=None,
        help="Path to Wan 2.1 repo (TencentARC_RollingForcing). "
        "Falls back to WAN_REPO_PATH env var if not set.",
    )
    args = parser.parse_args()

    # Resolve WAN_REPO: --wan_repo arg > WAN_REPO_PATH env var > error
    global WAN_REPO
    wan_repo = args.wan_repo or os.environ.get("WAN_REPO_PATH")
    if not wan_repo:
        parser.error(
            "Wan 2.1 repo path is required. Set --wan_repo or WAN_REPO_PATH env var."
        )
    WAN_REPO = wan_repo
    sys.path.insert(0, WAN_REPO)

    compile_decoder(args)
