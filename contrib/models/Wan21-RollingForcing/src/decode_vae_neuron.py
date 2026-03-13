#!/usr/bin/env python3
"""
Decode latents to video using Wan 2.1 VAE — hybrid CPU/Neuron strategy.

Strategy:
  - conv2 (post_quant_conv): CPU (1x1x1 conv, cheap, runs once)
  - Chunks 0+1: CPU Decoder3d (populates feat_cache, handles "Rep" string)
  - Chunks 2-N: Compiled Neuron decoder (all 32 cache slots are tensors)

At runtime:
  - Neuron decoder compiled for 2-frame input (padded from 1-frame)
  - Each Neuron call: pad 1-frame → 2-frame, decode, take last 4 pixel frames
   - Cache tensors are returned as explicit outputs by NxDModel (v4)
  - Runtime reconstructs correct [old_last, new_current] cache from returned values

Usage:
    # Full decode of latents (CPU chunks 0+1, Neuron chunks 2+)
    python decode_vae_neuron.py --latents /path/to/latents.pt

    # Compare with CPU-only decode (accuracy check)
    python decode_vae_neuron.py --latents /path/to/latents.pt --compare-cpu

    # Benchmark only (no video save)
    python decode_vae_neuron.py --latents /path/to/latents.pt --benchmark
"""

import os
import sys
import json
import time
import argparse

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

import torch
import torch.nn as nn

# Patch torch.cuda before importing wan
import torch.cuda as _tc

try:
    _tc.current_device()
except RuntimeError:
    _tc.current_device = lambda: 0
    _tc.is_available = lambda: False

from neuronx_distributed import NxDModel, NxDParallelState
from safetensors.torch import load_file as safetensors_load

# Add Wan 2.1 VAE source to path
# These module-level constants are used both when running directly (overridden by
# argparse in main()) and when imported as a library (set via environment variables).
# No hardcoded paths — callers must set env vars or pass paths explicitly.
WAN_REPO = os.environ.get("WAN_REPO_PATH", "")
if WAN_REPO:
    sys.path.insert(0, WAN_REPO)

VAE_PATH = os.environ.get("WAN_VAE_PATH", "")
COMPILED_DECODER_DIR = os.environ.get("WAN_COMPILED_DECODER_DIR", "")
OUTPUT_DIR = os.environ.get("WAN_OUTPUT_DIR", "./output")
FPS = 16

NUM_FEAT_CACHE = 32


def load_model_config(path):
    """Load model configuration."""
    with open(os.path.join(path, "config.json"), "r") as f:
        return json.load(f)


def load_duplicated_weights(model_path, world_size):
    """Load weights and duplicate for all TP ranks (decoder not sharded)."""
    weights_path = os.path.join(
        model_path, "weights", "tp0_sharded_checkpoint.safetensors"
    )
    base_weights = safetensors_load(weights_path)
    return [{k: v.clone() for k, v in base_weights.items()} for _ in range(world_size)]


class NeuronDecoderWrapper(nn.Module):
    """
    Runtime wrapper for compiled VAE decoder on Neuron.

    Handles:
    - dtype conversion (float32 ↔ bfloat16)
    - 1-frame padding to 2-frame for compiled model
    - feat_cache management between CPU and Neuron
    """

    def __init__(self, num_feat_cache=NUM_FEAT_CACHE):
        super().__init__()
        self.nxd_model = None
        self.num_feat_cache = num_feat_cache
        self.feat_cache_shapes = None

    def _init_feat_cache_shapes(self, height, width):
        """Initialize feat_cache shapes from config."""
        lat_h = height // 8
        lat_w = width // 8

        self.feat_cache_shapes = [
            (1, 16, 2, lat_h, lat_w),  # [0]
            (1, 384, 2, lat_h, lat_w),  # [1]
            (1, 384, 2, lat_h, lat_w),  # [2]
            (1, 384, 2, lat_h, lat_w),  # [3]
            (1, 384, 2, lat_h, lat_w),  # [4]
            (1, 384, 2, lat_h, lat_w),  # [5]
            (1, 384, 2, lat_h, lat_w),  # [6]
            (1, 384, 2, lat_h, lat_w),  # [7]
            (1, 384, 2, lat_h, lat_w),  # [8]
            (1, 384, 2, lat_h, lat_w),  # [9]
            (1, 384, 2, lat_h, lat_w),  # [10]
            (1, 384, 2, lat_h, lat_w),  # [11]
            (1, 192, 2, lat_h * 2, lat_w * 2),  # [12]
            (1, 384, 2, lat_h * 2, lat_w * 2),  # [13]
            (1, 384, 2, lat_h * 2, lat_w * 2),  # [14]
            (1, 384, 2, lat_h * 2, lat_w * 2),  # [15]
            (1, 384, 2, lat_h * 2, lat_w * 2),  # [16]
            (1, 384, 2, lat_h * 2, lat_w * 2),  # [17]
            (1, 384, 2, lat_h * 2, lat_w * 2),  # [18]
            (1, 192, 2, lat_h * 4, lat_w * 4),  # [19]
            (1, 192, 2, lat_h * 4, lat_w * 4),  # [20]
            (1, 192, 2, lat_h * 4, lat_w * 4),  # [21]
            (1, 192, 2, lat_h * 4, lat_w * 4),  # [22]
            (1, 192, 2, lat_h * 4, lat_w * 4),  # [23]
            (1, 192, 2, lat_h * 4, lat_w * 4),  # [24]
            (1, 96, 2, lat_h * 8, lat_w * 8),  # [25]
            (1, 96, 2, lat_h * 8, lat_w * 8),  # [26]
            (1, 96, 2, lat_h * 8, lat_w * 8),  # [27]
            (1, 96, 2, lat_h * 8, lat_w * 8),  # [28]
            (1, 96, 2, lat_h * 8, lat_w * 8),  # [29]
            (1, 96, 2, lat_h * 8, lat_w * 8),  # [30]
            (1, 96, 2, lat_h * 8, lat_w * 8),  # [31]
        ]

    def forward(self, x, feat_cache):
        """
        Run compiled decoder for one latent frame.

        Args:
            x: [1, 16, 1, H, W] single latent frame (already conv2'd)
            feat_cache: list of 32 tensors (steady-state cache from CPU chunks)

        Returns:
            output: [1, 3, 4, H*8, W*8] decoded pixel frames (after temporal upsample)
        """
        # V5: Pass 1-frame input directly (no padding needed).
        # The compiled model was traced with 1-frame input, matching the CPU path.
        x_bf16 = x.to(torch.bfloat16)

        # Prepare cache tensors in bfloat16
        feat_cache_tensors = []
        for i in range(self.num_feat_cache):
            if i < len(feat_cache) and feat_cache[i] is not None:
                if isinstance(feat_cache[i], str):
                    # Should not happen for chunks 2+, but handle gracefully
                    feat_cache_tensors.append(
                        torch.zeros(self.feat_cache_shapes[i], dtype=torch.bfloat16)
                    )
                else:
                    feat_cache_tensors.append(feat_cache[i].to(torch.bfloat16))
            else:
                feat_cache_tensors.append(
                    torch.zeros(self.feat_cache_shapes[i], dtype=torch.bfloat16)
                )

        # Call NxDModel with all tensor arguments
        # Returns tuple: (output, cache_0, cache_1, ..., cache_31) — 33 tensors
        results = self.nxd_model(
            x_bf16,
            feat_cache_tensors[0],
            feat_cache_tensors[1],
            feat_cache_tensors[2],
            feat_cache_tensors[3],
            feat_cache_tensors[4],
            feat_cache_tensors[5],
            feat_cache_tensors[6],
            feat_cache_tensors[7],
            feat_cache_tensors[8],
            feat_cache_tensors[9],
            feat_cache_tensors[10],
            feat_cache_tensors[11],
            feat_cache_tensors[12],
            feat_cache_tensors[13],
            feat_cache_tensors[14],
            feat_cache_tensors[15],
            feat_cache_tensors[16],
            feat_cache_tensors[17],
            feat_cache_tensors[18],
            feat_cache_tensors[19],
            feat_cache_tensors[20],
            feat_cache_tensors[21],
            feat_cache_tensors[22],
            feat_cache_tensors[23],
            feat_cache_tensors[24],
            feat_cache_tensors[25],
            feat_cache_tensors[26],
            feat_cache_tensors[27],
            feat_cache_tensors[28],
            feat_cache_tensors[29],
            feat_cache_tensors[30],
            feat_cache_tensors[31],
        )

        # Unpack: first element is output, rest are updated caches
        if isinstance(results, (tuple, list)):
            output = results[0]
            returned_caches = results[1:]
        else:
            output = results
            returned_caches = []

        # Convert output to float32
        output = output.to(torch.float32)

        # V5: With 1-frame input, the NEFF's internal cache update logic is correct:
        #   cache_x = x[:,:,-2:] → 1 frame → shape[2] < 2 → True
        #   cache_x = cat([old_cache_last, current_frame]) → correct [old, new]
        # So the returned caches are the actual updated values — use directly.
        for i in range(min(len(feat_cache), len(returned_caches))):
            feat_cache[i] = returned_caches[i]

        # Output already has the correct number of frames:
        # - Before temporal upsamples: 1 frame
        # - After 2x temporal upsamples: 2, then 4 frames
        # Final output shape: [1, 3, 4, H*8, W*8]
        output = output[:, :, -4:, :, :]

        return output


def decode_hybrid(latents, vae_model, neuron_decoder, height, width):
    """
    Decode latents using hybrid CPU/Neuron strategy.

    Args:
        latents: [1, 16, T_lat, H_lat, W_lat] latent tensor
        vae_model: WanVAE_ model on CPU (for conv2 + chunks 0+1)
        neuron_decoder: NeuronDecoderWrapper (for chunks 2+)
        height: pixel height (480)
        width: pixel width (832)

    Returns:
        output: [1, 3, T_pixel, H, W] decoded video tensor
    """
    z_dim = 16
    mean = torch.tensor(
        [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        dtype=torch.float32,
    )
    std = torch.tensor(
        [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        dtype=torch.float32,
    )
    scale = [mean, 1.0 / std]

    z = latents.float()
    num_latent_frames = z.shape[2]

    # Step 1: Unscale latents (reverse normalization)
    z = z / scale[1].view(1, z_dim, 1, 1, 1) + scale[0].view(1, z_dim, 1, 1, 1)

    # Step 2: conv2 (post_quant_conv) on all frames at once — CPU
    print(f"  conv2 on CPU ({num_latent_frames} frames)...")
    t0 = time.time()
    with torch.no_grad():
        x = vae_model.conv2(z)
    print(f"    conv2 took {time.time() - t0:.1f}s")

    # Step 3: Initialize feat_cache and run chunks 0+1 on CPU
    vae_model.clear_cache()
    feat_map = vae_model._feat_map  # [None] * 33

    # Chunk 0 (first frame, produces 1 pixel frame, writes "Rep" to cache)
    print(f"  Chunk 0 on CPU (first frame)...")
    t0 = time.time()
    vae_model._conv_idx = [0]
    with torch.no_grad():
        out = vae_model.decoder(
            x[:, :, 0:1, :, :], feat_cache=feat_map, feat_idx=vae_model._conv_idx
        )
    print(f"    Chunk 0 took {time.time() - t0:.1f}s, output: {out.shape}")

    # Chunk 1 (second frame, clears "Rep", produces 4 pixel frames)
    print(f"  Chunk 1 on CPU (second frame)...")
    t0 = time.time()
    vae_model._conv_idx = [0]
    with torch.no_grad():
        out_ = vae_model.decoder(
            x[:, :, 1:2, :, :], feat_cache=feat_map, feat_idx=vae_model._conv_idx
        )
    out = torch.cat([out, out_], 2)
    print(f"    Chunk 1 took {time.time() - t0:.1f}s, output: {out_.shape}")

    # Verify all cache slots are now tensors
    n_tensors = sum(1 for fc in feat_map if isinstance(fc, torch.Tensor))
    n_strings = sum(1 for fc in feat_map if isinstance(fc, str))
    n_nones = sum(1 for fc in feat_map if fc is None)
    print(f"    Cache state: {n_tensors} tensors, {n_strings} strings, {n_nones} None")
    if n_strings > 0:
        print("    WARNING: Still have 'Rep' strings in cache!")

    # Step 4: Run chunks 2-N on Neuron
    if num_latent_frames > 2:
        print(
            f"  Chunks 2-{num_latent_frames - 1} on Neuron ({num_latent_frames - 2} frames)..."
        )
        t0 = time.time()

        for i in range(2, num_latent_frames):
            t_chunk = time.time()
            with torch.no_grad():
                out_ = neuron_decoder(x[:, :, i : i + 1, :, :], feat_map)
            out = torch.cat([out, out_], 2)
            dt = time.time() - t_chunk
            if i == 2 or i == num_latent_frames - 1:
                print(f"    Chunk {i}: {dt:.2f}s, output: {out_.shape}")

        neuron_time = time.time() - t0
        print(
            f"    Neuron chunks took {neuron_time:.1f}s total "
            f"({neuron_time / (num_latent_frames - 2):.2f}s/chunk)"
        )

    return out


def decode_cpu_reference(latents, vae_model):
    """Full CPU decode for accuracy comparison."""
    z_dim = 16
    mean = torch.tensor(
        [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        dtype=torch.float32,
    )
    std = torch.tensor(
        [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        dtype=torch.float32,
    )
    scale = [mean, 1.0 / std]

    with torch.no_grad():
        video = vae_model.decode(latents.float(), scale)
    return video


def main():
    parser = argparse.ArgumentParser(
        description="Decode latents with Wan 2.1 VAE (hybrid CPU/Neuron)"
    )
    parser.add_argument(
        "--latents", type=str, required=True, help="Path to latent tensor (.pt)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output MP4 path")
    parser.add_argument(
        "--wan-repo",
        type=str,
        default=WAN_REPO or None,
        help="Path to Wan/RollingForcing repo (env: WAN_REPO_PATH)",
    )
    parser.add_argument(
        "--vae-weights",
        type=str,
        default=VAE_PATH or None,
        required=not VAE_PATH,
        help="VAE weights path (env: WAN_VAE_PATH)",
    )
    parser.add_argument(
        "--compiled-decoder",
        type=str,
        default=COMPILED_DECODER_DIR or None,
        required=not COMPILED_DECODER_DIR,
        help="Compiled decoder directory (env: WAN_COMPILED_DECODER_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for videos (env: WAN_OUTPUT_DIR, default: ./output)",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Output video FPS")
    parser.add_argument(
        "--compare-cpu", action="store_true", help="Compare with CPU decode"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark only (no video save)"
    )
    args = parser.parse_args()

    # If --wan-repo provided on CLI, ensure it's on sys.path
    if args.wan_repo and args.wan_repo not in sys.path:
        sys.path.insert(0, args.wan_repo)

    # Load latents
    print(f"\nLoading latents from {args.latents}...")
    latents = torch.load(args.latents, map_location="cpu")
    print(f"  Shape: {latents.shape}, dtype: {latents.dtype}")
    B, C, T_lat, H_lat, W_lat = latents.shape
    height = H_lat * 8
    width = W_lat * 8
    print(f"  B={B}, C={C}, T_lat={T_lat}, H_lat={H_lat}, W_lat={W_lat}")
    print(
        f"  Pixel resolution: {height}x{width}, expected frames: {(T_lat - 1) * 4 + 1}"
    )

    # Load VAE model (CPU, for conv2 + chunks 0+1)
    print(f"\nLoading Wan 2.1 VAE...")
    from wan.modules.vae import WanVAE_

    vae_model = WanVAE_(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    state_dict = torch.load(args.vae_weights, map_location="cpu")
    vae_model.load_state_dict(state_dict, strict=True)
    vae_model.eval()
    print(f"  VAE loaded")

    # Load compiled decoder config
    config = load_model_config(args.compiled_decoder)
    world_size = config["world_size"]
    tp_degree = config["tp_degree"]
    print(f"\nLoading compiled decoder from {args.compiled_decoder}...")
    print(
        f"  Config: world_size={world_size}, tp_degree={tp_degree}, dtype={config['dtype']}"
    )

    # Load and initialize NxDModel
    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        nxd_model = NxDModel.load(os.path.join(args.compiled_decoder, "nxd_model.pt"))
        weights = load_duplicated_weights(args.compiled_decoder, world_size)
        nxd_model.set_weights(weights)
        nxd_model.to_neuron()

        # Create runtime wrapper
        neuron_decoder = NeuronDecoderWrapper(num_feat_cache=NUM_FEAT_CACHE)
        neuron_decoder._init_feat_cache_shapes(height, width)
        neuron_decoder.nxd_model = nxd_model

        # ========== Hybrid decode ==========
        print(f"\n{'=' * 60}")
        print(f"Hybrid CPU/Neuron VAE Decode")
        print(f"{'=' * 60}")
        t_total = time.time()

        video_hybrid = decode_hybrid(
            latents[0:1], vae_model, neuron_decoder, height, width
        )
        video_hybrid = video_hybrid.float().clamp(-1, 1)

        total_time = time.time() - t_total
        print(f"\nTotal hybrid decode: {total_time:.1f}s")
        print(f"  Output shape: {video_hybrid.shape}")

        # ========== CPU reference (optional) ==========
        if args.compare_cpu:
            print(f"\n{'=' * 60}")
            print(f"CPU-only VAE Decode (reference)")
            print(f"{'=' * 60}")
            t_cpu = time.time()
            video_cpu = decode_cpu_reference(latents[0:1], vae_model)
            video_cpu = video_cpu.float().clamp(-1, 1)
            cpu_time = time.time() - t_cpu
            print(f"  CPU decode: {cpu_time:.1f}s, shape: {video_cpu.shape}")

            # Compare accuracy
            if video_hybrid.shape == video_cpu.shape:
                diff = (video_hybrid - video_cpu).abs()
                cos_sim = torch.nn.functional.cosine_similarity(
                    video_hybrid.flatten(), video_cpu.flatten(), dim=0
                ).item()
                print(f"\n  Accuracy comparison:")
                print(f"    Cosine similarity: {cos_sim:.6f}")
                print(f"    Max abs diff: {diff.max().item():.6f}")
                print(f"    Mean abs diff: {diff.mean().item():.6f}")
                print(
                    f"    PSNR: {-10 * torch.log10(diff.pow(2).mean()).item():.2f} dB"
                )
            else:
                print(
                    f"  Shape mismatch! Hybrid: {video_hybrid.shape}, CPU: {video_cpu.shape}"
                )

            print(f"\n  Speedup: {cpu_time / total_time:.1f}x")

    # Save video
    if not args.benchmark:
        video_out = (video_hybrid * 0.5 + 0.5).clamp(0, 1)

        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(os.path.basename(args.latents))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_neuron_vae.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        import imageio

        # video_out is [1, 3, T, H, W]
        frames_np = (255.0 * video_out[0]).clamp(0, 255).to(torch.uint8)
        frames_np = frames_np.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]
        imageio.mimwrite(output_path, frames_np, fps=args.fps)
        print(
            f"\nSaved video: {output_path} ({frames_np.shape[0]} frames, {args.fps} fps)"
        )

    print(f"\nDone! Total decode time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
