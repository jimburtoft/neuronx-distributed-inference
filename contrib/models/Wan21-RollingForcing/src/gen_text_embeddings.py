#!/usr/bin/env python3
"""
Generate real T5 text encoder embeddings and save them for use by the Neuron pipeline.

Usage:
    python gen_text_embeddings.py --wan-repo /path/to/TencentARC_RollingForcing --output-dir ./text_embeddings

Arguments can also be set via environment variables:
    WAN_REPO_PATH   Path to the TencentARC RollingForcing repository (required if --wan-repo not given)
"""

import argparse
import sys
import os
import time
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate T5 text encoder embeddings for the Neuron pipeline."
    )
    parser.add_argument(
        "--wan-repo",
        type=str,
        default=os.environ.get("WAN_REPO_PATH"),
        help="Path to the TencentARC RollingForcing repository. "
        "Falls back to WAN_REPO_PATH environment variable.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./text_embeddings",
        help="Directory to save generated embeddings (default: ./text_embeddings).",
    )
    args = parser.parse_args()

    if args.wan_repo is None:
        parser.error(
            "--wan-repo is required (or set WAN_REPO_PATH environment variable)"
        )

    return args


def main(args):
    # Setup paths
    repo_dir = args.wan_repo
    sys.path.insert(0, repo_dir)

    # Patch torch.cuda for CPU
    import torch.cuda

    torch.cuda.current_device = lambda: 0
    torch.cuda.is_available = lambda: False

    torch.set_grad_enabled(False)

    # Import text encoder
    from utils.wan_wrapper import WanTextEncoder

    # Patch device property
    WanTextEncoder.device = property(lambda self: torch.device("cpu"))

    print("Loading T5 text encoder...")
    t0 = time.time()
    text_encoder = WanTextEncoder()
    text_encoder = text_encoder.to(device="cpu", dtype=torch.float32)
    print(f"Text encoder loaded in {time.time() - t0:.1f}s")

    # Generate embeddings for test prompts
    prompts = [
        "A cat walks on the grass, realistic style.",
        "A beautiful sunset over the ocean with waves crashing on the shore.",
        "A person walking through a snowy forest at dusk.",
    ]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\nEncoding prompt {i}: '{prompt}'")
        t0 = time.time()
        conditional_dict = text_encoder(text_prompts=[prompt])
        prompt_embeds = conditional_dict["prompt_embeds"]
        print(f"  Shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
        print(f"  Range: [{prompt_embeds.min():.4f}, {prompt_embeds.max():.4f}]")
        print(
            f"  Non-zero tokens: {(prompt_embeds.abs() > 1e-6).any(dim=-1).sum().item()}"
        )
        print(f"  Encoding took {time.time() - t0:.1f}s")

        torch.save(
            {
                "prompt_embeds": prompt_embeds.cpu(),
                "prompt": prompt,
            },
            os.path.join(output_dir, f"enc_{i}.pt"),
        )

    # Also save the same noise used by the reference test for comparison
    noise = torch.randn(
        [1, 21, 16, 60, 104],
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(42),
    )
    torch.save(
        {"noise": noise, "seed": 42}, os.path.join(output_dir, "noise_seed42.pt")
    )

    print(f"\nAll embeddings saved to {output_dir}/")
    print("Files:")
    for f in os.listdir(output_dir):
        print(f"  {f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
