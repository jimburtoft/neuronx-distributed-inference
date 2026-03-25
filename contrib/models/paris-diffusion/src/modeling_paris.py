"""Paris Multi-Expert Diffusion Model for AWS Neuron.

Bagel Labs' Paris (bageldotcom/paris) -- an 8-expert DiT diffusion model
for 256x256 text-to-image generation, traced via torch_neuronx.trace().

Architecture:
  - 8x DiT-XL/2 experts (606M params each, AdaLN-Single conditioning)
  - 1x DiT-B router (129M params, AdaLN-Zero, CLS token classification)
  - CLIP ViT-L/14 text encoder (768-dim, 77 max tokens)
  - AutoencoderKL VAE decoder (sd-vae-ft-mse)
  - FlowMatchEulerDiscreteScheduler (velocity prediction)

Routing strategies:
  - top-1: Single best expert per step (fastest)
  - top-2: Weighted average of top-2 experts (best quality, FID 22.60)
  - full:  Weighted average of all 8 experts

CFG batching:
  Conditioned and unconditioned passes are batched into a single BS=2
  expert NEFF call per expert per step, following standard practice.

Reference: https://arxiv.org/abs/2510.03434
License: MIT
"""

import math
import os
import time
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer


# ============================================================
# Model Definitions
# ============================================================


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


def _t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT transformer block with AdaLN-Single conditioning and cross-attention."""

    def __init__(
        self, hidden_size: int = 1152, num_heads: int = 16, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor, t0: torch.Tensor, text_proj: torch.Tensor):
        B, N, C = x.shape
        modulation = self.scale_shift_table[None] + t0
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=1)
        )
        x_norm = F.layer_norm(x, (C,))
        x_mod = _t2i_modulate(x_norm, shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_msa * attn_out.reshape(B, N, C)
        cross_out, _ = self.cross_attn(x, text_proj, text_proj, need_weights=False)
        x = x + cross_out
        x_norm2 = F.layer_norm(x, (C,))
        x_mod2 = _t2i_modulate(x_norm2, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mod2)
        return x


class DiTExpert(nn.Module):
    """DiT-XL/2 expert with AdaLN-Single conditioning (PixArt-alpha style).

    28 layers, 1152 hidden, 16 heads, patch_size=2. Takes latent [B,4,32,32],
    timestep [B], and text_emb [B,77,768]. Returns velocity [B,4,32,32].
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.input_size = 32
        num_patches = (32 // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.time_embed = TimestepEmbedder(hidden_size)
        self.text_proj = nn.Linear(768, hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.layers = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = nn.Module()
        self.final_layer.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )
        self.final_layer.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, c = self.patch_size, self.out_channels
        h = w = self.input_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor):
        B, D = x.shape[0], self.hidden_size
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        t_emb = self.time_embed(t)
        t0 = self.t_block(t_emb).reshape(B, 6, D)
        text_proj = self.text_proj(text_emb)
        for layer in self.layers:
            x = layer(x, t0, text_proj)
        final_mod = self.final_layer.scale_shift_table[None] + t_emb[:, None]
        shift, scale = final_mod.chunk(2, dim=1)
        x = F.layer_norm(x, (D,))
        x = _t2i_modulate(x, shift, scale)
        x = self.final_layer.linear(x)
        return self.unpatchify(x)


class RouterBlock(nn.Module):
    """DiT-B transformer block with AdaLN-Zero conditioning (no cross-attention)."""

    def __init__(
        self, hidden_size: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        B, N, C = x.shape
        mod = self.adaLN_modulation(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.reshape(
            B, 6, C
        ).chunk(6, dim=1)
        x_norm = F.layer_norm(x, (C,))
        x_mod = _t2i_modulate(x_norm, shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_msa * attn_out.reshape(B, N, C)
        x_norm2 = F.layer_norm(x, (C,))
        x_mod2 = _t2i_modulate(x_norm2, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mod2)
        return x


class Router(nn.Module):
    """DiT-B router with CLS token and classification head.

    12 layers, 768 hidden, 12 heads. Takes latent [B,4,32,32] and timestep [B].
    Returns logits [B, 8] over 8 experts. Does NOT take text input.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 2,
        in_channels: int = 4,
        num_clusters: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        num_patches = (32 // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.time_embed = TimestepEmbedder(hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.layers = nn.ModuleList(
            [RouterBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, num_clusters),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        t_emb = self.time_embed(t)
        for layer in self.layers:
            x = layer(x, t_emb)
        cls_out = x[:, 0]
        return self.head(cls_out)


class CLIPTextEncoderWrapper(nn.Module):
    """Wrapper for tracing: takes input_ids, returns last_hidden_state."""

    def __init__(self, model: CLIPTextModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        return self.model(input_ids).last_hidden_state


class VAEDecoderWrapper(nn.Module):
    """Wrapper for tracing: takes latent, returns decoded image."""

    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv

    def forward(self, latent: torch.Tensor):
        return self.decoder(self.post_quant_conv(latent))


# ============================================================
# Tracing
# ============================================================

COMPILER_ARGS = ["--auto-cast", "matmult", "--auto-cast-type", "bf16"]


def trace_all(
    model_dir: str,
    output_dir: str,
    expert_batch_size: int = 2,
) -> dict:
    """Trace all Paris components to Neuron NEFFs.

    Args:
        model_dir: Path to HuggingFace model directory (bageldotcom/paris).
        output_dir: Path to save compiled NEFF files.
        expert_batch_size: Batch size for expert tracing. Use 2 for CFG batching
            (conditioned + unconditioned in one call). Use 1 for sequential CFG.

    Returns:
        Dict of tracing results per component.
    """
    import gc

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # CLIP text encoder (always BS=1)
    print("Tracing CLIP text encoder...")
    te = CLIPTextModel.from_pretrained(f"{model_dir}/text_encoder").eval()
    tok = CLIPTokenizer.from_pretrained(f"{model_dir}/tokenizer")
    wrapper = CLIPTextEncoderWrapper(te).eval()
    tokens = tok(
        "test",
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    t0 = time.time()
    clip_neuron = torch_neuronx.trace(
        wrapper, (tokens.input_ids,), compiler_args=COMPILER_ARGS
    )
    results["clip"] = {"compile_time_s": time.time() - t0}
    clip_neuron.save(f"{output_dir}/clip_text_encoder.pt")
    print(f"  Done ({results['clip']['compile_time_s']:.0f}s)")
    del te, wrapper
    gc.collect()

    # Router (always BS=1)
    print("Tracing router...")
    router = Router()
    router_w = load_file(f"{model_dir}/router/pytorch_model.safetensors")
    router.load_state_dict(router_w, strict=False)
    router.eval()

    t0 = time.time()
    router_neuron = torch_neuronx.trace(
        router,
        (torch.randn(1, 4, 32, 32), torch.tensor([500.0])),
        compiler_args=COMPILER_ARGS,
    )
    results["router"] = {"compile_time_s": time.time() - t0}
    router_neuron.save(f"{output_dir}/router.pt")
    print(f"  Done ({results['router']['compile_time_s']:.0f}s)")
    del router, router_w
    gc.collect()

    # 8 experts (BS=expert_batch_size for CFG batching)
    x_ex = torch.randn(expert_batch_size, 4, 32, 32)
    t_ex = torch.full((expert_batch_size,), 500.0)
    emb_ex = torch.randn(expert_batch_size, 77, 768)

    for i in range(8):
        print(f"Tracing expert {i}/7 (BS={expert_batch_size})...")
        expert = DiTExpert()
        w = load_file(f"{model_dir}/expert_{i}/diffusion_pytorch_model.safetensors")
        expert.load_state_dict(w, strict=False)
        expert.eval()

        t0 = time.time()
        expert_neuron = torch_neuronx.trace(
            expert, (x_ex, t_ex, emb_ex), compiler_args=COMPILER_ARGS
        )
        ct = time.time() - t0
        suffix = f"_bs{expert_batch_size}" if expert_batch_size > 1 else ""
        expert_neuron.save(f"{output_dir}/expert_{i}{suffix}.pt")
        results[f"expert_{i}"] = {"compile_time_s": ct}
        print(f"  Done ({ct:.0f}s)")
        del expert, w, expert_neuron
        gc.collect()

    # VAE decoder (always BS=1)
    print("Tracing VAE decoder...")
    vae = AutoencoderKL.from_pretrained(f"{model_dir}/vae").eval()
    vae_wrapper = VAEDecoderWrapper(vae).eval()

    t0 = time.time()
    vae_neuron = torch_neuronx.trace(
        vae_wrapper, (torch.randn(1, 4, 32, 32),), compiler_args=COMPILER_ARGS
    )
    results["vae"] = {"compile_time_s": time.time() - t0}
    vae_neuron.save(f"{output_dir}/vae_decoder.pt")
    print(f"  Done ({results['vae']['compile_time_s']:.0f}s)")
    del vae, vae_wrapper
    gc.collect()

    total = sum(r["compile_time_s"] for r in results.values())
    print(f"\nTotal compilation time: {total:.0f}s ({total / 60:.1f} min)")
    return results


# ============================================================
# Pipeline
# ============================================================


class ParisPipeline:
    """End-to-end Paris inference pipeline on Neuron.

    Loads pre-compiled NEFFs and runs the full diffusion loop with
    CFG-batched expert calls.

    Args:
        neff_dir: Path to directory containing compiled NEFF files.
        model_dir: Path to HuggingFace model directory (for tokenizer).
        expert_batch_size: Batch size used when tracing experts (1 or 2).
    """

    def __init__(self, neff_dir: str, model_dir: str, expert_batch_size: int = 2):
        self.expert_batch_size = expert_batch_size
        self.cfg_batched = expert_batch_size == 2

        print("Loading NEFFs...")
        self.clip = torch.jit.load(f"{neff_dir}/clip_text_encoder.pt")
        self.router = torch.jit.load(f"{neff_dir}/router.pt")

        suffix = f"_bs{expert_batch_size}" if expert_batch_size > 1 else ""
        self.experts = []
        for i in range(8):
            self.experts.append(torch.jit.load(f"{neff_dir}/expert_{i}{suffix}.pt"))
        self.vae = torch.jit.load(f"{neff_dir}/vae_decoder.pt")

        self.tokenizer = CLIPTokenizer.from_pretrained(f"{model_dir}/tokenizer")
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        print("Pipeline ready.")

    def encode_text(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return self.clip(tokens.input_ids)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        routing: Literal["top1", "top2", "full"] = "top2",
        cfg_scale: float = 7.5,
        num_steps: int = 50,
        seed: int = 42,
    ) -> Image.Image:
        """Generate a 256x256 image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            routing: Expert routing strategy ("top1", "top2", or "full").
            cfg_scale: Classifier-free guidance scale. 7.5 is recommended.
            num_steps: Number of diffusion steps. 50 is recommended.
            seed: Random seed for reproducibility.

        Returns:
            PIL Image (256x256 RGB).
        """
        self.scheduler.set_timesteps(num_steps)

        text_emb = self.encode_text(prompt)
        uncond_emb = self.encode_text("")

        torch.manual_seed(seed)
        x_t = torch.randn(1, 4, 32, 32)

        for t_idx in self.scheduler.timesteps:
            t_tensor = t_idx.unsqueeze(0)

            # Route
            router_logits = self.router(x_t, t_tensor)
            router_probs = F.softmax(router_logits, dim=-1)

            # Expert calls with CFG
            v = self._run_experts(
                x_t, t_tensor, text_emb, uncond_emb, router_probs, routing, cfg_scale
            )

            x_t = self.scheduler.step(v, t_idx, x_t).prev_sample

        # VAE decode
        img = self.vae(x_t / 0.18215)
        img = (img / 2 + 0.5).clamp(0, 1)
        return Image.fromarray(
            (img.permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)
        )

    def _run_experts(
        self, x_t, t_tensor, text_emb, uncond_emb, router_probs, routing, cfg_scale
    ):
        """Run expert forward passes with CFG, using batched or sequential calls."""
        if self.cfg_batched:
            x_batch = torch.cat([x_t, x_t], dim=0)
            t_batch = torch.cat([t_tensor, t_tensor], dim=0)
            emb_batch = torch.cat([text_emb, uncond_emb], dim=0)

        def _expert_cfg(idx):
            if self.cfg_batched:
                v_both = self.experts[idx](x_batch, t_batch, emb_batch)
                return v_both[1:2] + cfg_scale * (v_both[0:1] - v_both[1:2])
            else:
                v_cond = self.experts[idx](x_t, t_tensor, text_emb)
                v_uncond = self.experts[idx](x_t, t_tensor, uncond_emb)
                return v_uncond + cfg_scale * (v_cond - v_uncond)

        if routing == "top1":
            top_idx = router_probs.argmax(dim=-1).item()
            return _expert_cfg(top_idx)

        elif routing == "top2":
            top2_vals, top2_idx = router_probs.topk(2, dim=-1)
            top2_weights = top2_vals / top2_vals.sum(dim=-1, keepdim=True)
            idx0, idx1 = top2_idx[0, 0].item(), top2_idx[0, 1].item()
            w0, w1 = top2_weights[0, 0].item(), top2_weights[0, 1].item()
            return w0 * _expert_cfg(idx0) + w1 * _expert_cfg(idx1)

        elif routing == "full":
            v = torch.zeros_like(x_t)
            for k in range(8):
                wk = router_probs[0, k].item()
                if wk < 1e-6:
                    continue
                v = v + wk * _expert_cfg(k)
            return v
        else:
            raise ValueError(f"Unknown routing strategy: {routing}")
