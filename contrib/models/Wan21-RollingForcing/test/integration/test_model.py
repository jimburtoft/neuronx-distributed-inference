#!/usr/bin/env python3
"""
Integration tests for Wan 2.1 RollingForcing on Neuron.

Tests validate that:
1. The compiled DiT model loads and produces valid (non-NaN) output
2. The rolling-forcing pipeline generates latents of correct shape
3. The Neuron VAE decoder produces output matching CPU reference (>= 40 dB PSNR)

Prerequisites:
    - Compiled DiT model (see src/compile_transformer.py)
    - Compiled VAE decoder (see src/compile_vae_decoder.py)
    - Wan 2.1 base weights (from Wan-AI/Wan2.1-T2V-14B on HuggingFace)
    - TencentARC RollingForcing repository (for VAE code and DMD weights)

Environment variables:
    WEIGHT_PATH          Path to Wan diffusers weights directory
    DMD_WEIGHT_PATH      Path to converted DMD weights directory
    COMPILED_MODEL_PATH  Path to compiled unified DiT model directory
    COMPILED_VAE_PATH    Path to compiled VAE decoder directory
    VAE_WEIGHTS_PATH     Path to Wan2.1_VAE.pth
    WAN_REPO_PATH        Path to TencentARC RollingForcing repository

Usage:
    WEIGHT_PATH=/path/to/weights \
    COMPILED_MODEL_PATH=/path/to/compiled \
    pytest test/integration/test_model.py -v -s
"""

import os
import sys
import time

import pytest
import torch

# Add src/ to path for imports
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC_DIR)

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"


def get_env_or_skip(name, description):
    """Get an environment variable or skip the test."""
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set ({description})")
    return value


@pytest.fixture(scope="module")
def weight_path():
    return get_env_or_skip("WEIGHT_PATH", "Wan diffusers weights directory")


@pytest.fixture(scope="module")
def compiled_model_path():
    return get_env_or_skip(
        "COMPILED_MODEL_PATH", "compiled unified DiT model directory"
    )


@pytest.fixture(scope="module")
def tp_degree():
    return int(os.environ.get("TP_DEGREE", "4"))


@pytest.fixture(scope="module")
def resolution():
    """Return (height, width) for the test."""
    return (480, 832)


@pytest.fixture(scope="module")
def unified_app(weight_path, compiled_model_path, tp_degree, resolution):
    """Load the compiled unified DiT model."""
    from modeling_wan21 import (
        NeuronCausalWanUnifiedApplication,
        create_unified_causal_wan_config,
    )

    height, width = resolution
    config = create_unified_causal_wan_config(
        tp_degree=tp_degree, height=height, width=width
    )
    app = NeuronCausalWanUnifiedApplication(
        model_path=weight_path,
        config=config,
    )

    print(f"\nLoading unified model from {compiled_model_path}...")
    t0 = time.time()
    app.load(compiled_model_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    return app


@pytest.fixture(scope="module")
def pipeline(unified_app, tp_degree, resolution):
    """Create the rolling-forcing pipeline."""
    from pipeline import NxDIUnifiedRollingPipeline

    height, width = resolution
    return NxDIUnifiedRollingPipeline(
        unified_app=unified_app,
        tp_degree=tp_degree,
        height=height,
        width=width,
        dtype=torch.bfloat16,
    )


class TestDiTForwardPass:
    """Test that the compiled DiT model produces valid output."""

    def test_forward_produces_valid_output(self, unified_app, tp_degree, resolution):
        """Single forward pass should produce non-NaN, non-zero output."""
        import math
        from modeling_wan21 import (
            make_freqs,
            precompute_rope_embeddings,
            FRAME_SEQ_LENGTH,
            NUM_LAYERS,
            NUM_HEADS,
            HEAD_DIM,
            MAX_ATTENTION_SIZE,
            PATCH_T,
            PATCH_H,
            PATCH_W,
            TEXT_SEQ_LEN,
            TEXT_DIM,
        )

        height, width = resolution
        lat_h, lat_w = height // 8, width // 8
        dtype = torch.bfloat16
        num_frames = 15
        heads_per_rank = math.ceil(NUM_HEADS / tp_degree) * tp_degree // tp_degree

        hidden = torch.randn(1, 16, num_frames, lat_h, lat_w, dtype=dtype)
        timestep = torch.full((1, num_frames), 500.0, dtype=torch.float32)
        enc = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)

        post_f = num_frames // PATCH_T
        post_h = lat_h // PATCH_H
        post_w = lat_w // PATCH_W
        seq_len = post_f * post_h * post_w

        base_freqs = make_freqs(HEAD_DIM)
        rope_cos, rope_sin = precompute_rope_embeddings(
            base_freqs, post_f, post_h, post_w
        )
        rope_cos = rope_cos.to(torch.float32)
        rope_sin = rope_sin.to(torch.float32)

        # Self-mode mask (zero KV buffers)
        prefix_len = MAX_ATTENTION_SIZE - seq_len
        attn_mask = torch.full(
            (1, 1, seq_len, MAX_ATTENTION_SIZE),
            torch.finfo(dtype).min,
            dtype=dtype,
        )
        attn_mask[:, :, :seq_len, prefix_len : prefix_len + seq_len] = 0.0

        kv_tensors = []
        for _ in range(NUM_LAYERS):
            kv_tensors.append(
                torch.zeros(
                    1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype
                )
            )
            kv_tensors.append(
                torch.zeros(
                    1, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM, dtype=dtype
                )
            )

        with torch.no_grad():
            outputs = unified_app.forward_self(
                hidden, timestep, enc, rope_cos, rope_sin, attn_mask, *kv_tensors
            )

        output = outputs[0]
        assert output.shape == hidden.shape, (
            f"Output shape mismatch: {output.shape} vs {hidden.shape}"
        )
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        assert output.abs().max() > 0, "Output is all zeros"
        print(
            f"  Forward pass: shape={output.shape}, "
            f"range=[{output.min():.2f}, {output.max():.2f}]"
        )


class TestRollingForcingPipeline:
    """Test the full rolling-forcing generation pipeline."""

    def test_pipeline_generates_correct_shape(self, pipeline, resolution):
        """Pipeline should produce latents of shape [1, 16, 21, H/8, W/8]."""
        from modeling_wan21 import TEXT_SEQ_LEN, TEXT_DIM
        from window_schedule import NUM_FRAMES

        height, width = resolution
        lat_h, lat_w = height // 8, width // 8
        dtype = torch.bfloat16

        enc = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)
        noise = torch.randn(1, 16, NUM_FRAMES, lat_h, lat_w, dtype=dtype)

        with torch.no_grad():
            output = pipeline.generate(enc, noise)

        expected_shape = (1, 16, NUM_FRAMES, lat_h, lat_w)
        assert output.shape == expected_shape, (
            f"Output shape mismatch: {output.shape} vs {expected_shape}"
        )
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        print(
            f"  Pipeline output: shape={output.shape}, "
            f"range=[{output.min():.4f}, {output.max():.4f}], "
            f"mean={output.float().mean():.6f}, std={output.float().std():.6f}"
        )

    def test_pipeline_deterministic_with_seed(self, pipeline, resolution):
        """Same seed should produce identical output."""
        from modeling_wan21 import TEXT_SEQ_LEN, TEXT_DIM
        from window_schedule import NUM_FRAMES

        height, width = resolution
        lat_h, lat_w = height // 8, width // 8
        dtype = torch.bfloat16

        enc = torch.randn(1, TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)

        torch.manual_seed(42)
        noise1 = torch.randn(1, 16, NUM_FRAMES, lat_h, lat_w, dtype=dtype)
        with torch.no_grad():
            output1 = pipeline.generate(enc, noise1)

        torch.manual_seed(42)
        noise2 = torch.randn(1, 16, NUM_FRAMES, lat_h, lat_w, dtype=dtype)
        with torch.no_grad():
            output2 = pipeline.generate(enc, noise2)

        assert torch.allclose(output1, output2, atol=1e-5), (
            f"Non-deterministic output: max diff = "
            f"{(output1 - output2).abs().max().item()}"
        )
        print("  Deterministic: outputs match with same seed")


class TestVAEDecoder:
    """Test the Neuron VAE decoder accuracy against CPU reference."""

    @pytest.fixture(scope="class")
    def vae_paths(self):
        compiled_vae = get_env_or_skip(
            "COMPILED_VAE_PATH", "compiled VAE decoder directory"
        )
        vae_weights = get_env_or_skip("VAE_WEIGHTS_PATH", "Wan2.1_VAE.pth")
        wan_repo = get_env_or_skip(
            "WAN_REPO_PATH", "TencentARC RollingForcing repository"
        )
        return compiled_vae, vae_weights, wan_repo

    def test_neuron_vae_matches_cpu_reference(self, vae_paths):
        """Neuron VAE decoder PSNR should be >= 40 dB vs CPU reference."""
        compiled_vae, vae_weights, wan_repo = vae_paths

        if wan_repo not in sys.path:
            sys.path.insert(0, wan_repo)

        # Patch torch.cuda for CPU usage
        import torch.cuda as _tc

        _tc.current_device = lambda: 0
        _tc.is_available = lambda: False

        from wan.modules.vae import WanVAE_
        from decode_vae_neuron import (
            NeuronDecoderWrapper,
            decode_hybrid,
            load_model_config,
            load_duplicated_weights,
            NUM_FEAT_CACHE,
        )
        from neuronx_distributed import NxDModel, NxDParallelState

        height, width = 480, 832

        # Create test latents
        torch.manual_seed(123)
        latents = torch.randn(1, 16, 21, height // 8, width // 8, dtype=torch.float32)

        # CPU reference decode
        print("  Loading CPU VAE...")
        vae_cpu = WanVAE_(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
        vae_state = torch.load(vae_weights, map_location="cpu")
        vae_cpu.load_state_dict(vae_state, strict=True)
        vae_cpu.eval()

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
            ]
        )
        std_val = torch.tensor(
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
            ]
        )
        scale = [mean, 1.0 / std_val]

        print("  Running CPU VAE decode...")
        t0 = time.time()
        with torch.no_grad():
            video_cpu = vae_cpu.decode(latents.float(), scale)
        cpu_time = time.time() - t0
        video_cpu = (video_cpu * 0.5 + 0.5).clamp(0, 1)
        print(f"  CPU decode: {cpu_time:.1f}s, shape={video_cpu.shape}")

        # Neuron decode
        print("  Loading Neuron VAE decoder...")
        config_dec = load_model_config(compiled_vae)
        ws = config_dec["world_size"]
        tp_dec = config_dec["tp_degree"]

        with NxDParallelState(world_size=ws, tensor_model_parallel_size=tp_dec):
            nxd_model = NxDModel.load(os.path.join(compiled_vae, "nxd_model.pt"))
            weights_dec = load_duplicated_weights(compiled_vae, ws)
            nxd_model.set_weights(weights_dec)
            nxd_model.to_neuron()

            neuron_decoder = NeuronDecoderWrapper(num_feat_cache=NUM_FEAT_CACHE)
            neuron_decoder._init_feat_cache_shapes(height, width)
            neuron_decoder.nxd_model = nxd_model

            print("  Running Neuron VAE decode...")
            t0 = time.time()
            video_neuron = decode_hybrid(
                latents, vae_cpu, neuron_decoder, height, width
            )
            video_neuron = (video_neuron * 0.5 + 0.5).clamp(0, 1)
            neuron_time = time.time() - t0
            print(f"  Neuron decode: {neuron_time:.1f}s, shape={video_neuron.shape}")

        # Compute PSNR
        mse = ((video_cpu.float() - video_neuron.float()) ** 2).mean().item()
        if mse > 0:
            psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
        else:
            psnr = float("inf")

        cos_sim = torch.nn.functional.cosine_similarity(
            video_cpu.flatten().unsqueeze(0).float(),
            video_neuron.flatten().unsqueeze(0).float(),
        ).item()

        print(f"  PSNR: {psnr:.2f} dB, Cosine similarity: {cos_sim:.6f}")
        assert psnr >= 40.0, f"PSNR too low: {psnr:.2f} dB (expected >= 40 dB)"
        assert cos_sim >= 0.99, f"Cosine similarity too low: {cos_sim:.6f}"
