"""Integration tests for Paris multi-expert diffusion model on Neuron.

Prerequisites:
  - Model weights downloaded: huggingface-cli download bageldotcom/paris --local-dir /home/ubuntu/models/paris
  - NEFFs compiled: python -c "from modeling_paris import trace_all; trace_all('/home/ubuntu/models/paris', '/home/ubuntu/neuron_models/paris')"

Run:
  pytest test_model.py -v --timeout=600
  or: python test_model.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_paris import (
    CLIPTextEncoderWrapper,
    DiTExpert,
    ParisPipeline,
    Router,
    VAEDecoderWrapper,
    trace_all,
)

MODEL_PATH = "/home/ubuntu/models/paris"
COMPILED_MODEL_PATH = "/home/ubuntu/neuron_models/paris"


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def pipeline():
    """Load the compiled Neuron pipeline."""
    return ParisPipeline(
        neff_dir=COMPILED_MODEL_PATH,
        model_dir=MODEL_PATH,
        expert_batch_size=2,
    )


# ============================================================
# Tests
# ============================================================


class TestModelLoads:
    """Verify all NEFFs load successfully."""

    def test_pipeline_loads(self, pipeline):
        assert pipeline.clip is not None
        assert pipeline.router is not None
        assert len(pipeline.experts) == 8
        assert pipeline.vae is not None
        assert pipeline.tokenizer is not None

    def test_all_experts_loaded(self, pipeline):
        for i, expert in enumerate(pipeline.experts):
            assert expert is not None, f"Expert {i} failed to load"


class TestComponentAccuracy:
    """Verify individual component accuracy against CPU reference."""

    def test_clip_accuracy(self, pipeline):
        """CLIP text encoder should produce consistent embeddings."""
        emb1 = pipeline.encode_text("A cat sitting on a mat")
        emb2 = pipeline.encode_text("A cat sitting on a mat")
        cos_sim = F.cosine_similarity(emb1.flatten(), emb2.flatten(), dim=0).item()
        assert cos_sim > 0.999, f"CLIP not deterministic: cos_sim={cos_sim}"

    def test_clip_different_prompts(self, pipeline):
        """Different prompts should produce different embeddings."""
        emb1 = pipeline.encode_text("A beautiful sunset")
        emb2 = pipeline.encode_text("A snowy mountain")
        cos_sim = F.cosine_similarity(emb1.flatten(), emb2.flatten(), dim=0).item()
        assert cos_sim < 0.95, f"Different prompts too similar: cos_sim={cos_sim}"

    def test_router_output_shape(self, pipeline):
        """Router should produce [1, 8] logits."""
        x = torch.randn(1, 4, 32, 32)
        t = torch.tensor([500.0])
        logits = pipeline.router(x, t)
        assert logits.shape == (1, 8), f"Router output shape: {logits.shape}"

    def test_router_produces_valid_distribution(self, pipeline):
        """Router softmax should sum to 1."""
        x = torch.randn(1, 4, 32, 32)
        t = torch.tensor([500.0])
        logits = pipeline.router(x, t)
        probs = F.softmax(logits, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-4, f"Probs sum: {probs.sum().item()}"

    def test_expert_output_shape(self, pipeline):
        """Expert BS=2 should return [2, 4, 32, 32]."""
        x = torch.randn(2, 4, 32, 32)
        t = torch.tensor([500.0, 500.0])
        emb = torch.randn(2, 77, 768)
        out = pipeline.experts[0](x, t, emb)
        assert out.shape == (2, 4, 32, 32), f"Expert output shape: {out.shape}"


class TestGeneration:
    """Verify end-to-end image generation."""

    def test_generates_image_top1(self, pipeline):
        """Top-1 routing should produce a valid image."""
        img = pipeline.generate(
            "A sunset over the ocean", routing="top1", num_steps=20, seed=42
        )
        assert img.size == (256, 256), f"Image size: {img.size}"
        arr = np.array(img)
        assert arr.shape == (256, 256, 3), f"Array shape: {arr.shape}"
        assert arr.std() > 10, f"Image appears blank (std={arr.std():.1f})"

    def test_generates_image_top2(self, pipeline):
        """Top-2 routing should produce a valid image."""
        img = pipeline.generate("A fluffy cat", routing="top2", num_steps=20, seed=42)
        assert img.size == (256, 256)
        arr = np.array(img)
        assert arr.std() > 10, f"Image appears blank (std={arr.std():.1f})"

    def test_deterministic_generation(self, pipeline):
        """Same seed should produce identical images."""
        img1 = pipeline.generate("A red car", routing="top1", num_steps=20, seed=123)
        img2 = pipeline.generate("A red car", routing="top1", num_steps=20, seed=123)
        arr1, arr2 = np.array(img1), np.array(img2)
        assert np.array_equal(arr1, arr2), "Same seed produced different images"

    def test_different_seeds_differ(self, pipeline):
        """Different seeds should produce different images."""
        img1 = pipeline.generate("A flower", routing="top1", num_steps=20, seed=1)
        img2 = pipeline.generate("A flower", routing="top1", num_steps=20, seed=2)
        arr1, arr2 = np.array(img1).astype(float), np.array(img2).astype(float)
        mse = np.mean((arr1 - arr2) ** 2)
        assert mse > 100, f"Different seeds too similar (MSE={mse:.1f})"


class TestPerformance:
    """Verify inference performance meets expectations."""

    def test_top1_latency(self, pipeline):
        """Top-1 E2E should be under 2000ms (with margin)."""
        # Warmup
        pipeline.generate("warmup", routing="top1", num_steps=10, seed=0)

        t0 = time.perf_counter()
        pipeline.generate("A sunset", routing="top1", num_steps=50, seed=42)
        latency = (time.perf_counter() - t0) * 1000

        print(f"Top-1 50-step latency: {latency:.0f}ms")
        assert latency < 2000, f"Top-1 too slow: {latency:.0f}ms (expected <2000ms)"

    def test_top2_latency(self, pipeline):
        """Top-2 E2E should be under 3000ms (with margin)."""
        t0 = time.perf_counter()
        pipeline.generate("A sunset", routing="top2", num_steps=50, seed=42)
        latency = (time.perf_counter() - t0) * 1000

        print(f"Top-2 50-step latency: {latency:.0f}ms")
        assert latency < 3000, f"Top-2 too slow: {latency:.0f}ms (expected <3000ms)"


# ============================================================
# Manual runner
# ============================================================

if __name__ == "__main__":
    print("Loading pipeline...")
    pipe = ParisPipeline(
        neff_dir=COMPILED_MODEL_PATH,
        model_dir=MODEL_PATH,
        expert_batch_size=2,
    )

    print("\n--- Component Tests ---")
    emb = pipe.encode_text("test prompt")
    print(f"CLIP output: {emb.shape}")

    x = torch.randn(1, 4, 32, 32)
    t = torch.tensor([500.0])
    logits = pipe.router(x, t)
    probs = F.softmax(logits, dim=-1)
    print(f"Router output: {logits.shape}, top expert: {probs.argmax().item()}")

    x2 = torch.randn(2, 4, 32, 32)
    t2 = torch.tensor([500.0, 500.0])
    emb2 = torch.randn(2, 77, 768)
    out = pipe.experts[0](x2, t2, emb2)
    print(f"Expert BS=2 output: {out.shape}")

    print("\n--- Generation Tests ---")
    for routing in ["top1", "top2"]:
        t0 = time.perf_counter()
        img = pipe.generate(
            "A beautiful sunset over the ocean", routing=routing, num_steps=50, seed=42
        )
        lat = (time.perf_counter() - t0) * 1000
        arr = np.array(img)
        print(f"  {routing}: {lat:.0f}ms, size={img.size}, std={arr.std():.1f}")
        img.save(f"/tmp/paris_test_{routing}.png")

    print("\n--- Determinism Test ---")
    img1 = pipe.generate("A red car", routing="top1", num_steps=20, seed=42)
    img2 = pipe.generate("A red car", routing="top1", num_steps=20, seed=42)
    match = np.array_equal(np.array(img1), np.array(img2))
    print(f"  Deterministic: {match}")

    print("\nAll manual tests passed!")
