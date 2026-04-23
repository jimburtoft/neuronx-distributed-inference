#!/usr/bin/env python3
"""
Integration test for GLM-5 (zai-org/GLM-5) on NeuronX Distributed Inference.

Requires:
- trn2.48xlarge instance (TP=64, LNC=2)
- GLM-5 FP8 weights downloaded to MODEL_PATH
- Pre-compiled model at COMPILED_MODEL_PATH (or will compile on first run)

Usage:
    # Basic run (compile + generate):
    python test_model.py

    # With pytest:
    pytest test_model.py -v -s
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig

# Import from src directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_glm5 import NeuronGLM5ForCausalLM, GLM5InferenceConfig

# ---------------------------------------------------------------------------
# Test configuration -- update these paths for your environment
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("GLM5_MODEL_PATH", "/mnt/models/GLM-5-FP8")
COMPILED_MODEL_PATH = os.environ.get("GLM5_COMPILED_PATH", "/mnt/models/glm5_compiled")
CPU_REFERENCE_LOGITS_PATH = os.environ.get(
    "GLM5_REF_LOGITS_PATH", "/mnt/models/glm5_cpu_reference_logits.pt"
)

# Generation settings
NUM_TOKENS_TO_CHECK = 16
MAX_SEQ_LEN = 4096
BATCH_SIZE = 1

# Logit validation tolerances (FP8 + MoE + TP=64 = higher error)
DIVERGENCE_DIFFERENCE_TOL = 0.03
TOLERANCE_MAP = {
    5: (1e-5, 0.10),  # Top-5: FP8 + MoE expected higher
    50: (1e-5, 0.12),  # Top-50
    1000: (1e-5, 0.12),  # Top-1000
    None: (1e-5, 0.15),  # All tokens
}


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_model(model_path: str, compiled_path: str):
    """Create and configure the GLM-5 model for inference on trn2.48xlarge."""
    with open(f"{model_path}/config.json") as f:
        hf_config = json.load(f)

    neuron_config = MoENeuronConfig(
        tp_degree=64,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        n_active_tokens=MAX_SEQ_LEN,
        torch_dtype=torch.bfloat16,
        # MLA attention: no fused QKV (separate q_a/q_b/kv_a/kv_b projections)
        fused_qkv=False,
        # Attention NKI kernels
        qkv_kernel_enabled=True,
        qkv_nki_kernel_enabled=True,
        # MoE NKI kernels must be disabled:
        # moe_intermediate/TP = 2048/64 = 32, 32 % 128 != 0
        moe_fused_nki_kernel_enabled=False,
        expert_mlp_nki_kernel_enabled=False,
    )

    def load_config(c):
        for k, v in hf_config.items():
            setattr(c, k, v)

    config = GLM5InferenceConfig(neuron_config=neuron_config, load_config=load_config)
    model = NeuronGLM5ForCausalLM(model_path, config)
    return model


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_model():
    """Compile (or load from cache) and load the model."""
    model = create_model(MODEL_PATH, COMPILED_MODEL_PATH)
    model.compile(compiled_model_path=COMPILED_MODEL_PATH)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def generation_config(tokenizer):
    """Create generation config."""
    return GenerationConfig(
        do_sample=False,
        max_new_tokens=NUM_TOKENS_TO_CHECK,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_loads(compiled_model):
    """Smoke test: model loads and compiles successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    print("PASS: Model loaded and compiled successfully")


def test_greedy_generation(compiled_model, tokenizer, generation_config):
    """Test greedy generation produces coherent output."""
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    output = compiled_model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Prompt: '{prompt}'")
    print(f"  Generated: '{generated_text}'")

    # Basic sanity: output should be longer than input
    assert len(output[0]) > len(input_ids[0]), "No tokens generated"
    print(f"PASS: Generated {len(output[0]) - len(input_ids[0])} tokens")


def test_logit_accuracy(compiled_model, tokenizer, generation_config):
    """
    Validate Neuron model logits against CPU reference (if available).

    Requires pre-computed CPU reference logits. Skip if not available.
    """
    if not os.path.exists(CPU_REFERENCE_LOGITS_PATH):
        pytest.skip(f"CPU reference logits not found at {CPU_REFERENCE_LOGITS_PATH}")

    from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits_v2

    cpu_ref = torch.load(CPU_REFERENCE_LOGITS_PATH, weights_only=True)
    expected_logits = cpu_ref["expected_logits"]
    input_ids = cpu_ref["input_ids"]
    attention_mask = torch.ones_like(input_ids)

    print(f"  Reference logits shape: {expected_logits.shape}")
    print(f"  Prompt: '{cpu_ref.get('prompt', 'N/A')}'")

    check_accuracy_logits_v2(
        neuron_model=compiled_model,
        expected_logits=expected_logits,
        inputs_input_ids=input_ids,
        inputs_attention_mask=attention_mask,
        generation_config=generation_config,
        num_tokens_to_check=NUM_TOKENS_TO_CHECK,
        divergence_difference_tol=DIVERGENCE_DIFFERENCE_TOL,
        tol_map=TOLERANCE_MAP,
        tokenizer=tokenizer,
    )
    print(f"PASS: Logit accuracy validated ({NUM_TOKENS_TO_CHECK} tokens)")


def test_performance_tkg(compiled_model, tokenizer):
    """Measure TKG (token generation / decode) latency."""
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]

    # CTE (prefill) to set up KV cache
    output = compiled_model.forward(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
        seq_ids=torch.zeros(1, dtype=torch.int32),
    )
    logits = (output.logits if hasattr(output, "logits") else output[0])[0, -1, :]
    prev_token = torch.argmax(logits.float().cpu()).item()
    cur_pos = seq_len

    def run_tkg(token_id, pos):
        return compiled_model.forward(
            input_ids=torch.tensor([[token_id]], dtype=torch.int64),
            attention_mask=torch.ones(1, pos + 1, dtype=torch.int64),
            position_ids=torch.tensor([[pos]], dtype=torch.int32),
            seq_ids=torch.zeros(1, dtype=torch.int32),
        )

    # Warmup TKG
    for i in range(5):
        out = run_tkg(prev_token, cur_pos + i)
        logits = (out.logits if hasattr(out, "logits") else out[0])[0, -1, :]
        prev_token = torch.argmax(logits.float().cpu()).item()
    cur_pos += 5

    # Measure
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        out = run_tkg(prev_token, cur_pos + i)
        times.append((time.perf_counter() - t0) * 1000)
        logits = (out.logits if hasattr(out, "logits") else out[0])[0, -1, :]
        prev_token = torch.argmax(logits.float().cpu()).item()

    avg_ms = sum(times) / len(times)
    p50_ms = sorted(times)[len(times) // 2]
    tok_per_sec = 1000 / avg_ms
    print(
        f"  TKG latency: {avg_ms:.1f} ms avg, {p50_ms:.1f} ms p50 ({tok_per_sec:.1f} tok/s)"
    )
    # DS-V3 on trn2.48xlarge achieves ~48.7 tok/s. GLM-5 should be comparable.
    # Sanity: TKG under 200ms (conservative for first run)
    assert avg_ms < 200, f"TKG too slow: {avg_ms:.1f} ms"
    print(f"PASS: TKG latency {avg_ms:.1f} ms ({tok_per_sec:.1f} tok/s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 80)
    print("GLM-5 (754B MoE) Integration Test")
    print("=" * 80)
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Compiled: {COMPILED_MODEL_PATH}")
    print(f"  TP=64, EP=1, LNC=2, FP8")
    print()

    # Create model
    print("Creating model...")
    model = create_model(MODEL_PATH, COMPILED_MODEL_PATH)

    print("Compiling (may use cache)...")
    t0 = time.time()
    model.compile(compiled_model_path=COMPILED_MODEL_PATH)
    compile_time = time.time() - t0
    print(f"Compiled in {compile_time:.0f}s")

    print("Loading weights...")
    t0 = time.time()
    model.load(COMPILED_MODEL_PATH)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.0f}s")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=NUM_TOKENS_TO_CHECK,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )

    # Tests
    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    print("\n1. Smoke Test...")
    test_model_loads(model)

    print("\n2. Greedy Generation Test...")
    test_greedy_generation(model, tok, gen_config)

    print("\n3. Logit Accuracy Test...")
    if os.path.exists(CPU_REFERENCE_LOGITS_PATH):
        test_logit_accuracy(model, tok, gen_config)
    else:
        print(f"  SKIP: CPU reference logits not found at {CPU_REFERENCE_LOGITS_PATH}")

    print("\n4. TKG Performance Test...")
    test_performance_tkg(model, tok)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print(f"  Compile: {compile_time:.0f}s, Load: {load_time:.0f}s")
    print("=" * 80)
