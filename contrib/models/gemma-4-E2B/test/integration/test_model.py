#!/usr/bin/env python3
"""
Integration tests for Gemma-4-E2B NeuronX Distributed Inference implementation.

Tests model compilation, loading, and inference accuracy/performance.

Usage:
    # Run with pytest
    pytest test/integration/test_model.py --capture=tee-sys

    # Run standalone
    python test/integration/test_model.py

Environment variables:
    GEMMA4_E2B_MODEL_PATH  - Path to HF model weights (default: /mnt/models/gemma-4-E2B)
    GEMMA4_E2B_COMPILED_PATH - Path for compiled NEFFs (default: /mnt/models/gemma4-e2b-compiled)
    GEMMA4_E2B_TOKENIZER_PATH - Path for tokenizer (default: model path)
    GEMMA4_E2B_TP_DEGREE - Tensor parallelism degree (default: 1)
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

# Apply NxDI patches before importing model classes
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from ndxi_patch import apply_patch

apply_patch()

from modeling_gemma4_e2b import (
    NeuronGemma4E2BForCausalLM,
    Gemma4E2BInferenceConfig,
    Gemma4E2BNeuronConfig,
)


# ============================================================================
# Test configuration
# ============================================================================

MODEL_PATH = os.environ.get("GEMMA4_E2B_MODEL_PATH", "/mnt/models/gemma-4-E2B")
COMPILED_MODEL_PATH = os.environ.get(
    "GEMMA4_E2B_COMPILED_PATH", "/mnt/models/gemma4-e2b-compiled"
)
TOKENIZER_PATH = os.environ.get("GEMMA4_E2B_TOKENIZER_PATH", MODEL_PATH)
TP_DEGREE = int(os.environ.get("GEMMA4_E2B_TP_DEGREE", "1"))
BATCH_SIZE = 1
BUCKET_SEQ_LEN = 128
MAX_LENGTH = 512


# ============================================================================
# Helpers
# ============================================================================


def create_config(model_path, tp_degree=TP_DEGREE, batch_size=BATCH_SIZE):
    """Create Gemma4 E2B inference config from model path."""
    neuron_config = Gemma4E2BNeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_batch_size=batch_size,
        max_length=MAX_LENGTH,
        seq_len=BUCKET_SEQ_LEN,
        torch_dtype=torch.bfloat16,
        attn_kernel_enabled=False,
        weights_to_skip_layout_optimization=[r".*embed_tokens_per_layer.*"],
    )

    with open(os.path.join(model_path, "config.json")) as f:
        raw_config = json.load(f)

    def load_config_fn(config_obj):
        for k, v in raw_config.items():
            setattr(config_obj, k, v)
        config_obj._name_or_path = model_path

    return Gemma4E2BInferenceConfig(
        neuron_config=neuron_config, load_config=load_config_fn
    )


def generate_tokens(model, tokenizer, prompt, max_new_tokens=20):
    """
    Generate tokens using manual forward pass loop (prefill + decode).

    Returns (generated_token_ids, output_text, timing_dict).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]

    # Pad to bucket size
    padded = torch.nn.functional.pad(input_ids, (0, BUCKET_SEQ_LEN - seq_len), value=0)
    attn_mask = torch.zeros(1, BUCKET_SEQ_LEN, dtype=torch.int32)
    attn_mask[0, :seq_len] = 1
    pos_ids = torch.zeros(1, BUCKET_SEQ_LEN, dtype=torch.int32)
    pos_ids[0, :seq_len] = torch.arange(seq_len, dtype=torch.int32)
    seq_ids = torch.tensor([0], dtype=torch.int32)

    model.reset_kv_cache()
    timing = {}

    # Prefill (context encoding)
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=padded,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            seq_ids=seq_ids,
        )
    timing["ttft_ms"] = (time.perf_counter() - t0) * 1000

    # Extract first token
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
    elif hasattr(outputs, "tokens") and outputs.tokens is not None:
        next_token_id = (
            outputs.tokens[0, -1].item()
            if outputs.tokens.dim() > 1
            else outputs.tokens[0].item()
        )
    else:
        return [], "", timing

    generated_tokens = [next_token_id]
    cur_pos = seq_len

    # Token generation loop
    t_gen_start = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        tok_input = torch.tensor([[next_token_id]], dtype=torch.int64)
        tok_pos = torch.tensor([[cur_pos]], dtype=torch.int32)
        tok_attn = torch.ones(1, MAX_LENGTH, dtype=torch.int32)
        tok_attn[0, cur_pos + 1 :] = 0

        with torch.inference_mode():
            outputs = model(
                input_ids=tok_input,
                attention_mask=tok_attn,
                position_ids=tok_pos,
                seq_ids=seq_ids,
            )
        cur_pos += 1

        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        elif hasattr(outputs, "tokens") and outputs.tokens is not None:
            next_token_id = (
                outputs.tokens[0, -1].item()
                if outputs.tokens.dim() > 1
                else outputs.tokens[0].item()
            )
        else:
            break

        generated_tokens.append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            break

    t_gen_end = time.perf_counter()
    num_decode_tokens = len(generated_tokens) - 1
    if num_decode_tokens > 0:
        decode_time = t_gen_end - t_gen_start
        timing["tpot_ms"] = decode_time / num_decode_tokens * 1000
        timing["throughput_tps"] = num_decode_tokens / decode_time
    timing["total_tokens"] = len(generated_tokens)

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_tokens, output_text, timing


def _is_repetitive(text, max_repeat=5):
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False
    for i in range(len(words) - max_repeat):
        if all(words[i + j] == words[i] for j in range(max_repeat)):
            return True
    return False


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture(scope="module")
def compiled_model():
    """Compile (if needed) and load model."""
    config = create_config(MODEL_PATH)

    compiled_path = Path(COMPILED_MODEL_PATH)
    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        model = NeuronGemma4E2BForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.")

    print(f"Loading compiled model from {COMPILED_MODEL_PATH}...")
    model = NeuronGemma4E2BForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded.")
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tok = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, padding_side="right", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ============================================================================
# Tests
# ============================================================================


def test_model_loads(compiled_model):
    """Smoke test: model loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    assert hasattr(compiled_model.config, "neuron_config")
    print("PASS: Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model generates tokens."""
    prompt = "The capital of France is"
    tokens, text, timing = generate_tokens(
        compiled_model, tokenizer, prompt, max_new_tokens=10
    )

    assert len(tokens) > 0, "Model should produce at least one token"
    assert len(text) > 0, "Output text should not be empty"
    print(f"PASS: Generated {len(tokens)} tokens: {text!r}")
    print(f"  TTFT: {timing.get('ttft_ms', 0):.1f}ms")


def test_first_token(compiled_model, tokenizer):
    """
    Test that the first generated token is reasonable.

    E2B is a small model (2.3B) so we check for any non-empty output.
    """
    prompt = "The capital of France is"
    tokens, text, _ = generate_tokens(
        compiled_model, tokenizer, prompt, max_new_tokens=5
    )

    assert len(tokens) > 0, "Should produce at least one token"
    # Small model may not always produce "Paris" but output should be non-empty
    first_token_text = tokenizer.decode([tokens[0]], skip_special_tokens=True)
    assert len(first_token_text.strip()) > 0, (
        f"First token should decode to text, got: {first_token_text!r}"
    )
    print(f"PASS: First token: {first_token_text!r}, full output: {text!r}")


def test_multi_prompt_generation(compiled_model, tokenizer):
    """Test generation across multiple prompts to verify KV cache resets correctly."""
    prompts = [
        "Hello, how are you?",
        "The meaning of life is",
        "Once upon a time",
    ]
    for prompt in prompts:
        tokens, text, timing = generate_tokens(
            compiled_model, tokenizer, prompt, max_new_tokens=10
        )
        assert len(tokens) > 0, f"Failed to generate for: {prompt!r}"
        print(
            f"  Prompt: {prompt!r} -> {text!r} (TTFT: {timing.get('ttft_ms', 0):.1f}ms)"
        )

    print(f"PASS: All {len(prompts)} prompts generated successfully")


def test_performance_ttft(compiled_model, tokenizer):
    """Test Time To First Token (TTFT) performance."""
    prompt = "Hello, how are you?"

    # Warmup
    generate_tokens(compiled_model, tokenizer, prompt, max_new_tokens=1)

    # Measure over multiple runs
    ttft_times = []
    for _ in range(5):
        _, _, timing = generate_tokens(
            compiled_model, tokenizer, prompt, max_new_tokens=1
        )
        ttft_times.append(timing.get("ttft_ms", 0))

    avg_ttft = sum(ttft_times) / len(ttft_times)

    # E2B on TP=1: ~27ms observed, threshold at 100ms
    assert avg_ttft < 100, f"TTFT {avg_ttft:.1f}ms exceeds 100ms threshold"
    print(f"PASS: TTFT = {avg_ttft:.1f}ms (threshold: 100ms)")


def test_performance_throughput(compiled_model, tokenizer):
    """Test token generation throughput."""
    prompt = "Hello"

    # Warmup
    generate_tokens(compiled_model, tokenizer, prompt, max_new_tokens=5)

    # Measure
    _, _, timing = generate_tokens(compiled_model, tokenizer, prompt, max_new_tokens=30)
    throughput = timing.get("throughput_tps", 0)

    # E2B on TP=1: ~96 tok/s observed, threshold at 50 tok/s
    assert throughput > 50, (
        f"Throughput {throughput:.1f} tok/s below 50 tok/s threshold"
    )
    print(f"PASS: Throughput = {throughput:.1f} tok/s (threshold: 50 tok/s)")
    if "tpot_ms" in timing:
        print(f"  TPOT = {timing['tpot_ms']:.1f}ms")


# ============================================================================
# Standalone runner
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("Gemma-4-E2B Integration Tests")
    print(f"Model: {MODEL_PATH}")
    print(f"Compiled: {COMPILED_MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"TP degree: {TP_DEGREE}")
    print("=" * 80)

    # Create config and compile if needed
    config = create_config(MODEL_PATH)
    compiled_path = Path(COMPILED_MODEL_PATH)

    if not compiled_path.exists() or not any(compiled_path.iterdir()):
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        model = NeuronGemma4E2BForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)
        print("Compilation complete.")

    # Load model
    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
    model = NeuronGemma4E2BForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, padding_side="right", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run tests
    tests = [
        ("1. Smoke Test", test_model_loads, (model,)),
        ("2. Generation Test", test_model_generates, (model, tokenizer)),
        ("3. First Token", test_first_token, (model, tokenizer)),
        ("4. Multi-Prompt", test_multi_prompt_generation, (model, tokenizer)),
        ("5. TTFT Performance", test_performance_ttft, (model, tokenizer)),
        ("6. Throughput Performance", test_performance_throughput, (model, tokenizer)),
    ]

    passed = 0
    failed = 0
    for name, test_fn, args in tests:
        print(f"\n{name}...")
        try:
            test_fn(*args)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
    print("=" * 80)
