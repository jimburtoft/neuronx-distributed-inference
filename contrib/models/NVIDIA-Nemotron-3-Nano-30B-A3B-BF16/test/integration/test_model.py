#!/usr/bin/env python3
"""
Integration tests for NVIDIA Nemotron-3-Nano-30B-A3B-BF16 NeuronX implementation.

Tests model compilation, loading, and inference accuracy/performance.
This model is a hybrid Mamba2/Attention/MoE architecture (52 layers:
23 Mamba-2, 23 MoE, 6 GQA Attention) requiring Mamba state persistence
and MoE TP sharding across decode steps.

Tested on: trn2.3xlarge (TP=4, LNC=2, SDK 2.28)

Usage:
    # Run with pytest
    cd contrib/
    pytest test/integration/test_model.py -v

    # Or run directly (compiles if needed)
    python test/integration/test_model.py
"""

import pytest
import time
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)

# Import from src directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_nemotron_h import NeuronNemotronForCausalLM, NemotronHInferenceConfig


# Test configuration — UPDATE THESE PATHS for your environment
MODEL_PATH = "/mnt/models/nemotron-30b"
COMPILED_MODEL_PATH = "/mnt/models/nemotron_compiled_contrib"

# Compilation parameters (trn2.3xlarge with LNC=2 -> 4 logical cores)
TP_DEGREE = 4
BATCH_SIZE = 1
MAX_CONTEXT_LENGTH = 128
SEQ_LENGTH = 2048


def create_config(model_path: str):
    """Create NemotronHInferenceConfig."""
    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        max_context_length=MAX_CONTEXT_LENGTH,
        seq_len=SEQ_LENGTH,
        on_device_sampling_config=None,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        torch_dtype="bfloat16",
        save_sharded_checkpoint=True,
    )

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = NemotronHInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    return config


@pytest.fixture(scope="module")
def compiled_model():
    """Compile and load model."""
    compiled_path = Path(COMPILED_MODEL_PATH)
    if not (compiled_path / "model.pt").exists():
        print(f"Compiling model to {COMPILED_MODEL_PATH}...")
        config = create_config(MODEL_PATH)
        model = NeuronNemotronForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tokenizer.save_pretrained(COMPILED_MODEL_PATH)

    # Load compiled model
    config = create_config(MODEL_PATH)
    model = NeuronNemotronForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_model_loads(compiled_model):
    """Smoke test: model loads successfully."""
    assert compiled_model is not None
    assert hasattr(compiled_model, "config")
    print("PASS: Model loaded successfully")


def test_model_generates(compiled_model, tokenizer):
    """Test that model generates text (prefill + decode)."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_model = HuggingFaceGenerationAdapter(compiled_model)
    outputs = gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=20,
        do_sample=False,
    )

    new_tokens = outputs[0, inputs.input_ids.shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    assert len(output_text.strip()) > 0, "Should generate non-empty text"
    assert len(new_tokens) == 20, "Should generate exactly 20 new tokens"
    print(f"PASS: Generated '{output_text}'")


def test_first_token_correct(compiled_model, tokenizer):
    """Test that first greedy token for 'The capital of France is' is Paris-related."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_model = HuggingFaceGenerationAdapter(compiled_model)
    outputs = gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=1,
        do_sample=False,
    )

    first_token_id = outputs[0, inputs.input_ids.shape[1] :][0].item()
    first_token = tokenizer.decode([first_token_id])

    print(f"  First greedy token: '{first_token}' (id={first_token_id})")

    # Should not be EOS or padding
    assert first_token_id != 0, "Token ID 0 indicates generation failure"
    assert first_token_id != tokenizer.eos_token_id, (
        "Should not immediately produce EOS"
    )

    # The first token should be "Paris" or " Paris" (with or without leading space)
    assert "paris" in first_token.lower() or "Par" in first_token, (
        f"Expected Paris-related token, got '{first_token}'"
    )
    print(f"PASS: First token is '{first_token}' (correct)")


def test_output_coherence(compiled_model, tokenizer):
    """Test that output is coherent (not repetitive gibberish)."""
    prompt = "Albert Einstein was born in"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_model = HuggingFaceGenerationAdapter(compiled_model)
    outputs = gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=30,
        do_sample=False,
    )

    new_tokens = outputs[0, inputs.input_ids.shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    words = output_text.split()
    assert len(words) > 3, "Output should have multiple words"
    assert not _is_repetitive(output_text), (
        f"Output should not be repetitive: '{output_text}'"
    )
    print(f"PASS: Coherent output '{output_text[:100]}...'")


def test_multiple_prompts(compiled_model, tokenizer):
    """Test generation across diverse prompts."""
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "def fibonacci(n):",
        "Hello, how are you today?",
    ]

    gen_model = HuggingFaceGenerationAdapter(compiled_model)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = gen_model.generate(
            inputs.input_ids,
            attention_mask=torch.ones_like(inputs.input_ids),
            max_new_tokens=10,
            do_sample=False,
        )
        new_tokens = outputs[0, inputs.input_ids.shape[1] :]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        assert len(output_text.strip()) > 0, f"Empty output for '{prompt}'"
        print(f"  '{prompt}' -> '{output_text}'")

    print("PASS: All prompts generated text")


def _is_repetitive(text: str, max_repeat: int = 5) -> bool:
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False

    for i in range(len(words) - max_repeat):
        word = words[i]
        if all(words[i + j] == word for j in range(max_repeat)):
            return True

    # Check character-level repetition
    if len(text) > 20:
        tail = text[-100:] if len(text) > 100 else text
        char_counts = {}
        for c in tail:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_ratio = max(char_counts.values()) / len(tail)
        if max_ratio > 0.5:
            return True

    return False


def test_performance_throughput(compiled_model, tokenizer):
    """Measure token generation throughput."""
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    num_tokens = 50

    gen_model = HuggingFaceGenerationAdapter(compiled_model)

    # Warmup
    gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=5,
        do_sample=False,
    )

    start = time.perf_counter()
    gen_model.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=num_tokens,
        do_sample=False,
    )
    elapsed = time.perf_counter() - start

    throughput = num_tokens / elapsed
    print(
        f"PASS: Throughput = {throughput:.1f} tok/s ({elapsed:.2f}s for {num_tokens} tokens)"
    )

    # Should be at least 10 tok/s on trn2.3xlarge
    assert throughput > 10, (
        f"Throughput {throughput:.1f} tok/s is below minimum (10 tok/s)"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Nemotron-3-Nano-30B-A3B-BF16 Integration Tests")
    print("=" * 80)

    compiled_path = Path(COMPILED_MODEL_PATH)
    config = create_config(MODEL_PATH)

    if not (compiled_path / "model.pt").exists():
        print(f"\nCompiling model to {COMPILED_MODEL_PATH}...")
        model = NeuronNemotronForCausalLM(MODEL_PATH, config)
        model.compile(COMPILED_MODEL_PATH)

        tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tok.save_pretrained(COMPILED_MODEL_PATH)
        print("Compilation complete")

    # Load model
    print(f"\nLoading compiled model from {COMPILED_MODEL_PATH}...")
    model = NeuronNemotronForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_MODEL_PATH)
    print("Model loaded")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run tests
    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    print("\n1. Smoke Test...")
    test_model_loads(model)

    print("\n2. Generation Test...")
    test_model_generates(model, tokenizer)

    print("\n3. First Token Accuracy...")
    test_first_token_correct(model, tokenizer)

    print("\n4. Coherence Test...")
    test_output_coherence(model, tokenizer)

    print("\n5. Multiple Prompts...")
    test_multiple_prompts(model, tokenizer)

    print("\n6. Throughput...")
    test_performance_throughput(model, tokenizer)

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
