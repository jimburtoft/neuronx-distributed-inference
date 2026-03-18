#!/usr/bin/env python3
"""
Integration tests for Qwen3-VL-30B-A3B-Instruct NeuronX implementation.

Prerequisites:
  - trn2.3xlarge with Neuron SDK 2.28 (LNC=2, 4 logical cores)
  - source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
  - NEURON_RT_NUM_CORES=4, NEURON_PLATFORM_TARGET_OVERRIDE=trn2
  - HF model downloaded to MODEL_PATH
  - Model compiled to COMPILED_MODEL_PATH (using run_inference.py --compile-only
    or the compile fixture below)

Usage:
  # Run all tests (assumes pre-compiled model):
  pytest test_model.py -v -s

  # Run only text tests:
  pytest test_model.py -v -s -k "text"

  # Run only vision test:
  pytest test_model.py -v -s -k "vision"
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoProcessor, AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig, NeuronConfig
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Import from src directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from modeling_qwen3_vl_moe import (
    NeuronQwen3VLMoeForCausalLM,
    Qwen3VLMoeNeuronConfig,
    Qwen3VLMoeVLInferenceConfig,
)

# ---------------------------------------------------------------------------
# Paths -- override with environment variables if needed
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "QWEN3VLMOE_MODEL_PATH", "/home/ubuntu/models/Qwen3-VL-30B-A3B-Instruct"
)
COMPILED_MODEL_PATH = os.environ.get(
    "QWEN3VLMOE_COMPILED_PATH", "/home/ubuntu/neuron_models/Qwen3-VL-30B-A3B-Instruct"
)


# ---------------------------------------------------------------------------
# Config creation (mirrors run_inference.py)
# ---------------------------------------------------------------------------
def create_inference_config(model_path: str) -> Qwen3VLMoeVLInferenceConfig:
    """Create Qwen3VLMoeVLInferenceConfig for trn2.3xlarge (TP=4, LNC=2)."""
    DTYPE = torch.bfloat16
    TP_DEGREE = 4

    text_neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        ctx_batch_size=1,
        seq_len=2176,
        max_context_length=2048,
        max_new_tokens=128,
        torch_dtype=DTYPE,
        attention_dtype=DTYPE,
        rpl_reduce_dtype=DTYPE,
        on_device_sampling_config=None,
        moe_tp_degree=TP_DEGREE,
        moe_ep_degree=1,
        glu_mlp=True,
        blockwise_matmul_config={"block_size": 32768},
        enable_bucketing=True,
        buckets=[128, 512, 2048, 2176],
        context_encoding_buckets=[128, 512, 2048],
        token_generation_buckets=[128, 512, 2048, 2176],
        fused_qkv=True,
        qkv_kernel_enabled=True,
        attn_kernel_enabled=True,
        mlp_kernel_enabled=False,
        sequence_parallel_enabled=False,
        save_sharded_checkpoint=True,
        cc_pipeline_tiling_factor=2,
        logical_neuron_cores=2,
        cast_type="as-declared",
    )

    vision_neuron_config = Qwen3VLMoeNeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=1,
        ctx_batch_size=1,
        seq_len=4096,
        torch_dtype=DTYPE,
        attention_dtype=DTYPE,
        rpl_reduce_dtype=DTYPE,
        fused_qkv=True,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
        mlp_kernel_enabled=False,
        sequence_parallel_enabled=False,
        enable_bucketing=True,
        buckets=[1024, 4096],
        save_sharded_checkpoint=True,
        cc_pipeline_tiling_factor=2,
        logical_neuron_cores=2,
        cast_type="as-declared",
    )

    return Qwen3VLMoeVLInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prefill_and_decode(
    model,
    input_ids,
    attention_mask,
    position_ids,
    max_new_tokens,
    eos_ids=None,
    pixel_values=None,
    image_grid_thw=None,
):
    """Run manual prefill + decode loop. Returns (all_ids, prefill_time, decode_time)."""
    if eos_ids is None:
        eos_ids = {151643, 151645}  # Qwen EOS tokens

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Prefill
    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        sampling_params=prepare_sampling_params(batch_size),
    )
    if pixel_values is not None:
        prefill_kwargs["pixel_values"] = pixel_values
        prefill_kwargs["image_grid_thw"] = image_grid_thw

    t0 = time.perf_counter()
    outputs = model(**prefill_kwargs)
    prefill_time = time.perf_counter() - t0

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    all_ids = torch.cat([input_ids, next_token], dim=-1)

    # Decode
    t0 = time.perf_counter()
    for step in range(max_new_tokens - 1):
        cur_pos = seq_len + step + 1
        new_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
        pos_ids = torch.tensor([[cur_pos - 1]] * batch_size, dtype=torch.long)

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            sampling_params=prepare_sampling_params(batch_size),
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_ids = torch.cat([all_ids, next_token], dim=-1)

        if next_token[0, 0].item() in eos_ids:
            break

    decode_time = time.perf_counter() - t0
    return all_ids, prefill_time, decode_time


def _is_repetitive(text: str, max_repeat: int = 5) -> bool:
    """Check if text has excessive repetition."""
    words = text.split()
    if len(words) < 10:
        return False
    for i in range(len(words) - max_repeat):
        if all(words[i + j] == words[i] for j in range(max_repeat)):
            return True
    tail = text[-100:] if len(text) > 100 else text
    if len(tail) > 20:
        counts = {}
        for c in tail:
            counts[c] = counts.get(c, 0) + 1
        if max(counts.values()) / len(tail) > 0.5:
            return True
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    """Load pre-compiled Qwen3-VL-30B-A3B-Instruct model."""
    config = create_inference_config(MODEL_PATH)
    mdl = NeuronQwen3VLMoeForCausalLM(model_path=MODEL_PATH, config=config)
    mdl.load(COMPILED_MODEL_PATH)
    return mdl


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def processor():
    """Load processor (tokenizer + image processor)."""
    return AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_model_loads(model):
    """Smoke test: model loads and has expected attributes."""
    assert model is not None
    assert hasattr(model, "config")
    assert model.config.text_config.num_hidden_layers == 48
    assert model.config.text_config.num_experts == 128
    assert model.config.text_config.num_experts_per_tok == 8
    print("PASS: Model loaded successfully")


def test_text_generation(model, tokenizer):
    """Test text-only generation produces coherent output."""
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    model.reset()
    all_ids, prefill_time, decode_time = _prefill_and_decode(
        model,
        input_ids,
        attention_mask,
        position_ids,
        max_new_tokens=20,
    )
    output_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)

    assert len(output_text) > len(prompt), "Output should be longer than prompt"
    assert not _is_repetitive(output_text), "Output should not be repetitive"
    gen_tokens = all_ids.shape[1] - seq_len
    print(f"PASS: Text generation - '{output_text}'")
    print(
        f"  Prefill: {prefill_time * 1000:.0f}ms, Decode: {gen_tokens} tokens in {decode_time:.2f}s"
    )


def test_text_coherence(model, tokenizer):
    """Test that multi-prompt outputs are coherent and non-degenerate."""
    prompts = ["1 + 1 =", "Hello, how are you?"]
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        model.reset()
        all_ids, _, _ = _prefill_and_decode(
            model,
            input_ids,
            attention_mask,
            position_ids,
            max_new_tokens=30,
        )
        output_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)

        assert len(output_text.split()) > 2, (
            f"Output too short for '{prompt}': '{output_text}'"
        )
        assert not _is_repetitive(output_text), f"Repetitive output for '{prompt}'"
        print(f"PASS: Coherence - '{prompt}' -> '{output_text[:80]}...'")


def test_vision_text_generation(model, processor):
    """Test vision+text generation with a synthetic red image."""
    from PIL import Image
    import numpy as np

    # Create a 224x224 solid red image
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:, :, 0] = 255
    test_image = Image.fromarray(img_array)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What color is this image?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[test_image], return_tensors="pt")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    model.reset()
    all_ids, prefill_time, decode_time = _prefill_and_decode(
        model,
        input_ids,
        attention_mask,
        position_ids,
        max_new_tokens=30,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        eos_ids={processor.tokenizer.eos_token_id, 151643, 151645},
    )
    output_text = processor.decode(all_ids[0], skip_special_tokens=True)

    gen_tokens = all_ids.shape[1] - seq_len
    assert gen_tokens > 0, "Model should generate at least one token"
    assert "red" in output_text.lower(), f"Expected 'red' in output: '{output_text}'"
    print(f"PASS: Vision+text - '{output_text}'")
    print(
        f"  VL Prefill: {prefill_time * 1000:.0f}ms, Decode: {gen_tokens} tokens in {decode_time:.2f}s"
    )


def test_throughput(model, tokenizer):
    """Measure decode throughput (tokens/sec). Expected: ~95-99 tok/s with ISA kernels."""
    prompt = "Explain quantum computing in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Warmup run
    model.reset()
    _prefill_and_decode(
        model, input_ids, attention_mask, position_ids, max_new_tokens=5
    )

    # Measured run
    model.reset()
    num_tokens = 50
    all_ids, prefill_time, decode_time = _prefill_and_decode(
        model,
        input_ids,
        attention_mask,
        position_ids,
        max_new_tokens=num_tokens,
    )

    gen_tokens = all_ids.shape[1] - seq_len
    throughput = gen_tokens / decode_time if decode_time > 0 else 0

    assert throughput > 50, f"Throughput too low: {throughput:.1f} tok/s (expected >50)"
    print(
        f"PASS: Throughput - {throughput:.1f} tok/s ({gen_tokens} tokens in {decode_time:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Qwen3-VL-30B-A3B-Instruct Integration Tests")
    print("=" * 70)
    print(f"  Model path:    {MODEL_PATH}")
    print(f"  Compiled path: {COMPILED_MODEL_PATH}")
    print()
    print("Run with: pytest test_model.py -v -s")
    print("=" * 70)
