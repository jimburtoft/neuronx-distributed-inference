#!/usr/bin/env python3
"""Quick smoke test for Nemotron-3-Nano-30B NeuronX with existing compiled model."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer, AutoConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from modeling_nemotron_h import NeuronNemotronForCausalLM, NemotronHInferenceConfig

MODEL_PATH = "/mnt/models/nemotron-30b"
COMPILED_PATH = "/mnt/models/nemotron_compiled_v6"

neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    max_context_length=128,
    seq_len=2048,
    on_device_sampling_config=None,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    torch_dtype="bfloat16",
    save_sharded_checkpoint=True,
)

hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = NemotronHInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(hf_config=hf_config),
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading compiled model...")
model = NeuronNemotronForCausalLM(MODEL_PATH, config)
model.load(COMPILED_PATH)
print("Model loaded successfully!")

# Test 1: Generation
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    input_ids,
    attention_mask=torch.ones_like(input_ids),
    max_new_tokens=30,
    do_sample=False,
)
new_tokens = outputs[0, input_ids.shape[1] :]
text = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"Generated: {text}")

# Test 2: Coherence check
words = text.split()
is_coherent = len(words) > 3 and not any(
    all(words[i + j] == words[i] for j in range(5))
    for i in range(max(0, len(words) - 5))
)

# Test 3: First token should be Paris
first_token = tokenizer.decode([new_tokens[0].item()])
is_paris = "paris" in first_token.lower() or "Par" in first_token

print(f"\nResults:")
print(f"  Tokens generated: {len(new_tokens)}")
print(f"  First token: '{first_token}' (Paris? {is_paris})")
print(f"  Text coherent: {is_coherent}")
print(
    f"  Overall: {'PASS' if is_coherent and is_paris and len(text.strip()) > 10 else 'FAIL'}"
)
