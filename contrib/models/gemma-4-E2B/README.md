# Contrib Model: Gemma 4 E2B

NeuronX Distributed Inference implementation of Google's Gemma 4 E2B, a 2.3B-effective-parameter model with Per-Layer Embeddings (PLE) and KV cache sharing.

## Model Information

- **HuggingFace ID:** [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B)
- **Model Type:** Text decoder with PLE and KV sharing
- **Parameters:** 5.1B total, 2.3B effective (via KV sharing)
- **License:** Check HuggingFace model card

## Architecture Details

| Feature | Description |
|---------|-------------|
| **hidden_size** | 1536 |
| **num_hidden_layers** | 35 |
| **num_attention_heads** | 8 |
| **num_key_value_heads** | 1 (all layers, GQA) |
| **Heterogeneous layers** | SWA layers (head_dim=256) and Global layers (head_dim=512) |
| **sliding_window** | 512 (SWA layers only) |
| **intermediate_size** | 6144 (layers 0-14), 12288 (layers 15-34) |
| **Layer pattern** | Every 5th layer is global (4, 9, 14, 19, 24, 29, 34) |
| **QK normalization** | RMSNorm with learnable scale on Q and K |
| **Partial RoPE** | Global layers: `partial_rotary_factor=0.25` (128 of 512 dims rotated) |
| **Proportional RoPE** | Global layers use `1/(theta^(i/head_dim))` frequency computation |
| **4-norm decoder** | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` |
| **Scaled embeddings** | `embed * sqrt(hidden_size)` |
| **final_logit_softcapping** | `30 * tanh(logits / 30)` after lm_head |
| **layer_scalar** | Per-layer learned multiplicative factor |

### Per-Layer Embeddings (PLE)

PLE is a novel feature where each token gets an additional per-layer embedding vector:

| Property | Value |
|----------|-------|
| PLE dimension | 256 per layer |
| Total PLE dim | 35 x 256 = 8,960 |
| PLE vocab size | 262,144 |
| PLE embedding storage | bfloat16 (~4.6 GB) |

PLE flow per decoder layer:
1. Lookup: `embed_tokens_per_layer(input_ids) * sqrt(ple_dim)` -> reshape per layer
2. Project: `per_layer_model_projection(main_embeds) * (1/sqrt(hidden_size))` -> RMSNorm
3. Combine: `(normed_projection + ple_lookup) * rsqrt(2)`
4. In decoder layer: `act_fn(gate(hidden_states)) * per_layer_input` -> project -> norm -> residual add

### KV Cache Sharing

Layers 15-34 (20 shared layers) reuse KV from donor layers instead of computing their own:

| Property | Value |
|----------|-------|
| Shared layers | 15-34 (20 layers) |
| Donor for SWA shared layers | Layer 13 |
| Donor for Global shared layers | Layer 14 |
| Implementation | Donor K/V passed to shared layers; shared layers only compute Q |

This is critical for correctness: shared layers' own K/V projection weights produce random/uncorrelated outputs (cosine similarity ~0 with donor weights).

## Validation Results

**Validated:** 2026-04-07
**Configuration:** TP=1, batch_size=1, bfloat16, trn2.3xlarge (LNC=2)

### Accuracy

| Test | Cosine vs CPU Reference | Top-1 Match | Status |
|------|------------------------|-------------|--------|
| BOS-only, no PLE | 0.999999 | Yes | PASS |
| BOS-only, with PLE | 1.000004 | Yes | PASS |
| Multi-token, no PLE | 0.9945 | Yes | PASS |
| Multi-token, with PLE (f32) | 0.99999988 | Yes | PASS |
| f32 + PLE + KV sharing | 1.00000346 | Yes | PASS |
| **bf16 + PLE + KV sharing** | **0.99999905** | Yes | **PASS** |

### Performance (bf16, TP=1, batch=1)

| Metric | Value |
|--------|-------|
| TTFT (CTE, 128 bucket) | 27.3 ms |
| TPOT (TKG) | 10.4 ms |
| Throughput | 96 tok/s |
| Multi-prompt generation (6 x 30 tokens) | All pass |

## Prerequisites

- **Neuron SDK 2.28+** (DLAMI: `Deep Learning AMI Neuron (Ubuntu 24.04) 20260227`)
- **NxDI 0.8.0+** with transformers 4.57.6+
- **Instance:** trn2.3xlarge or larger (LNC=2)

## Usage

```python
import json
import os
import torch
from transformers import AutoTokenizer

# Apply NxDI patches BEFORE importing model classes
from src.ndxi_patch import apply_patch
apply_patch()

from src.modeling_gemma4_e2b import (
    NeuronGemma4E2BForCausalLM,
    Gemma4E2BInferenceConfig,
    Gemma4E2BNeuronConfig,
)

model_path = "/path/to/gemma-4-E2B"
compiled_path = "/path/to/compiled"

# Configure
neuron_config = Gemma4E2BNeuronConfig(
    tp_degree=1,
    batch_size=1,
    max_batch_size=1,
    max_length=512,
    seq_len=128,
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

config = Gemma4E2BInferenceConfig(
    neuron_config=neuron_config,
    load_config=load_config_fn,
)

# Compile (first time only)
model = NeuronGemma4E2BForCausalLM(model_path, config)
model.compile(compiled_path)

# Load onto Neuron
model = NeuronGemma4E2BForCausalLM(model_path, config)
model.load(compiled_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
seq_len = input_ids.shape[1]
bucket = 128

padded = torch.nn.functional.pad(input_ids, (0, bucket - seq_len), value=0)
attn_mask = torch.zeros(1, bucket, dtype=torch.int32)
attn_mask[0, :seq_len] = 1
pos_ids = torch.zeros(1, bucket, dtype=torch.int32)
pos_ids[0, :seq_len] = torch.arange(seq_len, dtype=torch.int32)
seq_ids = torch.tensor([0], dtype=torch.int32)

model.reset_kv_cache()

# Prefill
with torch.inference_mode():
    output = model(
        input_ids=padded,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        seq_ids=seq_ids,
    )

next_token = torch.argmax(output.logits[0, 0, :]).item()
generated = [next_token]
cur_pos = seq_len

# Token generation loop
max_length = 512
for _ in range(29):
    tok_input = torch.tensor([[next_token]], dtype=torch.int64)
    tok_pos = torch.tensor([[cur_pos]], dtype=torch.int32)
    tok_attn = torch.ones(1, max_length, dtype=torch.int32)
    tok_attn[0, cur_pos + 1:] = 0

    with torch.inference_mode():
        output = model(
            input_ids=tok_input,
            attention_mask=tok_attn,
            position_ids=tok_pos,
            seq_ids=seq_ids,
        )
    next_token = torch.argmax(output.logits[0, 0, :]).item()
    generated.append(next_token)
    cur_pos += 1
    if next_token == tokenizer.eos_token_id:
        break

print(tokenizer.decode(generated, skip_special_tokens=True))
```

See `test/integration/test_model.py` for a complete example with multiple prompts and performance measurement.

## Compatibility Matrix

| Instance Type | TP=1 | TP=4 |
|---------------|------|------|
| trn2.3xlarge (LNC=2) | **VALIDATED** (bf16) | Should work (untested) |
| trn2.48xlarge | Not tested | Not tested |
| Inf2 | N/A | N/A |

**Notes:**
- TP=1 is recommended for E2B (small model, 2.3B effective).
- `attn_kernel_enabled=False` required (head_dim 256/512 exceeds NxDI kernel limit of 128).
- `weights_to_skip_layout_optimization=[r".*embed_tokens_per_layer.*"]` required for bf16 PLE embedding.

## Testing

```bash
# Set environment variables
export GEMMA4_E2B_MODEL_PATH=/mnt/models/gemma-4-E2B
export GEMMA4_E2B_COMPILED_PATH=/mnt/models/gemma4-e2b-compiled
export GEMMA4_E2B_TOKENIZER_PATH=/mnt/models/gemma-4-E2B  # or path with tokenizer
export GEMMA4_E2B_TP_DEGREE=1

# Run with pytest
pytest nxdi_contrib_models/models/gemma-4-E2B/test/integration/test_model.py --capture=tee-sys

# Or run standalone
cd nxdi_contrib_models/models/gemma-4-E2B
python test/integration/test_model.py
```

## Known Limitations

- **Text-only decoder:** Vision and audio encoders are not yet implemented. The 31B contrib includes vision support which can be adapted.
- **No bidirectional vision attention:** Same limitation as 31B — standard causal masking for all tokens.
- **Greedy decode repetition:** Small models (2.3B) tend to produce repetitive output with greedy decoding. Use sampling (top-k, temperature) for better quality.
- **NxDI 0.8.0 monkey-patches:** Requires `ndxi_patch.py` to be applied before model creation due to API changes in NxDI 0.8.0.
- **PLE embedding in bf16:** The PLE embedding table is stored in bf16 to fit in HBM (~4.6 GB vs ~9.2 GB in f32). This has negligible accuracy impact (cosine > 0.99999).

## Related Models

- **[gemma-4-31b-it](../gemma-4-31b-it/)** — Full 31B model with text + vision support, NKI flash attention kernel

## Maintainer

Community contribution

**Last Updated:** 2026-04-07
