# Contrib Model: Qwen3.5-2B

NeuronX Distributed Inference implementation of Qwen3.5-2B, a 2B parameter dense model from Alibaba Cloud with a hybrid DeltaNet + GQA attention architecture. Adapted from the Qwen3.5-27B contrib -- same architecture family at a smaller scale.

## Model Family

| Model | HuggingFace ID | Params | Instance |
|-------|----------------|--------|----------|
| **Qwen3.5-2B** | `Qwen/Qwen3.5-2B` | 2B | trn2.3xlarge (TP=4) |

**License:** Apache 2.0

## Architecture Details

| Feature | Value |
|---------|-------|
| Layers | 24 (18 DeltaNet + 6 GQA) |
| Layer Pattern | [3 DeltaNet + 1 GQA] x 6 |
| Hidden Size | 2048 |
| GQA Attention | 8 heads, 2 KV heads, head_dim=256 |
| DeltaNet Attention | 16 value heads, 16 key heads, k_dim=v_dim=128 |
| Dense MLP | SwiGLU (gate_proj + up_proj: 2048 -> 6144, down_proj: 6144 -> 2048) |
| Position Encoding | Partial RoPE (25% of head_dim = 64 dims), mRoPE for VL |
| Vocabulary | 248,320 |
| Normalization | RMSNorm with +1 weight convention |
| Activation | SiLU gated MLP |
| Tied Embeddings | Yes (lm_head shares embed_tokens weights) |

### Unique Architecture Features

- **Hybrid DeltaNet + GQA:** 18 of 24 layers use Gated DeltaNet (linear recurrent attention), 6 layers use standard GQA with KV cache. The pattern repeats every 4 layers: 3 DeltaNet + 1 GQA.
- **DeltaNet Linear Attention:** Uses the delta rule for recurrent state updates with gated decay. Per-step: `state *= exp(g); delta = (v - state^T @ k) * beta; state += outer(k, delta); output = state^T @ q`. Runs as a chunked algorithm for context encoding, per-token recurrence for token generation.
- **Custom NKI Kernels:** Three NKI kernels implement the DeltaNet forward pass on Neuron: a per-token recurrent kernel (TKG), a per-chunk kernel (legacy), and a fused single-kernel chunked forward (CTE). The fused kernel uses a Neumann series for intra-chunk correction with state persistence in SBUF across chunks.
- **GQA Output Gate:** Attention layers use a sigmoid output gate. `q_proj` is 2x sized and interleaved: `[head0_query | head0_gate | head1_query | ...]`. The gate is split during weight conversion and applied after attention.
- **Partial RoPE:** Only 25% of head_dim (64 of 256 dimensions) receives rotary embeddings. The remaining 192 dimensions are identity (no rotation).
- **+1 RMSNorm Convention:** HF weights use `output = norm(x) * (1 + weight)` where weight is initialized to zeros. Converted to standard `output = norm(x) * weight` during loading by adding 1.0 to all RMSNorm weights (except DeltaNet internal norms, which use standard convention).
- **Tied Word Embeddings:** The lm_head shares weights with embed_tokens (`tie_word_embeddings=true`). Handled automatically by NxDI's base class.
- **Vision-Language Support:** Optional 24-layer ViT encoder (1024 hidden, out_hidden=2048). Vision embeddings are injected via a scatter mask at traced input positions.

## Usage

### Text-Only (trn2.3xlarge, TP=4)

```python
import json
import os
import torch
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

from src.modeling_qwen35 import Qwen35InferenceConfig, NeuronQwen35ForCausalLM

model_path = "/path/to/Qwen3.5-2B"
compiled_path = "/scratch/qwen35_2b_traced/"

neuron_config = NeuronConfig(
    tp_degree=4,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=128,
    torch_dtype=torch.bfloat16,
    logical_nc_config=2,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    save_sharded_checkpoint=True,
)

# Read config.json directly (model_type 'qwen3_5' may not be
# registered in all transformers versions)
with open(os.path.join(model_path, "config.json")) as f:
    hf_config = json.load(f)
text_config = hf_config.get("text_config", hf_config)
config_dict = dict(text_config)
config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)

config = Qwen35InferenceConfig(
    neuron_config=neuron_config,
    **config_dict,
)

model = NeuronQwen35ForCausalLM(model_path, config)
model.compile(compiled_path)

# Reload from compiled artifacts
model = NeuronQwen35ForCausalLM(compiled_path)
model.load(compiled_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
gen_config = GenerationConfig(
    do_sample=True, top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer("The capital of France is", return_tensors="pt")
gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=50,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Chat template formatting recommended:** Qwen3.5-2B is a chat model trained with `<|im_start|>`/`<|im_end|>` formatting. Raw text prompts may produce echoey output. Use `tokenizer.apply_chat_template()` for best results:

```python
messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, padding=True, return_tensors="pt")
outputs = gen_model.generate(
    inputs.input_ids,
    generation_config=gen_config,
    attention_mask=inputs.attention_mask,
    max_new_tokens=80,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: "The capital of France is **Paris**."
```

### Vision-Language (trn2.3xlarge, TP=4)

The VL pipeline uses the text decoder on Neuron and the vision encoder on CPU:

```python
from src.modeling_qwen35_vl import NeuronQwen35VLForCausalLM, Qwen35VLInferenceConfig

vl_model = NeuronQwen35VLForCausalLM(
    model_path="/path/to/Qwen3.5-2B",
    config=vl_config,
)
vl_model.compile(compiled_path)
vl_model.load(compiled_path)

# See test/integration/test_model.py for full VL usage example
```

### DeltaNet Kernel Selection

The DeltaNet forward path can be controlled via environment variables:

| Env Var | Forward Path | Use Case |
|---------|-------------|----------|
| *(default)* | Fused chunked NKI kernel | **Default CTE path** -- required for 2B (PyTorch chunked hits compiler ICE) |
| `USE_NKI_FUSED=0` | Disable fused | Falls through to other env-var options below |
| `USE_NKI_CHUNKED=1` | Per-chunk NKI kernel | Legacy, superseded by fused |
| `USE_NKI=1` | Per-token NKI kernel | TKG (always used for token generation) |
| `DELTANET_SEQUENTIAL=1` | Sequential PyTorch | Debugging/reference |
| `USE_PYTORCH_CHUNK=1` | PyTorch chunked | **Hits compiler ICE on 2B** -- do not use |

**Note:** The PyTorch chunked forward (`_chunk_forward`) creates 5D tensors that trigger neuronx-cc codegen ICE (NCC_INLA001) with 2B dimensions. The fused NKI kernel is the default and required path.

## Benchmarks

All benchmarks on trn2.3xlarge, TP=4, LNC=2, BF16, SDK 2.29. Chat-formatted prompt ("What is the capital of France?"), ~19 input tokens. Throughput is total tokens/sec across all batch items; per-request is throughput / batch_size.

### Baseline (BS=1, seq_len=128)

| Metric | Value |
|--------|-------|
| **TTFT** | 157.8 ms |
| **Throughput** | 114.5 tok/s |
| **Compilation time** | ~2 min (TKG + CTE) |
| **Model loading** | ~9 s |
| **BF16 model size** | ~4.3 GB |

### Batch Size Scaling (seq_len=128)

| Batch Size | TTFT (ms) | Throughput (tok/s) | Per-Request (tok/s) | Correct |
|:----------:|:---------:|:------------------:|:-------------------:|:-------:|
| 1 | 157.8 | 114.5 | 114.5 | PASS |
| 2 | 72.0 | 233.1 | 116.5 | PASS |
| 4 | 104.4 | 329.6 | 82.4 | PASS |
| 8 | 185.6 | 409.5 | 51.2 | PASS |

**Analysis:** Total throughput scales well from BS=1 to BS=8 (3.6x improvement). Per-request throughput peaks at BS=2 (116.5 tok/s) and degrades at higher batch sizes due to memory bandwidth saturation. BS=2 at 72 ms TTFT is notably faster than BS=1 at 158 ms -- likely a measurement artifact from warmup effects.

### Sequence Length Scaling (BS=1)

| seq_len | TTFT (ms) | Throughput (tok/s) | Correct |
|:-------:|:---------:|:------------------:|:-------:|
| 128 | 157.8 | 114.5 | PASS |
| 512 | 54.3 | 138.1 | PASS |
| 1024 | 102.7 | 125.3 | PASS |
| 2048 | 199.7 | 106.5 | PASS |
| 4096 | 401.7 | 80.3 | PASS |

**Analysis:** TTFT scales roughly linearly with seq_len (as expected -- CTE processes the full context). Throughput (TKG) is largely unaffected by seq_len for DeltaNet layers (O(1) per step), but GQA layers have KV cache that grows with seq_len. The seq_len=512 TTFT of 54 ms is lower than seq_len=128 because the model graph is compiled once and the actual prompt is only 19 tokens -- the difference is compilation/graph characteristics.

### Combined Configurations

| BS | seq_len | TTFT (ms) | Throughput (tok/s) | Per-Request (tok/s) | Correct |
|:--:|:-------:|:---------:|:------------------:|:-------------------:|:-------:|
| 2 | 1024 | 222.4 | 182.8 | 91.4 | PASS |
| 4 | 512 | 227.2 | 280.8 | 70.2 | PASS |

**Note on `seq_len`:** The `seq_len` parameter is the **total sequence budget** (input + generated tokens). Do not pad inputs to `max_length=seq_len` -- this leaves no room for generation. Use `padding=True` for automatic minimal padding.

## Caveats

1. **SDK 2.29+ required:** The NKI DeltaNet kernels require NKI 0.3.0 (SDK 2.29). No library modifications needed -- runs on stock SDK 2.29 DLAMI (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`).

2. **Chat template required for quality output:** Qwen3.5-2B is a chat model. Raw text prompts (e.g., `"The capital of France is"`) produce echoey/repetitive output. Always use `tokenizer.apply_chat_template()` for correct responses.

3. **No mini model test:** DeltaNet layers require NKI kernels that only execute on Neuron devices. Integration tests require a trn2 instance with model weights.

4. **+1 RMSNorm convention:** Qwen3.5 uses `output = norm(x) * (1 + weight)` for most RMSNorm layers, but DeltaNet internal norms use standard `output = norm(x) * weight`. The weight conversion handles this automatically, but custom weight loading must be aware of both conventions.

5. **Neumann series convergence:** The fused DeltaNet kernel uses a 6-round Neumann series for intra-chunk correction. This requires L2-normalized Q and K inputs. Unnormalized inputs will cause NaN divergence.

6. **DeltaNet state persistence:** DeltaNet recurrent and conv1d state buffers persist in HBM via `input_output_aliases`, the same mechanism used for KV cache. Both CTE and TKG share the same JIT ScriptModule, so state is automatically shared. The `_copy_past_key_values` override is only used for CPU execution paths (not on Neuron hardware).

## Compatibility Matrix

| Instance | TP | LNC | Notes |
|----------|-----|-----|-------|
| trn2.3xlarge | 4 | 2 | Recommended -- 2B model fits with large HBM headroom |
| inf2.8xlarge | 2 | - | Possible -- model is small enough |

### Tested Configurations (trn2.3xlarge, TP=4, LNC=2)

| Batch Size | seq_len | Status | Notes |
|:----------:|:-------:|:------:|-------|
| 1 | 128 | Verified | Baseline config |
| 2 | 128 | Verified | Best per-request throughput |
| 4 | 128 | Verified | Good total throughput |
| 8 | 128 | Verified | Highest total throughput |
| 1 | 512 | Verified | |
| 1 | 1024 | Verified | |
| 1 | 2048 | Verified | |
| 1 | 4096 | Verified | Long context |
| 2 | 1024 | Verified | Multi-user serving config |
| 4 | 512 | Verified | High-throughput short context |

### SDK Configuration

| Component | Version |
|-----------|---------|
| NxDI | 0.9.17334+ |
| neuronx-cc | 2.24+ |
| torch | 2.9+ |
| NKI | 0.3.0+ |
| NXDI venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |

## Testing

### Unit Tests (CPU only, no device needed)

```bash
cd contrib/models/Qwen3.5-2B/
# On DLAMI: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pytest test/unit/ -v
```

### Integration Tests (needs trn2 with NeuronCores)

```bash
cd contrib/models/Qwen3.5-2B/

QWEN35_MODEL_PATH=/mnt/models/Qwen3.5-2B \
QWEN35_COMPILED_PATH=/mnt/models/qwen35_2b_traced \
pytest test/integration/test_model.py --capture=tee-sys
```

## Key Porting Notes

This contrib is adapted from the Qwen3.5-27B contrib. The core DeltaNet NKI kernels, weight conversion logic, and hybrid state management are identical -- only dimensions and layer counts differ.

| Parameter | Qwen3.5-2B | Qwen3.5-27B |
|-----------|-----------|-------------|
| hidden_size | 2048 | 5120 |
| num_hidden_layers | 24 | 64 |
| num_attention_heads | 8 | 24 |
| num_key_value_heads | 2 | 4 |
| intermediate_size | 6144 | 17408 |
| DeltaNet value_heads | 16 | 48 |
| DeltaNet key_heads | 16 | 16 |
| Vision depth | 24 | 27 |
| Vision hidden | 1024 | 1152 |
| Vision out_hidden | 2048 | 5120 |
| tie_word_embeddings | true | false |
| BF16 model size | ~4 GB | ~52 GB |

## Example Checkpoints

- `Qwen/Qwen3.5-2B` (BF16, ~4 GB)

## Maintainer

AWS Neuron

**Last Updated:** 2026-04-23
