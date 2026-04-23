# Contrib Model: GLM-5

NeuronX Distributed Inference implementation of [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5).

## Model Information

- **HuggingFace ID:** `zai-org/GLM-5` (BF16), `zai-org/GLM-5-FP8` (E4M3 quantized)
- **Model Type:** Mixture of Experts (MoE) with Multi-head Latent Attention (MLA)
- **Parameters:** 754B total (40B active per token)
- **License:** MIT

## Architecture Details

- **Layers:** 78 decoder layers (3 dense + 75 MoE)
- **Hidden Size:** 6,144
- **Attention Heads:** 64 (MLA with compressed KV cache)
- **q_lora_rank:** 2,048
- **kv_lora_rank:** 512 (576 total with RoPE: 512 compressed + 64 rope)
- **qk_nope_head_dim:** 192
- **qk_rope_head_dim:** 64
- **v_head_dim:** 256
- **head_dim (output):** 64
- **Experts:** 256 routed + 1 shared per MoE layer
- **Active Experts:** 8 per token (top-8 sigmoid routing)
- **MoE Intermediate Size:** 2,048 (per expert)
- **Dense Intermediate Size:** 12,288 (layers 0-2 only)
- **Vocabulary:** 154,880 tokens
- **Max Position Embeddings:** 202,752
- **Position Encoding:** RoPE (theta=1,000,000, no YaRN)
- **Normalization:** RMSNorm
- **Activation:** SiLU (SwiGLU gating)
- **Router:** Sigmoid with e_score_correction_bias, routed_scaling_factor=2.5
- **DSA (DeepSeek Sparse Attention):** Present in weights, **skipped** (full-attention fallback)
- **MTP (Multi-Token Prediction):** Training-only, **skipped**

## Status

**Status:** IN DEVELOPMENT -- modeling code written, awaiting first compile/validation.

## Required Instance

- **trn2.48xlarge** with TP=64, EP=1, LNC=2
- FP8 weights required (BF16 does not fit: 1.49 TB / 1.54 TB = 96.8% HBM)
- FP8 weights: 746 GB / 1,536 GB = 48.6% HBM utilization
- trn2.3xlarge is NOT feasible (746 GB / 96 GB = 7.8x deficit)

## Usage

```python
import json
import torch
from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig

from src.modeling_glm5 import GLM5InferenceConfig, NeuronGLM5ForCausalLM

model_path = "/path/to/GLM-5-FP8"
compiled_path = "/path/to/compiled/"

# Load HuggingFace config
with open(f"{model_path}/config.json") as f:
    hf_config = json.load(f)

# Configure Neuron
neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    seq_len=4096,
    n_active_tokens=4096,
    torch_dtype=torch.bfloat16,
    # MLA: no fused QKV (separate q_a/q_b/kv_a/kv_b projections)
    fused_qkv=False,
    # Attention NKI kernels
    qkv_kernel_enabled=True,
    qkv_nki_kernel_enabled=True,
    # MoE NKI kernels disabled (2048/64=32, 32%128!=0)
    moe_fused_nki_kernel_enabled=False,
    expert_mlp_nki_kernel_enabled=False,
)

# Create inference config
config = GLM5InferenceConfig(
    neuron_config=neuron_config,
    load_config=lambda c: [setattr(c, k, v) for k, v in hf_config.items()],
)

# Compile, load, and generate
model = NeuronGLM5ForCausalLM(model_path, config)
model.compile(compiled_model_path=compiled_path)
model.load(compiled_path)

# Run inference
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = "The meaning of life is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
seq_len = input_ids.shape[1]

output = model.forward(
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    position_ids=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
    seq_ids=torch.zeros(1, dtype=torch.int32),
)

logits = (output.logits if hasattr(output, "logits") else output[0])[0, -1, :]
next_token = torch.argmax(logits).item()
print(f"Next token: {tokenizer.decode([next_token])}")
```

## Architecture Notes

GLM-5 is architecturally identical to DeepSeek-V3. vLLM implements it as an empty subclass:
```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass
```

### Key Differences from DeepSeek-V3

| Property | GLM-5 | DeepSeek-V3 |
|----------|-------|-------------|
| qk_nope_head_dim | 192 | 128 |
| v_head_dim | 256 | 128 |
| head_dim (output) | 64 | 128 |
| q_lora_rank | 2,048 | 1,536 |
| hidden_size | 6,144 | 7,168 |
| Layers | 78 (3 dense) | 61 (1 dense) |
| rope_theta | 1,000,000 | 10,000,000 |
| routed_scaling_factor | 2.5 | 2.827 |
| DSA | Yes | No |

### Implementation Decisions

1. **MLA Weight Absorption Fix**: The NxDI DS-V3 code's absorption slicing works only when `qk_nope_head_dim == v_head_dim` (both 128 in DS-V3). For GLM-5 we use the correct slicing: `q_absorb = wkv_b[:, :192]`, `v_absorb = wkv_b[:, 192:]`.

2. **routed_scaling_factor=2.5**: Handled in patched router forward. We set `normalize_top_k_affinities=False` and compute normalization + scaling ourselves (gather, normalize to sum=1, multiply by 2.5, scatter back).

3. **FP8 NaN clamping**: PyTorch `float8_e4m3fn` max=448, but Neuron treats exponent-15 as NaN. We clamp to 240 before dequantization.

4. **DSA skipped**: Full-attention fallback. DSA indexer weights are stripped during weight conversion.

5. **MTP skipped**: Layer 78 is training-only. Weights stripped during conversion.

### Known Limitations

- **MoE NKI kernels disabled:** `moe_intermediate/TP = 2048/64 = 32`, and `32 % 128 != 0`. Falls back to ISA router.
- **DSA not implemented:** Full attention used for all context lengths. This is correct but suboptimal for >4K tokens.
- **MTP not implemented:** No speculative decoding support.

## Testing Instructions

Run on a trn2.48xlarge instance:

```bash
# Activate NxDI venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Set environment variables
export GLM5_MODEL_PATH=/mnt/models/GLM-5-FP8
export GLM5_COMPILED_PATH=/mnt/models/glm5_compiled

# Run with pytest
pytest contrib/models/GLM-5/test/integration/test_model.py -v -s

# Or run directly
cd contrib/models/GLM-5
python3 test/integration/test_model.py
```

## Example Checkpoints

* [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) (BF16, 1.51 TB)
* [zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8) (E4M3, ~756 GB) -- **required for Neuron**

## SDK Requirements

- Neuron SDK 2.28+ (torch-neuronx 2.9.0, NxDI 0.8.0+)
- trn2.48xlarge instance with 64 Neuron cores (LNC=2)
- FP8 model weights (`zai-org/GLM-5-FP8`)
