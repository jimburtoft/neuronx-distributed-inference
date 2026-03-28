# Contrib Model: Qwen2.5-VL-3B-Instruct

NeuronX Distributed Inference implementation of Qwen2.5-VL-3B-Instruct.

> **Note:** This implementation supports the **text backbone only** with M-RoPE (Multimodal Rotary Position Embeddings). Vision encoder is not included. See [Related: Full Vision-Language Support](#related-full-vision-language-support) for a validated multimodal implementation of the same model family.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Model Type:** Vision-Language model (`qwen2_5_vl`). This contrib supports **text input/output only**.
- **License:** Check [HuggingFace model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## Architecture Details

### Text Backbone

- **Layers:** 36
- **Hidden Size:** 2048
- **Attention Heads:** 16 (Q) / 2 (KV) -- Grouped Query Attention
- **Head Dimension:** 128
- **Intermediate Size:** 11,008 (SwiGLU MLP)
- **Vocabulary:** 151,936
- **Max Position Embeddings:** 128,000
- **RoPE:** M-RoPE with sections [16, 24, 24] (temporal, height, width), theta=1e6
- **QKV Bias:** True, O Bias: False
- **Normalization:** RMSNorm (eps=1e-6)
- **Activation:** SiLU (SwiGLU)
- **Tied Weights:** embed_tokens and lm_head are tied
- **Model Size:** ~6 GB (BF16)

> Same architecture family as Qwen2.5-VL-7B but with fewer layers (36 vs 28), smaller hidden size (2048 vs 3584), and fewer KV heads (2 vs 4).

### Vision Encoder (not implemented in this contrib)

- **Layers:** 32, **Hidden Size:** 1280, **Heads:** 16
- **Output Hidden Size:** 2048 (projects to text hidden_size)
- **Patch Size:** 14, **Spatial Merge Size:** 2

## Validation Results

**Validated:** 2026-01-29 (original), updated 2026-03-27 (M-RoPE fix)
**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model loads successfully |
| Token Matching | *pending revalidation* | Previously 67.2% (without M-RoPE) |
| TTFT (P50) | PASS | 29.82ms (threshold: 100ms) |
| Throughput | PASS | 38.20 tok/s (threshold: 10 tok/s) |

> **M-RoPE fix (2026-03-27):** The original implementation used standard 1D RotaryEmbedding instead of M-RoPE, which likely caused the 67.2% token match. This update adds proper M-RoPE with section splitting [16, 24, 24]. Revalidation pending on Neuron instance.

### Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (P50) | 29.82ms |
| Throughput | 38.20 tokens/s |

### Device Profiling Metrics

**Configuration:** TP=2, batch_size=1, seq_len=128, bfloat16
**Instance:** trn1.32xlarge | **Profiled:** 2026-03-21

| Metric | Context Encoding | Token Generation |
|--------|-----------------|------------------|
| MFU (%) | 0.13 | 0.00 |
| MBU (%) | 0.27 | 0.29 |
| HFU (%) | 0.15 | 0.02 |
| Execution Time (us) | 0.03 | 0.03 |
| HBM Read | 3.15 GB | 3.09 GB |
| HBM Write | 62.86 MB | 3.35 MB |

**Throughput:** 32.98 tok/s | **Compile Time:** 224.93s

> Metrics from `neuron-profile capture` on compiled NEFFs. MFU = Model FLOPs Utilization,
> MBU = Memory Bandwidth Utilization, HFU = Hardware FLOPs Utilization.

## Related: Full Vision-Language Support

The **Qwen2.5-VL-7B-Instruct** contrib provides a complete vision-language implementation for the Qwen2.5-VL model family:

- **Full vision encoder** with hybrid windowed + global attention
- **M-RoPE** for spatial/temporal position encoding
- **Multi-bucket CTE** for 4.8x TTFT improvement on short inputs
- **86.4 tok/s** on trn2.3xlarge (TP=4), 100% token match

The 3B model's vision encoder uses the same architecture as the 7B, with output projection adjusted for the smaller hidden size (2048 vs 3584).

## Usage

```python
import torch
from neuronx_distributed_inference.models.config import NeuronConfig

# Import model classes from src
from src.modeling_qwen2vl import (
    NeuronQwen2_5_VL3BForCausalLM,
    Qwen2_5_VL3BInferenceConfig,
)

model_path = "/path/to/Qwen2.5-VL-3B-Instruct/"
compiled_model_path = "/path/to/compiled/"

# Configure
neuron_config = NeuronConfig(
    tp_degree=2,
    batch_size=1,
    seq_len=512,
    torch_dtype=torch.bfloat16,
)

config = Qwen2_5_VL3BInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config
)

# Compile and load
model = NeuronQwen2_5_VL3BForCausalLM(config)
model.compile(compiled_model_path)
model.load(compiled_model_path)

# Generate
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ... (see integration test for full example)
```

## Compatibility Matrix

| Instance/Version | 2.20+ | 2.19 and earlier |
|------------------|-------|------------------|
| Trn1             | Working | Not tested |
| Trn2             | Not tested | Not tested |
| Inf2             | Not tested | Not tested |

## Testing

Run integration tests:

```bash
pytest nxdi_contrib_models/models/Qwen2.5-VL-3B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd nxdi_contrib_models/models/Qwen2.5-VL-3B-Instruct
python3 test/integration/test_model.py
```

## Example Checkpoints

* Qwen/Qwen2.5-VL-3B-Instruct

## Maintainer

Annapurna Labs

**Last Updated:** 2026-03-27
