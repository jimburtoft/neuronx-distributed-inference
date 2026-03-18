# Contrib Model: Qwen3-VL-30B-A3B-Instruct

NeuronX Distributed Inference implementation of Qwen3-VL-30B-A3B-Instruct, a 30B-parameter Mixture-of-Experts vision-language model with 3B active parameters per token.

> **Note:** This is a vision-language MoE model. The implementation supports both text-only and vision+text inference. The vision encoder (ViT) reuses the built-in NxDI `qwen3_vl` vision pipeline. The text decoder is a custom 48-layer MoE implementation with 128 experts and top-8 routing.

## Model Information

- **HuggingFace ID:** `Qwen/Qwen3-VL-30B-A3B-Instruct`
- **Model Type:** Vision-Language Mixture-of-Experts (VL-MoE)
- **Architecture:** 27-layer ViT encoder + 48-layer MoE decoder
- **Parameters:** 30B total, 3B active per token
- **License:** Check HuggingFace model card

## Architecture Details

- **Text Decoder:** 48 MoE layers, hidden_size=2048, 32 attention heads, 4 KV heads (GQA 8:1)
- **MoE:** 128 routed experts, top-8 routing, no shared experts, moe_intermediate_size=768
- **Vision Encoder:** 27-layer ViT, hidden_size=1152, patch_size=16, out_hidden_size=2048
- **DeepStack:** Multi-level vision features at ViT layers [8, 16, 24]
- **Attention:** QK-Norm (RMSNorm per head), M-RoPE (multimodal rotary), head_dim=128
- **Total Size:** 62.14 GB (BF16)

## Validation Results

**Validated:** 2026-03-17
**Configuration:** TP=4, batch_size=1, seq_len=2176, max_context_length=2048, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles, loads, and runs |
| Text Generation | PASS | Correct factual responses ("Paris", "2") |
| Vision + Text | PASS | Correctly identifies colors in images |
| Decode Throughput | PASS | 95-99 tok/s with ISA kernels |

### Performance Metrics

| Metric | No ISA Kernels | ISA Kernels (QKV+Attn) |
|--------|:--------------:|:----------------------:|
| Compilation time | 13 min | 18 min |
| Text decode throughput | 65 tok/s | **95-99 tok/s** |
| Text prefill (5 tok) | 150ms | 1072ms |
| VL prefill (80 tok, 224x224) | 264ms | 1191ms |

**Status:** VALIDATED

### Performance Context

| Model | Instance | Decode tok/s | Notes |
|-------|----------|:------------:|-------|
| **Qwen3-VL-30B-A3B (this)** | **trn2.3xlarge** | **95-99** | **ISA kernels, BF16, batch=1** |
| Qwen3-VL-8B (dense) | trn2.3xlarge | 43.9 | Dense model, ISA kernels |
| Qwen3.5-35B-A3B | trn2.3xlarge | 54.9 | Similar MoE VL architecture |

The 95-99 tok/s result is 2.2x faster than dense Qwen3-VL-8B and 1.8x faster than Qwen3.5-35B-A3B, demonstrating MoE efficiency: only 3B active params execute per token.

## Source Code Structure

This model requires two modeling files due to its VL+MoE architecture:

| File | Description | Lines |
|------|-------------|-------|
| `modeling_qwen3_vl_moe.py` | VL orchestrator: config, vision pipeline, forward, weight loading | ~700 |
| `modeling_qwen3_vl_moe_text.py` | MoE text decoder: attention, MoE layers, state dict conversion | ~860 |

## Usage

```python
import os
import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# Add src to path
import sys
sys.path.insert(0, "path/to/Qwen3-VL-30B-A3B-Instruct/src")
from modeling_qwen3_vl_moe import (
    NeuronQwen3VLMoeForCausalLM,
    Qwen3VLMoeNeuronConfig,
    Qwen3VLMoeVLInferenceConfig,
)

os.environ["NEURON_RT_NUM_CORES"] = "4"

model_path = "/path/to/Qwen3-VL-30B-A3B-Instruct/"
compiled_path = "/path/to/compiled/"

DTYPE = torch.bfloat16
TP = 4  # trn2.3xlarge with LNC=2

# Text config (MoE-aware)
text_neuron_config = MoENeuronConfig(
    tp_degree=TP,
    batch_size=1,
    ctx_batch_size=1,
    seq_len=2176,
    max_context_length=2048,
    max_new_tokens=128,
    torch_dtype=DTYPE,
    attention_dtype=DTYPE,
    rpl_reduce_dtype=DTYPE,
    moe_tp_degree=TP,
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
)

# Vision config
vision_neuron_config = Qwen3VLMoeNeuronConfig(
    tp_degree=TP,
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
)

config = Qwen3VLMoeVLInferenceConfig(
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
    load_config=load_pretrained_config(model_path),
)

# Compile and load
model = NeuronQwen3VLMoeForCausalLM(model_path=model_path, config=config)
model.compile(compiled_path)
model.load(compiled_path)

# Generate (see test script for full text and vision examples)
```

## Compatibility Matrix

| Instance/Version | SDK 2.28+ | SDK 2.27 and earlier |
|------------------|-----------|----------------------|
| trn2.3xlarge     | VALIDATED | Not tested           |
| trn2.48xlarge    | Expected  | Not tested           |
| trn1             | Not supported (needs trn2 HBM) | Not supported |
| inf2             | Not tested | Not tested          |

## Testing

Run integration tests:

```bash
# Set environment
export NEURON_RT_NUM_CORES=4
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

# Run tests
pytest contrib/models/Qwen3-VL-30B-A3B-Instruct/test/integration/test_model.py --capture=tee-sys
```

Or run manually:

```bash
cd contrib/models/Qwen3-VL-30B-A3B-Instruct
python3 test/integration/test_model.py
```

## NKI Kernel Notes

| Kernel | Status | Notes |
|--------|--------|-------|
| QKV ISA | Enabled | +50% decode throughput |
| Attention ISA (CTE) | Enabled | Flash attention for context encoding |
| MLP ISA | Disabled | N/A for MoE layers |
| MoE Blockwise Matmul | Default | Active for CTE expert computation |
| MoE Fused TKG | Not recommended | Requires 33% padding (768->1024), net-negative performance |
| Sequence Parallel | Disabled | Risky with MoE routing |

## Memory Requirements

| Component | BF16 Size |
|-----------|-----------|
| Model weights | 62.14 GB |
| KV cache (2K context) | ~0.38 GB |
| Activations + overhead | ~5-10 GB |
| **Total** | **~68-75 GB** |
| trn2.3xlarge HBM | 96 GB |

## Example Checkpoints

* `Qwen/Qwen3-VL-30B-A3B-Instruct`

## Maintainer

Neuroboros Team - Annapurna Labs

**Last Updated:** 2026-03-17
