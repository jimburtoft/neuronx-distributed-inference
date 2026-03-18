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
**Instance:** trn2.3xlarge (ap-southeast-4), SDK 2.28, NxDI 0.8.0
**Configuration:** TP=4, LNC=2, batch_size=1, seq_len=2176, max_context_length=2048, bfloat16

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Smoke Test | PASS | Model compiles, loads, and runs |
| Text Generation | PASS | Correct factual responses ("Paris", "2") |
| Vision + Text | PASS | Correctly identifies colors in images |
| Decode Throughput | PASS | 96.7 tok/s (measured via pytest) |

### Performance Metrics

All benchmarks were run on trn2.3xlarge (TP=4, LNC=2, BF16, batch=1) by this project.

| Metric | No ISA Kernels | ISA Kernels (QKV+Attn) |
|--------|:--------------:|:----------------------:|
| Compilation time | 13 min | 18 min |
| Text decode throughput | 65 tok/s | **95-99 tok/s** |
| Text prefill (5 tok) | 150ms | 1072ms |
| VL prefill (80 tok, 224x224) | 264ms | 1191ms |

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

## NKI Kernel Analysis

### Kernels Enabled

| Kernel | Flag | Impact | Details |
|--------|------|--------|---------|
| **QKV ISA** | `qkv_kernel_enabled=True` | +50% decode throughput (65 -> 95-99 tok/s) | Fuses RMSNorm + QKV projection into a single NKI kernel. Compatible because head_dim=128 and fused_qkv=True are standard dimensions. QK-Norm (RMSNorm on Q/K) is applied after the kernel, outside its scope. |
| **Attention ISA (CTE)** | `attn_kernel_enabled=True` | Accelerates context encoding | Flash attention NKI kernel for prefill. head_dim=128 is natively supported. GQA 8:1 ratio works with TP=4 sharding. CTE bucket alignment verified: 128, 512, 2048 all satisfy the kernel's alignment requirements. |
| **MoE Blockwise Matmul** | `blockwise_matmul_config={"block_size": 32768}` | Default for CTE expert computation | Standard NxDI blockwise matmul for MoE context encoding. I_TP=192 at TP=4 satisfies the basic kernel alignment (192 % 16 = 0). Block size of 32768 exceeds seq_len * top_k for all bucket sizes. |

### Kernels Disabled (with reasoning)

| Kernel | Flag | Why Disabled |
|--------|------|-------------|
| **MLP ISA** | `mlp_kernel_enabled=False` | The MLP ISA kernel targets dense `NeuronLlamaMLP` layers. All 48 decoder layers in this model are MoE -- they use `initialize_moe_module()` which has its own compute path (blockwise matmul for CTE, standard dispatch for TKG). Setting this flag to True would be a no-op. |
| **MoE Fused TKG** | `moe_fused_nki_kernel_enabled=False` | The fused TKG mega-kernel combines router + expert gather + gate/up/down matmuls into one kernel. It requires `I_TP % 128 == 0`, where I_TP = moe_intermediate_size / moe_tp_degree. At TP=4: I_TP = 768/4 = 192. Since 192 % 128 = 64, the kernel cannot run without padding moe_intermediate_size from 768 to 1024 (+33%). **We tested this**: padding works, compilation succeeds, but performance drops to 80-88 tok/s (vs 95-99 without) because the 33% extra FLOPs across 128 experts x 48 layers x top-8 outweighs the kernel fusion savings. |
| **Shard-on-Intermediate Blockwise** | Not configured | The shard-on-intermediate variant of blockwise matmul requires `I_TP % 256 == 0`. At TP=4: 192 % 256 != 0. Would require the same 768->1024 padding as fused TKG, with the same 33% overhead. |
| **Sequence Parallel** | `sequence_parallel_enabled=False` | Sequence parallelism shards activations across TP ranks along the sequence dimension. This interacts poorly with MoE routing: the top-K expert selection and token-to-expert dispatch assume full sequence visibility. Enabling it risks incorrect routing decisions or silent accuracy degradation. |
| **Vision Encoder Kernels** | All `False` for vision config | The 27-layer ViT vision encoder has head_dim=72 (1152/16 heads), which is not supported by the attention ISA kernel (requires head_dim=128). QKV and MLP kernels are also disabled for vision -- this matches all other VL models in NxDI (Qwen2.5-VL, Qwen3-VL-8B, etc.). Vision compilation uses `-O1` with `--auto-cast=none`. |

### Kernels Not Applicable (architecture mismatch)

These kernels exist in other NxDI contrib models but do not apply to Qwen3-VL-30B-A3B-Instruct:

| Kernel | Used By | Why Not Applicable |
|--------|---------|-------------------|
| **DeltaNet NKI** (`nki_deltanet`) | Qwen3.5-35B-A3B | DeltaNet is a linear recurrent attention mechanism that replaces standard QKV softmax attention with a gated delta-rule state update. Qwen3.5 uses DeltaNet for 30 of 40 layers. Qwen3-VL-30B-A3B uses standard GQA attention for all 48 layers -- completely different attention mechanism. Cannot be applied without changing the model architecture and weights. |
| **Flash Attention d256** (`nki_flash_attn_d256`) | Qwen3.5-35B-A3B | Custom NKI kernel that tiles QK matmul into 128-dim chunks to support head_dim=256 (exceeds the standard kernel's 128 limit). Qwen3-VL-30B-A3B has head_dim=128, which the standard attention ISA kernel handles natively. This kernel solves a problem that does not exist for our model. |
| **Sigmoid-Gated Shared Expert** | Qwen3.5-35B-A3B | Qwen3.5 has a shared expert with sigmoid gating (`sigmoid(gate(x)) * expert(x)`). Qwen3-VL-30B-A3B has `n_shared_experts=0` -- no shared experts at all. |

### Future Kernel Opportunities

The primary optimization opportunity is a **custom MoE blockwise matmul kernel that handles I_TP=192 natively** without padding. The current fused TKG kernel requires 128-element SBUF partition alignment, forcing a 33% size increase. A kernel that supports non-128-aligned intermediate sizes (or uses a smaller partition granularity) could recover the fusion benefit. This would require authoring a new NKI kernel -- it is the only remaining kernel opportunity with meaningful performance potential for this model.

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

Jim Burtoft (jimburtoft)

**Last Updated:** 2026-03-18
