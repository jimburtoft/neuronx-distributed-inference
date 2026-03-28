# MiMo-V2-Flash Implementation for neuronx-distributed-inference

## Overview

MiMo-V2-Flash is a large MoE model from Xiaomi with unique architecture features:
- **48 layers** with hybrid attention patterns
- **256 routed experts** per layer, top-8 routing with sigmoid activation
- **Hybrid attention**: Full attention + Sliding window attention (alternating pattern)
- **Different Q/K dim (192) and V dim (128)** - asymmetric head dimensions
- **Partial RoPE**: Only 34% of dimensions use rotary position encoding

Reference: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

## Model Architecture

```
MiMo-V2-Flash Architecture
├── Embedding: vocab_size=151936, hidden_size=4096
├── 48 Decoder Layers
│   ├── Attention (hybrid pattern)
│   │   ├── Full Attention (pattern=0): num_kv_heads=4, head_dim=192, v_head_dim=128
│   │   └── Sliding Window (pattern=1): num_kv_heads=8, head_dim=192, v_head_dim=128, window=32768
│   ├── RMSNorm (eps=1e-5)
│   └── MoE Layer
│       ├── Router: 4096 → 256 (sigmoid activation)
│       ├── 256 Experts: gate_proj + up_proj + down_proj (GLU)
│       └── Intermediate size: 1536 per expert
├── Final RMSNorm
└── LM Head (tied with embeddings)

Total Parameters: ~143B (FP8) / ~286B (BF16 equivalent)
```

## Files

### Model Implementation
| File | Description |
|------|-------------|
| `src/neuronx_distributed_inference/models/mimo_v2/__init__.py` | Module exports |
| `src/neuronx_distributed_inference/models/mimo_v2/modeling_mimo_v2.py` | Main model implementation |

### Demo Script
| File | Description |
|------|-------------|
| `examples/generation_mimo_v2_demo.py` | Compilation and inference demo |

## Quick Start

### Prerequisites

The original HuggingFace checkpoint uses FP8 quantized weights. You need to convert them to BF16 first on a GPU machine:

```python
# Run this on a GPU machine with sufficient memory (~200GB)
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "XiaomiMiMo/MiMo-V2-Flash",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.save_pretrained("/path/to/MiMo-V2-Flash-BF16")
# Also copy the tokenizer files from the original checkpoint
```

### Environment Setup
```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Set PYTHONPATH for development
export PYTHONPATH=/home/ubuntu/neuronx-distributed-inference/src:$PYTHONPATH
```

### Compile and Run

```bash
# Compile model with TP=32 (recommended)
python examples/generation_mimo_v2_demo.py \
    --model-path /home/ubuntu/models/MiMo-V2-Flash-BF16 \
    --compiled-model-path /home/ubuntu/traced_model/MiMo-V2-Flash-BF16 \
    --tp-degree 32 \
    --batch-size 1 \
    --max-context-length 128 \
    --seq-len 512 \
    --compile-only

# Run inference (after compilation)
python examples/generation_mimo_v2_demo.py \
    --compiled-model-path /home/ubuntu/traced_model/MiMo-V2-Flash-BF16 \
    --tp-degree 32 \
    --skip-compile \
    --prompt "Hello, how are you?"
```

## Constraints

### TP Degree Requirements

| Mode | TP Range | Valid Values | Requirement |
|------|----------|--------------|-------------|
| Standard GQA | TP ≤ 4 | 1, 2, 4 | Must divide min(num_kv_heads) = 4 |
| CONVERT_TO_MHA | TP > 4 | 8, 16, 32, 64 | Must divide num_attention_heads = 64 |

**Recommended: TP=32** for trn2.48xlarge (32 Neuron cores)

### Memory Requirements

| TP Degree | Memory per Chip | Sharded Model Size | Status |
|-----------|-----------------|-------------------|--------|
| 4 | ~143GB | ~585GB | Exceeds 24GB HBM |
| 32 | ~4.5GB | ~585GB | Fits (recommended) |

### Hybrid Attention Configuration

| Attention Type | Pattern | num_kv_heads | With CONVERT_TO_MHA | head_dim | v_head_dim | window |
|---------------|---------|--------------|---------------------|----------|------------|--------|
| Full | 0 | 4 | 64 | 192 | 128 | ∞ |
| Sliding Window | 1 | 8 | 64 | 192 | 128 | 32768 |

## Implementation Details

### Model Components

| Component | Description |
|-----------|-------------|
| `MiMoV2InferenceConfig` | Config class with hybrid attention pattern parsing |
| `MiMoV2RotaryEmbedding` | Partial RoPE (34% of dimensions) |
| `NeuronMiMoV2Attention` | Hybrid attention with different K/V head dims |
| `NeuronMiMoV2DecoderLayer` | Decoder layer with hybrid attention + MoE |
| `NeuronMiMoV2Model` | Full transformer model |
| `NeuronMiMoV2ForCausalLM` | Causal LM wrapper with weight conversion |

### CONVERT_TO_MHA Mode

When TP > num_kv_heads (4 for MiMo), the implementation uses GQA CONVERT_TO_MHA strategy:

```
Original (GQA):
  Q: [batch, seq, 64, 192]  (64 heads)
  K: [batch, seq, 4, 192]   (4 heads, shared by 16 Q heads each)
  V: [batch, seq, 4, 128]   (4 heads)

CONVERT_TO_MHA (TP=32):
  Q: [batch, seq, 64, 192]  (64 heads, 2 per TP rank)
  K: [batch, seq, 64, 192]  (replicated to 64 heads, 2 per TP rank)
  V: [batch, seq, 64, 128]  (replicated to 64 heads, 2 per TP rank)
```

K/V replication uses `repeat_interleave` for correct GQA pattern:
```python
# Correct: [h0, h0, ..., h1, h1, ...] (each KV head serves consecutive Q heads)
k_proj = k_proj.repeat_interleave(repeat_factor, dim=0)
```

### KV Cache Workarounds

1. **Different K/V head dimensions**: V (128) is padded to K dimension (192) when storing to cache
2. **Hybrid KV heads**: Full attention (4 heads) pads to sliding window (8 heads) for unified cache format
3. **With CONVERT_TO_MHA**: All layers use 64 KV heads (same as Q heads)

### Token Generation

Uses decomposed attention approach for efficiency:
```
1. Prior attention: Compute attention on cached KV (all previous tokens)
2. Active attention: Compute attention on current token's KV
3. Combine: Weighted sum based on softmax normalization factors
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `TP degree must divide num_kv_heads` | TP ≤ 4 but doesn't divide 4 | Use TP = 1, 2, or 4 |
| `TP degree must divide num_attention_heads` | TP > 4 but doesn't divide 64 | Use TP = 8, 16, 32, or 64 |
| Garbage output | K/V replication using `repeat` instead of `repeat_interleave` | Fixed in current version |
| Garbage output | Missing `attention_sink_bias` in token generation | Fixed in current version |
| Slow checkpoint loading | Large model | Use `save_sharded_checkpoint=True` |
| Repetitive output | Missing EOS token handling | Set `eos_token_id` and use `max_new_tokens` |

### Key Bug Fixes (2025-01-28)

#### Issue: Garbage/Random Output During Inference

**Root Cause**: The token generation path was missing the `attention_sink_bias` handling that was present in the context encoding path.

**Technical Details**:
- MiMo-V2-Flash uses "attention sink bias" for Sliding Window Attention (SWA) layers
- This is a learnable parameter (`attention_sink_bias` with shape `[num_attention_heads]`) that adds an extra "sink" column to attention weights
- The HF implementation applies this in `eager_attention_forward`:
  ```python
  # Add sink column
  sinks = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(...)
  attn_weights = torch.cat([attn_weights, sinks], dim=-1)
  # Subtract max for numerical stability
  attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
  # Softmax
  probs = F.softmax(attn_weights, dim=-1)
  # Drop sink column after softmax
  probs = probs[..., :-1]
  ```

**Fix Applied**:
1. Added `attention_sink_bias` handling to the token generation path (was completely missing)
2. Both context encoding and token generation now follow the HF implementation:
   - Add sink bias column before softmax
   - Subtract max for numerical stability
   - Apply softmax
   - Drop sink column after softmax

**Verification**:
- Config: `add_swa_attention_sink_bias=True` (default), `add_full_attention_sink_bias=False`
- Only SWA layers (39 out of 48) have `attention_sink_bias` in the checkpoint
- Full attention layers (layer 0, 5, 11, 17, 23, 29, 35, 41, 47) do NOT have `attention_sink_bias`

## Expert Parallelism Note

The neuronx-distributed MoE module has a limitation for EP > 1:
```
NotImplementedError: Selective Loading with Expert parallelism is not supported in token generation.
```

With TP=32, EP is not required since the model fits in memory using tensor parallelism alone.

## Native FP8 Support (Not Currently Working)

The original HuggingFace checkpoint uses FP8 quantized weights with block-wise 128x128 scales. Native FP8 inference is **not currently supported** due to scale format incompatibilities:

- **HuggingFace format**: Block-wise quantization with per-block scales (e.g., `[256, 16, 2]` for MoE)
- **Neuron framework expectation**: Per-row or per-tensor scales (e.g., `[1, 1, 128]`)

Converting between these formats is not straightforward as it would require complete re-quantization, which loses the original quantization accuracy. For now, use BF16 mode by converting the checkpoint on a GPU machine first.

## vLLM-Neuron 部署（已验证）

### 环境准备

```bash
# 激活 vLLM 0.16.0 环境
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 设置超时（大模型编译和加载需要较长时间）
export VLLM_RPC_TIMEOUT=600000
export VLLM_ENGINE_READY_TIMEOUT_S=7200

# 使用 vllm-neuron 插件（release-0.5.0 分支）
cd /home/ubuntu/vllm-neuron
```

### 启动命令（EP=64，已验证）

以下命令已于 2026-03-27 在 trn2.48xlarge 上验证通过。

**关键点**: MiMo-V2-Flash 模型约 577GB（BF16），使用 TP=64 + 纯 tensor parallelism 在权重分片阶段会 OOM（峰值 >2TB）。
必须使用 **Expert Parallelism (EP=64)** 配置：`moe_tp_degree=1, moe_ep_degree=64`，这样每个 rank 分配 4 个完整的 expert，峰值内存约 1.15TB。

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/models/MiMo-V2-Flash-BF16" \
    --tokenizer "/opt/dlami/nvme/models/MiMo-V2-Flash-BF16" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --trust-remote-code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 64,
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "fused_qkv": false,
            "on_device_sampling_config": {
                "do_sample": true,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95
            },
            "enable_bucketing": true,
            "context_encoding_buckets": [1024],
            "token_generation_buckets": [1024],
            "flash_decoding_enabled": false,
            "logical_nc_config": 2,
            "sequence_parallel_enabled": true,
            "qkv_kernel_enabled": false,
            "qkv_nki_kernel_enabled": false,
            "qkv_cte_nki_kernel_fuse_rope": false,
            "attn_kernel_enabled": false,
            "strided_context_parallel_kernel_enabled": false,
            "async_mode": true,
            "glu_mlp": true,
            "normalize_top_k_affinities": true,
            "router_config": {
                "act_fn": "sigmoid",
                "dtype": "float32"
            },
            "use_index_calc_kernel": true,
            "moe_mask_padded_tokens": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": true,
                "skip_dma_token": true
            },
            "disable_numeric_cc_token": true,
            "scratchpad_page_size": 1024
        }
    }'
```

### 测试 API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/opt/dlami/nvme/models/MiMo-V2-Flash-BF16",
    "messages": [{"role": "user", "content": "What is 25 * 37? Think step by step."}],
    "max_tokens": 256,
    "temperature": 0.6
  }'
```

### EP=64 vs TP=64 对比

| 配置 | moe_tp_degree | moe_ep_degree | 权重分片峰值内存 | 结果 |
|------|--------------|--------------|----------------|------|
| 纯 TP | 64 | 1 | >2TB | OOM (内核 kill) |
| 纯 EP | 1 | 64 | ~1.15TB | 成功 |

**原理**: TP 需要将每个 expert 的权重切分到 64 个 rank 上（需要同时在内存中持有所有分片），而 EP 将完整的 expert 分配到不同 rank（256 experts / 64 ranks = 4 experts/rank），大幅降低分片阶段的内存占用。

### Qwen3 MoE 优化参数兼容性（已验证）

以下参数已于 2026-03-27 在 trn2.48xlarge 上测试验证。

| 参数 | 值 | 状态 | 备注 |
|------|-----|------|------|
| `async_mode` | true | ✅ 已验证 | baseline 已启用 |
| `is_continuous_batching` | true | ✅ 已验证 | baseline 已启用，需配合 batch_size=32 |
| `attn_kernel_enabled` | true | ✅ 已验证 | 输出正确，可提升注意力计算效率 |
| `qkv_kernel_enabled` | true | ❌ 不可用 | 需要 `fused_qkv=true`，但 MiMo Q/K head_dim=192, V head_dim=128，不能使用 fused QKV |
| `qkv_nki_kernel_enabled` | true | ❌ 不可用 | 依赖 `qkv_kernel_enabled` |
| `qkv_cte_nki_kernel_fuse_rope` | true | ❌ 不可用 | 依赖 `qkv_kernel_enabled` |
| `strided_context_parallel_kernel_enabled` | true | ❌ 不可用 | 需要 strided CP NKI kernel，MiMo 的 attention_sink_bias 机制不兼容 |
| `cp_degree` | >1 | ✅ 已实现 | Context Parallelism 支持已添加，可加速长序列 context encoding |

> **注意**: MiMo-V2-Flash 使用非对称的 Q/K/V 维度（Q/K: head_dim=192, V: head_dim=128），因此不能使用 `fused_qkv=true`，
> 这导致 QKV kernel 相关的三个优化都不可用。`attn_kernel_enabled` 是唯一可用的 kernel 优化。

### Context Parallelism (CP) 支持

MiMo-V2-Flash 的注意力模块已支持 Context Parallelism，可将序列维度分布到多个 CP rank 上并行计算，
显著加速长序列的 context encoding 阶段。

**实现原理**:
- CP 仅在 context encoding 阶段生效（token generation 不受影响）
- 每个 CP rank 处理 Q 的局部片段（S/CP），但使用完整的 K/V 计算注意力
- 支持 `sequence_parallel_enabled=True`（推荐）和 `sequence_parallel_enabled=False` 两种模式
- 与 MiMo 特有的 attention_sink_bias、partial RoPE、hybrid attention (Full+SWA) 完全兼容

**配置方法** (在 `override_neuron_config` 中添加):
```json
{
    "cp_degree": 2,
    "sequence_parallel_enabled": true
}
```

**约束条件**:
- `tp_degree` 必须能被 `cp_degree` 整除
- `strided_context_parallel_kernel_enabled` 不可用（MiMo 使用自定义注意力而非 NKI flash attention kernel）
- CP 不影响 token generation 性能，仅加速 context encoding

### 优化版启动命令（attn_kernel 启用）

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/models/MiMo-V2-Flash-BF16" \
    --tokenizer "/opt/dlami/nvme/models/MiMo-V2-Flash-BF16" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --trust-remote-code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 64,
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "fused_qkv": false,
            "on_device_sampling_config": {
                "do_sample": true,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95
            },
            "enable_bucketing": true,
            "context_encoding_buckets": [1024],
            "token_generation_buckets": [1024],
            "flash_decoding_enabled": false,
            "logical_nc_config": 2,
            "sequence_parallel_enabled": true,
            "qkv_kernel_enabled": false,
            "qkv_nki_kernel_enabled": false,
            "qkv_cte_nki_kernel_fuse_rope": false,
            "attn_kernel_enabled": true,
            "strided_context_parallel_kernel_enabled": false,
            "async_mode": true,
            "glu_mlp": true,
            "normalize_top_k_affinities": true,
            "router_config": {
                "act_fn": "sigmoid",
                "dtype": "float32"
            },
            "use_index_calc_kernel": true,
            "moe_mask_padded_tokens": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": true,
                "skip_dma_token": true
            },
            "disable_numeric_cc_token": true,
            "scratchpad_page_size": 1024
        }
    }'
```

## Benchmark 结果

### 测试环境

| 项目 | 值 |
|------|-----|
| 实例类型 | trn2.48xlarge |
| Neuron Cores | 32 (64 logical with NC=2) |
| 系统内存 | 2TB |
| 模型 | MiMo-V2-Flash-BF16 (~577GB) |
| 配置 | EP=64 (moe_tp_degree=1, moe_ep_degree=64) |
| max_model_len | 1024 |
| max_num_seqs | 32 |
| 日期 | 2026-03-27 |

### 性能结果

| 并发数 (Batch Size) | 吞吐量 (tok/s) | 单请求平均延迟 (256 tokens) |
|--------------------|---------------|--------------------------|
| BS=1 | 29.92 | 8.56s |
| BS=8 | 215.94 | 9.48s |
| BS=32 | **649.14** | 12.60s |

### 编译和加载时间

| 阶段 | 时间 |
|------|------|
| 编译（首次，-O3） | ~10 分钟（605 秒） |
| 权重分片加载 (EP=64, on-load sharding) | ~58 分钟 |
| 峰值内存（权重分片阶段） | ~1.15TB |
| 运行态内存 | ~49GB（权重已加载到 Neuron 设备） |

> **注意**: 编译产物会缓存到模型目录下的 `neuron-compiled-artifacts/`，后续启动只需权重加载。
> 如需加速权重加载，可考虑使用 `save_sharded_checkpoint: true` 预分片到磁盘（但需要足够的磁盘空间存储 64 份分片）。

## References

- [MiMo-V2-Flash on HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [Xiaomi MiMo Technical Report](https://github.com/XiaomiMiMo/MiMo)
- [neuronx-distributed-inference Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

## Version History

| Date | Changes |
|------|---------|
| 2026-03-27 | Added Context Parallelism (CP) support to attention module for long-sequence acceleration |
| 2026-03-27 | Added vLLM-Neuron deployment guide with EP=64 config, benchmark results, and Qwen3 MoE optimization compatibility table |
| 2025-01-28 | Fixed attention_sink_bias missing in token generation path (was causing garbage output) |
| 2025-01-21 | Simplified to BF16-only mode (native FP8 not supported due to scale format incompatibilities) |
| 2025-01-20 | Implemented CONVERT_TO_MHA for TP=32 support |
| 2025-01-19 | Initial implementation with hybrid attention and MoE |
