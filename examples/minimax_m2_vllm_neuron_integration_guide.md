# MiniMax M2 vLLM-Neuron 集成指南

本文档总结了 MiniMax M2 模型在 vLLM-Neuron 上的集成工作，包括独立编译脚本 v4 和 vLLM 自动编译流程的适配。

**日期**: 2025-02-02 (初始) / 2026-03-27 (vLLM 集成验证)
**状态**: 独立编译 ✓ | vLLM 集成 ✓

---

## 目录

1. [项目概述](#1-项目概述)
2. [v4 独立编译脚本](#2-v4-独立编译脚本)
3. [vLLM-Neuron 集成](#3-vllm-neuron-集成)
4. [关键问题与解决方案](#4-关键问题与解决方案)
5. [配置参考](#5-配置参考)
6. [运行指南](#6-运行指南)

---

## 1. 项目概述

### 1.1 目标

- **独立编译**: 创建 `generation_minimax_m2_v4_demo.py`，支持 Expert Parallelism (EP) 和长上下文
- **vLLM 集成**: 通过 vLLM-Neuron 的 `--additional-config` 参数自动编译和运行 MiniMax M2

### 1.2 关键成果

| 功能 | 状态 | 备注 |
|------|------|------|
| 独立编译 (v4 demo) | ✅ 成功 | 支持 EP、长上下文 |
| 独立推理 | ✅ 成功 | generation_minimax_m2_v4_demo.py |
| vLLM 自动编译 | ✅ 成功 | vLLM 0.16.0 + vllm-neuron release-0.5.0 |
| vLLM OpenAI API | ✅ 成功 | 2026-03-27 在 trn2.48xlarge 上验证通过 |

---

## 2. v4 独立编译脚本

### 2.1 新功能

`generation_minimax_m2_v4_demo.py` 借鉴 MiMo-V2-Flash 实现，新增以下功能：

#### Expert Parallelism (EP) 支持

```python
# EP 配置选项 (256 experts, tp_degree=64)
# EP=64, MoE_TP=1: 4 experts per rank (maximum EP)
# EP=32, MoE_TP=2: 8 experts per rank
# EP=16, MoE_TP=4: 16 experts per rank
# EP=8, MoE_TP=8: 32 experts per rank
# EP=1, MoE_TP=64: 256 experts per rank (default, no EP)

parser.add_argument("--moe-ep-degree", type=int, default=1)
parser.add_argument("--moe-tp-degree", type=int, default=None)
```

#### Hybrid Sharding (混合分片)

当 EP > 1 且 batch_size < 32 时，自动启用 hybrid sharding：
- **Prefill (CTE)**: 使用 EP 并行
- **Token Generation (TKG)**: 使用 EP=1 (全 TP)

```python
if use_hybrid_sharding:
    config_kwargs['hybrid_sharding_config'] = {
        'moe_cte_tp_degree': moe_tp_degree,
        'moe_cte_ep_degree': moe_ep_degree,
        'moe_tkg_tp_degree': args.tp_degree,  # Full TP
        'moe_tkg_ep_degree': 1,  # No EP for TKG
    }
```

#### 命令行参数

```bash
python generation_minimax_m2_v4_demo.py \
    --model-path /opt/dlami/nvme/model_hf/MiniMax-M2-BF16/ \
    --compiled-model-path /opt/dlami/nvme/traced_model/MiniMax-M2-BF16-v4/ \
    --tp-degree 64 \
    --batch-size 1 \
    --max-context-length 256 \
    --seq-len 512 \
    --compile-only
```

### 2.2 OOM 问题修复

**问题**: Token generation 编译时 OOM (需要 70GB vs 可用 24GB)

**根因**: `use_shard_on_intermediate_dynamic_while=True` 将 intermediate_size 从 1536 pad 到 16384 (10x)

**解决方案**:
```python
blockwise_matmul_config={
    "use_shard_on_intermediate_dynamic_while": False,  # 关键！
    "skip_dma_token": True,
}
```

**原理**:
- MiniMax M2 intermediate_size=1536
- 1536 / tp_degree(64) = 24
- 24 不能被 SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP(256) 整除
- 启用该选项会 pad 到 16384，导致内存暴增

---

## 3. vLLM-Neuron 集成

### 3.1 架构注册

MiniMax M2 已在 NxDI constants.py 中注册：

```python
# neuronx_distributed_inference/utils/constants.py
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_v3 import NeuronMiniMaxM2ForCausalLMV3

MODEL_TYPES = {
    # ...
    "minimaxm2": {"causal-lm": NeuronMiniMaxM2ForCausalLMV3},
    # ...
}
```

vLLM-Neuron 自动映射：
```
HuggingFace: MiniMaxM2ForCausalLM
    ↓ model.lower()
NxDI key: minimaxm2
    ↓ MODEL_TYPES lookup
Class: NeuronMiniMaxM2ForCausalLMV3
```

### 3.2 vLLM 配置传递

通过 `--additional-config` 传递 Neuron 配置：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16" \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 1 \
    --additional-config '{
        "override_neuron_config": {
            "glu_mlp": true,
            "moe_tp_degree": 64,
            "moe_ep_degree": 1,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"},
            "sequence_parallel_enabled": true,
            "logical_nc_config": 2,
            ...
        }
    }'
```

---

## 4. 关键问题与解决方案

### 4.1 XLA 类型不匹配 (S32 vs S64)

**错误信息**:
```
Status: INVALID_ARGUMENT: Cannot concatenate arrays with different element types: S32 vs S64.
*** torch_xla::IndexByTensors ***
```

**根因分析**:

XLA 的 `IndexByTensors` 操作会 stack 多个 index tensors，要求类型一致。代码中存在：
- `SPMDRank.rank`: int32 (来自 neuronx_distributed)
- `torch.topk` 返回值: int64 (torch.long)

**第一次尝试 (失败)**:
将所有 rank tensors 改为 int64 (torch.long)

```python
# 修改 state_dict 中的 rank tensors
neuron_state_dict["rank_util.rank"] = torch.arange(
    0, config.neuron_config.tp_degree, dtype=torch.long  # int64
)
```

**新问题**:
```
TypeError: Unsupported dtype 'int64' of operand 'src' in 'load', expected one of the following dtypes: 'int32', ...
```

**根因**: NKI kernel (`find_nonzero_indices`) 的 `nl.load` 不支持 int64

**最终解决方案**:

1. **保持 int32**: rank tensors 必须是 int32 以兼容 NKI kernels
2. **禁用 index_calc_kernel**: 避免触发不兼容的 NKI kernel

```python
# 在 override_neuron_config 中添加
"use_index_calc_kernel": false
```

### 4.2 SPMDRank 类型问题

**问题位置**: `/neuronx_distributed/parallel_layers/layers.py`

```python
class SPMDRank(torch.nn.Module):
    def __init__(self, world_size, ...):
        # 创建 int32 parameter
        self.rank = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int32),  # 必须是 int32
            requires_grad=False
        )
```

**MiniMax M2 的适配**:

```python
# modeling_minimax_m2_v3.py - state_dict 中设置
neuron_state_dict["rank_util.rank"] = torch.arange(
    0, config.neuron_config.tp_degree, dtype=torch.int32  # 保持 int32
)

# ReplicatedRMSNorm.forward() - 使用时保持 int32
rank_index = rank_util.rank[:1]  # 不转换类型
local_weight = torch.index_select(weight_reshaped, 0, rank_index).squeeze(0)
```

### 4.3 MiniMax M2 特有的适配

#### ReplicatedRMSNorm (分布式 QK Norm)

MiniMax M2 的 QK Norm 在 reshape 前应用于完整投影输出，需要特殊处理：

```python
class MiniMaxM2QKNorm(nn.Module):
    """
    分布式 QK Norm:
    - 每个 TP rank 计算本地 sum(x²)
    - All-reduce 获取全局统计量
    - 使用 rank_util tensor 动态切片权重
    """
    def forward(self, hidden_states, rank_util=None):
        # 1. 本地计算
        local_sum_sq = hidden_states.pow(2).sum(-1, keepdim=True)

        # 2. All-reduce
        if self.tp_degree > 1:
            global_sum_sq = reduce_from_tensor_model_parallel_region(local_sum_sq)

        # 3. 全局 RMS
        hidden_states = hidden_states * torch.rsqrt(global_sum_sq / self.full_hidden_size + eps)

        # 4. 动态权重切片 (使用 rank_util tensor)
        if rank_util is not None and self.tp_degree > 1:
            weight_reshaped = self.weight.view(self.tp_degree, self.hidden_size)
            rank_index = rank_util.rank[:1]  # int32 tensor
            local_weight = torch.index_select(weight_reshaped, 0, rank_index).squeeze(0)

        return local_weight * hidden_states
```

#### Sigmoid Router with Bias

```python
class RouterTopKWithBias(RouterTopK):
    """MiniMax M2 使用 sigmoid + e_score_correction_bias"""

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)  # sigmoid

        # Bias 只影响选择，不影响权重
        scores_for_choice = expert_affinities + self.e_score_correction_bias.unsqueeze(0)
        _, expert_index = torch.topk(scores_for_choice, self.top_k, dim=-1)

        # 保持 int32 以兼容 NKI
        expert_index = expert_index.detach().to(dtype=torch.int32)

        return router_logits, expert_affinities, expert_index
```

---

## 5. 配置参考

### 5.1 完整 vLLM 启动命令（已验证）

以下命令已于 2026-03-27 在 trn2.48xlarge 上验证通过。

**环境准备**:
```bash
# 激活 vLLM 0.16.0 环境
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 设置超时（大模型编译和加载需要较长时间）
export VLLM_RPC_TIMEOUT=600000
export VLLM_ENGINE_READY_TIMEOUT_S=7200

# 使用 vllm-neuron 插件（release-0.5.0 分支）
cd /home/ubuntu/vllm-neuron
```

**启动命令**:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16" \
    --tokenizer "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16" \
    --tensor-parallel-size 64 \
    --max-model-len 512 \
    --max-num-seqs 1 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --trust-remote-code \
    --additional-config '{
        "override_neuron_config": {
            "sequence_parallel_enabled": true,
            "logical_nc_config": 2,
            "fused_qkv": true,
            "is_continuous_batching": false,
            "async_mode": false,
            "flash_decoding_enabled": false,
            "enable_bucketing": true,
            "context_encoding_buckets": [256],
            "token_generation_buckets": [512],
            "use_index_calc_kernel": true,
            "moe_mask_padded_tokens": true,
            "qkv_kernel_enabled": false,
            "qkv_nki_kernel_enabled": false,
            "attn_kernel_enabled": false,
            "strided_context_parallel_kernel_enabled": false,
            "router_config": {
                "act_fn": "sigmoid",
                "dtype": "float32"
            },
            "glu_mlp": true,
            "save_sharded_checkpoint": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": false,
                "skip_dma_token": true
            },
            "disable_numeric_cc_token": true,
            "scratchpad_page_size": 1024,
            "moe_tp_degree": 64,
            "moe_ep_degree": 1
        }
    }'
```

> **注意**: 首次启动需要编译，编译时间约 22 分钟。编译产物会缓存到模型目录下的 `neuron-compiled-artifacts/`，后续启动将跳过编译。权重预分片（`save_sharded_checkpoint: true`）约需 20 分钟，峰值内存约 1.2TB。

### 5.2 优化版启动命令（高吞吐，已验证）

参考 Qwen3 MoE 优化参数，启用 continuous batching + NKI kernel 优化。
以下所有优化参数已于 2026-03-27 在 trn2.48xlarge 上逐一测试验证输出正确性。

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16" \
    --tokenizer "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16" \
    --tensor-parallel-size 64 \
    --max-model-len 512 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --trust-remote-code \
    --additional-config '{
        "override_neuron_config": {
            "sequence_parallel_enabled": true,
            "logical_nc_config": 2,
            "fused_qkv": true,
            "is_continuous_batching": true,
            "batch_size": 32,
            "ctx_batch_size": 1,
            "tkg_batch_size": 32,
            "async_mode": true,
            "flash_decoding_enabled": false,
            "enable_bucketing": true,
            "context_encoding_buckets": [256],
            "token_generation_buckets": [512],
            "use_index_calc_kernel": true,
            "moe_mask_padded_tokens": true,
            "qkv_kernel_enabled": true,
            "qkv_nki_kernel_enabled": true,
            "qkv_cte_nki_kernel_fuse_rope": true,
            "attn_kernel_enabled": true,
            "strided_context_parallel_kernel_enabled": false,
            "router_config": {
                "act_fn": "sigmoid",
                "dtype": "float32"
            },
            "glu_mlp": true,
            "save_sharded_checkpoint": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": false,
                "skip_dma_token": true
            },
            "disable_numeric_cc_token": true,
            "scratchpad_page_size": 1024,
            "moe_tp_degree": 64,
            "moe_ep_degree": 1
        }
    }'
```

### 5.3 Qwen3 MoE 优化参数兼容性（已验证）

| 参数 | 值 | 状态 | 备注 |
|------|-----|------|------|
| `async_mode` | true | ✅ 已验证 | |
| `is_continuous_batching` | true | ✅ 已验证 | 需配合 batch_size=32 |
| `qkv_kernel_enabled` | true | ✅ 已验证 | 需 fused_qkv=true |
| `qkv_nki_kernel_enabled` | true | ✅ 已验证 | 需 qkv_kernel_enabled=true |
| `qkv_cte_nki_kernel_fuse_rope` | true | ✅ 已验证 | partial RoPE (rotary_dim=64) 也正常 |
| `attn_kernel_enabled` | true | ✅ 已验证 | |
| `strided_context_parallel_kernel_enabled` | true | ❌ 不可用 | 需要 Context Parallelism，报错 "CP must also be enabled" |
| `attention_dp_degree` | >1 | ❌ 不可用 | 模型未实现 |
| `cp_degree` | >1 | ❌ 不可用 | 模型未实现 |

### 5.4 关键配置参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `tensor_parallel_size` | 64 | 必须与 tp_degree 匹配 |
| `logical_nc_config` | 2 | 64 TP 需要 logical NC 拆分 |
| `glu_mlp` | true | MiniMax M2 使用 GLU MLP |
| `router_config.act_fn` | "sigmoid" | MiniMax M2 使用 sigmoid |
| `use_shard_on_intermediate_dynamic_while` | false | 避免 OOM（intermediate_size=1536 不能被 256 整除） |

### 5.5 使用预编译模型

如果使用 `generation_minimax_m2_v4_demo.py` 预编译的模型：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/traced_model/MiniMax-M2-BF16-v4/" \
    --tensor-parallel-size 64 \
    --max-model-len 512 \
    --max-num-seqs 1 \
    --block-size 512 \
    --additional-config '{"override_neuron_config": {...}}'
```

---

## 6. 运行指南

### 6.1 方式一：独立编译后运行

```bash
# Step 1: 编译
cd /home/ubuntu/neuronx-distributed-inference/examples
python generation_minimax_m2_v4_demo.py --compile-only \
    --model-path /opt/dlami/nvme/model_hf/MiniMax-M2-BF16/ \
    --compiled-model-path /opt/dlami/nvme/traced_model/MiniMax-M2-BF16-v4/ \
    --tp-degree 64 --batch-size 1 --max-context-length 256 --seq-len 512

# Step 2: 推理
python generation_minimax_m2_v4_demo.py --skip-compile \
    --compiled-model-path /opt/dlami/nvme/traced_model/MiniMax-M2-BF16-v4/ \
    --prompt "Give me a short introduction to large language models."
```

### 6.2 方式二：vLLM 自动编译

```bash
# 一步完成编译和启动服务
cd /home/ubuntu/vllm-neuron
python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16" \
    ... (完整配置见上方)
```

### 6.3 测试 API

```bash
# Chat Completion API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16",
    "messages": [{"role": "user", "content": "What is 25 * 37? Think step by step."}],
    "max_tokens": 256,
    "temperature": 0.6
  }'

# Completion API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/opt/dlami/nvme/model_hf/MiniMax-M2-BF16",
    "prompt": "Hello, ",
    "max_tokens": 50
  }'
```

---

## 7. Benchmark 结果

### 7.1 测试环境

| 项目 | 值 |
|------|-----|
| 实例类型 | trn2.48xlarge |
| Neuron Cores | 32 (64 logical with NC=2) |
| 系统内存 | 2TB |
| 模型 | MiniMax-M2-BF16 (~434GB) |
| TP | 64 (moe_tp_degree=64, moe_ep_degree=1) |
| max_model_len | 512 |
| 日期 | 2026-03-27 |

### 7.2 性能结果

**配置 A: 低延迟（单请求）**
- `is_continuous_batching=false, max_num_seqs=1, async_mode=false`

| 指标 | 值 |
|------|-----|
| 单请求吞吐 (BS=1, 256 output tokens) | **62.24 tok/s** |

**配置 B: 高吞吐（Continuous Batching + 全优化）**
- `is_continuous_batching=true, batch_size=32, async_mode=true`
- `qkv_kernel_enabled=true, qkv_nki_kernel_enabled=true, qkv_cte_nki_kernel_fuse_rope=true, attn_kernel_enabled=true`

| 并发数 | 吞吐量 (tok/s) | 单请求平均延迟 (256 tokens) |
|----|----|----|
| BS=1 | 3.06 | 83.78s |
| BS=8 | 22.85 | 89.65s |
| BS=32 | **75.35** | 108.72s |

**配置 C: Continuous Batching 无 kernel 优化**（对照组）
- `is_continuous_batching=true, batch_size=32, async_mode=true`
- `qkv_kernel_enabled=false, attn_kernel_enabled=false`

| 并发数 | 吞吐量 (tok/s) |
|----|-----|
| BS=1 | 2.98 |
| BS=8 | 21.55 |
| BS=32 | 73.28 |

> **结论**:
> - 单请求场景推荐配置 A（62 tok/s），CB 开启后单请求延迟大幅增加（因 token generation 按 BS=32 pad 运算）
> - 高并发场景推荐配置 B（75 tok/s at BS=32），kernel 优化有约 3% 提升
> - kernel 优化在长序列场景（>1024 tokens）预期有更大收益

### 7.3 编译和加载时间

| 阶段 | 配置 A (baseline) | 配置 B (全优化) |
|------|------|------|
| 编译（首次） | ~22 分钟 | ~30 分钟（NKI kernel HLO 生成较慢） |
| 权重预分片 (save_sharded_checkpoint=true) | ~20 分钟 | ~20 分钟 |
| 加载已编译模型 + 已分片权重 | ~5 分钟 | ~5 分钟 |

---

## 附录

### A. 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `modeling_minimax_m2_v3.py` | 添加 `MiniMaxM2QKNorm`，使用 `initialize_moe_module`，int32 dtype 修复 |
| `generation_minimax_m2_v4_demo.py` | 新建 v4 demo，支持 EP 和长上下文 |
| `constants.py` | 注册 `minimaxm2` 模型类型 |
| `vllm_minimax_m2_example.md` | vLLM 使用文档 |
| `vllm_minimax_m2_test.py` | vLLM Python API 测试脚本 |

### B. 调试技巧

1. **检查 NxDI 模型类是否正确加载**:
```python
from neuronx_distributed_inference.utils.constants import MODEL_TYPES
print(MODEL_TYPES.get("minimaxm2"))
```

2. **验证 dtype 一致性**:
```python
# 在 convert_hf_to_neuron_state_dict 中添加
for k, v in neuron_state_dict.items():
    if 'rank' in k:
        print(f"{k}: dtype={v.dtype}")
```

3. **查看 NKI kernel 支持的 dtype**:
```
支持: int32, float32, bfloat16, float16, int8, uint8, bool
不支持: int64 (torch.long)
```

### C. 已知限制

1. **NKI 不支持 int64**: 所有传递给 NKI kernel 的 index tensors 必须是 int32
2. **hybrid sharding 限制**: Token generation 不支持 EP (batch_size < 32 时)
3. **intermediate_size 限制**: 不能启用 `use_shard_on_intermediate_dynamic_while` 因为会 OOM

---

**更新日期**: 2026-03-27
**版本**: v4 + vLLM 集成验证
**技术栈**: AWS Trainium2, Neuron SDK, vLLM 0.16.0, vllm-neuron release-0.5.0, NeuronX Distributed Inference
