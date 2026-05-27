# Contrib Model: Qwen3-Coder-Next on AWS Trainium2

Optimized NxD Inference implementation for **Qwen3-Coder-Next** (`Qwen/Qwen3-Coder-Next`) on trn2.

## Model Information

- **HuggingFace ID:** [`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- **Model Type:** Hybrid DeltaNet + GQA + MoE (qwen3_next)
- **Total Parameters:** 80B (3B active per token)
- **License:** Apache 2.0

## Architecture Details

Qwen3-Coder-Next is a hybrid sparse model with identical core architecture to Qwen3.5-35B-A3B:

| Parameter | Qwen3-Coder-Next | Qwen3.5-35B-A3B |
|-----------|-----------------|-----------------|
| Total params | 80B | 35B |
| Active params | ~3B | ~3B |
| Hidden size | 2048 | 2048 |
| Layers | 48 (36 DeltaNet + 12 GQA) | 40 (30 DeltaNet + 10 GQA) |
| Attention heads | 16 Q / 2 KV | 16 Q / 2 KV |
| Head dim | 256 | 256 |
| Partial RoPE | 25% (64 dims) | 25% (64 dims) |
| Experts | 512, top-10 | 256, top-8 |
| Expert intermediate | 512 | 1024 |
| Shared experts | 1 (sigmoid gated) | 1 (sigmoid gated) |
| Context length | 262,144 | 131,072 |

## Hardware Requirements

- **Instance:** `trn2.3xlarge` (TP=4) or `trn2.48xlarge` (TP=8)
- **LNC Mode:** LNC=2 (default, 24 GB HBM per logical core)
- **Disk:** ~160 GB for model weights
- **Neuron SDK:** 2.30+
- **DLAMI:** `Deep Learning AMI Neuron (Ubuntu 24.04) 20260522`

## Quick Start

### 1. Download Model

```bash
pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-Coder-Next \
  --local-dir /mnt/models/Qwen3-Coder-Next/
```

### 2. Activate Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

### 3. Run Test

```bash
cd contrib/models/Qwen3-Coder-Next/
MODEL_PATH=/mnt/models/Qwen3-Coder-Next python test/integration/test_model.py
```

## NKI Kernels

This contrib includes custom NKI kernels:

| Kernel | Purpose | File |
|--------|---------|------|
| DeltaNet Recurrent | Token generation (sequential) | `nki_deltanet.py` |
| DeltaNet Chunked | Context encoding (parallel chunks) | `nki_deltanet_chunked.py` |
| Flash Attention d=256 | Standard attention (pipelined) | `nki_flash_attn_d256_pipe.py` |
| Flash Attention d=256 (simple) | Testing/reference | `nki_flash_attn_d256.py` |

## Configuration Notes

### SDK 2.30 MoE CTE Workaround

The `shard_hidden` MoE kernel is broken on SDK 2.30. Use `shard_on_block` with PING_PONG:

```python
blockwise_matmul_config={
    "block_size": 128,
    "use_shard_on_block_dynamic_while": True,
    "block_sharding_strategy": "PING_PONG",
}
```

### Float32 Router

MoE router uses float32 for accuracy. The fused TKG kernel is patched to use the ISA router fallback for mixed-dtype support.

## Provenance

Forked from `contrib/qwen3.5-35b-a3b-v2` (Qwen3.5-35B-A3B, identical architecture).
Core changes: dynamic layer_types generation, VL files removed, SDK 2.30 default config.
