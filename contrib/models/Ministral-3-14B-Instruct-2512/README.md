# Contrib Model: Ministral-3-14B-Instruct-2512 (Leanstral)

NeuronX Distributed Inference contrib for Ministral-3-14B-Instruct-2512 on AWS Trainium 2.
This model uses Mistral's 14B dense GQA text decoder with 8 KV heads, served via the
LlamaForCausalLM code path in NxDI with custom NKI kernels for multi-KV-head attention.

## Model Information

- **HuggingFace ID:** `mistralai/Ministral-3-14B-Instruct-2512`
- **Architecture:** Dense GQA (runs as LlamaForCausalLM via hf-overrides)
- **Parameters:** 14B (40 layers, hidden=5120, 32 Q / 8 KV heads, d_head=128)
- **Vocab:** 32768 (text-only extraction from VL checkpoint)
- **License:** Check HuggingFace model card (gated access)

## Architecture Details

- 40 layers, hidden\_size=5120 (mapped to 3584 for text extraction), intermediate\_size=16384
- num\_attention\_heads=32, num\_kv\_heads=8, head\_dim=128, rope\_theta=1e9
- At TP=4: q\_heads\_per\_rank=8, kv\_heads\_per\_rank=2
- Original checkpoint is FP8 E4M3 — dequantized to BF16 via `extract_text_model.py`

### Key Adaptations for SDK 2.29

1. **LlamaForCausalLM code path**: vLLM 0.16 auto-promotes MistralForCausalLM to Pixtral.
   We use `--hf-overrides '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}'`
   to force the Llama code path, which handles the GQA sharding natively.

2. **Multi-KV-head TKG kernel**: Modified `attention_block_tkg` kernel supporting
   kv\_heads\_per\_rank > 1 via virtual-batch approach. Installed via `setup_patches.py`.

3. **FP8→BF16 text extraction**: The HuggingFace checkpoint is a VL model with FP8 weights.
   `extract_text_model.py` strips vision keys, dequantizes FP8→BF16, fixes tokenizer issues,
   and writes a clean text-only checkpoint.

4. **NKI 0.3.0 compatibility**: V cache update uses HBM-based DMA instead of SBUF nc_transpose
   (which exceeds `gemm_stationary_fmax=128` when `kv_heads*d_head=256`).

## Prerequisites

- **SDK 2.29** (neuronx-cc >= 2.24, neuronx-distributed-inference >= 0.9, vLLM 0.16 + vllm-neuron 0.5)
- **trn2.3xlarge** (TP=4, LNC=2, 96 GB HBM)
- **Model checkpoint**: `mistralai/Ministral-3-14B-Instruct-2512` from HuggingFace (gated)
- **Disk**: ~300 GB EBS for checkpoint + compiled model artifacts

### Environment Setup

```bash
# Activate pre-installed vLLM 0.16 environment (SDK 2.29)
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

# Install aiohttp for benchmark script
pip install aiohttp
```

## Quick Start

### Step 1: Download Model

```bash
huggingface-cli download mistralai/Ministral-3-14B-Instruct-2512 \
  --local-dir /home/ubuntu/models/Ministral-3-14B-Instruct-2512
```

### Step 2: Extract Text-Only BF16 Checkpoint

```bash
python src/extract_text_model.py \
  --input-dir /home/ubuntu/models/Ministral-3-14B-Instruct-2512 \
  --output-dir /home/ubuntu/models/Ministral-3-14B-text-bf16
```

This produces a ~27 GB checkpoint with:
- 6 safetensors shards (BF16, vision keys removed, `language_model.` prefix stripped)
- Fixed `tokenizer_config.json` (removes Pixtral processor references)
- Proper `config.json` for LlamaForCausalLM

### Step 3: Apply Runtime Patches

```bash
python src/setup_patches.py
```

Applies 6 patches to the installed NxDI/nkilib packages:
1. `rms_norm_eps` pass-through in model base
2. nkilib QKV kernel epsilon guard
3. neuronxcc QKV kernel epsilon guard
4. `convert_state_dict_to_fused_qkv` fix for non-standard head counts
5. Fused RMSNorm config support
6. Multi-KV TKG kernel + NKI 0.3.0 V cache fix + attention adapter

### Step 4: Launch vLLM Server

```bash
export NEURON_CC_FLAGS="--auto-cast=matmult"

python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/Ministral-3-14B-text-bf16 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --block-size 8 \
  --no-enable-prefix-caching \
  --port 8000 \
  --hf-overrides '{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}' \
  --additional-config '{"override_neuron_config": {"fused_qkv": true, "qkv_nki_kernel_enabled": true, "qkv_kernel_enabled": true, "attn_block_tkg_nki_kernel_enabled": true, "attn_block_tkg_nki_kernel_cache_update": true}}'
```

First launch compiles the model (~5 minutes). Subsequent launches use the NCC cache.

### Step 5: Query

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "leanstral",
       "messages": [{"role": "user", "content": "What is the capital of France?"}],
       "max_tokens": 256}'
```

## Performance Results

Measured on trn2.3xlarge (TP=4, LNC=2, SDK 2.29) via vLLM 0.16:

### vLLM Serving — QKV NKI Kernel (baseline, no TKG)

| Workload | Conc | TTFT P50 (ms) | tok/s P50 | TPOT P50 (ms) | E2E P50 (ms) |
|----------|------|---------------|-----------|---------------|--------------|
| short-short (128/128) | 1 | 100.6 | 63.3 | 15.8 | 2106.9 |
| short-short (128/128) | 4 | 200.0 | 58.5 | 15.9 | 2465.7 |
| short-long (128/512) | 1 | 101.4 | 62.8 | 15.9 | 8234.2 |
| short-long (128/512) | 4 | 200.0 | 62.3 | 15.9 | 8498.9 |
| long-short (2048/128) | 1 | 303.6 | 57.4 | 17.4 | 2514.7 |
| long-short (2048/128) | 4 | 609.2 | 50.9 | 17.3 | 3400.3 |
| long-long (2048/512) | 1 | 304.0 | 57.7 | 17.3 | 9156.5 |
| long-long (2048/512) | 4 | 608.9 | 55.9 | 17.3 | 10053.9 |

### vLLM Serving — Full TKG Kernel (fused attention + cache update)

| Workload | Conc | TTFT P50 (ms) | tok/s P50 | TPOT P50 (ms) | E2E P50 (ms) |
|----------|------|---------------|-----------|---------------|--------------|
| short-short (128/128) | 1 | 100.1 | 56.8 | 17.6 | 2335.1 |
| short-short (128/128) | 4 | 197.0 | 54.6 | 17.6 | 2613.9 |
| short-long (128/512) | 1 | 100.3 | 54.8 | 17.7 | 9299.9 |
| short-long (128/512) | 4 | 197.6 | 54.2 | 17.7 | 9557.2 |
| long-short (2048/128) | 1 | 302.9 | 52.2 | 19.2 | 2737.9 |
| long-short (2048/128) | 4 | 606.5 | 46.6 | 19.1 | 3618.1 |
| long-long (2048/512) | 1 | 303.3 | 52.1 | 19.2 | 10103.4 |
| long-long (2048/512) | 4 | 607.5 | 50.4 | 19.2 | 10991.7 |

**Notes:**
- QKV NKI kernel (baseline) achieves higher throughput in vLLM serving mode because it avoids
  the extra V cache HBM roundtrip required by the TKG kernel's multi-KV-head adaptation.
- The TKG kernel's architectural advantage (single fused kernel per layer) manifests more in
  direct model inference (213.7 tok/s at BS=4 on SDK 2.28) where vLLM scheduling overhead
  is absent.
- For vLLM serving, the recommended config uses both QKV NKI kernel and TKG for correctness
  validation; production deployments may prefer QKV-only for peak throughput.

### SDK 2.28 Reference (direct model inference, no vLLM)

| Platform | Config | Text tok/s | VL tok/s |
|----------|--------|-----------|---------|
| **trn2.3xlarge** | BS=4, fused QKV+TKG | **213.7** | **199.9** |
| p5.48xlarge | 1x H100 80GB, FP8 | 140.3 | 139.4 |

## Known Limitations

1. **TKG V cache overhead**: The multi-KV-head TKG kernel routes V cache updates through HBM
   (to work around NKI 0.3.0 `gemm_stationary_fmax=128` constraint), adding ~10% TPOT
   overhead vs the baseline QKV-only path.

2. **KVDP not supported**: KV data parallelism is not compatible with the multi-KV-head
   kernel path.

3. **FP8 checkpoint**: The original checkpoint uses FP8 E4M3 weights. These are dequantized
   to BF16 during extraction. Runtime FP8 inference is not currently supported.

4. **Pixtral auto-promotion**: vLLM 0.16 auto-promotes Mistral models to Pixtral even with
   tokenizer fixes. The `--hf-overrides` flag is mandatory to force LlamaForCausalLM.

5. **Text-only**: This contrib extracts and serves only the text decoder. Vision-language
   inference requires the full VL model and additional patches not included here.

## Compatibility Matrix

| Instance | SDK 2.29 | SDK 2.28 | Earlier |
|----------|----------|----------|---------|
| trn2.3xlarge (TP=4) | **Tested** | Tested (prior version) | Not supported |
| trn2.48xlarge | Not tested | Not tested | Not tested |
| trn1 / inf2 | Not supported | Not supported | Not supported |

## Source Files

| File | Description |
|------|-------------|
| `src/setup_patches.py` | SDK 2.29 runtime patch installer (6 patches) |
| `src/extract_text_model.py` | FP8→BF16 text-only checkpoint extraction |
| `src/attention_block_tkg_multi_kv.py` | Multi-KV-head TKG kernel (NKI 0.3.0 compatible) |
| `src/multi_kv_adapter.py` | TKG kernel adapter for attention_base.py |
| `src/fix_nki030.py` | NKI 0.3.0 compatibility fixes |
| `src/modeling_leanstral.py` | Legacy model class (SDK 2.28, reference only) |
| `src/patch_native_multi_kv.py` | Legacy adapter (SDK 2.28, reference only) |
| `bench.py` | Async streaming benchmark script |
| `test/integration/test_model.py` | Integration test |

## Upstream NxDI Gaps

This contrib identifies NxDI gaps that would benefit from upstream support:

1. **Multi-KV-head TKG kernel** — the bundled kernel hardcodes kv\_heads=1. The nki-library
   kernel fork adds `n_kv_heads` parameter with virtual-batch dispatch.
2. **Fused QKV conversion** — `convert_state_dict_to_fused_qkv` assumes standard Llama head
   ratios; non-standard ratios (32Q/8KV at TP=4) need a fix to compute interleave groups.
3. **RMS norm epsilon** — NxDI model base doesn't pass `rms_norm_eps` from config, defaulting
   to 1e-5 which differs from Mistral's 1e-5 (same in this case, but other models differ).

## Maintainer

Leanstral Project

**Last Updated:** 2026-04-21
