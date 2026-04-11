# Contrib Model: MiniMax M2

NeuronX Distributed Inference implementation of [MiniMax/MiniMax-M2](https://huggingface.co/MiniMax/MiniMax-M2).

## Model Information

- **HuggingFace ID:** `MiniMax/MiniMax-M2`
- **Model Type:** Decoder-only MoE transformer
- **Architecture:** Custom MoE with sigmoid routing and e_score_correction_bias
- **Parameters:** 229B total, ~10B active per token
- **License:** Check HuggingFace model card

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Hidden Size | 3072 |
| Layers | 62 (all MoE, no dense) |
| Attention Heads | 48 Q / 8 KV (GQA) |
| Head Dim | 128 |
| Experts | 256 (top-8 routing) |
| Expert Intermediate | 1536 |
| Vocab Size | 200,064 |
| RoPE | Partial (rotary_dim=64, 50% of head_dim), theta=5M |
| Max Position | 196,608 |
| Weight Format | FP8 (float8_e4m3fn) with block-wise scales [128,128] |

Key features:
- **QK Norm**: RMSNorm applied before reshape on full Q/K projection output (per-rank, no all-reduce)
- **Partial RoPE**: Only first 64 of 128 dims use rotary encoding
- **Sigmoid Router**: With learnable `e_score_correction_bias` for expert selection (bias values ~8.0-9.5 dominate sigmoid scores 0-1)
- **SwiGLU experts**: gate (w1) + up (w3) with SiLU activation, down (w2)
- **FP8 dequantization**: Weights dequantized from float8_e4m3fn to BF16 at load time

## Test Results

Tested on trn2.48xlarge with SDK 2.29 (NxDI 0.9.0, torch-neuronx 2.9.0.2.13, NKI 0.3.0).

**Configuration**: `max_context_length=32, seq_len=64, TP=32, batch=1`

### Accuracy (8/8 correct)

| Prompt | First Token | Correct |
|--------|-------------|---------|
| The capital of France is | Paris | Yes |
| 1 + 1 = | 2 | Yes |
| The color of the sky is | blue | Yes |
| Water boils at | 100 | Yes |
| The largest planet in our solar system is | Jupiter | Yes |
| def fibonacci(n): | (code) | Yes |
| In 1969, humans first | set | Yes |
| The chemical formula for water is | H2O | Yes |

### Throughput

| Metric | Value |
|--------|-------|
| Average TPS | 50.3 tok/s |
| Per-token latency | 19.9 ms |
| Compile time | 8.5 min |
| Load time | 512 s |

## Prerequisites

- **Instance**: trn2.48xlarge (16 Neuron devices, LNC=2 -> 32 logical NeuronCores)
- **SDK**: 2.29+ (NxDI 0.9+, torch-neuronx 2.9+)
- **Weights**: Original FP8 checkpoint (~215 GB, 130 safetensor shards). Dequantized to BF16 at load time.
- **Disk**: 300+ GB EBS (for checkpoint + compiled model)

### HBM Constraints

At TP=32, MoE expert weights consume ~22 GB of 24 GB HBM per core (NxDI duplicates weights between CE and TKG NEFFs: 62 CE layers + 40 TKG layers). This limits context length to short sequences. Larger contexts require either:
- INT8 quantization to reduce weight memory
- Framework support for CE/TKG weight sharing (not yet available)

## Usage

```python
import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)

from src.modeling_minimax_m2 import NeuronMiniMaxM2ForCausalLM, MiniMaxM2InferenceConfig

model_path = "/path/to/MiniMax-M2/"
compiled_path = "/path/to/compiled/"

hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

neuron_config = MoENeuronConfig(
    tp_degree=32,
    batch_size=1,
    max_context_length=32,
    seq_len=64,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=False, global_topk=1
    ),
    enable_bucketing=False,
)

config = MiniMaxM2InferenceConfig(
    neuron_config=neuron_config,
    load_config=load_pretrained_config(hf_config=hf_config),
)

# Compile
model = NeuronMiniMaxM2ForCausalLM(model_path, config)
model.compile(compiled_path)

# Load
model = NeuronMiniMaxM2ForCausalLM(compiled_path, config)
model.load(compiled_path)

# Generate
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gen_model = HuggingFaceGenerationAdapter(model)
gen_config = GenerationConfig(
    max_new_tokens=20,
    do_sample=False,
    top_k=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

inputs = tokenizer("The capital of France is", return_tensors="pt", padding=True)
outputs = gen_model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    generation_config=gen_config,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# -> "The capital of France is Paris, and the capital of the United States is Washington, D.C. ..."
```

## vLLM Integration

MiniMax-M2 can be served via [vllm-neuron](https://github.com/aws-neuron/vllm-neuron). A patch is required to add MiniMax architecture support.

### Setup

```bash
# 1. Install vllm-neuron
pip install vllm-neuron

# 2. Apply the MiMo/MiniMax patch
cd /path/to/vllm-neuron
git apply /path/to/neuronx-distributed-inference/perf_test/vllm-neuron-mimo-minimax.patch
pip install -e .
```

### Serving

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/MiniMax-M2 \
    --tensor-parallel-size 64 \
    --max-model-len 1024 \
    --max-num-seqs 256 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust_remote_code \
    --additional-config '{
        "override_neuron_config": {
            "tp_degree": 64,
            "logical_nc_config": 2,
            "flash_decoding_enabled": false,
            "sequence_parallel_enabled": true,
            "glu_mlp": true,
            "moe_mask_padded_tokens": true,
            "disable_numeric_cc_token": true,
            "router_config": {"act_fn": "sigmoid", "dtype": "float32"},
            "moe_tp_degree": 1,
            "moe_ep_degree": 64,
            "batch_size": 256,
            "ctx_batch_size": 1,
            "tkg_batch_size": 256,
            "max_context_length": 1024,
            "seq_len": 1024,
            "is_continuous_batching": true,
            "fused_qkv": false,
            "enable_bucketing": true,
            "normalize_top_k_affinities": true,
            "use_index_calc_kernel": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": true,
                "skip_dma_token": true
            },
            "scratchpad_page_size": 1024
        }
    }'
```

### Key vLLM Patch Changes

The patch (`perf_test/vllm-neuron-mimo-minimax.patch`) modifies vllm-neuron to:
- Pass `hf_config` from vLLM to NxDI (required for `trust_remote_code` models)
- Replace `AutoModelForCausalLM.from_pretrained` with `snapshot_download` for model loading

See `perf_test/2_bench_minimax_m2.sh` for full benchmark configurations with BS=1/256.

## Implementation Notes

### Key Design Decisions

1. **RouterTopKWithBias**: Uses `nn.Parameter` (not `register_buffer`) with non-uniform initialization (`torch.arange`, bfloat16). Uniform init values (zeros/ones) allow XLA to prove the bias add is a no-op and eliminate it from the HLO, making the real bias values impossible to load at inference time.

2. **Direct MoE construction**: The MoE module is constructed directly (RouterTopKWithBias + ExpertMLPsV2 + MoE) rather than using `initialize_moe_module` + router replacement. This ensures the model's parameter tree matches the weight converter's key mapping.

3. **Neuron-native QK normalization**: Uses `RmsNorm.apply` from `neuronx_distributed_inference.modules.custom_calls` instead of hand-rolled PyTorch RMSNorm. Hand-rolled pow/mean/rsqrt compiles into different HLO in context encoding vs token generation NEFFs, producing incorrect token generation results.

4. **expert_index dtype**: Must be `torch.long` (not `torch.int32`). SDK 2.29's ExpertMLPsV2 indexing requires S64 element type.

5. **Weight converter maps e_score_correction_bias** to `router.e_score_correction_bias` (not deleted). The bias values (~8.0-9.5) dominate sigmoid scores (0-1) and are critical for correct expert selection.

### NKI Kernel Compatibility

At TP=32, the expert intermediate dimension per rank is `1536/32 = 48`, which falls below the 128-element minimum required by fused MoE NKI kernels. This means the highest-impact kernels are structurally incompatible at the only viable TP degree.

| Kernel | Compatible? | Impact | Notes |
|--------|-------------|--------|-------|
| Fused MoE expert MLP | No (I=48 < 128) | Very High | Blocked at TP=32 |
| Router Top-K | Yes | Low | Works with sigmoid + bias |
| Attention TKG | Yes | Moderate | Can be enabled via flag |
| Fused QKV | Partially | Low | Small matmul at TP=32 |

Potential paths to enable fused MoE NKI kernels:
- **EP=4 at TP=8**: I/TP = 1536/8 = 192 (NKI-compatible), but requires 32 total ranks (multi-node)
- **INT8 quantization + lower TP**: May allow TP=4 (I=384, NKI-compatible) if weights fit in HBM
- **NKI kernel enhancement**: Relax the 128-element alignment for sub-128 dimensions

## Compatibility Matrix

| Instance/Version | SDK 2.29 (PyTorch 2.9) | SDK 2.28 | Earlier |
|------------------|------------------------|----------|---------|
| Trn2 (trn2.48xlarge) | Tested (50.3 tok/s) | Tested (43.5 tok/s) | Not tested |
| Trn1 | Not supported (insufficient NeuronCores) | Not supported | Not supported |
| Inf2 | Not supported | Not supported | Not supported |

SDK 2.29 delivers +15.6% throughput improvement over SDK 2.28 with zero code changes.

## Testing

```bash
pytest contrib/models/MiniMax-M2/test/integration/test_model.py -v
```

## Example Checkpoints

* [MiniMax/MiniMax-M2](https://huggingface.co/MiniMax/MiniMax-M2) (FP8 original, dequantized to BF16 at load time)

## Maintainer

Henan Wan (whn09)

**Last Updated:** 2026-04-11
