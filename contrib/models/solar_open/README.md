# Contrib Model: Solar Open 100B MoE

NeuronX Distributed Inference implementation of [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B), a 100B Mixture-of-Experts language model.

## Model Information

- **HuggingFace ID:** `upstage/Solar-Open-100B`
- **Model Type:** Decoder-only MoE transformer
- **Architecture:** 128 routed experts + 1 shared expert per layer, top-8 routing with sigmoid activation
- **Parameters:** ~100B total, ~22B active per token
- **License:** Check HuggingFace model card

> **Note:** Solar Open is available in `transformers` 4.57+ as a built-in model type (`SolarOpenForCausalLM`). `trust_remote_code` is NOT required.

## Architecture Details

Solar Open is derived from the same author's GLM-4.5 MoE contrib (PR #58). Key differences from GLM-4.5 MoE:

| Property | Solar Open | GLM-4.5 MoE |
|----------|-----------|-------------|
| `partial_rotary_factor` | 1.0 (full RoPE) | < 1.0 (partial RoPE) |
| `attention_bias` | False | True |
| `use_qk_norm` | False | True |
| `first_k_dense_replace` | **0** (ALL layers MoE) | > 0 (some dense layers) |
| `rope_scaling` | `yarn` (factor 2.0, 128K context) | None |
| In `transformers` | Yes (4.57+) | Yes |

### MoE Configuration (100B model)

- `n_routed_experts`: 128
- `n_shared_experts`: 1
- `num_experts_per_tok`: 8 (top-8 routing)
- `n_group`: 1, `topk_group`: 1 (global top-8, no group constraint)
- `norm_topk_prob`: True
- `routed_scaling_factor`: 1.0
- Router: **sigmoid** + `e_score_correction_bias` (same as DeepSeek-V3/GPT-OSS)

### Expert Parallelism

EP support requires `n_group == 1` (which Solar Open satisfies). Recommended production config: `tp_degree=64` on trn2.48xlarge (128 NeuronCores).

## Hardware Requirements

| Configuration | Instance | TP Degree | Notes |
|--------------|----------|-----------|-------|
| Production (100B, full model) | **trn2.48xlarge** (128 NeuronCores) | 64 | Validated on SDK 2.29 |
| Development (unit tests only) | Any machine with Python 3.10+ | N/A | No Neuron hardware needed |

> **Note:** trn2.48xlarge is required for the full 100B model due to NEFF I/O constraints at 128 experts. Smaller instances (trn2.3xlarge) cannot fit the expert weights at lower TP degrees.

> **Note:** NxD Inference 0.9.x (SDK 2.29) drops trn1/inf2 support. trn2 only going forward.

## Usage

```python
import sys
sys.path.insert(0, "contrib/models/solar_open/src")

import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from solar_open.modeling_solar_open import (
    SolarOpenInferenceConfig,
    NeuronSolarOpenForCausalLM,
    load_solar_open_config,
)

model_path = "/path/to/upstage/Solar-Open-100B"
traced_model_path = "/path/to/traced_model"

neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    seq_len=4096,
    n_active_tokens=128,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.6, top_k=20, top_p=0.95
    ),
    fused_qkv=True,
    qkv_kernel_enabled=True,         # Attention NKI ON (recommended)
    qkv_nki_kernel_enabled=True,
    attn_kernel_enabled=True,
    moe_fused_nki_kernel_enabled=False,  # MoE NKI OFF (see Known Issues)
)

config = SolarOpenInferenceConfig(
    neuron_config,
    load_config=load_solar_open_config(model_path),
)

# Compile
model = NeuronSolarOpenForCausalLM(model_path, config)
model.compile(traced_model_path)

# Load and run
model = NeuronSolarOpenForCausalLM(traced_model_path)
model.load(traced_model_path)
```

See `examples/generation_solar_open_demo.py` for a full end-to-end example.

## Environment Setup (SDK 2.29)

```bash
# Activate the pre-installed NxD Inference venv on the Neuron DLAMI (Ubuntu 24.04, 20260410)
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

## Testing

### Unit Tests (CPU, no Neuron hardware required)

```bash
cd contrib/models/solar_open
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m pytest test/unit/ -v
```

### Integration Tests (requires Neuron hardware)

```bash
cd contrib/models/solar_open
python -m pytest test/integration/ -v --capture=tee-sys
```

Integration tests compile a 2-layer tiny random model and verify:
1. **Smoke test** — model compiles and loads without error
2. **Logit accuracy** — Neuron logits match HuggingFace CPU reference via `check_accuracy_logits_v2` (divergence_difference_tol=0.001)
3. **Context encoding** — forward pass completes without error

## Validation Results

**Tested with:** Reduced 2-layer config (`hidden_size=512`, `n_routed_experts=8`, random weights), TP=2, `bfloat16`
**Full model:** `upstage/Solar-Open-100B` on `trn2.48xlarge`, TP=64, SDK 2.29

| Test | Status | Notes |
|------|--------|-------|
| Model compilation (2-layer) | PASS | Reduced config, TP=2 |
| Model load | PASS | |
| Logit accuracy (`check_accuracy_logits_v2`) | PASS | `divergence_difference_tol=0.001` |
| Full model logit accuracy (100B, trn2.48xlarge) | PASS | 0 divergence, SDK 2.29 |
| Unit: router top-k (10 tests) | PASS | CPU-only |
| Unit: partial RoPE (9 tests) | PASS | CPU-only |
| Unit: decoder layer dispatch (11 tests) | PASS | CPU-only |

## Performance (SDK 2.29, trn2.48xlarge)

Validated on `Deep Learning AMI Neuron (Ubuntu 24.04) 20260410` with NxD Inference 0.9.x and neuronx-cc 2.24.

| Metric | SDK 2.29 | SDK 2.28 | Change |
|--------|----------|----------|--------|
| TKG throughput | **82.4 tok/s** | 12.6 tok/s | **6.54x faster** |
| TKG latency | 12.1 ms/tok | 79.4 ms/tok | 6.54x |
| Compile time | 11.2 min | 10.5 min | +7% |
| Model load time | 4.0 min | 3.8 min | +5% |
| `check_accuracy_logits_v2` | PASS (0 divergence) | N/A | Logit validation confirmed |

The 6.54x TKG improvement comes from `neuronx-cc` 2.24 compiler optimizations for NKI attention kernels on trn2 architecture, not from kernel code changes.

### Recommended NKI Kernel Configuration

| Kernel | Setting | Reason |
|--------|---------|--------|
| Attention NKI (`qkv_kernel_enabled`, `qkv_nki_kernel_enabled`) | **ON** | 82.4 tok/s — sole optimization path |
| MoE fused NKI (`moe_fused_nki_kernel_enabled`) | **OFF** | 71.9 tok/s — 13% slower than ISA path |

## Example Checkpoints

- [`upstage/Solar-Open-100B`](https://huggingface.co/upstage/Solar-Open-100B) — Full 100B model (128 routed experts + 1 shared, 64 layers)

## Known Issues

### MoE NKI Kernels Not Beneficial

The fused MoE NKI kernel (`moe_fused_nki_kernel_enabled`) compiles successfully on SDK 2.29 (the tripcount=1 issue from 2.28 is fixed), but produces **13% lower throughput** (71.9 tok/s vs 82.4 tok/s) compared to the ISA path.

**Root cause:** Solar Open's narrow expert dimension (moe_intermediate_size=1280) results in only 20 elements per TP shard at tp_degree=64 (1280/64=20). This is too small to benefit from NKI's SBUF tiling optimizations. Keep `moe_fused_nki_kernel_enabled=False`.

### Sigmoid Router and Fused TKG Kernel

The fused MoE TKG kernel's built-in router only supports softmax activation. Solar Open uses sigmoid routing. This model includes a runtime patch (`_patch_fused_tkg_for_sigmoid()`) that forces the ISA router fallback when the fused TKG kernel is active, ensuring correct routing behavior. No user action needed — the patch is applied automatically at import time.

## Compatibility Matrix

| Instance | NxDI Version | SDK | Status |
|----------|-------------|-----|--------|
| trn2.48xlarge | 0.9.x | 2.29 | **Validated** — 82.4 tok/s, logits verified |
| trn2.48xlarge | 0.7.x | 2.28 | Validated — 12.6 tok/s |
| trn1.32xlarge | 0.7.x | 2.28 | Unit tests only |
| trn1 / inf2 | 0.9.x | 2.29 | **Not supported** (NxDI 0.9.x drops trn1/inf2) |

## Maintainer

Contributed by: gmkim (lifelongeeek)

SDK 2.29 validation, NKI kernel optimization, and hf_adapter fix contributed by: jimburtoft

**Last Updated:** 2026-04-22
