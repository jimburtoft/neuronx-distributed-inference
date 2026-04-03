# Contrib Model: NVIDIA Nemotron-3-Nano-30B-A3B-BF16

NeuronX Distributed Inference implementation of NVIDIA's [Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) (NemotronHForCausalLM).

## Model Information

- **HuggingFace ID:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- **Model Type:** Hybrid Mamba2/Attention with MoE
- **Parameters:** 31.58B total (~3.5B active per token with top-6 routing)
- **Architecture:** Mamba-2 SSM + GQA Attention + MoE (128 routed + 1 shared expert, top-6 sigmoid routing)
- **License:** NVIDIA Open Model License

## Architecture Details

| Property | Value |
|----------|-------|
| Hidden size | 2688 |
| Layers | 52 (23 Mamba-2 + 23 MoE + 6 GQA Attention) |
| Layer pattern | `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` |
| Attention heads | 32 Q / 2 KV (16:1 GQA ratio) |
| Attention head_dim | 128 (explicit, not hidden_size // num_heads) |
| RoPE theta | 10000 |
| MoE experts | 128 routed + 1 shared per layer, top-6 sigmoid routing |
| MoE routing | Sigmoid with `e_score_correction_bias` (DeepSeek-V3 style) |
| MoE activation | relu2 (non-gated MLP: up_proj + down_proj only) |
| Mamba-2 heads | 64 (head_dim=64) |
| SSM state size | 128 |
| Conv kernel | 4 |
| Mamba chunk size | 128 |
| Vocab size | 131072 |
| tie_word_embeddings | False |
| Model size (BF16) | 58.83 GB |

## Implementation Notes

### Single-Block-Per-Layer Design

Unlike Granite4's dual-block layers, Nemotron uses a single block per layer. Each of the 52 layers has ONE mixer: either Mamba-2 OR MoE OR Attention. The layer type is determined by a `hybrid_override_pattern` string in the HF config. Simple residual: `output = residual + mixer(norm(hidden_states))`.

### Mamba State Persistence

The key challenge for hybrid architectures is persisting Mamba2 recurrent state (conv_state and ssm_state) across XLA graph executions during autoregressive decode. We solve this using the same `input_output_aliases` mechanism that NxDI uses for KV cache:

1. `NeuronNemotronModel` maintains a `nn.ParameterList` (`mamba_states`) containing conv_state and ssm_state buffers for each of the 23 Mamba layers (46 parameters total)
2. `NemotronDecoderModelInstance` (extends `DecoderModelInstance`) adds these parameters to `input_output_aliases` after the standard KV cache entries
3. `NeuronNemotronMamba2Layer.forward()` accepts and returns state as explicit tensor arguments
4. The output list is: `[logits, K0, V0, ..., conv_state_0, ssm_state_0, ...]`
5. Non-attention layers return dummy `(batch_size,1,1,1)` KV tuples to satisfy `KVCacheManager`

### MoE with TP Sharding

The 128-expert MoE layers use tensor-parallel sharding on the intermediate dimension:
- Expert weights stored at per-TP-rank size: `up (E, H, I/TP)`, `down (E, I/TP, H)`
- Shared expert uses `ColumnParallelLinear(gather_output=False)`
- Delayed all-reduce after combining routed + shared expert outputs
- Sigmoid routing with `e_score_correction_bias` (top-6 selection per token)
- relu2 activation (non-gated MLP)
- Per-expert Python loop (avoids HBM allocation failure from batched BMM)

### Manual Depthwise Conv1d

SDK 2.28 has a compiler issue (TEN404) where the auto-inserted NKI Conv1d kernel crashes on `seq_len=1` (decode path). We work around this by implementing depthwise convolution manually using weight parameters and a loop over kernel positions.

### Gated RMSNorm with Per-Group Normalization

Nemotron's Mamba layers use gated RMSNorm with `norm_before_gate=False` and per-group normalization (`group_size = intermediate_size / n_groups = 4096 / 8 = 512`). The CUDA Triton kernel (`rmsnorm_fn`) handles this correctly, but the PyTorch fallback in the original HF code ignores `group_size` entirely — causing incorrect normalization and incoherent decode output. Our `NemotronRMSNormGated` implements correct per-group normalization for all backends.

### NKI Selective Scan (Optional)

The codebase includes an optional O(L) NKI selective scan kernel (ported from Granite4 contrib) using `nisa.tensor_tensor_scan`. However, benchmarking showed that at `max_context_length=128`, the quadratic O(L^2) parallel scan is actually **23x faster for TTFT** (211 ms vs 4932 ms) and **3x faster for decode** (18.3 vs 6.6 tok/s). This is because the NKI kernel invocation overhead per layer (23 Mamba layers) dominates at short sequence lengths. The quadratic scan is the default (`USE_NKI_SCAN = False`). Set `USE_NKI_SCAN = True` to experiment with the NKI path for longer sequences.

## Validation Results

**Validated:** 2026-04-03
**Configuration:** TP=4, batch_size=1, seq_len=2048, max_context_length=128, bfloat16
**Instance:** trn2.3xlarge (LNC=2)
**SDK:** Neuron SDK 2.28, PyTorch 2.9

### Prefill Accuracy (vs HF BF16 CPU, 11 prompts)

| Metric | Value |
|--------|-------|
| **Average cosine similarity** | **0.968** |
| **Argmax match rate** | **9/11 (82%)** |
| Min cosine | 0.931 |
| Max cosine | 0.995 |

### Decode Quality (greedy, 20 tokens)

| Prompt | First Token | Quality |
|--------|-------------|---------|
| "The capital of France is" | Paris | Correct factual answer |
| "Albert Einstein was born in" | 1885 | Approximately correct |
| "1 + 1 =" | 2 | Correct |

Both Neuron and HF reference produce correct first tokens, followed by greedy repetition patterns typical of base (non-instruct) models.

### Inference Performance

| Configuration | TTFT (ms) | Decode (tok/s) | TPOT (ms) |
|--------------|-----------|----------------|-----------|
| **BS=1, seq_len=2048** | 211 | 18.3 | 54.6 |
| **BS=2, seq_len=2048** | 263 | 22.0 | — |

All measurements on trn2.3xlarge (TP=4, LNC=2, BF16).

| Metric | Value (BS=1) |
|--------|-------|
| Model load time | 16.9 s |
| HBM per core (est.) | ~14.7 GB / 24 GB (61%) |

TPOT is extremely stable at BS=1: P50-P99 spread < 0.3 ms.

### Compilation

| Metric | Value |
|--------|-------|
| Compile time | ~32 min (trn2.3xlarge, first compile) |
| Compiler flags | `-O1 --auto-cast=none --enable-mixed-precision-accumulation` |
| Compiler RAM | >88 GB (requires 128 GB swap on NVMe) |

## Usage

```python
from transformers import AutoTokenizer, AutoConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from modeling_nemotron_h import NeuronNemotronForCausalLM, NemotronHInferenceConfig

MODEL_PATH = "/path/to/nemotron-30b/"
COMPILED_PATH = "/path/to/compiled_model/"

# Configure
neuron_config = MoENeuronConfig(
    tp_degree=4,
    batch_size=1,
    max_context_length=128,
    seq_len=2048,
    on_device_sampling_config=None,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    torch_dtype="bfloat16",
    save_sharded_checkpoint=True,
)

hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = NemotronHInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(hf_config=hf_config),
)

# Compile (first time only, ~32 min on trn2.3xlarge)
model = NeuronNemotronForCausalLM(MODEL_PATH, config)
model.compile(COMPILED_PATH)

# Load compiled model
model = NeuronNemotronForCausalLM(MODEL_PATH, config)
model.load(COMPILED_PATH)

# Generate
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
inputs = tokenizer("The capital of France is", return_tensors="pt")

gen_model = HuggingFaceGenerationAdapter(model)
outputs = gen_model.generate(
    inputs.input_ids,
    attention_mask=torch.ones_like(inputs.input_ids),
    max_new_tokens=50,
    do_sample=False,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Prerequisites

### HuggingFace Model

The model requires `trust_remote_code=True` because it uses a custom architecture class (`NemotronHForCausalLM`). Download from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir /mnt/models/nemotron-30b
```

### HF Model Patches Required

The original HuggingFace `modeling_nemotron_h.py` has several issues that must be patched for non-CUDA execution:

1. **CUDA import fallbacks** — `rmsnorm_fn`, `selective_state_update`, `causal_conv1d` imports fail without CUDA. Wrap in try/except.
2. **Per-group RMSNorm** — `MambaRMSNormGated.forward()` PyTorch fallback ignores `group_size`, producing incorrect normalization. Add per-group CPU fallback.
3. **Cache attribute bugs** — `HybridMambaAttentionDynamicCache` accesses `.device` on Python lists instead of tensor elements.
4. **`torch.cuda.stream`** — Replace with `if True:` for non-CUDA backends.

### Swap Space

The 52-layer model produces ~1.16M HLO instructions. The neuronx-cc compiler needs >88 GB RAM. Configure 128 GB swap on NVMe:

```bash
sudo fallocate -l 128G /mnt/models/swapfile
sudo chmod 600 /mnt/models/swapfile
sudo mkswap /mnt/models/swapfile
sudo swapon /mnt/models/swapfile
```

## Compatibility Matrix

| Instance Type | SDK 2.28 | SDK 2.27 |
|--------------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | Validated | Not tested |
| trn2.48xlarge (TP=4, LNC=2) | Not tested | Not tested |
| trn1.32xlarge | Not tested | Not tested |

## Example Checkpoints

* [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)

## Testing Instructions

```bash
# Run standard integration tests (compile + behavioral accuracy + throughput)
# First compile takes ~32 min; subsequent runs load from cache
pytest test/integration/test_model.py -v

# Run all tests including logit validation against CPU BF16 reference
# Requires ~60 GB free CPU RAM to load the 30B HF model
pytest test/integration/test_model.py -v --run-slow

# Or run directly (compiles if needed)
python test/integration/test_model.py

# Enable logit validation when running directly
RUN_LOGIT_VALIDATION=1 python test/integration/test_model.py

# Quick smoke test (requires pre-compiled model)
python test_smoke.py
```

**Environment variables:**
- `NEMOTRON_MODEL_PATH` — Path to HF model (default: `/mnt/models/nemotron-30b`)
- `NEMOTRON_COMPILED_PATH` — Path for compiled model (default: `/mnt/models/nemotron_compiled_contrib`)
- `RUN_LOGIT_VALIDATION` — Set to `1` to enable logit validation when running directly

## Known Issues

1. **Maximum context length is 128.** The 23 Mamba layers require persistent state buffers (conv_state + ssm_state) per core. At longer context lengths, per-core I/O tensors exceed the 24 GB HBM bank limit.
2. **Maximum batch size is 2 on trn2.3xlarge (LNC=2).** BS=4 compiles successfully but exceeds HBM during model load (CE model allocation fails on 24 GB/core). BS=4 would require trn2.48xlarge or LNC=1 (not tested).
3. **Maximum seq_len validated is 2048.** Higher seq_len (4096, 8192) not tested due to compilation disk space constraints.
4. **No on-device sampling tested.** Current validation uses raw logits (`on_device_sampling_config=None`).
4. **Per-expert loop for MoE.** The 128-expert routing uses a Python loop over selected experts. A fused NKI MoE kernel would improve throughput but requires relu2 activation support not currently available.
5. **Conv1d workaround.** Manual depthwise convolution avoids TEN404 but may be slower than native conv1d once the SDK issue is fixed.
6. **Base model behavior.** This is a base (non-instruct) model. Greedy decoding produces repetitive output after the first few correct tokens, consistent with the HF reference.

## HuggingFace Model Issues Found

During development, we discovered and documented several issues in the original HuggingFace `modeling_nemotron_h.py`:

1. **`MambaRMSNormGated` ignores `group_size`** — The PyTorch fallback normalizes over the full hidden dimension instead of per-group. The CUDA Triton kernel is correct. This causes incoherent decode output on CPU/non-CUDA backends.
2. **`HybridMambaAttentionDynamicCache` attribute bugs** — `self.ssm_states` and `self.conv_states` are Python lists but accessed as tensors (`.device`, `.zero_()`).
3. **Cache key mismatch** — `prepare_inputs_for_generation()` stores cache under `"past_key_values"` but `forward()` expects `"cache_params"`, preventing proper state persistence in HF's `generate()`.

## Source Files

| File | Description | Lines |
|------|-------------|-------|
| `src/modeling_nemotron_h.py` | Full model implementation (config, Mamba layer, attention, MoE, NKI scan, model wrapper, state dict conversion) | ~1893 |
| `src/__init__.py` | Public exports | ~27 |
| `test/integration/test_model.py` | Integration tests (compile, load, generate, logit validation, throughput) | ~441 |
| `test_smoke.py` | Quick smoke test for pre-compiled model | ~79 |

## Maintainer

Jim Burtoft ([@jimburtoft](https://github.com/jimburtoft))

**Last Updated:** 2026-04-03
