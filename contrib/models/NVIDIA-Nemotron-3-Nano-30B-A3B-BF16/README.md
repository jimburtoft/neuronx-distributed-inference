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

### MoE with TP Sharding and Sparse Expert Dispatch

The 128-expert MoE layers use tensor-parallel sharding on the intermediate dimension:
- Expert weights stored at per-TP-rank size: `up (E, H, I/TP)`, `down (E, I/TP, H)`
- Shared expert uses `ColumnParallelLinear(gather_output=False)`
- Delayed all-reduce after combining routed + shared expert outputs
- Sigmoid routing with `e_score_correction_bias` (top-6 selection per token)
- relu2 activation (non-gated MLP)

**Dual-path MoE dispatch** optimizes decode throughput by 3.8x:
- **Decode path** (`batch_tokens <= 2`): Sparse dispatch using `torch.index_select` to load only the 6 selected experts' weights, followed by `torch.bmm`. This reduces HBM bandwidth from 640 MB → 30 MB per MoE layer.
- **Prefill path** (`batch_tokens > 2`): Dense per-expert Python loop over all 128 experts (sparse dispatch creates HLO graph explosion at `seq_len=128`).
- NxDI traces CE and TG as separate XLA graphs, so `batch_tokens` is a static compile-time constant. The `if/else` is resolved during tracing — no dynamic control flow at runtime.
- Output is **bit-for-bit identical** between sparse and dense paths (verified 30/30 tokens).

### Manual Depthwise Conv1d

The auto-inserted NKI Conv1d kernel has previously crashed on `seq_len=1` (decode path) with compiler error TEN404. We implement depthwise convolution manually using explicit weight parameters and a loop over kernel positions to avoid this dependency.

### Gated RMSNorm with Per-Group Normalization

Nemotron's Mamba layers use gated RMSNorm with `norm_before_gate=False` and per-group normalization (`group_size = intermediate_size / n_groups = 4096 / 8 = 512`). The CUDA Triton kernel (`rmsnorm_fn`) handles this correctly, but the PyTorch fallback in the original HF code ignores `group_size` entirely — causing incorrect normalization and incoherent decode output. Our `NemotronRMSNormGated` implements correct per-group normalization for all backends.

### NKI Selective Scan (Optional)

The codebase includes an optional O(L) NKI selective scan kernel (ported from Granite4 contrib) using `nisa.tensor_tensor_scan`. It is off by default (`USE_NKI_SCAN = False`) because the head-grouped quadratic scan and the chunked NKI SSD kernel (see below) are faster for the sequence lengths this model targets.

## Validation Results

**Instance:** trn2.3xlarge (LNC=2, TP=4)
**SDK:** Neuron SDK 2.31, DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`
**Configuration:** batch_size=1, bfloat16

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

### Chunked NKI SSD end-to-end correctness

Bit-exact A/B token comparison between the chunked NKI SSD path and the head-grouped quadratic scan on the same model, greedy decoding:

| ctx | Chunks | Tokens generated | Match |
|-----|--------|------------------|-------|
| 128 | 1 | 50 | **50/50 identical** |
| 1024 | 8 | 30 | **30/30 identical** |

### Inference Performance

All measurements on trn2.3xlarge (TP=4, LNC=2, BF16) with SDK 2.31 (`Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`). BS=1, greedy decoding, single warmup, 50-token generation. Correct " Paris" first token at every ctx.

#### Default path: head-grouped quadratic scan

Ships with the model. No environment variable required.

| Configuration | TTFT (ms) | Decode (tok/s) | TPOT (ms) | Load (s) |
|--------------|-----------|----------------|-----------|----------|
| BS=1, seq=2048, ctx=128 | **67.5** | **105.66** | **9.46** | 56.5 |
| BS=1, seq=2048, ctx=256 | 124.6 | 103.38 | 9.67 | 76.7 |
| BS=1, seq=2048, ctx=512 | 225.1 | 102.62 | 9.74 | 81.0 |
| BS=1, seq=2048, ctx=640 | 425.3 | 105.64 | 9.47 | 67.9 |
| BS=1, seq=2048, ctx=768 | 514.7 | 106.66 | 9.38 | 73.5 |
| BS=1, seq=2048, ctx=1024 | 459.8 | 105.61 | 9.47 | 59.7 |
| **BS=1, seq=4096, ctx=2048** | **1081.7** | **102.06** | **9.80** | 87.1 |

Decode throughput is essentially flat at ~103-106 tok/s across the full ctx range.

#### Optional path: chunked NKI SSD scan (`USE_CHUNKED_NKI_SCAN=1`)

Opt-in via environment variable. Uses the cheap-ops-only chunked NKI SSD kernel at `src/nki_kernels/nki_mamba2_ssd_chunked.py`. Faster prefill at every context length AND extends the max context on trn2.3xlarge from 2048 to 8192.

| Configuration | TTFT (ms) | Decode (tok/s) | TPOT (ms) | Load (s) |
|--------------|-----------|----------------|-----------|----------|
| BS=1, seq=2048, ctx=128 | **63.2** | 104.34 | 9.58 | 52.7 |
| BS=1, seq=2048, ctx=512 | 172.8 | 102.30 | 9.77 | 64.5 |
| BS=1, seq=2048, ctx=1024 | 325.2 | 103.16 | 9.69 | 83.8 |
| BS=1, seq=4096, ctx=2048 | 633.8 | 99.39 | 10.06 | 101.0 |
| **BS=1, seq=8192, ctx=4096** | **1450.1** | 100.99 | 9.90 | 110.9 |
| **BS=1, seq=16384, ctx=8192** | **2503.4** | 101.29 | 9.87 | 151.5 |

**Correctness**: bit-for-bit identical token sequences vs the default quadratic path under greedy decoding (50/50 tokens at ctx=128, 30/30 tokens at ctx=1024). See Validation Results above.

**Delta vs default quadratic path**:

| ctx | Default TTFT | Chunked TTFT | Delta |
|-----|--------------|--------------|-------|
| 128 | 67.5 ms | **63.2 ms** | 6% faster |
| 512 | 225.1 ms | **172.8 ms** | 23% faster |
| 1024 | 459.8 ms | **325.2 ms** | 29% faster |
| 2048 | 1081.7 ms | **633.8 ms** | 41% faster |
| 4096 | not available | **1450.1 ms** | new capability |
| 8192 | not available | **2503.4 ms** | new capability |

Decode throughput unchanged (~100-104 tok/s) in both paths.

**Enabling**:

```bash
export USE_CHUNKED_NKI_SCAN=1
python your_script.py
```

Or set the flag directly in `modeling_nemotron_h.py`:

```python
USE_CHUNKED_NKI_SCAN = os.environ.get("USE_CHUNKED_NKI_SCAN", "0") == "1"
```

### Compilation

| Metric | Value |
|--------|-------|
| Compiler flags | `-O1 --auto-cast=none --enable-mixed-precision-accumulation --model-type transformer` |
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

# Compile (first time only)
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
3. **Cache attribute issues** — `HybridMambaAttentionDynamicCache` accesses `.device` on Python lists instead of tensor elements.
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

| Instance Type | SDK 2.31 |
|--------------|----------|
| trn2.3xlarge (TP=4, LNC=2) | **Validated** — ctx<=2048 (default quadratic scan), ctx<=8192 (chunked NKI SSD, opt-in via `USE_CHUNKED_NKI_SCAN=1`) |
| trn2.48xlarge (TP=8, LNC=2) | Not tested on SDK 2.31 |
| trn1.32xlarge | Not supported (NxDI 2.29+ requires Trn2 or newer) |

Verified on `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`. Package versions: torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.26.6360.0, NxDI 0.10.18399, NKI 0.5.0, Runtime 2.33.10, Driver 2.29.0.

## Example Checkpoints

* [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)

## Testing Instructions

```bash
# Run standard integration tests (compile + behavioral accuracy + throughput)
# Subsequent runs load compiled artifacts from cache
pytest test/integration/test_model.py -v

# Run all tests including logit validation against CPU BF16 reference
# Requires ~60 GB free CPU RAM to load the 30B HF model
pytest test/integration/test_model.py -v --run-slow

# Or run directly (compiles if needed)
python test/integration/test_model.py

# Enable logit validation when running directly
RUN_LOGIT_VALIDATION=1 python test/integration/test_model.py

# Enable the chunked NKI SSD path (opt-in)
USE_CHUNKED_NKI_SCAN=1 python test/integration/compile_bench_231.py --ctx 128 --seq 2048

# Standalone kernel unit tests (no HF checkpoint required)
python test/unit/test_mamba2_ssd_chunked.py
python test/unit/test_chunked_d64.py
```

**Environment variables:**
- `NEMOTRON_MODEL_PATH` — Path to HF model (default: `/mnt/models/nemotron-30b`)
- `NEMOTRON_COMPILED_PATH` — Path for compiled model (default: `/mnt/models/nemotron_compiled_contrib`)
- `RUN_LOGIT_VALIDATION` — Set to `1` to enable logit validation when running directly

## Known Issues

1. **Maximum context length on trn2.3xlarge (TP=4, LNC=2)** depends on which scan path is enabled:
   - **Default (head-grouped quadratic scan)**: **ctx=2048** validated (seq=4096). Beyond ctx=2048, the O(L^2) SBUF scratchpad exceeds the 24 GB/core HBM.
   - **Chunked NKI SSD** (`USE_CHUNKED_NKI_SCAN=1`): **ctx=8192** validated (seq=16384). ctx=16384 fails compilation with `[F137] neuronx-cc was forcibly killed` — the CE HLO exceeds available host RAM during `neuronx-cc` compilation (TG NEFF compiles fine). Device HBM is not the limit; a larger-host instance or a future compiler improvement could push the ceiling higher.
   - The chunked NKI SSD path works because it uses O(chunk^2) SBUF scratchpad (chunk=128) instead of O(L^2), and stores state as a fixed 4 MB/layer HBM tensor. The kernel uses only "cheap ops" that survive the DGE budget in full-model compilation. See `src/nki_kernels/README.md` for the pattern documentation.
2. **Maximum batch size is 2 on trn2.3xlarge (LNC=2).** BS=4 compiles successfully but exceeds HBM during model load (CE model allocation fails on 24 GB/core). BS=4 would require trn2.48xlarge or LNC=1 (not tested).
3. **The `USE_SSD_SCAN=True` / `USE_PYTORCH_SSD=True` paths remain broken** in the full 52-layer graph: the NKI SSD kernel with 128-expert MoE triggers a runtime error, `scatter/gather (indirect memory copy via vector DGE) out-of-bound access`. This is a compiler-level issue independent of our modeling code. The chunked NKI SSD path (`USE_CHUNKED_NKI_SCAN=1`) is the supported way to get chunk-based SSD scan into the full model.
4. **No on-device sampling tested.** Current validation uses raw logits (`on_device_sampling_config=None`).
5. **Manual depthwise conv1d.** We use an explicit weight loop instead of the auto-inserted NKI Conv1d kernel to avoid past crashes on `seq_len=1`. This may be slower than a native conv1d once the SDK issue is fully resolved.
6. **Base model behavior.** This is a base (non-instruct) model. Greedy decoding produces repetitive output after the first few correct tokens, consistent with the HF reference.
7. **Sparse dispatch prefill fallback.** The prefill (context encoding) path uses a dense per-expert loop because sparse `index_select` on 128 experts at `seq_len=128` creates an HLO graph too large for the 5M instruction limit. A fused NKI MoE kernel could address this.
8. **BS>1 blocked on vLLM-neuron.** When launching vLLM with `--max-num-seqs >1`, NEFFs compile correctly for the larger batch size, but Mamba state buffers (`mamba_states`) are initialized with `batch_size=1` (from `config.neuron_config.batch_size`). The runtime rejects the shape mismatch (e.g., "received 1 8 64 128, expected 4 8 64 128"). Fix requires plumbing `max_num_seqs` through to `neuron_config.batch_size` in `NeuronNemotronModel.init_model()`.

## HuggingFace Model Issues Found

During development, we discovered and documented several issues in the original HuggingFace `modeling_nemotron_h.py`:

1. **`MambaRMSNormGated` ignores `group_size`** — The PyTorch fallback normalizes over the full hidden dimension instead of per-group. The CUDA Triton kernel is correct. This causes incoherent decode output on CPU/non-CUDA backends.
2. **`HybridMambaAttentionDynamicCache` attribute issues** — `self.ssm_states` and `self.conv_states` are Python lists but accessed as tensors (`.device`, `.zero_()`).
3. **Cache key mismatch** — `prepare_inputs_for_generation()` stores cache under `"past_key_values"` but `forward()` expects `"cache_params"`, preventing proper state persistence in HF's `generate()`.

## Source Files

| File | Description |
|------|-------------|
| `src/modeling_nemotron_h.py` | Full model implementation (config, Mamba-2 layer, GQA attention, MoE with sparse dispatch, preshard_hook for TP, model wrapper, state dict conversion, all scan-path flags) |
| `src/__init__.py` | Public exports |
| `src/nki_kernels/nki_mamba2_ssd_chunked.py` | Chunked NKI SSD scan kernel (opt-in via `USE_CHUNKED_NKI_SCAN=1`) |
| `src/nki_kernels/nki_mamba2_ssd_recurrent.py` | Reference per-token recurrent kernel (foundation for the chunked variant) |
| `src/nki_kernels/chunked_ssd_wrapper.py` | PyTorch wrapper that adapts Nemotron Mamba-2 prefill inputs to the chunked NKI kernel |
| `src/nki_kernels/README.md` | Pattern documentation: "cheap-ops-only" DGE-OOB-survivor kernel style |
| `test/integration/test_model.py` | Integration tests (compile, load, generate, logit validation, throughput) |
| `test/integration/compile_bench_231.py` | Single-ctx TTFT/decode/TPOT benchmark |
| `test/integration/ctx_sweep_driver.py` | Context-ceiling sweep runner |
| `test/integration/compare_scans.py` | A/B token comparison between default and chunked scan paths |
| `test/unit/test_mamba2_ssd_chunked.py`, `test/unit/test_chunked_long_seq.py`, `test/unit/test_chunked_d64.py` | Standalone kernel-level correctness tests |

## Maintainer

Jim Burtoft ([@jimburtoft](https://github.com/jimburtoft))

**Last Updated:** 2026-07-12 (SDK 2.31 chunked NKI SSD, max ctx 8192 on trn2.3xlarge)
