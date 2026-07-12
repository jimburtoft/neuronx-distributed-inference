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

#### trn2.3xlarge (TP=4, LNC=2, BF16, SDK 2.31, chunked NKI SSD scan)

Measured 2026-07-10 on `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`. Requires `USE_CHUNKED_NKI_SCAN=True` in `modeling_nemotron_h.py` (default is False for safety). Kernel: `contrib/src/nki_kernels/nki_mamba2_ssd_chunked.py`.

| Configuration | TTFT (ms) | Decode (tok/s) | TPOT (ms) | Load (s) |
|--------------|-----------|----------------|-----------|----------|
| BS=1, seq=2048, ctx=128 | **63.2** | 104.34 | 9.58 | 52.7 |
| BS=1, seq=2048, ctx=512 | 172.8 | 102.30 | 9.77 | 64.5 |
| BS=1, seq=2048, ctx=1024 | 325.2 | 103.16 | 9.69 | 83.8 |
| BS=1, seq=4096, ctx=2048 | 633.8 | 99.39 | 10.06 | 101.0 |
| **BS=1, seq=8192, ctx=4096** (new capability) | **1450.1** | 100.99 | 9.90 | 110.9 |
| **BS=1, seq=16384, ctx=8192** (new capability) | **2503.4** | 101.29 | 9.87 | 151.5 |
| ~~ctx=16384~~ | ~~compile fails~~ | -- | -- | -- |

**Ceiling on trn2.3xlarge**: ctx=8192 works; ctx=16384 fails with `[F137] neuronx-cc was forcibly killed` (compiler host RAM OOM, not device HBM). TG NEFF at ctx=16384 compiles fine; only the CE NEFF exceeds 128 GB host RAM. A larger-host instance (or compiler improvements) could push higher.

**Correctness validated** by A/B token comparison vs the head-grouped quadratic scan on the same model: **50/50 tokens identical at ctx=128**, **30/30 tokens identical at ctx=1024 (8 chunks)** under greedy decoding.

**Delta vs head-grouped quadratic** (numbers below):

| ctx | Quadratic TTFT | Chunked TTFT | Delta |
|-----|---------------|--------------|-------|
| 128 | 67.5 ms | **63.2 ms** | 6% faster |
| 512 | 225.1 ms | **172.8 ms** | 23% faster |
| 1024 | 459.8 ms | **325.2 ms** | 29% faster |
| 2048 | 1081.7 ms | **633.8 ms** | 41% faster |
| 4096 | HBM OOM | **1450.1 ms** | new capability |
| 8192 | HBM OOM | **2503.4 ms** | new capability |

Decode throughput unchanged (~100-104 tok/s) in both paths.

#### trn2.3xlarge (TP=4, LNC=2, BF16, SDK 2.31, head-grouped scan G=8)

Measured 2026-07-09 on `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`.

| Configuration | TTFT (ms) | Decode (tok/s) | TPOT (ms) | Load (s) |
|--------------|-----------|----------------|-----------|----------|
| BS=1, seq=2048, ctx=128 | **67.5** | **105.66** | **9.46** | 56.5 |
| BS=1, seq=2048, ctx=256 | 124.6 | 103.38 | 9.67 | 76.7 |
| BS=1, seq=2048, ctx=512 | 225.1 | 102.62 | 9.74 | 81.0 |
| BS=1, seq=2048, ctx=640 | 425.3 | 105.64 | 9.47 | 67.9 |
| BS=1, seq=2048, ctx=768 | 514.7 | 106.66 | 9.38 | 73.5 |
| BS=1, seq=2048, ctx=1024 | 459.8 | 105.61 | 9.47 | 59.7 |
| **BS=1, seq=4096, ctx=2048** | **1081.7** | **102.06** | **9.80** | 87.1 |

All greedy decoding, 50-token generation, single warmup. Correct " Paris" first token at every ctx. Decode throughput is essentially flat at ~103-106 tok/s across the full ctx range.

#### trn2.3xlarge (TP=4, LNC=2, BF16, SDK 2.28, sparse MoE)

Historical numbers preserved from Task 010 for comparison:

| Configuration | TTFT (ms) | Decode (tok/s) | TPOT (ms) |
|--------------|-----------|----------------|-----------|
| BS=1, seq=2048, ctx=128 (sparse MoE) | 211 | 66.6 | 15.0 |
| BS=1, seq=2048, ctx=128 (dense MoE) | 211 | 17.4 | 57.5 |
| BS=1, seq=4096, ctx=256 (sparse MoE) | 436 | 45.8 | -- |
| BS=2, seq=2048, ctx=128 | 263 | 22.0 | -- |
| BS=1, seq=4096, ctx=128 | 210.5 | 16.0 | -- |
| BS=1, seq=8192, ctx=128 | 211.3 | 15.8 | -- |

**Perf gains SDK 2.28 -> SDK 2.31 (BS=1, ctx=128):** decode 66.6 -> **105.7 tok/s** (+58%), TTFT 211 -> **67.5 ms** (3.1x faster).

All measurements on trn2.3xlarge (TP=4, LNC=2, BF16).

#### trn2.48xlarge (TP=8, LNC=2)

| Configuration | TTFT (ms) | Decode (tok/s) | TPOT P50 (ms) | E2E P50 (ms) |
|--------------|-----------|----------------|---------------|--------------|
| **BS=1, ctx=2048, 50 output tokens** | 1,698 | 22.9 | 9.93 | 2,184 |
| **BS=1, ctx=2048, 100 output tokens** | 1,698 | 37.3 | 9.94 | 2,682 |
| **BS=1, ctx=2048, 200 output tokens** | 1,698 | 54.4 | 9.95 | 3,678 |
| **BS=1, ctx=2048, 300 output tokens** | 1,699 | 64.1 | 9.96 | 4,677 |
| **BS=1, ctx=2048, 400 output tokens** | 1,698 | 70.5 | 9.96 | 5,673 |
| **BS=1, ctx=2048, 512 output tokens** | 1,698 | **75.4** | **9.98** | 6,796 |

Measured via `vllm bench serve` with `vllm-neuron` on trn2.48xlarge (TP=8, LNC=2, BF16). Input: 1800-token prefix + 141 random tokens = 1941 total. 20 prompts, 4 warmup. Prefix caching disabled (Mamba recurrent state). TPOT P50-P99 spread < 0.1 ms.

**Maximum context length on trn2 (LNC=2) is 2048.** All trn2 instances have 24 GB per logical core at LNC=2, regardless of instance size. The Mamba-2 quadratic scan at 4224 tokens produces scratchpad + weight requirements of 22.765 GB per core, leaving only ~1.2 GB free. The CTE transpose operation needs 1.031 GB more than available. Tested with both `-O1` and `-O2` compiler optimization on trn2.48xlarge TP=8 — same result. See Known Issues #8.

**Sparse expert dispatch** (default) achieves **3.83x decode speedup** by loading only the 6 active expert weights per MoE layer during decode, instead of all 128. Output is bit-for-bit identical to the dense path.

| Metric | Value (BS=1) |
|--------|-------|
| Model load time | 16.9 s |
| HBM per core (est.) | ~14.7 GB / 24 GB (61%) |

TPOT is extremely stable at BS=1: P50-P99 spread < 0.3 ms.

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

| Instance Type | SDK 2.31 | SDK 2.29 | SDK 2.28 | SDK 2.27 |
|--------------|----------|----------|----------|----------|
| trn2.3xlarge (TP=4, LNC=2) | **Validated (ctx<=2048 quadratic, ctx<=8192 chunked NKI SSD)** | Validated (ctx<=640) | Validated (ctx<=256) | Not tested |
| trn2.48xlarge (TP=8, LNC=2) | Not tested | Not tested | Validated (ctx<=2048) | Not tested |
| trn1.32xlarge | Not supported | Not supported | Not tested | Not tested |

> **NxDI 2.29+ requires Trn2 or newer hardware.** For Trn1 support, pin to SDK 2.28.

**SDK 2.31 notes (verified 2026-07-09 on `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`)**:
- Working configuration (default): Falcon-H1-style TP sharding + head-grouped O(L^2) quadratic scan (`SCAN_HEAD_GROUP=8`).
- New chunked NKI SSD path (opt-in via `USE_CHUNKED_NKI_SCAN=True`) delivers 6-41% faster prefill on matched contexts AND extends the max ctx on trn2.3xlarge from **2048 to 8192 (4x)**. Correctness proven bit-for-bit vs quadratic (50/50 tokens at ctx=128, 30/30 at ctx=1024).
- Decode throughput 102-106 tok/s across ctx=128..2048 (BS=1) -- **~40% faster** than SDK 2.29 (73-74 tok/s).
- TTFT 2.7-4.4x faster than SDK 2.29 at matched context lengths.
- Max context on trn2.3xlarge raised from ctx=640 (SDK 2.29) to **ctx=2048** (SDK 2.31, seq=4096) -- matches trn2.48xlarge SDK 2.28 milestone on the smaller instance.
- Still blocked: NKI SSD kernel with full MoE graph triggers `scatter/gather (indirect memory copy via vector DGE) out-of-bound access` at runtime. Same failure as SDK 2.29. Do NOT set `USE_SSD_SCAN=True` / `USE_PYTORCH_SSD=True` for the full 52-layer model.
- Package versions: torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.26.6360.0, NxDI 0.10.18399, NKI 0.5.0, Runtime 2.33.10, Driver 2.29.0.

**SDK 2.29 note**: The model compiles and loads on SDK 2.29 (NxDI 0.9.17334) and produces correct inference output after the Falcon-H1 TP rewrite (2026-04-11). Use SDK 2.31 for best performance and highest context length.

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

# Quick smoke test (requires pre-compiled model)
python test_smoke.py
```

**Environment variables:**
- `NEMOTRON_MODEL_PATH` — Path to HF model (default: `/mnt/models/nemotron-30b`)
- `NEMOTRON_COMPILED_PATH` — Path for compiled model (default: `/mnt/models/nemotron_compiled_contrib`)
- `RUN_LOGIT_VALIDATION` — Set to `1` to enable logit validation when running directly

## Known Issues

1. **Maximum context length on trn2.3xlarge (TP=4, LNC=2) depends on SDK version and scan path:**
   - **SDK 2.31 + chunked NKI SSD** (`USE_CHUNKED_NKI_SCAN=True`): **ctx=8192** validated (seq=16384). ctx=16384 fails compilation with `[F137] neuronx-cc was forcibly killed` -- compiler host RAM OOM at CE HLO compile step (TG NEFF compiles fine). Device HBM is not the limit.
   - **SDK 2.31 default (head-grouped scan)**: **ctx=2048** validated (seq=4096, decode 102 tok/s). Beyond ctx=2048, the head-grouped O(L^2) scratchpad exceeds the 24 GB/core HBM.
   - **SDK 2.29**: ctx=640 max (head-grouped scan G=8). ctx=768 fails at load with HBM OOM (34.5 GB required vs 24 GB available).
   - **SDK 2.28**: ctx=256 max. ctx=512 compiles but OOM on load -- the CE model's 14.8 GB tensors + 7.5 GB scratchpad exceed 24 GB/core.
   - The SDK 2.31 improvement (2048) over 2.29 (640) comes from a redesigned graph compiler code generation backend (default on Trn2/Trn3 in 2.31).
   - The chunked NKI SSD path (8192) works because it uses O(chunk^2) SBUF scratchpad (chunk=128) instead of O(L^2), and stores state as a fixed 4 MB/layer HBM tensor. Kernel uses only "cheap ops" that survive the DGE budget in full-model compilation.
2. **Maximum batch size is 2 on trn2.3xlarge (LNC=2).** BS=4 compiles successfully but exceeds HBM during model load (CE model allocation fails on 24 GB/core). BS=4 would require trn2.48xlarge or LNC=1 (not tested).
3. **Validated seq_len up to 8192.** seq_len=4096 and seq_len=8192 both compile, load, and generate correctly with stable throughput (~16 tok/s) and TTFT (~211 ms).
4. **No on-device sampling tested.** Current validation uses raw logits (`on_device_sampling_config=None`).
5. **Conv1d workaround.** Manual depthwise convolution avoids TEN404 but may be slower than native conv1d once the SDK issue is fixed.
6. **Base model behavior.** This is a base (non-instruct) model. Greedy decoding produces repetitive output after the first few correct tokens, consistent with the HF reference.
7. **Sparse dispatch prefill fallback.** The prefill (context encoding) path uses a dense per-expert loop because sparse `index_select` on 128 experts at `seq_len=128` creates HLO graph explosion exceeding the 5M instruction limit. A fused NKI MoE kernel could address this.
8. **Maximum context length is 2048 on all trn2 instances (LNC=2).** All trn2 instances have 24 GB per logical core at LNC=2, so this limit applies regardless of instance size. ctx=2048 is validated on trn2.48xlarge TP=8. ctx=4224 compiles successfully but fails at NEFF load time — the Mamba-2 quadratic scan scratchpad consumes 22.765 GB per logical core (weights 7.5 GB + scratchpad 7.0 GB + shared scratchpad 8.25 GB), leaving only ~1.2 GB free. The CTE transpose operation (`transpose.215_sg0002`) requires an additional 1.031 GB that cannot be allocated. Tested both `-O1` and `-O2` compiler optimization — identical failure. The scratchpad requirement scales quadratically with context length due to the Mamba-2 parallel scan. Possible mitigations: chunked CTE (multiple smaller passes), TP=16 (requires `n_groups` divisible by 16; currently `n_groups=8`), or switching to the O(L) NKI selective scan for CTE.
9. **BS>1 blocked on vLLM-neuron.** When launching vLLM with `--max-num-seqs >1`, NEFFs compile correctly for the larger batch size, but Mamba state buffers (`mamba_states`) are initialized with `batch_size=1` (from `config.neuron_config.batch_size`). The runtime rejects the shape mismatch (e.g., "received 1 8 64 128, expected 4 8 64 128"). Fix requires plumbing `max_num_seqs` through to `neuron_config.batch_size` in `NeuronNemotronModel.init_model()`.

## HuggingFace Model Issues Found

During development, we discovered and documented several issues in the original HuggingFace `modeling_nemotron_h.py`:

1. **`MambaRMSNormGated` ignores `group_size`** — The PyTorch fallback normalizes over the full hidden dimension instead of per-group. The CUDA Triton kernel is correct. This causes incoherent decode output on CPU/non-CUDA backends.
2. **`HybridMambaAttentionDynamicCache` attribute issues** — `self.ssm_states` and `self.conv_states` are Python lists but accessed as tensors (`.device`, `.zero_()`).
3. **Cache key mismatch** — `prepare_inputs_for_generation()` stores cache under `"past_key_values"` but `forward()` expects `"cache_params"`, preventing proper state persistence in HF's `generate()`.

## Source Files

| File | Description | Lines |
|------|-------------|-------|
| `src/modeling_nemotron_h.py` | Full model implementation (config, Mamba layer, attention, MoE with sparse dispatch, NKI scan, preshard_hook for TP, model wrapper, state dict conversion) | ~2280 |
| `src/__init__.py` | Public exports | ~27 |
| `test/integration/test_model.py` | Integration tests (compile, load, generate, logit validation, throughput) | ~441 |
| `test_smoke.py` | Quick smoke test for pre-compiled model | ~79 |

## Maintainer

Jim Burtoft ([@jimburtoft](https://github.com/jimburtoft))

**Last Updated:** 2026-07-10 (SDK 2.31 chunked NKI SSD integration by agent nemo -- max ctx 8192 on trn2.3xlarge)
