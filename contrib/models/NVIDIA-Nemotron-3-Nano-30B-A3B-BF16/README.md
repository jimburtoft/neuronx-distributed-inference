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

The default prefill path implements depthwise convolution manually using explicit weight parameters and a loop over kernel positions. This was originally added as a Granite4-era workaround for a compiler crash (`TEN404`) on SDK 2.28. On SDK 2.31 the compiler no longer crashes on the native `F.conv1d(groups=conv_dim)` -- set `USE_NATIVE_CONV1D=1` to opt into it. The native path is currently ~11% slower on decode, so the manual loop remains the default. See Known Issue #5.

### Gated RMSNorm with Per-Group Normalization

Nemotron's Mamba layers use gated RMSNorm with `norm_before_gate=False` and per-group normalization (`group_size = intermediate_size / n_groups = 4096 / 8 = 512`). The CUDA Triton kernel (`rmsnorm_fn` from `mamba-ssm`) handles this correctly, but the PyTorch fallback in the original HF code has **two bugs** that any naive port will inherit:

1. It ignores `group_size` entirely, normalizing over the full hidden dim.
2. The `norm_before_gate=False` naming is counter-intuitive. It means the gate is applied to `x` BEFORE variance is computed: `y = RMSNorm(x * SiLU(z)) * w`. The HF fallback (and initial versions of this file) reads the name literally and computes `y = RMSNorm(x) * SiLU(z) * w`, which is a different function.

Symptom of either bug: the model produces the correct first token, then degenerates into repetition. With `apply_chat_template`, output becomes coherent English words but semantically garbage.

Our `NemotronRMSNormGated` implements both fixes correctly for all backends. Reference: [state-spaces/mamba layernorm_gated.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py) function `rms_norm_ref`.

### NKI Selective Scan (Optional)

The codebase includes an optional O(L) NKI selective scan kernel (ported from Granite4 contrib) using `nisa.tensor_tensor_scan`. It is off by default (`USE_NKI_SCAN = False`) because the head-grouped quadratic scan and the chunked NKI SSD kernel (see below) are faster for the sequence lengths this model targets.

## Validation Results

**Instance:** trn2.3xlarge (LNC=2, TP=4)
**SDK:** Neuron SDK 2.31, DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`
**Configuration:** batch_size=1, bfloat16

### End-to-end coherent output with chat template

The model produces **coherent, factually correct, reasoning-quality output** when invoked with `apply_chat_template()`. Sample greedy generations at ctx=128 with the model card's recommended invocation (system prompt empty, `enable_thinking=True` default) exactly match what a properly-deployed Nemotron 3 Nano is expected to produce:

| User prompt | Generated response (greedy, ctx=128) |
|-------------|---------------------------------------|
| "What is 2+2?" | `The user asks a simple question: "What is 2+2?" The answer is 4. Provide a concise answer.</think>4<\|im_end\|>` |
| "What is the capital of France?" | `The user asks: "What is the capital of France?" The answer is Paris. Provide concise answer.</think>The capital of France is **Paris**.<\|im_end\|>` |
| "Write a haiku about GPUs" | `We need to respond with a haiku about GPUs. Haiku is 5-7-5 syllable structure. Let's craft: "Silicon whispers / Parallel hearts beat in silicon / Light streams forth". Count syllables...` |

The `<think>...</think>` reasoning trace followed by a concise final answer + `<|im_end|>` is exactly the behavior documented on the model card.

### Raw completion prompts (no chat template)

For raw completion prompts, the model still produces the correct first token but degenerates into repetition, because Nemotron 3 Nano is instruction-tuned + RLHF'd -- it expects chat-templated input. Both Neuron and HF CPU reference behave the same way on raw completion prompts:

| Prompt | Neuron first token | HF CPU first token | Match |
|--------|---------------------|---------------------|-------|
| "The capital of France is" | ` Paris` | ` Paris` | ✓ |
| "Albert Einstein was born in" | ` 1` (then diverges) | ` 1` (then diverges) | ✓ first token only |

### Chunked NKI SSD end-to-end correctness

Bit-exact A/B token comparison between the chunked NKI SSD path and the head-grouped quadratic scan on the same model, greedy decoding on raw completion prompts:

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

1. **CUDA import fallbacks** -- `rmsnorm_fn`, `selective_state_update`, `causal_conv1d` imports fail without CUDA. Change `raise ImportError(...)` at line ~66 to `rmsnorm_fn = None`.
2. **`MambaRMSNormGated` PyTorch fallback** -- two bugs: ignores `group_size` and applies gate AFTER normalization instead of BEFORE. For `norm_before_gate=False` (Nemotron's setting), the correct math is `y = RMSNorm(x * SiLU(z)) * w`. See our `test/integration/test_model.py::_patch_hf_rmsnorm` for the corrected implementation.
3. **`HybridMambaAttentionDynamicCache` attribute issues** -- accesses `.device` on Python lists instead of tensor elements. Replace `self.conv_states.device` with `new_conv_state.device` (and similar for `ssm_states`).
4. **`torch.cuda.stream(torch.cuda.default_stream(...))`** -- Replace with `if True:` for non-CUDA backends.

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
2. **Maximum batch size is 2 on trn2.3xlarge (LNC=2, verified 2026-07-13).** BS=2 compiles and generates correctly (TTFT=139 ms, aggregate decode 127.8 tok/s, coherent output on both stream). BS=4 compiles successfully but exceeds HBM during model load (CE model allocation fails on 24 GB/core). BS=4 would require trn2.48xlarge or LNC=1 (not tested).
3. **The `USE_SSD_SCAN=True` / `USE_PYTORCH_SSD=True` paths remain broken** in the full 52-layer graph: the NKI SSD kernel with 128-expert MoE triggers a runtime error, `scatter/gather (indirect memory copy via vector DGE) out-of-bound access`. This is a compiler-level issue independent of our modeling code. The chunked NKI SSD path (`USE_CHUNKED_NKI_SCAN=1`) is the supported way to get chunk-based SSD scan into the full model.
4. **No on-device sampling tested.** Current validation uses raw logits (`on_device_sampling_config=None`).
5. **Manual depthwise conv1d (default).** The prefill path defaults to an explicit weight-loop implementation of the depthwise conv1d. Set `USE_NATIVE_CONV1D=1` to opt into `F.conv1d(groups=conv_dim)` instead. On SDK 2.31 the native path compiles successfully -- the compiler auto-inserts NKI `Conv1d_depthwise_bf01_oi01_bf01` kernels and produces coherent output. Decode throughput is currently **~11% lower** (94.4 vs 105.9 tok/s at ctx=128, BS=1) with the native path, so the manual loop remains the default. The original TEN404 compiler crash that motivated this workaround on SDK 2.28 is no longer reproducible on SDK 2.31 (verified 2026-07-13).
6. **Model expects chat-templated input.** Nemotron 3 Nano is fully instruction-tuned + RLHF'd (not a base model). Feed prompts through `tokenizer.apply_chat_template(messages, add_generation_prompt=True, ...)` to get coherent output. Raw completion prompts (e.g., `"The capital of France is"`) get the correct first token but degenerate into repetition, which is expected instruct-model behavior on unaligned input.
7. **Sparse dispatch prefill fallback (still required on SDK 2.31).** The prefill (context encoding) path uses a dense per-expert loop because sparse `index_select` on 128 experts at `seq_len=128` still exceeds the compiler instruction limit. Setting `FORCE_SPARSE_MOE_PREFILL=1` compiles the HLO but `neuronx-cc` fails with `[NCC_EBVF030] Instructions generated by compiler 19633667 exceeds the typical limit of 15000000` (verified 2026-07-13 on SDK 2.31). A fused NKI MoE kernel could address this.
8. **BS>1 works when compiled with the correct `batch_size` (verified 2026-07-13).** Compiling with `neuron_config.batch_size=2` produces working BS=2 NEFFs. Both batch items generate coherent output identical to BS=1 (e.g., `"The capital of France is **Paris**."` on the chat-templated prompt). TTFT at BS=2 is 139 ms and aggregate decode is 127.8 tok/s. **The vLLM-neuron shape-mismatch failure** (`"received 1 8 64 128, expected 4 8 64 128"`) is a plumbing bug in the vLLM adapter, not the model: it passes `--max-num-seqs >1` to a NEFF compiled with `batch_size=1`. Fix requires threading `max_num_seqs` through to `neuron_config.batch_size` in `NeuronNemotronModel.init_model()`. Also note the HBM ceiling: BS=4 compiles but exceeds 24 GB/core at load (would require trn2.48xlarge or LNC=1).

## HuggingFace Model Issues Found

During development, we discovered and documented several issues in the original HuggingFace `modeling_nemotron_h.py`:

1. **`MambaRMSNormGated` has two related bugs in its PyTorch/CUDA-fallback path**:
   - **(a) Ignores `group_size`** -- normalizes over the full hidden dimension instead of per-group. The Triton `rmsnorm_fn` from `mamba-ssm` is correct.
   - **(b) Wrong gate order for `norm_before_gate=False`** -- the CUDA `rmsnorm_fn` with `norm_before_gate=False` computes `y = RMSNorm(x * SiLU(z)) * w` (gate applied to x BEFORE variance). The HF fallback (and any naive port that reads the name literally) computes `y = RMSNorm(x) * SiLU(z) * w`, which is mathematically different.
   - **Symptom:** On CPU without the CUDA fast path, the model produces the correct first token, then degenerates into repetition. On instruction-following prompts (with `apply_chat_template`), output becomes coherent English words but semantically garbage (e.g., `"I'm sorry, I'm not sure I'm going to be able to..."`).
   - Both bugs are fixed in our `src/modeling_nemotron_h.py` (`NemotronRMSNormGated` and `_gated_rmsnorm_with_weight`) and in the HF CPU reference patch in `test/integration/test_model.py::_patch_hf_rmsnorm`.
2. **`HybridMambaAttentionDynamicCache` attribute issues** -- `self.ssm_states` and `self.conv_states` are Python lists but accessed as tensors (`.device`, `.zero_()`).
3. **Cache key mismatch** -- `prepare_inputs_for_generation()` stores cache under `"past_key_values"` but `forward()` expects `"cache_params"`, preventing proper state persistence in HF's `generate()`.
4. **`torch.cuda.stream(torch.cuda.default_stream(...))` in `NemotronHBlock.forward`** -- fails without CUDA. Must be wrapped in `if True: ...` or `contextlib.nullcontext()` for CPU/Neuron.
5. **`mamba-ssm` `ImportError` at import time** -- the module fails to load without the mamba-ssm package installed. Change `raise ImportError(...)` to `rmsnorm_fn = None` so the fallback path can be used.

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

**Last Updated:** 2026-07-13 (Known Issues #1 conv1d + #7 sparse prefill + #8 BS>1 retested on SDK 2.31; RMSNorm gate-order bug fix from 2026-07-12 unchanged)
