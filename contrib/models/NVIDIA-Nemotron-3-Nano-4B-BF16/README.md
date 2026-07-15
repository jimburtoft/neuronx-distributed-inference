# NVIDIA-Nemotron-3-Nano-4B-BF16 -- NeuronX port

This directory contains the NeuronX Distributed Inference (NxDI) port of NVIDIA's
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, a **dense** hybrid Mamba2-Transformer model
(no MoE) targeting edge and small-batch inference. The port is a fork of the sibling
[30B-A3B-BF16 port](../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/) with MoE stripped and
a dense MLP layer added.

## Model overview

- **Type**: `NemotronHForCausalLM` (hybrid Mamba2-Transformer, dense variant)
- **Parameters**: 3.97 B (all-active; not a MoE)
- **Layers**: 42 (21 Mamba-2, 17 dense MLP, 4 GQA attention)
- **Pattern**: `M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-` (M=Mamba, `-`=MLP, `*`=attention)
- **Hidden size**: 3136
- **MLP intermediate**: 12544 (relu2 activation, `mlp_bias=false`)
- **Attention**: 40 Q heads, 8 KV heads (5:1 GQA), `head_dim=128`
- **Mamba-2**: 96 heads, `mamba_head_dim=80`, `ssm_state_size=128`, `n_groups=8`, `chunk_size=256`, `conv_kernel=4`
- **Context**: `max_position_embeddings=262144` (declared; actual Neuron ceiling depends on scan path)
- **Trained from**: NVIDIA-Nemotron-Nano-9B-v2 via the Elastic compression framework
- **Model card**: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

## Port summary

The Neuron port carries over verbatim from the 30B-A3B port:
- `NemotronRMSNormGated` with the July 12 gate-order fix (`RMSNorm(x * SiLU(z))` for `norm_before_gate=False`)
- `USE_NATIVE_CONV1D` toggle for the depthwise conv1d
- All four scan paths (head-grouped quadratic default, chunked NKI SSD, O(L) NKI, PyTorch SSD)
- `NeuronNemotronMamba2Layer`, `NeuronNemotronAttention`, and the NxDI wrapper machinery

What changed vs the 30B port:
- Replaced `NeuronNemotronRouter` + `NeuronNemotronMoELayer` with a plain `NeuronNemotronMLP` class:
  ColumnParallelLinear(`up_proj`) -> relu2 -> RowParallelLinear(`down_proj`), no gate, no bias.
- `PATTERN_MAP` grammar changed: `{'M': 'mamba', '-': 'mlp', '*': 'attention'}` (30B used `'E'` for MoE).
- `NemotronHInferenceConfig.get_neuron_config_cls()` returns `NeuronConfig` instead of `MoENeuronConfig`.
- Dropped MoE fields (`n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size`, ...);
  added `intermediate_size` to required attributes.
- Config reader accepts either `norm_eps` (30B) or `rms_norm_eps` / `layer_norm_epsilon` (4B).
- State-dict conversion simplified: no expert stacking, no shared-expert transpose.
- Chunked NKI SSD kernel wrapper hardcodes `chunk_size=128`; the modeling code overrides the
  4B config's `chunk_size=256` to `128` when `USE_CHUNKED_NKI_SCAN=1`.

## Instance and SDK

All measurements below on:
- **Instance**: trn2.3xlarge (1× Neuron device, 4 logical cores at LNC=2, 24 GB HBM/core)
- **SDK**: Neuron SDK 2.31
- **DLAMI**: `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708` (`ami-0b658f2749710f886` in ap-southeast-4)
- **TP**: 4 across the 4 logical cores
- **Weights**: bf16

## Benchmark results

BS=1 sweep (default head-grouped quadratic scan G=8, unless noted):

| Config | Scan path | Compile (s) | Load (s) | TTFT (ms) | Decode (tok/s) | TPOT (ms) | Coherent? |
|---|---|---|---|---|---|---|---|
| ctx=128, seq=2048 | quadratic | 212 | 10.8 | **39.1** | **144.7** | 6.91 | yes |
| ctx=256, seq=2048 | quadratic | 375 | 12.6 | 123.3 | 144.1 | 6.94 | yes |
| ctx=512, seq=2048 | quadratic | 459 | 14.7 | 283.0 | 146.5 | 6.82 | yes |
| ctx=1024, seq=2048 | quadratic | -- HBM OOM: 27.63 GB > 24 GB/core | | | | | |
| ctx=1024, seq=2048 | chunked NKI SSD | 1938 | 14.3 | 392.9 | 146.1 | 6.84 | yes |
| ctx=2048, seq=4096 | chunked NKI SSD | 1010 | 20.9 | 782.5 | 141.8 | 7.05 | yes |

**Coherence check (chat-templated, greedy):**

Input: `[{"role": "user", "content": "What is the capital of France?"}]` -> apply_chat_template

Output at every ctx above:
```
The user asks: "What is the capital of France?" The answer is Paris.
So respond: "The capital of France is Paris." Probably just answer.
</think>
The capital of France is Paris.<|im_end|>
```

The reasoning-trace + concise-final-answer + `<|im_end|>` termination is exactly what the
Nemotron 3 Nano model card advertises. Ported model behaves correctly.

Batching sweep (ctx=128, seq=2048, quadratic scan):

| BS | Compile (s) | Load (s) | TTFT (ms) | Decode per-stream (tok/s) | Decode aggregate (tok/s) | TPOT (ms) |
|---|---|---|---|---|---|---|
| 1 | 212 | 10.8 | 39.1 | 144.7 | **144.7** | 6.91 |
| 4 | 2012 | 24.4 | 384.5 | 106.8 | **427.0** | 9.37 |
| 8 | -- HBM OOM: 27.27 GB > 24 GB/core | | | | | |

Per-stream decode drops from 144.7 (BS=1) to 106.8 (BS=4) as batches share Mamba SSM state
buffers; aggregate throughput scales to ~427 tok/s at BS=4.

### vs 30B-A3B on the same instance

At BS=1, ctx=128, SDK 2.31, trn2.3xlarge TP=4:

| Metric | 4B (dense) | 30B-A3B (MoE, ~3B active) | Ratio |
|---|---|---|---|
| Compile time | 212 s | 1131 s | 5.3× faster (4B) |
| Load time | 10.8 s | 323 s | 30× faster (4B) |
| TTFT | 39.1 ms | 67.5 ms | 1.73× faster (4B) |
| Decode | 144.7 tok/s | 105.7 tok/s | 1.37× faster (4B) |
| TPOT | 6.91 ms | 9.46 ms | 1.37× faster (4B) |
| First token | " Paris" ✓ | " Paris" ✓ | tie |
| Chat coherence | yes ✓ | yes ✓ | tie |

The 4B ports much more cleanly than the 30B: no MoE routing, no sparse-dispatch HBM budget
issues, no expert-loop instruction-limit issues. Compile and load are dramatically faster.
At BS=1 the 4B is ~1.4× faster on decode and ~1.7× faster on TTFT despite being the same
per-token active-parameter count (~3 B).

## NKI kernel status (Task 020, 2026-07-15)

Audit of which nki-lib kernels fire for the 4B in the default compile, and
which additional ones could be wired in:

| Component | NKI kernel used? | Notes |
|---|---|---|
| Attention Softmax (CE) | **YES** (auto) | `NativeToCustomSoftmax` pass replaces 4 softmax ops per CE compile (one per attention layer). |
| Attention Q/K/V/O matmul | No | Generic matmul. NxDI's `attention_cte`/`qkv_cte`/`output_projection_cte` NKI kernels are not auto-selected -- suspect shape mismatch, not audited further. |
| Input RMSNorm | No | Runs as generic HLO ops. NKI `rmsnorm` kernel not wired for this pre-norm. |
| Gated RMSNorm (Mamba layer) | No | Our fixed implementation in Python -- compiled to generic HLO. |
| Depthwise Conv1d (Mamba prefill) | Off by default | With `USE_NATIVE_CONV1D=1`, SDK 2.31 auto-inserts 15 `NativeNkiKernel-*_Conv1d_depthwise_bf01_oi01_bf01` invocations across the 21 Mamba layers x CE+TG buckets. But this path is **1.5% slower on decode** than the manual weight-loop path, so it stays off by default. Manual loop remains the default. |
| Mamba SSM step (decode) | No | Head-grouped quadratic scan compiled from our PyTorch code. |
| Mamba SSD scan (prefill) | Off by default | `USE_CHUNKED_NKI_SCAN=1` invokes our own chunked NKI SSD kernel (`src/nki_kernels/nki_mamba2_ssd_chunked.py`). The upstream nkilib `scan/ssd` kernel is not usable (`n_groups=1` constraint; 4B uses `n_groups=8`). |
| Dense MLP up_proj / relu2 / down_proj | Blocked | See below. |

### Dense MLP kernel: BLOCKED by hidden_size=3136 not being a multiple of 128

Attempted to wire the 4B's dense MLP through `non_gated_mlp_nkilib`
(the utility Llama uses for its `mlp_kernel_enabled` path). The kernel supports
`skip_gate_proj=True` (matches 4B: no gate) and `NormType.NO_NORM`, which
would allow keeping the external RMSNorm as-is and still fusing
`up_proj + activation + down_proj + all-reduce` into one HBM-round-trip kernel.

**Blocker**: `nkilib.core.mlp.mlp_parameters` asserts `H % 128 == 0`. The 4B's
`hidden_size=3136` is not a multiple of 128 (3136/128 = 24.5). The 128-alignment
is fundamental to how the kernel tiles into SBUF partitions -- confirmed by
`H // 128` divisions in `mlp_torch.py`, `mlp_tkg_down_projection.py`,
`mlp_tkg_layernorm.py`, `mlp_tkg_rmsnorm.py`, and `down_projection_mx_shard_H.py`.
It's not a shallow assert that can just be relaxed.

**Second blocker**: the `ActFnType` enum lacks `SquaredReLU`. The 4B's
`mlp_hidden_act="relu2"` is not one of the currently-supported activations
(SiLU/GELU/GELU_Tanh_Approx/Swish/ReLU). Task 015 found upstream/main added
SquaredReLU to `moe_cte` but not to the generic MLP path.

### Options to unblock (not attempted)

1. **Pad hidden_size 3136 → 3200 (2% overhead)**. Requires modifying up_proj,
   down_proj, RMSNorm gamma, embeddings, and lm_head all consistently; and
   adding SquaredReLU to the nkilib. Non-trivial but tractable.
2. **Write a custom H=3136 MLP kernel**. Substantial NKI work (~1-2 weeks).
3. **Accept the current baseline**. 4B is already fast (145 tok/s BS=1,
   6.87 ms TPOT). Expected win from fused MLP is 10-20% decode at best.

Recommend Option 3 for now; revisit if a customer deployment has a concrete
target that requires the extra perf.

### Coincidental finding: SDK 2.31 auto-inserts NKI Conv1d without ANY code change

If we simply removed the manual conv1d loop and used `F.conv1d(groups=conv_dim)`
directly, the compiler would auto-insert the NKI Conv1d kernel with no other
opt-in required. On SDK 2.31 this is fully supported (the TEN404 crash that
motivated the manual loop for SDK 2.28 is fixed). The reason we keep the manual
loop is purely a 1.5% decode perf preference, not correctness.

## GPU comparison (L40S, 1× g6e.4xlarge, vLLM 0.25.1, BF16, TP=1)

Real GPU baseline captured 2026-07-14 in us-east-2 (Task 019). Note: p5.4xlarge
(H100) spot capacity was globally exhausted at the time of the run; we used
g6e.4xlarge (L40S, Ada Lovelace, 48 GB HBM, 864 GB/s memory bandwidth) as the
best available fallback. **L40S is not H100.** L40S has ~26% of H100's HBM
bandwidth (864 GB/s vs 3.35 TB/s), so these numbers are a conservative GPU
comparison — a real H100 would be meaningfully faster on decode.

**TTFT (from a `max_tokens=1` sweep, 10 samples each):**

| Input tokens | L40S BS=1 | L40S BS=4 | L40S BS=8 |
|---|---|---|---|
| 128 | **34.0 ms** | 65.4 ms | 73.7 ms |
| 2048 | 79.4 ms | 317.7 ms | 641.9 ms |

**Decode throughput (per-stream and aggregate, computed as `output_tokens / (e2e - ttft)`):**

| Workload | Input | Output | BS | TTFT (ms) | Decode per-stream (tok/s) | Decode aggregate (tok/s) | TPOT (ms) | E2E (ms) |
|---|---|---|---|---|---|---|---|---|
| short-short | 128 | 128 | 1 | 34.0 | 95.0 | 95.0 | 10.53 | 1382 |
| short-short | 128 | 128 | 4 | 65.4 | 83.7 | **334.9** | 11.94 | 1594 |
| short-short | 128 | 128 | 8 | 73.7 | 77.4 | **619.3** | 12.92 | 1727 |
| short-long | 128 | 512 | 1 | 34.0 | 94.4 | 94.4 | 10.59 | 5456 |
| short-long | 128 | 512 | 4 | 65.4 | 83.8 | 335.3 | 11.93 | 6174 |
| short-long | 128 | 512 | 8 | 73.7 | 77.0 | **616.3** | 12.98 | 6720 |
| long-short | 2048 | 128 | 1 | 79.4 | 94.0 | 94.0 | 10.63 | 1441 |
| long-short | 2048 | 128 | 4 | 317.7 | 83.2 | 332.8 | 12.02 | 1856 |
| long-short | 2048 | 128 | 8 | 641.9 | 75.8 | **606.3** | 13.19 | 2331 |
| long-long | 2048 | 512 | 1 | 79.4 | 93.5 | 93.5 | 10.69 | 5553 |
| long-long | 2048 | 512 | 4 | 317.7 | 82.7 | 330.6 | 12.10 | 6512 |
| long-long | 2048 | 512 | 8 | 641.9 | 74.9 | **599.3** | 13.35 | 7477 |

### Head-to-head: 4B Neuron vs 4B L40S (BS=1, ctx=128)

| Metric | Neuron trn2.3xlarge (TP=4, LNC=2) | L40S g6e.4xlarge (TP=1) | Winner |
|---|---|---|---|
| TTFT (in=128) | 39.1 ms | **34.0 ms** | L40S (1.15×) |
| TTFT (in=2048) | 782 ms | **79.4 ms** | L40S (9.9×) |
| Decode tok/s | **144.7** | 95.0 | Neuron (1.52×) |
| TPOT (BS=1) | **6.91 ms** | 10.53 ms | Neuron (1.52×) |
| Aggregate at BS=8 | HBM OOM | **~600 tok/s** | L40S |
| Load time | 10.8 s (post-compile) | 1.4 s | L40S |

**Key takeaways:**
- **BS=1 decode: Neuron wins by 1.52×.** Memory-bandwidth-bound; four LNC=2 logical cores at ~1.6 TB/s effective bandwidth aggregate to more usable decode throughput than a single L40S at 864 GB/s.
- **TTFT: L40S wins decisively**, especially at longer contexts (9.9× faster at ctx=2048). vLLM's chunked prefill + FlashAttention 2 + fused CUDA kernels are much more efficient than our Mamba-2 prefill scan on Neuron.
- **Batching: L40S wins** at BS=4 and BS=8. Neuron hits HBM OOM at BS=8, L40S handles it easily on 48 GB HBM.
- **On H100 (not yet measured)**: expected decode is ~2-3× faster than L40S (extrapolating from the 30B's 269 tok/s H100 number in Task 007, since H100 has ~3.9× L40S's HBM bandwidth). A real H100 measurement would likely flip decode from Neuron-favored (1.52×) to H100-favored (~1.5-2×).

### Cost per million output tokens (BS=1)

| Instance | $/hr | Decode tok/s (BS=1) | $/M output tokens |
|---|---|---|---|
| **trn2.3xlarge** (capacity block) | $2.24 | 144.7 | **$4.30** |
| **g6e.4xlarge** (L40S spot, us-east-2b) | $1.03 | 95.0 | **$3.01** |
| **g6e.4xlarge** (L40S on-demand) | $3.02 | 95.0 | $8.83 |
| **p5.4xlarge** (H100 spot, when available) | $2.53 | ~285 (est) | **~$2.46 (est)** |
| **p5.48xlarge** (H100 spot, 1 of 8 GPUs) | $15.90 (whole box) | 269 (measured 30B) | $16.42 |

L40S spot is currently the cheapest per-token option ($3.01/M tokens) for the 4B
if you can get spot capacity. Neuron capacity block is the second cheapest ($4.30/M).
H100 spot would be the fastest and probably the cheapest if available, but p5 spot
was globally unavailable when we ran this benchmark.

## Known Issues

1. **HBM budget is tight even at 4B.** Compared to the 30B, the 4B has a much wider dense
   MLP (`intermediate_size=12544` vs 30B's per-expert 1856) and larger Mamba intermediate
   (`96*80=7680` vs 30B's `64*64=4096`). Consequences on trn2.3xlarge (24 GB/core at LNC=2):
   - **Default head-grouped quadratic scan**: max ctx **512** at BS=1. ctx=1024 fails with
     `[NCC_EOOM002] Maximum peak HBM usage of 27.63GB exceeds HBM limit of 24.00GB`.
   - **Chunked NKI SSD scan** (`USE_CHUNKED_NKI_SCAN=1`): ctx **2048** verified at BS=1.
     Higher ctx values likely work but not yet tested.
   - **BS=4** works at ctx=128 (default scan). **BS=8** fails with the same HBM OOM
     (27.27 GB > 24 GB/core). BS=8 would require trn2.48xlarge or LNC=1.
2. **Chunked NKI SSD kernel requires `chunk_size=128`.** The kernel wrapper at
   `src/nki_kernels/chunked_ssd_wrapper.py` hardcodes `chunk_size=128` (P_MAX / SBUF partition
   width). The 4B config declares `chunk_size=256`. Our modeling code overrides to 128 when
   `USE_CHUNKED_NKI_SCAN=1`. This is safe (chunk_size is a scan implementation detail, not a
   model parameter). A future kernel update could support `chunk_size=256` natively.
3. **`num_key_value_heads=8` is not divisible by TP=4.** NxDI logs a warning
   (`TP degree (4) and KV heads (8) are not divisible. Overriding attention sharding
   strategy to GQA.CONVERT_TO_MHA!`) but handles this automatically -- each rank ends up
   with 2 KV heads. **Not an error**, functionally correct, and included here as a note
   only.
4. **`mamba_head_dim=80` on chunked NKI SSD not yet audited.** The kernel wrapper's assert
   is `head_dim <= 128`, which passes. Runtime correctness at head_dim=80 confirmed
   end-to-end (coherent output at ctx=1024 and ctx=2048), but bit-exact A/B comparison vs
   the quadratic path has not been done for 4B (only for 30B at head_dim=64).
5. **`intermediate_size` is a scalar in the 4B config.** The HF `NemotronHMLP` supports
   per-layer intermediate sizes via `config.intermediate_size` as a list (e.g., for
   compressed models where different MLPs have different widths). The 4B config uses a
   single scalar, so we treat it as a scalar. If future NVIDIA models use the per-layer
   list form, this port needs updates.
6. **`ssm_state_size==128` required for chunked NKI SSD.** 4B satisfies this. Wouldn't
   apply on models with different state sizes.
7. **The 4B is instruction-tuned.** Feed prompts through `tokenizer.apply_chat_template()`
   for coherent output. Raw completion prompts (e.g., `"The capital of France is"`) get
   the correct first token but degenerate into repetition -- expected instruct-model
   behavior on unaligned input, matches the 30B behavior.

## File layout

```
src/
  __init__.py            -- exports (NemotronHInferenceConfig, NeuronNemotronForCausalLM, ...)
  modeling_nemotron_h.py -- 3085-line modeling module (fork of 30B port)
  nki_kernels/           -- symbolic copies of the 30B port's kernels
    chunked_ssd_wrapper.py
    nki_mamba2_ssd_chunked.py
    nki_mamba2_ssd_recurrent.py
    README.md
test/
  integration/
    __init__.py
    compile_bench.py     -- compile+bench driver, mirrors 30B's compile_bench_231.py
```

## How to use

Environment: SDK 2.31 DLAMI, activate `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate`.

Download the model:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --exclude "*.bin" "*.pth"
```

Compile and benchmark at ctx=128, BS=1:
```bash
cd /path/to/nxdi-nemotron/contrib/models/NVIDIA-Nemotron-3-Nano-4B-BF16
MODEL_PATH=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/*/)
NEMOTRON_MODEL_PATH=$MODEL_PATH TP_DEGREE=4 BATCH_SIZE=1 \
    python test/integration/compile_bench.py --ctx 128 --seq 2048 \
        --out-dir /tmp/artifacts --tag nemotron4b_ctx128_bs1
```

For ctx>=1024, enable the chunked NKI SSD scan:
```bash
USE_CHUNKED_NKI_SCAN=1 NEMOTRON_MODEL_PATH=$MODEL_PATH TP_DEGREE=4 BATCH_SIZE=1 \
    python test/integration/compile_bench.py --ctx 2048 --seq 4096 \
        --out-dir /tmp/artifacts --tag nemotron4b_ctx2048_chunked
```

For BS>1 (up to BS=4 verified):
```bash
NEMOTRON_MODEL_PATH=$MODEL_PATH TP_DEGREE=4 BATCH_SIZE=4 \
    python test/integration/compile_bench.py --ctx 128 --seq 2048 \
        --out-dir /tmp/artifacts --tag nemotron4b_ctx128_bs4
```

## Not tested / future work

- **LNC=1 vs LNC=2 shootout.** Steering doc recommends benchmarking both for models under
  1 B parameters; 4B is above that threshold but worth trying since it fits comfortably at
  LNC=2.
- **trn2.48xlarge (TP=8) BS>=4 at higher ctx.** Would need more HBM / more logical cores
  to escape the OOM ceiling.
- **H100 GPU baseline** on p5.4xlarge for a proper head-to-head. Task 019 attempted this
  2026-07-14 but p5 spot capacity was globally exhausted -- fell back to L40S. Retry when
  p5 spot capacity is available. Expected H100 decode ~2-3× faster than L40S (extrapolating
  from the 30B Task 007 numbers).
- **Bit-exact quadratic vs chunked NKI SSD comparison** at head_dim=80. Coherence-check-
  by-eye passes; formal token match rate not yet measured.
- **Higher ctx (4096, 8192, 16384)** with chunked NKI SSD. Kernel supports arbitrary ctx
  (state is 4 MB/layer fixed), so likely works; not yet compiled.
- **Fused MLP kernel via hidden_size padding**. `hidden_size=3136 → 3200` (2% padding overhead)
  would unblock `non_gated_mlp_nkilib` for the dense MLP. Also needs SquaredReLU added to
  nkilib. Estimated ~10-20% decode win but non-trivial to wire up. See "NKI kernel status"
  section above.

**Last Updated:** 2026-07-15 (Task 020: NKI kernel audit; USE_NATIVE_CONV1D retested; fused MLP kernel blocked by H%128 constraint)
