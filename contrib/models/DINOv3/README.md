# DINOv3 on AWS Neuron

Compile and run [Meta DINOv3](https://github.com/facebookresearch/dinov3) self-supervised vision foundation models on AWS Neuron (Trainium2 and Inferentia2).

## Model

| Property | Value |
|----------|-------|
| **Models** | DINOv3 ViT-S/B/L/H+ (21M-840M), ConvNeXt-T/B (28M-88M), ViT-7B (6.7B) |
| **Architecture** | Encoder-only vision transformer / ConvNeXt backbone |
| **Parameters** | 21M to 6.7B (FP32 by default) |
| **Input** | 224x224 RGB images |
| **Output** | Dense feature embeddings (CLS token for ViT, pooled features for ConvNeXt) |
| **Source** | https://github.com/facebookresearch/dinov3 |
| **License** | DINOv3 License (not Apache/MIT -- check distribution terms) |

## Compilation Approaches

DINOv3 models are compiled using two different approaches depending on model size:

| Model | Params | Approach | Instance |
|-------|--------|----------|----------|
| ViT-S/16 | 21.6M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-B/16 | 85.7M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-L/16 | 303.2M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-H+/16 | 840.6M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ViT-7B/16 | 6,716M | `neuronx-distributed` ModelBuilder TP>=2 | inf2.24xlarge (TP>=2) / trn2.3xlarge (TP=4) |
| ConvNeXt-T | 27.8M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |
| ConvNeXt-B | 87.6M | `torch_neuronx.trace()` | inf2.xlarge / trn2.3xlarge |

**Key insight**: ViT-7B is the first encoder-only vision model to use tensor parallelism on Neuron. The BF16 weights do not fit in a single NeuronCore's HBM, so TP via `neuronx-distributed` is required. TP=2 is the minimum on Inferentia2; TP=4 is used on trn2.3xlarge.

### Maximum model size on Inferentia2

**All 7 DINOv3 variants -- including the 6.7B ViT-7B -- run on Inferentia2, and all of them fit on a single inf2.xlarge** (1 device, 2 cores). Measured on inf2.xlarge and inf2.24xlarge, SDK 2.31.

Inferentia2 HBM: **32 GB per device, 16 GB per NeuronCore** (2 cores per device).

| Model | Params | Min config on inf2 | Peak core HBM | Status |
|-------|-------:|--------------------|--------------:|--------|
| ViT-S/16 | 21.6M | 1 core | < 0.1 GB | Fits easily |
| ViT-B/16 | 85.7M | 1 core | ~0.2 GB | Fits easily |
| ViT-L/16 | 303.2M | 1 core | ~0.5 GB | Fits easily |
| ViT-H+/16 | 840.6M | 1 core | ~1.4 GB | Fits easily |
| **ViT-7B/16** | **6,716M** | **TP=2 (1 device)** | **14.72 GB of 16 GB** | **Fits, TP>=2 required** |
| ViT-7B/16 | 6,716M | TP=1 | **15.95 GB -> OOM** | Does not fit |

Because TP=2 fits within a single Inferentia2 device, **inf2.xlarge is sufficient for every DINOv3 model** -- verified at 11.4 img/s for ViT-7B on inf2.xlarge.


**The binding constraint is per-core HBM, and ViT-7B at TP=2 is right at the edge**: 14.72 GB of the 16 GB core (92%), of which 14.51 GB is tensors. At TP=1 the runtime reaches 15.95 GB and then fails to allocate the next 64 MB block (`TDRV:dmem_alloc_internal Failed to allocate DEVICE memory`).

**Practical ceiling**: ViT-7B at TP=2 consumes 3,360M params/rank. Since a BF16 param costs 2 bytes, one 16 GB core holds roughly **7.2B params/rank** before activations. This means:
- **~7B is the practical max at TP=2** (1 Inferentia2 device)
- **~14B at TP=4**, **~28B at TP=8** -- if a larger DINOv3-family backbone existed
- **TP must be a power of 2.** TP=12 fails with `4096 is not divisible by 12` (embed_dim 4096).

ViT-7B is the largest DINOv3 variant Meta publishes, so **inf2 covers the entire DINOv3 model family**. Headroom exists for larger models at higher TP.

## Results

### Accuracy

Measured on **inf2.24xlarge, SDK 2.31**, FP32 weights with `--auto-cast=matmult --auto-cast-type=bf16`, BS=1, vs CPU FP32 reference.

| Model | Cosine Similarity | Max Abs Diff |
|-------|------------------:|-------------:|
| ViT-S/16 | 1.000000 | 3.6e-06 |
| ViT-B/16 | 1.000000 | < 1e-05 |
| ViT-L/16 | 1.000000 | < 1e-05 |
| ViT-H+/16 | 1.000000 | < 1e-05 |
| ViT-7B/16 | Deterministic (random weights) | -- |
| ConvNeXt-T | 0.999988 | < 0.001 |
| ConvNeXt-B | 0.999989 | < 0.001 |

### Benchmark: inf2.xlarge -- maximum throughput (2 NeuronCores, SDK 2.31)

inf2.xlarge is the smallest and cheapest Inferentia2 instance: **1 Inferentia2 device, 2 NeuronCores, 4 vCPU, 16 GB host RAM**. Each config below is the best of a (workers x batch size) sweep, so these are **maximum achievable** throughput, not single-point samples.

| Model | Params | **Max img/s** | Workers | Batch | p50 (ms) | p99 (ms) |
|-------|-------:|--------------:|--------:|------:|---------:|---------:|
| ViT-S/16 | 21.6M | **832.4** | 2 | 1 | 2.35 | 4.38 |
| ConvNeXt-T | 27.8M | **501.9** | 2 | 1 | 3.97 | 4.27 |
| ViT-B/16 | 85.7M | **421.4** | 2 | 1 | 4.72 | 5.14 |
| ConvNeXt-B | 87.6M | **275.7** | 2 | 16 | 115.33 | 124.26 |
| ViT-L/16 | 303.2M | **130.8** | 2 | 4 | 56.99 | 79.26 |
| ViT-H+/16 | 840.6M | **69.1** | 2 | 4 | 115.73 | 116.36 |
| ViT-7B/16 | 6,716M | **11.4** | 1 (TP=2) | 1 | 83.19 | 118.52 |

**The entire DINOv3 family -- including the 6.7B ViT-7B -- runs on a single inf2.xlarge.** ViT-7B uses both cores as one TP=2 replica.

#### Two rules for picking batch size

Optimal batch size depends on whether the model is dispatch-bound or bandwidth-bound:

- **Small models (ViT-S/B, ConvNeXt-T) peak at BS=1 or BS=4.** Per-call overhead is already amortized; larger batches only add latency. ViT-B: 421.4 img/s at BS=1 vs 294.9 at BS=16.
- **Large models (ViT-L, ViT-H+, ConvNeXt-B) need batching.** Batching amortizes weight loading over more images. **ViT-H+ gains 6.8x: 5.1 img/s at BS=1 -> 34.6 at BS=4** (single worker).

#### Scaling to larger inf2 instances

inf2.xlarge delivers **essentially the same per-core throughput as inf2.24xlarge**, so a
larger instance buys you cores, not efficiency. Per-core, best-batch on each:

| Model | inf2.xlarge (2 cores) | inf2.24xlarge (12 cores) | Ratio |
|-------|----------------------:|-------------------------:|------:|
| ViT-S/16 | 416.2 | 429.3 | 0.97x |
| ViT-B/16 | 210.7 | 213.6 | 0.99x |
| ConvNeXt-T | 250.9 | 266.0 | 0.94x |
| ConvNeXt-B | 137.8 | 143.0 | 0.96x |
| ViT-L/16 | 65.4 | 78.4 | 0.83x |

4 vCPU is sufficient to feed 2 NeuronCores. **Estimate larger instances as
(per-core rate) x (core count)** and choose instance size on cost and required
aggregate throughput. For reference, the measured 12-core inf2.24xlarge peaks are
ViT-S 5,151.7 / ViT-B 2,562.8 / ConvNeXt-T 3,192.2 / ConvNeXt-B 1,716.3 /
ViT-L 940.4 / ViT-H+ 413.7 img/s.

#### ViT-7B tensor parallelism on inf2

| TP | Params/rank | Compile | img/s | p50 (ms) | Peak core HBM |
|---:|------------:|--------:|------:|---------:|--------------:|
| 1 | 6,716M | -- | **OOM** | -- | 15.95 GB -> fail |
| 2 | 3,360M | 8.5s | **12.5** | 76.9 | 14.72 GB (92%) |
| 4 | 1,682M | 7.1s | 11.9 | 79.7 | -- |
| 8 | 843M | 6.5s | **15.9** | 70.5 | -- |

TP=2 and TP=4 are within noise of each other; TP=8 is ~27% faster than TP=2. Compile time is dominated by the TP sharding path and is fast (6-9s) because each rank compiles a smaller graph.

### Tested and rejected: NKI exact-erf GELU kernel

A custom NKI kernel replacing `nn.GELU` with a single Scalar/activation-engine op was
built, validated, and **rejected on measurement**. Recording it here so it is not
re-attempted.

The rationale was sound on paper: `neuronx-cc` lowers exact erf-GELU as **1664 matmuls +
256 reciprocals + 2816 vector ops** where the kernel emits ~26 activation-table ops, and
the DINOv3 forward is **Vector-engine bound (86% active, Tensor only 57%)** with GELU at
**34% of forward Vector+Scalar time**. The kernel is also numerically exact
(**cos_sim 1.000000** on ViT, 0.99999 on ConvNeXt).

It is nonetheless slower almost everywhere. inf2.xlarge, 1 core, best-of-N per side:

| Model | BS=1 | BS=4 | BS=8 |
|-------|-----:|-----:|-----:|
| ViT-S/16 | -11.8% | -7.8% | -5.8% |
| ViT-B/16 | -10.7% | -8.1% | -7.9% |
| ViT-L/16 | +2.8% | -- | -5.5% |
| ConvNeXt-T | -12.7% | -8.4% | -9.7% |
| ConvNeXt-B | -14.5% | -7.3% | -10.8% |

**13 of 14 configurations regress.** The one positive (ViT-L BS=1, +2.8%) is within
run-to-run noise.

**Measurement caveat worth internalizing:** an earlier version of this table reported
**+32%** for ViT-L at BS=1. That was wrong -- the *baseline* had not converged. Measured
back-to-back in one process, the ViT-L BS=1 baseline climbs **52.0 -> 62.4 -> 68.4 img/s**
across three repeats, while the NKI path is flat at 70.2 from the first. Comparing a
warm kernel against a cold baseline manufactured the entire gain. **Always repeat the
baseline until it converges before claiming a kernel win.**

Root cause of the regression: the compiler already fuses GELU into the surrounding MLP
GEMMs, so a standalone kernel adds an HBM round-trip per call (load -> activate -> store)
that outweighs the cheaper transcendental. This matches a prior independent finding that
a fused-MLP NKI kernel also regressed ~23%, and the general rule that kernel *boundary
count* dominates kernel *content* on this path.

The kernel remains in `src/nki_gelu_trace.py` as a **reference implementation** of the NKI
`nki_jit` calling convention for `torch_neuronx.trace()` -- it is not wired into
`trace_dinov3()` and should not be enabled without re-measuring.

### Alternative: PyTorch Native -- faster for ViT, SLOWER for ConvNeXt

`torch.compile(backend="neuron")` (PyTorch Native, Beta 6 / SDK 2.32) was measured against
`torch_neuronx.trace()` on inf2. It is numerically sound (cos_sim 0.999996 vs CPU FP32) and
**the verdict splits by architecture** -- do not assume the ViT result generalizes.

Per-instance, **inf2.24xlarge (12 NeuronCores)**, each side at its own best batch size:

| Model | trace (FP32 + matmult) | PyTorch Native (BF16) | Ratio | Use |
|-------|----------------------:|----------------------:|------:|-----|
| ViT-S/16 | 5,151.7 | **8,118.1** | 1.58x | **Native** |
| ViT-B/16 | 2,562.8 | **4,176.3** | 1.63x | **Native** |
| ViT-L/16 | 940.4 | **1,542.3** | 1.64x | **Native** |
| ConvNeXt-T | **3,192.2** | 2,550.9 | 0.80x | **trace** |
| ConvNeXt-B | **1,716.3** | 1,320.1 | 0.77x | **trace** |

ViT-H+ (840M) was measured separately on inf2.8xlarge (2 cores), since it uses **SwiGLU**
rather than `nn.GELU` and so exercises a different FFN path:

| ViT-H+/16 config | trace | Native | Ratio |
|------------------|------:|-------:|------:|
| 1 core, BS=1 | 5.1 | **47.1** | **9.24x** |
| 1 core, BS=4 | 34.6 | **83.4** | 2.41x |
| 2 cores, best | 69.1 | **143.9** | **2.08x** |

**ViT-H+ is the strongest Native win** -- 9.2x at batch 1. The traced path lowers SwiGLU
poorly at BS=1 (5.1 img/s) and needs batching to recover; Native does not. Native also
scales well on ViT-H+ (86-92% efficiency across 2 cores).

**ConvNeXt is 1.25-1.30x FASTER on the traced path.** Batching does not rescue it on Native
(ConvNeXt-T: 2,335.7 / 2,529.4 / 2,550.9 img/s at BS=1/4/8 -- still short of trace's 3,192.2).
The conv lowering evidently does not benefit from `torch.compile` the way the transformer
ops do.

#### Native multi-core scaling is essentially perfect

ViT-L BF16 BS=1, independent worker processes, one per core:

| Cores | img/s | Scaling | Efficiency | p50 (ms) |
|------:|------:|--------:|-----------:|---------:|
| 1 | 103.8 | 1.00x | 100.0% | 10.84 |
| 2 | 206.5 | 1.99x | 99.5% | 10.90 |
| 4 | 413.3 | 3.98x | 99.5% | 10.87 |
| 8 | 825.4 | 7.95x | 99.4% | 10.96 |
| **12** | **1,237.5** | **11.92x** | **99.3%** | 10.90 |

**99.3% efficiency at 12 cores with latency flat at ~10.9 ms.** Per-core replicas are fully
independent, so per-instance throughput is just (per-core rate) x (core count).

#### The gain is a compilation-path effect that only appears in BF16

Single core, ViT-L / ViT-B:

| | ViT-L | ViT-B |
|---|---:|---:|
| trace FP32 -> trace BF16 | 1.12x | 1.02x |
| trace BF16 -> Native BF16 (matched precision) | **1.70x** | **1.99x** |
| trace FP32 -> Native FP32 | **0.73x** | **0.86x** |

So tracing BF16 weights is **not** a shortcut to the same gain -- it only buys 2-12%, because
`--auto-cast=matmult` already runs the matmuls in bf16. And at FP32, Native is *slower* than
trace.

#### This contrib intentionally stays on `torch_neuronx.trace()`

- **ViT-7B has no Native path here.** TP uses `neuronx-distributed` ModelBuilder, which is **not present on SDK 2.32**, and trace+NxD measured 1.83x *faster* than Native for ViT-7B TP elsewhere. Switching would regress the largest model and lose the max-model-size result.
- **ConvNeXt would regress 1.25-1.30x** (measured above).
- ViT-H+ and all ViT variants *would* gain on Native (1.6x-9.2x), so a mixed deployment is the real tradeoff, not a clean switch.
- PyTorch Native is **beta software in a private ECR container** whose `:latest` tag has repointed three times.

Use PyTorch Native for **ViT-S/B/L BF16** when a ~1.6x matters and beta tooling is
acceptable. Use the traced path for ConvNeXt, ViT-7B, and anything needing a released SDK.

### Critical: `torch_neuronx.DataParallel` defaults to 2 worker threads

**`torch_neuronx.DataParallel` sets `num_workers = 2` regardless of how many NeuronCores you give it.** When device IDs are consecutive (the default), `_load_modules()` batch-loads into a single module object and sets `num_workers = 2 * 1 = 2`, so `parallel_apply()`'s `ThreadPoolExecutor` serializes all but two core dispatches.

Measured on inf2.24xlarge, ViT-L, 12 cores (single-core reference 65.1 img/s):

| `num_workers` | Throughput | Scaling | p50 (ms) |
|--------------:|-----------:|--------:|---------:|
| **2 (library default)** | 155.7 img/s | 2.39x | 77.05 |
| 4 | 307.9 img/s | 4.73x | 38.93 |
| 8 | 458.0 img/s | 7.04x | 26.19 |
| **12 (= core count)** | **806.2 img/s** | **12.38x** | **14.96** |
| 16 | 802.7 img/s | 12.33x | 14.97 |

**The one-line fix is worth 5.18x.** `num_workers` is not a constructor argument -- set it as an attribute after construction. Raising it above the core count gives nothing.

```python
model_dp = torch_neuronx.DataParallel(model_neuron, device_ids=list(range(num_cores)), dim=0)
model_dp.num_workers = num_cores   # REQUIRED -- default of 2 serializes across cores
```

`benchmark_dataparallel()` in `src/modeling_dinov3.py` now applies this automatically.

With the fix, `DataParallel` (806.2 img/s) comes within **14%** of independent worker processes (939.5 img/s). Processes still win -- they avoid the GIL entirely for input/output marshaling -- and remain the recommendation for production serving, but `DataParallel` is no longer catastrophically slow.

**Use exactly one worker per NeuronCore.** Oversubscribing fails: 4 processes on 2 cores produced `RuntimeError: The PyTorch Neuron Runtime could not be initialized` for the extras. A NeuronCore is single-tenant.


### Key Findings

1. **`--auto-cast=matmult` is critical**: FP32 models get 50-60% speedup with matmult bf16 autocast, consistent with SigLIP and MoLFormer results
2. **Inferentia2 runs the entire DINOv3 family**, including ViT-7B at TP=2 -- and all of it fits on a single **inf2.xlarge** (see "Maximum model size on Inferentia2")
3. **PyTorch Native is faster for every ViT (1.58-1.64x, and 9.2x for ViT-H+ at BS=1) but 1.25-1.30x SLOWER for ConvNeXt.** The verdict splits by architecture. Native multi-core scaling is 99.3% efficient at 12 cores
4. **A custom NKI GELU kernel does NOT help** -- 13 of 14 model/batch configurations regress 5-15%, despite being numerically exact. The compiler already fuses GELU into the surrounding GEMMs (see "Tested and rejected: NKI exact-erf GELU kernel")
5. **`DataParallel` needs `num_workers = num_cores`.** The library default of 2 costs **1.82-1.87x** on 4 trn2 cores and **5.18x** on 12 inf2 cores (the penalty scales with core count, since the default caps concurrency at 2). This is an API fix for anyone using `DataParallel` -- it does not affect the tables in this README, which all use independent worker processes
6. **Inline weights into the NEFF.** `inline_weights=False` costs **6.8x** on inf2 (ViT-L 56.9 -> 8.4 img/s)
7. **Optimal batch size is model-dependent**: small models (ViT-S/B, ConvNeXt-T) peak at BS=1; large models must batch -- **ViT-H+ gains 6.8x from BS=1 -> BS=4**
8. **inf2.xlarge delivers essentially full per-core performance** (0.83-0.99x of inf2.24xlarge). Buy cores for throughput, not efficiency
9. **ViT-H+ must be batched on the traced path.** Its poor BS=1 number is a batching artifact, not a bandwidth limit: on inf2 one core goes 5.1 -> 34.6 img/s from BS=1 -> BS=4, and on trn2 four cores go 11.2 -> 119.8 img/s (10.7x). On PyTorch Native the penalty largely disappears (47.1 img/s at BS=1)
10. **ViT-7B requires TP>=2 on inf2**: at TP=1 the runtime reaches 15.95 GB of the 16 GB core and OOMs. TP must be a power of 2
11. **ConvNeXt is now competitive.** The older claim that "ViT is 1.7x faster than ConvNeXt" no longer holds on SDK 2.31. At the only near-matched pair (ViT-B 85.7M vs ConvNeXt-B 87.6M) **ViT-B is 1.53x faster** on inf2.xlarge (421.4 vs 275.7 img/s) -- ViT still wins at equal size, but by less than before

### Benchmark: Trainium2 (trn2.3xlarge, SDK 2.31)

Re-measured 2026-09-23 with independent worker processes and a batch-size sweep.
trn2.3xlarge is one Trainium2 device: LNC=2 (default) gives 4 logical cores at 24 GB each,
LNC=1 gives 8 at 12 GB.

| Model | LNC=2 (4 cores) | LNC=1 (8 cores) | Best | Winner |
|-------|----------------:|----------------:|-----:|--------|
| ViT-S/16 | 3,780.4 | **3,893.7** | 3,893.7 | LNC=1 |
| ConvNeXt-T | 2,310.9 | **2,401.8** | 2,401.8 | LNC=1 |
| ViT-B/16 | 1,930.5 | **1,969.5** | 1,969.5 | LNC=1 |
| ConvNeXt-B | 1,285.6 | 1,298.6 | 1,298.6 | tie |
| ViT-L/16 | 348.8 | **739.7** | 739.7 | **LNC=1 (2.12x)** |

**LNC=1 is the better default on trn2 for every model tested.** The margin is small
(1.0-1.04x) except for **ViT-L, where LNC=1 is 2.12x faster** -- consistent with the earlier
finding that ViT-L is memory-bandwidth bound and benefits from the smaller per-core NEFF.

#### These numbers supersede the previous trn2 table, which was understated 4.2-5.4x

The prior table reported DP=4 peaks of ViT-S 722.8 / ViT-B 422.9 / ViT-L 174.7 /
ConvNeXt-T 522.6 / ConvNeXt-B 257.8 img/s, measured with `torch_neuronx.DataParallel` at
its **broken `num_workers=2` default** and at **BS=1 only**. Re-measured properly they are
**4.23x to 5.39x higher**.

**Two independent causes were at work, and their relative weight differs sharply by model** --
worth knowing, because the fix differs. Decomposing ViT-L:

| Step | img/s | Factor |
|------|------:|-------:|
| old published (DP, `num_workers=2`, BS=1) | 174.7 | -- |
| re-measured, same config | 164.4 | reproduces the old value |
| + `num_workers=4` | 305.8 | **1.86x** |
| + independent processes and batch sweep | 348.8 | 1.14x |

For **ViT-L the `num_workers` defect is the dominant term** (1.86x of a 2.00x gap). For
**ViT-H+ it contributes nothing**: BS=1 re-measures at 11.2 against an old value of 10.5,
and the entire 10.7x comes from batching (see below). Do not read the 4.2-5.4x range as a
single effect.

The `num_workers` defect itself reproduces on trn2 as on inf2 (LNC=2, 4 cores, BS=1):

| Model | `num_workers=2` (default) | `num_workers=4` (fixed) | Gain |
|-------|--------------------------:|------------------------:|-----:|
| ViT-S/16 | 651.4 img/s | 1,216.0 | 1.87x |
| ViT-B/16 | 400.1 img/s | 726.8 | 1.82x |
| ViT-L/16 | 164.4 img/s | 305.8 | 1.86x |

Note ViT-L's broken-default figure (164.4) closely reproduces the old table's 174.7 --
confirming the historical numbers were measured with the defect.

#### Setting LNC on trn2: `NEURON_CC_FLAGS` is NOT enough

`trace_dinov3()` passes an explicit `compiler_args` list to `torch_neuronx.trace()`, which
**overrides the `NEURON_CC_FLAGS` environment variable**. Exporting
`NEURON_CC_FLAGS="--lnc=1"` therefore has no effect -- the NEFF compiles at the default
`--lnc=2` and the runtime rejects it at load:

```
NRT:nrt_load_util  Mismatch detected between Runtime configuration and NEFF.
    Runtime currently configured with `NEURON_LOGICAL_NC_CONFIG=1`
    but NEFF ... was compiled with `--lnc=2`.
```

Pass the target through `compiler_args` instead, alongside the runtime variable:

```python
compiler_args = COMPILER_ARGS_VIT + ["--logical-nc-config", "1"]   # compile-time
# and export NEURON_LOGICAL_NC_CONFIG=1                            # runtime
```

#### ViT-H+ and ViT-7B on trn2 (SDK 2.31)

| Model | Config | BS=1 | BS=4 | Compile |
|-------|--------|-----:|-----:|--------:|
| ViT-H+/16 | 4 independent procs, LNC=2 | 11.2 | **119.8** | 950s / 525s |
| ViT-7B/16 | TP=4, LNC=2 | **34.9** | -- | -- |

**ViT-H+ gains 10.7x from BS=1 -> BS=4** (11.2 -> 119.8 img/s). Its old SDK 2.28 figure of
10.5 img/s was understated **11.4x**, but note the cause: BS=1 re-measures at 11.2, almost
exactly the old value, so the understatement came **entirely from not batching**, not from
the `num_workers` defect. ViT-H+ must be batched.

ViT-7B TP=4 reproduces at 34.9 img/s vs the old 38.8 (-10%), i.e. no meaningful change on
SDK 2.31. TP models dispatch from a single process, so the `num_workers` defect never
applied to it.

## Compatibility

| Component | Version |
|-----------|---------|
| **Neuron SDK** | 2.31 |
| **torch-neuronx** | 2.9.0.2.15.32035 |
| **neuronx-cc** | 2.26.6360.0 |
| **neuronx-distributed** | 0.19.28492 (for ViT-7B TP only) |
| **torch** | 2.9.1 |
| **Instance (ViT-S/B/L/H+, ConvNeXt-T/B)** | inf2.xlarge and up, trn2.3xlarge |
| **Instance (ViT-7B)** | inf2.24xlarge (TP>=2), trn2.3xlarge (TP=4, LNC=2) |
| **DLAMI** | Deep Learning AMI Neuron (Ubuntu 24.04) 20260708 |

> **SDK 2.32 is not supported.** SDK 2.32 removed `neuronx-distributed` and
> `neuronx-distributed-inference` from the DLAMI entirely, and its vLLM venv ships only
> `libtorch-neuronx-lite` (no `torch_neuronx`). The venv path used below does not exist on
> 2.32. **Use SDK 2.31 (DLAMI 20260708 or 20260813).**


## Usage

### Setup

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate  # SDK 2.31

# Clone DINOv3 repository
git clone https://github.com/facebookresearch/dinov3.git /mnt/models/dinov3
```

### Trace a ViT Model

```python
import sys
sys.path.insert(0, "contrib/models/DINOv3/src")
from modeling_dinov3 import load_dinov3_model, trace_dinov3, validate_accuracy, benchmark_model

# Load model
model = load_dinov3_model("dinov3_vitb16", repo_dir="/mnt/models/dinov3")

# Compile for Neuron
model_neuron = trace_dinov3(model, is_convnext=False, save_path="/tmp/dinov3_vit_b.pt")

# Validate accuracy
metrics = validate_accuracy(model, model_neuron)
print(f"Cosine similarity: {metrics['cosine_sim']:.6f}")

# Benchmark
perf = benchmark_model(model_neuron)
print(f"Throughput: {perf['throughput_img_s']:.1f} img/s")
```

### Trace a ConvNeXt Model

```python
model = load_dinov3_model("dinov3_convnext_tiny", repo_dir="/mnt/models/dinov3")
model_neuron = trace_dinov3(model, is_convnext=True, save_path="/tmp/dinov3_convnext_t.pt")
```

### Compile ViT-7B with Tensor Parallelism

```python
from modeling_dinov3 import compile_vit7b_tp, benchmark_model

# On inf2: TP=2 is the minimum (TP=1 OOMs at 16 GB/core). TP must be a power of 2.
# Requires NEURON_RT_NUM_CORES >= tp_degree.
nxd_model = compile_vit7b_tp(tp_degree=2)     # inf2.24xlarge
# nxd_model = compile_vit7b_tp(tp_degree=4)   # trn2.3xlarge
perf = benchmark_model(nxd_model)
print(f"throughput: {perf['throughput_img_s']:.1f} img/s")
```

### Multi-core throughput

Two options. `DataParallel` is fine **if you set `num_workers`** (see above); independent
processes are slightly faster and are the production recommendation:

```bash
# Launch one worker per core, each pinned to its own NeuronCore.
for core in $(seq 0 11); do
    NEURON_RT_VISIBLE_CORES=$core NEURON_RT_NUM_CORES=1 \
        python my_server.py --port $((8000 + core)) &
done
```

```python
from modeling_dinov3 import benchmark_dataparallel

# Convenience API -- benchmark_dataparallel() sets num_workers = num_cores for you
dp_results = benchmark_dataparallel(model_neuron, num_cores=12)
```

## Running Tests

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate  # SDK 2.31

# Run integration tests
python -m pytest contrib/models/DINOv3/test/integration/test_model.py -v

# Or standalone (with detailed output)
python contrib/models/DINOv3/test/integration/test_model.py

# Set custom paths
DINOV3_REPO_DIR=/path/to/dinov3 python -m pytest contrib/models/DINOv3/test/ -v
```

## Dependencies

Pre-installed in DLAMI PyTorch inference venv:
- torch-neuronx
- neuronx-distributed (for ViT-7B TP)
- numpy

Required (clone separately):
- DINOv3 repository: `git clone https://github.com/facebookresearch/dinov3.git`

## Notes

- All models use `pretrained=False` (random weights) for architecture validation. Replace with pretrained weights for production use.
- `--model-type=transformer` compiler flag is used for ViT models only (not ConvNeXt).
- ConvNeXt models exercise different Neuron ops (Conv2d, depthwise conv, GroupNorm) -- good diversity test for Neuron compiler.
- ViT-H+ traces successfully but is HBM-bandwidth-limited (1.36 GB artifact, 5.1 img/s per core). It still scales ~12x across 12 cores, so it is usable when throughput matters more than latency.
- **Keep `inline_weights=True` (the default) on inf2.** Disabling it costs 6.8x throughput -- see "Critical: always inline weights into the NEFF on inf2". A prior recommendation to set `inline_weights=False` at BS>=4 came from a trn2/SDK 2.30 async XLA error and does not apply to inf2 on SDK 2.31.
- **Older SDKs miscompiled RoPE on Inferentia2.** `torch.Tensor.split()` on dim>=2 of a 4D tensor was compiled incorrectly by neuronx-cc 2.25.3371, collapsing ViT-L cosine similarity to 0.13 on inf2. Current upstream DINOv3 uses index slicing in `Attention.apply_rope`, and **cosine similarity is 1.000000 on inf2 with SDK 2.31** -- verified, no workaround needed. If you pin an older DINOv3 commit that still uses `split()`, replace it with index slicing or `.narrow()`.
- DINOv3 License is not Apache/MIT -- review before redistribution.


## Maintainer

Jim Burtoft (`jimburtoft`)
