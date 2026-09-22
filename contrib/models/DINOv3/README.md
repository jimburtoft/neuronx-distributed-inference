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
3. **A custom NKI GELU kernel does NOT help** -- 13 of 14 model/batch configurations regress 5-15%, despite being numerically exact. The compiler already fuses GELU into the surrounding GEMMs (see "Tested and rejected: NKI exact-erf GELU kernel")
4. **`DataParallel` needs `num_workers = num_cores`.** The library default of 2 costs **5.18x** on 12 cores (155.7 -> 806.2 img/s). This is the single largest configuration issue found
5. **Inline weights into the NEFF.** `inline_weights=False` costs **6.8x** on inf2 (ViT-L 56.9 -> 8.4 img/s)
6. **Optimal batch size is model-dependent**: small models (ViT-S/B, ConvNeXt-T) peak at BS=1; large models must batch -- **ViT-H+ gains 6.8x from BS=1 -> BS=4**
7. **inf2.xlarge delivers essentially full per-core performance** (0.83-0.99x of inf2.24xlarge). Buy cores for throughput, not efficiency
8. **ViT-H+ is not fundamentally bandwidth-starved.** Its poor BS=1 number (5.1 img/s/core) is mostly a batching artifact: BS=4 reaches 34.6 img/s on one core
9. **ViT-7B requires TP>=2 on inf2**: at TP=1 the runtime reaches 15.95 GB of the 16 GB core and OOMs. TP must be a power of 2
10. **ConvNeXt is now competitive.** The older claim that "ViT is 1.7x faster than ConvNeXt" no longer holds on SDK 2.31. At the only near-matched pair (ViT-B 85.7M vs ConvNeXt-B 87.6M) **ViT-B is 1.53x faster** on inf2.xlarge (421.4 vs 275.7 img/s) -- ViT still wins at equal size, but by less than before

### Benchmark: Trainium2 (trn2.3xlarge, LNC=2)

Retained for reference. These are historical numbers from SDK 2.28/2.29 and use
`torch_neuronx.DataParallel`, so the DP=4 column understates the hardware for the same
reason described above.

| Model | Compile Time | 1-Core (img/s) | DP=4 Peak (img/s) |
|-------|-------------:|----------------:|-------------------:|
| ViT-S/16 | 84s | 367 | 722.8 |
| ViT-B/16 | 89s | 214.4 | 422.9 |
| ViT-L/16 | 123s | 87.6 | 174.7 |
| ViT-H+/16 | 688s | 5.2 | 10.5 |
| ViT-7B/16 | 5.9s | OOM | 38.8 (TP=4) |
| ConvNeXt-T | 34s | 264.4 | 522.6 |
| ConvNeXt-B | 63s | 130 | 257.8 |

*ViT-S/L/H+/7B and ConvNeXt-B from SDK 2.28; ViT-B and ConvNeXt-T validated on SDK 2.29.*

On trn2, **LNC=1 outperforms LNC=2 for ViT-L** (207.6 vs 114.7 img/s peak, 1.81x) because
the model is memory-bandwidth bound and the LNC=1 NEFF is smaller (468 MB vs 935 MB).

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
