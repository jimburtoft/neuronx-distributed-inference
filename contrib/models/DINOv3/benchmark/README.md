# Reproducing the DINOv3 benchmarks

Every number in the parent [README.md](../README.md) was produced by a script in this
directory. This file maps each table to its script and gives the exact command.

All results were measured on **Neuron SDK 2.31**:

| Component | Version |
|---|---|
| DLAMI | `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708` |
| neuronx-cc | 2.26.6360.0 |
| torch-neuronx | 2.9.0.2.15.32035 |
| neuronx-distributed | 0.19.28492 (ViT-7B TP only) |
| torch | 2.9.1 |
| venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` |

> **SDK 2.32 will not work.** It removed `torch_neuronx` and `neuronx_distributed` from the
> DLAMI entirely. Use a 2.31 DLAMI (`20260708` or `20260813`).

## Setup

```bash
# 1. Launch an inf2 or trn2 instance with the SDK 2.31 DLAMI above.
#    inf2.xlarge (2 cores) is enough for the single-instance tables.
#    inf2.24xlarge (12 cores) is needed for the 12-core scaling numbers.

# 2. Add swap. inf2.xlarge has only 15 GB RAM and neuronx-cc needs far more per
#    graph; without swap, large-model compiles are OOM-killed.
sudo fallocate -l 96G /swapfile && sudo chmod 600 /swapfile
sudo mkswap --force /swapfile && sudo swapon /swapfile

# 3. Clone the DINOv3 reference repo (the model source).
sudo mkdir -p /mnt/models && sudo chown $USER /mnt/models
git clone --depth 1 https://github.com/facebookresearch/dinov3.git /mnt/models/dinov3

# 4. Activate the venv and point the scripts at the repo + model source.
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export DINOV3_REPO_DIR=/mnt/models/dinov3
export DINOV3_CONTRIB_SRC=$PWD/../src
```

## Which script produces which table

| README table | Script | Command |
|---|---|---|
| **inf2.xlarge max throughput** | `bench_max_throughput.py` | `--models vit_s,vit_b,vit_l,convnext_tiny,convnext_base --workers 2 --batch-sizes 1,4,8,16 --n-cores 2` |
| **inf2 single-core + accuracy + artifact size** | `bench_inf2.py` | `--models vit_s,vit_b,vit_l,vit_h_plus,convnext_tiny,convnext_base --dp-cores 12` |
| **Maximum model size / ViT-7B TP** | `bench_vit7b_tp_inf2.py` | `NEURON_RT_NUM_CORES=8 python bench_vit7b_tp_inf2.py --tp 2,4,8` |
| **trn2 LNC=2 / LNC=1** | `bench_trn2.py` | see [trn2](#trn2) below |
| **trn2 ViT-H+ and ViT-7B** | `bench_trn2_big.py` | `--mode vith --cores 4 --batch-sizes 1,4` then `--mode vit7b --tp 4` |
| **`DataParallel` num_workers** | `bench_dp_numworkers.py` | `--model vit_l --cores 12 --workers 2,4,8,12,16` |
| **Multi-process vs DataParallel** | `bench_dp_mp.py` | `--model vit_l --cores 1,12 --batch-size 1` (one `--model` at a time) |
| **BF16 vs FP32+matmult** | `bench_bf16_default.py` | `--models vit_s,vit_b,vit_l,convnext_tiny,convnext_base --cores 12 --batch-sizes 1,4,8` |
| **Accuracy on real pretrained weights** | `validate_real_weights.py` | `--models vit_s,vit_b,vit_l --dtypes fp32,bf16 --n 64` |
| **recall@1 near-tie analysis** | `diag_recall.py` | `--model vit_l --dtype bf16 --n 64` |
| **Rejected NKI GELU kernel** | `ab_gelu_all.py` | `--models vit_s,vit_b,convnext_tiny,convnext_base --batch-sizes 1,4,8` |

Supporting modules (not run directly): `hf_to_repo_weights.py` remaps HuggingFace
checkpoints into the reference-repo architecture; `test_nki_gelu_inf2.py` is the 3-gate
harness for the NKI kernel.

## Headline numbers and how to get them

### inf2.xlarge (2 NeuronCores) -- maximum throughput

```bash
python bench_max_throughput.py \
    --models vit_s,vit_b,vit_l,convnext_tiny,convnext_base \
    --workers 2 --batch-sizes 1,4,8,16 --n-cores 2 \
    --out results_max.json
```
Reports the best (workers x batch) per model. Expect ViT-S ~832, ConvNeXt-T ~502,
ViT-B ~421, ConvNeXt-B ~276, ViT-L ~131 img/s.

### inf2.24xlarge (12 NeuronCores)

Same script with `--workers 12 --n-cores 12`. Expect ViT-S ~5,152, ConvNeXt-T ~3,192,
ViT-B ~2,563, ConvNeXt-B ~1,716, ViT-L ~940 img/s.

### ViT-7B with tensor parallelism

```bash
# TP must be a power of 2, and TP>=2 is required (TP=1 OOMs at 16 GB/core).
NEURON_RT_NUM_CORES=8 python bench_vit7b_tp_inf2.py --tp 2,4,8 --out results_7b.json
```

### trn2

LNC must be set in **two** places -- the runtime env var **and** the compile target -- or the
runtime rejects the NEFF at load with a `Mismatch detected between Runtime configuration and
NEFF` error.

`NEURON_CC_FLAGS="--lnc=N"` alone is **not** sufficient: `trace_dinov3()` passes an explicit
`compiler_args` list, which overrides the env var, so the NEFF silently compiles at the
default `--lnc=2`. `bench_trn2.py` reads `NEURON_LOGICAL_NC_CONFIG` and appends
`--logical-nc-config` to `compiler_args` for you, so setting the one env var is enough here:

```bash
export NEURON_LOGICAL_NC_CONFIG=2          # runtime: 4 logical cores at 24 GB
python bench_trn2.py --models vit_s,vit_b,vit_l,convnext_tiny,convnext_base \
    --cores 4 --batch-sizes 1,4,8 --mode mp --out results_trn2_lnc2.json

export NEURON_LOGICAL_NC_CONFIG=1          # runtime: 8 logical cores at 12 GB
python bench_trn2.py --models vit_s,vit_b,vit_l,convnext_tiny,convnext_base \
    --cores 8 --batch-sizes 1,4,8 --mode mp --out results_trn2_lnc1.json
```

### trn2 ViT-H+ and ViT-7B

`bench_trn2_big.py` selects the model with `--mode`, not `--models`:

```bash
export NEURON_LOGICAL_NC_CONFIG=2
python bench_trn2_big.py --mode vith  --cores 4 --batch-sizes 1,4   # ViT-H+ (trace)
NEURON_RT_NUM_CORES=4 python bench_trn2_big.py --mode vit7b --tp 4  # ViT-7B (TP)
```

ViT-H+ compiles take ~950 s at BS=1 on trn2, so allow time.

### Accuracy on real pretrained weights

Needs an **approved HuggingFace token** -- the `facebook/dinov3-*` checkpoints are
manually gated. Request access on the model page first, then:

```bash
mkdir -p ~/.cache/huggingface && echo "$HF_TOKEN" > ~/.cache/huggingface/token
pip install 'transformers>=4.57'
python validate_real_weights.py --models vit_s,vit_b,vit_l --dtypes fp32,bf16 --n 64
```

The checkpoints ship in `transformers` format, which **`torch_neuronx.trace()`
miscompiles** (cos 0.47-0.62). `hf_to_repo_weights.py` remaps them into the reference-repo
architecture and self-verifies on CPU -- it aborts unless HF-vs-repo agreement is ~1.0.

## Methodology notes -- read these before comparing your numbers

These are not incidental. Each one produced a materially wrong result during this work.

1. **Use independent worker processes, not `torch_neuronx.DataParallel`.** `DataParallel`
   pins `num_workers = 2` regardless of core count, costing up to 5.18x on 12 cores. All
   tables here use one process per core, pinned with `NEURON_RT_VISIBLE_CORES`. If you do
   use `DataParallel`, set `model_dp.num_workers = num_cores` after construction.

2. **One worker per NeuronCore. No oversubscription.** A NeuronCore is single-tenant; extra
   processes fail with `The PyTorch Neuron Runtime could not be initialized`.

3. **Never compile and measure at the same time on a small host.** On a 15 GB inf2.xlarge
   this drove 10.7 GB into swap and corrupted the numbers -- one config read 147.6 img/s
   against ~310 for its neighbours. `bench_bf16_default.py` compiles everything in a first
   phase, then measures. Prefer that pattern.

4. **A loaded traced model pins its core for the process lifetime.** `gc.collect()` does not
   release it; only process exit does. That is why the scripts fan out to subprocesses and
   sleep between phases.

5. **Repeat the baseline until it converges.** A cold baseline against a warm variant
   manufactures gains. One ViT-L BS=1 baseline climbed 52.0 -> 62.4 -> 68.4 img/s across
   three back-to-back runs; comparing against the first produced a fabricated +32%. Always
   take best-of-N, and treat a variant with much lower variance than its baseline as a
   measurement smell.

6. **Keep `inline_weights=True`** (the default). Disabling it costs 6.8x on inf2 -- weights
   are re-fed from the host on every call.

7. **Optimal batch size is per-model.** Small models (ViT-S/B, ConvNeXt-T) peak at BS=1-4;
   ViT-H+ must be batched (or run in BF16). Sweep it rather than assuming.

## Expected run times

| Scope | Approx wall clock |
|---|---|
| `bench_max_throughput.py`, 5 models x 4 batches, 2 cores | ~2-3 h |
| Same on 12 cores | ~3-4 h |
| `bench_inf2.py` full sweep (includes ViT-H+, ~330 s compile) | ~1.5 h |
| `bench_trn2.py` one LNC mode | ~2 h |
| `validate_real_weights.py` 3 models x 2 dtypes | ~40 min |

ViT-H+ dominates compile time (330 s on inf2.24xlarge, ~950 s on trn2). ViT-7B TP compiles
are fast (6-9 s per rank) because each rank compiles a smaller graph.
