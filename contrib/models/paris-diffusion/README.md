# Contrib Model: Paris Multi-Expert Diffusion

NeuronX inference implementation of Bagel Labs' **Paris** — an 8-expert Mixture-of-Experts
DiT (Diffusion Transformer) for 256x256 text-to-image generation, compiled via `torch_neuronx.trace()`.

## Model Information

- **HuggingFace ID:** [`bageldotcom/paris`](https://huggingface.co/bageldotcom/paris)
- **Model Type:** Multi-expert DiT (Diffusion Transformer) for text-to-image
- **Parameters:** 8 x 606M experts + 129M router = ~5B total
- **License:** MIT
- **Paper:** [Scaling Diffusion Models with Mixture of Expert Routing](https://arxiv.org/abs/2510.03434)

## Architecture Details

Paris is a **model-level Mixture-of-Experts** diffusion model with five components:

| Component | Architecture | Params | Purpose |
|-----------|-------------|--------|---------|
| Text encoder | CLIP ViT-L/14 | 123M | 77-token text embeddings (768-dim) |
| Router | DiT-B (AdaLN-Zero, CLS token) | 129M | Per-timestep expert selection over 8 experts |
| Experts (x8) | DiT-XL/2 (AdaLN-Single, PixArt-alpha style) | 606M each | Velocity prediction on 32x32 latents |
| VAE decoder | AutoencoderKL (sd-vae-ft-mse) | 50M | Latent-to-pixel decoding |
| Scheduler | FlowMatchEulerDiscreteScheduler | — | 50-step flow matching |

**Key difference from standard LLM MoE:** The router is a full 129M-parameter DiT-B transformer
(not a single linear layer). It makes a **per-timestep** routing decision for the entire image,
selecting which expert(s) to run at each diffusion step.

### Routing Strategies

- **top-1:** Run only the highest-scoring expert (fastest, 1 expert call per step)
- **top-2:** Weighted average of top-2 experts (best FID 22.60, 2 expert calls per step)
- **full:** Weighted average of all 8 experts (8 expert calls per step)

### CFG Batching

Classifier-free guidance (CFG) is optimized by batching the conditioned and unconditioned
forward passes into a single BS=2 expert call. Experts are traced at batch size 2 for this purpose.
This yields ~30% speedup over sequential CFG.

## Validation Results

**Validated:** 2026-03-23
**Configuration:** trn2.3xlarge, LNC=2, SDK 2.28, experts at BS=2 (CFG-batched)

### Benchmark: Neuron vs GPU (50-step generation)

| Strategy | GPU (A10G) | GPU + torch.compile | Neuron (trn2) | Speedup vs GPU | Speedup vs compiled |
|----------|:---:|:---:|:---:|:---:|:---:|
| Top-1 | 2,851 ms | 1,603 ms | **880 ms** | **3.24x** | **1.82x** |
| Top-2 | 5,318 ms | 2,950 ms | **1,535 ms** | **3.46x** | **1.92x** |
| Full | 20,202 ms | — | **5,413 ms** | **3.73x** | — |

### Cost Efficiency

| | g5.2xlarge (A10G) | trn2.3xlarge |
|---|:---:|:---:|
| Hourly cost | $2.06 | $2.24 |
| Top-1 images/hr | 1,263 | 4,091 |
| Cost per 1k images | $1.63 | **$0.55** |
| **Cost ratio** | baseline | **2.97x cheaper** |

### Component Accuracy (Neuron vs CPU reference)

| Component | Cosine Similarity |
|-----------|:-:|
| CLIP text encoder | 0.999990 |
| Router | 1.000000 |
| Experts (avg of 8) | > 0.99998 |
| VAE decoder | 0.999970 |

### Test Results

| Test | Status | Result |
|------|--------|--------|
| Pipeline Load | PASS | All 11 NEFFs load successfully |
| CLIP Determinism | PASS | cos_sim > 0.999 |
| Router Distribution | PASS | softmax sums to 1.0 |
| Expert Shape (BS=2) | PASS | [2, 4, 32, 32] output |
| Top-1 Generation | PASS | 256x256 RGB, non-blank |
| Top-2 Generation | PASS | 256x256 RGB, non-blank |
| Determinism (same seed) | PASS | Pixel-identical outputs |
| Different Seeds Differ | PASS | MSE > 100 |
| Top-1 Latency | PASS | < 2000 ms (50 steps) |
| Top-2 Latency | PASS | < 3000 ms (50 steps) |

## Usage

### 1. Download Model Weights

```bash
huggingface-cli download bageldotcom/paris --local-dir /home/ubuntu/models/paris
```

### 2. Compile (Trace) All Components

```python
from src.modeling_paris import trace_all

results = trace_all(
    model_dir="/home/ubuntu/models/paris",
    output_dir="/home/ubuntu/neuron_models/paris",
    expert_batch_size=2,  # 2 = CFG batching (recommended)
)
# Compiles: CLIP, router, 8 experts (BS=2), VAE decoder
# Total compile time: ~25 minutes on trn2.3xlarge
```

### 3. Run Inference

```python
from src.modeling_paris import ParisPipeline

pipeline = ParisPipeline(
    neff_dir="/home/ubuntu/neuron_models/paris",
    model_dir="/home/ubuntu/models/paris",
    expert_batch_size=2,
)

# Generate with top-2 routing (best quality)
image = pipeline.generate(
    prompt="A beautiful sunset over the ocean",
    routing="top2",
    cfg_scale=7.5,
    num_steps=50,
    seed=42,
)
image.save("output.png")
```

### Routing Strategy Selection

```python
# Fastest (880ms): single expert per step
image = pipeline.generate("A cat", routing="top1")

# Best quality (1,535ms): weighted blend of top-2 experts
image = pipeline.generate("A cat", routing="top2")

# Maximum ensemble (5,413ms): all 8 experts
image = pipeline.generate("A cat", routing="full")
```

## Compatibility Matrix

| Instance / SDK | 2.28 | 2.27 and earlier |
|----------------|------|-------------------|
| Trn2 (trn2.3xlarge) | PASS | Not tested |
| Trn1 | Not tested | Not tested |
| Inf2 | Not tested | Not tested |

## Testing

### Prerequisites

1. Download model weights to `/home/ubuntu/models/paris`
2. Compile NEFFs to `/home/ubuntu/neuron_models/paris` (see Usage above)

### Run Integration Tests

```bash
pytest contrib/models/paris-diffusion/test/integration/test_model.py -v --timeout=600
```

Or run manually:

```bash
cd contrib/models/paris-diffusion
python test/integration/test_model.py
```

## Dependencies

- `torch-neuronx >= 2.5`
- `torch >= 2.5`
- `transformers`
- `diffusers`
- `safetensors`
- `Pillow`
- `numpy`

## Design Notes

This model uses `torch_neuronx.trace()` directly rather than NxDI's `NeuronBaseForCausalLM`.
The Paris architecture (multi-expert diffusion with per-timestep routing) does not fit the
autoregressive LLM paradigm that NxDI's base classes are designed for. The pipeline orchestration
(routing, CFG, scheduler loop) runs on CPU while individual component forward passes execute on
Neuron cores.

## Example Checkpoints

- [bageldotcom/paris](https://huggingface.co/bageldotcom/paris) (MIT license)

## Maintainer

Jim Burtoft

**Last Updated:** 2026-03-24
