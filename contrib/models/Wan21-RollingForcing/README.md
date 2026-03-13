# Contrib Model: Wan 2.1 RollingForcing

Wan 2.1 14B DiT with RollingForcing temporal acceleration running on AWS Trainium 2 via NxD Inference. Generates 81-frame (480x832) video from text prompts using a 5-step graduated-noise rolling denoising pipeline with DMD distilled weights.

## Model Information

- **HuggingFace ID:** [`Wan-AI/Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) (base weights)
- **Paper:** [RollingForcing: Graduated Noise for Video Diffusion Generation](https://github.com/TencentARC/RollingForcing)
- **Model Type:** DiT (Diffusion Transformer) for text-to-video generation with rolling-forcing temporal acceleration
- **Parameters:** ~14B (BF16) -- 30 transformer blocks, 12 heads, 128 head dim, 1536 hidden dim
- **Architecture:** Full self-attention + cross-attention with RoPE, single-NEFF unified compilation (all attention paths through one cached NEFF), causal 3D VAE for video decoding
- **License:** See HuggingFace model card and TencentARC RollingForcing repository
- **Algorithm:** RollingForcing divides the 21-latent-frame sequence into 7 blocks of 3 frames each, denoising them in 11 sliding windows with 5 graduated noise levels (1000, 800, 600, 400, 200). Each window denoises all blocks, stores predictions, re-noises to the next cleaner level, and runs a cache update call at timestep=0.

## Validation Results

**Validated:** 2026-03-13
**Instance:** trn2.3xlarge (TP=4, LNC=2, 4 logical NeuronCores)
**SDK:** Neuron SDK 2.28, PyTorch 2.9, Deep Learning AMI Neuron (Ubuntu 24.04) 20260227

### Accuracy Validation

| Component | Metric | Value | Notes |
|-----------|--------|-------|-------|
| Neuron VAE decoder vs CPU | PSNR | 50.22 dB | v5 1-frame compilation, correct cache propagation |
| Neuron VAE decoder vs CPU | Cosine similarity | > 0.999 | Full 81-frame video decode |
| DiT forward pass | Non-NaN, non-Inf | PASS | All attention modes validated |

### Benchmark Results

| Stage | Time | Notes |
|-------|------|-------|
| DiT model loading | ~90s | Pre-compiled unified NEFF + pre-sharded weights, 4 NeuronCores |
| Pipeline generation (11 windows) | ~40s | 22 model calls (11 main + 11 update), warm steady-state |
| VAE decode (CPU) | ~480s | Full 21-chunk causal decode |
| VAE decode (Neuron hybrid) | ~60s | CPU for chunks 0-1, Neuron for chunks 2-20 |
| **Total end-to-end (Neuron VAE)** | **~190s** | Including model load, generation, and decode |

### Component Distribution

| Component | Location | Notes |
|-----------|----------|-------|
| DiT transformer (30 blocks) | **Neuron** (TP=4) | Single-NEFF unified compilation |
| 3D Causal VAE (chunks 2-20) | **Neuron** (TP=1) | Compiled decoder, 3.7 GB NEFF |
| 3D Causal VAE (conv2 + chunks 0-1) | CPU | Handles "Rep" string ops, not compilable |
| T5 text encoder (UMT5-XXL) | CPU | ~11 GB, used for text embedding generation |
| Flow matching scheduler | CPU | `convert_flow_pred_to_x0`, `add_noise` |

## Usage

### Prerequisites

```bash
# On trn2.3xlarge with Deep Learning AMI Neuron (Ubuntu 24.04) 20260227
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Clone TencentARC RollingForcing repository (needed for VAE and DMD weights)
git clone https://github.com/TencentARC/RollingForcing.git
cd RollingForcing
# Download DMD weights per repository instructions

# Download Wan 2.1 base weights
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir /path/to/wan_weights/
```

### Step 1: Convert DMD Weights

```bash
python src/convert_dmd_weights.py \
  --dmd-checkpoint /path/to/RollingForcing/checkpoints/rolling_forcing_dmd.pt \
  --output /path/to/dmd_converted/model.safetensors
```

### Step 2: Generate Text Embeddings

```bash
python src/gen_text_embeddings.py \
  --wan-repo /path/to/RollingForcing \
  --output-dir /path/to/text_embeddings
```

### Step 3: Compile the DiT Backbone

```bash
# Phase 2 (main compilation) -- produces single unified model.pt
python src/compile_transformer.py \
  --weight-path /path/to/wan_weights/ \
  --compiled-path /path/to/compiled_dit/ \
  --tp-degree 4 \
  --phase 2
```

Compilation takes approximately 30-60 minutes. The compiled model is a single NEFF (~2 GB) with shared weights.

### Step 4: Compile the VAE Decoder

```bash
export WAN_REPO_PATH=/path/to/RollingForcing
python src/compile_vae_decoder.py \
  --vae_path /path/to/wan_weights/Wan2.1_VAE.pth \
  --compiled_models_dir /path/to/compiled_vae/ \
  --model-version v5
```

### Step 5: Run End-to-End Generation

```bash
WEIGHT_PATH=/path/to/dmd_converted/ \
COMPILED_MODEL_PATH=/path/to/compiled_dit/ \
WAN_REPO_PATH=/path/to/RollingForcing \
python test/integration/test_model.py -v -s
```

## Compatibility Matrix

| Instance/Version | SDK 2.28 |
|------------------|----------|
| trn2.3xlarge (TP=4, LNC=2) | VALIDATED |

## Example Checkpoints

* [`Wan-AI/Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) -- Wan 2.1 base weights
* [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing) -- DMD distilled weights and VAE code

## Testing Instructions

```bash
# Set environment variables pointing to your model artifacts
WEIGHT_PATH=/path/to/dmd_converted/ \
COMPILED_MODEL_PATH=/path/to/compiled_dit/ \
COMPILED_VAE_PATH=/path/to/compiled_vae/ \
VAE_WEIGHTS_PATH=/path/to/wan_weights/Wan2.1_VAE.pth \
WAN_REPO_PATH=/path/to/RollingForcing \
  pytest test/integration/test_model.py -v -s
```

Tests validate:
- DiT forward pass produces valid (non-NaN) output
- Rolling-forcing pipeline generates latents of correct shape [1, 16, 21, 60, 104]
- Pipeline is deterministic with the same seed
- Neuron VAE decoder achieves >= 40 dB PSNR vs CPU reference

## Architecture Details

### Single-NEFF Unified Compilation

The DiT backbone is compiled as a single NEFF that handles all three attention modes:
- **Self-attention**: Zero KV buffers + current-only mask (windows 0-4 main calls)
- **Cached attention**: Anchor KV + working cache + current input (windows 5-10)
- **Update-via-cached**: Padded f15 with full cache KV (all update calls)

This eliminates Neuron compiler numerical non-determinism between separate NEFFs.

### Backbone Input Signature

| Input | Shape | Description |
|-------|-------|-------------|
| hidden_states | (1, 16, 15, 60, 104) | Latent video tensor (padded to 15 frames) |
| timestep | (1, 15) | Per-frame timestep (float32) |
| encoder_hidden_states | (1, 512, 4096) | T5 text embeddings |
| rope_cos, rope_sin | (1, 12, seq, 64) | Precomputed RoPE embeddings |
| attn_mask | (1, 1, seq, 32760) | Additive attention mask |
| kv_buffers | 30 x (1, 32760, 3, 128) x 2 | Per-layer K and V cache buffers |

### VAE Architecture

- 3D causal VAE: `dim=96`, `z_dim=16`, `dim_mult=[1,2,4,4]`
- 33 CausalConv3d modules, 32 active feat_cache slots
- Hybrid decode strategy: CPU for conv2 + chunks 0-1 (handle "Rep" string ops), Neuron for chunks 2-20
- Temporal compression: 4x (each latent frame -> 4 pixel frames, except first -> 1)
- 21 latent frames -> 81 pixel frames at 480x832

### Compiler Flags

**DiT backbone:**
```
--model-type=transformer -O1 --auto-cast matmult --lnc 2
```

Environment: `NEURON_FUSE_SOFTMAX=1`, `NEURON_CUSTOM_SILU=1`

**VAE decoder:**
```
--auto-cast matmult
```

## Known Issues

- **Cold start latency**: First forward pass through the DiT NEFF takes ~140s for Neuron device initialization. Subsequent calls are fast (~1.5s per window).
- **VAE chunks 0-1 on CPU**: The first two VAE decoder chunks use Python string operations ("Rep") in layer names that cannot be traced by the Neuron compiler. These run on CPU; chunks 2-20 run on Neuron.
- **Block boundary artifacts**: The RollingForcing algorithm produces subtle visual jumps at block boundaries (every 12 pixel frames). This is inherent to the graduated-noise rolling approach and present in the reference implementation as well.
- **T5 text encoder on CPU**: The UMT5-XXL text encoder (~11 GB) runs on CPU and takes ~160s. Pre-compute embeddings with `gen_text_embeddings.py` to avoid this at inference time.
- **Memory**: The DiT backbone + VAE decoder both need to fit on 4 NeuronCores (96 GB total HBM). Sequential loading is used -- DiT is unloaded before VAE loads.

## Source Files

| File | Purpose |
|------|---------|
| `src/modeling_wan21.py` | Core NxDI model: TP sharding, self/cached/update attention, RoPE, KV cache management |
| `src/pipeline.py` | Rolling-forcing pipeline: 11-window graduated-noise loop, flow matching scheduler, VAE orchestration |
| `src/window_schedule.py` | Pre-computed deterministic schedule for all 22 model calls |
| `src/compile_transformer.py` | DiT backbone compilation (single-NEFF unified architecture) |
| `src/compile_vae_decoder.py` | VAE decoder compilation (hybrid CPU/Neuron) |
| `src/decode_vae_neuron.py` | VAE decode runtime with cache propagation |
| `src/gen_text_embeddings.py` | T5 text encoder embedding generation utility |
| `src/convert_dmd_weights.py` | DMD checkpoint to safetensors weight conversion |
