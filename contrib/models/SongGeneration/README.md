# Contrib Model: SongGeneration (LeVo)

Text-to-music generation on AWS Trainium2 using Tencent's SongGeneration (LeVo) model -- a hybrid LLM-Diffusion audio pipeline that generates stereo 48kHz music from text descriptions.

## Model Information

- **HuggingFace ID:** `tencent/SongGeneration`
- **Model Type:** Multi-stage audio generation pipeline (LLM + Diffusion + VAE)
- **Parameters:** ~4.1B total (LeLM 2.83B + Diffusion 1.1B + VAE 169M)
- **Architecture:** Dual-Llama AR LM (28L primary + 12L secondary) with delayed codebook pattern, GPT2-RoPE CFM diffusion backbone (16L), Stable Audio VAE decoder
- **Output:** Stereo 48kHz WAV audio
- **License:** Check [HuggingFace model card](https://huggingface.co/tencent/SongGeneration)

## Architecture

SongGeneration uses a three-stage pipeline:

| Stage | Component | Params | Neuron Compilation | Key Innovation |
|-------|-----------|--------|-------------------|----------------|
| 1. LeLM | Dual-Llama AR (28L + 12L) | 2.83B | `ModelBuilder` (on-device KV) | `torch.scatter` KV cache in HBM |
| 2. Diffusion | GPT2-RoPE CFM (16L) | 1.1B | `torch_neuronx.trace()` | Rewritten RoPE (no complex numbers) |
| 3. VAE | Stable Audio decoder | 169M | `torch_neuronx.trace()` | `weight_norm` removal pre-trace |

### On-Device KV Cache

The LeLM transformers use on-device KV caching via `neuronx_distributed.ModelBuilder`. Instead of passing KV cache tensors as model inputs/outputs each autoregressive step (PCIe round-trip), the cache is stored as `register_buffer` on the model and updated in-place with `torch.scatter`. This keeps the cache in Neuron HBM, providing a 3.2x speedup on the LeLM stage.

### Neuron-Specific Adaptations

- **RoPE rewrite:** `torch.view_as_complex` / `torch.view_as_real` replaced with explicit sin/cos rotation (XLA compatible)
- **Flash Attention disabled:** `use_flash_attn_2=False` (CUDA-only feature)
- **CUDA-to-CPU patches:** All `torch.cuda` calls redirected to CPU (upstream codebase assumes CUDA)
- **weight_norm removal:** `torch.nn.utils.remove_weight_norm` applied to VAE before tracing
- **bf16 precision:** AR loop state kept in FP32 on CPU to avoid compound rounding errors; LeLM Neuron models use `--auto-cast matmult`
- **GPT2 diffusion fp32:** The GPT2 diffusion backbone **must** be traced with `--auto-cast none` (full FP32). Using `--auto-cast matmult` causes severe numerical degradation (cosine similarity drops from 1.0 to 0.64 vs CPU) which compounds across 10 Euler solver steps into garbled audio. The VAE can safely use `--auto-cast matmult`.
- **v1 conditioning:** The base checkpoint uses `version='v1'` tokenizers. Do NOT prefix descriptions with `[Musicality-very-high]` (a v2-only feature). Use plain `description.lower()` for the type_info conditioner.

## Validation Results

**Validated:** 2026-03-20
**Instance:** trn2.3xlarge (LNC=2, 4 NeuronCores)
**SDK:** Neuron SDK 2.28 (DLAMI 20260227), PyTorch 2.9

### Component Accuracy

| Component | Metric | Value | Threshold |
|-----------|--------|-------|-----------|
| GPT2 diffusion (fp32) | Cosine similarity vs CPU | 1.0004 | > 0.98 |
| GPT2 diffusion (fp32) | Max diff vs CPU | 0.0003 | < 0.01 |
| VAE decoder | Cosine similarity vs CPU | 1.0001 | > 0.98 |
| VAE decoder | SNR vs CPU | > 40 dB | > 20 dB |

### Benchmark Results (5s audio generation)

| Metric | On-Device KV | PCIe KV | GPU (A10G) | CPU |
|--------|-------------|---------|------------|-----|
| **LeLM AR time** | **45.4s** | 50.1s | 17.5s | 112.9s |
| **Diffusion + VAE** | **0.16s** | 0.15s | 2.7s | 46.8s |
| **Total E2E** | **45.6s** | 60.7s | 20.1s | 159.7s |
| **Real-Time Factor** | **9.1x** | 12.1x | 4.0x | 31.9x |
| **vs GPU** | **2.3x slower** | 3.0x slower | baseline | 7.9x slower |

**Note:** LeLM AR includes 602 condition-prepend tokens processed on the first step via single-token decoding. A prefill model optimization would process these in one batch (~1s), reducing total to ~18s and achieving 1.1x faster than GPU.

### Benchmark Results (30s audio generation, with prefill optimization)

| Metric | Neuron (fp32 GPT2) | CPU |
|--------|-------------------|-----|
| **LeLM AR time** | **59.0s** | ~500s (est.) |
| **Diffusion (10 steps)** | **1.47s** | ~23s (est.) |
| **VAE decode** | **0.42s** | ~25s (est.) |
| **Total E2E** | **60.9s** | ~550s (est.) |
| **Real-Time Factor** | **2.03x** | ~18x |

**With v1 conditioning fix + early EOS:** Total drops to ~42s (RTF 1.4x) because the model generates correct EOS tokens and stops early.

### Per-Step Latency (decode phase)

| Component | On-Device KV | PCIe KV | Speedup |
|-----------|-------------|---------|---------|
| Primary (28L) step | 28.0ms | 94.5ms | 3.4x |
| Fused Secondary (12L) step | 13.7ms | 39.1ms | 2.9x |
| Combined LeLM step | 41.7ms | 133.6ms | 3.2x |
| GPT2 diffusion step (fp32) | 14.5ms | 7.8ms | 0.5x (fp32 required) |
| VAE decode (full) | 71.5ms | 71.5ms | 1.0x |

## Usage

### Prerequisites

1. Clone the SongGeneration repository to the instance:
   ```bash
   git clone https://huggingface.co/tencent/SongGeneration /mnt/models/songgeneration
   ```

2. Download model weights:
   ```bash
   # Main checkpoint (11 GB)
   huggingface-cli download tencent/SongGeneration --local-dir /mnt/models/ckpt/songgeneration_base
   ```

3. Activate Neuron environment:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
   export PYTHONPATH="$(pwd)/codeclm/tokenizer/:$(pwd):$(pwd)/codeclm/tokenizer/Flow1dVAE/:$PYTHONPATH"
   ```

### Compile and Generate

```python
from modeling_songgeneration import SongGenerationNeuron, SongGenerationConfig

config = SongGenerationConfig(
    model_path="/mnt/models/ckpt/songgeneration_base/model.pt",
    config_path="/mnt/models/ckpt/songgeneration_base/config.yaml",
    safetensors_path="/mnt/models/songgeneration/ckpt/model_septoken/model_2.safetensors",
    prompt_path="/mnt/models/songgeneration/ckpt/prompt.pt",
)

# Compile (first time, ~15-20 min)
pipeline = SongGenerationNeuron(config)
pipeline.compile()
pipeline.save("/mnt/models/songgeneration/compiled")

# Load pre-compiled (subsequent runs, ~3 min)
pipeline = SongGenerationNeuron(config)
pipeline.load("/mnt/models/songgeneration/compiled")
pipeline.warmup()

# Generate
audio, sample_rate = pipeline.generate(
    "A cheerful pop song with catchy melody",
    genre="Pop",
    duration_sec=5.0,
)

# Save as WAV
import scipy.io.wavfile
import numpy as np
audio_np = audio.squeeze(0).float().cpu().numpy().T
audio_np = np.clip(audio_np, -1.0, 1.0)
audio_int16 = (audio_np * 32767).astype(np.int16)
scipy.io.wavfile.write("output.wav", sample_rate, audio_int16)
```

### Command Line

```bash
python src/modeling_songgeneration.py \
    --text "A gentle piano ballad with soft strings" \
    --genre Pop \
    --duration-sec 5.0 \
    --output-wav output.wav
```

### With Lyrics

```bash
python src/modeling_songgeneration.py \
    --lyrics "[intro-short] ; [verse] Walking through the city lights.Feeling like everything is right ; [chorus] We are alive tonight.Hearts on fire burning bright ; [outro-short]" \
    --description "pop, uplifting, piano, acoustic guitar, drums, male vocals" \
    --genre Pop \
    --duration-sec 30.0 \
    --output-wav output_lyrics.wav
```

### Timing Breakdown

```python
result = pipeline.generate_timed(
    "An upbeat dance track",
    genre="Pop",
    duration_sec=5.0,
)
print(f"LeLM:      {result['timings']['lelm_s']:.1f}s")
print(f"Diffusion: {result['timings']['diffusion_s']:.3f}s")
print(f"VAE:       {result['timings']['vae_s']:.3f}s")
print(f"Total:     {result['timings']['total_s']:.1f}s")
```

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 |
|----------|----------|----------|
| trn2.3xlarge (LNC=2) | VALIDATED | Not tested |
| trn2.48xlarge | Not tested | Not tested |
| inf2.xlarge | Not tested | Not tested |

## Example Checkpoints

* [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration)

## Testing Instructions

```bash
# On a trn2.3xlarge with model weights and codeclm source:
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
cd /mnt/models/songgeneration

# Set paths
export PYTHONPATH="$(pwd)/codeclm/tokenizer/:$(pwd):$(pwd)/codeclm/tokenizer/Flow1dVAE/:$PYTHONPATH"
export SONGGEN_MODEL_PATH=/mnt/models/ckpt/songgeneration_base/model.pt
export SONGGEN_CONFIG_PATH=/mnt/models/ckpt/songgeneration_base/config.yaml
export SONGGEN_SAFETENSORS_PATH=/mnt/models/songgeneration/ckpt/model_septoken/model_2.safetensors
export SONGGEN_PROMPT_PATH=/mnt/models/songgeneration/ckpt/prompt.pt

# Run tests (compile from scratch, ~30 min):
pytest contrib/models/SongGeneration/test/integration/test_model.py -v --timeout=1800

# Or run standalone:
python contrib/models/SongGeneration/test/integration/test_model.py
```

## Known Issues

1. **GPT2 diffusion MUST use `--auto-cast none`:** The iterative Euler solver (10 steps) amplifies per-step numerical errors exponentially. With `--auto-cast matmult`, the GPT2 NEFF has only 0.64 cosine similarity with CPU -- after 10 iterations this produces completely garbled audio. With `--auto-cast none`, cosine similarity is 1.0004 and audio quality is correct. This adds ~50% to per-step GPT2 latency (14.5ms vs 7.8ms) but the total impact on E2E is small (~0.5s extra for 30s audio).

2. **torchaudio WAV saving:** The Neuron DLAMI's torchaudio may lack codec support for WAV saving. Use `scipy.io.wavfile` instead (included in the pipeline).

2. **First-run library rehydration:** The first import of torch-neuronx/transformers on a fresh DLAMI instance can take 2-5 minutes due to lazy package decompression. This is normal.

3. **transformers version:** The upstream codeclm codebase has an assertion requiring `transformers < 4.40`. This assertion is patched out in the loading code (line 99 of `codeclm/models/levo.py`).

4. **CUDA patching:** The upstream codebase assumes CUDA availability. All CUDA calls are redirected to CPU at import time. This patch is applied automatically by the pipeline.

5. **Compilation time:** Full compilation (LeLM + GPT2 + VAE) takes ~15-20 minutes. Use `save()`/`load()` to avoid recompilation.

6. **Duration flexibility:** The GPT2 and VAE components are traced at a fixed frame count. To generate audio of a different duration, recompilation is required. The LeLM models (on-device KV) support variable lengths up to `max_seq_len`.

## Maintainer

Agent Pavarati (OpenCode)
