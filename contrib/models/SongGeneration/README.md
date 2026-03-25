# Contrib Model: SongGeneration (LeVo)

Text-to-music generation on AWS Trainium2 using Tencent's SongGeneration (LeVo) model -- a hybrid LLM-Diffusion audio pipeline that generates stereo 48kHz music with vocals from lyrics and text descriptions.

## Model Information

- **HuggingFace ID:** `lglg666/SongGeneration-base-new` (English + Chinese)
- **Also compatible:** `tencent/SongGeneration` (Chinese-only base model)
- **Model Type:** Multi-stage audio generation pipeline (LLM + Diffusion + VAE)
- **Parameters:** ~4.1B total (LeLM 2.83B + Diffusion 1.1B + VAE 169M)
- **Architecture:** Dual-Llama AR LM (28L primary + 12L secondary) with delayed codebook pattern, GPT2-RoPE CFM diffusion backbone (16L), Stable Audio VAE decoder
- **Output:** Stereo 48kHz WAV audio
- **License:** Check [HuggingFace model card](https://huggingface.co/tencent/SongGeneration)

## Architecture

SongGeneration uses a three-stage pipeline:

| Stage | Component | Params | Neuron Compilation | Key Innovation |
|-------|-----------|--------|-------------------|----------------|
| 1. LeLM | Dual-Llama AR (28L + 12L) | 2.83B | `ModelBuilder` (on-device KV) | Prefill + `torch.scatter` KV cache in HBM |
| 2. Diffusion | GPT2-RoPE CFM (16L) | 1.1B | `torch_neuronx.trace()` | Rewritten RoPE (no complex numbers) |
| 3. VAE | Stable Audio decoder | 169M | `torch_neuronx.trace()` | `weight_norm` removal pre-trace |

### On-Device KV Cache

The LeLM transformers use on-device KV caching via `neuronx_distributed.ModelBuilder`. Instead of passing KV cache tensors as model inputs/outputs each autoregressive step (PCIe round-trip), the cache is stored as `register_buffer` on the model and updated in-place with `torch.scatter`. This keeps the cache in Neuron HBM, providing a 3.2x speedup on the LeLM stage.

### Prefill Optimization

The first 512 of ~602 condition-prepend tokens (text encoding + prompt audio + description) are processed in a single Neuron call via a dedicated "prefill" NEFF, rather than one-at-a-time through the decode NEFF. This reduces total LeLM time by ~45%.

### TP=2 NxDI Mode

An optional TP=2 mode uses `NxDParallelState` from `neuronx_distributed` with NxDI `NeuronLlamaDecoderLayer` wrappers to shard the LeLM transformers across 2 NeuronCores. This provides a further 2x speedup on the LeLM stage, bringing total generation time close to real-time.

### Neuron-Specific Adaptations

- **RoPE rewrite:** `torch.view_as_complex` / `torch.view_as_real` replaced with explicit sin/cos rotation (XLA compatible)
- **Flash Attention disabled:** `use_flash_attn_2=False` (CUDA-only feature)
- **CUDA-to-CPU patches:** All `torch.cuda` calls redirected to CPU (upstream codebase assumes CUDA)
- **weight_norm removal:** `torch.nn.utils.remove_weight_norm` applied to VAE before tracing
- **bf16 precision:** AR loop state kept in FP32 on CPU to avoid compound rounding errors; LeLM Neuron models use `--auto-cast matmult`
- **GPT2 diffusion fp32:** The GPT2 diffusion backbone **must** be traced with `--auto-cast none` (full FP32). Using `--auto-cast matmult` causes severe numerical degradation (cosine similarity drops from 1.0 to 0.64 vs CPU) which compounds across 10 Euler solver steps into garbled audio. The VAE can safely use `--auto-cast matmult`.
- **Language-aware prompts:** The `new_auto_prompt.pt` file provides per-language prompt audio tokens (`['Pop']['en']` for English, `['Pop']['zh']` for Chinese). Using the correct language prompt is essential for generating vocals in the target language.

## Validation Results

**Validated:** 2026-03-25
**Instance:** trn2.3xlarge (LNC=2, 4 NeuronCores)
**SDK:** Neuron SDK 2.28 (DLAMI 20260227), PyTorch 2.9

### Component Accuracy

| Component | Metric | Value | Threshold |
|-----------|--------|-------|-----------|
| GPT2 diffusion (fp32) | Cosine similarity vs CPU | 1.0004 | > 0.98 |
| GPT2 diffusion (fp32) | Max diff vs CPU | 0.0003 | < 0.01 |
| VAE decoder | Cosine similarity vs CPU | 1.0001 | > 0.98 |
| VAE decoder | SNR vs CPU | > 40 dB | > 20 dB |

### Benchmark Results (15s audio generation, English vocals)

Model: `SongGeneration-base-new` with English prompt audio and structured lyrics.

| Metric | Baseline (TP=1) | NxDI TP=2 | Speedup |
|--------|----------------|-----------|---------|
| **LeLM AR time** | **39.7s** | **19.4s** | **2.0x** |
| **Diffusion + VAE** | **0.6s** | **0.6s** | 1.0x |
| **Total E2E** | **40.3s** | **20.0s** | **2.0x** |
| **Real-Time Factor** | **2.69x** | **1.33x** | -- |
| **LeLM per-step** | **32.3ms** | **15.8ms** | **2.0x** |
| **AR steps** | 1227 | 1227 | -- |

Audio quality validated by human listening across 3 seeds per configuration. Both TP=1 and TP=2 produce clear English vocals with comparable quality.

### Benchmark Results (5s audio generation, Chinese vocals)

Model: `songgeneration_base` (Chinese-only) with Chinese prompt audio.

| Metric | Baseline (TP=1) | NxDI TP=2 |
|--------|----------------|-----------|
| **LeLM AR time** | **24.0s** | **12.1s** |
| **Total E2E** | **24.2s** | **12.3s** |
| **Real-Time Factor** | **4.84x** | **2.46x** |

### Per-Step Latency

| Component | TP=1 (baseline) | TP=2 (NxDI) | Speedup |
|-----------|-----------------|--------------|---------|
| Primary (28L) step | ~20ms | ~10ms | 2.0x |
| Fused Secondary (12L) step | ~12ms | ~6ms | 2.0x |
| Combined LeLM step | ~32ms | ~16ms | 2.0x |

## Usage

### Prerequisites

1. Clone the SongGeneration repository:
   ```bash
   git clone https://github.com/tencent-ailab/songgeneration.git /mnt/models/songgeneration/codeclm_repo
   cp -r /mnt/models/songgeneration/codeclm_repo/codeclm /mnt/models/songgeneration/
   ```

2. Download model weights (English-capable):
   ```bash
   pip install huggingface_hub
   python -c "
   from huggingface_hub import snapshot_download
   # English + Chinese model
   snapshot_download('lglg666/SongGeneration-base-new',
                     local_dir='/mnt/models/ckpt/songgeneration_base_new',
                     ignore_patterns=['*.md'])
   # Shared assets (diffusion, VAE, tokenizer, prompts)
   snapshot_download('tencent/SongGeneration',
                     local_dir='/mnt/models/ckpt/songgeneration_base',
                     ignore_patterns=['*.md'])
   "
   ```

3. Pull language-aware prompt audio (requires git-lfs):
   ```bash
   cd /mnt/models/songgeneration/codeclm_repo
   git lfs pull --include='tools/new_auto_prompt.pt'
   ```

4. Set up symlinks and paths:
   ```bash
   cd /mnt/models/songgeneration
   ln -sf /mnt/models/ckpt/songgeneration_base/third_party third_party
   ln -sf /mnt/models/ckpt/songgeneration_base/ckpt ckpt
   mkdir -p conf && cp codeclm_repo/conf/vocab.yaml conf/vocab.yaml
   ```

5. Activate Neuron environment and install dependencies:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
   pip install accelerate flashy alias-free-torch descript-audio-codec \
       k-diffusion vector-quantize-pytorch einops-exts x-transformers \
       diffusers==0.37.0 peft==0.18.0 lightning openunmix
   pip install protobuf==5.29.3  # Must be after descript-audio-codec
   export PYTHONPATH="$(pwd)/codeclm/tokenizer/:$(pwd):$(pwd)/codeclm/tokenizer/Flow1dVAE/:$PYTHONPATH"
   ```

6. Apply patches (required on each new instance):
   ```bash
   # SequenceSummary stub
   UTILS_FILE=$(python3 -c "import transformers.modeling_utils; print(transformers.modeling_utils.__file__)")
   echo '
   class SequenceSummary:
       pass' >> "$UTILS_FILE"

   # Flash attention import fix
   find codeclm/ -name "*.py" -exec sed -i "s/is_flash_attn_available/is_flash_attn_2_available/g" {} +

   # Remove transformers version assertion
   sed -i "/assert.*transformers.*version/d" codeclm/models/levo.py
   ```

### Compile and Generate (Baseline TP=1)

```python
from modeling_songgeneration import SongGenerationNeuron, SongGenerationConfig
import torch

config = SongGenerationConfig(
    model_path="/mnt/models/ckpt/songgeneration_base_new/model.pt",
    config_path="/mnt/models/ckpt/songgeneration_base_new/config.yaml",
    safetensors_path="/mnt/models/songgeneration/ckpt/model_septoken/model_2.safetensors",
    prompt_path="/mnt/models/songgeneration/ckpt/prompt.pt",  # placeholder, overridden below
    default_duration_sec=15.0,
)

pipeline = SongGenerationNeuron(config)
pipeline.compile()  # ~20 min first time

# Load English prompt audio
auto_prompt = torch.load(
    '/mnt/models/songgeneration/codeclm_repo/tools/new_auto_prompt.pt',
    map_location='cpu', weights_only=False
)
pipeline._prompt_data = {g: auto_prompt[g]['en'] for g in auto_prompt if 'en' in auto_prompt[g]}

pipeline.warmup()

# Generate with English lyrics
lyrics = (
    "[intro-short] ; "
    "[verse] Sunlight breaks through morning haze."
    "Golden fields stretch far away ; "
    "[chorus] Sing along.Let the music carry you home ; "
    "[outro-short]"
)
audio, sr = pipeline.generate(lyrics, genre="Pop", duration_sec=15.0)

# Save as WAV
import scipy.io.wavfile, numpy as np
audio_np = audio.squeeze(0).float().cpu().numpy()
if audio_np.ndim > 1:
    audio_np = audio_np.mean(axis=0)
peak = max(abs(audio_np.max()), abs(audio_np.min()), 1e-10)
audio_int16 = (audio_np / peak * 32767).astype(np.int16)
scipy.io.wavfile.write("output.wav", sr, audio_int16)
```

### Generate with NxDI TP=2 (2x faster)

```python
from modeling_songgeneration import SongGenerationNeuron, SongGenerationConfig
from hybrid_benchmark import compile_hybrid
import torch

config = SongGenerationConfig(
    model_path="/mnt/models/ckpt/songgeneration_base_new/model.pt",
    config_path="/mnt/models/ckpt/songgeneration_base_new/config.yaml",
    safetensors_path="/mnt/models/songgeneration/ckpt/model_septoken/model_2.safetensors",
    prompt_path="/mnt/models/songgeneration/ckpt/prompt.pt",
    default_duration_sec=15.0,
)

pipeline = SongGenerationNeuron(config)
compile_hybrid(pipeline, "full-nxdi")  # ~25 min, uses TP=2

# Load English prompts
auto_prompt = torch.load(
    '/mnt/models/songgeneration/codeclm_repo/tools/new_auto_prompt.pt',
    map_location='cpu', weights_only=False
)
pipeline._prompt_data = {g: auto_prompt[g]['en'] for g in auto_prompt if 'en' in auto_prompt[g]}

pipeline.warmup()

result = pipeline.generate_timed(lyrics, genre="Pop", duration_sec=15.0)
print(f"Total: {result['timings']['total_s']:.1f}s")  # ~20s for 15s audio
```

### Lyrics Format

The model expects structured lyrics with section tags separated by ` ; ` and lines separated by `.`:

```
[intro-short] ; [verse] First line of verse.Second line of verse ; [chorus] Chorus line one.Chorus line two ; [outro-short]
```

**Structure tags:** `[verse]`, `[chorus]`, `[bridge]`, `[intro-short/medium/long]`, `[outro-short/medium/long]`, `[inst-short/medium/long]`, `[silence]`

**Language:** The model generates vocals in the language of the lyrics. Use English lyrics for English vocals, Chinese for Chinese. The prompt audio language should match (use `new_auto_prompt.pt` with `['en']` or `['zh']` key).

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 |
|----------|----------|----------|
| trn2.3xlarge (LNC=2, TP=1) | VALIDATED | Not tested |
| trn2.3xlarge (LNC=2, TP=2) | VALIDATED | Not tested |
| trn2.48xlarge | Not tested | Not tested |

## Example Checkpoints

* [lglg666/SongGeneration-base-new](https://huggingface.co/lglg666/SongGeneration-base-new) (English + Chinese, recommended)
* [tencent/SongGeneration](https://huggingface.co/tencent/SongGeneration) (Chinese-only base + shared assets)

## Testing Instructions

```bash
# On a trn2.3xlarge with model weights and codeclm source:
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
cd /mnt/models/songgeneration

# Set paths
export PYTHONPATH="$(pwd)/codeclm/tokenizer/:$(pwd):$(pwd)/codeclm/tokenizer/Flow1dVAE/:$PYTHONPATH"
export SONGGEN_MODEL_PATH=/mnt/models/ckpt/songgeneration_base_new/model.pt
export SONGGEN_CONFIG_PATH=/mnt/models/ckpt/songgeneration_base_new/config.yaml
export SONGGEN_SAFETENSORS_PATH=/mnt/models/songgeneration/ckpt/model_septoken/model_2.safetensors
export SONGGEN_PROMPT_PATH=/mnt/models/songgeneration/codeclm_repo/tools/new_auto_prompt.pt

# Run tests (compile from scratch, ~30 min):
pytest contrib/models/SongGeneration/test/integration/test_model.py -v --timeout=3600

# Or run standalone:
python contrib/models/SongGeneration/test/integration/test_model.py
```

## Known Issues

1. **GPT2 diffusion MUST use `--auto-cast none`:** The iterative Euler solver (10 steps) amplifies per-step numerical errors exponentially. With `--auto-cast matmult`, the GPT2 NEFF has only 0.64 cosine similarity with CPU -- after 10 iterations this produces completely garbled audio. With `--auto-cast none`, cosine similarity is 1.0004 and audio quality is correct.

2. **Language-aware prompt audio is essential:** Using the old `prompt.pt` (Chinese-only) with English lyrics produces Chinese vocals regardless of lyric language. Always use `new_auto_prompt.pt` with the correct language key.

3. **Duration affects compilation:** The GPT2 and VAE components are traced at a fixed frame count (`T_frames = duration_sec * 25`). Changing duration requires recompilation. The LeLM models support variable lengths up to `max_seq_len`.

4. **torchaudio WAV saving:** The Neuron DLAMI's torchaudio may lack codec support for WAV saving. Use `scipy.io.wavfile` instead.

5. **First-run library rehydration:** The first import of torch-neuronx/transformers on a fresh DLAMI instance can take 2-5 minutes due to lazy package decompression.

6. **NxDI TP=2 requires LNC=2:** The `NxDParallelState(world_size=2)` compiles NEFFs with Logical Core Size 2. The instance must be configured with `NEURON_LOGICAL_NC_CONFIG=2` (the default on trn2.3xlarge).

7. **Compilation not cached across sessions:** ModelBuilder does not persist compiled NEFFs between Python sessions. Each run recompiles (~20 min for TP=1, ~25 min for TP=2). Use `save()`/`load()` for the baseline pipeline to avoid recompilation.

## Maintainer

Agent Pavarati (OpenCode)
