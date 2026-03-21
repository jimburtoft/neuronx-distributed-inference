# Contrib Model: Ministral-3-14B-Instruct-2512 (Leanstral)

NeuronX Distributed Inference implementation of the Ministral-3-14B vision-language model
on AWS Trainium 2. This model combines a Pixtral vision encoder with a Llama-compatible
text decoder and a Mistral3-specific PatchMerger projector.

## Model Information

- **HuggingFace ID:** `mistralai/Ministral-3-14B-Instruct-2512`
- **Model Type:** Vision-Language (Pixtral ViT + Llama text decoder)
- **Architecture:** `Mistral3ForConditionalGeneration`
- **Parameters:** 14B (dense GQA, 40 layers, hidden=5120, 32 Q / 8 KV heads)
- **License:** Check HuggingFace model card (gated access)

## Architecture Details

- Text decoder: 40 layers, hidden\_size=5120, num\_attention\_heads=32, num\_kv\_heads=8,
  vocab\_size=131072, intermediate\_size=16384, head\_dim=128, rope\_theta=1e9
- Vision encoder: Pixtral ViT (patch\_size=16, hidden=1024, 24 layers, 16 heads)
- Projector: Mistral3 PatchMerger (spatial 2x2 merge via F.unfold) + 2-layer MLP (runs on CPU)
- At TP=4: q\_heads\_per\_rank=8, kv\_heads\_per\_rank=2

### Key Adaptations

This model requires three runtime patches not yet available in upstream NxDI:

1. **SHARD\_OVER\_HEADS GQA**: With 8 KV heads at TP=4, each rank gets kv\_heads\_per\_rank=2
   instead of stock NxDI's replication to 8 (CONVERT\_TO\_MHA behavior). This avoids 4x
   KV cache inflation.

2. **Multi-KV-head TKG kernel**: A modified nki-library `attention_block_tkg` kernel that
   supports kv\_heads\_per\_rank > 1 via a virtual-batch approach. The stock bundled kernel
   hardcodes kv\_heads=1.

3. **CPU projector**: The Mistral3 PatchMerger uses spatial merging not present in NxDI's
   Pixtral projector, so it runs on CPU.

All patches are applied automatically when you call `get_model_cls()`.

## Prerequisites

- **SDK 2.28** (neuronx-cc >= 2.23, neuronx-distributed-inference >= 0.8)
- **trn2.3xlarge** (TP=4, LNC=2, 96 GB HBM)
- **Environment variable**: `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` must be set
- **Model checkpoint**: `mistralai/Ministral-3-14B-Instruct-2512` from HuggingFace (gated)
- **Disk**: ~300 GB EBS for checkpoint + compiled model artifacts

### Environment Setup

```bash
# Activate pre-installed PyTorch inference environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Required environment variable
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_COMPILE_CACHE_URL=""
```

## Usage

```python
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import build_inference_config, get_model_cls

MODEL_PATH = "/mnt/models/Ministral-3-14B-Instruct-2512"
COMPILED_PATH = "/mnt/models/compiled_leanstral"

# Build config
config = build_inference_config(
    model_path=MODEL_PATH,
    tp_degree=4,
    batch_size=1,
    seq_len=2048,
    n_positions=4096,
    vision_seq_len=4096,
    enable_tkg_kernel=True,
)

# Get model class (applies patches automatically)
ModelCls = get_model_cls()

# Instantiate, compile, and load
model = ModelCls(MODEL_PATH, config)
model.compile(COMPILED_PATH)
model.load(COMPILED_PATH)

# Enable vision encoder for VL inference
model.enable_vision_encoder()

# --- Text-only generation ---
import torch
from tokenizers import Tokenizer

tok = Tokenizer.from_file(f"{MODEL_PATH}/tokenizer.json")
encoded = tok.encode("The theory of general relativity")
input_ids = torch.tensor([encoded.ids], dtype=torch.long)
prompt_len = input_ids.shape[1]

# Prefill
out = model(
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    position_ids=torch.arange(prompt_len, dtype=torch.int32).unsqueeze(0),
    seq_ids=torch.zeros(1, dtype=torch.int32),
    sampling_params=torch.zeros(1, 3, dtype=torch.float32),
)

# Greedy decode loop
next_token = out.logits[:, -1, :].argmax(dim=-1).item()
for step in range(50):
    total_len = prompt_len + step + 1
    out = model(
        input_ids=torch.tensor([[next_token]], dtype=torch.long),
        attention_mask=torch.ones(1, total_len, dtype=torch.int32),
        position_ids=torch.tensor([[total_len - 1]], dtype=torch.int32),
        seq_ids=torch.zeros(1, dtype=torch.int32),
        sampling_params=torch.zeros(1, 3, dtype=torch.float32),
    )
    next_token = out.logits[:, -1, :].argmax(dim=-1).item()
```

### Vision-Language Inference

```python
from PIL import Image
from torchvision import transforms

# Load and preprocess image
image = Image.open("image.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16)
image_sizes = torch.tensor([[1024, 1024]], dtype=torch.int32)

# Build input with [IMG] tokens (token_id=10)
# The number of [IMG] tokens must match: (H/patch_size/2) * (W/patch_size/2)
# For 1024x1024 with patch=16, merge=2: (64/2)*(64/2) = 1024 tokens
num_img_tokens = 1024
img_tokens = torch.full((1, num_img_tokens), 10, dtype=torch.long)
text_tokens = torch.tensor([encoded.ids], dtype=torch.long)
input_ids = torch.cat([img_tokens, text_tokens], dim=1)

# VL prefill
out = model(
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    position_ids=torch.arange(input_ids.shape[1], dtype=torch.int32).unsqueeze(0),
    seq_ids=torch.zeros(1, dtype=torch.int32),
    sampling_params=torch.zeros(1, 3, dtype=torch.float32),
    pixel_values=pixel_values,
    image_sizes=image_sizes,
)
```

## Performance Results

Measured on trn2.3xlarge (TP=4, LNC=2, SDK 2.28):

| Mode | Throughput | Notes |
|------|-----------|-------|
| Text-only (TKG kernel) | 68.8 tok/s | grid=1 workaround |
| Vision-Language (TKG kernel) | 72.9 tok/s | With Pixtral ViT on Neuron |
| Text-only (no TKG, baseline) | 71.7 tok/s | Stock NxDI attention |

The ~4% text-only gap vs baseline is due to the `grid=1` workaround for compiler issue
NCC\_IXLV002 (LNC barrier mismatch with B\_virt > 1). This will resolve when the compiler
issue is fixed.

### Accuracy

- **Kernel correctness**: 100% token match over 20 decode steps in forced-token test
  (Neuron TKG kernel vs Neuron baseline)
- **FP8 dequantization**: Checkpoint weights are FP8 E4M3; dequantized to bf16 during
  state\_dict conversion with no accuracy loss

## Known Limitations

1. **grid=1 TKG kernel**: The multi-KV-head TKG kernel runs with `grid=1` to work around
   compiler issue NCC\_IXLV002 (LNC barrier mismatch when `B_virt > 1`). This costs ~4%
   throughput on text-only workloads.

2. **KVDP not supported**: KV data parallelism is not compatible with the multi-KV-head
   kernel path.

3. **Batch size 1 only**: VL pipeline is validated with batch\_size=1.

4. **FP8 checkpoint**: The original checkpoint uses FP8 E4M3 weights. These are dequantized
   to bf16 during state\_dict conversion. Runtime FP8 inference is not currently supported.

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 | Earlier |
|----------|----------|----------|---------|
| trn2.3xlarge (TP=4) | Tested | Not tested | Not supported |
| trn2.48xlarge | Not tested | Not tested | Not tested |
| trn1 / inf2 | Not supported | Not supported | Not supported |

## Source Files

| File | Lines | Description |
|------|-------|-------------|
| `src/modeling_leanstral.py` | ~960 | Model classes, config builder, state\_dict conversion, patches |
| `src/patch_native_multi_kv.py` | ~435 | NxDI TKG kernel adapter for multi-KV-head dispatch |
| `src/attention_block_tkg_multi_kv.py` | ~1927 | Modified nki-library outer kernel (virtual-batch approach) |

## Testing

Run integration tests on a trn2.3xlarge instance:

```bash
# Ensure model is downloaded and compiled first
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_COMPILE_CACHE_URL=""
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

pytest test/integration/test_model.py -v --capture=tee-sys
```

## Example Checkpoints

* [mistralai/Ministral-3-14B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) (gated, requires HuggingFace access)

## Upstream NxDI Gaps

This contrib model identifies 3 NxDI gaps that would benefit from upstream support:

1. **SHARD\_OVER\_HEADS GQA strategy** -- when `kv_heads >= tp_degree`, shard KV heads
   across ranks instead of replicating. See fork branch `feature/shard-over-heads-gqa`.
2. **Multi-KV-head TKG kernel** -- the bundled kernel hardcodes kv\_heads=1. The nki-library
   kernel fork adds `n_kv_heads` parameter with virtual-batch dispatch.
3. **Mistral3ForConditionalGeneration** config -- not registered in HuggingFace AutoConfig,
   requiring manual `load_config` callback.

## Maintainer

Leanstral Project

**Last Updated:** 2026-03-20
