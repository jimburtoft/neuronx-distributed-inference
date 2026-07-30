# Speculative decoding for Voxtral-Mini-3B

A 4-layer distilled draft model for speculative decoding with Voxtral-Mini-3B-2507 is available at:

**[huggingface.co/jburtoft/Voxtral-Mini-3B-2507-draft-4layer](https://huggingface.co/jburtoft/Voxtral-Mini-3B-2507-draft-4layer)**

Training methodology, dataset, reproduction scripts, and detailed measurements are on the HF model card.

## Quick summary

- Draft shares audio encoder, projector, tokenizer, and `lm_head` with the target. Only the LLM decoder is shrunk from 30 layers to 4 (kept: indices `[0, 10, 20, 29]`).
- Trained via distillation on public LibriSpeech pseudo-labels (~2000 clips, ~230 min audio).
- Achieves ~66.67% top-1 argmax agreement with the target on a held-out English audio benchmark; expected 1.3x-1.6x mean speedup in a correctly-implemented spec-decode wrapper.
- Draft is ~4.4x faster per generated token than the target on GPU L40S.

## Download

```bash
# Clone the model repo into a local drafts directory
huggingface-cli download jburtoft/Voxtral-Mini-3B-2507-draft-4layer \
  --local-dir /mnt/drafts/Voxtral-Mini-3B-2507-draft-4layer
```

Or in Python:

```python
from huggingface_hub import snapshot_download
draft_dir = snapshot_download(
    "jburtoft/Voxtral-Mini-3B-2507-draft-4layer",
    local_dir="/mnt/drafts/Voxtral-Mini-3B-2507-draft-4layer",
)
```

## Basic usage on GPU (HuggingFace `transformers`)

```python
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor

target = VoxtralForConditionalGeneration.from_pretrained(
    "mistralai/Voxtral-Mini-3B-2507",
    dtype=torch.bfloat16,
).cuda().eval()

draft = VoxtralForConditionalGeneration.from_pretrained(
    "jburtoft/Voxtral-Mini-3B-2507-draft-4layer",
    dtype=torch.bfloat16,
).cuda().eval()

processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")

inputs = processor.apply_transcription_request(
    language="en",
    audio="path/to/audio.wav",
    model_id="mistralai/Voxtral-Mini-3B-2507",
).to("cuda", dtype=torch.bfloat16)

# HF's built-in speculative decoding via assistant_model=
out = target.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    assistant_model=draft,
)
transcript = processor.tokenizer.decode(
    out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
)
print(transcript)
```

## Neuron / NxD Inference integration

Not yet included in this contrib. Planned as a follow-up: a `spec_decode=True` flag on `NeuronApplicationVoxtral` that traces bucketed target verify-K NEFFs plus a single-step draft NEFF, wired into a Neuron-native spec-decode loop. When this lands, it will live alongside the base `src/modeling_voxtral.py` and reuse the same downloaded draft.

For now, use the model on GPU per above. See the HF model card for details on why HF's `assistant_model=` path currently underperforms with `VoxtralForConditionalGeneration` (a known plumbing issue) and how to work around it with a manual measurement/decode loop.
