"""Task 003 preparation: extract text-only weights for both target and draft.

For each model we save `text_only/` with just the LLM decoder (VoxtralForConditionalGeneration.language_model),
plus tokenizer/processor files. That's the input to NxDI's NeuronApplicationVoxtral.

Only needed once per model. Idempotent (skips if text_only/ already exists).
"""
from __future__ import annotations
import argparse
import os
import shutil
import time
import torch
from huggingface_hub import snapshot_download
from transformers import VoxtralForConditionalGeneration


TARGET_MODEL = "mistralai/Voxtral-Mini-3B-2507"
DRAFT_MODEL = "jburtoft/Voxtral-Mini-3B-2507-draft-4layer"


def extract(model_path: str) -> str:
    text_only = os.path.join(model_path, "text_only")
    if os.path.exists(os.path.join(text_only, "config.json")):
        print(f"[extract] {model_path}: text_only/ already exists, skipping")
        return text_only
    print(f"[extract] Loading full model from {model_path}")
    t0 = time.time()
    full = VoxtralForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print(f"[extract] Loaded in {time.time()-t0:.1f}s")
    os.makedirs(text_only, exist_ok=True)
    full.language_model.save_pretrained(text_only)
    # Copy the tokenizer/processor files
    for fname in ["tekken.json", "params.json", "preprocessor_config.json", "generation_config.json", "processor_config.json"]:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, text_only)
    print(f"[extract] Saved text-only weights to {text_only}")
    return text_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["target", "draft", "both"], default="both")
    args = p.parse_args()

    if args.which in ("target", "both"):
        target_path = snapshot_download(TARGET_MODEL)
        extract(target_path)
        print(f"target text_only: {target_path}/text_only")

    if args.which in ("draft", "both"):
        draft_path = snapshot_download(DRAFT_MODEL)
        extract(draft_path)
        print(f"draft text_only: {draft_path}/text_only")


if __name__ == "__main__":
    raise SystemExit(main())
