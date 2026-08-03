"""Correctness + speedup test for fused-spec Voxtral.

Compares the fused-spec model's transcription output against the baseline
(target-only) Voxtral on real customer audio clips.

Uses:
  - Audio encoder: reuse the traced audio_encoder.pt from the baseline compile.
  - Projector: CPU.
  - Baseline vl_model: the existing target_tp4 compiled dir.
  - Fused-spec vl_model: /mnt/models/nxdi_compiled/voxtral_fused_spec_K3.

Run with:
    PYTHONPATH=/home/ubuntu/nxdi_edit python test_fused_spec_correctness.py
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

CONTRIB_SRC = "/home/ubuntu/nxdi-fork/contrib/models/voxtral-mini-3B/src"
if CONTRIB_SRC not in sys.path:
    sys.path.insert(0, CONTRIB_SRC)

import torch
from huggingface_hub import snapshot_download
from transformers.tokenization_mistral_common import MistralCommonTokenizer

TARGET = "mistralai/Voxtral-Mini-3B-2507"
DRAFT = "jburtoft/Voxtral-Mini-3B-2507-draft-4layer"
AUDIO_INTERMEDIATE_SIZE = 5120


def load_audio_encoder(path):
    import torch_neuronx
    enc = torch.jit.load(path)
    torch_neuronx.async_load(enc)
    return enc


def run_audio_pipeline(audio_encoder, projector, input_features, dtype):
    with torch.no_grad():
        enc_output = audio_encoder(input_features.to(dtype))
    if isinstance(enc_output, dict):
        audio_hidden = enc_output.get("last_hidden_state", list(enc_output.values())[0])
    elif isinstance(enc_output, tuple):
        audio_hidden = enc_output[0]
    else:
        audio_hidden = enc_output
    audio_hidden_flat = audio_hidden.reshape(-1, AUDIO_INTERMEDIATE_SIZE)
    with torch.no_grad():
        audio_embeds = projector(audio_hidden_flat)
    return audio_embeds


def build_fused_spec_vl_model(target_text, draft_text, K, seq_len=768, n_pos=768, tp=4, dtype=torch.bfloat16):
    from compile_voxtral_fused_spec import build_config
    from modeling_voxtral import VoxtralForCausalLM
    cfg = build_config(target_text, draft_text, K=K, tp=tp, seq_len=seq_len, n_pos=n_pos, dtype=dtype)
    model = VoxtralForCausalLM(target_text, cfg)
    return model


def build_target_only_vl_model(target_text, seq_len=768, n_pos=768, tp=4, dtype=torch.bfloat16):
    from modeling_voxtral import VoxtralInferenceConfig, VoxtralForCausalLM
    with open(os.path.join(target_text, "config.json")) as f:
        full_cfg = json.load(f)
    VoxtralForCausalLM._seq_len = seq_len
    VoxtralForCausalLM._n_positions = n_pos
    VoxtralForCausalLM._text_hidden_size = full_cfg["hidden_size"]
    VoxtralForCausalLM._batch_size = 1
    cfg = VoxtralInferenceConfig(
        text_model_path=target_text, full_config=full_cfg,
        tp_degree=tp, batch_size=1, seq_len=seq_len, n_positions=n_pos,
        dtype=dtype, on_device_sampling=True,
    ).build()
    return VoxtralForCausalLM(target_text, cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--num-clips", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--manifest", default="/home/ubuntu/voxtral_spec_decode/dataset/manifest.csv")
    p.add_argument("--fused-dir", default="/mnt/models/nxdi_compiled/voxtral_fused_spec_K3")
    p.add_argument("--target-dir", default="/mnt/models/nxdi_compiled/target_tp4_np768_sl768/text_decoder")
    p.add_argument("--audio-encoder", default="/mnt/models/nxdi_compiled/target_tp4_np768_sl768/audio_encoder.pt")
    args = p.parse_args()

    dtype = torch.bfloat16
    target_text = os.path.join(snapshot_download(TARGET), "text_only")
    draft_text = os.path.join(snapshot_download(DRAFT), "text_only")

    from modeling_voxtral import (
        generate_positions_from_mask, pad_positions, pad_vision_embeddings,
    )
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    tok = MistralCommonTokenizer.from_pretrained(target_text)
    processor = AutoProcessor.from_pretrained(TARGET)
    audio_token_id = json.load(open(os.path.join(target_text, "config.json"))).get("audio_token_id", 24)

    # Audio encoder + projector
    print("[test] Loading audio encoder + projector")
    audio_encoder = load_audio_encoder(args.audio_encoder)
    full_model = VoxtralForConditionalGeneration.from_pretrained(TARGET, dtype=dtype)
    projector = full_model.multi_modal_projector.eval()
    del full_model
    import gc; gc.collect()

    # Read clips
    dataset_root = os.path.dirname(args.manifest)
    clips = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            ap = row["audio_path"]
            ap = ap if os.path.isabs(ap) else os.path.join(dataset_root, ap)
            clips.append({"clip": os.path.basename(ap), "audio_path": ap,
                          "duration_sec": float(row["duration_sec"]), "reference": row["transcript"]})
            if len(clips) >= args.num_clips:
                break

    def transcribe(adapter, clip, use_fused):
        inputs = processor.apply_transcription_request(language="en", audio=clip["audio_path"], model_id=TARGET)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        input_features = inputs["input_features"].to(dtype)
        audio_embeds = run_audio_pipeline(audio_encoder, projector, input_features, dtype)
        audio_embeds_3d = audio_embeds.unsqueeze(0).to(dtype)
        audio_positions = (input_ids == audio_token_id).squeeze(0)
        vision_mask = generate_positions_from_mask(audio_positions)
        vision_mask_padded = pad_positions(vision_mask, 768, 767)
        audio_embeds_padded = pad_vision_embeddings(audio_embeds_3d, 768)
        gen_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                          max_new_tokens=args.max_new_tokens, do_sample=False,
                          vision_embeddings=audio_embeds_padded, vision_mask=vision_mask_padded)
        if use_fused:
            gen_kwargs["prompt_lookup_num_tokens"] = args.K
        t0 = time.perf_counter()
        out = adapter.generate(**gen_kwargs)
        wall = time.perf_counter() - t0
        gen = out[0, input_ids.shape[1]:]
        text = tok.decode(gen.tolist(), skip_special_tokens=True).strip()
        return text, wall, int(gen.shape[0])

    # --- Baseline (target-only)
    print("\n[test] Building + loading target-only baseline")
    base_model = build_target_only_vl_model(target_text)
    base_model.load(args.target_dir)
    base_adapter = HuggingFaceGenerationAdapter(base_model)
    # warmup
    _ = transcribe(base_adapter, clips[0], use_fused=False)
    base_results = {}
    for clip in clips:
        text, wall, ntok = transcribe(base_adapter, clip, use_fused=False)
        base_results[clip["clip"]] = {"text": text, "wall": wall, "ntok": ntok}
        print(f"[test]   baseline {clip['clip']}: {wall*1000:.0f}ms {ntok}tok :: {text[:80]!r}")
    del base_model, base_adapter
    gc.collect()

    # --- Fused-spec
    print("\n[test] Building + loading fused-spec")
    fs_model = build_fused_spec_vl_model(target_text, draft_text, K=args.K)
    fs_model.load(args.fused_dir)
    fs_adapter = HuggingFaceGenerationAdapter(fs_model)
    _ = transcribe(fs_adapter, clips[0], use_fused=True)
    print()
    n_identical = 0
    for clip in clips:
        text, wall, ntok = transcribe(fs_adapter, clip, use_fused=True)
        base = base_results[clip["clip"]]
        identical = text == base["text"]
        n_identical += identical
        speedup = base["wall"] / wall if wall > 0 else 0
        print(f"[test]   fused {clip['clip']}: {wall*1000:.0f}ms {ntok}tok speedup={speedup:.2f}x byte-id={identical}")
        if not identical:
            print(f"[test]     baseline: {base['text'][:100]!r}")
            print(f"[test]     fused:    {text[:100]!r}")

    print(f"\n[test] === byte-identical: {n_identical}/{len(clips)} ===")


if __name__ == "__main__":
    raise SystemExit(main())
