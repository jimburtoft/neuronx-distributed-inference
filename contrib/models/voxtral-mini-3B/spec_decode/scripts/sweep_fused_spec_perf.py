"""Performance sweep for fused-spec Voxtral: K values + async + ODS.

For each config variant: compile (if needed), then measure mean speedup +
byte-identical rate on the customer clips vs the target-only baseline.

Run with:
    PYTHONPATH=/home/ubuntu/nxdi_edit python sweep_fused_spec_perf.py
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import statistics
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

# Variants to test: (name, K, extra_flags)
VARIANTS = [
    ("K2", 2, {}),
    ("K3", 3, {}),
    ("K4", 4, {}),
    ("K5", 5, {}),
    ("K6", 6, {}),
    ("K4_async", 4, {"async_mode": True}),
    ("K4_ods", 4, {"is_hidden_dim_shuffled": True}),
]


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


def read_clips(manifest, num_clips):
    dataset_root = os.path.dirname(manifest)
    clips = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            ap = row["audio_path"]
            ap = ap if os.path.isabs(ap) else os.path.join(dataset_root, ap)
            clips.append({"clip": os.path.basename(ap), "audio_path": ap,
                          "duration_sec": float(row["duration_sec"]), "reference": row["transcript"]})
            if len(clips) >= num_clips:
                break
    return clips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-clips", type=int, default=18)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--manifest", default="/home/ubuntu/voxtral_spec_decode/dataset/manifest.csv")
    p.add_argument("--target-dir", default="/mnt/models/nxdi_compiled/target_tp4_np768_sl768/text_decoder")
    p.add_argument("--audio-encoder", default="/mnt/models/nxdi_compiled/target_tp4_np768_sl768/audio_encoder.pt")
    p.add_argument("--variants", nargs="+", default=[v[0] for v in VARIANTS])
    p.add_argument("--output", default="/home/ubuntu/voxtral_spec_decode/tmp/sweep_fused_perf.json")
    args = p.parse_args()

    dtype = torch.bfloat16
    target_text = os.path.join(snapshot_download(TARGET), "text_only")
    draft_text = os.path.join(snapshot_download(DRAFT), "text_only")

    from compile_voxtral_fused_spec import build_config
    from modeling_voxtral import (
        VoxtralForCausalLM, VoxtralInferenceConfig,
        generate_positions_from_mask, pad_positions, pad_vision_embeddings,
    )
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    tok = MistralCommonTokenizer.from_pretrained(target_text)
    processor = AutoProcessor.from_pretrained(TARGET)
    audio_token_id = json.load(open(os.path.join(target_text, "config.json"))).get("audio_token_id", 24)

    audio_encoder = load_audio_encoder(args.audio_encoder)
    full_model = VoxtralForConditionalGeneration.from_pretrained(TARGET, dtype=dtype)
    projector = full_model.multi_modal_projector.eval()
    del full_model; gc.collect()

    clips = read_clips(args.manifest, args.num_clips)

    def transcribe(adapter, clip, K=None):
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
        gk = dict(input_ids=input_ids, attention_mask=attention_mask,
                  max_new_tokens=args.max_new_tokens, do_sample=False,
                  vision_embeddings=audio_embeds_padded, vision_mask=vision_mask_padded)
        if K is not None:
            gk["prompt_lookup_num_tokens"] = K
        t0 = time.perf_counter()
        out = adapter.generate(**gk)
        wall = time.perf_counter() - t0
        gen = out[0, input_ids.shape[1]:]
        return tok.decode(gen.tolist(), skip_special_tokens=True).strip(), wall, int(gen.shape[0])

    # --- Baseline (target-only)
    print("[sweep] Baseline target-only")
    with open(os.path.join(target_text, "config.json")) as f:
        full_cfg = json.load(f)
    VoxtralForCausalLM._seq_len = 768; VoxtralForCausalLM._n_positions = 768
    VoxtralForCausalLM._text_hidden_size = full_cfg["hidden_size"]; VoxtralForCausalLM._batch_size = 1
    base_cfg = VoxtralInferenceConfig(text_model_path=target_text, full_config=full_cfg,
        tp_degree=4, batch_size=1, seq_len=768, n_positions=768, dtype=dtype, on_device_sampling=True).build()
    base_model = VoxtralForCausalLM(target_text, base_cfg)
    base_model.load(args.target_dir)
    base_adapter = HuggingFaceGenerationAdapter(base_model)
    _ = transcribe(base_adapter, clips[0])
    base = {}
    for clip in clips:
        text, wall, ntok = transcribe(base_adapter, clip)
        base[clip["clip"]] = {"text": text, "wall": wall, "ntok": ntok}
    del base_model, base_adapter; gc.collect()
    print(f"[sweep] Baseline mean wall: {statistics.mean(v['wall'] for v in base.values())*1000:.0f}ms")

    variant_map = {v[0]: v for v in VARIANTS}
    results = {}
    for vname in args.variants:
        _, K, flags = variant_map[vname]
        compiled_dir = f"/mnt/models/nxdi_compiled/vox_fs_{vname}"
        print(f"\n[sweep] === {vname} (K={K}, flags={flags}) ===")

        # Compile if needed
        if not os.path.exists(os.path.join(compiled_dir, "text_model", "model.pt")):
            print(f"[sweep] Compiling {vname}")
            os.makedirs(compiled_dir, exist_ok=True)
            cfg = build_config(target_text, draft_text, K=K, extra_flags=flags)
            m = VoxtralForCausalLM(target_text, cfg)
            t0 = time.time()
            try:
                m.compile(compiled_dir)
                print(f"[sweep]   compiled in {time.time()-t0:.0f}s")
            except Exception as e:
                print(f"[sweep]   COMPILE FAILED: {type(e).__name__}: {str(e)[:200]}")
                results[vname] = {"error": f"compile: {str(e)[:150]}"}
                del m; gc.collect()
                continue
            del m; gc.collect()

        # Load + measure
        try:
            cfg = build_config(target_text, draft_text, K=K, extra_flags=flags)
            m = VoxtralForCausalLM(target_text, cfg)
            m.load(compiled_dir)
            adapter = HuggingFaceGenerationAdapter(m)
            _ = transcribe(adapter, clips[0], K=K)
            speedups, long_speedups, n_id = [], [], 0
            for clip in clips:
                text, wall, ntok = transcribe(adapter, clip, K=K)
                b = base[clip["clip"]]
                sp = b["wall"] / wall if wall > 0 else 0
                speedups.append(sp)
                if b["ntok"] >= 40:
                    long_speedups.append(sp)
                n_id += (text == b["text"])
            results[vname] = {
                "K": K, "flags": flags,
                "mean_speedup": statistics.mean(speedups),
                "median_speedup": statistics.median(speedups),
                "long_speedup": statistics.mean(long_speedups) if long_speedups else None,
                "max_speedup": max(speedups),
                "byte_identical": f"{n_id}/{len(clips)}",
            }
            r = results[vname]
            print(f"[sweep]   mean={r['mean_speedup']:.3f}x median={r['median_speedup']:.3f}x "
                  f"long={r['long_speedup']:.3f}x max={r['max_speedup']:.3f}x byte-id={r['byte_identical']}")
            del m, adapter; gc.collect()
        except Exception as e:
            print(f"[sweep]   MEASURE FAILED: {type(e).__name__}: {str(e)[:200]}")
            results[vname] = {"error": f"measure: {str(e)[:150]}"}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[sweep] === SUMMARY ===")
    print(f"{'variant':<12} {'mean':>8} {'median':>8} {'long':>8} {'max':>8} {'byte-id':>8}")
    for vname, r in results.items():
        if "error" in r:
            print(f"{vname:<12} ERROR: {r['error'][:50]}")
        else:
            ls = f"{r['long_speedup']:.3f}" if r['long_speedup'] else "n/a"
            print(f"{vname:<12} {r['mean_speedup']:>7.3f}x {r['median_speedup']:>7.3f}x {ls:>8} {r['max_speedup']:>7.3f}x {r['byte_identical']:>8}")
    print(f"[sweep] Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
