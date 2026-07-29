"""Task 005: Neuron correctness verification.

Runs greedy target-only vs greedy spec-decode on 50 clips from LibriSpeech test-clean
plus 10 clips from LibriSpeech test-other. Verifies byte-identical output.

Uses the orchestrator from Task 004 (POC v1, full-prefill sync). Correctness is
independent of the speed profile.
"""

import argparse
import json
import time

import numpy as np
import torch
from datasets import load_dataset

import whisper as openai_whisper
from whisper.tokenizer import get_tokenizer

# Reuse orchestrator functions
import sys
sys.path.insert(0, "/home/ubuntu/whisper_spec_decode")
from spec_decode_orchestrator import (
    load_model, greedy_target_only, spec_decode_greedy, audio_to_mel,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-traced", default="/home/ubuntu/traced/whisper-large-v3-K4")
    ap.add_argument("--draft-traced", default="/home/ubuntu/traced/distil-large-v3-K4")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-clean", type=int, default=50)
    ap.add_argument("--n-other", type=int, default=10)
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--out-file", default="/home/ubuntu/task005_results.json")
    args = ap.parse_args()

    target = load_model("/home/ubuntu/models/whisper-large-v3", args.target_traced, args.K, "TARGET")
    draft = load_model("/home/ubuntu/models/distil-large-v3", args.draft_traced, args.K, "DRAFT")

    tokenizer = get_tokenizer(multilingual=True, task="transcribe", language="en", num_languages=100)
    sot, en, transcribe, notimestamps = tokenizer.sot, tokenizer.to_language_token("en"), tokenizer.transcribe, tokenizer.no_timestamps
    initial_tokens = (sot, en, transcribe, notimestamps)
    eot = tokenizer.eot
    special_tokens = {sot, en, transcribe, notimestamps, eot}
    print(f"initial_tokens={list(initial_tokens)}, eot={eot}", flush=True)

    # Load clips
    all_clips = []
    if args.n_clean > 0:
        print(f"Loading {args.n_clean} test-clean clips (min {args.min_dur}s)...", flush=True)
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
        for ex in ds:
            dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
            if dur < args.min_dur:
                continue
            all_clips.append({"ex": ex, "split": "test-clean", "dur": dur})
            if sum(1 for c in all_clips if c["split"] == "test-clean") >= args.n_clean:
                break
    if args.n_other > 0:
        print(f"Loading {args.n_other} test-other clips...", flush=True)
        ds = load_dataset("openslr/librispeech_asr", "other", split="test", streaming=True)
        for ex in ds:
            dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
            if dur < args.min_dur:
                continue
            all_clips.append({"ex": ex, "split": "test-other", "dur": dur})
            if sum(1 for c in all_clips if c["split"] == "test-other") >= args.n_other:
                break

    print(f"total {len(all_clips)} clips loaded", flush=True)

    results = []
    n_bi = 0
    n_bi_by_split = {"test-clean": 0, "test-other": 0}
    counts = {"test-clean": 0, "test-other": 0}
    n_zero_acc = 0
    t_start = time.perf_counter()
    for i, c in enumerate(all_clips):
        ex = c["ex"]
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        counts[c["split"]] += 1
        mel = audio_to_mel(audio).to(torch.float16)
        enc = target.encoder(mel)

        # Greedy target-only
        g_tokens, g_time, g_steps = greedy_target_only(target, enc, initial_tokens, eot=eot)
        # Spec-decode
        s_tokens, s_time, s_stats = spec_decode_greedy(target, draft, enc, initial_tokens, K=args.K, eot=eot)

        bi = (g_tokens == s_tokens)
        n_bi += int(bi)
        n_bi_by_split[c["split"]] += int(bi)
        acc = s_stats["accepted"] / s_stats["proposals"] if s_stats["proposals"] > 0 else 0
        if s_stats["proposals"] == 0:
            n_zero_acc += 1

        g_text = tokenizer.decode([t for t in g_tokens if t not in special_tokens])
        s_text = tokenizer.decode([t for t in s_tokens if t not in special_tokens])

        results.append({
            "id": ex["id"], "split": c["split"], "duration": c["dur"],
            "reference": ex["text"],
            "greedy_text": g_text, "spec_text": s_text,
            "greedy_tokens_n": len(g_tokens), "spec_tokens_n": len(s_tokens),
            "greedy_ms": g_time * 1000, "spec_ms": s_time * 1000,
            "greedy_steps": g_steps, "spec_iters": s_stats["iters"],
            "acceptance_rate": acc,
            "byte_identical": bi,
            "tokens_diff_at": None if bi else next(
                (k for k in range(min(len(g_tokens), len(s_tokens))) if g_tokens[k] != s_tokens[k]),
                min(len(g_tokens), len(s_tokens))
            ),
        })

        elapsed = time.perf_counter() - t_start
        print(f"[{i+1:3d}/{len(all_clips)}] {c['split']} {ex['id']} dur={c['dur']:.1f}s "
              f"bi={bi} acc={acc:.3f} g={g_time*1000:.0f}ms s={s_time*1000:.0f}ms "
              f"(elapsed {elapsed:.0f}s)", flush=True)
        if not bi:
            diff_at = results[-1]["tokens_diff_at"]
            print(f"    DIFFER at token {diff_at}", flush=True)
            print(f"    greedy: {g_text!r}", flush=True)
            print(f"    spec:   {s_text!r}", flush=True)

    # Aggregate
    summary = {
        "config": {"K": args.K, "target": "whisper-large-v3", "draft": "distil-large-v3"},
        "n_clips": len(all_clips),
        "n_test_clean": counts["test-clean"],
        "n_test_other": counts["test-other"],
        "byte_identical_all": n_bi,
        "byte_identical_test_clean": n_bi_by_split["test-clean"],
        "byte_identical_test_other": n_bi_by_split["test-other"],
        "n_zero_acc": n_zero_acc,
    }
    print(f"\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    with open(args.out_file, "w") as f:
        json.dump({"summary": summary, "per_clip": results}, f, indent=2)
    print(f"Wrote {args.out_file}", flush=True)

    if n_bi != len(all_clips):
        print(f"\n*** CORRECTNESS GATE FAILURE: {len(all_clips) - n_bi} / {len(all_clips)} clips differ ***", flush=True)


if __name__ == "__main__":
    main()
