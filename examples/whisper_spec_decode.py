"""Task 004: NxDI Whisper spec-decode orchestrator (POC v1, correctness-first).

Implements greedy speculative decoding on top of two NeuronApplicationWhisper models:
  - target = whisper-large-v3
  - draft  = distil-whisper/distil-large-v3

Design constraints observed in Task 003:
  1. The DecoderDecode NEFF advances the KV cache by ONE position per call. It returns
     logits at only the LAST position.
  2. The DecoderPrefillShort NEFF at whisper_prompt_len=K+1=5 writes K+1 positions at once
     but is currently only dispatched during initial prefill (see NeuronInference.logits in
     decoding.py:38 -- "if tokens.shape[-1] > self.initial_token_length: ... single-token").
  3. On partial rejection of proposals, draft's KV cache has stale entries at positions
     num_accepted..K-1 that must be rewritten before draft can propose the next batch.
     There is no NxDI-supported way to write a targeted slot in the KV cache without
     re-running that position's compute.

Given these constraints, this POC v1 uses the **safest correctness pattern**: at the end
of each iter, do a fresh full-length prefill on both models with the current committed
tokens list. This rewrites both KV caches from scratch, guaranteeing correctness.

Trade-off: expensive per-iter (adds 1 target full-prefill + 1 draft full-prefill), but
lets us verify byte-identical correctness against greedy target-only (Task 005 gate).

Task 006 benchmarks will show whether the algorithm's correctness ceiling is achievable
without the full-prefill overhead. If yes (via a smarter KV cache sync), we recommend
the fix in Task 008.

Algorithm (greedy):
  Init: initial_tokens = [SOT, lang, task, notimestamps] (4 tokens for Whisper)
  Loop until EOT or max_new_tokens:
    (a) Full prefill both models with `tokens`. This sets target_next_opinion.
    (b) Draft proposes K tokens greedily (K draft.decoder calls, one per proposal).
    (c) Sequentially verify with target:
        For k in 0..K-1:
          if target_next_opinion == proposals[k]:
            accept; call target.logits(tokens + proposals[:k+1]) to advance target's
              state; update target_next_opinion from last logits.
          else:
            reject; commit proposals[:num_accepted] + [target_next_opinion]; break.
        If all accepted: commit proposals + [target_next_opinion] as bonus.
    (d) On EOT in committed tokens, terminate.
    (e) Loop back to (a) -- re-prefill both models.

Correctness argument: after step (a), target's next-opinion is grounded in truth (target
has processed the current committed sequence, no stale KV). Draft's cache too. The K
draft proposals in (b) are made against a fully-synced draft cache. The K accept checks
in (c) use target-computed opinions at the correct positions.
"""

import argparse
import json
import time

import numpy as np
import torch
from datasets import load_dataset

import whisper as openai_whisper
from whisper.tokenizer import get_tokenizer

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.whisper.modeling_whisper import (
    WhisperInferenceConfig, NeuronApplicationWhisper,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


def make_config(model_path, K):
    nc = NeuronConfig(batch_size=1, torch_dtype=torch.float16, tp_degree=4, logical_nc_config=2)
    nc.whisper_prompt_len = K + 1
    return WhisperInferenceConfig(nc, load_config=load_pretrained_config(model_path))


def load_model(model_path, traced_path, K, tag):
    print(f"[{tag}] loading from {traced_path}...", flush=True)
    cfg = make_config(model_path, K)
    model = NeuronApplicationWhisper(traced_path, config=cfg)
    model.load(traced_path)
    return model


@torch.inference_mode()
def greedy_target_only(target, encoded, initial_tokens, max_new=224, eot=50257):
    """Baseline: greedy decode using target-only, full-prefill per step for pure correctness."""
    tokens = list(initial_tokens)
    n_steps = 0
    t0 = time.perf_counter()
    while len(tokens) < len(initial_tokens) + max_new:
        tok = torch.tensor([tokens], dtype=torch.int32)
        logits = target.logits(tok, encoded)
        next_id = int(torch.argmax(logits[0, -1]).item())
        tokens.append(next_id)
        n_steps += 1
        if next_id == eot:
            break
    return tokens, time.perf_counter() - t0, n_steps


@torch.inference_mode()
def spec_decode_greedy(target, draft, encoded, initial_tokens, K=4, max_new=224, eot=50257, log=False):
    """
    POC v1: full-prefill both models each iter.

    Returns:
        tokens: full decoded token list including initial prompt
        elapsed_s: wall time
        stats: dict
    """
    tokens = list(initial_tokens)
    stats = {
        "iters": 0, "proposals": 0, "accepted": 0,
        "draft_calls": 0, "target_verify_calls": 0,
        "target_prefill_calls": 0, "draft_prefill_calls": 0,
        "accepted_per_iter": [],
    }
    t0 = time.perf_counter()

    while len(tokens) < len(initial_tokens) + max_new:
        stats["iters"] += 1

        # (a) Full prefill both models to sync KV caches to the current tokens list.
        prompt = torch.tensor([tokens], dtype=torch.int32)
        target_logits = target.logits(prompt, encoded)
        stats["target_prefill_calls"] += 1
        draft.logits(prompt, encoded)
        stats["draft_prefill_calls"] += 1
        # target's opinion for the NEXT token position (len(tokens)):
        target_opinion = int(torch.argmax(target_logits[0, -1]).item())

        # If target's own opinion at this position is EOT, we're done -- just commit EOT.
        if target_opinion == eot:
            tokens.append(eot)
            stats["accepted_per_iter"].append(0)
            break

        # (b) Draft proposes K tokens greedily
        proposals = []
        for k in range(K):
            dtok = torch.tensor([tokens + proposals], dtype=torch.int32)
            dlogits = draft.logits(dtok, encoded)
            stats["draft_calls"] += 1
            p = int(torch.argmax(dlogits[0, -1]).item())
            proposals.append(p)
            if p == eot:
                break

        # (c) Sequentially verify against target
        num_accepted = 0
        while num_accepted < len(proposals):
            k = num_accepted
            p = proposals[k]
            if target_opinion == p:
                # accepted; advance target's state by feeding tokens + proposals[:k+1]
                ttok = torch.tensor([tokens + proposals[:k + 1]], dtype=torch.int32)
                tlogits = target.logits(ttok, encoded)
                stats["target_verify_calls"] += 1
                target_opinion = int(torch.argmax(tlogits[0, -1]).item())
                num_accepted += 1
                if p == eot:
                    break
            else:
                # rejected at position k; target_opinion is target's replacement token
                break

        # Commit
        tokens.extend(proposals[:num_accepted])
        stats["accepted"] += num_accepted
        stats["proposals"] += len(proposals)
        stats["accepted_per_iter"].append(num_accepted)

        # If all proposals were accepted AND none were EOT, target_opinion is a bonus token.
        # If num_accepted < len(proposals), target_opinion is the replacement for proposals[num_accepted].
        # Either way, commit target_opinion.
        # Exception: if the last accepted proposal was already EOT, don't append anything more.
        if num_accepted > 0 and proposals[num_accepted - 1] == eot:
            break
        tokens.append(target_opinion)
        if log:
            print(f"    iter {stats['iters']}: proposed={len(proposals)} accepted={num_accepted} tokens_len={len(tokens)}", flush=True)
        if target_opinion == eot:
            break

    return tokens, time.perf_counter() - t0, stats


def audio_to_mel(audio, n_mels=128):
    audio = openai_whisper.pad_or_trim(audio)
    return openai_whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-traced", default="/home/ubuntu/traced/whisper-large-v3-K4")
    ap.add_argument("--draft-traced", default="/home/ubuntu/traced/distil-large-v3-K4")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--min-dur", type=float, default=15.0)
    ap.add_argument("--out-file", default="/home/ubuntu/task004_results.json")
    ap.add_argument("--target-only", action="store_true")
    ap.add_argument("--spec-only", action="store_true")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    target = load_model("/home/ubuntu/models/whisper-large-v3", args.target_traced, args.K, "TARGET")
    draft = load_model("/home/ubuntu/models/distil-large-v3", args.draft_traced, args.K, "DRAFT")

    # whisper-large-v3 uses the v3 tokenizer (num_languages=100) which has different special-token IDs
    # from v2. Wrong tokenizer -> target predicts EOT after prompt because prompt looks like garbage.
    tokenizer = get_tokenizer(multilingual=True, task="transcribe", language="en", num_languages=100)
    sot = tokenizer.sot
    lang = tokenizer.to_language_token("en")
    task_tok = tokenizer.transcribe
    notimestamps = tokenizer.no_timestamps
    initial_tokens = (sot, lang, task_tok, notimestamps)
    eot = tokenizer.eot
    print(f"initial_tokens={list(initial_tokens)}, eot={eot}", flush=True)

    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
    clips = []
    for ex in ds:
        dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
        if dur >= args.min_dur:
            clips.append(ex)
            if len(clips) >= args.n_clips:
                break

    results = []
    for i, ex in enumerate(clips):
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        dur = len(audio) / ex["audio"]["sampling_rate"]
        print(f"\n=== clip {i+1}/{len(clips)}: {ex['id']} ({dur:.1f}s) ===", flush=True)
        mel = audio_to_mel(audio).to(torch.float16)
        enc = target.encoder(mel)

        clip_res = {"id": ex["id"], "duration": dur, "reference": ex["text"]}

        if not args.spec_only:
            print("greedy target-only...", flush=True)
            g_tokens, g_time, g_steps = greedy_target_only(target, enc, initial_tokens, eot=eot)
            g_text = tokenizer.decode([t for t in g_tokens if t not in (sot, lang, task_tok, notimestamps, eot)])
            print(f"  {g_time*1000:.0f}ms, {g_steps} steps, {len(g_tokens)} tokens", flush=True)
            print(f"  {g_text!r}", flush=True)
            clip_res["greedy"] = {"tokens": g_tokens, "text": g_text, "time_s": g_time, "steps": g_steps}

        if not args.target_only:
            print(f"spec-decode K={args.K}...", flush=True)
            s_tokens, s_time, s_stats = spec_decode_greedy(target, draft, enc, initial_tokens, K=args.K, eot=eot, log=args.log)
            s_text = tokenizer.decode([t for t in s_tokens if t not in (sot, lang, task_tok, notimestamps, eot)])
            print(f"  {s_time*1000:.0f}ms, iters={s_stats['iters']}, tokens={len(s_tokens)}", flush=True)
            print(f"  {s_text!r}", flush=True)
            print(f"  stats={s_stats}", flush=True)
            clip_res["spec"] = {"tokens": s_tokens, "text": s_text, "time_s": s_time, "stats": s_stats, "K": args.K}
            if not args.spec_only:
                clip_res["byte_identical"] = (g_tokens == s_tokens)
                clip_res["speedup"] = g_time / s_time
                acc_rate = s_stats["accepted"] / s_stats["proposals"] if s_stats["proposals"] > 0 else 0
                clip_res["acceptance_rate"] = acc_rate
                print(f"  byte-identical: {clip_res['byte_identical']}, speedup: {clip_res['speedup']:.2f}x, acc={acc_rate:.3f}", flush=True)

        results.append(clip_res)

    with open(args.out_file, "w") as f:
        out = []
        for r in results:
            r2 = dict(r)
            for k in ("greedy", "spec"):
                if k in r2:
                    r2[k] = dict(r2[k])
                    r2[k]["tokens"] = list(r2[k]["tokens"])
            out.append(r2)
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out_file}", flush=True)

    if not args.target_only and not args.spec_only:
        bi = sum(1 for r in results if r.get("byte_identical"))
        speedups = [r["speedup"] for r in results if "speedup" in r]
        accs = [r["acceptance_rate"] for r in results if "acceptance_rate" in r]
        print(f"\n=== Summary ===", flush=True)
        print(f"  byte-identical: {bi}/{len(results)}", flush=True)
        if speedups:
            print(f"  mean speedup: {sum(speedups)/len(speedups):.2f}x  min: {min(speedups):.2f}  max: {max(speedups):.2f}", flush=True)
        if accs:
            print(f"  mean acceptance: {sum(accs)/len(accs):.3f}", flush=True)


if __name__ == "__main__":
    main()
