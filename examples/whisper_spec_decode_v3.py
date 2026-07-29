"""POC v3.1: verify using K sequential target DecoderDecodes instead of 1 target full-prefill.
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
    cfg = make_config(model_path, K)
    m = NeuronApplicationWhisper(traced_path, config=cfg)
    m.load(traced_path)
    return m


def _decode_step(model, token_id, kv_write_pos, encoded):
    """Single-token DecoderDecode. Returns (next_argmax, logits_tensor)."""
    text = torch.tensor([[token_id]], dtype=torch.int32)
    audio = encoded.to(model.config.neuron_config.torch_dtype)
    last_pos = torch.tensor([kv_write_pos], dtype=torch.int32)
    n_ctx = model.dims.n_text_ctx
    pad_mask = torch.zeros((1, n_ctx), dtype=torch.int32)
    pad_mask[:, :kv_write_pos + 1] = 1
    out = model.decoder(text, audio, last_pos, pad_mask)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return int(torch.argmax(logits[0, -1]).item()), logits


@torch.inference_mode()
def greedy_target_only_fast(target, encoded, initial_tokens, max_new=224, eot=50257):
    tokens = list(initial_tokens)
    t0 = time.perf_counter()
    prompt = torch.tensor([tokens], dtype=torch.int32)
    logits = target.logits(prompt, encoded)
    next_id = int(torch.argmax(logits[0, -1]).item())
    tokens.append(next_id)
    n_steps = 1
    while next_id != eot and len(tokens) < len(initial_tokens) + max_new:
        pos = len(tokens) - 1
        next_id, _ = _decode_step(target, tokens[-1], pos, encoded)
        tokens.append(next_id)
        n_steps += 1
    return tokens, time.perf_counter() - t0, n_steps


@torch.inference_mode()
def spec_decode_v3_1(target, draft, encoded, initial_tokens, K=4, max_new=224, eot=50257, log=False):
    """POC v3.1: verify using K sequential target DecoderDecodes."""
    tokens = list(initial_tokens)
    stats = {
        "iters": 0, "proposals": 0, "accepted": 0,
        "draft_decode_calls": 0, "target_decode_calls": 0,
        "draft_prefill_calls": 0, "target_prefill_calls": 0,
        "all_accepted_iters": 0, "partial_reject_iters": 0,
        "accepted_per_iter": [],
    }
    t0 = time.perf_counter()

    # Bootstrap: prefill both models
    prompt = torch.tensor([tokens], dtype=torch.int32)
    tlogits = target.logits(prompt, encoded)
    stats["target_prefill_calls"] += 1
    dlogits = draft.logits(prompt, encoded)
    stats["draft_prefill_calls"] += 1
    target_opinion = int(torch.argmax(tlogits[0, -1]).item())
    draft_opinion = int(torch.argmax(dlogits[0, -1]).item())

    if target_opinion == eot:
        tokens.append(eot)
        return tokens, time.perf_counter() - t0, stats

    while len(tokens) < len(initial_tokens) + max_new:
        stats["iters"] += 1
        N = len(tokens)

        # (1) Draft proposes K tokens via K single-token DecoderDecodes.
        # Start from draft_opinion (for position N), advance draft by K-1 more steps.
        proposals = [draft_opinion]
        cur_input = draft_opinion
        for k in range(1, K):
            # write cur_input at position N+k-1, read draft's opinion for position N+k
            next_tok, _ = _decode_step(draft, cur_input, N + k - 1, encoded)
            stats["draft_decode_calls"] += 1
            proposals.append(next_tok)
            cur_input = next_tok
            if next_tok == eot:
                break
        num_prop = len(proposals)

        # (2) Verify: K sequential target DecoderDecodes, each writes at KV positions N..N+K-1.
        # We already have target_opinion = target's opinion for position N (from previous iter's
        # bootstrap or bonus). Now feed proposals[0..K-1] to target sequentially.
        target_opinions = [target_opinion]  # for position N
        for k in range(num_prop):
            # write proposals[k] at position N+k, read target's opinion for position N+k+1
            next_opinion, _ = _decode_step(target, proposals[k], N + k, encoded)
            stats["target_decode_calls"] += 1
            target_opinions.append(next_opinion)

        # (3) Accept/reject
        num_accepted = 0
        for j in range(num_prop):
            if target_opinions[j] == proposals[j]:
                num_accepted += 1
                if proposals[j] == eot:
                    break
            else:
                break

        # (4) Commit accepted proposals + bonus/replacement
        tokens.extend(proposals[:num_accepted])
        stats["accepted"] += num_accepted
        stats["proposals"] += num_prop
        stats["accepted_per_iter"].append(num_accepted)

        if num_accepted > 0 and proposals[num_accepted - 1] == eot:
            break

        # next token: bonus (if all K accepted) or replacement (partial reject)
        next_tok = target_opinions[num_accepted]
        tokens.append(next_tok)
        if next_tok == eot:
            break

        # (5) KV sync bookkeeping
        # Target's KV: positions 0..N+num_prop-1 are written (K target decodes wrote at N..N+K-1).
        # Committed length now = N + num_accepted + 1.
        # Consistency: for j < num_accepted, target wrote proposals[j] at position N+j (correct).
        # For j >= num_accepted, target wrote proposals[j] at position N+j (stale; those slots
        # should be replaced by bonus/replacement).
        #
        # On all-accepted (num_accepted == K):
        #   - target's KV has positions 0..N+K-1 with committed tokens (correct so far).
        #   - Position N+K = tokens[-1] = bonus (not yet written to target KV).
        #   - target_opinion for next iter's first verify (position N+K = new_N-1): we have
        #     target_opinions[K] = target's opinion for position N+K, which is what target
        #     would have said AFTER seeing proposals[K-1] at position N+K-1. That equals
        #     the "bonus" token we just committed as tokens[-1]. So target_opinion (next iter's
        #     seed for position new_N) requires target to see the bonus at N+K first.
        #     -> We need one more target decode: feed bonus at position N+K, get opinion for N+K+1.
        #   - Draft's KV: it has proposals[0..K-1] at positions N..N+K-1 (from draft decode chain).
        #     For next iter, first draft proposal is at position new_N = N+K+1. draft_opinion for
        #     that position requires draft to see the bonus at N+K. -> One more draft decode too.
        #
        # On partial-reject at j (num_accepted = j):
        #   - target's KV at positions N+j..N+K-1 have stale proposals[j..K-1].
        #     Committed tokens now include replacement=target_opinions[j] at position N+j.
        #     For NEXT iter's target verify, we'll re-run target decodes at N+j+1, N+j+2, ...
        #     But target's KV at position N+j has stale proposals[j], not replacement.
        #     So we need to fix target's KV at position N+j: decode replacement at N+j.
        #     Then target's KV positions N+j+1..N+K-1 are still stale but will be overwritten
        #     by next iter's target decodes.
        #   - target_opinion for next iter's position new_N = N+j+1: we need it. One target decode
        #     of replacement at N+j gives target's opinion for N+j+1.
        #   - Draft's KV at positions N+j..N+K-1 also stale. Same fix: one draft decode of
        #     replacement at N+j.

        # In BOTH cases, we do:
        #   target decode of next_tok at position len(tokens)-1 -> new target_opinion
        #   draft decode of next_tok at position len(tokens)-1 -> new draft_opinion
        pos = len(tokens) - 1
        target_opinion, _ = _decode_step(target, next_tok, pos, encoded)
        stats["target_decode_calls"] += 1
        draft_opinion, _ = _decode_step(draft, next_tok, pos, encoded)
        stats["draft_decode_calls"] += 1

        if num_accepted == num_prop:
            stats["all_accepted_iters"] += 1
        else:
            stats["partial_reject_iters"] += 1

        if log:
            print(f"    iter {stats['iters']}: proposed={num_prop} accepted={num_accepted} "
                  f"tokens_len={len(tokens)}", flush=True)

    return tokens, time.perf_counter() - t0, stats


def audio_to_mel(audio, n_mels=128):
    audio = openai_whisper.pad_or_trim(audio)
    return openai_whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-traced", default="/home/ubuntu/traced/whisper-large-v3-K4")
    ap.add_argument("--draft-traced", default="/home/ubuntu/traced/distil-large-v3-K4")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-clips", type=int, default=5)
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--split", default="test.clean")
    ap.add_argument("--out-file", default="/home/ubuntu/task011_v3_1_results.json")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    target = load_model("/home/ubuntu/models/whisper-large-v3", args.target_traced, args.K, "TARGET")
    draft = load_model("/home/ubuntu/models/distil-large-v3", args.draft_traced, args.K, "DRAFT")

    tokenizer = get_tokenizer(multilingual=True, task="transcribe", language="en", num_languages=100)
    initial_tokens = (tokenizer.sot, tokenizer.to_language_token("en"), tokenizer.transcribe, tokenizer.no_timestamps)
    eot = tokenizer.eot

    split_name = "clean" if "clean" in args.split else "other"
    ds = load_dataset("openslr/librispeech_asr", split_name, split="test", streaming=True)
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

        clip_res = {"id": ex["id"], "duration": dur}

        gf_tokens, gf_time, gf_steps = greedy_target_only_fast(target, enc, initial_tokens, eot=eot)
        gf_text = tokenizer.decode([t for t in gf_tokens if t < 50257])
        print(f"  greedy-fast:    {gf_time*1000:.0f}ms, {gf_steps} steps, {len(gf_tokens)} tokens", flush=True)

        s_tokens, s_time, s_stats = spec_decode_v3_1(target, draft, enc, initial_tokens, K=args.K, eot=eot, log=args.log)
        s_text = tokenizer.decode([t for t in s_tokens if t < 50257])
        print(f"  spec v3.1:      {s_time*1000:.0f}ms, iters={s_stats['iters']}, "
              f"tokens={len(s_tokens)} tD={s_stats['target_decode_calls']} dD={s_stats['draft_decode_calls']}", flush=True)

        clip_res["greedy"] = {"tokens": gf_tokens, "text": gf_text, "time_s": gf_time, "steps": gf_steps}
        clip_res["spec"] = {"tokens": s_tokens, "text": s_text, "time_s": s_time, "stats": s_stats, "K": args.K}
        clip_res["byte_identical"] = (gf_tokens == s_tokens)
        clip_res["speedup"] = gf_time / s_time
        clip_res["acceptance_rate"] = s_stats["accepted"] / s_stats["proposals"] if s_stats["proposals"] > 0 else 0
        print(f"  BI={clip_res['byte_identical']} SU={clip_res['speedup']:.2f}x acc={clip_res['acceptance_rate']:.2f}", flush=True)

        if not clip_res["byte_identical"]:
            print(f"    fast tokens[:30]={gf_tokens[:30]}", flush=True)
            print(f"    spec tokens[:30]={s_tokens[:30]}", flush=True)

        results.append(clip_res)

    with open(args.out_file, "w") as f:
        json.dump([{**r, "greedy": {**r["greedy"], "tokens": list(r["greedy"]["tokens"])},
                    "spec": {**r["spec"], "tokens": list(r["spec"]["tokens"])}} for r in results], f, indent=2)

    bi = sum(1 for r in results if r["byte_identical"])
    speedups = [r["speedup"] for r in results]
    accs = [r["acceptance_rate"] for r in results]
    print(f"\n=== Summary ===")
    print(f"  byte-identical: {bi}/{len(results)}")
    print(f"  mean speedup: {sum(speedups)/len(speedups):.2f}x  min: {min(speedups):.2f}  max: {max(speedups):.2f}")
    print(f"  mean acceptance: {sum(accs)/len(accs):.3f}")


if __name__ == "__main__":
    main()
