"""POC v4.b: bootstrap via DecoderPrefillShort (task 014).

Extends POC v4.a by eliminating the two full-prefill `.logits()` calls in the bootstrap phase.
Instead, both target and draft prime their KV cache via the DecoderPrefillShort NEFF (traced at
[1, K+1] = [1, 5]) applied to a padded copy of the 4-token initial prompt.

Bootstrap savings (Task 010 measurements):
  * target: full-prefill 44 ms -> DecoderPrefillShort 9.6 ms  (~34 ms saved)
  * draft : full-prefill 29 ms -> DecoderPrefillShort 5.0 ms  (~24 ms saved)
  Total: ~58 ms saved per clip.

Design:
  Initial prompt is 4 tokens [SOT, LANG, TASK, NO_TIMESTAMPS], but the short NEFF is compiled at
  seq_len = K+1 = 5. To reuse the NEFF, we pad the prompt from 4 to 5 tokens with a dummy value
  (0) at the tail. Call model.decoder(text=[SOT,LANG,TASK,NOTS,DUMMY], last_pos=4). The NEFF
  writes KV at positions 0..4:
      * positions 0..3 hold correct K/V for the four real tokens (used by subsequent iters)
      * position 4 holds garbage K/V from the dummy token
        -> harmless because the FIRST spec-decode iter's verify-NEFF writes positions
        N-1..N+K-1 = 3..3+K, overwriting position 4 with a real proposal token before it is
        ever read as context.
  The prediction for position 4 (first token to generate) is argmax(logits[0, 3]) -- the logit
  at the last REAL input position. logits[0, 4] uses the dummy token and is discarded.

Correctness proof:
  Position 3 (last real input) gets rewritten by the first iter's verify NEFF with the same token
  value (tokens[-1] = NO_TIMESTAMPS). Idempotent rewrite. Positions 0..2 are never rewritten and
  hold correct KV. Position 4 (dummy) is overwritten before being read. => byte-identical to
  greedy baseline that uses full-prefill.

Everything else is identical to POC v4.a from Task 012.
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


K = 4  # spec-decode K


def make_config(model_path):
    nc = NeuronConfig(batch_size=1, torch_dtype=torch.float16, tp_degree=4, logical_nc_config=2)
    nc.whisper_prompt_len = K + 1  # short NEFF traced at K+1 = 5
    return WhisperInferenceConfig(nc, load_config=load_pretrained_config(model_path))


def load_model(model_path, traced_path):
    cfg = make_config(model_path)
    m = NeuronApplicationWhisper(traced_path, config=cfg)
    m.load(traced_path)
    return m


def _decode_step(model, token_id, kv_write_pos, encoded):
    """Single-token DecoderDecode."""
    text = torch.tensor([[token_id]], dtype=torch.int32)
    audio = encoded.to(model.config.neuron_config.torch_dtype)
    last_pos = torch.tensor([kv_write_pos], dtype=torch.int32)
    n_ctx = model.dims.n_text_ctx
    pad_mask = torch.zeros((1, n_ctx), dtype=torch.int32)
    pad_mask[:, :kv_write_pos + 1] = 1
    out = model.decoder(text, audio, last_pos, pad_mask)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return int(torch.argmax(logits[0, -1]).item()), logits


def _verify_k_trailing(target, verify_tokens_kp1, last_pos_of_last, encoded):
    """POC v4.a verify-K at trailing positions. Unchanged from Task 012.

    verify_tokens_kp1: K+1 int token ids (positions N-1..N+K-1)
    last_pos_of_last: KV position of the LAST input token = N + K - 1.

    Returns [1, K+1, vocab] logits.
    """
    text = torch.tensor([verify_tokens_kp1], dtype=torch.int32)
    audio = encoded.to(target.config.neuron_config.torch_dtype)
    last_pos = torch.tensor([last_pos_of_last], dtype=torch.int32)
    n_ctx = target.dims.n_text_ctx
    pad_mask = torch.zeros((1, n_ctx), dtype=torch.int32)
    pad_mask[:, :last_pos_of_last + 1] = 1
    out = target.decoder(text, audio, last_pos, pad_mask)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits


def _bootstrap_short_prefill(model, initial_tokens, encoded):
    """Task 014 bootstrap via DecoderPrefillShort.

    Args:
      model: NeuronApplicationWhisper
      initial_tokens: sequence of len < K+1 (typically 4) -- the forced Whisper prompt.
      encoded: encoder output tensor

    Returns:
      opinion: int, argmax of logits at the last REAL prompt position.
               = model's prediction for the first token to generate.

    Side effect: model's KV cache is populated at positions [0..prompt_len-1] with the
      forced-prompt K/V, and at position [prompt_len..K] with garbage from dummy padding
      (safe because those positions get overwritten by the first spec-decode iter).
    """
    prompt_len = len(initial_tokens)  # e.g., 4
    neff_len = K + 1  # 5
    assert prompt_len <= neff_len, f"initial_tokens ({prompt_len}) longer than DecoderPrefillShort seq_len ({neff_len})"

    # Pad with dummy tokens (0) at the tail.
    padded = list(initial_tokens) + [0] * (neff_len - prompt_len)  # length neff_len
    text = torch.tensor([padded], dtype=torch.int32)
    audio = encoded.to(model.config.neuron_config.torch_dtype)

    # last_pos = neff_len - 1 makes offset = last_pos - neff_len + 1 = 0.
    # KV writes go to positions [0..neff_len-1].
    last_pos = torch.tensor([neff_len - 1], dtype=torch.int32)

    # pad_mask must mark positions 0..prompt_len-1 as valid (the real tokens).
    # The dummy positions [prompt_len..neff_len-1] are NOT part of the actual context.
    # However, the causal mask in the NEFF is built from arange comparisons using the
    # write_offset (=0) and the Q's own row index. The pad_mask primarily constrains the
    # cross-attention side; for self-attention within this call it's less critical. We
    # set pad_mask valid up to prompt_len to be conservative.
    n_ctx = model.dims.n_text_ctx
    pad_mask = torch.zeros((1, n_ctx), dtype=torch.int32)
    pad_mask[:, :prompt_len] = 1

    out = model.decoder(text, audio, last_pos, pad_mask)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    # Logits shape [1, neff_len, vocab]. logits[0, prompt_len - 1] uses the last real input
    # and predicts the first generation position.
    return int(torch.argmax(logits[0, prompt_len - 1]).item())


@torch.inference_mode()
def greedy_target_only_fast(target, encoded, initial_tokens, max_new=224, eot=50257):
    """Baseline: greedy decode with target only. Uses the same bootstrap short-prefill
    so both baseline and spec-decode start from the same KV state. This keeps the
    comparison fair (both pay ~9.6 ms bootstrap on target instead of ~44 ms).
    """
    tokens = list(initial_tokens)
    t0 = time.perf_counter()
    # Task 014: bootstrap via DecoderPrefillShort (was: target.logits(prompt, encoded))
    next_id = _bootstrap_short_prefill(target, initial_tokens, encoded)
    tokens.append(next_id)
    n_steps = 1
    while next_id != eot and len(tokens) < len(initial_tokens) + max_new:
        pos = len(tokens) - 1
        next_id, _ = _decode_step(target, tokens[-1], pos, encoded)
        tokens.append(next_id)
        n_steps += 1
    return tokens, time.perf_counter() - t0, n_steps


@torch.inference_mode()
def greedy_target_only_full_prefill(target, encoded, initial_tokens, max_new=224, eot=50257):
    """Legacy baseline: greedy with FULL DecoderPrefill for bootstrap.
    Same as POC v4.a's `greedy_target_only_fast`. Kept for A/B comparison with the new bootstrap.
    """
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
def spec_decode_v4b(target, draft, encoded, initial_tokens, K_=K, max_new=224, eot=50257, log=False):
    """POC v4.b spec-decode: identical to v4.a except bootstrap uses DecoderPrefillShort.
    """
    tokens = list(initial_tokens)
    stats = {
        "iters": 0, "proposals": 0, "accepted": 0,
        "draft_decode_calls": 0, "target_verify_k_calls": 0,
        "draft_prefill_calls": 0, "target_prefill_calls": 0,
        "all_accepted_iters": 0, "partial_reject_iters": 0,
        "accepted_per_iter": [],
    }
    t0 = time.perf_counter()

    # Task 014: bootstrap via SHORT prefill on both models (was: full DecoderPrefill x2)
    target_opinion = _bootstrap_short_prefill(target, initial_tokens, encoded)
    stats["target_prefill_calls"] += 1
    draft_opinion = _bootstrap_short_prefill(draft, initial_tokens, encoded)
    stats["draft_prefill_calls"] += 1

    if target_opinion == eot:
        tokens.append(eot)
        return tokens, time.perf_counter() - t0, stats

    while len(tokens) < len(initial_tokens) + max_new:
        stats["iters"] += 1
        N = len(tokens)

        # (1) Draft proposes K tokens
        proposals = [draft_opinion]
        cur_input = draft_opinion
        for k in range(1, K_):
            next_tok, _ = _decode_step(draft, cur_input, N + k - 1, encoded)
            stats["draft_decode_calls"] += 1
            proposals.append(next_tok)
            cur_input = next_tok
            if next_tok == eot:
                break
        num_prop = len(proposals)

        # (2) Target verify-K trailing (unchanged from v4.a)
        pad_needed = K_ - num_prop
        if pad_needed > 0:
            verify_tokens = [tokens[-1]] + proposals + [0] * pad_needed
        else:
            verify_tokens = [tokens[-1]] + proposals

        last_pos_of_last = (N - 1) + K_
        vlogits = _verify_k_trailing(target, verify_tokens, last_pos_of_last, encoded)
        stats["target_verify_k_calls"] += 1
        opinion_slice = vlogits[0, :num_prop + 1]
        opinions = torch.argmax(opinion_slice, dim=-1).tolist()

        # (3) Accept/reject
        num_accepted = 0
        for j in range(num_prop):
            if opinions[j] == proposals[j]:
                num_accepted += 1
                if proposals[j] == eot:
                    break
            else:
                break

        # (4) Commit
        tokens.extend(proposals[:num_accepted])
        stats["accepted"] += num_accepted
        stats["proposals"] += num_prop
        stats["accepted_per_iter"].append(num_accepted)

        if num_accepted > 0 and proposals[num_accepted - 1] == eot:
            break

        next_tok = opinions[num_accepted]
        tokens.append(next_tok)
        if next_tok == eot:
            break

        # (5) Draft bookkeep for next iter
        pos = len(tokens) - 1
        if num_accepted == num_prop:
            stats["all_accepted_iters"] += 1
        else:
            stats["partial_reject_iters"] += 1
        next_draft_opinion, _ = _decode_step(draft, next_tok, pos, encoded)
        stats["draft_decode_calls"] += 1
        draft_opinion = next_draft_opinion

        if log:
            print(f"    iter {stats['iters']}: proposed={num_prop} accepted={num_accepted} tokens_len={len(tokens)}", flush=True)

    return tokens, time.perf_counter() - t0, stats


def audio_to_mel(audio, n_mels=128):
    audio = openai_whisper.pad_or_trim(audio)
    return openai_whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-traced", default="/home/ubuntu/traced/whisper-large-v3-K4")
    ap.add_argument("--draft-traced", default="/home/ubuntu/traced/distil-large-v3-K4")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--split", default="test.clean")
    ap.add_argument("--out-file", default="/home/ubuntu/task014_v4b_results.json")
    ap.add_argument("--baseline", choices=["short", "full"], default="short",
                    help="which bootstrap the greedy baseline uses. 'short'=fair (both use short); 'full'=legacy v4.a comparison")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    target = load_model("/home/ubuntu/models/whisper-large-v3", args.target_traced)
    draft = load_model("/home/ubuntu/models/distil-large-v3", args.draft_traced)

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

    baseline_fn = greedy_target_only_fast if args.baseline == "short" else greedy_target_only_full_prefill
    print(f"Using baseline: {args.baseline} (fn={baseline_fn.__name__})", flush=True)

    results = []
    for i, ex in enumerate(clips):
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        dur = len(audio) / ex["audio"]["sampling_rate"]
        print(f"\n=== clip {i+1}/{len(clips)}: {ex['id']} ({dur:.1f}s) ===", flush=True)
        mel = audio_to_mel(audio).to(torch.float16)
        enc = target.encoder(mel)

        clip_res = {"id": ex["id"], "duration": dur}

        gf_tokens, gf_time, gf_steps = baseline_fn(target, enc, initial_tokens, eot=eot)
        gf_text = tokenizer.decode([t for t in gf_tokens if t < 50257])
        print(f"  baseline:     {gf_time*1000:.0f}ms, {gf_steps} steps, {len(gf_tokens)} tokens", flush=True)

        s_tokens, s_time, s_stats = spec_decode_v4b(target, draft, enc, initial_tokens, eot=eot, log=args.log)
        s_text = tokenizer.decode([t for t in s_tokens if t < 50257])
        print(f"  spec v4.b:    {s_time*1000:.0f}ms, iters={s_stats['iters']}, tokens={len(s_tokens)}", flush=True)

        clip_res["greedy"] = {"tokens": gf_tokens, "text": gf_text, "time_s": gf_time, "steps": gf_steps}
        clip_res["spec"] = {"tokens": s_tokens, "text": s_text, "time_s": s_time, "stats": s_stats, "K": K}
        clip_res["byte_identical"] = (gf_tokens == s_tokens)
        clip_res["speedup"] = gf_time / s_time
        clip_res["acceptance_rate"] = s_stats["accepted"] / s_stats["proposals"] if s_stats["proposals"] > 0 else 0
        print(f"  BI={clip_res['byte_identical']} SU={clip_res['speedup']:.2f}x acc={clip_res['acceptance_rate']:.2f}", flush=True)

        if not clip_res["byte_identical"]:
            print(f"    base tokens[:30]={gf_tokens[:30]}", flush=True)
            print(f"    spec tokens[:30]={s_tokens[:30]}", flush=True)
            for k, (g, s) in enumerate(zip(gf_tokens, s_tokens)):
                if g != s:
                    print(f"    diverge at index {k}: greedy={g} spec={s}", flush=True)
                    break

        results.append(clip_res)

    with open(args.out_file, "w") as f:
        json.dump([{**r, "greedy": {**r["greedy"], "tokens": list(r["greedy"]["tokens"])},
                    "spec": {**r["spec"], "tokens": list(r["spec"]["tokens"])}} for r in results], f, indent=2)

    bi = sum(1 for r in results if r["byte_identical"])
    speedups = [r["speedup"] for r in results]
    accs = [r["acceptance_rate"] for r in results]
    print(f"\n=== Summary ===")
    print(f"  byte-identical: {bi}/{len(results)}")
    print(f"  mean speedup: {sum(speedups)/len(speedups):.3f}x  min: {min(speedups):.3f}  max: {max(speedups):.3f}")
    print(f"  mean acceptance: {sum(accs)/len(accs):.3f}")


if __name__ == "__main__":
    main()
