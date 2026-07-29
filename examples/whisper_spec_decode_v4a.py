"""POC v4.a: verify-K via DecoderPrefillShort at trailing positions.

Uses the modified NeuronMultiHeadAttention + NeuronTextDecoder that support the trailing-write
mode when calling the SHORT prefill NEFF with a `last_pos` value indicating trailing positions.

Design:
  1. Draft proposes K tokens via K single-token DecoderDecode calls (unchanged from POC v3.1)
  2. Target verify uses ONE call to DecoderPrefillShort at [1, K+1] tokens, with last_pos set
     to N + K (so the write range is N..N+K). This costs ~10 ms instead of 4 * 6.16 = 24.6 ms
     for K=4 sequential target decodes, or 44 ms for one full-prefill.
  3. Return logits for the K+1 verify positions, do accept/reject.
  4. On both all-accepted and partial-reject, next iter's first target verify will call
     DecoderPrefillShort again at the new trailing positions, overwriting stale KV.
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
    """Task 012 verify-K at trailing positions.

    Args:
      target: NeuronApplicationWhisper for the target model
      verify_tokens_kp1: list of K+1 int token ids -- the "context predecessor" + K proposals
                         (positions N-1, N, ..., N+K-1 in the committed+proposal sequence)
      last_pos_of_last: the KV position where the LAST token in verify_tokens_kp1 gets written.
                        This equals N + K - 1 where N = number of committed tokens before this iter.
                        (I.e., we're writing at positions N-1, N, ..., N+K-1.)
      encoded: the encoder output tensor.

    Note: passing verify_tokens_kp1 including the LAST COMMITTED TOKEN + K proposals means
    K+1 positions are written. Position N-1 (last committed) is a REWRITE (idempotent).
    Positions N..N+K-1 are new writes for the K proposals.

    Returns [1, K+1, vocab] logits.
    """
    text = torch.tensor([verify_tokens_kp1], dtype=torch.int32)
    audio = encoded.to(target.config.neuron_config.torch_dtype)
    last_pos = torch.tensor([last_pos_of_last], dtype=torch.int32)
    n_ctx = target.dims.n_text_ctx
    pad_mask = torch.zeros((1, n_ctx), dtype=torch.int32)
    pad_mask[:, :last_pos_of_last + 1] = 1  # valid positions 0..last_pos_of_last
    out = target.decoder(text, audio, last_pos, pad_mask)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits  # [1, K+1, vocab]


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
def spec_decode_v4a(target, draft, encoded, initial_tokens, K_=K, max_new=224, eot=50257, log=False):
    tokens = list(initial_tokens)
    stats = {
        "iters": 0, "proposals": 0, "accepted": 0,
        "draft_decode_calls": 0, "target_verify_k_calls": 0,
        "draft_prefill_calls": 0, "target_prefill_calls": 0,
        "all_accepted_iters": 0, "partial_reject_iters": 0,
        "accepted_per_iter": [],
    }
    t0 = time.perf_counter()

    # Bootstrap: full-prefill both models on initial_tokens
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

        # (2) Target verify: ONE call to DecoderPrefillShort at trailing positions.
        # verify_tokens_kp1 = [tokens[-1], proposals[0], proposals[1], ..., proposals[K-1]]
        # These are input positions N-1, N, N+1, ..., N+K-1.
        # last_pos_of_last = N + K - 1 (where proposals[K-1] gets written).
        #
        # BUT: the NEFF is compiled at prompt_len = K+1. So we always need K+1 tokens.
        # If num_prop < K (draft stopped early on EOT), we still need K+1 inputs. Pad with anything
        # (say, 0 or eot) at the tail -- those positions' logits will be discarded.
        #
        # Also, the last committed token position (N-1) is a REWRITE. Draft's KV at N-1 already has
        # tokens[-1] (from previous iter's bookkeep decode). Target's KV at N-1 was written by the
        # bootstrap prefill or the previous iter's verify. Writing again is idempotent for this token.
        pad_needed = K_ - num_prop
        if pad_needed > 0:
            verify_tokens = [tokens[-1]] + proposals + [0] * pad_needed  # length K+1
        else:
            verify_tokens = [tokens[-1]] + proposals  # length K+1

        last_pos_of_last = (N - 1) + K_  # = N + K - 1
        vlogits = _verify_k_trailing(target, verify_tokens, last_pos_of_last, encoded)
        stats["target_verify_k_calls"] += 1
        # vlogits shape [1, K+1, vocab]. Index i corresponds to Q-position i, which is input at
        # sequence position (N-1)+i, so it produces target's opinion for position N+i.
        # target's opinion for position N+j (j=0..K) = argmax(vlogits[0, j+1])
        #   (j+1 because index 0 is input at position N-1, whose logit predicts position N. Hmm wait.)
        # Actually, index i in vlogits corresponds to Q-position i in the [K+1] input, which is
        # sequence position (N-1)+i. That Q's logit predicts position (N-1)+i+1 = N+i.
        # So target's opinion for position N+j = argmax(vlogits[0, j]) for j in 0..K.
        # (Note: index 0 corresponds to position N-1's input -> logit for position N = "verify" of proposals[0])
        # (Index K corresponds to position N+K-1's input -> logit for position N+K = bonus if all K accepted)
        opinion_slice = vlogits[0, :num_prop + 1]  # only need indices 0..num_prop
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

        # (5) Update draft_opinion for next iter.
        # Same argument as POC v3.1: one draft decode of next_tok at len(tokens)-1 gives
        # draft's opinion for position len(tokens), which is the next iter's first proposal.
        # Also target: no bookkeep call needed -- next iter's verify NEFF rewrites all positions.
        # We track target_opinion via target_opinions but we won't use it directly (verify NEFF
        # handles it internally).
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
    ap.add_argument("--out-file", default="/home/ubuntu/task012_v4a_results.json")
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
        print(f"  greedy-fast:  {gf_time*1000:.0f}ms, {gf_steps} steps, {len(gf_tokens)} tokens", flush=True)

        s_tokens, s_time, s_stats = spec_decode_v4a(target, draft, enc, initial_tokens, eot=eot, log=args.log)
        s_text = tokenizer.decode([t for t in s_tokens if t < 50257])
        print(f"  spec v4.a:    {s_time*1000:.0f}ms, iters={s_stats['iters']}, tokens={len(s_tokens)}", flush=True)

        clip_res["greedy"] = {"tokens": gf_tokens, "text": gf_text, "time_s": gf_time, "steps": gf_steps}
        clip_res["spec"] = {"tokens": s_tokens, "text": s_text, "time_s": s_time, "stats": s_stats, "K": K}
        clip_res["byte_identical"] = (gf_tokens == s_tokens)
        clip_res["speedup"] = gf_time / s_time
        clip_res["acceptance_rate"] = s_stats["accepted"] / s_stats["proposals"] if s_stats["proposals"] > 0 else 0
        print(f"  BI={clip_res['byte_identical']} SU={clip_res['speedup']:.2f}x acc={clip_res['acceptance_rate']:.2f}", flush=True)

        if not clip_res["byte_identical"]:
            print(f"    fast tokens[:30]={gf_tokens[:30]}", flush=True)
            print(f"    spec tokens[:30]={s_tokens[:30]}", flush=True)
            # find first divergence
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
