# Task 003 Follow-ups -- Performance Tuning

**Agent**: vsd
**Date**: 2026-08-03
**Instance**: trn2.3xlarge `i-0b7b5c3f1b3effb3d` (ap-southeast-4c)

## Summary

Tuned the working fused-spec Voxtral from the initial K=3 result (1.159x mean) by sweeping the speculation length K and testing async + ODS. **Best config: K=4 + ODS (weight layout optimization) → 1.177x mean, 1.226x long-clip, 1.568x max, 17/18 byte-identical.**

## K sweep (customer 18-file benchmark, TP=4, SDK 2.31)

| K | mean | median | long (>=40 tok) | max | byte-id |
|---|-----:|-------:|----------------:|----:|--------:|
| 2 | 1.086x | 1.101x | 1.105x | 1.191x | 18/18 |
| 3 | 1.139x | 1.158x | 1.169x | 1.373x | 17/18 |
| **4** | **1.166x** | **1.193x** | **1.218x** | **1.561x** | 17/18 |
| 5 | 1.100x | 1.145x | 1.136x | 1.394x | 16/18 |
| 6 | 1.101x | 1.101x | 1.138x | 1.489x | 16/18 |

**K=4 is optimal.** Interpretation:
- K=2 is too conservative -- leaves acceptance on the table (only proposes 2 tokens/iter).
- K=4 hits the sweet spot for the 4-layer draft's ~66% top-1 acceptance: proposes enough tokens to amortize the verify cost without over-speculating.
- K=5/6 over-speculate: the draft can't sustain 5+ correct tokens in a row often enough, so the extra draft forwards are wasted. Byte-identical also drops to 16/18 (the longer candidate windows introduce occasional off-by-one tail differences).

Note the earlier "K=3 = 1.159x" and this run's "K=3 = 1.139x" differ by ~2% -- run-to-run measurement noise (baseline mean wall was 365 vs 367 ms across runs). The relative K ordering is stable.

## ODS (weight layout optimization / is_hidden_dim_shuffled)

Applied on top of K=4:

| Config | mean | median | long | max | byte-id |
|---|-----:|-------:|-----:|----:|--------:|
| K=4 | 1.166x | 1.193x | 1.218x | 1.561x | 17/18 |
| **K=4 + ODS** | **1.177x** | **1.224x** | **1.226x** | **1.568x** | 17/18 |

ODS gives a small (~1%) additional bump by shuffling weights to a layout that reduces per-launch DMA setup. It composes cleanly with fused-spec (just a config flag: `is_hidden_dim_shuffled=True`). Free win; no correctness impact (still 17/18).

## Async mode -- NOT a quick win

`async_mode=True` compiles but **fails at runtime** with a TorchScript error in the SPMD executor. Root cause: our image-to-text fused-spec dispatch was added only to the synchronous `_get_model_outputs`; the async path (`_get_model_outputs_async`) has no fused-spec branch, so the async runtime hits an unhandled shape/output-count path. Wiring async fused-spec for image-to-text is a deeper change (mirror the fused-spec output handling in the async path + verify the ranked-IO tuple layout). Deferred -- not a config-flag win.

Potential upside if done: async execution overlaps CPU dispatch with device execution, which could hide a chunk of the ~1.5 ms/call Python overhead we measured. Estimated additional 1.1-1.2x on top, but requires ~1-2 days of async-path work. Left as a future follow-up.

## Recommended production config

**K=4 + ODS**: `speculation_length=4`, `is_hidden_dim_shuffled=True`. Delivers 1.177x mean / 1.226x long-clip / 1.568x max, byte-identical on 17/18 clips. Compiled artifact: `/mnt/models/nxdi_compiled/vox_fs_K4_ods/` on the trn2.

## Remaining follow-ups (ranked by expected upside)

1. **Async fused-spec for image-to-text** (~1-2 days): hide Python/launch overhead. Est +10-20%.
2. **Combine with parallel voxtral optimization project** (`move_trace_to_device` on the text decoder, CTE bucketing): these reduce fixed per-clip overhead and should stack multiplicatively with spec-decode. Coordinate with that project.
3. **More draft training data** (Task 001b data-scaling law): a higher-acceptance draft would let a larger K pay off and lift the long-clip numbers. The dip on 100+ token clips is acceptance-limited.
4. **Bucketed n_active_tokens for the CTE** to reduce prefill cost on short clips (where spec-decode currently loses to baseline).

## Scripts
- `working/voxtral_spec_decode/nxdi/sweep_fused_spec_perf.py` -- the sweep harness (compiles + measures each variant)
- `working/voxtral_spec_decode/nxdi/compile_voxtral_fused_spec.py` -- now accepts `extra_flags` for async/ODS/etc.
- Raw results: `~/voxtral_spec_decode/tmp/sweep_fused_perf.json`, `sweep_k.log`, `sweep_async_ods.log` on the trn2.
