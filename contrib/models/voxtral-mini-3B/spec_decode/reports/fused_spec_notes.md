# Fused speculative decoding on Voxtral (NxDI): implementation notes

Fused speculative decoding integrated into Voxtral-Mini-3B on Trainium (NxDI 0.10,
SDK 2.31, trn2.3xlarge, TP=4, LNC=2), using the 4-layer distilled draft
`jburtoft/Voxtral-Mini-3B-2507-draft-4layer`.

Results on public benchmarks to be added.

## What made it work

The key change is **on-device audio scatter to BOTH target and draft during
prefill** (edits 4/5 below). The draft was trained on audio-conditioned inputs,
so it only reaches its expected acceptance rate when its KV cache holds the audio
context. Two earlier failure modes: (a) decoding with `LlamaTokenizerFast`
instead of `MistralCommonTokenizer`, and (b) a draft KV cache without audio
context (plain-text prefill gives much lower acceptance).

## The changes

### NxDI source (8 edits against installed NxDI 0.10)

**model_base.py**:
1. `NeuronFusedSpecModel.__init__`: `hidden_size` falls back to `config.text_config.hidden_size` (nested multimodal config).
2. `NeuronFusedSpecModel.__init__`: target worker gets `config.text_config` when nested.
3. `_speculative_token_selection`: `pad_token_id` falls back to `text_config.pad_token_id`.
4. `_context_encoding_forward`: accepts `vision_embeddings`/`vision_mask` and passes them to BOTH target and draft. **This is the critical audio-scatter-to-draft change.**
5. `NeuronFusedSpecModel.forward`: adds `vision_embeddings`/`vision_mask` params after `computed_context_lens`; empty -> None via `torch.numel`; threads them to `_context_encoding_forward`.

**image_to_text_model_base.py**:
6. `__init__`: fused-spec setup (propagate `fused_spec_config` to `text_config`, `_model_cls=NeuronFusedSpecModel`, enable context encoding + fused spec), guard removed. `get_text_builder`: wire `init_custom_process_group_fn`.
7. `_get_model_outputs`: fused-spec branch (base-class version; Voxtral overrides this, so it helps other image-to-text models).

**image_to_text_model_wrapper.py**:
8. `input_generator`: FUSED_SPECULATION branch producing the 13-arg fused layout.

**pixtral/modeling_pixtral.py**: removed the "Pixtral does not yet support fused speculation" guard.

### Voxtral contrib (2 edits in `src/modeling_voxtral.py`)
- `convert_hf_to_neuron_state_dict`: handle a flat config (draft/target sub-model loading passes `text_config` directly).
- `_get_model_outputs`: fused-spec branch (CTE with real audio scatter; TKG -> `fused_spec_model` with empty vision inputs). This is the path that runs for Voxtral (it overrides the base).

### The 13-arg fused-spec positional layout (source of truth: `NeuronFusedSpecModel.forward`)
```
0 input_ids, 1 attention_mask, 2 position_ids, 3 seq_ids, 4 sampling_params,
5 prev_hidden, 6 adapter_ids, 7 slot_mapping, 8 active_block_table, 9 num_queries,
10 computed_context_lens, 11 vision_embeddings, 12 vision_mask
```

## How to run
```bash
# Patched NxDI 0.10 source on PYTHONPATH (see ../nxdi_patches/README.md)
PYTHONPATH=/home/ubuntu/nxdi_edit python ../scripts/compile_voxtral_fused_spec.py --stage compile --K 4
PYTHONPATH=/home/ubuntu/nxdi_edit python ../scripts/test_fused_spec_correctness.py --manifest <your manifest.csv>
PYTHONPATH=/home/ubuntu/nxdi_edit python ../scripts/sweep_fused_spec_perf.py --manifest <your manifest.csv>
```
Decode with `MistralCommonTokenizer`.

## Tuning notes
- **Speculation length K**: the sweep script compares K=2..6. Small K leaves
  acceptance unused; large K over-speculates (the draft rarely sustains 5+
  correct tokens in a row), which wastes draft forwards and can introduce
  occasional tail differences at the `max_new_tokens` boundary.
- **ODS** (`is_hidden_dim_shuffled=True`) composes cleanly with fused spec (config flag only).
- **Async mode** (`async_mode=True`) compiles but fails at runtime: the
  image-to-text fused-spec dispatch exists only in the synchronous
  `_get_model_outputs`; `_get_model_outputs_async` has no fused-spec branch.
  Wiring it requires mirroring the fused-spec output handling in the async path.
- Very short clips (a few tokens) do not amortize spec-decode setup.

## Follow-ups
1. Async fused spec for image-to-text (hide Python/launch overhead).
2. Combine with `move_trace_to_device` on the text decoder and CTE bucketing.
3. More draft training data to raise acceptance on long transcripts.
4. Bucketed `n_active_tokens` for the CTE to reduce prefill cost on short clips.
