# NxDI patches for fused speculative decoding on image-to-text models

These are modified copies of `neuronx_distributed_inference` **0.10.0** source files
that add support for **fused speculative decoding** to `NeuronBaseForImageToText`
models (Voxtral, and by extension Pixtral / other multimodal-to-text models).

Stock NxDI 0.10 explicitly blocks fused speculation for image-to-text models
(`raise ValueError("NeuronBaseForImageToText does not yet support fused speculation")`).
These patches lift that restriction and wire the modality embeddings (Voxtral
audio) through the fused-spec context-encoding path so **both** the target and the
draft get their KV caches conditioned on the audio during prefill -- which is what
lets a draft trained on audio-conditioned inputs reach its real acceptance rate.

## Result

With these patches + the 4-layer distilled draft
(`jburtoft/Voxtral-Mini-3B-2507-draft-4layer`), Voxtral-Mini-3B fused spec-decode
on trn2.3xlarge (K=3, TP=4, SDK 2.31) delivers **1.159x mean speedup** (1.193x on
clips >=40 tokens, 1.39x max) with **17/18 byte-identical** greedy transcripts on
an 18-file TED audio benchmark. See `../reports/task003_working_result.md`.

## Files (modified vs stock NxDI 0.10.0)

| File | Change |
|---|---|
| `models/model_base.py` | `NeuronFusedSpecModel`: nested-config fallbacks (hidden_size, target worker config, pad_token_id); `_context_encoding_forward` + `forward` accept & scatter `vision_embeddings`/`vision_mask` to target AND draft during prefill. |
| `models/image_to_text_model_base.py` | `__init__` fused-spec setup + guard removal; `get_text_builder` draft-group init; `_get_model_outputs` fused-spec branch. |
| `models/image_to_text_model_wrapper.py` | `input_generator` FUSED_SPECULATION_MODEL_TAG case (13-arg fused layout). |
| `models/pixtral/modeling_pixtral.py` | Removed the "Pixtral does not yet support fused speculation" guard. |

## Version note

These are against NxDI **0.10.0** (the version shipped in the SDK 2.31 DLAMI
`aws_neuronx_venv_pytorch_2_9_nxd_inference`). The parent fork's `src/` tree is
0.9.0; do NOT drop these files directly onto 0.9. To use:

1. On an SDK 2.31 instance, copy the installed NxDI into a writable dir:
   ```bash
   cp -r $VENV/lib/python3.12/site-packages/neuronx_distributed_inference /home/ubuntu/nxdi_edit/
   ```
2. Overlay these patched files onto that copy (same relative paths under
   `neuronx_distributed_inference/`).
3. Run with `PYTHONPATH=/home/ubuntu/nxdi_edit` so the patched source overrides
   the installed package.

Or, for a proper upstream contribution, apply the semantic diffs (see the report)
to the current NxDI main.

## The 13-arg fused-spec positional layout

Source of truth = `NeuronFusedSpecModel.forward`:
```
0 input_ids, 1 attention_mask, 2 position_ids, 3 seq_ids, 4 sampling_params,
5 prev_hidden, 6 adapter_ids, 7 slot_mapping, 8 active_block_table, 9 num_queries,
10 computed_context_lens, 11 vision_embeddings, 12 vision_mask
```
The `ImageToTextModelWrapper.input_generator` (trace) and the Voxtral
`_get_model_outputs` (runtime) both produce/consume this exact layout.
