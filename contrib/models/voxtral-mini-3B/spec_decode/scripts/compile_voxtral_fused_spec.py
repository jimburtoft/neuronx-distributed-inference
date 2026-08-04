"""Compile fused-spec Voxtral directly via the modified NxDI source (no monkey-patches).

Run with:
    PYTHONPATH=/home/ubuntu/nxdi_edit python compile_voxtral_fused_spec.py --stage compile
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback

CONTRIB_SRC = "/home/ubuntu/nxdi-fork/contrib/models/voxtral-mini-3B/src"
if CONTRIB_SRC not in sys.path:
    sys.path.insert(0, CONTRIB_SRC)

import torch
from huggingface_hub import snapshot_download

from neuronx_distributed_inference.models.config import FusedSpecNeuronConfig

TARGET = "mistralai/Voxtral-Mini-3B-2507"
DRAFT = "jburtoft/Voxtral-Mini-3B-2507-draft-4layer"


def build_config(target_text, draft_text, K, tp=4, batch=1, seq_len=768, n_pos=768, dtype=torch.bfloat16, extra_flags=None):
    from modeling_voxtral import VoxtralInferenceConfig, VoxtralForCausalLM

    with open(os.path.join(target_text, "config.json")) as f:
        target_full_cfg = json.load(f)
    with open(os.path.join(draft_text, "config.json")) as f:
        draft_full_cfg = json.load(f)

    VoxtralForCausalLM._seq_len = seq_len
    VoxtralForCausalLM._n_positions = n_pos
    VoxtralForCausalLM._text_hidden_size = target_full_cfg["hidden_size"]
    VoxtralForCausalLM._batch_size = batch

    # Target PixtralInferenceConfig
    target_cfg = VoxtralInferenceConfig(
        text_model_path=target_text, full_config=target_full_cfg,
        tp_degree=tp, batch_size=batch, seq_len=seq_len, n_positions=n_pos,
        dtype=dtype, on_device_sampling=True,
    ).build()

    # Enable fused-spec on the target (outer + text_config)
    for nc in (target_cfg.neuron_config, target_cfg.text_config.neuron_config):
        nc.enable_fused_speculation = True
        nc.speculation_length = K
        nc.spec_batch_size = batch
        for k, v in (extra_flags or {}).items():
            setattr(nc, k, v)

    # Draft PixtralInferenceConfig -> use its text_config as the draft_config
    draft_cfg = VoxtralInferenceConfig(
        text_model_path=draft_text, full_config=draft_full_cfg,
        tp_degree=tp, batch_size=batch, seq_len=seq_len, n_positions=n_pos,
        dtype=dtype, on_device_sampling=True,
    ).build()
    draft_text_cfg = draft_cfg.text_config
    draft_text_cfg.neuron_config.enable_fused_speculation = False
    draft_text_cfg.neuron_config.speculation_length = 0
    draft_text_cfg.neuron_config.spec_batch_size = batch
    for k, v in (extra_flags or {}).items():
        setattr(draft_text_cfg.neuron_config, k, v)

    from modeling_voxtral import VoxtralTextModel
    fused = FusedSpecNeuronConfig(
        worker_cls=VoxtralTextModel,
        draft_config=draft_text_cfg,
        draft_model_path=os.path.join(draft_text),
    )
    target_cfg.fused_spec_config = fused
    target_cfg.text_config.fused_spec_config = fused
    return target_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["instantiate", "compile"], default="instantiate")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--compiled-dir", default="/mnt/models/nxdi_compiled/voxtral_fused_spec_K3")
    args = p.parse_args()

    from modeling_voxtral import VoxtralForCausalLM
    target_text = os.path.join(snapshot_download(TARGET), "text_only")
    draft_text = os.path.join(snapshot_download(DRAFT), "text_only")

    print(f"[main] Building fused-spec config K={args.K}")
    cfg = build_config(target_text, draft_text, K=args.K)
    print(f"[main] enable_fused_speculation={cfg.neuron_config.enable_fused_speculation}, "
          f"speculation_length={cfg.neuron_config.speculation_length}")

    try:
        print(f"[main] Instantiating VoxtralForCausalLM")
        model = VoxtralForCausalLM(target_text, cfg)
        print(f"[main] Instantiated OK. text_models tags: {[m.tag for m in model.text_models]}")
        print(f"[main] has fused_spec_model: {hasattr(model, 'fused_spec_model')}")
    except Exception as e:
        print(f"[main] INSTANTIATE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    if args.stage == "compile":
        os.makedirs(args.compiled_dir, exist_ok=True)
        print(f"[main] Compiling to {args.compiled_dir}")
        t0 = time.time()
        try:
            model.compile(args.compiled_dir)
            print(f"[main] COMPILE SUCCEEDED in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[main] COMPILE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
