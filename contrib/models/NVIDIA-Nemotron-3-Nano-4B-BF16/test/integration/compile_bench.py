#!/usr/bin/env python3
"""SDK 2.31 retest driver for Nemotron-3-Nano-4B-BF16 (dense variant).

Compiles at a target context length using the default head-grouped scan (G=8),
then measures TTFT, decode tok/s, TPOT, and quality on the standard capital-of-France prompt.

Usage:
    python compile_bench.py --ctx 128 --seq 2048
    python compile_bench.py --ctx 2048 --seq 4096
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

# Suppress deprecation warnings from NxDI
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from modeling_nemotron_h import (  # noqa: E402
    NeuronNemotronForCausalLM,
    NemotronHInferenceConfig,
    HAS_NKI,
    USE_NKI_SCAN,
    USE_SSD_SCAN,
    SCAN_HEAD_GROUP,
    USE_NATIVE_CONV1D,
    USE_CHUNKED_NKI_SCAN,
)

MODEL_PATH = os.environ.get("NEMOTRON_MODEL_PATH", "/mnt/models/nemotron-4b")

TP_DEGREE = int(os.environ.get("TP_DEGREE", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))


def build_config(ctx: int, seq: int) -> NemotronHInferenceConfig:
    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        max_context_length=ctx,
        seq_len=seq,
        on_device_sampling_config=None,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        torch_dtype="bfloat16",
        save_sharded_checkpoint=True,
    )
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return NemotronHInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )


def compile_or_load(ctx: int, seq: int, compiled_path: Path):
    config = build_config(ctx, seq)

    if not (compiled_path / "model.pt").exists():
        print(f"[compile] target {compiled_path}", flush=True)
        t0 = time.perf_counter()
        model = NeuronNemotronForCausalLM(MODEL_PATH, config)
        model.compile(str(compiled_path))
        compile_s = time.perf_counter() - t0
        print(f"[compile] done in {compile_s:.1f}s", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tokenizer.save_pretrained(str(compiled_path))
        del model
        gc.collect()
    else:
        print(f"[compile] cache hit at {compiled_path}", flush=True)
        compile_s = 0.0

    print("[load] loading compiled model...", flush=True)
    t0 = time.perf_counter()
    model = NeuronNemotronForCausalLM(MODEL_PATH, config)
    model.load(str(compiled_path))
    load_s = time.perf_counter() - t0
    print(f"[load] done in {load_s:.1f}s", flush=True)
    return model, compile_s, load_s


def bench(model, tokenizer, num_gen: int = 50, warmup: int = 1):
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    adapter = HuggingFaceGenerationAdapter(model)

    # Warmup
    for _ in range(warmup):
        adapter.generate(
            inputs.input_ids,
            attention_mask=torch.ones_like(inputs.input_ids),
            max_new_tokens=5,
            do_sample=False,
        )

    # TTFT measurement: generate 1 token
    t0 = time.perf_counter()
    _ = adapter.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=1,
        do_sample=False,
    )
    ttft = (time.perf_counter() - t0) * 1000  # ms

    # Throughput measurement: generate N tokens
    t0 = time.perf_counter()
    outN = adapter.generate(
        inputs.input_ids,
        attention_mask=torch.ones_like(inputs.input_ids),
        max_new_tokens=num_gen,
        do_sample=False,
    )
    total_s = time.perf_counter() - t0
    decode_tokens = num_gen - 1
    decode_s = total_s - (ttft / 1000.0)
    decode_tps = decode_tokens / decode_s if decode_s > 0 else 0.0
    tpot_ms = (decode_s / decode_tokens) * 1000 if decode_tokens > 0 else 0.0

    new_tokens = outN[0, inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    first_tok = tokenizer.decode(outN[0, inputs.input_ids.shape[1]:inputs.input_ids.shape[1]+1])

    return {
        "ttft_ms": ttft,
        "decode_tps": decode_tps,
        "tpot_ms": tpot_ms,
        "first_token": first_tok,
        "output_preview": output_text[:120],
    }


def bench_chat(model, tokenizer, num_gen: int = 50):
    """Coherent-output check via apply_chat_template. The 4B is instruct-tuned."""
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    adapter = HuggingFaceGenerationAdapter(model)
    output = adapter.generate(
        input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=num_gen, do_sample=False,
    )
    gen = output[0][input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--num-gen", type=int, default=50)
    ap.add_argument("--out-dir", default="/mnt/models")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    default_tag = f"nemotron4b_sdk231_hg{SCAN_HEAD_GROUP}_ctx{args.ctx}_seq{args.seq}_bs{BATCH_SIZE}"
    tag = args.tag or default_tag
    compiled_path = Path(args.out_dir) / tag

    print(f"===== 4B SDK 2.31: ctx={args.ctx}, seq={args.seq}, TP={TP_DEGREE}, BS={BATCH_SIZE} =====", flush=True)
    print(f"  HAS_NKI={HAS_NKI}  USE_NKI_SCAN={USE_NKI_SCAN}  USE_SSD_SCAN={USE_SSD_SCAN}", flush=True)
    print(f"  SCAN_HEAD_GROUP={SCAN_HEAD_GROUP}  USE_NATIVE_CONV1D={USE_NATIVE_CONV1D}  USE_CHUNKED_NKI_SCAN={USE_CHUNKED_NKI_SCAN}", flush=True)
    print(f"  compiled_path={compiled_path}", flush=True)

    model, compile_s, load_s = compile_or_load(args.ctx, args.seq, compiled_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[bench] raw prompt...", flush=True)
    r = bench(model, tokenizer, num_gen=args.num_gen)

    print("[bench] chat template...", flush=True)
    try:
        chat_out = bench_chat(model, tokenizer, num_gen=args.num_gen)
    except Exception as e:
        chat_out = f"<chat bench failed: {e}>"

    print("", flush=True)
    print("===== RESULTS =====", flush=True)
    print(f"  ctx={args.ctx} seq={args.seq} bs={BATCH_SIZE}", flush=True)
    print(f"  compile_s={compile_s:.1f}", flush=True)
    print(f"  load_s={load_s:.1f}", flush=True)
    print(f"  TTFT={r['ttft_ms']:.1f} ms", flush=True)
    print(f"  decode={r['decode_tps']:.2f} tok/s", flush=True)
    print(f"  TPOT={r['tpot_ms']:.2f} ms", flush=True)
    print(f"  first_token='{r['first_token']}'", flush=True)
    print(f"  raw_output_preview='{r['output_preview']}'", flush=True)
    print(f"  chat_output='{chat_out[:200]}'", flush=True)


if __name__ == "__main__":
    main()
