#!/usr/bin/env python3
"""
Hybrid benchmark: mix NxDI and baseline components to isolate quality issues.

Modes:
  --mode nxdi-primary    : NxDI primary + baseline secondary (test if NxDI primary is OK)
  --mode nxdi-secondary  : Baseline primary + NxDI secondary (test if NxDI secondary is the issue)
  --mode full-nxdi       : Both NxDI (same as benchmark_nxdi.py)
  --mode baseline        : Both baseline (same as run_baseline.py)

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    cd /mnt/models/songgeneration
    export PYTHONPATH="$(pwd)/codeclm/tokenizer/:$(pwd):$(pwd)/codeclm/tokenizer/Flow1dVAE/"
    export TRANSFORMERS_CACHE="$(pwd)/third_party/hub"
    export NEURON_COMPILE_CACHE_URL="/mnt/models/neuron_cache"
    python hybrid_benchmark.py --mode nxdi-primary --output-wav hybrid_nxdi_pri.wav
"""

import os
import sys
import time

sys.path.insert(0, "/mnt/models/songgeneration")

from modeling_songgeneration_prefill import (
    SongGenerationConfig,
    SongGenerationNeuron,
    _NeuronPrimaryTransformer,
    _NeuronFusedSecondary,
    _patch_cuda_to_cpu,
    _NeuronGPT2Model,
    _load_gpt2_weights,
    _remove_all_weight_norm,
    _VAEDecoderWrapper,
    _load_cpu_diffusion_components,
)

from nxdi_transformers import (
    TP_DEGREE,
    NxDIPrimaryTransformer,
    NxDIFusedSecondary,
    _build_primary_sd,
    _build_secondary_sd,
    shard_model_weights,
    init_parallel_state,
    # Legacy TP=1 functions
    convert_primary_weights,
    convert_secondary_weights,
)

import torch
import torch.nn as nn


def compile_hybrid(pipeline, mode):
    """
    Compile with mixed NxDI/baseline components.

    For NxDI components: uses TP=TP_DEGREE with NxDParallelState context.
    Baseline components always use TP=1 as before.

    mode: 'nxdi-primary', 'nxdi-secondary', 'full-nxdi', 'baseline'
    """
    import torch_neuronx
    from neuronx_distributed import ModelBuilder, NxDParallelState

    cfg = pipeline.config
    T_frames = int(cfg.default_duration_sec * 25)

    PREPEND_LEN = 602
    required = PREPEND_LEN + T_frames + 260 + 10
    for candidate in [512, 768, 1024, 1536, 2048, 3072, 4096]:
        if candidate >= required:
            cfg.max_seq_len = candidate
            break

    print(f"[1/5] Loading LeLM model on CPU...")
    lm_model = pipeline._load_lelm_cpu()

    use_nxdi_primary = mode in ("nxdi-primary", "full-nxdi")
    use_nxdi_secondary = mode in ("nxdi-secondary", "full-nxdi")

    # ===================== PRIMARY =====================
    if use_nxdi_primary:
        print(
            f"[2/5] Building NxDI primary ({cfg.primary_layers}L) with TP={TP_DEGREE}..."
        )

        # Build full unsharded state dict first (before NxDParallelState context)
        # We need a temporary model to inspect NeuronConfig, but the real model
        # must be constructed inside NxDParallelState.
        # Strategy: build SD outside context (just key mapping), construct model inside.

        # Construct model and compile inside NxDParallelState
        nxdi_ctx = NxDParallelState(
            world_size=TP_DEGREE, tensor_model_parallel_size=TP_DEGREE
        )
        with nxdi_ctx:
            primary_wrapper = NxDIPrimaryTransformer(lm_model.transformer, cfg)
            primary_wrapper.eval()

            # Build full unsharded weights
            primary_sd = _build_primary_sd(lm_model.transformer, primary_wrapper, cfg)

            # Trace and compile
            builder = ModelBuilder(model=primary_wrapper)
            example_kwargs = {
                "inputs_embeds": torch.randn(cfg.batch_size, 1, cfg.dim),
                "position_ids": torch.zeros(cfg.batch_size, 1, dtype=torch.long),
                "cache_position": torch.tensor([0], dtype=torch.long),
                "attn_mask": pipeline._build_attn_mask(0),
            }
            builder.trace(kwargs=example_kwargs, tag="decode")

            if cfg.prefill_len > 0:
                prefill_positions = torch.arange(cfg.prefill_len, dtype=torch.long)
                prefill_kwargs = {
                    "inputs_embeds": torch.randn(
                        cfg.batch_size, cfg.prefill_len, cfg.dim
                    ),
                    "position_ids": prefill_positions.unsqueeze(0).expand(
                        cfg.batch_size, -1
                    ),
                    "cache_position": prefill_positions,
                    "attn_mask": pipeline._build_attn_mask(prefill_positions),
                }
                builder.trace(kwargs=prefill_kwargs, tag="prefill")

            pipeline._primary_neuron = builder.compile(
                priority_model_key="decode", compiler_args=cfg.compiler_args
            )

        # Shard weights and load to Neuron
        with NxDParallelState(
            world_size=TP_DEGREE, tensor_model_parallel_size=TP_DEGREE
        ):
            primary_sharded = shard_model_weights(primary_sd, primary_wrapper)
        pipeline._primary_neuron.set_weights(primary_sharded)
        pipeline._primary_neuron.to_neuron()

    else:
        print(f"[2/5] Building BASELINE primary ({cfg.primary_layers}L)...")
        primary_wrapper = _NeuronPrimaryTransformer(lm_model.transformer, cfg)
        primary_wrapper.eval()

        builder = ModelBuilder(model=primary_wrapper)
        example_kwargs = {
            "inputs_embeds": torch.randn(cfg.batch_size, 1, cfg.dim),
            "position_ids": torch.zeros(cfg.batch_size, 1, dtype=torch.long),
            "cache_position": torch.tensor([0], dtype=torch.long),
            "attn_mask": pipeline._build_attn_mask(0),
        }
        builder.trace(kwargs=example_kwargs, tag="decode")

        if cfg.prefill_len > 0:
            prefill_positions = torch.arange(cfg.prefill_len, dtype=torch.long)
            prefill_kwargs = {
                "inputs_embeds": torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                "position_ids": prefill_positions.unsqueeze(0).expand(
                    cfg.batch_size, -1
                ),
                "cache_position": prefill_positions,
                "attn_mask": pipeline._build_attn_mask(prefill_positions),
            }
            builder.trace(kwargs=prefill_kwargs, tag="prefill")

        pipeline._primary_neuron = builder.compile(
            priority_model_key="decode", compiler_args=cfg.compiler_args
        )
        pipeline._primary_neuron.set_weights([primary_wrapper.state_dict()])
        pipeline._primary_neuron.to_neuron()

    # ===================== SECONDARY =====================
    if use_nxdi_secondary:
        print(
            f"[3/5] Building NxDI secondary ({cfg.secondary_layers}L) with TP={TP_DEGREE}..."
        )

        nxdi_ctx = NxDParallelState(
            world_size=TP_DEGREE, tensor_model_parallel_size=TP_DEGREE
        )
        with nxdi_ctx:
            secondary_wrapper = NxDIFusedSecondary(
                lm_model.transformer2, lm_model.mlp, lm_model.linears, cfg
            )
            secondary_wrapper.eval()

            # Build full unsharded weights
            secondary_sd = _build_secondary_sd(
                lm_model.transformer2,
                lm_model.mlp,
                lm_model.linears,
                secondary_wrapper,
                cfg,
            )

            builder = ModelBuilder(model=secondary_wrapper)
            example_kwargs = {
                "fused_input2": torch.randn(cfg.batch_size, 1, cfg.dim),
                "primary_hidden": torch.randn(cfg.batch_size, 1, cfg.dim),
                "position_ids": torch.zeros(cfg.batch_size, 1, dtype=torch.long),
                "cache_position": torch.tensor([0], dtype=torch.long),
                "attn_mask": pipeline._build_attn_mask(0),
            }
            builder.trace(kwargs=example_kwargs, tag="decode")

            if cfg.prefill_len > 0:
                prefill_positions = torch.arange(cfg.prefill_len, dtype=torch.long)
                prefill_kwargs = {
                    "fused_input2": torch.randn(
                        cfg.batch_size, cfg.prefill_len, cfg.dim
                    ),
                    "primary_hidden": torch.randn(
                        cfg.batch_size, cfg.prefill_len, cfg.dim
                    ),
                    "position_ids": prefill_positions.unsqueeze(0).expand(
                        cfg.batch_size, -1
                    ),
                    "cache_position": prefill_positions,
                    "attn_mask": pipeline._build_attn_mask(prefill_positions),
                }
                builder.trace(kwargs=prefill_kwargs, tag="prefill")

            pipeline._secondary_neuron = builder.compile(
                priority_model_key="decode", compiler_args=cfg.compiler_args
            )

        # Shard weights and load to Neuron
        with NxDParallelState(
            world_size=TP_DEGREE, tensor_model_parallel_size=TP_DEGREE
        ):
            secondary_sharded = shard_model_weights(secondary_sd, secondary_wrapper)
        pipeline._secondary_neuron.set_weights(secondary_sharded)
        pipeline._secondary_neuron.to_neuron()

    else:
        print(f"[3/5] Building BASELINE secondary ({cfg.secondary_layers}L)...")
        secondary_wrapper = _NeuronFusedSecondary(
            lm_model.transformer2, lm_model.mlp, lm_model.linears, cfg
        )
        secondary_wrapper.eval()

        builder = ModelBuilder(model=secondary_wrapper)
        example_kwargs = {
            "fused_input2": torch.randn(cfg.batch_size, 1, cfg.dim),
            "primary_hidden": torch.randn(cfg.batch_size, 1, cfg.dim),
            "position_ids": torch.zeros(cfg.batch_size, 1, dtype=torch.long),
            "cache_position": torch.tensor([0], dtype=torch.long),
            "attn_mask": pipeline._build_attn_mask(0),
        }
        builder.trace(kwargs=example_kwargs, tag="decode")

        if cfg.prefill_len > 0:
            prefill_positions = torch.arange(cfg.prefill_len, dtype=torch.long)
            prefill_kwargs = {
                "fused_input2": torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                "primary_hidden": torch.randn(cfg.batch_size, cfg.prefill_len, cfg.dim),
                "position_ids": prefill_positions.unsqueeze(0).expand(
                    cfg.batch_size, -1
                ),
                "cache_position": prefill_positions,
                "attn_mask": pipeline._build_attn_mask(prefill_positions),
            }
            builder.trace(kwargs=prefill_kwargs, tag="prefill")

        pipeline._secondary_neuron = builder.compile(
            priority_model_key="decode", compiler_args=cfg.compiler_args
        )
        pipeline._secondary_neuron.set_weights([secondary_wrapper.state_dict()])
        pipeline._secondary_neuron.to_neuron()

    # ===================== GPT2 + VAE (always baseline) =====================
    print("[4/5] Tracing GPT2 diffusion backbone...")
    pipeline._setup_codeclm_paths()
    sys.path.insert(
        0,
        os.path.join(cfg.codeclm_path, "codeclm/tokenizer/Flow1dVAE/models_gpt/models"),
    )
    from gpt2_config import GPT2Config
    from gpt2_rope2_time_new_correct_mask_noncasual_reflow import (
        GPT2Model as OrigGPT2Model,
    )
    from safetensors.torch import load_file

    gpt2_config = GPT2Config(
        n_positions=1000,
        n_layer=16,
        n_head=20,
        n_embd=2200,
        n_inner=4400,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    gpt2_config._attn_implementation = "eager"

    full_sd = load_file(cfg.safetensors_path)
    gpt2_sd = {
        k[len("cfm_wrapper.estimator.") :]: v
        for k, v in full_sd.items()
        if k.startswith("cfm_wrapper.estimator.")
    }
    orig_gpt2 = OrigGPT2Model(gpt2_config)
    orig_gpt2.load_state_dict(gpt2_sd, strict=False)
    orig_gpt2.eval()

    neuron_gpt2 = _NeuronGPT2Model(gpt2_config)
    _load_gpt2_weights(neuron_gpt2, orig_gpt2)
    neuron_gpt2.eval()

    B_diff = cfg.batch_size
    example_inputs = (
        torch.randn(B_diff, T_frames, 2200),
        torch.ones(B_diff, 1, T_frames, T_frames),
        torch.tensor([0.5] * B_diff),
    )
    pipeline._neuron_gpt2 = torch_neuronx.trace(
        neuron_gpt2,
        example_inputs,
        compiler_args=["--auto-cast", "none", "--model-type", "transformer"],
    )

    print("[5/5] Tracing VAE decoder...")
    sys.path.insert(0, os.path.join(cfg.codeclm_path, "codeclm/tokenizer/Flow1dVAE"))
    from tools.get_1dvae_large import get_model as get_vae_model

    vae_config_path = os.path.join(
        os.path.dirname(cfg.safetensors_path), "../vae/stable_audio_1920_vae.json"
    )
    vae_weights_path = os.path.join(
        os.path.dirname(cfg.safetensors_path), "../vae/autoencoder_music_1320k.ckpt"
    )
    vae = get_vae_model(vae_config_path, vae_weights_path)
    vae.eval()
    _remove_all_weight_norm(vae)

    vae_wrapper = _VAEDecoderWrapper(vae)
    vae_wrapper.eval()
    pipeline._neuron_vae = torch_neuronx.trace(
        vae_wrapper,
        (torch.randn(1, 64, T_frames),),
        compiler_args=["--auto-cast", "matmult"],
    )

    print("[+] Loading CPU diffusion components...")
    (
        pipeline._rvq_vocal,
        pipeline._rvq_bgm,
        pipeline._normfeat,
        pipeline._mask_emb,
        pipeline._zero_cond,
    ) = _load_cpu_diffusion_components(cfg.safetensors_path)

    pipeline._prompt_data = torch.load(
        cfg.prompt_path, map_location="cpu", weights_only=False
    )

    pipeline._compiled = True
    print(f"Hybrid compilation complete (mode={mode}).")


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Hybrid SongGeneration Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["nxdi-primary", "nxdi-secondary", "full-nxdi", "baseline"],
    )
    parser.add_argument(
        "--text", type=str, default="A cheerful pop song with catchy melody"
    )
    parser.add_argument("--genre", type=str, default="Pop")
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--output-wav", type=str, default="hybrid_output.wav")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SongGenerationConfig(
        model_path="/mnt/models/ckpt/songgeneration_base/model.pt",
        config_path="/mnt/models/ckpt/songgeneration_base/config.yaml",
        safetensors_path="/mnt/models/songgeneration/ckpt/model_septoken/model_2.safetensors",
        prompt_path="/mnt/models/songgeneration/ckpt/prompt.pt",
    )

    pipeline = SongGenerationNeuron(config)
    compile_hybrid(pipeline, args.mode)
    pipeline.warmup()

    result = pipeline.generate_timed(
        args.text,
        genre=args.genre,
        duration_sec=args.duration_sec,
        seed=args.seed,
    )

    audio = result["audio"]
    timings = result["timings"]

    print(f"\n{args.mode} Timings:")
    print(f"  LeLM:      {timings['lelm_s']:.1f}s ({timings['lelm_steps']} steps)")
    print(f"  Diffusion: {timings['diffusion_s']:.3f}s")
    print(f"  VAE:       {timings['vae_s']:.3f}s")
    print(f"  Total:     {timings['total_s']:.1f}s")

    try:
        import scipy.io.wavfile

        audio_np = audio.squeeze(0).float().cpu().numpy()
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
        audio_np = audio_np.T
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        scipy.io.wavfile.write(args.output_wav, config.sample_rate, audio_int16)
        print(f"Audio saved to {args.output_wav}")
    except Exception as e:
        print(f"Could not save WAV: {e}")

    # Quick quality check
    try:
        import scipy.io.wavfile

        sr, d = scipy.io.wavfile.read(args.output_wav)
        rms = np.sqrt(np.mean(d.astype(float) ** 2))
        print(f"Audio quality: RMS={rms:.0f}, range=[{d.min()}, {d.max()}]")
        # Baseline reference: RMS ~2855, range ~[-15835, 15951]
        if rms < 1000:
            print("WARNING: Low RMS energy suggests degraded audio quality!")
        else:
            print("Audio energy looks healthy (comparable to baseline)")
    except Exception:
        pass
