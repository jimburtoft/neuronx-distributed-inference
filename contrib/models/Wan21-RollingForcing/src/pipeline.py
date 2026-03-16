"""
NxDI Rolling Pipeline — CPU orchestrator for RollingForcing on Neuron (trn2).

This module orchestrates the full 11-window rolling-forcing pipeline using
NxDI applications (self, cached, update) with tensor parallelism (TP=4).

Key design:
  1. Self mode: 5-bucket multi-bucket model.pt (all buckets co-resident)
  2. Cached mode: single-bucket model.pt per frame count (swap between windows)
     - 5 cached NEFFs can't co-reside due to I/O buffer pre-allocation (18.3 GB)
     - Each cached model loaded/unloaded as needed (~45-57s per swap)
  3. Update mode: 1-bucket model.pt (always resident)
  4. KV cache uses TP-sharded heads (heads_per_rank = NUM_HEADS / TP)
  5. RoPE uses real-valued cos/sin (not complex numbers)
  6. Per-layer KV inputs passed as separate tensors (60 total per call)

Usage:
    pipeline = NxDIRollingPipeline(
        self_app=self_app,
        cached_model_paths={3: "/path/cached_f3_tp4", 6: ..., ...},
        update_app=update_app,
        weight_path="/path/to/weights",
        ...
    )
    latents = pipeline.generate(encoder_hidden_states, noise_latents)
"""

import torch
import torch.nn as nn
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from window_schedule import (
    compute_schedule,
    WindowSchedule,
    MainCallSchedule,
    UpdateCallSchedule,
    FRAME_SEQ_LENGTH,
    NUM_FRAME_PER_BLOCK,
    BLOCK_LENGTH,
    MAX_ATTENTION_SIZE,
    KV_CACHE_CAPACITY,
    SINK_TOKENS,
    NUM_HEADS,
    HEAD_DIM,
    NUM_TRANSFORMER_LAYERS,
    WINDOW_NUM,
    NUM_FRAMES,
    NUM_BLOCKS,
    DENOISING_STEPS,
    ROLLING_WINDOW_LENGTH_BLOCKS,
)

from modeling_wan21 import (
    NeuronCausalWanApplication,
    create_causal_wan_config,
    make_freqs,
    precompute_rope_embeddings,
    apply_rope_precomputed,
    PATCH_T,
    PATCH_H,
    PATCH_W,
    TEXT_SEQ_LEN,
    TEXT_DIM,
    IN_CHANNELS,
    DIM,
    NUM_LAYERS,
)


# ─── Flow Match Scheduler ──────────────────────────────────────────────────


class FlowMatchScheduler:
    """
    Flow matching noise scheduler for Wan 2.1 / RollingForcing.

    Implements the shifted linear sigma schedule used by Wan's flow matching formulation.
    Key operations:
      - add_noise(x0, noise, timestep): forward corruption  = (1-sigma)*x0 + sigma*noise
      - get_sigma(timestep): look up sigma for given timestep values
      - step(flow_pred, timestep, xt): one denoising step using flow prediction

    The schedule uses shift=5.0 (from Wan 2.1 config) which maps linear sigmas
    through: sigma_shifted = shift * sigma / (1 + (shift-1) * sigma)

    Args:
        num_train_timesteps: Total training timesteps (1000 for Wan)
        shift: Sigma shift factor (5.0 for Wan 2.1)
        num_inference_steps: Number of inference steps (1000 for full schedule lookup)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        num_inference_steps: int = 1000,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        # Match CPU reference: sigma_min=0.0, extra_one_step=True
        sigma_min = 0.0
        sigma_max = 1.0

        # Build full sigma and timestep schedules
        # extra_one_step: linspace N+1 then drop last
        sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

    def get_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Look up sigma values for given timestep values.

        Args:
            timestep: [N] tensor of timestep values (in [0, 1000] range)

        Returns:
            sigma: [N] tensor of corresponding sigma values
        """
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        return self.sigmas[timestep_id]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: corrupt clean samples with noise.

        Args:
            original_samples: [B, C, H, W] clean latents
            noise: [B, C, H, W] Gaussian noise
            timestep: [B] timestep values

        Returns:
            noisy_samples: [B, C, H, W] = (1-sigma)*x0 + sigma*noise
        """
        sigma = self.get_sigma(timestep).reshape(-1, 1, 1, 1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)

    def convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Convert flow prediction to x0 prediction.

        In flow matching: pred = noise - x0, and x_t = (1-sigma)*x0 + sigma*noise
        Therefore: x0 = x_t - sigma * pred

        Args:
            flow_pred: [B, C, H, W] model output (flow/velocity prediction)
            xt: [B, C, H, W] noisy input
            timestep: [B] timestep values

        Returns:
            x0_pred: [B, C, H, W] predicted clean latent
        """
        original_dtype = flow_pred.dtype
        flow_pred_d = flow_pred.double()
        xt_d = xt.double()
        sigma = self.get_sigma(timestep).double().reshape(-1, 1, 1, 1)
        x0_pred = xt_d - sigma * flow_pred_d
        return x0_pred.to(original_dtype)


class KVCacheManagerTP:
    """
    Manages KV cache state for all transformer layers with TP-sharded heads.

    Cache shape per layer: [B, KV_CACHE_CAPACITY, heads_per_rank, HEAD_DIM]
    The cache stores:
      - Raw (un-normed, un-roped) K for anchor blocks (first BLOCK_LENGTH tokens)
      - Roped K for all other blocks
      - V values for all blocks
    """

    def __init__(
        self,
        batch_size: int,
        tp_degree: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.tp_degree = tp_degree
        self.heads_per_rank = math.ceil(NUM_HEADS / tp_degree) * tp_degree // tp_degree

        # Per-layer caches: [B, KV_CACHE_CAPACITY, heads_per_rank, HEAD_DIM]
        self.cache_k = [
            torch.zeros(
                batch_size,
                KV_CACHE_CAPACITY,
                self.heads_per_rank,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            for _ in range(NUM_TRANSFORMER_LAYERS)
        ]
        self.cache_v = [
            torch.zeros(
                batch_size,
                KV_CACHE_CAPACITY,
                self.heads_per_rank,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            for _ in range(NUM_TRANSFORMER_LAYERS)
        ]

        # Track cache fill levels
        self.global_end_index = 0
        self.local_end_index = 0

    def reset(self):
        for layer in range(NUM_TRANSFORMER_LAYERS):
            self.cache_k[layer].zero_()
            self.cache_v[layer].zero_()
        self.global_end_index = 0
        self.local_end_index = 0

    def write_block(
        self,
        layer_idx: int,
        local_start: int,
        local_end: int,
        k_block: torch.Tensor,  # [B, BLOCK_LENGTH, heads_per_rank, HEAD_DIM]
        v_block: torch.Tensor,
    ):
        """Write a block of K/V into the cache for one layer."""
        self.cache_k[layer_idx][:, local_start:local_end] = k_block
        self.cache_v[layer_idx][:, local_start:local_end] = v_block

    def assemble_kv_buffer_cached(
        self,
        layer_idx: int,
        schedule: MainCallSchedule,
        anchor_roped_k: torch.Tensor,  # [B, BLOCK_LENGTH, heads_per_rank, HEAD_DIM]
        anchor_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Assemble padded KV buffer for Path B (cached) attention.

        Buffer layout: [anchor(4680) | working(variable) | padding | current_space(input_len)]
        Total length = MAX_ATTENTION_SIZE.

        The model writes its own computed K/V into the last input_len positions.
        The attention mask unmasks anchor + working + current, masks padding.

        Returns:
            kv_k: [B, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM]
            kv_v: [B, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM]
            valid_len: number of filled cache positions (anchor + working)
        """
        B = self.batch_size
        kv_k = torch.zeros(
            B,
            MAX_ATTENTION_SIZE,
            self.heads_per_rank,
            HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        kv_v = torch.zeros(
            B,
            MAX_ATTENTION_SIZE,
            self.heads_per_rank,
            HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )

        # 1. Anchor block (re-roped)
        pos = 0
        kv_k[:, pos : pos + BLOCK_LENGTH] = anchor_roped_k
        kv_v[:, pos : pos + BLOCK_LENGTH] = anchor_v
        pos += BLOCK_LENGTH

        # 2. Working cache
        if (
            schedule.working_cache_length is not None
            and schedule.working_cache_length > 0
        ):
            wc_start = schedule.extract_cache_start
            wc_end = schedule.extract_cache_end
            kv_k[:, pos : pos + schedule.working_cache_length] = self.cache_k[
                layer_idx
            ][:, wc_start:wc_end]
            kv_v[:, pos : pos + schedule.working_cache_length] = self.cache_v[
                layer_idx
            ][:, wc_start:wc_end]
            pos += schedule.working_cache_length

        # Note: current input K/V are NOT included here — they're computed inside the model.
        # The attn_mask handles telling the model which positions are valid.
        valid_len = pos

        return kv_k, kv_v, valid_len

    def assemble_kv_buffer_update(
        self,
        layer_idx: int,
        schedule: UpdateCallSchedule,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Assemble padded KV buffer for Path C (update) attention.

        Layout: [cache_slice(variable) | zeros(padding)]
        Total padded to MAX_ATTENTION_SIZE.

        Returns:
            kv_k: [B, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM]
            kv_v: [B, MAX_ATTENTION_SIZE, heads_per_rank, HEAD_DIM]
            valid_len: number of valid KV positions
        """
        B = self.batch_size
        kv_k = torch.zeros(
            B,
            MAX_ATTENTION_SIZE,
            self.heads_per_rank,
            HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )
        kv_v = torch.zeros(
            B,
            MAX_ATTENTION_SIZE,
            self.heads_per_rank,
            HEAD_DIM,
            dtype=self.dtype,
            device=self.device,
        )

        cs = schedule.extract_cache_start
        ce = schedule.extract_cache_end
        valid_len = ce - cs

        kv_k[:, :valid_len] = self.cache_k[layer_idx][:, cs:ce]
        kv_v[:, :valid_len] = self.cache_v[layer_idx][:, cs:ce]

        return kv_k, kv_v, valid_len

    def update_indices(self, global_end: int, local_end: int):
        self.global_end_index = global_end
        self.local_end_index = local_end


class NxDIRollingPipeline:
    """
    CPU-side orchestrator for RollingForcing using NxDI applications.

    HBM constraint: only ONE model can be loaded at a time on 4 NeuronCores
    (self 5-bucket = 19 GB, no room for cached or update NEFFs alongside).

    Strategy:
      Phase 1 (windows 0-4): Load self model, run main + update (both path A)
      Phase 2 (windows 5-10): For each window:
        - Load cached_fN, run main call (path B), unload cached
        - Load update, run update call (path C), unload update

    The pipeline manages:
      1. KV cache across windows (CPU-side)
      2. RoPE computation (CPU-side, real-valued cos/sin)
      3. Attention mask construction (CPU-side)
      4. Noise schedule (CPU-side)
      5. Model loading/unloading per phase
    """

    def __init__(
        self,
        self_model_path: str,  # Path to self-mode 5-bucket model.pt directory
        cached_model_paths: Dict[int, str],  # {frame_count: "/path/to/cached_fN_tp4"}
        update_model_path: str,  # Path to update-mode 1-bucket model.pt directory
        weight_path: str,  # Path to HF/diffusers weights
        tp_degree: int = 4,
        height: int = 480,
        width: int = 832,
        dtype: torch.dtype = torch.bfloat16,
        denoising_steps: list = None,
    ):
        self.self_model_path = self_model_path
        self.cached_model_paths = cached_model_paths
        self.update_model_path = update_model_path
        self.weight_path = weight_path
        self.tp_degree = tp_degree
        self.dtype = dtype

        # Currently loaded model state
        self._current_app = None
        self._current_app_name = None

        # Resolution
        self.height = height
        self.width = width
        self.lat_h = height // 8
        self.lat_w = width // 8
        self.post_h = self.lat_h // PATCH_H
        self.post_w = self.lat_w // PATCH_W

        # Noise schedule
        if denoising_steps is None:
            denoising_steps = [1000, 800, 600, 400, 200]
        self.denoising_steps = denoising_steps

        # Pre-compute schedule and RoPE
        self.schedule = compute_schedule()
        self.base_freqs = make_freqs(HEAD_DIM)
        self._precompute_all_rope()

    def _unload_current(self):
        """Unload the currently loaded model, releasing NeuronCores."""
        if self._current_app is not None:
            print(f"  Unloading {self._current_app_name}...")
            del self._current_app
            self._current_app = None
            self._current_app_name = None
            import gc

            gc.collect()

    def _load_self(self):
        """Load the self-mode 5-bucket model."""
        if self._current_app_name == "self":
            return
        self._unload_current()

        print(f"  Loading self model from {self.self_model_path}...")
        t0 = time.time()

        config = create_causal_wan_config(
            mode="self",
            tp_degree=self.tp_degree,
            height=self.height,
            width=self.width,
        )
        app = NeuronCausalWanApplication(
            model_path=self.weight_path,
            config=config,
        )
        app.load(self.self_model_path)

        self._current_app = app
        self._current_app_name = "self"
        print(f"  Loaded self model in {time.time() - t0:.1f}s")

    def _load_cached(self, num_frames: int):
        """Load a single cached-mode model for the given frame count."""
        name = f"cached_f{num_frames}"
        if self._current_app_name == name:
            return
        self._unload_current()

        if num_frames not in self.cached_model_paths:
            raise ValueError(
                f"No cached model path for frames={num_frames}. "
                f"Available: {list(self.cached_model_paths.keys())}"
            )

        model_path = self.cached_model_paths[num_frames]
        print(f"  Loading {name} from {model_path}...")
        t0 = time.time()

        config = create_causal_wan_config(
            mode="cached",
            tp_degree=self.tp_degree,
            height=self.height,
            width=self.width,
        )
        app = NeuronCausalWanApplication(
            model_path=self.weight_path,
            config=config,
        )
        app.model.frame_counts = [num_frames]
        app.load(model_path)

        self._current_app = app
        self._current_app_name = name
        print(f"  Loaded {name} in {time.time() - t0:.1f}s")

    def _load_update(self):
        """Load the update-mode 1-bucket model."""
        if self._current_app_name == "update":
            return
        self._unload_current()

        print(f"  Loading update model from {self.update_model_path}...")
        t0 = time.time()

        config = create_causal_wan_config(
            mode="update",
            tp_degree=self.tp_degree,
            height=self.height,
            width=self.width,
        )
        app = NeuronCausalWanApplication(
            model_path=self.weight_path,
            config=config,
        )
        app.load(self.update_model_path)

        self._current_app = app
        self._current_app_name = "update"
        print(f"  Loaded update model in {time.time() - t0:.1f}s")

    def _precompute_all_rope(self):
        """Pre-compute all RoPE cos/sin tensors for every call in the schedule."""
        self.rope_cache = {}

        for ws in self.schedule:
            mc = ws.main_call
            uc = ws.update_call

            # Main call: current input RoPE
            post_f = mc.num_input_frames // PATCH_T
            key_main = (post_f, mc.current_start_frame_for_rope)
            if key_main not in self.rope_cache:
                cos, sin = precompute_rope_embeddings(
                    self.base_freqs,
                    post_f,
                    self.post_h,
                    self.post_w,
                    start_frame=mc.current_start_frame_for_rope,
                )
                self.rope_cache[key_main] = (cos.float(), sin.float())

            # Main call: anchor re-RoPE (Path B)
            if mc.attention_path == "B" and mc.rope_start_frame_anchor is not None:
                post_f_anchor = NUM_FRAME_PER_BLOCK // PATCH_T
                key_anchor = (post_f_anchor, mc.rope_start_frame_anchor)
                if key_anchor not in self.rope_cache:
                    cos, sin = precompute_rope_embeddings(
                        self.base_freqs,
                        post_f_anchor,
                        self.post_h,
                        self.post_w,
                        start_frame=mc.rope_start_frame_anchor,
                    )
                    self.rope_cache[key_anchor] = (cos.float(), sin.float())

            # Update call: current input RoPE
            post_f_u = uc.num_input_frames // PATCH_T
            key_update = (post_f_u, uc.current_start_frame_for_rope)
            if key_update not in self.rope_cache:
                cos, sin = precompute_rope_embeddings(
                    self.base_freqs,
                    post_f_u,
                    self.post_h,
                    self.post_w,
                    start_frame=uc.current_start_frame_for_rope,
                )
                self.rope_cache[key_update] = (cos.float(), sin.float())

            # Update call: anchor re-RoPE (Path C)
            if uc.attention_path == "C" and uc.rope_start_frame_anchor is not None:
                post_f_anchor_u = NUM_FRAME_PER_BLOCK // PATCH_T
                key_anchor_u = (post_f_anchor_u, uc.rope_start_frame_anchor)
                if key_anchor_u not in self.rope_cache:
                    cos, sin = precompute_rope_embeddings(
                        self.base_freqs,
                        post_f_anchor_u,
                        self.post_h,
                        self.post_w,
                        start_frame=uc.rope_start_frame_anchor,
                    )
                    self.rope_cache[key_anchor_u] = (cos.float(), sin.float())

        print(f"Pre-computed {len(self.rope_cache)} distinct RoPE tensors")

    def _make_attn_mask(self, query_len, valid_kv_len):
        """Create additive attention mask for SDPA."""
        mask = torch.zeros(1, 1, query_len, MAX_ATTENTION_SIZE, dtype=self.dtype)
        mask[:, :, :, :valid_kv_len] = 0.0
        mask[:, :, :, valid_kv_len:] = torch.finfo(self.dtype).min
        return mask

    def _call_self_model(
        self,
        hidden_states: torch.Tensor,  # [B, C, F, H, W]
        timestep: torch.Tensor,  # [B]
        encoder_hidden_states: torch.Tensor,  # [B, T, text_dim]
        mc: MainCallSchedule,
    ):
        """Dispatch to self-attention model (Path A). Loads model if needed."""
        self._load_self()

        post_f = mc.num_input_frames // PATCH_T
        rope_cos, rope_sin = self.rope_cache[(post_f, mc.current_start_frame_for_rope)]

        with torch.no_grad():
            outputs = self._current_app(
                hidden_states, timestep, encoder_hidden_states, rope_cos, rope_sin
            )

        # outputs = (video_output, raw_k_0, v_0, roped_k_0, ..., raw_k_29, v_29, roped_k_29)
        return outputs

    def _call_cached_model(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        mc: MainCallSchedule,
        kv_cache: KVCacheManagerTP,
    ):
        """Dispatch to cached attention model (Path B). Loads model if needed."""
        # Load the correct cached model for this frame count
        self._load_cached(mc.num_input_frames)

        post_f = mc.num_input_frames // PATCH_T
        rope_cos, rope_sin = self.rope_cache[(post_f, mc.current_start_frame_for_rope)]

        # Get anchor re-RoPE
        post_f_anchor = NUM_FRAME_PER_BLOCK // PATCH_T
        anchor_cos, anchor_sin = self.rope_cache[
            (post_f_anchor, mc.rope_start_frame_anchor)
        ]

        # Assemble per-layer KV buffers
        # Compute valid_len from schedule (same for all layers)
        valid_len = BLOCK_LENGTH  # anchor
        if mc.working_cache_length is not None and mc.working_cache_length > 0:
            valid_len += mc.working_cache_length

        kv_tensors = []
        for layer_idx in range(NUM_TRANSFORMER_LAYERS):
            # Get anchor block (un-roped K) and re-rope it
            anchor_raw_k = kv_cache.cache_k[layer_idx][:, :BLOCK_LENGTH]
            anchor_v = kv_cache.cache_v[layer_idx][:, :BLOCK_LENGTH]

            # Re-rope anchor K
            anchor_roped_k = apply_rope_precomputed(
                anchor_raw_k, anchor_cos, anchor_sin
            )

            # Assemble full KV buffer
            kv_k, kv_v, _ = kv_cache.assemble_kv_buffer_cached(
                layer_idx, mc, anchor_roped_k, anchor_v
            )

            kv_tensors.append(kv_k)
            kv_tensors.append(kv_v)

        # Build attention mask
        # Buffer layout: [anchor | working | padding | current(last input_len positions)]
        # The model writes current K/V at positions [MAX_ATTENTION_SIZE - seq_len, MAX_ATTENTION_SIZE)
        # Mask: attend to anchor + working + current, mask padding
        seq_len = mc.input_seq_len
        prefix_len = MAX_ATTENTION_SIZE - seq_len

        attn_mask = torch.full(
            (1, 1, seq_len, MAX_ATTENTION_SIZE),
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
        )
        attn_mask[:, :, :, :valid_len] = 0.0  # anchor + working
        attn_mask[:, :, :, prefix_len:] = 0.0  # current (model fills K/V here)

        with torch.no_grad():
            outputs = self._current_app(
                hidden_states,
                timestep,
                encoder_hidden_states,
                rope_cos,
                rope_sin,
                attn_mask,
                *kv_tensors,
            )

        return outputs

    def _call_update_model(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        uc: UpdateCallSchedule,
        kv_cache: KVCacheManagerTP,
    ):
        """Dispatch to update attention model (Path C). Loads model if needed."""
        self._load_update()

        post_f = uc.num_input_frames // PATCH_T
        rope_cos, rope_sin = self.rope_cache[(post_f, uc.current_start_frame_for_rope)]

        # Assemble per-layer KV buffers
        kv_tensors = []

        # Compute valid_len from schedule (same for all layers)
        if uc.extract_cache_start is not None and uc.extract_cache_end is not None:
            valid_len = uc.extract_cache_end - uc.extract_cache_start
        else:
            valid_len = BLOCK_LENGTH

        # If anchor needs re-roping, do it before assembly
        if uc.attention_path == "C" and uc.rope_start_frame_anchor is not None:
            post_f_anchor = NUM_FRAME_PER_BLOCK // PATCH_T
            anchor_cos, anchor_sin = self.rope_cache[
                (post_f_anchor, uc.rope_start_frame_anchor)
            ]
        else:
            anchor_cos = anchor_sin = None

        for layer_idx in range(NUM_TRANSFORMER_LAYERS):
            kv_k, kv_v, _ = kv_cache.assemble_kv_buffer_update(layer_idx, uc)

            # Re-rope anchor if needed
            if anchor_cos is not None and uc.extract_cache_start == 0:
                anchor_raw_k = kv_k[:, :BLOCK_LENGTH]
                kv_k[:, :BLOCK_LENGTH] = apply_rope_precomputed(
                    anchor_raw_k, anchor_cos, anchor_sin
                )

            kv_tensors.append(kv_k)
            kv_tensors.append(kv_v)

        # Attention mask
        attn_mask = self._make_attn_mask(BLOCK_LENGTH, valid_len)

        with torch.no_grad():
            outputs = self._current_app(
                hidden_states,
                timestep,
                encoder_hidden_states,
                rope_cos,
                rope_sin,
                attn_mask,
                *kv_tensors,
            )

        return outputs

    def _process_model_output(
        self,
        outputs,
        mc_or_uc,
        kv_cache: KVCacheManagerTP,
        is_update: bool = False,
    ):
        """
        Process model outputs: extract video output and update KV cache.

        Model output format:
          (video_output, raw_k_0, v_0, roped_k_0, ..., raw_k_29, v_29, roped_k_29)

        For cache write:
          - If is_first_block: write raw_k (un-roped) for anchor
          - Otherwise: write roped_k
        """
        video_output = outputs[0]
        cache_tensors = outputs[1:]  # 90 tensors: (raw_k, v, roped_k) × 30 layers

        for layer_idx in range(NUM_TRANSFORMER_LAYERS):
            raw_k = cache_tensors[
                layer_idx * 3
            ]  # [B, BLOCK_LENGTH, heads_per_rank, HEAD_DIM]
            v = cache_tensors[layer_idx * 3 + 1]
            roped_k = cache_tensors[layer_idx * 3 + 2]

            # Choose which K to cache
            if mc_or_uc.is_first_block:
                k_to_cache = raw_k  # Un-roped for anchor
            else:
                k_to_cache = roped_k  # Roped for non-anchor

            kv_cache.write_block(
                layer_idx,
                mc_or_uc.local_start_index,
                mc_or_uc.local_end_index,
                k_to_cache,
                v,
            )

        # Update cache indices
        kv_cache.update_indices(
            mc_or_uc.global_end_index_after,
            mc_or_uc.local_end_index_after,
        )

        return video_output

    def generate(
        self,
        encoder_hidden_states: torch.Tensor,  # [B, T, text_dim]
        noise_latents: torch.Tensor,  # [B, C, num_frames, lat_h, lat_w]
    ) -> torch.Tensor:
        """
        Run the full RollingForcing pipeline.

        Args:
            encoder_hidden_states: text embeddings [B, 512, 4096]
            noise_latents: initial noise [B, 16, 21, lat_h, lat_w]

        Returns:
            denoised_latents: [B, 16, 21, lat_h, lat_w]
        """
        B = noise_latents.shape[0]
        num_frames = noise_latents.shape[2]
        assert num_frames == NUM_FRAMES, (
            f"Expected {NUM_FRAMES} frames, got {num_frames}"
        )

        # Initialize KV cache
        kv_cache = KVCacheManagerTP(
            batch_size=B,
            tp_degree=self.tp_degree,
            dtype=self.dtype,
        )

        # Output buffer: accumulate denoised frames
        denoised = noise_latents.clone()

        # Noise schedule
        denoising_steps = self.denoising_steps

        print(f"Starting rolling-forcing pipeline: {WINDOW_NUM} windows")
        t_start = time.time()

        for ws in self.schedule:
            mc = ws.main_call
            uc = ws.update_call
            t_window = time.time()

            # --- MAIN CALL (denoising) ---
            start_frame = mc.current_start_frame
            end_frame = start_frame + mc.num_input_frames

            # Assemble noisy input for this window
            # In RollingForcing, each block gets a different noise level:
            # blocks closer to the leading edge get more noise
            noisy_input = self._assemble_noisy_input(denoised, mc, noise_latents)

            # Timestep for this window (use the appropriate denoising step)
            # The main call uses the graduated noise timestep
            window_timestep = self._get_window_timestep(mc)
            timestep = torch.tensor([window_timestep], dtype=self.dtype)

            # Dispatch to correct model
            if mc.attention_path == "A":
                outputs = self._call_self_model(
                    noisy_input, timestep, encoder_hidden_states, mc
                )
            else:
                outputs = self._call_cached_model(
                    noisy_input, timestep, encoder_hidden_states, mc, kv_cache
                )

            # Process output and update cache
            video_output = self._process_model_output(outputs, mc, kv_cache)

            # Store denoised output
            denoised[:, :, start_frame:end_frame] = video_output

            # --- UPDATE CALL (cache clean frame) ---
            # Re-run with only the first block of the denoised output at t=0
            update_start = mc.current_start_frame
            update_end = update_start + NUM_FRAME_PER_BLOCK
            update_input = denoised[:, :, update_start:update_end].clone()

            timestep_zero = torch.tensor([0.0], dtype=self.dtype)

            if uc.attention_path == "A":
                outputs = self._call_self_model(
                    update_input, timestep_zero, encoder_hidden_states, uc
                )
            else:
                outputs = self._call_update_model(
                    update_input, timestep_zero, encoder_hidden_states, uc, kv_cache
                )

            # Process update output (updates cache but we don't use the video output)
            _ = self._process_model_output(outputs, uc, kv_cache, is_update=True)

            elapsed = time.time() - t_window
            print(
                f"  Window {ws.window_index}: {mc.attention_path}/{uc.attention_path} "
                f"f{mc.num_input_frames} -> {elapsed:.2f}s"
            )

        total = time.time() - t_start
        print(f"Rolling loop completed in {total:.1f}s")

        return denoised

    def _assemble_noisy_input(
        self,
        denoised: torch.Tensor,
        mc: MainCallSchedule,
        noise_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assemble the noisy input for a window's main call.

        In RollingForcing, each block in the window gets a graduated noise level:
        - Block closest to output (newest): highest noise (denoising_steps[0])
        - Block at trailing edge: lowest noise (denoising_steps[-1])
        - Already-denoised blocks: no noise (t=0)

        For simplicity in this initial implementation, we use uniform noise.
        TODO: Implement graduated noise schedule per block.
        """
        start_frame = mc.current_start_frame
        num_frames = mc.num_input_frames
        end_frame = start_frame + num_frames

        # Take the current state of the latents for this window
        window_latents = denoised[:, :, start_frame:end_frame].clone()

        return window_latents

    def _get_window_timestep(self, mc: MainCallSchedule) -> float:
        """Get the timestep for a window's main call."""
        # In RollingForcing, the timestep varies per block within the window.
        # For now, use a representative timestep.
        # TODO: Implement per-block timestep scheduling.
        #
        # The actual schedule from rolling_forcing_dmd.yaml:
        # denoising_step_list = [1000, 800, 600, 400, 200]
        # Each block in the window gets a different timestep based on its position.
        #
        # For the initial implementation, we use the middle timestep.
        return self.denoising_steps[len(self.denoising_steps) // 2]


# ─── Unified Pipeline (no model swapping) ─────────────────────────────────


class NxDIUnifiedRollingPipeline:
    """
    CPU-side orchestrator using the 2-NEFF unified model (self + cached).

    All modes co-resident on NeuronCores — NO model swapping needed.
    Update calls go through the cached NEFF with padded f15 input.

    Self-mode inputs with fewer than 15 frames are padded to f15 and
    an attention mask zeros out the padding tokens.

    Expected performance: ~50-80s total (vs 700s with model swapping).
    """

    def __init__(
        self,
        unified_app,  # NeuronCausalWanUnifiedApplication (loaded)
        tp_degree: int = 4,
        height: int = 480,
        width: int = 832,
        dtype: torch.dtype = torch.bfloat16,
        denoising_steps: list = None,
    ):
        self.app = unified_app
        self.tp_degree = tp_degree
        self.dtype = dtype

        # Resolution
        self.height = height
        self.width = width
        self.lat_h = height // 8
        self.lat_w = width // 8
        self.post_h = self.lat_h // PATCH_H
        self.post_w = self.lat_w // PATCH_W

        # TP-related
        self.heads_per_rank = math.ceil(NUM_HEADS / tp_degree) * tp_degree // tp_degree

        # --- Flow Match Scheduler and Graduated Noise Schedule ---
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=5.0,
            num_inference_steps=1000,
        )

        # Raw denoising step list (from config)
        if denoising_steps is None:
            denoising_steps = [1000, 800, 600, 400, 200]
        self.denoising_steps_raw = denoising_steps

        # Warp denoising steps through the scheduler's timestep schedule
        # Reference: timesteps[1000 - step] where timesteps has 1000 entries + [0]
        scheduler_timesteps = torch.cat((self.scheduler.timesteps, torch.tensor([0.0])))
        self.denoising_step_list = torch.tensor(
            [scheduler_timesteps[1000 - step].item() for step in denoising_steps],
            dtype=torch.float32,
        )
        print(f"Warped denoising steps: {self.denoising_step_list.tolist()}")

        # Build shared_timestep tensor [1, rolling_window_frames]
        # Layout: [block_0(cleanest) | block_1 | ... | block_N-1(noisiest)]
        # From reference: reversed(denoising_step_list) → from clean to noisy
        num_blocks_per_window = len(denoising_steps)  # 5
        window_frames = num_blocks_per_window * NUM_FRAME_PER_BLOCK  # 5*3=15
        self.shared_timestep = torch.ones(1, window_frames, dtype=torch.float32)
        for index, current_timestep in enumerate(reversed(self.denoising_step_list)):
            start = index * NUM_FRAME_PER_BLOCK
            end = (index + 1) * NUM_FRAME_PER_BLOCK
            self.shared_timestep[:, start:end] *= current_timestep.item()
        print(
            f"Shared timestep per block: {[self.shared_timestep[0, i * NUM_FRAME_PER_BLOCK].item() for i in range(num_blocks_per_window)]}"
        )

        self.denoising_steps = denoising_steps  # keep for backward compat
        self.num_denoising_steps = len(denoising_steps)

        # Pre-compute schedule and RoPE
        self.schedule = compute_schedule()
        self.base_freqs = make_freqs(HEAD_DIM)

        # Pre-compute the f15 constants for padding (BEFORE _precompute_all_rope)
        self.f15_post_f = 15 // PATCH_T
        self.f15_seq_len = self.f15_post_f * self.post_h * self.post_w  # 23400

        self._precompute_all_rope()

    def _precompute_all_rope(self):
        """Pre-compute all RoPE cos/sin tensors for every call in the schedule."""
        self.rope_cache = {}

        for ws in self.schedule:
            mc = ws.main_call
            uc = ws.update_call

            # Main call: current input RoPE
            post_f = mc.num_input_frames // PATCH_T
            key_main = (post_f, mc.current_start_frame_for_rope)
            if key_main not in self.rope_cache:
                cos, sin = precompute_rope_embeddings(
                    self.base_freqs,
                    post_f,
                    self.post_h,
                    self.post_w,
                    start_frame=mc.current_start_frame_for_rope,
                )
                self.rope_cache[key_main] = (cos.float(), sin.float())

            # Main call: anchor re-RoPE (Path B)
            if mc.attention_path == "B" and mc.rope_start_frame_anchor is not None:
                post_f_anchor = NUM_FRAME_PER_BLOCK // PATCH_T
                key_anchor = (post_f_anchor, mc.rope_start_frame_anchor)
                if key_anchor not in self.rope_cache:
                    cos, sin = precompute_rope_embeddings(
                        self.base_freqs,
                        post_f_anchor,
                        self.post_h,
                        self.post_w,
                        start_frame=mc.rope_start_frame_anchor,
                    )
                    self.rope_cache[key_anchor] = (cos.float(), sin.float())

            # Update call: current input RoPE
            post_f_u = uc.num_input_frames // PATCH_T
            key_update = (post_f_u, uc.current_start_frame_for_rope)
            if key_update not in self.rope_cache:
                cos, sin = precompute_rope_embeddings(
                    self.base_freqs,
                    post_f_u,
                    self.post_h,
                    self.post_w,
                    start_frame=uc.current_start_frame_for_rope,
                )
                self.rope_cache[key_update] = (cos.float(), sin.float())

            # Update call: anchor re-RoPE (Path C)
            if uc.attention_path == "C" and uc.rope_start_frame_anchor is not None:
                post_f_anchor_u = NUM_FRAME_PER_BLOCK // PATCH_T
                key_anchor_u = (post_f_anchor_u, uc.rope_start_frame_anchor)
                if key_anchor_u not in self.rope_cache:
                    cos, sin = precompute_rope_embeddings(
                        self.base_freqs,
                        post_f_anchor_u,
                        self.post_h,
                        self.post_w,
                        start_frame=uc.rope_start_frame_anchor,
                    )
                    self.rope_cache[key_anchor_u] = (cos.float(), sin.float())

        # Also pre-compute f15 RoPE for ALL calls (for padding)
        # Each call may have a different start_frame_for_rope
        for ws in self.schedule:
            for call in [ws.main_call, ws.update_call]:
                key_f15 = (self.f15_post_f, call.current_start_frame_for_rope)
                if key_f15 not in self.rope_cache:
                    cos, sin = precompute_rope_embeddings(
                        self.base_freqs,
                        self.f15_post_f,
                        self.post_h,
                        self.post_w,
                        start_frame=call.current_start_frame_for_rope,
                    )
                    self.rope_cache[key_f15] = (cos.float(), sin.float())

        print(f"Pre-computed {len(self.rope_cache)} distinct RoPE tensors")

    def _pad_to_f15(self, hidden_states, num_frames):
        """Pad hidden_states from num_frames to 15 frames."""
        if num_frames == 15:
            return hidden_states
        B, C, F, H, W = hidden_states.shape
        padded = torch.zeros(B, C, 15, H, W, dtype=hidden_states.dtype)
        padded[:, :, :F] = hidden_states
        return padded

    def _self_mask_padded(self, real_seq_len):
        """Create self-attention mask for padded f15 input.

        Real tokens attend to all real tokens; padding tokens are fully masked.
        Mask shape: [1, 1, f15_seq_len, f15_seq_len]
        """
        f15_len = self.f15_seq_len
        mask = torch.full(
            (1, 1, f15_len, f15_len),
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
        )
        # Real tokens attend to all real tokens
        mask[:, :, :real_seq_len, :real_seq_len] = 0.0
        return mask

    def _pad_timestep_to_f15(self, timestep, num_frames):
        """Pad a per-frame timestep tensor from [B, num_frames] to [B, 15].

        Padding frames get timestep=0 (clean/zero noise).
        """
        if num_frames == 15:
            return timestep
        B = timestep.shape[0]
        padded = torch.zeros(B, 15, dtype=timestep.dtype)
        padded[:, :num_frames] = timestep
        return padded

    def _call_self_unified(self, hidden_states, timestep, enc, mc_or_uc):
        """Call self mode through the cached NEFF with zero KV buffers.

        Single-NEFF architecture: all calls go through the cached NEFF.
        Self-attention is achieved by passing zero KV buffers with a mask
        that only unmasks the current (f15) positions.

        Args:
            hidden_states: [B, C, F, H, W] input latents
            timestep: [B, F] per-frame timestep values
            enc: [B, T, text_dim] encoder hidden states
            mc_or_uc: schedule object with RoPE info
        """
        num_frames = hidden_states.shape[2]
        post_f = num_frames // PATCH_T
        real_seq_len = post_f * self.post_h * self.post_w

        # Pad input to f15
        hidden_padded = self._pad_to_f15(hidden_states, num_frames)

        # Pad timestep to [B, 15]
        timestep_padded = self._pad_timestep_to_f15(timestep, num_frames)

        # RoPE for f15 (always use f15 post_f for padded input)
        rope_cos, rope_sin = self.rope_cache[
            (self.f15_post_f, mc_or_uc.current_start_frame_for_rope)
        ]

        # Build cached-style attention mask: [1, 1, f15_seq, MAX_ATTENTION_SIZE]
        # Only unmask current positions at [prefix_len : prefix_len + real_seq_len]
        # where prefix_len = MAX_ATTENTION_SIZE - f15_seq_len
        prefix_len = MAX_ATTENTION_SIZE - self.f15_seq_len
        attn_mask = torch.full(
            (1, 1, self.f15_seq_len, MAX_ATTENTION_SIZE),
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
        )
        # Unmask current positions (the model writes K/V here)
        attn_mask[:, :, :real_seq_len, prefix_len : prefix_len + real_seq_len] = 0.0
        # Padding Q rows (if num_frames < 15) stay fully masked

        # Zero KV buffers for all layers
        kv_tensors = []
        for _ in range(NUM_TRANSFORMER_LAYERS):
            kv_tensors.append(
                torch.zeros(
                    1,
                    MAX_ATTENTION_SIZE,
                    self.heads_per_rank,
                    HEAD_DIM,
                    dtype=self.dtype,
                )
            )
            kv_tensors.append(
                torch.zeros(
                    1,
                    MAX_ATTENTION_SIZE,
                    self.heads_per_rank,
                    HEAD_DIM,
                    dtype=self.dtype,
                )
            )

        with torch.no_grad():
            outputs = self.app.forward_cached(
                hidden_padded,
                timestep_padded,
                enc,
                rope_cos,
                rope_sin,
                attn_mask,
                *kv_tensors,
            )

        # Unpad output: extract first real_seq_len from video output
        video_output = outputs[0]
        if num_frames < 15:
            video_output = video_output[:, :, :num_frames]

        # Cache tensors
        cache_tensors = outputs[1:]

        return (video_output,) + tuple(cache_tensors)

    def _call_cached_unified(self, hidden_states, timestep, enc, mc, kv_cache):
        """Call cached mode through unified app. Pads to f15 if needed.

        Args:
            hidden_states: [B, C, F, H, W] input latents
            timestep: [B, F] per-frame timestep values
            enc: [B, T, text_dim] encoder hidden states
            mc: MainCallSchedule or UpdateCallSchedule
            kv_cache: KVCacheManagerTP
        """
        num_frames = mc.num_input_frames
        post_f = num_frames // PATCH_T
        real_seq_len = post_f * self.post_h * self.post_w

        # Pad input to f15
        hidden_padded = self._pad_to_f15(hidden_states, num_frames)

        # Pad timestep to [B, 15]
        timestep_padded = self._pad_timestep_to_f15(timestep, num_frames)

        # RoPE for f15 (padded)
        rope_cos, rope_sin = self.rope_cache[
            (self.f15_post_f, mc.current_start_frame_for_rope)
        ]

        # Get anchor re-RoPE
        post_f_anchor = NUM_FRAME_PER_BLOCK // PATCH_T
        anchor_cos, anchor_sin = self.rope_cache[
            (post_f_anchor, mc.rope_start_frame_anchor)
        ]

        # Compute valid_len from schedule
        valid_len = BLOCK_LENGTH  # anchor
        if mc.working_cache_length is not None and mc.working_cache_length > 0:
            valid_len += mc.working_cache_length

        # Assemble per-layer KV buffers
        kv_tensors = []
        for layer_idx in range(NUM_TRANSFORMER_LAYERS):
            anchor_raw_k = kv_cache.cache_k[layer_idx][:, :BLOCK_LENGTH]
            anchor_v = kv_cache.cache_v[layer_idx][:, :BLOCK_LENGTH]
            anchor_roped_k = apply_rope_precomputed(
                anchor_raw_k, anchor_cos, anchor_sin
            )
            kv_k, kv_v, _ = kv_cache.assemble_kv_buffer_cached(
                layer_idx, mc, anchor_roped_k, anchor_v
            )
            kv_tensors.append(kv_k)
            kv_tensors.append(kv_v)

        # Build attention mask for padded f15
        # Buffer layout: [anchor | working | padding_kv | current(last f15_seq positions)]
        # The model writes current K/V at [MAX_ATTENTION_SIZE - f15_seq_len, MAX_ATTENTION_SIZE)
        prefix_len = MAX_ATTENTION_SIZE - self.f15_seq_len

        attn_mask = torch.full(
            (1, 1, self.f15_seq_len, MAX_ATTENTION_SIZE),
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
        )
        # Real Q rows attend to anchor+working and their own positions
        attn_mask[:, :, :real_seq_len, :valid_len] = 0.0  # anchor + working
        attn_mask[:, :, :real_seq_len, prefix_len : prefix_len + real_seq_len] = (
            0.0  # current (real only)
        )
        # Padding Q rows stay fully masked (output discarded)

        with torch.no_grad():
            outputs = self.app.forward_cached(
                hidden_padded,
                timestep_padded,
                enc,
                rope_cos,
                rope_sin,
                attn_mask,
                *kv_tensors,
            )

        # Unpad output
        video_output = outputs[0]
        if num_frames < 15:
            video_output = video_output[:, :, :num_frames]

        cache_tensors = outputs[1:]
        return (video_output,) + tuple(cache_tensors)

    def _call_update_via_cached(self, hidden_states, timestep, enc, uc, kv_cache):
        """Call update mode through the cached NEFF with f3 padded to f15.

        GPU reference behavior (causal_model.py lines 248-262):
          1. The update call computes K/V from the clean input
          2. Writes them into kv_cache at [local_start_index:local_end_index],
             OVERWRITING the main call's noisy K/V at that position
          3. Extracts attention context: kv_cache[extract_start:extract_end]
             where extract_end = local_end_index (INCLUDES the just-written clean K/V)
          4. Q attends to: prior_blocks_KV + update_own_clean_KV

        Neuron adaptation:
          - Pre-assemble prior blocks' KV (excluding current block) at buffer [0:prior_len]
          - The model writes its own K/V at [prefix_len:prefix_len+f15_seq_len]
          - Unmask BOTH the prior KV AND the model's self-written positions
          - This way Q sees: prior_blocks_KV + update_clean_KV (matching GPU)

        Args:
            hidden_states: [B, C, F, H, W] input latents (F=3)
            timestep: [B, F] per-frame timestep values
            enc: [B, T, text_dim] encoder hidden states
            uc: UpdateCallSchedule
            kv_cache: KVCacheManagerTP
        """
        num_frames = uc.num_input_frames  # always 3 for update
        post_f = num_frames // PATCH_T
        real_seq_len = post_f * self.post_h * self.post_w  # 4680

        # Pad input to f15
        hidden_padded = self._pad_to_f15(hidden_states, num_frames)

        # Pad timestep to [B, 15]
        timestep_padded = self._pad_timestep_to_f15(timestep, num_frames)

        # RoPE for f15 (padded)
        rope_cos, rope_sin = self.rope_cache[
            (self.f15_post_f, uc.current_start_frame_for_rope)
        ]

        # Compute prior_len: KV from prior blocks EXCLUDING the current block.
        # GPU reference includes the current block's KV (overwritten with clean KV),
        # but we get the clean KV from the model's self-written positions instead.
        if uc.extract_cache_start is not None and uc.extract_cache_end is not None:
            full_valid_len = uc.extract_cache_end - uc.extract_cache_start
            prior_len = full_valid_len - BLOCK_LENGTH
            if prior_len < 0:
                prior_len = 0
        else:
            prior_len = 0
            full_valid_len = 0

        # Check if anchor needs re-roping
        if uc.attention_path == "C" and uc.rope_start_frame_anchor is not None:
            post_f_anchor = NUM_FRAME_PER_BLOCK // PATCH_T
            anchor_cos, anchor_sin = self.rope_cache[
                (post_f_anchor, uc.rope_start_frame_anchor)
            ]
        else:
            anchor_cos = anchor_sin = None

        # Assemble per-layer KV buffers with ONLY prior blocks' KV
        kv_tensors = []
        for layer_idx in range(NUM_TRANSFORMER_LAYERS):
            kv_k, kv_v, _ = kv_cache.assemble_kv_buffer_update(layer_idx, uc)

            # Re-rope anchor if needed
            if anchor_cos is not None and uc.extract_cache_start == 0:
                anchor_raw_k = kv_k[:, :BLOCK_LENGTH]
                kv_k[:, :BLOCK_LENGTH] = apply_rope_precomputed(
                    anchor_raw_k, anchor_cos, anchor_sin
                )

            # Zero out the current block's main-call KV (at positions prior_len to
            # full_valid_len) to ensure Q only sees prior blocks' KV + its own clean KV
            if prior_len < full_valid_len:
                kv_k[:, prior_len:full_valid_len] = 0.0
                kv_v[:, prior_len:full_valid_len] = 0.0

            kv_tensors.append(kv_k)
            kv_tensors.append(kv_v)

        # Build attention mask
        # The cached NEFF writes its own K/V at [prefix_len : prefix_len + f15_seq_len].
        # We unmask:
        #   1. Prior blocks' KV at [0 : prior_len]
        #   2. Model's own clean KV at [prefix_len : prefix_len + real_seq_len]
        # This matches GPU reference: Q sees prior_blocks_KV + update_clean_KV
        prefix_len = MAX_ATTENTION_SIZE - self.f15_seq_len
        attn_mask = torch.full(
            (1, 1, self.f15_seq_len, MAX_ATTENTION_SIZE),
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
        )
        # Unmask prior blocks' cached KV
        if prior_len > 0:
            attn_mask[:, :, :real_seq_len, :prior_len] = 0.0
        # Unmask model's own K/V (update's clean KV for current block)
        attn_mask[:, :, :real_seq_len, prefix_len : prefix_len + real_seq_len] = 0.0

        with torch.no_grad():
            outputs = self.app.forward_cached(
                hidden_padded,
                timestep_padded,
                enc,
                rope_cos,
                rope_sin,
                attn_mask,
                *kv_tensors,
            )

        # Unpad output: take first 3 frames
        video_output = outputs[0][:, :, :num_frames]
        cache_tensors = outputs[1:]

        return (video_output,) + tuple(cache_tensors)

    def _process_model_output(self, outputs, mc_or_uc, kv_cache, is_update=False):
        """Process model outputs: extract video output and update KV cache."""
        video_output = outputs[0]
        cache_tensors = outputs[1:]  # 90 tensors: (raw_k, v, roped_k) × 30 layers

        for layer_idx in range(NUM_TRANSFORMER_LAYERS):
            raw_k = cache_tensors[layer_idx * 3]
            v = cache_tensors[layer_idx * 3 + 1]
            roped_k = cache_tensors[layer_idx * 3 + 2]

            if mc_or_uc.is_first_block:
                k_to_cache = raw_k
            else:
                k_to_cache = roped_k

            kv_cache.write_block(
                layer_idx,
                mc_or_uc.local_start_index,
                mc_or_uc.local_end_index,
                k_to_cache,
                v,
            )

        kv_cache.update_indices(
            mc_or_uc.global_end_index_after,
            mc_or_uc.local_end_index_after,
        )

        return video_output

    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        noise_latents: torch.Tensor,
        force_self_mode: bool = False,
        force_self_update: bool = False,
        force_self_main: bool = False,
    ) -> torch.Tensor:
        """Run the full RollingForcing pipeline with graduated noise schedule.

        Args:
            force_self_mode: If True, ALL windows use self-attention (Path A)
                instead of cached attention (Path B/C). This disables KV cache
                usage for windows 5-10, useful for diagnosing cached-path bugs.
            force_self_update: If True, UPDATE calls always use self-attention
                (Path A) while MAIN calls use their normal scheduled path.
                Tests whether the update call's cached mode corrupts the cache.
            force_self_main: If True, MAIN calls always use self-attention
                (Path A) while UPDATE calls use their normal scheduled path.
                Tests whether the main call's cached mode is the issue.

        This implements the core RollingForcing algorithm:
        1. Sliding window over video blocks with graduated noise levels
        2. Each block in the window has a different denoising timestep
        3. Model predicts flow (velocity), converted to x0 prediction
        4. After denoising, each block is re-noised to its next cleaner level
        5. Update call caches the first block's KV at timestep=0

        Data layout:
            noise_latents: [B, C, F, H, W] — channel-first (NxDI convention)
            Reference uses: [B, F, C, H, W] — frame-first
            We operate in channel-first throughout, converting only for scheduler ops.

        Args:
            encoder_hidden_states: [B, T, text_dim] text embeddings
            noise_latents: [B, C, F, H, W] initial Gaussian noise (F=21)

        Returns:
            output: [B, C, F, H, W] denoised video latents
        """
        B, C, F, H, W = noise_latents.shape
        assert F == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {F}"

        kv_cache = KVCacheManagerTP(
            batch_size=B,
            tp_degree=self.tp_degree,
            dtype=self.dtype,
        )

        # --- Initialize noisy_cache and output ---
        # noisy_cache: [B, C, F, H, W] — stores re-noised intermediate results
        # output: [B, C, F, H, W] — stores final denoised predictions
        # CRITICAL: Use float32 for pipeline state to avoid bf16 compound error
        # across autoregressive windows. Only cast to bf16 for Neuron model input.
        noisy_cache = torch.zeros(B, C, F, H, W, dtype=torch.float32)
        output = torch.zeros(B, C, F, H, W, dtype=torch.float32)
        noise = noise_latents.float().clone()  # original noise in float32

        # --- Window schedule parameters ---
        num_blocks = NUM_BLOCKS  # 7
        num_denoising_steps = self.num_denoising_steps  # 5
        rolling_window_length_blocks = num_denoising_steps  # 5
        window_num = num_blocks + rolling_window_length_blocks - 1  # 11

        # Pre-compute window boundaries
        window_start_blocks = []
        window_end_blocks = []
        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)

        mode_str = ""
        if force_self_mode:
            mode_str = " [FORCE_SELF_MODE]"
        elif force_self_update:
            mode_str = " [FORCE_SELF_UPDATE]"
        elif force_self_main:
            mode_str = " [FORCE_SELF_MAIN]"
        print(
            f"Starting graduated-noise rolling-forcing pipeline: {window_num} windows"
            + mode_str
        )
        print(f"  Blocks: {num_blocks}, Steps: {num_denoising_steps}")
        print(f"  Warped timesteps: {self.denoising_step_list.tolist()}")
        t_start = time.time()

        enc = encoder_hidden_states

        for window_index in range(window_num):
            t_window = time.time()

            # Get schedule entry for model calls
            ws = self.schedule[window_index]
            mc = ws.main_call
            uc = ws.update_call

            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]  # inclusive
            current_start_frame = start_block * NUM_FRAME_PER_BLOCK
            current_end_frame = (end_block + 1) * NUM_FRAME_PER_BLOCK
            current_num_frames = current_end_frame - current_start_frame

            # --- Assemble noisy input ---
            # Blocks 0 to N-2: from noisy_cache (re-noised outputs from prior windows)
            # Block N-1 (last): fresh Gaussian noise
            if (
                current_num_frames == rolling_window_length_blocks * NUM_FRAME_PER_BLOCK
                or current_start_frame == 0
            ):
                # Full-size window OR ramp-up: take noisy_cache + fresh noise for last block
                noisy_input = torch.cat(
                    [
                        noisy_cache[
                            :,
                            :,
                            current_start_frame : current_end_frame
                            - NUM_FRAME_PER_BLOCK,
                        ],
                        noise[
                            :,
                            :,
                            current_end_frame - NUM_FRAME_PER_BLOCK : current_end_frame,
                        ],
                    ],
                    dim=2,
                )
            else:
                # Ramp-down (end of video): all from noisy_cache
                noisy_input = noisy_cache[:, :, current_start_frame:current_end_frame]

            # --- Compute per-frame timestep for this window ---
            # shared_timestep is [1, 15] for full 5-block window
            # Ramp-up windows (start_frame==0, fewer blocks): use tail of shared_timestep
            # Ramp-down windows (end of video): use head of shared_timestep
            # Full windows: use full shared_timestep
            if current_num_frames == rolling_window_length_blocks * NUM_FRAME_PER_BLOCK:
                current_timestep = self.shared_timestep  # [1, 15]
            elif current_start_frame == 0:
                # Ramp-up: take the LAST current_num_frames from shared_timestep
                current_timestep = self.shared_timestep[:, -current_num_frames:]
            elif current_end_frame == NUM_FRAMES:
                # Ramp-down: take the FIRST current_num_frames from shared_timestep
                current_timestep = self.shared_timestep[:, :current_num_frames]
            else:
                raise ValueError(
                    f"Unexpected window: start={current_start_frame}, "
                    f"end={current_end_frame}, num_frames={current_num_frames}"
                )

            # Keep timestep in float32 for precision in sinusoidal embedding.
            # CRITICAL: bf16 rounds e.g. 558.0→560.0, causing ~10% error in
            # time embeddings. The compiled model accepts float32 timestep input.
            timestep_f32 = current_timestep.expand(B, -1).to(torch.float32)

            # --- MAIN CALL: Denoising ---
            # Cast noisy_input to bf16 for Neuron model (pipeline state is float32)
            noisy_input_bf16 = noisy_input.to(self.dtype)
            use_self = force_self_mode or force_self_main or mc.attention_path == "A"

            if use_self:
                outputs = self._call_self_unified(
                    noisy_input_bf16, timestep_f32, enc, mc
                )
            else:
                outputs = self._call_cached_unified(
                    noisy_input_bf16, timestep_f32, enc, mc, kv_cache
                )

            # Process model output and update KV cache
            flow_pred = self._process_model_output(outputs, mc, kv_cache)
            # flow_pred: [B, C, current_num_frames, H, W]

            # --- Convert flow prediction to x0 ---
            # Reference: x0 = xt - sigma_t * flow_pred (per-frame sigma)
            # Flatten to per-frame for sigma lookup: [B*F, C, H, W]
            # CRITICAL: Use the SAME xt values the model saw (bf16→float32).
            # The GPU reference uses one tensor for both model input and x0 formula.
            # Using the original float32 noisy_input here would introduce a mismatch:
            # the model computed flow_pred based on bf16 xt, but x0 would subtract
            # sigma * flow_pred from a different (float32) xt value.
            B_f = B * current_num_frames
            flow_flat = flow_pred.float().permute(0, 2, 1, 3, 4).reshape(B_f, C, H, W)
            xt_flat = (
                noisy_input_bf16.float().permute(0, 2, 1, 3, 4).reshape(B_f, C, H, W)
            )
            t_flat = current_timestep.expand(B, -1).reshape(B_f)

            denoised_pred_flat = self.scheduler.convert_flow_pred_to_x0(
                flow_flat, xt_flat, t_flat
            )
            # Reshape back to [B, C, F, H, W]
            denoised_pred = denoised_pred_flat.reshape(
                B, current_num_frames, C, H, W
            ).permute(0, 2, 1, 3, 4)

            # Store in output
            output[:, :, current_start_frame:current_end_frame] = denoised_pred

            # --- Re-noise each block for the noisy_cache ---
            with torch.no_grad():
                for block_idx in range(start_block, end_block + 1):
                    # Get this block's timestep value
                    local_block_offset = block_idx - start_block
                    block_frame_start = local_block_offset * NUM_FRAME_PER_BLOCK
                    block_frame_end = (local_block_offset + 1) * NUM_FRAME_PER_BLOCK
                    block_timestep_val = current_timestep[0, block_frame_start].item()

                    # Find which denoising step this corresponds to
                    matches = (
                        self.denoising_step_list - block_timestep_val
                    ).abs() < 1e-4
                    match_indices = torch.nonzero(matches, as_tuple=True)[0]
                    if len(match_indices) == 0:
                        continue

                    block_timestep_index = match_indices[0].item()

                    # If already at the cleanest level (last index), done
                    if block_timestep_index >= len(self.denoising_step_list) - 1:
                        continue

                    # Get next cleaner timestep
                    next_timestep = self.denoising_step_list[block_timestep_index + 1]

                    # Extract this block's denoised prediction
                    block_x0 = denoised_pred[
                        :,
                        :,
                        block_frame_start:block_frame_end,
                    ]  # [B, C, 3, H, W]

                    # Re-noise: add_noise expects [B*F, C, H, W]
                    block_x0_flat = block_x0.permute(0, 2, 1, 3, 4).reshape(
                        B * NUM_FRAME_PER_BLOCK, C, H, W
                    )
                    block_noise = torch.randn_like(block_x0_flat)
                    t_renoise = next_timestep * torch.ones(
                        B * NUM_FRAME_PER_BLOCK, dtype=torch.float32
                    )

                    renoised_flat = self.scheduler.add_noise(
                        block_x0_flat, block_noise, t_renoise
                    )
                    renoised = renoised_flat.reshape(
                        B, NUM_FRAME_PER_BLOCK, C, H, W
                    ).permute(0, 2, 1, 3, 4)

                    # Store in noisy_cache
                    global_frame_start = block_idx * NUM_FRAME_PER_BLOCK
                    global_frame_end = (block_idx + 1) * NUM_FRAME_PER_BLOCK
                    noisy_cache[:, :, global_frame_start:global_frame_end] = renoised

            # --- UPDATE CALL: Cache first block at timestep=0 ---
            # Only cache the first block of the window (3 frames)
            update_start = mc.current_start_frame
            update_end = update_start + NUM_FRAME_PER_BLOCK
            update_input = denoised_pred[:, :, :NUM_FRAME_PER_BLOCK].clone()
            # Cast to bf16 for Neuron model (pipeline state is float32)
            update_input_bf16 = update_input.to(self.dtype)

            # context_noise = 0 → timestep = 0 for all frames (float32 for precision)
            timestep_zero = torch.zeros(B, NUM_FRAME_PER_BLOCK, dtype=torch.float32)

            # Use cached attention for update calls when the schedule says Path C,
            # matching GPU reference behavior. Path A windows (0-4) use self-attention.
            # The GPU reference (causal_model.py:248-262) runs update calls with
            # cache-augmented attention where Q attends to all prior cached K/V.
            if uc.attention_path == "C" and not force_self_mode:
                outputs = self._call_update_via_cached(
                    update_input_bf16, timestep_zero, enc, uc, kv_cache
                )
            else:
                # Path A: self-attention only (windows 0-4), or forced self mode
                outputs = self._call_self_unified(
                    update_input_bf16, timestep_zero, enc, uc
                )

            _ = self._process_model_output(outputs, uc, kv_cache, is_update=True)

            elapsed = time.time() - t_window
            t_vals = [
                f"{current_timestep[0, i * NUM_FRAME_PER_BLOCK].item():.0f}"
                for i in range((end_block - start_block + 1))
            ]
            actual_main_mode = mc.attention_path
            if force_self_mode or force_self_main:
                if mc.attention_path != "A":
                    actual_main_mode = f"A(forced)"
            # Update always uses self-attention now (fix for cache corruption)
            actual_update_mode = "A" if uc.attention_path == "A" else "A(fix)"
            print(
                f"  Window {window_index}: blocks [{start_block}-{end_block}] "
                f"{actual_main_mode}/{actual_update_mode} "
                f"t=[{','.join(t_vals)}] {elapsed:.2f}s"
            )

        total = time.time() - t_start
        print(f"Graduated-noise rolling loop completed in {total:.1f}s")

        return output
