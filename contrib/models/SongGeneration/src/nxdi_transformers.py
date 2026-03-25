# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NxDI-powered transformer wrappers for SongGeneration LeLM.

Replaces the manual attention loop in _NeuronPrimaryTransformer and
_NeuronFusedSecondary with NxDI decoder layers that get NKI kernel
acceleration (attention_tkg_fwd_isa_kernel builtin kernel, etc.)
for free via NeuronAttentionBase and NeuronLlamaMLP.

Usage:
    These classes are drop-in replacements for _NeuronPrimaryTransformer
    and _NeuronFusedSecondary. They accept the same forward() signature
    and return the same outputs, so the generation loop in
    modeling_songgeneration_prefill.py does not need changes.

Architecture:
    - TP_DEGREE=2 (requires LNC=1 on trn2.3xlarge → 8 logical cores)
    - NxDIPrimaryTransformer: 28-layer Llama with NxDI decoder layers + direct KV buffers
    - NxDIFusedSecondary: MLP bridge + 12-layer Llama + per-codebook heads + direct KV buffers
    - Both use register_buffer + setattr for on-device KV cache (ModelBuilder-compatible)
    - Both use NeuronAttentionBase with attn_tkg_builtin_kernel_enabled for
      fused RoPE + attention ISA kernel during token generation (decode)
    - KV cache has num_heads_per_rank = num_heads // TP_DEGREE per rank

KV Cache Flow (direct buffers — same pattern as baseline):
    Both prefill and decode:
      -> Each layer produces new (K, V) tensors (per-rank: num_heads // TP_DEGREE heads)
      -> torch.scatter writes them into the per-layer cache buffers
      -> setattr(self, f"cache_k_{i}", ...) ensures ModelBuilder creates aliases
      -> For decode: past_key_value=[k_cache, v_cache] passed to NxDI layer (triggers TKG)
      -> For prefill: past_key_value=None (triggers CTE mode)

Weight Flow (TP=2):
    1. _convert_layer_weights() maps HF keys → NxDI keys with FULL (unsharded) tensors
    2. shard_checkpoint() splits them per-rank based on layer types
       (ColumnParallelLinear splits output dim, RowParallelLinear splits input dim)
    3. nxd_model.set_weights([rank0_sd, rank1_sd]) loads per-rank weights
"""

# TP degree: 2 gives 6 heads/core on our 12-head model.
# LNC=1 on trn2.3xlarge gives 8 logical cores, TP=2 uses 2 of them.
# NOTE: attn_tkg_builtin_kernel_enabled is DISABLED — the ISA kernel
# fails with 6 heads/core (assertion in NKI trace). Trying without kernel
# first to see if TP=2's reduced per-core work itself gives a speedup.
TP_DEGREE = 2

import os
import math
import logging
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn

logger = logging.getLogger("Neuron")


# ============================================================================
# Lazy imports — these are only available on Neuron instances
# ============================================================================


def _lazy_nxdi_imports():
    """Import NxDI modules. Called at construction time, not at import time."""
    import neuronx_distributed.parallel_layers.parallel_state as parallel_state
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.modules.attention.attention_base import (
        NeuronAttentionBase,
    )
    from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        NeuronLlamaMLP,
        NeuronLlamaAttention,
        NeuronLlamaDecoderLayer,
        get_rmsnorm_cls,
    )

    return {
        "parallel_state": parallel_state,
        "NeuronConfig": NeuronConfig,
        "NeuronAttentionBase": NeuronAttentionBase,
        "RotaryEmbedding": RotaryEmbedding,
        "NeuronLlamaMLP": NeuronLlamaMLP,
        "NeuronLlamaAttention": NeuronLlamaAttention,
        "NeuronLlamaDecoderLayer": NeuronLlamaDecoderLayer,
        "get_rmsnorm_cls": get_rmsnorm_cls,
    }


# ============================================================================
# Parallel state initialization
# ============================================================================


def init_parallel_state(tp_degree=TP_DEGREE):
    """
    Initialize torch.distributed + parallel_state for the given TP degree.

    For TP=1: Uses gloo backend (single process, single rank).
    For TP>1: Uses NxDParallelState context manager which handles SPMD
    parallel state internally without requiring torchrun.

    Returns an NxDParallelState context manager for TP>1, or None for TP=1.
    The caller should use this as a context manager wrapping model
    construction and compilation.
    """
    from neuronx_distributed import NxDParallelState
    from neuronx_distributed.parallel_layers import parallel_state

    if tp_degree == 1:
        # TP=1: simple gloo-based init (same as before)
        if parallel_state.model_parallel_is_initialized():
            return None
        if torch.distributed.is_initialized():
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
            return None

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        return None
    else:
        # TP>1: return NxDParallelState context manager
        # Caller must use: `with init_parallel_state(tp_degree=2): ...`
        return NxDParallelState(
            world_size=tp_degree,
            tensor_model_parallel_size=tp_degree,
        )


# ============================================================================
# Config adapter: bridge SongGenerationConfig -> NxDI InferenceConfig
# ============================================================================


class SongGenInferenceConfig:
    """
    Minimal InferenceConfig-compatible object that satisfies NxDI's
    NeuronLlamaAttention, NeuronLlamaMLP, NeuronLlamaDecoderLayer, and KVCacheManager.

    Instead of subclassing InferenceConfig (which has complex __init__),
    we create a simple namespace with the required fields.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        head_dim: int = 128,
        intermediate_size: int = 8960,
        num_hidden_layers: int = 28,
        rope_theta: float = 100000.0,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",
        batch_size: int = 2,
        max_seq_len: int = 512,
        torch_dtype: torch.dtype = torch.float32,
        tp_degree: int = TP_DEGREE,
    ):
        # Core architecture fields (read by NeuronLlamaAttention, NeuronLlamaMLP)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act

        # Fields read by NeuronLlamaDecoderLayer
        self.num_cores_per_group = tp_degree

        # No rope scaling for LeLM
        # (NeuronLlamaAttention.get_rope checks hasattr(config, 'rope_scaling'))
        # We intentionally do NOT set rope_scaling so it uses basic RotaryEmbedding.

        # NeuronConfig sub-object
        nxdi = _lazy_nxdi_imports()
        NeuronConfig = nxdi["NeuronConfig"]

        self.neuron_config = NeuronConfig(
            tp_degree=tp_degree,
            batch_size=batch_size,
            seq_len=max_seq_len,
            n_positions=max_seq_len,
            max_length=max_seq_len,
            max_context_length=max_seq_len,
            torch_dtype=torch_dtype,
            # NKI kernel configuration for TP=2:
            # - Attention builtin TKG kernel: DISABLED — fails with 6 heads/core
            #   (NKI trace assertion failure). Testing without kernel first.
            # - MLP kernel: DISABLED — CTE (prefill) path has intermediate_size <= 4096
            #   assertion. Even with TP=2, intermediate_size/2=4480 > 4096.
            # - Mega-fused kernel: DISABLED — requires KVCacheManager
            attn_block_tkg_nki_kernel_enabled=False,
            attn_block_tkg_nki_kernel_cache_update=False,
            mlp_kernel_enabled=False,
            mlp_tkg_nki_kernel_enabled=False,
            attn_tkg_nki_kernel_enabled=False,
            attn_tkg_builtin_kernel_enabled=False,
            attn_kernel_enabled=False,
            attn_block_cte_nki_kernel_enabled=False,
            qkv_kernel_enabled=False,
            qkv_nki_kernel_enabled=False,
            out_proj_kernel_enabled=False,
            mlp_kernel_fuse_residual_add=False,
            qkv_kernel_fuse_residual_add=False,
            # Standard settings
            fused_qkv=False,
            sequence_parallel_enabled=False,
            on_cpu=False,
            padding_side="right",
            flash_decoding_enabled=False,
            k_cache_transposed=False,
            qk_layernorm=False,
            enable_fused_speculation=False,
            is_eagle3=False,
            is_eagle_draft=False,
        )

    @classmethod
    def for_primary(cls, sg_config) -> "SongGenInferenceConfig":
        """Create config for the 28-layer primary transformer."""
        return cls(
            hidden_size=sg_config.dim,
            num_attention_heads=sg_config.num_heads,
            num_key_value_heads=sg_config.num_heads,  # MHA, not GQA
            head_dim=sg_config.head_dim,
            intermediate_size=8960,
            num_hidden_layers=sg_config.primary_layers,
            rope_theta=sg_config.primary_rope_theta,
            max_position_embeddings=sg_config.max_seq_len * 4,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            batch_size=sg_config.batch_size,
            max_seq_len=sg_config.max_seq_len,
        )

    @classmethod
    def for_secondary(cls, sg_config) -> "SongGenInferenceConfig":
        """Create config for the 12-layer secondary transformer."""
        return cls(
            hidden_size=sg_config.dim,
            num_attention_heads=sg_config.num_heads,
            num_key_value_heads=sg_config.num_heads,
            head_dim=sg_config.head_dim,
            intermediate_size=8960,
            num_hidden_layers=sg_config.secondary_layers,
            rope_theta=sg_config.secondary_rope_theta,
            max_position_embeddings=sg_config.max_seq_len * 4,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            batch_size=sg_config.batch_size,
            max_seq_len=sg_config.max_seq_len,
        )


# ============================================================================
# Direct-buffer KV cache helpers (ModelBuilder-compatible)
# ============================================================================


def _register_kv_buffers(
    module,
    num_layers,
    batch_size,
    num_heads,
    max_seq_len,
    head_dim,
    tp_degree=TP_DEGREE,
):
    """
    Register per-layer KV cache buffers as module attributes using register_buffer.

    With TP>1, each rank only holds num_heads // tp_degree heads, so the
    cache buffer is sized accordingly. ModelBuilder traces per-rank, so
    the buffer shape matches what each rank's NxDI layers produce.

    This uses the same pattern as the baseline (_NeuronPrimaryTransformer,
    _NeuronFusedSecondary) which ModelBuilder correctly aliases as
    input/output state tensors.
    """
    num_heads_per_rank = num_heads // tp_degree
    for i in range(num_layers):
        module.register_buffer(
            f"cache_k_{i}",
            torch.zeros(batch_size, num_heads_per_rank, max_seq_len, head_dim),
        )
        module.register_buffer(
            f"cache_v_{i}",
            torch.zeros(batch_size, num_heads_per_rank, max_seq_len, head_dim),
        )


def _run_decoder_layers(
    module: nn.Module,
    layers: nn.ModuleList,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    is_context_encoding: bool,
    max_seq_len: int,
    num_heads_per_rank: int,
    head_dim: int,
):
    """
    Run NxDI decoder layers with direct-buffer KV cache management.

    Uses the same getattr/setattr + torch.scatter pattern as the baseline
    for KV cache updates. This ensures ModelBuilder correctly creates
    input/output aliases for the cache tensors.

    The NxDI decoder layers handle everything internally (RMSNorm, QKV,
    RoPE, attention, MLP, residuals), but we manage the KV cache externally
    to maintain compatibility with ModelBuilder's aliasing mechanism.

    Args:
        module: The parent module that owns the cache_k_{i}/cache_v_{i} buffers
        layers: nn.ModuleList of NeuronLlamaDecoderLayer
        hidden_states: [B, S, D]
        attention_mask: [B, 1, S, max_seq] (boolean: True=attend)
        position_ids: [B, S]
        cache_position: [S] — position indices for scatter
        is_context_encoding: True for prefill (CTE), False for decode (TKG)
        max_seq_len: max sequence length
        num_heads_per_rank: number of attention heads PER RANK (num_heads // tp_degree)
        head_dim: head dimension

    Returns:
        hidden_states: [B, S, D] — output of the last layer (before final norm)
    """
    num_layers = len(layers)
    bsz = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    cos_cache = None
    sin_cache = None
    residual = None

    for idx in range(num_layers):
        decoder_layer = layers[idx]

        # Read existing KV cache for this layer
        k_cache = getattr(module, f"cache_k_{idx}")
        v_cache = getattr(module, f"cache_v_{idx}")

        # For decode (TKG): pass existing cache so NxDI uses decomposed attention
        # For prefill (CTE): pass None so NxDI uses context encoding attention
        if is_context_encoding:
            past_key_value = None
        else:
            past_key_value = [k_cache, v_cache]

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            residual=residual,
            # These kwargs are passed through to NeuronAttentionBase.forward()
            # We set them to values appropriate for our direct-buffer approach
            kv_mgr=None,
            get_kv_per_layer=False,
            update_kv_per_layer=False,
            idx=idx,
            is_for_context_encoding=is_context_encoding,
            seq_len=max_seq_len,
            kvcache_buffer=None,
            active_block_table=None,
            active_mask=None,
            seq_ids=None,
        )

        hidden_states = layer_outputs[0]
        new_kv = layer_outputs[1]  # tuple (K_new, V_new)
        cos_cache = layer_outputs[2]
        sin_cache = layer_outputs[3]
        residual = layer_outputs[4]

        # Extract new K, V from layer output
        new_k, new_v = new_kv[0], new_kv[1]
        # new_k/new_v shape: [B, num_heads_per_rank, seq_len, head_dim]

        # Scatter new K, V into cache at the correct positions
        # (same pattern as baseline: _NeuronPrimaryTransformer.forward)
        scatter_idx = cache_position.view(1, 1, seq_len, 1).expand(
            bsz, num_heads_per_rank, seq_len, head_dim
        )
        setattr(module, f"cache_k_{idx}", torch.scatter(k_cache, 2, scatter_idx, new_k))
        setattr(module, f"cache_v_{idx}", torch.scatter(v_cache, 2, scatter_idx, new_v))

    return hidden_states


# ============================================================================
# NxDI Primary Transformer (28L)
# ============================================================================


class NxDIPrimaryTransformer(nn.Module):
    """
    28-layer primary LeLM transformer using NxDI decoder layers.

    Drop-in replacement for _NeuronPrimaryTransformer. Same forward() signature:
        forward(inputs_embeds, position_ids, cache_position, attn_mask)
        -> (hidden_states, logits)

    Must be constructed inside NxDParallelState(tp=TP_DEGREE) context.
    """

    def __init__(self, hf_causal_lm, sg_config):
        super().__init__()

        # parallel_state is already initialized by the NxDParallelState context manager
        self.nxdi_config = SongGenInferenceConfig.for_primary(sg_config)
        self.config = sg_config

        nxdi = _lazy_nxdi_imports()

        # NxDI decoder layers
        self.num_layers = sg_config.primary_layers
        self.layers = nn.ModuleList(
            [
                nxdi["NeuronLlamaDecoderLayer"](self.nxdi_config)
                for _ in range(self.num_layers)
            ]
        )

        # Direct KV cache buffers (ModelBuilder-compatible)
        # With TP>1, each rank holds num_heads // TP_DEGREE heads
        _register_kv_buffers(
            self,
            self.num_layers,
            sg_config.batch_size,
            sg_config.num_heads,
            sg_config.max_seq_len,
            sg_config.head_dim,
            tp_degree=TP_DEGREE,
        )

        # Final norm and LM head (replicated across ranks)
        self.norm = nxdi["get_rmsnorm_cls"](hidden_size=sg_config.dim, eps=1e-5)
        self.lm_head = nn.Linear(sg_config.dim, sg_config.vocab_size, bias=False)

        self.max_seq_len = sg_config.max_seq_len
        self.num_heads = sg_config.num_heads
        self.num_heads_per_rank = sg_config.num_heads // TP_DEGREE
        self.head_dim = sg_config.head_dim

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs_embeds: [B, S, D]
            position_ids: [B, S]
            cache_position: [S] — position indices for scatter
            attn_mask: [B, 1, S, max_seq] — additive mask (0.0=attend, -inf=mask)

        Returns:
            (hidden_states, logits)
        """
        bsz, seq_len, _ = inputs_embeds.shape
        is_context_encoding = seq_len > 1

        # Convert additive mask to boolean (NxDI uses torch.where(mask.bool(), QK, min))
        # Additive: 0.0 = attend, -inf = mask → Boolean: True = attend, False = mask
        nxdi_mask = attn_mask == 0.0

        # For prefill (CTE), NxDI attention only uses K[:, :, :q_len, :],
        # so mask must be [B, 1, S, S] not [B, 1, S, max_seq]
        if is_context_encoding:
            nxdi_mask = nxdi_mask[:, :, :, :seq_len]

        hidden_states = _run_decoder_layers(
            module=self,
            layers=self.layers,
            hidden_states=inputs_embeds,
            attention_mask=nxdi_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            is_context_encoding=is_context_encoding,
            max_seq_len=self.max_seq_len,
            num_heads_per_rank=self.num_heads_per_rank,
            head_dim=self.head_dim,
        )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()

        return hidden_states, logits


# ============================================================================
# NxDI Fused Secondary Transformer (12L + bridge + output heads)
# ============================================================================


class NxDIFusedSecondary(nn.Module):
    """
    12-layer secondary LeLM transformer with MLP bridge and per-codebook heads.

    Drop-in replacement for _NeuronFusedSecondary. Same forward() signature:
        forward(fused_input2, primary_hidden, position_ids, cache_position, attn_mask)
        -> res_logits  [B, code_depth-1, S, V]

    Must be constructed inside NxDParallelState(tp=TP_DEGREE) context.
    """

    def __init__(self, hf_causal_lm, mlp_bridge, output_linears, sg_config):
        super().__init__()

        # parallel_state is already initialized by the NxDParallelState context manager
        self.nxdi_config = SongGenInferenceConfig.for_secondary(sg_config)
        self.config = sg_config

        nxdi = _lazy_nxdi_imports()

        # Custom pre/post components (replicated across ranks)
        self.mlp_bridge = mlp_bridge
        self.output_linears = nn.ModuleList(list(output_linears))
        self.code_depth = sg_config.code_depth

        # NxDI decoder layers
        self.num_layers = sg_config.secondary_layers
        self.layers = nn.ModuleList(
            [
                nxdi["NeuronLlamaDecoderLayer"](self.nxdi_config)
                for _ in range(self.num_layers)
            ]
        )

        # Direct KV cache buffers (ModelBuilder-compatible)
        _register_kv_buffers(
            self,
            self.num_layers,
            sg_config.batch_size,
            sg_config.num_heads,
            sg_config.max_seq_len,
            sg_config.head_dim,
            tp_degree=TP_DEGREE,
        )

        # Final norm (replicated)
        self.norm = nxdi["get_rmsnorm_cls"](hidden_size=sg_config.dim, eps=1e-5)

        self.max_seq_len = sg_config.max_seq_len
        self.num_heads = sg_config.num_heads
        self.num_heads_per_rank = sg_config.num_heads // TP_DEGREE
        self.head_dim = sg_config.head_dim

    def forward(
        self,
        fused_input2: torch.Tensor,
        primary_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_input2: [B, S, D]
            primary_hidden: [B, S, D]
            position_ids: [B, S]
            cache_position: [S]
            attn_mask: [B, 1, S, max_seq] — additive mask (0.0=attend, -inf=mask)

        Returns:
            res_logits: [B, code_depth-1, S, V]
        """
        # MLP bridge
        bridge_input = torch.cat([fused_input2, primary_hidden], dim=-1)
        hidden_states = self.mlp_bridge(bridge_input)

        bsz, seq_len, _ = hidden_states.shape
        is_context_encoding = seq_len > 1

        # Convert additive mask to boolean (NxDI uses torch.where(mask.bool(), QK, min))
        nxdi_mask = attn_mask == 0.0

        # For prefill (CTE), slice to [B, 1, S, S]
        if is_context_encoding:
            nxdi_mask = nxdi_mask[:, :, :, :seq_len]

        hidden_states = _run_decoder_layers(
            module=self,
            layers=self.layers,
            hidden_states=hidden_states,
            attention_mask=nxdi_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            is_context_encoding=is_context_encoding,
            max_seq_len=self.max_seq_len,
            num_heads_per_rank=self.num_heads_per_rank,
            head_dim=self.head_dim,
        )

        hidden_states = self.norm(hidden_states)

        # Per-codebook output heads
        res_logits = torch.stack(
            [
                self.output_linears[k](hidden_states).float()
                for k in range(self.code_depth - 1)
            ],
            dim=1,
        )

        return res_logits


# ============================================================================
# Weight conversion: HuggingFace -> NxDI state dict
# ============================================================================


def _convert_layer_weights(
    hf_sd, nxdi_sd, num_layers, mlp_kernel_enabled=False, hf_layer_prefix="model.layers"
):
    """
    Convert per-layer weights from HF Llama keys to NxDI keys.
    Produces FULL (unsharded) tensors — shard_checkpoint() handles splitting for TP>1.

    With fused_qkv=False:
        HF: {hf_layer_prefix}.{i}.self_attn.{q,k,v,o}_proj.weight
        NxDI: layers.{i}.self_attn.qkv_proj.{q,k,v}_proj.weight  [hidden, hidden]
              layers.{i}.self_attn.o_proj.o_proj.weight  (double-nested)

    MLP weights:
        When mlp_kernel_enabled=False:
            HF [out, in] == NxDI [out, in] — no transpose needed.
        When mlp_kernel_enabled=True:
            NxDI init transposes weights via transpose_parallel_linear_layer()
            to [in, out] layout for NKI kernels. We must provide transposed weights
            to match the parameter shapes after init.

    Note: rank_util.rank is NOT included here — shard_checkpoint handles it,
    or it's auto-initialized by NxDI.
    """
    for i in range(num_layers):
        hf_p = f"{hf_layer_prefix}.{i}"
        nx_p = f"layers.{i}"

        # Separate Q, K, V projections (fused_qkv=False)
        # Full unsharded shapes: [hidden, hidden] = [1536, 1536]
        nxdi_sd[f"{nx_p}.self_attn.qkv_proj.q_proj.weight"] = hf_sd[
            f"{hf_p}.self_attn.q_proj.weight"
        ].clone()
        nxdi_sd[f"{nx_p}.self_attn.qkv_proj.k_proj.weight"] = hf_sd[
            f"{hf_p}.self_attn.k_proj.weight"
        ].clone()
        nxdi_sd[f"{nx_p}.self_attn.qkv_proj.v_proj.weight"] = hf_sd[
            f"{hf_p}.self_attn.v_proj.weight"
        ].clone()

        # O projection (double-nested: self_attn.o_proj.o_proj.weight)
        # preshard_hook reads this key and writes the triple-nested version.
        # Full unsharded: [hidden, hidden] = [1536, 1536]
        nxdi_sd[f"{nx_p}.self_attn.o_proj.o_proj.weight"] = hf_sd[
            f"{hf_p}.self_attn.o_proj.weight"
        ].clone()

        # MLP weights — full unsharded
        if mlp_kernel_enabled:
            nxdi_sd[f"{nx_p}.mlp.gate_proj.weight"] = (
                hf_sd[f"{hf_p}.mlp.gate_proj.weight"].clone().t()
            )
            nxdi_sd[f"{nx_p}.mlp.up_proj.weight"] = (
                hf_sd[f"{hf_p}.mlp.up_proj.weight"].clone().t()
            )
            nxdi_sd[f"{nx_p}.mlp.down_proj.weight"] = (
                hf_sd[f"{hf_p}.mlp.down_proj.weight"].clone().t()
            )
        else:
            nxdi_sd[f"{nx_p}.mlp.gate_proj.weight"] = hf_sd[
                f"{hf_p}.mlp.gate_proj.weight"
            ].clone()
            nxdi_sd[f"{nx_p}.mlp.up_proj.weight"] = hf_sd[
                f"{hf_p}.mlp.up_proj.weight"
            ].clone()
            nxdi_sd[f"{nx_p}.mlp.down_proj.weight"] = hf_sd[
                f"{hf_p}.mlp.down_proj.weight"
            ].clone()

        # Layer norms (replicated across ranks — shard_checkpoint leaves these intact)
        nxdi_sd[f"{nx_p}.input_layernorm.weight"] = hf_sd[
            f"{hf_p}.input_layernorm.weight"
        ].clone()
        nxdi_sd[f"{nx_p}.post_attention_layernorm.weight"] = hf_sd[
            f"{hf_p}.post_attention_layernorm.weight"
        ].clone()


def _build_primary_sd(hf_causal_lm, nxdi_primary, sg_config) -> Dict[str, torch.Tensor]:
    """Build FULL (unsharded) NxDI state dict for primary transformer."""
    hf_sd = hf_causal_lm.state_dict()
    nxdi_sd = {}

    mlp_kernel = nxdi_primary.nxdi_config.neuron_config.mlp_kernel_enabled
    _convert_layer_weights(
        hf_sd, nxdi_sd, sg_config.primary_layers, mlp_kernel_enabled=mlp_kernel
    )

    nxdi_sd["norm.weight"] = hf_sd["model.norm.weight"].clone()
    nxdi_sd["lm_head.weight"] = hf_sd["lm_head.weight"].clone()

    return nxdi_sd


def _build_secondary_sd(
    hf_causal_lm, mlp_bridge, output_linears, nxdi_secondary, sg_config
) -> Dict[str, torch.Tensor]:
    """Build FULL (unsharded) NxDI state dict for secondary transformer."""
    hf_sd = hf_causal_lm.state_dict()
    nxdi_sd = {}

    mlp_kernel = nxdi_secondary.nxdi_config.neuron_config.mlp_kernel_enabled
    _convert_layer_weights(
        hf_sd, nxdi_sd, sg_config.secondary_layers, mlp_kernel_enabled=mlp_kernel
    )

    nxdi_sd["norm.weight"] = hf_sd["model.norm.weight"].clone()

    # MLP bridge (replicated — not a parallel layer)
    for k, v in mlp_bridge.state_dict().items():
        nxdi_sd[f"mlp_bridge.{k}"] = v.clone()

    # Per-codebook output heads (replicated)
    for head_idx, linear in enumerate(output_linears):
        for k, v in linear.state_dict().items():
            nxdi_sd[f"output_linears.{head_idx}.{k}"] = v.clone()

    return nxdi_sd


def shard_model_weights(
    full_sd: Dict[str, torch.Tensor],
    model: nn.Module,
) -> List[Dict[str, torch.Tensor]]:
    """
    Shard a full (unsharded) state dict into per-rank dicts using shard_checkpoint.

    Must be called inside NxDParallelState(tp=TP_DEGREE) context.
    Returns a list of TP_DEGREE dicts, one per rank.
    """
    from neuronx_distributed import shard_checkpoint

    return shard_checkpoint(
        checkpoint=full_sd,
        model=model,
        start_rank=0,
        end_rank=TP_DEGREE - 1,
        load_on_device=True,
    )


# Legacy TP=1 functions kept for backward compatibility
def convert_primary_weights(hf_causal_lm, nxdi_primary, sg_config):
    """Convert HF weights and load for TP=1 (legacy — use _build_primary_sd + shard for TP>1)."""
    nxdi_sd = _build_primary_sd(hf_causal_lm, nxdi_primary, sg_config)
    # For TP=1, can load directly
    missing, unexpected = nxdi_primary.load_state_dict(nxdi_sd, strict=False)
    if missing:
        logger.warning(f"Primary weight conversion - missing keys: {missing}")
    if unexpected:
        logger.warning(f"Primary weight conversion - unexpected keys: {unexpected}")
    return nxdi_primary


def convert_secondary_weights(
    hf_causal_lm, mlp_bridge, output_linears, nxdi_secondary, sg_config
):
    """Convert HF weights and load for TP=1 (legacy — use _build_secondary_sd + shard for TP>1)."""
    nxdi_sd = _build_secondary_sd(
        hf_causal_lm, mlp_bridge, output_linears, nxdi_secondary, sg_config
    )
    missing, unexpected = nxdi_secondary.load_state_dict(nxdi_sd, strict=False)
    if missing:
        logger.warning(f"Secondary weight conversion - missing keys: {missing}")
    if unexpected:
        logger.warning(f"Secondary weight conversion - unexpected keys: {unexpected}")
    return nxdi_secondary
