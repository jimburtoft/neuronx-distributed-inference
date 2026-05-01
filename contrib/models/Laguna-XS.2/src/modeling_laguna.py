# coding=utf-8
# Copyright 2026 Poolside and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeuronX Distributed Inference implementation of poolside/Laguna-XS.2.

Laguna-XS.2 is a 33B total / 3B active MoE model for agentic coding:
- 40 layers, 256 routed experts + 1 shared expert, top-8 routing
- Mixed attention: 10 full-attention layers (48 heads) + 30 SWA layers (64 heads)
- Softplus attention gating per head
- Dual RoPE: YaRN (full_attention) + default (sliding_attention)
- Sigmoid routing with e_score_correction_bias and L1 normalization

SCAFFOLD STATUS: This is the initial scaffold with simplified components.
Novel features are added incrementally via Task 008 sub-tasks:
  008a: Per-layer variable attention heads (48 vs 64)  [PENDING]
  008b: Mixed SWA/full attention dispatch               [PENDING]
  008c: Softplus attention gating                        [PENDING]
  008d: Dual RoPE (YaRN + default)                       [PENDING]
  008e: Mixed MLP types (dense + MoE)                    [PENDING]
  008f: MoE with 256 experts                             [PENDING]
  008g: MoE output scaling (2.5x)                        [PENDING]
"""

import copy
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
)
from neuronx_distributed_inference.modules.attention.gqa import (
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed.modules.moe.routing import RouterTopK

logger = logging.getLogger(__name__)


# ====================================================================================
# Normalization
# ====================================================================================


def get_rmsnorm_cls():
    """Return appropriate RMSNorm for current execution context."""
    if cpu_mode():

        class StandardRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )
                return (self.weight * hidden_states).to(input_dtype)

        return StandardRMSNorm
    else:
        return CustomRMSNorm


# ====================================================================================
# Configuration
# ====================================================================================


class LagunaInferenceConfig(InferenceConfig):
    """Configuration for Laguna-XS.2 inference on NeuronX.

    Reads from HuggingFace config.json. Key fields:
    - num_attention_heads_per_layer: [48, 64, 64, 64, 48, ...] (40 entries)
    - layer_types: [full_attention, sliding_attention, ...] (40 entries)
    - mlp_layer_types: [dense, sparse, ...] (40 entries)
    - rope_parameters: {full_attention: {...}, sliding_attention: {...}}
    - num_experts: 256, num_experts_per_tok: 8
    - moe_intermediate_size: 512, shared_expert_intermediate_size: 512
    - moe_routed_scaling_factor: 2.5
    - gating: true (softplus attention gating)
    """

    def __init__(self, neuron_config=None, **kwargs):
        self.vocab_size = kwargs.pop("vocab_size", 100352)
        self.hidden_size = kwargs.pop("hidden_size", 2048)
        self.intermediate_size = kwargs.pop("intermediate_size", 8192)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 40)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 48)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", 8)
        self.head_dim = kwargs.pop("head_dim", 128)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 131072)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-6)
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.attention_bias = kwargs.pop("attention_bias", False)
        self.hidden_act = (
            kwargs.pop("hidden_act", "silu") or "silu"
        )  # HF config has None
        self.sliding_window = kwargs.pop("sliding_window", 512)

        # Per-layer attention heads
        self.num_attention_heads_per_layer = kwargs.pop(
            "num_attention_heads_per_layer",
            [self.num_attention_heads] * self.num_hidden_layers,
        )

        # Layer types
        self.layer_types = kwargs.pop("layer_types", None)
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        self.mlp_layer_types = kwargs.pop("mlp_layer_types", None)
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * self.num_hidden_layers

        # MoE config
        self.num_experts = kwargs.pop("num_experts", 256)
        self.num_experts_per_tok = kwargs.pop("num_experts_per_tok", 8)
        self.moe_intermediate_size = kwargs.pop("moe_intermediate_size", 512)
        self.shared_expert_intermediate_size = kwargs.pop(
            "shared_expert_intermediate_size", 512
        )
        self.moe_routed_scaling_factor = kwargs.pop("moe_routed_scaling_factor", 2.5)
        self.moe_apply_router_weight_on_input = kwargs.pop(
            "moe_apply_router_weight_on_input", False
        )
        self.router_aux_loss_coef = kwargs.pop("router_aux_loss_coef", 0.0)

        # Aliases required by initialize_moe_module()
        self.num_local_experts = self.num_experts
        self.n_shared_experts = 1  # Laguna always has 1 shared expert

        # RoPE
        self.rope_parameters = kwargs.pop("rope_parameters", None)
        self.partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.5)

        # Attention gating
        self.gating = kwargs.pop("gating", True)

        # Standard attributes
        self.pad_token_id = kwargs.pop("pad_token_id", 9)
        self.bos_token_id = kwargs.pop("bos_token_id", 2)
        self.eos_token_id = kwargs.pop("eos_token_id", [2, 24])
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)

        # Pop HF-specific keys not used by our config
        for hf_key in [
            "auto_map",
            "architectures",
            "model_type",
            "transformers_version",
            "dtype",
            "torch_dtype",
            "use_cache",
        ]:
            kwargs.pop(hf_key, None)

        super().__init__(neuron_config=neuron_config, **kwargs)

    def add_derived_config(self):
        self.num_cores_per_group = 1

        # MoE process group config (read by initialize_moe_process_group)
        tp = (
            self.neuron_config.tp_degree
            if hasattr(self, "neuron_config") and self.neuron_config
            else 4
        )
        self.moe_cte_ep_degree = 1
        self.moe_cte_tp_degree = tp
        self.moe_tkg_ep_degree = 1
        self.moe_tkg_tp_degree = tp

        if hasattr(self, "neuron_config") and self.neuron_config is not None:
            nc = self.neuron_config

            # Laguna uses sigmoid routing (not softmax) with L1 normalization
            if hasattr(nc, "router_config") and nc.router_config is not None:
                nc.router_config.act_fn = "sigmoid"

            # GLU MLP for MoE experts (SiLU gating)
            # MoENeuronConfig sets glu_mlp=True by default, but ensure glu_type
            nc.glu_type = "glu"

            # MoE TP degree should match overall TP
            nc.moe_tp_degree = tp

            # Use torch blockwise matmul (NKI shard-hidden kernel not available in SDK 2.29)
            if hasattr(nc, "blockwise_matmul_config"):
                nc.blockwise_matmul_config.use_torch_block_wise = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "LagunaInferenceConfig":
        neuron_config = kwargs.pop("neuron_config", None)
        model_path = os.path.expanduser(model_path)
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


def get_updated_configs(config: LagunaInferenceConfig):
    """Generate per-layer configs for heterogeneous attention.

    SCAFFOLD: All layers use num_attention_heads (48) for now.
    Task 008a will add per-layer variable heads (48 vs 64).
    Task 008b will add mixed SWA/full attention dispatch.
    Task 008d will add dual RoPE configs per layer.
    """
    updated_configs = []

    for i in range(config.num_hidden_layers):
        layer_config = copy.deepcopy(config)

        # SCAFFOLD: Use default head count for all layers.
        # 008a will replace with: config.num_attention_heads_per_layer[i]
        layer_config._layer_num_attention_heads = config.num_attention_heads

        # SCAFFOLD: No SWA distinction yet.
        # 008b will add: layer_config._layer_is_sliding based on layer_types
        layer_config._layer_is_sliding = False

        # SCAFFOLD: Single RoPE config.
        # 008d will add: per-layer rope_theta, partial_rotary_factor, rope_type
        rope_params = {}
        if config.rope_parameters:
            rope_params = config.rope_parameters.get("sliding_attention", {})
        layer_config._layer_rope_theta = rope_params.get("rope_theta", 10000.0)
        layer_config._layer_partial_rotary_factor = 1.0

        updated_configs.append(layer_config)

    return updated_configs


# ====================================================================================
# Attention
# ====================================================================================


class NeuronLagunaAttention(NeuronAttentionBase):
    """Laguna attention with QK norms.

    SCAFFOLD: Basic attention without gating or variable heads.
    Task 008a: Variable num_attention_heads per layer
    Task 008c: Softplus gating (g_proj + F.softplus)
    Task 008d: Partial rotary embedding for full_attention layers
    """

    def __init__(self, config: LagunaInferenceConfig):
        num_heads = config._layer_num_attention_heads
        rope_theta = config._layer_rope_theta

        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,  # SCAFFOLD: full rotation for all layers
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

        # QK norms
        rmsnorm_cls = get_rmsnorm_cls()
        q_norm = rmsnorm_cls(config.head_dim, eps=config.rms_norm_eps)
        k_norm = rmsnorm_cls(config.head_dim, eps=config.rms_norm_eps)

        # SCAFFOLD: Pass sliding_window=None for ALL layers.
        # This routes all layers through standard_causal_attention_forward.
        # 008b will add mixed attention dispatch via local_mask.
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
            q_layernorm=q_norm,
            k_layernorm=k_norm,
            sliding_window=None,
        )


# ====================================================================================
# MLP
# ====================================================================================


class NeuronLagunaMLP(nn.Module):
    """Laguna dense MLP with SiLU gating.

    Used for layer 0 (dense) and as scaffold for all layers.
    Task 008e will add MoE dispatch for layers 1-39.
    """

    def __init__(self, config: LagunaInferenceConfig, intermediate_size: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ====================================================================================
# MoE (008e + 008f)
# ====================================================================================


class RouterTopKWithBias(RouterTopK):
    """RouterTopK with expert_bias for Laguna's sigmoid routing.

    Laguna routing: sigmoid(logits) + expert_bias for top-k selection,
    then L1-normalized sigmoid scores (no bias) as routing weights.
    Based on Trinity's RouterTopKWithBias.
    """

    def __init__(self, expert_bias_size, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            "expert_bias",
            torch.zeros(expert_bias_size, dtype=torch.float32),
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        # Top-k selection with expert_bias added to scores
        scores_for_selection = expert_affinities.float() + self.expert_bias.float()
        _, expert_index = torch.topk(scores_for_selection, self.top_k)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


def _patch_fused_tkg_for_sigmoid():
    """Patch MoEFusedTKG kernel to use ISA router fallback for sigmoid routing.

    The fused MoE TKG NKI kernel's router only supports softmax. Laguna uses
    sigmoid routing. This patch forces use_router_topk_nki_kernel=False to use
    the ISA router fallback which supports both sigmoid and softmax.

    Must be called before model.compile().
    """
    try:
        import neuronx_distributed.modules.moe.moe_fused_tkg as fused_tkg_mod

        original_kernel = fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call
        if original_kernel is None:
            logger.warning(
                "Fused TKG selective load kernel not available, skipping patch"
            )
            return

        class _PatchedKernelCall:
            def __init__(self, original):
                self._original = original

            def __getitem__(self, grid):
                original_grid_call = self._original[grid]

                def patched_call(*args, **kwargs):
                    kwargs["use_router_topk_nki_kernel"] = False
                    return original_grid_call(*args, **kwargs)

                return patched_call

        fused_tkg_mod._moe_token_gen_selective_load_kernel_nki_call = (
            _PatchedKernelCall(original_kernel)
        )

        original_all = fused_tkg_mod._moe_tkg_forward_all_experts_nki_call
        if original_all is not None:
            fused_tkg_mod._moe_tkg_forward_all_experts_nki_call = _PatchedKernelCall(
                original_all
            )

        logger.warning("Patched MoEFusedTKG for sigmoid routing (ISA fallback).")
    except ImportError:
        logger.info("moe_fused_tkg module not available, skipping patch")
    except Exception as e:
        logger.warning("Failed to patch MoEFusedTKG for sigmoid: %s", e)


def initialize_laguna_moe(config: "LagunaInferenceConfig", rmsnorm=None):
    """Initialize MoE module for Laguna with sigmoid routing and expert bias.

    Creates an MoE module via NxDI's initialize_moe_module, then replaces the
    default RouterTopK with RouterTopKWithBias for expert_bias support.

    Args:
        config: LagunaInferenceConfig with MoE fields set for the layer.
            Must have intermediate_size=moe_intermediate_size (512) before calling.
        rmsnorm: Optional RMSNorm for fused TKG path.
    """
    try:
        moe = initialize_moe_module(
            config=config, init_tkg_module=True, rmsnorm=rmsnorm
        )
    except (TypeError, Exception) as e:
        logger.warning("Fused MoE TKG init failed: %s. Falling back.", e)
        moe = initialize_moe_module(config=config)

    # Replace router with bias-aware version (Trinity pattern)
    old_router = moe.router
    new_router = RouterTopKWithBias(
        expert_bias_size=config.num_local_experts,
        num_experts=old_router.num_experts,
        top_k=old_router.top_k,
        hidden_size=old_router.hidden_size,
        dtype=old_router.dtype,
        device=old_router.device,
        act_fn=old_router.act_fn,
        sequence_parallel_enabled=old_router.sequence_parallel_enabled,
        sequence_dimension=old_router.sequence_dimension,
        bias=old_router.bias,
        apply_act_fn_over_topk=old_router.apply_act_fn_over_topk,
        store_transposed_weights=old_router.store_transposed_weights,
    )
    new_router.linear_router = old_router.linear_router
    if hasattr(old_router, "weight_T"):
        new_router.weight_T = old_router.weight_T
    moe.router = new_router
    moe.eval()
    return moe


# ====================================================================================
# Decoder Layer
# ====================================================================================


class NeuronLagunaDecoderLayer(nn.Module):
    """Laguna decoder layer with pre/post norms for attention and MLP.

    Laguna uses 2 norms per block (input_layernorm + post_attention_layernorm).
    Structure: LN -> Attn -> LN -> residual -> MLP -> residual

    Layer 0: Dense MLP (intermediate=8192)
    Layers 1-39: MoE (256 experts, intermediate=512, + shared expert)
    Task 008c: Softplus gating on attention output
    """

    def __init__(self, config: LagunaInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe_layer = (
            hasattr(config, "mlp_layer_types")
            and config.mlp_layer_types[layer_idx] == "sparse"
        )

        self.self_attn = NeuronLagunaAttention(config)

        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.is_moe_layer:
            # MoE layers (1-39): Create config copy with intermediate_size=512
            moe_config = copy.deepcopy(config)
            moe_config.intermediate_size = config.moe_intermediate_size  # 512
            # Disable internal shared expert in MoE module — we handle shared expert
            # separately so we can apply 2.5x scaling to routed output in 008g:
            # result = routed_output * 2.5 + shared_output
            moe_config.n_shared_experts = 0
            self.mlp = initialize_laguna_moe(moe_config)
            # For fused TKG: provide a separate RMSNorm for the kernel's internal
            # normalization. We pass rmsnorm=None to MoE init so CTE doesn't
            # double-apply the norm (following Trinity pattern).
            if (
                hasattr(self.mlp, "moe_fused_tkg")
                and self.mlp.moe_fused_tkg is not None
            ):
                moe_rmsnorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
                self.mlp.moe_fused_tkg.post_attention_layernorm = moe_rmsnorm
            # Shared expert (standalone dense MLP with intermediate=512)
            self.shared_expert = NeuronLagunaMLP(
                config, intermediate_size=config.shared_expert_intermediate_size
            )
        else:
            # Dense layer (layer 0): standard SwiGLU MLP with intermediate=8192
            self.mlp = NeuronLagunaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        # SCAFFOLD: Force recomputation of cos/sin per layer.
        # Required when dual RoPE is added (008d).
        kwargs.pop("cos_cache", None)
        kwargs.pop("sin_cache", None)

        # SCAFFOLD: No mask selection yet.
        # 008b will add: local_mask for SWA layers, attention_mask for full layers.
        kwargs.pop("local_mask", None)

        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # SCAFFOLD: No gating. 008c will add softplus gating here.
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Residual connection after attention (matches HF: residual + attn_output)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
            # MoE: routed experts (no internal shared expert) + separate shared expert.
            # Laguna formula: result = routed_output * 2.5 + shared_output (008g scaling).
            # For now (008e/f), use simple sum: routed_output + shared_output.
            # The MoE rmsnorm was set to post_attention_layernorm, but since we already
            # applied it above, pass hidden_states directly (MoE norm is a no-op when
            # hidden_states is already normed; but the fused TKG path expects it).
            routed_output = self.mlp(hidden_states)[0]  # MoE returns (output, ...)
            shared_output = self.shared_expert(hidden_states)
            hidden_states = routed_output + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ====================================================================================
# Model
# ====================================================================================


class NeuronLagunaModel(NeuronBaseModel):
    """Laguna text model: embeddings + decoder layers + final norm + lm_head."""

    def setup_attr_for_model(self, config: LagunaInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: LagunaInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        # Per-layer configs
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [
                NeuronLagunaDecoderLayer(conf, idx)
                for idx, conf in enumerate(updated_configs)
            ]
        )

        rmsnorm_cls = get_rmsnorm_cls()
        self.norm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

    def init_inference_optimization(self, config: LagunaInferenceConfig):
        """Initialize KV cache and optional on-device sampling."""
        if self.on_device_sampling:
            try:
                from neuronx_distributed_inference.modules.generation.sampling import (
                    Sampler,
                )
            except ImportError:
                from neuronx_distributed_inference.modules.sampling.utils import (
                    create_sampler as Sampler,
                )
            self.sampler = Sampler(config.neuron_config)

        # SCAFFOLD: Standard KV cache (uniform shape for all layers).
        # 008b will add per-layer cache sizing (SWA=512, full=max_length).
        from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
            KVCacheManager,
        )

        self.kv_mgr = KVCacheManager(
            config,
            num_kv_head=self.num_key_value_heads,
            global_rank=self.rank_util,
        )


# ====================================================================================
# Top-level Model Class
# ====================================================================================


class NeuronLagunaForCausalLM(NeuronBaseForCausalLM):
    """Laguna causal LM for NeuronX inference.

    Handles weight loading, state dict conversion, and model initialization.
    """

    _model_cls = NeuronLagunaModel

    def __init__(self, *args, **kwargs):
        # Patch fused TKG for sigmoid routing before any compilation
        _patch_fused_tkg_for_sigmoid()
        super().__init__(*args, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return LagunaInferenceConfig

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load HF model for weight extraction.

        Note: trust_remote_code=True fails on transformers 4.57 due to
        missing imports (RopeParameters, initialization). We load via
        safetensors directly in convert_hf_to_neuron_state_dict instead.
        """
        # Try native transformers first
        try:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **kwargs
            )
        except (ImportError, Exception) as e:
            logger.warning("HF AutoModel loading failed: %s. Using safetensors.", e)
            return None

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        config: LagunaInferenceConfig,
    ) -> Dict[str, torch.Tensor]:
        """Convert HuggingFace Laguna state dict to NeuronX format.

        Key transformations:
        1. Strip 'model.' prefix from HF keys
        2. Remap q_norm/k_norm -> q_layernorm/k_layernorm
        3. Stack expert weights [256, H, 2*I] for gate_up, [256, I, H] for down
        4. Remap router and expert_bias keys
        5. Map shared expert weights
        6. Add rank_util tensors for TP
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        new_state_dict = {}
        target_dtype = torch.bfloat16

        # Detect whether keys still have 'model.' prefix.
        # The framework's application_base.get_state_dict() strips 'model.' BEFORE
        # calling convert_hf_to_neuron_state_dict, so normally keys arrive without it.
        # But handle both cases for robustness (matches Trinity pattern).
        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())

        def strip_prefix(key):
            if has_model_prefix and key.startswith("model."):
                return key[6:]
            return key

        def hf_key(layer_idx, suffix):
            """Build HF state_dict key for a layer, respecting prefix state."""
            if has_model_prefix:
                return f"model.layers.{layer_idx}.{suffix}"
            return f"layers.{layer_idx}.{suffix}"

        # Identify MoE layers
        moe_layers = set()
        if hasattr(config, "mlp_layer_types"):
            for i, t in enumerate(config.mlp_layer_types):
                if t == "sparse":
                    moe_layers.add(i)

        # SCAFFOLD: Identify layers where head count differs from scaffold default.
        # Task 008a will add per-layer variable heads and remove this.
        scaffold_heads = config.num_attention_heads  # 48
        mismatched_layers = set()
        if hasattr(config, "num_attention_heads_per_layer"):
            for i, h in enumerate(config.num_attention_heads_per_layer):
                if h != scaffold_heads:
                    mismatched_layers.add(i)

        attn_weight_suffixes = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ]

        # --- Pass 1: Map non-MoE keys ---
        for key, weight in state_dict.items():
            new_key = strip_prefix(key)

            # Skip all MoE-related keys (handled in Pass 2)
            if any(
                x in new_key
                for x in ["mlp.experts.", "mlp.gate.", "mlp.shared_expert."]
            ):
                continue

            # SCAFFOLD: Skip g_proj (attention gating) — 008c will add
            if "g_proj" in new_key:
                continue

            # SCAFFOLD: Skip attention weights for mismatched-head layers — 008a will fix
            skip = False
            for i in mismatched_layers:
                prefix = f"layers.{i}."
                if new_key.startswith(prefix):
                    for suffix in attn_weight_suffixes:
                        if new_key.endswith(suffix):
                            skip = True
                            break
                if skip:
                    break
            if skip:
                continue

            # Skip dense MLP keys for MoE layers (they don't exist in HF)
            for i in moe_layers:
                prefix = f"layers.{i}.mlp."
                if new_key.startswith(prefix):
                    skip = True
                    break
            if skip:
                continue

            # Remap QK norm keys
            new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

            new_state_dict[new_key] = weight.detach().clone().to(target_dtype)

        # --- Pass 2: Stack MoE expert weights per layer ---
        num_experts = config.num_experts  # 256
        hidden_size = config.hidden_size  # 2048
        moe_intermediate = config.moe_intermediate_size  # 512

        for layer_idx in moe_layers:
            neuron_prefix = f"layers.{layer_idx}"

            # Router: mlp.gate.weight -> mlp.router.linear_router.weight
            router_key = hf_key(layer_idx, "mlp.gate.weight")
            if router_key in state_dict:
                new_state_dict[f"{neuron_prefix}.mlp.router.linear_router.weight"] = (
                    state_dict[router_key].to(target_dtype)
                )

            # Expert bias: mlp.experts.e_score_correction_bias -> mlp.router.expert_bias
            bias_key = hf_key(layer_idx, "mlp.experts.e_score_correction_bias")
            if bias_key in state_dict:
                new_state_dict[f"{neuron_prefix}.mlp.router.expert_bias"] = state_dict[
                    bias_key
                ].to(torch.float32)

            # Stack expert weights: gate+up -> [num_experts, H, 2*I], down -> [num_experts, I, H]
            gate_up_proj = torch.empty(
                num_experts, hidden_size, 2 * moe_intermediate, dtype=target_dtype
            )
            down_proj = torch.empty(
                num_experts, moe_intermediate, hidden_size, dtype=target_dtype
            )

            all_found = True
            for e in range(num_experts):
                gate_k = hf_key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")
                up_k = hf_key(layer_idx, f"mlp.experts.{e}.up_proj.weight")
                down_k = hf_key(layer_idx, f"mlp.experts.{e}.down_proj.weight")

                if gate_k in state_dict and up_k in state_dict and down_k in state_dict:
                    gate_w = state_dict[gate_k].to(target_dtype)  # [I, H]
                    up_w = state_dict[up_k].to(target_dtype)  # [I, H]
                    down_w = state_dict[down_k].to(target_dtype)  # [H, I]

                    # Concat gate+up -> [2*I, H], transpose -> [H, 2*I]
                    gate_up_proj[e] = torch.cat([gate_w, up_w], dim=0).T
                    # Transpose down -> [I, H]
                    down_proj[e] = down_w.T
                else:
                    all_found = False
                    logger.warning(
                        "Missing expert weights for layer %d expert %d", layer_idx, e
                    )
                    break

            if all_found:
                new_state_dict[
                    f"{neuron_prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                ] = gate_up_proj
                new_state_dict[
                    f"{neuron_prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"
                ] = down_proj

            # Shared expert: mlp.shared_expert.* -> shared_expert.*
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                se_key = hf_key(layer_idx, f"mlp.shared_expert.{proj_name}.weight")
                if se_key in state_dict:
                    new_state_dict[
                        f"{neuron_prefix}.shared_expert.{proj_name}.weight"
                    ] = state_dict[se_key].to(target_dtype)

            # Fused MoE TKG aliased weights (Trinity pattern).
            # The MoEFusedTKG module has a separate RMSNorm that needs the same
            # weights as the layer's post_attention_layernorm.
            post_attn_key = f"{neuron_prefix}.post_attention_layernorm.weight"
            if post_attn_key in new_state_dict:
                new_state_dict[
                    f"{neuron_prefix}.mlp.moe_fused_tkg.post_attention_layernorm.weight"
                ] = new_state_dict[post_attn_key].clone()

            # Router transposed weight for fused TKG kernel.
            router_w_key = f"{neuron_prefix}.mlp.router.linear_router.weight"
            if router_w_key in new_state_dict:
                new_state_dict[f"{neuron_prefix}.mlp.router.weight_T"] = (
                    new_state_dict[router_w_key].detach().T.clone()
                )

        # --- Dummy attention weights for mismatched-head layers (scaffold) ---
        # Task 008a will remove this section.
        head_dim = config.head_dim  # 128
        kv_heads = config.num_key_value_heads  # 8
        q_size = scaffold_heads * head_dim  # 48 * 128 = 6144
        kv_size = kv_heads * head_dim  # 8 * 128 = 1024
        for i in mismatched_layers:
            pfx = f"layers.{i}.self_attn"
            new_state_dict[f"{pfx}.q_proj.weight"] = torch.zeros(
                q_size, hidden_size, dtype=target_dtype
            )
            new_state_dict[f"{pfx}.k_proj.weight"] = torch.zeros(
                kv_size, hidden_size, dtype=target_dtype
            )
            new_state_dict[f"{pfx}.v_proj.weight"] = torch.zeros(
                kv_size, hidden_size, dtype=target_dtype
            )
            new_state_dict[f"{pfx}.o_proj.weight"] = torch.zeros(
                hidden_size, q_size, dtype=target_dtype
            )
            new_state_dict[f"{pfx}.q_layernorm.weight"] = torch.ones(
                head_dim, dtype=target_dtype
            )
            new_state_dict[f"{pfx}.k_layernorm.weight"] = torch.ones(
                head_dim, dtype=target_dtype
            )

        # --- rank_util tensors ---
        new_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        for i in range(config.num_hidden_layers):
            new_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Laguna has tie_word_embeddings=False, so no tying needed."""
        pass

    def get_compiler_args(self):
        """Get compiler arguments for Laguna."""
        return "--model-type=transformer"
