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
        self.hidden_act = kwargs.pop("hidden_act", "silu")
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
        return NeuronConfig

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
# Decoder Layer
# ====================================================================================


class NeuronLagunaDecoderLayer(nn.Module):
    """Laguna decoder layer with pre/post norms for attention and MLP.

    Laguna uses 2 norms per block (input_layernorm + post_attention_layernorm).
    Structure: LN -> Attn -> LN -> residual -> MLP -> residual

    SCAFFOLD: All dense MLP, no MoE, no gating.
    Task 008e: Mixed MLP types (dense layer 0, MoE layers 1-39)
    Task 008c: Softplus gating on attention output
    """

    def __init__(self, config: LagunaInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = NeuronLagunaAttention(config)

        # SCAFFOLD: All layers use dense MLP.
        # 008e will add MoE dispatch for layers 1-39.
        self.mlp = NeuronLagunaMLP(config)

        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )

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

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
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
        3. Add rank_util tensors for TP

        SCAFFOLD: Basic weight mapping for dense-only model.
        Task 008a: Handle per-layer variable head counts
        Task 008c: Map g_proj weights (attention gating)
        Task 008e: Map dense MLP for layer 0
        Task 008f: Stack expert weights for MoE layers
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        new_state_dict = {}
        target_dtype = torch.bfloat16

        # SCAFFOLD: Identify layers where the scaffold head count differs from actual.
        # The scaffold uses num_attention_heads (48) for ALL layers, but layers with
        # 64 heads (SWA layers) have differently-shaped QKVO weights. Skip those
        # attention weights to avoid shape mismatches during weight loading.
        # Task 008a will add per-layer variable heads and load all weights correctly.
        scaffold_heads = config.num_attention_heads  # 48
        mismatched_layers = set()
        if hasattr(config, "num_attention_heads_per_layer"):
            for i, h in enumerate(config.num_attention_heads_per_layer):
                if h != scaffold_heads:
                    mismatched_layers.add(i)

        # Also identify MoE layers: scaffold uses dense MLP for all layers,
        # so MoE layers' dense MLP weights don't exist in HF checkpoint.
        moe_layers = set()
        if hasattr(config, "mlp_layer_types"):
            for i, t in enumerate(config.mlp_layer_types):
                if t == "sparse":
                    moe_layers.add(i)

        attn_weight_suffixes = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        ]

        for key, weight in state_dict.items():
            new_key = key

            # Strip 'model.' prefix
            if new_key.startswith("model."):
                new_key = new_key[6:]

            # SCAFFOLD: Skip MoE keys (experts, router, shared_expert)
            # 008f will handle expert weight stacking
            if any(
                x in new_key
                for x in [
                    "mlp.experts.",
                    "mlp.gate.",
                    "mlp.shared_expert.",
                ]
            ):
                continue

            # SCAFFOLD: Skip g_proj (attention gating)
            # 008c will handle g_proj mapping
            if "g_proj" in new_key:
                continue

            # SCAFFOLD: Skip attention weights for layers with mismatched head counts.
            # These layers have 64 heads in HF but the scaffold uses 48 for all layers.
            # 008a will load all attention weights correctly with per-layer head counts.
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

            # SCAFFOLD: Skip dense MLP weights for MoE layers (they don't exist in HF).
            # The scaffold creates dense MLP modules for all layers but MoE layers
            # only have expert/shared_expert/router weights in the HF checkpoint.
            # 008e will add MoE modules and load these correctly.
            for i in moe_layers:
                prefix = f"layers.{i}.mlp."
                if new_key.startswith(prefix) and not any(
                    x in new_key for x in ["experts.", "gate.", "shared_expert."]
                ):
                    skip = True
                    break
            if skip:
                continue

            # Remap QK norm keys: q_norm -> q_layernorm, k_norm -> k_layernorm
            new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_layernorm.")
            new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_layernorm.")

            new_state_dict[new_key] = weight.detach().clone().to(target_dtype)

        # Add rank_util tensors
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
