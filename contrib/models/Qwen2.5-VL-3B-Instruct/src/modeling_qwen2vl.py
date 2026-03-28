# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team.
# All rights reserved.
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
Qwen2.5-VL-3B-Instruct text backbone for NeuronX Distributed Inference.

Implements the text decoder with M-RoPE (Multimodal Rotary Position Embeddings).
Vision encoder is not included -- this is a text-only port.

The text backbone uses the same architecture as Qwen2.5-VL-7B with different
dimensions: 36 layers, 2048 hidden, 16/2 GQA, 11008 intermediate, 151936 vocab.

Based on: contrib/qwen2.5-vl-7b (validated at 100% token match on trn2.3xlarge)
"""

import gc
import json
import logging
import os
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import _rotate_half
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

logger = logging.getLogger("Neuron")


# ---------------------------------------------------------------------------
# M-RoPE: Multimodal Rotary Position Embedding
# Splits head_dim into [temporal, height, width] sections and applies
# separate rotary embeddings per axis. For text-only input all 3 axes
# receive identical position IDs, but the frequency interleaving still
# differs from standard 1D RoPE.
# ---------------------------------------------------------------------------


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Apply M-RoPE to query and key tensors.

    Args:
        q, k: Query/key tensors of shape (batch, heads, seq, head_dim)
        cos, sin: Rotary embeddings of shape (3, batch, seq, head_dim)
        mrope_section: Section sizes [temporal, height, width] (e.g. [16, 24, 24])
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
    """
    mrope_section = mrope_section * 2  # double for cos/sin pairs
    split_indices = [sum(mrope_section[: i + 1]) for i in range(len(mrope_section) - 1)]
    cos = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(torch.tensor_split(cos, split_indices, dim=-1))
        ],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [
            m[i % 3]
            for i, m in enumerate(torch.tensor_split(sin, split_indices, dim=-1))
        ],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def get_rmsnorm_cls():
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


# ---------------------------------------------------------------------------
# Rotary Embedding (3D for M-RoPE)
# ---------------------------------------------------------------------------


class Qwen2_5_VL3BRotaryEmbedding(nn.Module):
    """3D rotary embedding for M-RoPE (temporal, height, width)."""

    def __init__(self, config: InferenceConfig, device=None):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.base = getattr(config, "rope_theta", 1000000.0)
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", None, persistent=False)
        self.inv_freq = self._compute_inv_freq(device)

    def _compute_inv_freq(self, device=None):
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.base ** (freq_indices / self.dim))

    def forward(self, x, position_ids):
        # position_ids: (3, batch, seq) for [temporal, height, width]
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(
            3, position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[None, :, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# NeuronConfig subclass
# ---------------------------------------------------------------------------


class Qwen2_5_VL3BNeuronConfig(NeuronConfig):
    """NeuronConfig that wires the correct attention class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronQwen2_5_VL3BAttention


# ---------------------------------------------------------------------------
# InferenceConfig subclass
# ---------------------------------------------------------------------------


class Qwen2_5_VL3BInferenceConfig(InferenceConfig):
    """Configuration for Qwen2.5-VL-3B text backbone on NeuronX."""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.qkv_bias = True
        self.o_bias = False
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "use_return_dict"):
            self.use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Qwen2_5_VL3BNeuronConfig]:
        return Qwen2_5_VL3BNeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load config from pretrained model directory.

        Handles the qwen2_5_vl config structure where text params are at top level
        (not nested under text_config like Qwen2.5-Omni).
        """
        neuron_config = kwargs.pop("neuron_config", None)

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found at {config_path}")

        with open(config_path, "r") as f:
            hf_config = json.load(f)

        # Qwen2.5-VL stores text params at top level (not in text_config)
        config_dict = {
            "hidden_size": hf_config.get("hidden_size"),
            "num_attention_heads": hf_config.get("num_attention_heads"),
            "num_hidden_layers": hf_config.get("num_hidden_layers"),
            "num_key_value_heads": hf_config.get("num_key_value_heads"),
            "vocab_size": hf_config.get("vocab_size"),
            "max_position_embeddings": hf_config.get("max_position_embeddings"),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "hidden_act": hf_config.get("hidden_act", "silu"),
            "intermediate_size": hf_config.get("intermediate_size"),
            "pad_token_id": hf_config.get(
                "pad_token_id", hf_config.get("eos_token_id", 151645)
            ),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", True),
        }

        # M-RoPE configuration
        rope_scaling = hf_config.get("rope_scaling", {})
        if rope_scaling:
            config_dict["rope_scaling"] = rope_scaling

        config_dict.update(kwargs)
        return cls(neuron_config=neuron_config, **config_dict)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class NeuronQwen2_5_VL3BAttention(NeuronAttentionBase):
    """Qwen2.5-VL attention with M-RoPE: GQA, QKV bias, 3D position encoding."""

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=True,
            o_bias=False,
            rotary_emb=Qwen2_5_VL3BRotaryEmbedding(config),
            rms_norm_eps=config.rms_norm_eps,
        )
        self.rope_scaling = config.rope_scaling
        self.mrope_section = config.rope_scaling["mrope_section"]

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_multimodal_rotary_pos_emb(
                Q, K, cos_cache, sin_cache, self.mrope_section
            )
        return Q, K, cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class NeuronQwen2_5_VL3BDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer: GQA + SwiGLU MLP."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen2_5_VL3BAttention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuronQwen2_5_VL3BTextModel(NeuronBaseModel):
    """Qwen2.5-VL-3B text decoder model."""

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size, self.hidden_size, self.padding_idx
            )
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.layers = nn.ModuleList(
            [
                NeuronQwen2_5_VL3BDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------


class NeuronQwen2_5_VL3BForCausalLM(NeuronBaseForCausalLM):
    """Qwen2.5-VL-3B causal LM: compilation, weight conversion, generation."""

    _model_cls = NeuronQwen2_5_VL3BTextModel

    @staticmethod
    def load_hf_model(model_path):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """Convert HF state dict to Neuron format.

        Qwen2.5-VL uses 'model.' prefix for text weights. Visual weights are skipped.
        QKV projections are remapped for NeuronAttentionBase.
        """
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        }
        new_state_dict = {}
        for key in state_dict:
            # Skip vision encoder weights
            if "visual" in key or "vision" in key:
                continue

            if "model." in key:
                new_key = key.replace("model.", "")
                if not inference_config.neuron_config.fused_qkv:
                    for attn_key in attention_keys:
                        if attn_key in new_key:
                            new_key = new_key.replace(
                                attn_key, attention_keys[attn_key]
                            )
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]

        if inference_config.neuron_config.fused_qkv:
            new_state_dict = _convert_state_dict_to_fused_qkv(
                new_state_dict, inference_config
            )

        return new_state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Qwen2.5-VL-3B ties embed_tokens and lm_head weights."""
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Qwen2_5_VL3BInferenceConfig


# ---------------------------------------------------------------------------
# Fused QKV helpers
# ---------------------------------------------------------------------------


def _helper_concat_and_delete_qkv(state_dict, layer_num, attr):
    q_key = f"layers.{layer_num}.self_attn.q_proj.{attr}"
    k_key = f"layers.{layer_num}.self_attn.k_proj.{attr}"
    v_key = f"layers.{layer_num}.self_attn.v_proj.{attr}"
    fused_key = f"layers.{layer_num}.self_attn.Wqkv.{attr}"

    if q_key in state_dict and k_key in state_dict and v_key in state_dict:
        state_dict[fused_key] = torch.cat(
            [state_dict[q_key], state_dict[k_key], state_dict[v_key]]
        )
        del state_dict[q_key]
        del state_dict[k_key]
        del state_dict[v_key]


def _convert_state_dict_to_fused_qkv(state_dict, cfg: InferenceConfig):
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None) or []
    for layer in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, layer, "weight")
        # Qwen2.5-VL has QKV bias -- must fuse both weight AND bias
        _helper_concat_and_delete_qkv(state_dict, layer, "bias")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled
            or cfg.neuron_config.quantized
        ) and f"layers.{layer}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, layer, "scale")
    gc.collect()
    return state_dict
