# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
PyTorch Gemma3 model for NXD inference
"""

from typing import List, Optional, Tuple, Type
import copy
import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Gemma3ForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    repeat_kv,
)

logger = logging.getLogger(__name__)

# Maximum head dimension before chunked attention is needed to avoid
# Neuron compiler DGE out-of-bounds errors.  Gemma-3 1B has head_dim=256
# which exceeds this limit.
_MAX_UNCHUNKED_HEAD_DIM = 128


def _chunked_qk(
    Q: Tensor, K: Tensor, scale: float, chunk_size: int = _MAX_UNCHUNKED_HEAD_DIM
) -> Tensor:
    """Compute Q @ K^T / scale by chunking along head_dim.

    This avoids generating a single matmul with a >128-wide inner dimension
    that triggers DGE out-of-bounds on Neuron hardware.

    Args:
        Q: (B, H, S_q, D) query tensor
        K: (B, H, S_k, D) key tensor (NOT transposed)
        scale: divisor applied to the dot-product (typically sqrt(head_dim))
        chunk_size: max inner-dim width per matmul chunk
    Returns:
        QK: (B, H, S_q, S_k) attention scores
    """
    head_dim = Q.shape[-1]
    if head_dim <= chunk_size:
        return torch.matmul(Q, K.transpose(2, 3)) / scale

    # Accumulate partial dot-products: sum_i Q_i @ K_i^T
    QK = torch.matmul(Q[..., :chunk_size], K[..., :chunk_size].transpose(2, 3))
    for start in range(chunk_size, head_dim, chunk_size):
        end = min(start + chunk_size, head_dim)
        QK = QK + torch.matmul(Q[..., start:end], K[..., start:end].transpose(2, 3))
    return QK / scale


def _chunked_v_matmul(
    scores: Tensor, V: Tensor, chunk_size: int = _MAX_UNCHUNKED_HEAD_DIM
) -> Tensor:
    """Compute scores @ V by chunking V along head_dim.

    This avoids generating a single matmul with a >128-wide output dimension
    that triggers DGE out-of-bounds on Neuron hardware.

    Args:
        scores: (B, H, S_q, S_k) attention weights (after softmax)
        V: (B, H, S_k, D) value tensor
        chunk_size: max output-dim width per matmul chunk
    Returns:
        output: (B, H, S_q, D)
    """
    head_dim = V.shape[-1]
    if head_dim <= chunk_size:
        return torch.matmul(scores, V)

    # Compute scores @ V_i for each chunk and concatenate
    chunks = []
    for start in range(0, head_dim, chunk_size):
        end = min(start + chunk_size, head_dim)
        chunks.append(torch.matmul(scores, V[..., start:end]))
    return torch.cat(chunks, dim=-1)


def _chunked_qk_transposed(
    Q: Tensor, K_t: Tensor, scale: float, chunk_size: int = _MAX_UNCHUNKED_HEAD_DIM
) -> Tensor:
    """Compute Q @ K_t / scale where K_t is already transposed to (B, H, D, S_k).

    Args:
        Q: (B, H, S_q, D)
        K_t: (B, H, D, S_k) — already transposed key
        scale: divisor
        chunk_size: max inner-dim width per matmul chunk
    Returns:
        QK: (B, H, S_q, S_k)
    """
    head_dim = Q.shape[-1]
    if head_dim <= chunk_size:
        return torch.matmul(Q, K_t) / scale

    QK = torch.matmul(Q[..., :chunk_size], K_t[..., :chunk_size, :])
    for start in range(chunk_size, head_dim, chunk_size):
        end = min(start + chunk_size, head_dim)
        QK = QK + torch.matmul(Q[..., start:end], K_t[..., start:end, :])
    return QK / scale


class NeuronGemma3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return Gemma3RMSNorm if cpu_mode() else NeuronGemma3RMSNorm


def get_updated_configs(config: InferenceConfig):
    """
    Generate a list of configurations for each hidden layer in a Gemma3 model.

    Args:
    config (InferenceConfig): The inference configuration for the model.

    Returns:
    list[InferenceConfig]: A list of InferenceConfig objects, one for each layer in the model.
                           Each config may be either the original config or a modified version.
    """
    updated_configs = []

    for i in range(config.num_hidden_layers):
        updated_config = copy.deepcopy(config)

        swa_layer = (i + 1) % 6 != 0

        if not swa_layer:
            updated_config.sliding_window = None

        updated_configs.append(updated_config)

    return updated_configs


class Gemma3NeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attn_cls = NeuronGemma3Attention


class Gemma3InferenceConfig(InferenceConfig):
    def __init__(
        self, neuron_config: NeuronConfig, fused_spec_config=None, load_config=None
    ):
        self.attributes = [
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "sliding_window",
        ]

        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config

        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        text_config = getattr(self, "text_config", None)

        if text_config is not None:
            for attribute in self.attributes:
                setattr(self, attribute, getattr(text_config, attribute))

        # These are not defined in the standard HF Gemma3 config json
        setattr(self, "max_position_embeddings", 131072)
        setattr(self, "local_rope_theta", 10000.0)
        setattr(self, "rope_scaling", 8.0)
        setattr(self, "global_rope_theta", 1000000.0)
        # Use vocab_size from HF config if already set (varies by variant:
        # 262144 for 1B, 262208 for 4B/12B/27B). Only set default if missing.
        if not hasattr(self, "vocab_size") or self.vocab_size is None:
            setattr(self, "vocab_size", 262208)
        setattr(self, "pad_token_id", 0)
        setattr(self, "rms_norm_eps", 1e-06)
        setattr(self, "hidden_act", "gelu_pytorch_tanh")

        # Auto-disable NKI flash attention kernel when head_dim > 128
        # (the NKI kernel asserts head_dim <= 128)
        head_dim = getattr(
            self, "head_dim", self.hidden_size // self.num_attention_heads
        )
        if (
            head_dim > _MAX_UNCHUNKED_HEAD_DIM
            and self.neuron_config.attn_kernel_enabled is not False
        ):
            logger.warning(
                f"Gemma3: head_dim={head_dim} > {_MAX_UNCHUNKED_HEAD_DIM}, "
                f"auto-disabling NKI attention kernel (attn_kernel_enabled=False)"
            )
            self.neuron_config.attn_kernel_enabled = False

        self.add_derived_config()
        self.validate_config()

    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return self.attributes

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma3NeuronConfig]:
        return Gemma3NeuronConfig


class NeuronGemma3Attention(NeuronAttentionBase):
    """Gemma3 attention with chunked matmuls for head_dim > 128.

    The Neuron compiler generates DGE (Data Group Engine) scatter/gather
    instructions that produce out-of-bounds memory accesses when operating on
    tensors with a head_dim of 256.  This subclass overrides the attention
    computation methods to split all Q@K^T and scores@V matmuls along head_dim
    into 128-wide chunks, keeping the results mathematically identical while
    staying within hardware addressing limits.
    """

    def __init__(self, config: Gemma3InferenceConfig):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        local_rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.local_rope_theta,
        )

        global_rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.global_rope_theta,
            factor=config.rope_scaling,
        )

        rotary_emb = local_rotary_emb
        if config.sliding_window is None:
            rotary_emb = global_rotary_emb

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
            use_scaled_rope=None,
            sliding_window=config.sliding_window,
            post_transpose_layernorm=True,
            # TODO: Enable this
            # query_pre_attn_scalar=config.query_pre_attn_scalar**(.5) # QK/sqrt(head_dim) is replaced with QK/sqrt(query_pre_attn_scalar) in Gemma3
        )

        self.q_layernorm = get_rmsnorm_cls()(
            hidden_size=head_dim, eps=config.rms_norm_eps
        )
        self.k_layernorm = get_rmsnorm_cls()(
            hidden_size=head_dim, eps=config.rms_norm_eps
        )
        self._needs_chunked_attn = head_dim > _MAX_UNCHUNKED_HEAD_DIM

        # The base class forces k_cache_transposed=False for SWA layers
        # (attention_base.py line 316).  However, the KV cache manager uses
        # the NeuronConfig value for ALL layers, so the K cache is BHDS for
        # every layer when k_cache_transposed=True.  We restore the config
        # value here so that SWA layers correctly interpret the cache layout.
        # Our compute_for_token_gen override handles the transposed repeat_kv.
        self.k_cache_transposed = config.neuron_config.k_cache_transposed

    # ------------------------------------------------------------------
    # Overrides for chunked attention (head_dim > 128)
    # ------------------------------------------------------------------

    def scaled_qk(self, Q, K, attention_mask):
        """Override: chunk Q@K^T for large head_dim."""
        if self._needs_chunked_attn:
            QK = _chunked_qk(Q, K, scale=math.sqrt(self.head_dim))
        else:
            QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            QK = torch.where(
                attention_mask.to(torch.bool), QK, torch.finfo(QK.dtype).min
            )
        return QK

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """Override: use chunked V matmul for the flat-compiler CTE path."""
        from neuronx_distributed_inference.modules.attention.attention_base import (
            FlashAttentionStrategy,
        )

        flash_attn_strategy = self.get_flash_attention_strategy(
            q_len, attention_mask is not None
        )
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            # Kernel paths — delegate to base class (should not be reached for
            # head_dim > 128 since the kernel is auto-disabled)
            return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)

        # Flat compiler path — use chunked matmuls
        logger.debug("ATTN: native compiler (Gemma3 chunked)")
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = self.scaled_qk(Q, K_active, attention_mask)
        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            assert learned_sinks.ndim == 1 and learned_sinks.shape[0] == self.num_heads
            learned_sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(
                bsz, -1, q_len, -1
            )
            active_scores = torch.cat((active_scores, learned_sinks), dim=-1)
        active_scores = nn.functional.softmax(
            active_scores, dim=-1, dtype=torch.float32
        ).to(Q.dtype)
        if learned_sinks is not None:
            active_scores = active_scores[..., :-1]

        if self._needs_chunked_attn:
            attn_output = _chunked_v_matmul(active_scores, V_active)
        else:
            attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    def perform_prefill_windowed_attn(
        self, Q, K, V, q_len, bsz, attention_mask, window_size
    ) -> Tensor:
        """Override: use chunked matmuls for windowed (SWA) CTE path."""
        from neuronx_distributed_inference.modules.attention.attention_base import (
            FlashAttentionStrategy,
        )

        flash_attn_strategy = self.get_flash_attention_strategy(
            q_len, attention_mask is not None
        )
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if (
            flash_attn_strategy != FlashAttentionStrategy.NONE
            and flash_attn_strategy != FlashAttentionStrategy.SLIDING_WINDOW_KERNEL
        ):
            attn_output, _ = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
            assert attn_output.shape == (bsz, self.num_heads, self.head_dim, q_len)
            return attn_output, flash_attn_strategy

        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        if flash_attn_strategy == FlashAttentionStrategy.SLIDING_WINDOW_KERNEL:
            # Kernel path — delegate to base class
            return super().perform_prefill_windowed_attn(
                Q, K, V, q_len, bsz, attention_mask, window_size
            )

        # Flat compiler implementation with chunked matmuls
        logger.debug("Windowed ATTN: native compiler (Gemma3 chunked)")
        active_scores = self.scaled_qk(Q, K_active, attention_mask)
        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            assert learned_sinks.ndim == 1 and learned_sinks.shape[0] == self.num_heads
            learned_sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(
                bsz, -1, q_len, -1
            )
            active_scores = torch.cat((active_scores, learned_sinks), dim=-1)
        active_scores = nn.functional.softmax(
            active_scores, dim=-1, dtype=torch.float32
        ).to(Q.dtype)
        if learned_sinks is not None:
            active_scores = active_scores[..., :-1]

        if self._needs_chunked_attn:
            attn_output = _chunked_v_matmul(active_scores, V_active)
        else:
            attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    def compute_for_token_gen(
        self,
        Q,
        K,
        V,
        position_ids,
        past_key_value,
        attention_mask,
        active_mask,
        is_prefix_caching=False,
    ) -> Tensor:
        """Override: use chunked matmuls for token generation attention.

        This is a copy of the base class method with all Q@K^T and scores@V
        matmuls replaced by chunked versions.
        """
        if not self._needs_chunked_attn:
            return super().compute_for_token_gen(
                Q,
                K,
                V,
                position_ids,
                past_key_value,
                attention_mask,
                active_mask,
                is_prefix_caching,
            )

        from neuronx_distributed_inference.modules.attention.attention_base import (
            manual_softmax,
        )

        is_speculation = False if position_ids is None else position_ids.shape[-1] > 1
        if self.attention_chunk_size and is_speculation:
            raise NotImplementedError(
                "Speculative decoding is not supported by chunked attention yet."
            )

        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]

        if self.k_cache_transposed:
            # K_prior is BHDS (B, H_kv, D, S) from the cache.  repeat_kv
            # expects BHSD layout, so transpose before GQA expansion, then
            # transpose back for the transposed matmul path.
            K_prior = K_prior.transpose(2, 3)  # BHDS -> BHSD
            K_prior = repeat_kv(K_prior, self.num_key_value_groups)  # expand KV heads
            K_prior = K_prior.transpose(2, 3)  # BHSD -> BHDS
            V_prior = repeat_kv(V_prior, self.num_key_value_groups)
            prior_scores = _chunked_qk_transposed(
                Q, K_prior, scale=math.sqrt(self.head_dim)
            )
        else:
            # K_prior is BHSD — standard path
            K_prior = repeat_kv(K_prior, self.num_key_value_groups)
            V_prior = repeat_kv(V_prior, self.num_key_value_groups)
            prior_scores = _chunked_qk(Q, K_prior, scale=math.sqrt(self.head_dim))

        # pad the attention mask if the KV cache is padded
        if (
            prior_scores.shape[-1] > attention_mask.shape[-1]
            and self.neuron_config.apply_seq_ids_mask
        ):
            attention_mask = F.pad(
                attention_mask,
                (0, prior_scores.shape[-1] - attention_mask.shape[-1]),
                "constant",
                0,
            )

        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = _chunked_qk(Q, K_active, scale=math.sqrt(self.head_dim))
        if is_speculation or is_prefix_caching:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            assert learned_sinks.ndim == 1 and learned_sinks.shape[0] == self.num_heads
            bsz, _, seqlen, _ = active_scores.shape
            sinks = learned_sinks.reshape(1, self.num_heads, 1, 1).expand(
                bsz, -1, seqlen, -1
            )
            prior_scores = torch.cat((prior_scores, sinks), dim=-1)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(
            prior_scores, active_scores, is_speculation or is_prefix_caching
        )

        if learned_sinks is not None:
            softmax_prior = softmax_prior[..., :-1]

        softmax_prior, softmax_active = (
            softmax_prior.to(Q.dtype),
            softmax_active.to(Q.dtype),
        )
        attn_prior = _chunked_v_matmul(softmax_prior, V_prior)
        attn_active = _chunked_v_matmul(softmax_active, V_active)

        return attn_prior + attn_active


class NeuronGemma3DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Gemma3InferenceConfig, layer_idx: int):
        super().__init__()

        self.is_sliding_window_attention = config.sliding_window is not None
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.query_pre_attn_scalar = config.query_pre_attn_scalar

        self.self_attn = NeuronGemma3Attention(config)
        self.mlp = NeuronLlamaMLP(config)  # can reuse LlamaMLP module
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        mask = local_mask
        if not self.is_sliding_window_attention or local_mask is None:
            mask = attention_mask

        # Gemma3 uses a scaled word embedding
        # (Normal embedding) * (sqrt(hidden_size) downcast to bfloat 16)
        if self.layer_idx == 0:
            hidden_states = hidden_states * (self.hidden_size**0.5)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronGemma3TextModel(NeuronBaseModel):
    def setup_attr_for_model(self, config: Gemma3InferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Gemma3InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [
                NeuronGemma3DecoderLayer(conf, idx)
                for idx, conf in enumerate(updated_configs)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


class NeuronGemma3ForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as Gemma3ForCausalLM
    """

    _model_cls = NeuronGemma3TextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return Gemma3ForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree

        for i in range(num_layers):
            # To facilitate rank usage in attention
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # Rename q_norm and k_norm
            state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = (
                state_dict[f"layers.{i}.self_attn.q_norm.weight"].detach().clone()
            )
            del state_dict[f"layers.{i}.self_attn.q_norm.weight"]

            state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = (
                state_dict[f"layers.{i}.self_attn.k_norm.weight"].detach().clone()
            )
            del state_dict[f"layers.{i}.self_attn.k_norm.weight"]

        # To facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma3InferenceConfig
