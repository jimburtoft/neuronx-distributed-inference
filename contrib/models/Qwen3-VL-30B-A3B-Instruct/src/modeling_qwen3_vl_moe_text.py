# coding=utf-8
# Qwen3-VL-MoE Text Decoder for NxDI
# Combines qwen3_vl attention (mRoPE, QK-Norm, DeepStack) with qwen3_moe MLP (MoE routing)

import gc
import math
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
    MOE_TKG_MK_INTERMEDIATE_PER_TP,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    scatter_by_index_put,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


# =============================================================================
# mRoPE -- Reused from qwen3_vl (identical implementation)
# =============================================================================


class NeuronQwen3VLMoeRotaryEmbedding(nn.Module):
    """Multimodal Rotary Position Embedding for Qwen3-VL-MoE.
    Identical to NeuronQwen3VLRotaryEmbedding -- 3D mRoPE with interleaved T/H/W."""

    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = self.config.rope_scaling["rope_type"]

        assert self.rope_type == "default", "Only 'default' rope_type is supported"

        inv_freq, self.attention_scaling = self._compute_default_rope_parameters(
            config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    @staticmethod
    def _compute_default_rope_parameters(config, device=None):
        base = config.rope_theta
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )
        freqs = self._neuron_compute_freqs_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def _neuron_compute_freqs_mrope(
        freqs: torch.Tensor, mrope_section: list
    ) -> torch.Tensor:
        """XLA-friendly multimodal RoPE frequency computation."""
        last_dim = freqs.shape[-1]
        indices = torch.arange(last_dim, device=freqs.device, dtype=torch.int64)
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            mask = (indices % 3 == offset) & (indices < length)
            freqs_t = torch.where(mask, freqs[dim], freqs_t)
        return freqs_t


# =============================================================================
# Attention -- Reused from qwen3_vl (mRoPE + QK-Norm)
# =============================================================================


class NeuronQwen3VLMoeAttention(NeuronAttentionBase):
    """GQA attention with mRoPE and QK-Norm. Same as NeuronQwen3VLAttention."""

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        super().__init__(
            config=config,
            tensor_model_parallel_group=tensor_model_parallel_group,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            num_cores_per_group=config.num_cores_per_group,
            qkv_bias=False,
            o_bias=False,
            rotary_emb=NeuronQwen3VLMoeRotaryEmbedding(config),
            rms_norm_eps=config.rms_norm_eps,
            attention_chunk_size=getattr(config, "attention_chunk_size", None),
            sliding_window=getattr(config, "sliding_window", None),
            q_layernorm=get_rmsnorm_cls()(head_dim, config.rms_norm_eps),
            k_layernorm=get_rmsnorm_cls()(head_dim, config.rms_norm_eps),
        )
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.mrope_section = config.rope_scaling["mrope_section"]


# =============================================================================
# Decoder Layer -- qwen3_vl attention + qwen3_moe MLP (MoE)
# =============================================================================


class NeuronQwen3VLMoeDecoderLayer(nn.Module):
    """Decoder layer combining VL attention (mRoPE, QK-Norm) with MoE MLP."""

    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3VLMoeAttention(config)
        self.moe_fused_nki_kernel_enabled = getattr(
            config, "moe_fused_nki_kernel_enabled", False
        )

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE MLP -- key difference from qwen3_vl which uses NeuronLlamaMLP
        if self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config,
                rmsnorm=self.post_attention_layernorm,
                init_tkg_module=True,
            )
        else:
            self.mlp = initialize_moe_module(config=config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_position_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        qkv_fused_rmsnorm = None
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with mRoPE (rotary_position_ids is 3D: time/height/width)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rotary_position_ids=rotary_position_ids,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE MLP
        residual = hidden_states
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        is_speculative_decoding = (
            self.config.neuron_config.enable_fused_speculation
            and (not self.config.neuron_config.is_prefill_stage)
        )
        hidden_states = self.mlp(
            hidden_states, padding_mask, is_speculative_decoding=is_speculative_decoding
        )[0]
        hidden_states = residual + hidden_states

        hidden_states = ModuleMarkerEndWrapper()(hidden_states)
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


# =============================================================================
# Text Model -- NeuronBaseModel with MoE layers + DeepStack + vision scatter
# =============================================================================


class NeuronQwen3VLMoeTextModel(NeuronBaseModel):
    @staticmethod
    def deepstack_process_xla(
        hidden_states: torch.Tensor,
        visual_embeds: torch.Tensor,
        vision_mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Integrates visual embeddings into hidden states at specified positions (XLA-friendly)."""
        expanded_visual_embeds = torch.zeros_like(hidden_states)
        expanded_visual_embeds = scatter_by_index_put(
            expanded_visual_embeds, visual_embeds, vision_mask_positions
        )
        return hidden_states + expanded_visual_embeds

    def encode_vision_to_input(
        self, inputs_embeds, vision_embeddings, vision_mask
    ) -> torch.Tensor:
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

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
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size, self.hidden_size, self.padding_idx
            )
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # MoE decoder layers (all 48 layers are MoE since decoder_sparse_step=1)
        self.layers = nn.ModuleList(
            [
                NeuronQwen3VLMoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


# =============================================================================
# State dict conversion -- chain VL prefix stripping + MoE expert stacking
# =============================================================================


def maybe_dequantize_layer(neuron_state_dict, config):
    """Dequantize FP8 layers if present."""
    scale_layers = []
    for layer_key in neuron_state_dict.keys():
        if layer_key.endswith("_scale_inv"):
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)
            fp8_layer_name = layer_key.replace("_scale_inv", "")
            fp8_layer = neuron_state_dict[fp8_layer_name]
            block_size = config.quantization_config["weight_block_size"]
            scales_expanded = scales.repeat_interleave(
                block_size[0], dim=0
            ).repeat_interleave(block_size[1], dim=1)
            scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(
                torch.float32
            )
            neuron_state_dict[fp8_layer_name] = scaled_layer.to(
                config.neuron_config.torch_dtype
            )
    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


def _helper_concat_and_delete_qkv(state_dict, layer_num, attr):
    state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ]
    )
    del state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(state_dict, cfg):
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None) or []
    for layer in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, layer, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled
            or cfg.neuron_config.quantized
        ) and f"layers.{layer}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, layer, "scale")
    gc.collect()
    return state_dict


def convert_qwen3_vl_moe_hf_to_neuron_state_dict(state_dict, config):
    """Convert HF Qwen3-VL-MoE state dict to NxDI format.

    Combines:
    1. VL text prefix stripping (language_model.* -> *)
    2. Attention key renaming (q/k/v_proj -> qkv_proj.*)
    3. QK-Norm renaming (q_norm -> q_layernorm)
    4. MoE expert weight stacking (per-expert -> 3D tensors)
    5. Router renaming (mlp.gate -> mlp.router.linear_router)
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # Step 1: Strip language_model prefix and rename attention keys
    attention_keys = {
        ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
        ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
        ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
        ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
    }
    new_state_dict = {}
    for dict_key in state_dict:
        if "language_model." in dict_key:
            new_key = dict_key.replace("language_model.", "")
            if not config.neuron_config.fused_qkv:
                for atten_key in attention_keys:
                    if atten_key in new_key:
                        new_key = new_key.replace(atten_key, attention_keys[atten_key])
            # QK-Norm rename
            if ".q_norm." in new_key:
                new_key = new_key.replace(".q_norm.", ".q_layernorm.")
            if ".k_norm." in new_key:
                new_key = new_key.replace(".k_norm.", ".k_layernorm.")
            new_state_dict[new_key] = state_dict[dict_key]
        else:
            new_state_dict[dict_key] = state_dict[dict_key]

    # Step 2: Dequantize FP8 if needed
    maybe_dequantize_layer(new_state_dict, config)

    # Step 3: Add rank utility tensors
    new_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    # Step 4: MoE expert weight stacking + router rename
    for l in range(config.num_hidden_layers):
        # Rank util for attention
        new_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Router rename: mlp.gate.weight -> mlp.router.linear_router.weight
        gate_key = f"layers.{l}.mlp.gate.weight"
        if gate_key in new_state_dict:
            new_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                new_state_dict[gate_key].detach().clone()
            )
            del new_state_dict[gate_key]

        # Expert weight stacking
        # Qwen3-VL-MoE stores experts as fused 3D tensors in HF:
        #   mlp.experts.gate_up_proj: [num_experts, hidden_size, 2*intermediate_size]
        #   mlp.experts.down_proj: [num_experts, intermediate_size, hidden_size]
        # But the HF checkpoint may have per-expert keys depending on version.
        # Check both formats.

        fused_gate_up_key = f"layers.{l}.mlp.experts.gate_up_proj"
        per_expert_key = f"layers.{l}.mlp.experts.0.gate_proj.weight"

        if fused_gate_up_key in new_state_dict:
            # Already fused 3D tensors -- just need to transpose and rename
            gate_up_proj = new_state_dict[fused_gate_up_key]
            # HF stores as [num_experts, hidden_size, 2*intermediate_size] but we need
            # [num_experts, hidden_size, 2*intermediate_size] (same layout)
            # Actually NxDI expects: [num_experts, hidden_size, 2*intermediate_size]
            # which matches the HF fused format. Just rename.
            pad_size = getattr(config, "moe_intermediate_pad_size", 0)
            if pad_size > 0:
                # gate_up_proj shape: [num_experts, hidden_size, 2*intermediate_size]
                num_experts = gate_up_proj.shape[0]
                hidden_size = gate_up_proj.shape[1]
                gate_up_proj = gate_up_proj.reshape(num_experts, hidden_size, 2, -1)
                gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
                gate_up_proj = gate_up_proj.reshape(num_experts, hidden_size, -1)
            new_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = (
                gate_up_proj
            )
            del new_state_dict[fused_gate_up_key]

            fused_down_key = f"layers.{l}.mlp.experts.down_proj"
            down_proj = new_state_dict[fused_down_key]
            if pad_size > 0:
                down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))
            new_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = (
                down_proj
            )
            del new_state_dict[fused_down_key]

        elif per_expert_key in new_state_dict:
            # Per-expert format -- stack into 3D tensors (same as qwen3_moe conversion)
            intermediate_size, hidden_size = new_state_dict[per_expert_key].shape
            device = new_state_dict[per_expert_key].device
            dtype = new_state_dict[per_expert_key].dtype

            gate_up_proj = torch.empty(
                config.num_experts,
                hidden_size,
                2 * intermediate_size,
                dtype=dtype,
                device=device,
            )
            for e in range(config.num_experts):
                gate_proj_weights = (
                    new_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                    .T.detach()
                    .clone()
                )
                up_proj_weights = (
                    new_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
                    .T.detach()
                    .clone()
                )
                gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
                torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size).copy_(
                    gate_proj_weights
                )
                torch.narrow(
                    gate_up_proj_slice, 2, intermediate_size, intermediate_size
                ).copy_(up_proj_weights)
                del new_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
                del new_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]

            pad_size = getattr(config, "moe_intermediate_pad_size", 0)
            if pad_size > 0:
                gate_up_proj = gate_up_proj.reshape(
                    config.num_experts, hidden_size, 2, -1
                )
                gate_up_proj = torch.nn.functional.pad(gate_up_proj, (0, pad_size))
                gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, -1)
            new_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = (
                gate_up_proj
            )

            down_proj = torch.empty(
                config.num_experts,
                intermediate_size,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            for e in range(config.num_experts):
                down_proj_weights = (
                    new_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
                    .T.detach()
                    .clone()
                )
                torch.narrow(down_proj, 0, e, 1).copy_(down_proj_weights)
                del new_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]

            if pad_size > 0:
                down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_size))
            new_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = (
                down_proj
            )

        gc.collect()

    # Step 5: Fused QKV if enabled
    if config.neuron_config.fused_qkv:
        new_state_dict = convert_state_dict_to_fused_qkv(new_state_dict, config)

    return new_state_dict


# =============================================================================
# CausalLM wrapper
# =============================================================================


class NeuronQwen3VLMoeTextForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen3VLMoeTextModel

    @staticmethod
    def load_hf_model(model_path):
        from transformers import AutoModelForCausalLM

        try:
            from transformers import Qwen3VLMoeForConditionalGeneration

            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path)
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
        return model

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        return convert_qwen3_vl_moe_hf_to_neuron_state_dict(
            state_dict, inference_config
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Qwen3VLMoeInferenceConfig


# =============================================================================
# Inference Config -- MoE + VL combined
# =============================================================================


class Qwen3VLMoeInferenceConfig(InferenceConfig):
    """Text model config for Qwen3-VL-MoE, combining VL text fields with MoE fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MoE config (from qwen3_moe)
        self.num_local_experts = self.num_experts
        self.n_shared_experts = 0

        # Check intermediate padding for MoE
        self.maybe_pad_intermediate()
        self.enable_moe_fused_nki_kernel()

        # Override intermediate_size for MoE expert MLPs
        self.intermediate_size = self.moe_intermediate_size

        # Router config
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.disable_numeric_cc_token = True
        self.neuron_config.normalize_top_k_affinities = True

    def maybe_pad_intermediate(self):
        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree

        # Pad for shard-on-I CTE kernel (SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP=256)
        if getattr(
            self.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = (
                    math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(
                    padded - self.moe_intermediate_size, 0
                )
                self.moe_intermediate_size = padded

        # Pad for fused TKG mega-kernel (MOE_TKG_MK_INTERMEDIATE_PER_TP=128)
        I_TP = self.moe_intermediate_size // moe_tp_degree
        if getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False):
            if I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP != 0:
                padded = (
                    math.ceil(I_TP / MOE_TKG_MK_INTERMEDIATE_PER_TP)
                    * MOE_TKG_MK_INTERMEDIATE_PER_TP
                    * moe_tp_degree
                )
                pad_size = max(padded - self.moe_intermediate_size, 0)
                existing_pad = getattr(self, "moe_intermediate_pad_size", 0)
                self.moe_intermediate_pad_size = existing_pad + pad_size
                self.moe_intermediate_size = padded
                logger.info(
                    f"Padded moe_intermediate_size for fused TKG kernel: "
                    f"{self.moe_intermediate_size - pad_size} -> {self.moe_intermediate_size} "
                    f"(I_TP: {(self.moe_intermediate_size - pad_size) // moe_tp_degree} -> "
                    f"{self.moe_intermediate_size // moe_tp_degree}, pad={pad_size})"
                )

    def enable_moe_fused_nki_kernel(self):
        I_TP = self.moe_intermediate_size // self.neuron_config.moe_tp_degree
        if (
            getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False)
            and I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0
        ):
            self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_scaling",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


# =============================================================================
# Model Wrapper -- Input generator for tracing (25-element tuple with VL slots)
# =============================================================================


class NeuronQwen3VLMoeTextModelWrapper(ImageToTextModelWrapper):
    def pad_inputs(self, *args, pad_type="first_fit"):
        """Override pad_inputs to preserve deepstack_vision_embeds (arg 24).

        The base ModelWrapper.pad_inputs drops arg 24 when re-padding vision
        inputs at line `args = (*args[:22], *padded_args)`. We call the base
        implementation then re-append deepstack_vision_embeds if it was dropped.
        """
        # Save deepstack_vision_embeds before padding
        has_deepstack = len(args) >= 25
        deepstack_vision_embeds = args[24] if has_deepstack else None

        padded_args = super().pad_inputs(*args, pad_type=pad_type)

        # If base class dropped arg 24, re-append it
        if has_deepstack and len(padded_args) < 25:
            # Need to re-pad deepstack to match new padded seq length
            if (
                deepstack_vision_embeds is not None
                and deepstack_vision_embeds.numel() > 0
                and len(deepstack_vision_embeds.shape) >= 3
            ):
                padded_seq_len = padded_args[0].shape[1]
                ds_seq_len = deepstack_vision_embeds.shape[-2]
                if ds_seq_len != padded_seq_len:
                    # Re-generate deepstack at correct size
                    num_ds_layers = deepstack_vision_embeds.shape[0]
                    batch_size = padded_args[0].shape[0]
                    hidden_size = self.config.hidden_size
                    deepstack_vision_embeds = torch.zeros(
                        num_ds_layers,
                        batch_size,
                        padded_seq_len,
                        hidden_size,
                        dtype=self.config.neuron_config.torch_dtype,
                    )
            padded_args = (*padded_args, deepstack_vision_embeds)

        return padded_args

    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        input_batch_size = input_ids.shape[0]
        input_sequence_len = input_ids.shape[-1]
        if input_sequence_len > 1:  # prefill
            vision_embeddings = torch.zeros(
                input_batch_size,
                n_active_tokens,
                config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                size=(input_batch_size, n_active_tokens, 1),
                fill_value=fill_value,
                dtype=torch.int32,
            )
            deepstack_vision_embeds = [
                torch.zeros(
                    input_batch_size,
                    n_active_tokens,
                    config.hidden_size,
                    dtype=config.neuron_config.torch_dtype,
                )
                for _ in config.deepstack_visual_indexes
            ]
            if len(deepstack_vision_embeds) > 0:
                deepstack_vision_embeds = torch.stack(deepstack_vision_embeds)
            else:
                deepstack_vision_embeds = torch.zeros(
                    (0), dtype=config.neuron_config.torch_dtype
                )
        else:  # decode
            vision_embeddings = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
            vision_mask = torch.zeros((0), dtype=torch.bool)
            deepstack_vision_embeds = torch.zeros(
                (0), dtype=config.neuron_config.torch_dtype
            )
        return vision_embeddings, vision_mask, deepstack_vision_embeds

    def input_generator(self):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )
            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros(
                (self.neuron_config.batch_size, bucket), dtype=torch.int32
            )
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len),
                dtype=torch.float32,
            )

            vision_embeddings, vision_mask, deepstack_vision_embeds = (
                self.get_dummy_vision_inputs(
                    config=self.config,
                    input_ids=input_ids,
                    n_active_tokens=n_active_tokens,
                    fill_value=0,
                )
            )
            rotary_position_ids = torch.zeros(
                (3, self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )

            if (
                self.tag == CONTEXT_ENCODING_MODEL_TAG
                or self.tag == TOKEN_GENERATION_MODEL_TAG
            ):
                inputs.append(
                    (
                        input_ids,  # 0
                        attention_mask,  # 1
                        position_ids,  # 2
                        seq_ids,  # 3
                        sampling_params,  # 4
                        torch.empty(0),  # 5  prev_hidden
                        torch.empty(0),  # 6  adapter_ids
                        torch.empty(0),  # 7  accepted_indices
                        torch.empty(0),  # 8  current_length
                        torch.empty(0),  # 9  medusa_mask
                        torch.empty(0),  # 10 scatter_index
                        torch.empty(0),  # 11 slot_mapping
                        torch.empty(0),  # 12 active_block_table
                        torch.empty(0),  # 13 num_queries
                        torch.empty(0),  # 14 computed_context_lens
                        torch.empty(0),  # 15 tile_q_indices
                        torch.empty(0),  # 16 tile_block_tables
                        torch.empty(0),  # 17 tile_masks
                        torch.empty(0),  # 18 inputs_embeds
                        torch.empty(0),  # 19 kv_cache
                        torch.empty(0),  # 20 active_mask
                        rotary_position_ids,  # 21 (3D mRoPE position IDs)
                        vision_embeddings,  # 22
                        vision_mask,  # 23
                        deepstack_vision_embeds,  # 24 (DeepStack features)
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported model tag '{self.tag}' for ImageToText models"
                )
        return inputs
