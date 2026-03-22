"""
Ministral-3-14B-Instruct-2512 (Leanstral) on AWS Neuron via NxDI.

A vision-language model combining a Pixtral vision encoder with a Llama-compatible
text decoder and a Mistral3-specific PatchMerger projector. Architecture: 40 layers,
hidden=5120, heads=32/8kv, vocab=131072, intermediate=16384, head_dim=128.

This model reuses NxDI's Pixtral VL pipeline (NeuronPixtralVisionModel for the
vision encoder, NeuronLlamaModel for the text decoder) with three key adaptations:

1. CPU projector: The Mistral3 PatchMerger (spatial 2x2 merge via F.unfold + 2-layer MLP)
   runs on CPU since it has no NxDI equivalent.
2. SHARD_OVER_HEADS GQA: Avoids replicating KV heads when kv_heads >= tp_degree. With 8 KV
   heads at TP=4, each rank gets kv_heads_per_rank=2 instead of stock NxDI's replication to 8.
3. Multi-KV-head TKG kernel: A modified nki-library attention_block_tkg kernel that supports
   kv_heads_per_rank > 1 via a virtual-batch approach.

NKI Kernel Optimizations (default in vllm-neuron):
- fused_qkv: Single fused Wqkv weight matrix instead of separate Q/K/V projections
- qkv_nki_kernel_enabled: NKI kernel for fused RMSNorm + QKV projection matmul

NKI Kernel Optimizations (opt-in, blocked by neuronx-cc 2.28 ICE):
- attn_block_tkg_nki_kernel_enabled: Fused TKG attention block (RMSNorm -> QKV -> RoPE ->
  attention -> KV cache update -> output projection) in a single NKI kernel call
- attn_block_tkg_nki_kernel_cache_update: In-kernel KV cache update

Known limitations:
- TKG kernel uses grid=1 for multi-KV-head (NCC_IXLV002 workaround), ~4% text throughput cost
- FP8 checkpoint weights are dequantized to bf16 during state_dict conversion
"""

import copy
import importlib
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.modeling_outputs import CausalLMOutputWithPast

import neuronx_distributed_inference.modules.autobucketing as autobucketing
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
)
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaModel
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
    scatter_by_index_put,
)
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG
from neuronx_distributed_inference.models.pixtral.modeling_pixtral_vision import (
    NeuronPixtralForImageEncoding,
    NeuronPixtralVisionModel,
    PixtralVisionModelWrapper,
)
from neuronx_distributed_inference.modules.flashdecode.utils import (
    calculate_num_cores_per_group,
)

logger = logging.getLogger("Neuron")

# Mistral3 uses image_token_id=10 for [IMG] tokens in the vocabulary
IMAGE_TOKEN_ID = 10
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2


# ---------------------------------------------------------------------------
# SHARD_OVER_HEADS GQA patch
# ---------------------------------------------------------------------------
_shard_over_heads_applied = False


def apply_shard_over_heads_patch():
    """Patch NxDI's GQA sharding to support kv_heads >= tp_degree without replication.

    NxDI 0.8 only supports CONVERT_TO_MHA (replicates KV heads to match Q heads)
    and REPLICATE_TO_TP_DEGREE. For models where kv_heads >= tp_degree and
    kv_heads % tp_degree == 0, we can shard KV heads across ranks instead.
    This avoids inflating KV cache memory by 4x and enables the multi-KV-head
    TKG kernel path.
    """
    global _shard_over_heads_applied
    if _shard_over_heads_applied:
        return

    import neuronx_distributed_inference.modules.attention.gqa as gqa_module

    _orig_determine = gqa_module.determine_sharding_strategy
    _orig_get_shardable = gqa_module.get_shardable_head_counts

    def _patched_determine(
        tp_degree, source_key_value_heads, desired_sharding_strategy=None
    ):
        if (
            source_key_value_heads >= tp_degree
            and source_key_value_heads % tp_degree == 0
        ):
            return gqa_module.GQA.CONVERT_TO_MHA
        return _orig_determine(
            tp_degree, source_key_value_heads, desired_sharding_strategy
        )

    def _patched_get_shardable(
        tp_degree, num_attention_heads, num_key_value_heads, sharding_strategy
    ):
        if (
            sharding_strategy == gqa_module.GQA.CONVERT_TO_MHA
            and num_key_value_heads >= tp_degree
            and num_key_value_heads % tp_degree == 0
        ):
            from neuronx_distributed_inference.modules.attention.gqa import (
                get_number_of_extra_heads,
            )

            updated = num_attention_heads + get_number_of_extra_heads(
                num_attention_heads, tp_degree
            )
            return updated, num_key_value_heads
        return _orig_get_shardable(
            tp_degree, num_attention_heads, num_key_value_heads, sharding_strategy
        )

    for module_path in [
        "neuronx_distributed_inference.modules.attention.gqa",
        "neuronx_distributed_inference.modules.kvcache.kv_cache_manager",
    ]:
        try:
            mod = importlib.import_module(module_path)
            mod.determine_sharding_strategy = _patched_determine
            mod.get_shardable_head_counts = _patched_get_shardable
        except (ImportError, AttributeError):
            pass

    try:
        import neuronx_distributed_inference.modules.kvcache.gpt_kv_cache_manager as gpt_kv

        gpt_kv.determine_sharding_strategy = _patched_determine
        gpt_kv.get_shardable_head_counts = _patched_get_shardable
    except (ImportError, AttributeError):
        pass

    _shard_over_heads_applied = True
    logger.info("SHARD_OVER_HEADS GQA patch applied for Leanstral")


# ---------------------------------------------------------------------------
# Multi-KV-head TKG kernel adapter patch
# ---------------------------------------------------------------------------
_multi_kv_patch_applied = False


def apply_multi_kv_tkg_patch():
    """Patch NxDI's TKG kernel dispatch for multi-KV-head support."""
    global _multi_kv_patch_applied
    if _multi_kv_patch_applied:
        return

    from neuronx_distributed_inference.models.leanstral import patch_native_multi_kv

    patch_native_multi_kv.apply_patch()
    _multi_kv_patch_applied = True
    logger.info("Multi-KV-head TKG kernel adapter patch applied for Leanstral")


def _ensure_patches_applied():
    """Ensure both patches are applied. Safe to call multiple times."""
    apply_shard_over_heads_patch()
    apply_multi_kv_tkg_patch()


# ---------------------------------------------------------------------------
# CPU Projector (Mistral3 PatchMerger + MLP)
# ---------------------------------------------------------------------------


class Mistral3PatchMerger(nn.Module):
    """Spatial 2x2 patch merger using F.unfold for correct spatial ordering."""

    def __init__(self, hidden_size, spatial_merge_size=SPATIAL_MERGE_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.merging_layer = nn.Linear(
            hidden_size * spatial_merge_size * spatial_merge_size,
            hidden_size,
            bias=False,
        )

    def forward(self, features, ph, pw):
        merge = self.spatial_merge_size
        ph_m = (ph // merge) * merge
        pw_m = (pw // merge) * merge
        feats = features.view(ph, pw, self.hidden_size)[:ph_m, :pw_m, :]
        image_grid = feats.permute(2, 0, 1).unsqueeze(0)
        grid = torch.nn.functional.unfold(image_grid, kernel_size=merge, stride=merge)
        grid = grid.view(self.hidden_size * merge * merge, -1).t()
        return self.merging_layer(grid)


class Mistral3CPUProjector(nn.Module):
    """CPU-side vision-to-text projector: RMSNorm -> PatchMerger -> 2-layer MLP."""

    def __init__(
        self,
        vision_hidden_size,
        text_hidden_size,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(vision_hidden_size, eps=1e-5)
        self.patch_merger = Mistral3PatchMerger(vision_hidden_size, spatial_merge_size)
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=False)

    def forward(self, vision_features, image_h, image_w):
        ph = image_h // PATCH_SIZE
        pw = image_w // PATCH_SIZE
        feats = vision_features.squeeze(0)
        feats = self.norm(feats.float()).to(feats.dtype)
        feats = self.patch_merger(feats, ph, pw)
        feats = self.linear_1(feats)
        feats = self.act(feats)
        feats = self.linear_2(feats)
        return feats


def _load_cpu_projector(model_path, vision_hidden_size, text_hidden_size):
    """Load Mistral3CPUProjector weights from safetensors checkpoint."""
    projector = Mistral3CPUProjector(vision_hidden_size, text_hidden_size)
    weight_mapping = {
        "multi_modal_projector.norm.weight": "norm.weight",
        "multi_modal_projector.patch_merger.merging_layer.weight": "patch_merger.merging_layer.weight",
        "multi_modal_projector.linear_1.weight": "linear_1.weight",
        "multi_modal_projector.linear_2.weight": "linear_2.weight",
    }
    safetensors_files = sorted(
        f
        for f in os.listdir(model_path)
        if f.endswith(".safetensors") and "consolidated" not in f
    )
    for fname in safetensors_files:
        with safe_open(os.path.join(model_path, fname), framework="pt") as f:
            for key in f.keys():
                if key in weight_mapping:
                    target_key = weight_mapping[key]
                    parts = target_key.split(".")
                    module = projector
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    existing = getattr(module, parts[-1])
                    with torch.no_grad():
                        existing.copy_(f.get_tensor(key).to(existing.dtype))
    projector.eval().to(torch.bfloat16)
    return projector


# ---------------------------------------------------------------------------
# Inference Config
# ---------------------------------------------------------------------------


class LeanstralInferenceConfig(ImageToTextInferenceConfig):
    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )
        self.validate_vision_model_supported_configs()

        # Mistral3 HF config uses "image_token_id" but NxDI expects "image_token_index"
        if not hasattr(self, "image_token_index"):
            self.image_token_index = getattr(self, "image_token_id", IMAGE_TOKEN_ID)

        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Leanstral does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Leanstral does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Leanstral does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Leanstral does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Leanstral does not yet support fused speculation.")

        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads = self.text_config.num_attention_heads
            num_kv_heads = self.text_config.num_key_value_heads
            num_attn_heads = (
                num_attn_heads // self.neuron_config.tp_degree + 1
            ) * self.neuron_config.tp_degree
            self.text_config.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def validate_vision_model_supported_configs(self):
        LEANSTRAL_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG = [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "attn_kernel_enabled",
            "fused_qkv",
            "qkv_kernel_enabled",
            "mlp_kernel_enabled",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
        ]
        for unsupported_config in LEANSTRAL_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG:
            if (
                getattr(self.vision_config.neuron_config, unsupported_config, False)
                is not False
            ):
                setattr(self.vision_config.neuron_config, unsupported_config, False)
                logger.warning(
                    f"Leanstral vision model does not yet support '{unsupported_config}'. Will be disabled."
                )

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.pad_token_id",
            "text_config.vocab_size",
            "text_config.max_position_embeddings",
            "text_config.rope_theta",
            "text_config.rms_norm_eps",
            "text_config.hidden_act",
            "vision_config.image_size",
            "vision_config.patch_size",
            "vision_config.num_hidden_layers",
            "vision_config.num_channels",
            "vision_config.hidden_size",
            "vision_config.num_attention_heads",
            "vision_config.rope_theta",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ---------------------------------------------------------------------------
# Model Classes
# ---------------------------------------------------------------------------


class NeuronLeanstralTextModel(NeuronLlamaModel):
    """Llama text model with vision embedding injection via scatter_by_index_put."""

    def encode_vision_to_input(
        self, inputs_embeds, vision_embeddings, vision_mask
    ) -> torch.Tensor:
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)


class NeuronLeanstralVisionModel(NeuronPixtralVisionModel):
    """Pixtral ViT without built-in projector (Leanstral uses CPU PatchMerger)."""

    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, "multi_modal_projector"):
            del self.multi_modal_projector

    def forward(self, patch_embeds, attention_mask, position_ids):
        patch_embeds = self.vision_patch_conv_linear(patch_embeds)
        patch_embeds = self.vision_ln_pre(patch_embeds)
        return self.vision_transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


class LeanstralVisionWrapper(PixtralVisionModelWrapper):
    """Fix unpad slice for no-projector output."""

    def pad_inputs(self, patch_embeds, attention_mask, position_ids):
        result = super().pad_inputs(patch_embeds, attention_mask, position_ids)
        if self.original_patch_embed_slices is not None:
            self.original_patch_embed_slices[-1][-1] = (
                self.config.vision_config.hidden_size
            )
        return result


class NeuronLeanstralForCausalLM(NeuronBaseForImageToText):
    """Full VL model: Pixtral vision + Llama text + CPU PatchMerger projector.

    NKI kernel optimizations enabled by default in vllm-neuron:
    - fused_qkv + qkv_nki_kernel_enabled (+14-17% decode throughput)
    - SHARD_OVER_HEADS GQA (kv_heads sharded across TP ranks, not replicated)
    - Multi-KV-head TKG kernel adapter (virtual batch approach)

    NOT enabled by default (opt-in via neuron_config.json):
    - attn_block_tkg_nki_kernel_enabled: Blocked by neuronx-cc 2.28 compiler ICE
      (NCC_ITEN404) on TKG buckets >= 512 with multi-KV-head SHARD_OVER_HEADS.
    """

    text_model_cls = NeuronLeanstralTextModel
    vision_model_cls = NeuronLeanstralVisionModel
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = LeanstralVisionWrapper

    def __init__(self, *args, **kwargs):
        # Apply patches before any model construction
        _ensure_patches_applied()

        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )
        # Load the CPU projector if we have a model path (not loading from compiled artifacts)
        model_path = args[0] if args else kwargs.get("model_path")
        if (
            model_path
            and os.path.isdir(model_path)
            and os.path.exists(os.path.join(model_path, "config.json"))
        ):
            self.cpu_projector = _load_cpu_projector(
                model_path,
                vision_hidden_size=self.config.vision_config.hidden_size,
                text_hidden_size=self.config.text_config.hidden_size,
            )
        else:
            self.cpu_projector = None

    @classmethod
    def get_config_cls(cls):
        return LeanstralInferenceConfig

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig

    @staticmethod
    def load_pretrained_config(model_path):
        """Build a load_config callable for Mistral3, bypassing AutoConfig.

        AutoConfig fails for Mistral3 because the text sub-config uses
        model_type="ministral3" which is not registered in the transformers
        CONFIG_MAPPING. We load config.json directly and build SimpleNamespace
        objects manually.
        """
        import json
        from types import SimpleNamespace

        with open(os.path.join(model_path, "config.json")) as f:
            full_config = json.load(f)
        text_cfg = full_config.get("text_config", {})
        vision_cfg = full_config.get("vision_config", {})
        rope_params = text_cfg.get("rope_scaling", {})

        def load_config(config_obj):
            rope_scaling_dict = {
                "rope_type": rope_params.get("rope_type", "yarn"),
                "type": rope_params.get("type", "yarn"),
                "factor": rope_params.get("factor", 16.0),
                "beta_fast": rope_params.get("beta_fast", 32.0),
                "beta_slow": rope_params.get("beta_slow", 1.0),
                "original_max_position_embeddings": rope_params.get(
                    "original_max_position_embeddings", 16384
                ),
                "mscale": rope_params.get("mscale", 1.0),
                "mscale_all_dim": rope_params.get("mscale_all_dim", 1.0),
            }
            tc = SimpleNamespace(
                hidden_size=text_cfg["hidden_size"],
                num_attention_heads=text_cfg["num_attention_heads"],
                num_hidden_layers=text_cfg["num_hidden_layers"],
                num_key_value_heads=text_cfg["num_key_value_heads"],
                vocab_size=text_cfg["vocab_size"],
                max_position_embeddings=text_cfg["max_position_embeddings"],
                rope_theta=rope_params.get("rope_theta", 1e9),
                rope_scaling=rope_scaling_dict,
                rms_norm_eps=text_cfg["rms_norm_eps"],
                hidden_act=text_cfg["hidden_act"],
                intermediate_size=text_cfg["intermediate_size"],
                head_dim=text_cfg.get("head_dim", 128),
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=full_config.get("tie_word_embeddings", False),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            vc = SimpleNamespace(
                hidden_size=vision_cfg["hidden_size"],
                num_attention_heads=vision_cfg["num_attention_heads"],
                num_hidden_layers=vision_cfg["num_hidden_layers"],
                num_channels=vision_cfg["num_channels"],
                patch_size=vision_cfg["patch_size"],
                image_size=vision_cfg["image_size"],
                rope_theta=vision_cfg.get("rope_parameters", {}).get(
                    "rope_theta", 10000.0
                ),
                head_dim=vision_cfg.get("head_dim", 64),
                intermediate_size=vision_cfg.get("intermediate_size", 4096),
                hidden_act=vision_cfg.get("hidden_act", "silu"),
            )
            config_obj.text_config = tc
            config_obj.vision_config = vc
            config_obj.multimodal_projector_bias = False
            config_obj.projector_hidden_act = "gelu"
            config_obj.vision_feature_layer = -1
            config_obj.spatial_merge_size = full_config.get(
                "spatial_merge_size", SPATIAL_MERGE_SIZE
            )
            config_obj.image_token_index = IMAGE_TOKEN_ID
            config_obj._name_or_path = model_path
            # vllm-neuron checks model_type to dispatch multimodal data processing.
            # Leanstral uses the Pixtral/Llava VL pipeline. vllm's Pixtral preprocessor
            # produces "images" key, while NxDI's Llava handler expects "pixel_values".
            # We use "leanstral" model_type so vllm-neuron can remap keys before
            # delegating to the Llava processing path.
            config_obj.model_type = "leanstral"
            config_obj.output_attentions = False
            config_obj.output_hidden_states = False
            config_obj.return_dict = True

        return load_config

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = (
            self.vision_config.neuron_config.cc_pipeline_tiling_factor
        )
        return (
            f"--enable-saturate-infinity --auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} ' -O1 "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = (
            self.text_config.neuron_config.cc_pipeline_tiling_factor
        )
        return (
            f"--enable-saturate-infinity --auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} --vectorize-strided-dma ' -O1 "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def enable_vision_encoder(
        self, enable_wlt_optimization: bool = True, **model_init_kwargs
    ):
        new_config = copy.deepcopy(self.config)
        if new_config.vision_config.neuron_config.enable_bucketing:
            vc_nc = new_config.vision_config.neuron_config
            if vc_nc.buckets == [vc_nc.seq_len] or vc_nc.buckets is None:
                if vc_nc.seq_len > 1024:
                    vc_nc.buckets = autobucketing.generate_buckets(1024, vc_nc.seq_len)
                else:
                    vc_nc.buckets = [vc_nc.seq_len]
        new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=True,
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """Convert HuggingFace Ministral3 weights to NxDI format.

        Follows the Pixtral conversion pattern exactly:
        1. FP8 dequantization (Ministral3-specific — the checkpoint uses FP8)
        2. Strip HF prefixes and remap attention keys (same as Pixtral)
        3. Call NeuronLlamaForCausalLM converter for text (handles QKV fusion, rank utils)
        4. Call NeuronPixtralForImageEncoding converter for vision (handles key rename, dtype)

        Projector keys (multi_modal_projector.*) are filtered out here because
        Leanstral's CPU projector loads them separately via _load_cpu_projector().
        """
        # Phase 1: FP8 dequantize in-place
        fp8_count = 0
        keys_to_remove = []
        for key, val in state_dict.items():
            if ".activation_scale" in key or ".weight_scale_inv" in key:
                keys_to_remove.append(key)
                continue
            if val.dtype == torch.float8_e4m3fn:
                scale_key = key.replace(".weight", ".weight_scale_inv")
                scale = (
                    state_dict[scale_key].float()
                    if scale_key in state_dict
                    else torch.tensor(1.0)
                )
                state_dict[key] = (val.float() * scale).to(torch.bfloat16)
                fp8_count += 1
        for key in keys_to_remove:
            del state_dict[key]
        logger.info("Dequantized %d FP8 tensors to bf16", fp8_count)

        # Phase 2: Strip HF prefixes, remap attention keys for non-fused case
        # Follows Pixtral's convert_hf_to_neuron_state_dict pattern exactly.
        # Vision keys keep their "vision_tower." prefix for the Pixtral vision converter.
        # Projector keys are filtered out (loaded separately by CPU projector).
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        }
        new_state_dict = {}
        for dict_key in state_dict:
            # Skip projector keys — loaded separately by _load_cpu_projector
            if "multi_modal_projector" in dict_key:
                continue
            if "language_model.model." in dict_key:
                new_key = dict_key.replace("language_model.model.", "")
                if not inference_config.neuron_config.fused_qkv:
                    for atten_key in attention_keys:
                        if atten_key in new_key:
                            new_key = new_key.replace(
                                atten_key, attention_keys[atten_key]
                            )
                new_state_dict[new_key] = state_dict[dict_key]
            elif "language_model." in dict_key:
                new_key = dict_key.replace("language_model.", "")
                new_state_dict[new_key] = state_dict[dict_key]
            else:
                new_state_dict[dict_key] = state_dict[dict_key]

        # Phase 3: Text model conversion (QKV fusion, rank utils, etc.)
        from neuronx_distributed_inference.models.llama.modeling_llama import (
            NeuronLlamaForCausalLM,
        )

        state_dict = NeuronLlamaForCausalLM.convert_hf_to_neuron_state_dict(
            new_state_dict, inference_config.text_config
        )

        # Phase 4: Vision model conversion (key rename, dtype cast, patch_conv reshape)
        state_dict = NeuronPixtralForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config
        )

        return state_dict

    def get_required_kwargs(self) -> List[str]:
        return ["pixel_values", "vision_mask", "image_sizes"]

    def get_padding_length(self, input_ids):
        buckets = self.context_encoding_model.config.neuron_config.buckets
        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise ValueError(
            f"No bucket found for input_ids length {input_ids.shape[1]}. "
            f"Available buckets: {buckets}"
        )

    def concat_causal_lm_outputs(self, outputs_list):
        concatenated_logits = []
        concatenated_hidden_states = []
        concatenated_tokens = []
        for output in outputs_list:
            if isinstance(output.logits, torch.Tensor):
                concatenated_logits.append(output.logits)
            if isinstance(output.hidden_states, torch.Tensor):
                concatenated_hidden_states.append(output.hidden_states)
            elif isinstance(output.hidden_states, list):
                concatenated_hidden_states.extend(output.hidden_states)
            if hasattr(output, "tokens") and isinstance(output.tokens, torch.Tensor):
                concatenated_tokens.append(output.tokens)
        concatenated_logits = (
            torch.cat(concatenated_logits, dim=0)
            if len(concatenated_logits) > 0
            else None
        )
        concatenated_tokens = (
            torch.cat(concatenated_tokens, dim=0) if len(concatenated_tokens) else None
        )

        concatentated_output = CausalLMOutputWithPast(
            logits=concatenated_logits,
            hidden_states=concatenated_hidden_states,
        )
        if concatenated_tokens is not None:
            concatentated_output.tokens = concatenated_tokens
        return concatentated_output

    def forward_atomic_prefill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
    ):
        """Run vision encoder + CPU projector + text prefill for one batch item."""
        if image_sizes is None:
            assert len(pixel_values.shape) == 4
            img_h = pixel_values.shape[2]
            img_w = pixel_values.shape[3]
            image_sizes = torch.tensor([[img_h, img_w]], dtype=torch.int32)

        if vision_mask is None:
            vision_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
        vision_mask = generate_positions_from_mask(vision_mask.squeeze())

        assert image_sizes.dtype in [torch.int, torch.int32, torch.int64], (
            f"Parameter `image_sizes` must be of type int, received {image_sizes.dtype}"
        )

        # 1. Vision encoder (on Neuron)
        vision_embeddings = self.vision_encoder_model(
            pixel_values.to(self.vision_config.neuron_config.torch_dtype), image_sizes
        )

        # 2. CPU projector: RMSNorm -> PatchMerger -> MLP
        if self.cpu_projector is not None:
            img_h = image_sizes[0, 0].item()
            img_w = image_sizes[0, 1].item()
            with torch.no_grad():
                projected = self.cpu_projector(vision_embeddings, img_h, img_w)
            vision_embeddings = projected.unsqueeze(0).to(
                self.text_config.neuron_config.torch_dtype
            )
        else:
            vision_embeddings = vision_embeddings.to(
                self.text_config.neuron_config.torch_dtype
            )

        # 3. Pad to text bucket
        pad_limit = self.get_padding_length(input_ids)
        vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))
        vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

    def check_empty_pixel_values(self, pixel_values):
        if pixel_values is None:
            return True
        elif isinstance(pixel_values, torch.Tensor):
            return pixel_values.sum() == 0
        elif isinstance(pixel_values, list):
            for pixel_value in pixel_values:
                if pixel_value.sum() != 0:
                    return False
            return True
        else:
            raise ValueError(
                f"Unsupported type for pixel_values {type(pixel_values)}, expecting list, tensor, or None."
            )

    def get_batch_line_mm_input(self, mm_input, index):
        if mm_input is None:
            return None
        elif isinstance(mm_input, list):
            return mm_input[index]
        elif isinstance(mm_input, torch.Tensor):
            return mm_input[index].unsqueeze(0)
        else:
            raise ValueError(
                f"Unsupported type for mm_input:{type(mm_input)}, expecting list, tensor, or None."
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[
            Union[torch.FloatTensor, List[torch.FloatTensor]]
        ] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if input_ids.shape[-1] > 1 and not self.check_empty_pixel_values(pixel_values):
            # VL prefill: process each batch item through vision pipeline
            outputs = []
            for index in range(input_ids.shape[0]):
                outputs.append(
                    self.forward_atomic_prefill(
                        input_ids[index].unsqueeze(0),
                        attention_mask[index].unsqueeze(0)
                        if attention_mask is not None
                        else attention_mask,
                        position_ids[index].unsqueeze(0)
                        if position_ids is not None
                        else position_ids,
                        seq_ids[index].unsqueeze(0) if seq_ids is not None else seq_ids,
                        sampling_params[index].unsqueeze(0)
                        if sampling_params is not None
                        else sampling_params,
                        self.get_batch_line_mm_input(pixel_values, index),
                        self.get_batch_line_mm_input(vision_mask, index),
                        self.get_batch_line_mm_input(image_sizes, index),
                    )
                )
            return self.concat_causal_lm_outputs(outputs)
        else:
            # Text-only prefill or TKG decode
            pad_limit = (
                self.get_padding_length(input_ids) if input_ids.shape[-1] > 1 else 1
            )
            vision_embeddings, vision_mask = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask,
            )

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        """Load the Ministral3 checkpoint. Tries HuggingFace AutoModel first,
        falls back to direct safetensors loading."""
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        except Exception:
            # Fallback: direct safetensors loading for older transformers versions
            state_dict = {}
            safetensors_files = sorted(
                f
                for f in os.listdir(model_path)
                if f.endswith(".safetensors") and "consolidated" not in f
            )
            for fname in safetensors_files:
                with safe_open(os.path.join(model_path, fname), framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

            class _StateDict:
                def __init__(self, sd):
                    self._sd = sd

                def state_dict(self):
                    return self._sd

            return _StateDict(state_dict)

    def to_cpu(self):
        raise NotImplementedError("to_cpu() is not implemented")
