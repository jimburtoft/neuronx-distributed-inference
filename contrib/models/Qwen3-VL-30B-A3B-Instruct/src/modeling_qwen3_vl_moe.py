# coding=utf-8
# Qwen3-VL-MoE VL Orchestrator for NxDI
# Top-level model combining qwen3_vl vision encoder with qwen3_moe text decoder

import copy
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from neuronx_distributed_inference.modules.autobucketing import generate_buckets
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    VISION_ENCODER_MODEL_TAG,
)

# Reuse qwen3_vl vision encoder directly (identical architecture)
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
    NeuronQwen3VLForImageEncoding,
    NeuronQwen3VLVisionModel,
    NeuronQwen3VLVisionModelWrapper,
)
from neuronx_distributed_inference.modules.flashdecode.utils import (
    calculate_num_cores_per_group,
)

# Import our MoE text model
from modeling_qwen3_vl_moe_text import (
    NeuronQwen3VLMoeTextForCausalLM,
    NeuronQwen3VLMoeTextModel,
    NeuronQwen3VLMoeTextModelWrapper,
    Qwen3VLMoeInferenceConfig,
)

logger = logging.getLogger("Neuron")


# =============================================================================
# NeuronConfig for VL -- thin wrapper
# =============================================================================


class Qwen3VLMoeNeuronConfig(NeuronConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


# =============================================================================
# VL + MoE Inference Config
# =============================================================================


class Qwen3VLMoeVLInferenceConfig(ImageToTextInferenceConfig):
    """Combined VL inference config for Qwen3-VL-MoE.

    Text config uses MoENeuronConfig (from Qwen3VLMoeInferenceConfig).
    Vision config uses standard NeuronConfig.
    """

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

        # CRITICAL: ImageToTextInferenceConfig creates self.text_config as a plain
        # InferenceConfig, losing our MoE-specific init. Apply MoE settings here.
        self._apply_moe_text_config()

        # Validate vision config
        self.validate_vision_model_supported_configs()

        # Copy deepstack_visual_indexes from vision to text config
        setattr(
            self.text_config,
            "deepstack_visual_indexes",
            copy.deepcopy(self.vision_config.deepstack_visual_indexes),
        )

        # Unsupported features
        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Qwen3VLMoe does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Qwen3VLMoe does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Qwen3VLMoe does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Qwen3VLMoe does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Qwen3VLMoe does not yet support fused speculation.")
        if self.text_config.neuron_config.attention_dp_degree > 1:
            raise ValueError("Qwen3VLMoe does not yet support attention data parallel")
        if self.text_config.neuron_config.cp_degree > 1:
            raise ValueError("Qwen3VLMoe does not yet support context parallel")
        if self.text_config.neuron_config.seq_len > 10240:
            os.environ["NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE"] = (
                f"{140 * 1024 * 1024}"
            )

        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads = self.text_config.num_attention_heads
            num_kv_heads = self.text_config.num_key_value_heads
            num_attn_heads = (
                num_attn_heads // self.neuron_config.tp_degree + 1
            ) * self.neuron_config.tp_degree
            self.text_config.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

        # Vision bucketing
        if not self.vision_config.neuron_config.enable_bucketing:
            VISION_SEQ_LENGTH = self.vision_config.neuron_config.seq_len
            self.vision_config.neuron_config.enable_bucketing = True
            self.vision_config.neuron_config.buckets = generate_buckets(
                VISION_SEQ_LENGTH, VISION_SEQ_LENGTH
            )

        logger.info(
            f"Bucketing Qwen3 VL MoE vision model on seq len. Buckets: {self.vision_config.neuron_config.buckets}"
        )

        # Ensure text context >= compressed vision seq len
        vision_seq_len_to_text = self.vision_config.neuron_config.seq_len // (
            self.vision_config.spatial_merge_size**2
        )
        assert (
            self.text_config.neuron_config.max_context_length >= vision_seq_len_to_text
        ), (
            f"Text max context {self.text_config.neuron_config.max_context_length} < compressed vision seq len {vision_seq_len_to_text}"
        )

    def validate_vision_model_supported_configs(self):
        unsupported = [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "mlp_kernel_enabled",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
            "attn_kernel_enabled",
        ]
        for config_name in unsupported:
            if (
                getattr(self.vision_config.neuron_config, config_name, False)
                is not False
            ):
                setattr(self.vision_config.neuron_config, config_name, False)
                logger.warning(
                    f"Qwen3VLMoe vision does not support '{config_name}'. Disabled."
                )

    def _apply_moe_text_config(self):
        """Apply MoE-specific settings to text_config.

        ImageToTextInferenceConfig creates text_config as a plain InferenceConfig,
        so we must manually apply the MoE fields that Qwen3VLMoeInferenceConfig
        would normally set in its __init__.
        """
        import math
        from neuronx_distributed_inference.models.config import (
            SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
            MOE_TKG_MK_INTERMEDIATE_PER_TP,
        )

        tc = self.text_config

        # Core MoE fields
        tc.num_local_experts = tc.num_experts
        tc.n_shared_experts = 0
        tc.intermediate_size = tc.moe_intermediate_size

        # GLU MLP assertion
        assert tc.neuron_config.glu_mlp is True, "Only GLU MLP is supported for MoE"

        # Intermediate padding for MoE
        moe_tp_degree = tc.neuron_config.moe_tp_degree
        I_TP = tc.moe_intermediate_size // moe_tp_degree
        if getattr(
            tc.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded = (
                    math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                tc.moe_intermediate_pad_size = max(padded - tc.moe_intermediate_size, 0)
                tc.moe_intermediate_size = padded
                tc.intermediate_size = tc.moe_intermediate_size

        # Pad for fused TKG mega-kernel (MOE_TKG_MK_INTERMEDIATE_PER_TP=128)
        I_TP = tc.moe_intermediate_size // moe_tp_degree
        if getattr(tc.neuron_config, "moe_fused_nki_kernel_enabled", False):
            if I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP != 0:
                padded = (
                    math.ceil(I_TP / MOE_TKG_MK_INTERMEDIATE_PER_TP)
                    * MOE_TKG_MK_INTERMEDIATE_PER_TP
                    * moe_tp_degree
                )
                pad_size = max(padded - tc.moe_intermediate_size, 0)
                existing_pad = getattr(tc, "moe_intermediate_pad_size", 0)
                tc.moe_intermediate_pad_size = existing_pad + pad_size
                tc.moe_intermediate_size = padded
                tc.intermediate_size = tc.moe_intermediate_size
                logger.info(
                    f"Padded moe_intermediate_size for fused TKG kernel: "
                    f"{tc.moe_intermediate_size - pad_size} -> {tc.moe_intermediate_size} "
                    f"(I_TP: {(tc.moe_intermediate_size - pad_size) // moe_tp_degree} -> "
                    f"{tc.moe_intermediate_size // moe_tp_degree}, pad={pad_size})"
                )
            # Now I_TP should be divisible — enable the kernel flag
            I_TP = tc.moe_intermediate_size // moe_tp_degree
            if I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0:
                tc.moe_fused_nki_kernel_enabled = True
                logger.info("Fused TKG mega-kernel ENABLED for MoE token generation")
            else:
                logger.warning(
                    f"Fused TKG requested but I_TP={I_TP} still not divisible by "
                    f"{MOE_TKG_MK_INTERMEDIATE_PER_TP} after padding. Kernel DISABLED."
                )

        # Router config
        tc.neuron_config.router_config.dtype = torch.float32
        tc.neuron_config.router_config.act_fn = "softmax"
        tc.neuron_config.disable_numeric_cc_token = True
        tc.neuron_config.normalize_top_k_affinities = True

        logger.info(
            f"MoE text config applied: num_experts={tc.num_experts}, "
            f"top_k={tc.num_experts_per_tok}, intermediate={tc.intermediate_size}, "
            f"moe_tp={moe_tp_degree}, moe_ep={tc.neuron_config.moe_ep_degree}"
        )

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.attention_bias",
            "text_config.attention_dropout",
            "text_config.num_attention_heads",
            "text_config.bos_token_id",
            "text_config.dtype",
            "text_config.eos_token_id",
            "text_config.head_dim",
            "text_config.hidden_act",
            "text_config.hidden_size",
            "text_config.initializer_range",
            "text_config.max_position_embeddings",
            "text_config.moe_intermediate_size",
            "text_config.num_experts",
            "text_config.num_experts_per_tok",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.rms_norm_eps",
            "text_config.rope_scaling",
            "text_config.rope_theta",
            "text_config.vocab_size",
            "vision_config.deepstack_visual_indexes",
            "vision_config.depth",
            "vision_config.hidden_act",
            "vision_config.hidden_size",
            "vision_config.in_channels",
            "vision_config.initializer_range",
            "vision_config.intermediate_size",
            "vision_config.num_heads",
            "vision_config.num_position_embeddings",
            "vision_config.out_hidden_size",
            "vision_config.patch_size",
            "vision_config.spatial_merge_size",
            "vision_config.temporal_patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Qwen3VLMoeNeuronConfig


# =============================================================================
# Top-level VL Model
# =============================================================================


class NeuronQwen3VLMoeForCausalLM(NeuronBaseForImageToText):
    """Qwen3-VL-MoE: VL model with MoE text decoder.

    Reuses qwen3_vl vision encoder and VL pipeline.
    Text decoder uses MoE routing (128 experts, top-8).
    """

    # Vision model: reuse from qwen3_vl (identical architecture)
    vision_model_cls = NeuronQwen3VLVisionModel
    vision_model_wrapper = NeuronQwen3VLVisionModelWrapper

    # Text model: MoE variant
    text_model_cls = NeuronQwen3VLMoeTextModel
    text_model_wrapper = NeuronQwen3VLMoeTextModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )
        self.rope_deltas = None

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = (
            self.vision_config.neuron_config.cc_pipeline_tiling_factor
        )
        return (
            f"--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 "
            f"--internal-max-instruction-limit=15000000"
        )

    def get_compiler_args(self) -> str:
        """Compiler args for MoE text model -- merged from qwen3_moe and qwen3_vl."""
        cc_pipeline_tiling_factor = (
            self.text_config.neuron_config.cc_pipeline_tiling_factor
        )

        if (
            hasattr(self, "compile_tag")
            and self.compile_tag == TOKEN_GENERATION_MODEL_TAG
        ):
            optimization_level = (
                "-O3" if self.text_config.neuron_config.moe_ep_degree > 1 else "-O1"
            )
        else:
            optimization_level = "-O1"

        compiler_args = (
            f"--enable-saturate-infinity --enable-mixed-precision-accumulation "
            f"--model-type transformer {optimization_level} "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' "
            f"--auto-cast=none "
            f"--internal-enable-dge-levels vector_dynamic_offsets "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true' "
            f"--internal-max-instruction-limit=15000000"
        )

        if self.text_config.neuron_config.scratchpad_page_size:
            compiler_args += f" --hbm-scratchpad-page-size={self.text_config.neuron_config.scratchpad_page_size}"

        return compiler_args

    def get_required_kwargs(self) -> List[str]:
        return ["pixel_values", "image_grid_thw"]

    def enable_vision_encoder(
        self, enable_wlt_optimization: bool = True, **model_init_kwargs
    ):
        new_config = copy.deepcopy(self.config)
        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=False,
            return_ranked_to_cpu=True,
        )
        self.vision_models.append(self.vision_encoder_model)

    # Wraps enable_context_encoding/token_generation to add compile_tag
    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration

            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path, **kwargs
            )
        except ImportError:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **kwargs
            )
        return model

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, inference_config) -> dict:
        # Chain: vision conversion then text conversion
        state_dict = NeuronQwen3VLForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config
        )
        state_dict = NeuronQwen3VLMoeTextForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, inference_config.text_config
        )
        return state_dict

    def get_padding_length(self, input_ids):
        buckets = self.context_encoding_model.config.neuron_config.buckets
        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

    def get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        """Compute 3D mRoPE position IDs. Identical to qwen3_vl implementation."""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            for i, input_ids_i in enumerate(total_input_ids):
                input_ids_i = input_ids_i[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(
                    input_ids_i == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids_i[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    ed_image = (
                        input_tokens.index(image_token_id, st)
                        if image_token_id in input_tokens and remain_images > 0
                        else len(input_tokens) + 1
                    )
                    ed_video = (
                        input_tokens.index(video_token_id, st)
                        if video_token_id in input_tokens and remain_videos > 0
                        else len(input_tokens) + 1
                    )

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = t.item()
                    llm_grid_h = h.item() // spatial_merge_size
                    llm_grid_w = w.item() // spatial_merge_size
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = (
                    torch.cat(llm_pos_ids_list, dim=1)
                    .reshape(3, -1)
                    .to(total_input_ids.dtype)
                )
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        vision_attention_mask: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        pad_limit = self.get_padding_length(input_ids)

        if (
            pixel_values is not None
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0
        ):
            # Vision+Text Prefill
            vision_mask = (
                (input_ids == self.config.image_token_id).unsqueeze(-1).to(torch.bool)
            )
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))

            vision_embeddings, deepstack_vision_embeds = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
                image_grid_thw,
            )
            vision_embeddings = vision_embeddings.to(
                self.text_config.neuron_config.torch_dtype
            )
            embedding_dim = vision_embeddings.shape[-1]
            vision_embeddings = vision_embeddings.view(-1, embedding_dim).unsqueeze(0)
            vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)

            for i, feat in enumerate(deepstack_vision_embeds):
                feat = (
                    feat.view(-1, embedding_dim)
                    .unsqueeze(0)
                    .to(self.text_config.neuron_config.torch_dtype)
                )
                deepstack_vision_embeds[i] = pad_vision_embeddings(feat, pad_limit)
            deepstack_vision_embeds = torch.stack(deepstack_vision_embeds)
        else:
            # Text-only Prefill or Decode
            vision_embeddings, vision_mask, deepstack_vision_embeds = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )

        # mRoPE computation
        if input_ids.shape[-1] > 1:
            rotary_position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size = input_ids.shape[0]
            if self.rope_deltas is not None:
                delta = self.rope_deltas.to(input_ids.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            else:
                delta = 0
            rotary_position_ids = copy.deepcopy(position_ids)
            rotary_position_ids = (
                rotary_position_ids.view(1, -1).expand(batch_size, -1).add(delta)
            )
            rotary_position_ids = rotary_position_ids.unsqueeze(0).expand(3, -1, -1)

        output_token = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            input_capture_hook=input_capture_hook,
            tensor_capture_hook=tensor_capture_hook,
            rotary_position_ids=rotary_position_ids,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            deepstack_vision_embeds=deepstack_vision_embeds,
        )
        return output_token

    @classmethod
    def get_config_cls(cls):
        return Qwen3VLMoeVLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        return NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts, images, processor, role, config
        )
