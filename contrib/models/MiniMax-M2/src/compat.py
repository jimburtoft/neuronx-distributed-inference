# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FP8 in-graph dequant compatibility patch for MiniMax-M2 at TP=32 (no EP).

When running MiniMax-M2 with TP=32 and no expert parallelism (all 256 experts
on each rank), the MoE dispatch paths are:

  CTE (context encoding, seq_len > 1):
    ExpertMLPsV2.forward_blockwise -> BlockwiseMatmulNKIFunc -> _call_shard_hidden_kernel

  TKG (token generation, seq_len = 1):
    ExpertMLPsV2.forward_selective_loading -> Experts.forward ->
      ExpertFusedColumnParallelLinear.forward -> torch.einsum (bare FP8 matmul)

Problem: NKI custom calls don't register FP8 scale tensors in XLA's computation
graph, so XLA eliminates them during HLO generation. The TKG path via
ExpertFusedLinear also does bare FP8 matmul without scale application.

Solution: Two patches that dequant FP8->BF16 in-graph using PyTorch ops:
  1. _patch_blockwise_shard_hidden: For the CTE path, dequants before the
     shard_hidden NKI kernel.
  2. _patch_expert_fused_linear_forward: For the TKG path, dequants inside
     ExpertFusedColumnParallelLinear.forward and ExpertFusedRowParallelLinear.forward
     AFTER expert_indices slicing (only top-k experts, not all 256).

This ensures both FP8 weights AND FP32 scales are traced through XLA and
appear in the compiled NEFF.  Weights are stored as FP8 in HBM (half the
memory), and the dequant happens at compute time inside the NEFF.

Usage:
    import compat  # patches are applied on import
"""

import logging
import importlib

import torch

logger = logging.getLogger(__name__)


def _patch_blockwise_shard_hidden():
    """Patch NxD blockwise.py _call_shard_hidden_kernel if it's a stub.

    NxD 0.18 (SDK 2.29) removed the neuronxcc.nki._private.blockwise_mm
    module, leaving _call_shard_hidden_kernel as a stub. This restores
    it from nkilib and adds FP8 in-graph dequant support.
    """
    try:
        import neuronx_distributed.modules.moe.blockwise as bw
    except ImportError:
        logger.debug(
            "neuronx_distributed.modules.moe.blockwise not available, skipping patch"
        )
        return False

    # Check if the function is a stub (raises NotImplementedError)
    try:
        bw._call_shard_hidden_kernel(None)
    except NotImplementedError:
        pass  # Confirmed stub, proceed with patch
    except (TypeError, AttributeError):
        logger.debug("_call_shard_hidden_kernel appears functional, skipping patch")
        return False

    try:
        mod = importlib.import_module("nkilib.experimental.moe.forward.bwmm_shard_on_H")
        kernel_fn = getattr(mod, "blockwise_mm_baseline_shard_hidden")

        import nki

        wrapped_kernel = nki.jit(kernel_fn, mode="torchxla")
        bw._blockwise_mm_baseline_shard_hidden_nki_call = wrapped_kernel

        def _call_shard_hidden_kernel_patched(args):
            """Call the nkilib shard_hidden kernel for blockwise matmul.

            When FP8 scales are present, dequant in-graph (FP8->BF16 * scale)
            then pass BF16 weights to shard_hidden. This ensures XLA traces
            through both weights and scales.
            """
            import os

            strategy = os.environ.get("COMPAT_FP8_STRATEGY", "dequant_shard_hidden")

            if args.gate_up_proj_scale is not None:
                if strategy == "dequant_shard_hidden":
                    # Dequant FP8->BF16 in XLA graph, then use shard_hidden
                    # gate_up_proj_weight: [E, H, 2, I_TP] FP8 (4D)
                    # gate_up_proj_scale: [E, 2*I_TP] FP32 (2D)
                    gup_w = args.gate_up_proj_weight
                    gup_s = args.gate_up_proj_scale
                    if gup_s.dim() == 3:
                        gup_s = gup_s.squeeze(1)

                    gup_w_bf16 = gup_w.to(torch.bfloat16)
                    E_w = gup_w.shape[0]
                    I_TP_w = gup_w.shape[3]
                    gup_s_4d = (
                        gup_s.reshape(E_w, 2, I_TP_w).unsqueeze(1).to(torch.bfloat16)
                    )
                    args.gate_up_proj_weight = gup_w_bf16 * gup_s_4d

                    # down_proj_weight: [E, I_TP, H] FP8
                    dp_w = args.down_proj_weight.to(torch.bfloat16)
                    dp_s = args.down_proj_scale
                    if dp_s is not None:
                        if dp_s.dim() == 3:
                            dp_s = dp_s.squeeze(1)
                        args.down_proj_weight = dp_w * dp_s.unsqueeze(1).to(
                            torch.bfloat16
                        )
                    else:
                        args.down_proj_weight = dp_w

                    args.gate_up_proj_scale = None
                    args.down_proj_scale = None
                    # Fall through to shard_hidden BF16 path below
                else:
                    # Native FP8: delegate to shard_on_block kernel
                    args.block_sharding_strategy = bw.BlockShardStrategy.PING_PONG
                    if args.gate_up_proj_scale.dim() == 2:
                        args.gate_up_proj_scale = args.gate_up_proj_scale.unsqueeze(1)
                    if (
                        args.down_proj_scale is not None
                        and args.down_proj_scale.dim() == 2
                    ):
                        args.down_proj_scale = args.down_proj_scale.unsqueeze(1)
                    output = bw._call_bwmm_shard_on_block_kernel(args)
                    return output, args.gate_up_activations_T, args.down_activations

            # BF16 path: use shard_hidden kernel
            output = wrapped_kernel[2](
                hidden_states=args.hidden_states,
                expert_affinities_masked=args.expert_affinities_masked,
                gate_up_proj_weight=args.gate_up_proj_weight,
                down_proj_weight=args.down_proj_weight,
                block_size=args.block_size,
                token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
                block_to_expert=args.block_to_expert.to(dtype=torch.int32),
                gate_up_activations_T=args.gate_up_activations_T,
                down_activations=args.down_activations,
                skip_dma=args.skip_dma,
                is_tensor_update_accumulating=args.is_tensor_update_accumulating,
                expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
            )
            return output, args.gate_up_activations_T, args.down_activations

        bw._call_shard_hidden_kernel = _call_shard_hidden_kernel_patched
        logger.info(
            "Patched NxD blockwise.py _call_shard_hidden_kernel with nkilib kernel"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to patch blockwise.py _call_shard_hidden_kernel: {e}")
        return False


def _patch_expert_fused_linear_forward():
    """Patch ExpertFusedColumnParallelLinear and ExpertFusedRowParallelLinear
    to dequant FP8 weights inside their forward methods.

    For the TKG path (forward_selective_loading), the original code does:
      weight = self.weight[expert_indices, :, :]  # FP8
      output = einsum("e...h,ehi->e...i", input, weight)  # bare FP8 matmul

    This patch intercepts after the expert_indices slice and dequants:
      weight_bf16 = weight_fp8.to(bf16) * scale_sliced.unsqueeze(dim).to(bf16)

    Only top-k experts (typically 8) are dequanted, not all 256.
    """
    try:
        from neuronx_distributed.modules.moe.moe_parallel_layers import (
            ExpertFusedColumnParallelLinear,
            ExpertFusedRowParallelLinear,
        )
    except ImportError:
        logger.debug("ExpertFused linear layers not available, skipping forward patch")
        return False

    import os

    # --- Patch ExpertFusedColumnParallelLinear.forward ---
    _orig_col_forward = ExpertFusedColumnParallelLinear.forward

    def _col_forward_with_dequant(self, input_, expert_indices=None, *args_):
        strategy = os.environ.get("COMPAT_FP8_STRATEGY", "dequant_shard_hidden")
        scale = getattr(self, "scale", None)
        is_fp8 = self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

        if is_fp8 and scale is not None and strategy == "dequant_shard_hidden":
            from neuronx_distributed.parallel_layers import mappings

            if (
                self.async_tensor_model_parallel_allreduce
                or self.sequence_parallel_enabled
            ):
                input_parallel = input_
            else:
                input_parallel = mappings.copy_to_tensor_model_parallel_region(
                    input_,
                    process_group=self.tensor_parallel_group,
                )

            if expert_indices is not None:
                weight_fp8 = self.weight[expert_indices, :, :]
                scale_sliced = scale[expert_indices, :]
            else:
                weight_fp8 = self.weight
                scale_sliced = scale

            # Dequant: [E', H, 2*I/tp] * [E', 1, 2*I/tp]
            weight_bf16 = weight_fp8.to(torch.bfloat16) * scale_sliced.unsqueeze(1).to(
                torch.bfloat16
            )

            output = self._forward_impl(
                input=input_parallel,
                weight=weight_bf16,
                bias=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                autograd_func_class=self.autograd_func_class,
                process_group=self.tensor_parallel_group,
            )

            if self.bias is not None:
                if expert_indices is not None:
                    bias = self.bias[expert_indices, :]
                else:
                    bias = self.bias
                bias = bias.unsqueeze(1).unsqueeze(2)
            else:
                bias = None
            output = (output + bias) if bias is not None else output
            return output
        else:
            return _orig_col_forward(self, input_, expert_indices, *args_)

    ExpertFusedColumnParallelLinear.forward = _col_forward_with_dequant

    # --- Patch ExpertFusedRowParallelLinear.forward ---
    _orig_row_forward = ExpertFusedRowParallelLinear.forward

    def _row_forward_with_dequant(self, input_, expert_indices=None):
        strategy = os.environ.get("COMPAT_FP8_STRATEGY", "dequant_shard_hidden")
        scale = getattr(self, "scale", None)
        is_fp8 = self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

        if is_fp8 and scale is not None and strategy == "dequant_shard_hidden":
            from neuronx_distributed.parallel_layers import mappings

            if expert_indices is not None:
                weight_fp8 = self.weight[expert_indices, :, :]
                scale_sliced = scale[expert_indices, :]
            else:
                weight_fp8 = self.weight
                scale_sliced = scale

            # Dequant: [E', I/tp, H] * [E', 1, H]
            weight_bf16 = weight_fp8.to(torch.bfloat16) * scale_sliced.unsqueeze(1).to(
                torch.bfloat16
            )

            output_parallel = self._forward_impl(
                input=input_,
                weight=weight_bf16,
                bias=None,
                async_grad_allreduce=False,
                sequence_parallel_enabled=False,
                autograd_func_class=self.autograd_func_class,
                process_group=self.tensor_parallel_group,
            )

            if self.reduce_output:
                output = mappings.reduce_from_tensor_model_parallel_region(
                    output_parallel,
                    process_group=self.tensor_parallel_group,
                )
            else:
                output = output_parallel

            if self.bias is not None:
                if expert_indices is not None:
                    bias = self.bias[expert_indices, :]
                else:
                    bias = self.bias
                bias = bias.unsqueeze(1).unsqueeze(2)
            else:
                bias = None
            output = (output + bias) if bias is not None else output
            return output
        else:
            return _orig_row_forward(self, input_, expert_indices)

    ExpertFusedRowParallelLinear.forward = _row_forward_with_dequant

    logger.info(
        "Patched ExpertFused linear layers to dequant FP8 after expert_indices slice"
    )
    return True


def _patch_fused_tkg_with_selection_bias(model):
    """Patch MoEFusedTKG._moe_fused_tkg_kernel on each MoE layer to inject
    MiniMax-M2's e_score_correction_bias as selection_bias.

    MiniMax-M2 uses sigmoid routing with a learned per-expert bias for top-K
    expert *selection* (the bias doesn't affect affinity weights). The stock
    MoEFusedTKG kernel doesn't know about this bias. This patch:

    1. Copies the bias from RouterTopKWithBias on each decoder layer
    2. Registers it as a buffer on the fused_tkg module (so it moves to XLA device)
    3. Replaces _moe_fused_tkg_kernel with a version that passes
       selection_bias=bias.unsqueeze(0)  [1, E] to the nkilib moe_block_tkg kernel

    The nkilib kernel (branch feature/selection-bias-routing) adds the bias to
    sigmoid scores for top-K selection only, then uses the unbiased sigmoid
    scores as affinity weights — matching MiniMax-M2's RouterTopKWithBias.forward().

    Args:
        model: NeuronMiniMaxM2ForCausalLM instance (after super().__init__).
    """
    import types

    try:
        from neuronx_distributed.modules.moe.moe_fused_tkg import (
            ExpertAffinityScaleMode,
            ROUTER_ACT_FN_MAPPING,
            get_kernel_activation_func_id,
            ACTFunc,
            ActFnType,
            DEFAULT_SELECTIVE_LOADING_THRESHOLD,
        )
    except ImportError:
        logger.warning(
            "Cannot import moe_fused_tkg components, skipping selection_bias patch"
        )
        return 0

    # Import the nkilib moe_block_tkg kernel directly and wrap with nki.jit.
    # NxDI SDK 2.28 doesn't expose moe_block_tkg_kernel as a module-level export
    # (SDK 2.29+ does). We import from nkilib and wrap ourselves.
    try:
        from nkilib.core.moe_block.moe_block_tkg import moe_block_tkg
        import nki

        moe_block_tkg_kernel = nki.jit(moe_block_tkg, mode="torchxla")
    except ImportError as e:
        logger.warning(
            "Cannot import nkilib moe_block_tkg kernel: %s. "
            "Install nkilib fork (branch feature/selection-bias-routing).",
            e,
        )
        return 0

    # Walk decoder layers to find MoE modules with fused TKG.
    # NxDI model hierarchy: CausalLM -> models[] (wrappers) -> model -> layers[]
    # We need to patch ALL model instances (CTE and TKG have separate models).
    # The TKG model is the one that actually uses MoEFusedTKG.
    all_layers = []
    if hasattr(model, "models"):
        for m in model.models:
            inner = getattr(m, "model", m)
            if hasattr(inner, "layers"):
                all_layers.extend(enumerate(inner.layers))
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        all_layers = list(enumerate(model.model.layers))
    elif hasattr(model, "layers"):
        all_layers = list(enumerate(model.layers))

    if not all_layers:
        logger.warning("Could not find decoder layers on model, skipping patch")
        return 0

    patched_count = 0
    for layer_idx, layer in all_layers:
        # Get the MoE module
        moe_module = getattr(layer, "block_sparse_moe", None)
        if moe_module is None:
            continue

        # The MoE module wraps: moe_module.moe is the NxDI MoE, or it IS the MoE
        # In MiniMax-M2, block_sparse_moe IS the MoE wrapper from initialize_minimax_m2_moe_module
        # which returns MoE(...). The fused_tkg is on moe_module.moe_fused_tkg.
        fused_tkg = getattr(moe_module, "moe_fused_tkg", None)
        if fused_tkg is None or not hasattr(fused_tkg, "_moe_fused_tkg_kernel"):
            continue

        # Get the selection bias from the router
        # The router is on fused_tkg.router (set during MoEFusedTKG.__init__)
        router = getattr(fused_tkg, "router", None)
        if router is None:
            logger.warning("Layer %d: No router on fused_tkg, skipping", layer_idx)
            continue

        bias_param = getattr(router, "e_score_correction_bias", None)
        if bias_param is None:
            logger.warning(
                "Layer %d: Router has no e_score_correction_bias, skipping",
                layer_idx,
            )
            continue

        # Register bias as a buffer on fused_tkg so it's moved to XLA device.
        # Using .data.clone() to detach from the original parameter's graph.
        # NOTE: This buffer holds the init-time placeholder value (arange).
        # The real checkpoint values are loaded into router.e_score_correction_bias
        # via replace_weights(). The replacement kernel accesses the parameter
        # directly (self.router.e_score_correction_bias), not this buffer.
        # The buffer exists only as a backup / debugging aid.
        fused_tkg.register_buffer("_minimax_selection_bias", bias_param.data.clone())

        def _make_replacement_method():
            """Create a replacement for _moe_fused_tkg_kernel with selection_bias."""

            def replacement_moe_fused_tkg_kernel(self, hidden_states, residual=None):
                """Replacement _moe_fused_tkg_kernel that injects selection_bias.

                Based on NxDI stock MoEFusedTKG._moe_fused_tkg_kernel with one
                addition: selection_bias kwarg passed to the NKI kernel.
                """
                import neuronxcc.nki.language as nl

                hidden_states_shape = hidden_states.shape
                if self.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
                    expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
                else:
                    expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE
                local_rank = self.expert_mlps.spmd_rank.get_rank()
                local_ep_rank = (
                    local_rank
                    // self.expert_mlps.moe_tensor_model_parallel_group.size()
                )
                grid = self.logical_nc_config
                (
                    shared_experts_gate_proj_weight,
                    shared_experts_up_proj_weight,
                    shared_experts_down_proj_weight,
                ) = self._slice_shared_experts_weights()

                def get_data(t):
                    return t.data if t is not None and hasattr(t, "data") else t

                # Stock NxDI hardcodes router_mm_dtype=nl.float32.
                # MiniMax-M2's router config dtype is torch.float32, so this matches.
                router_mm_dtype = nl.float32

                # FP8 scales: use getattr since the module may be non-quantized
                # (ExpertFusedColumnParallelLinear without .scale) when
                # modules_to_not_convert excluded expert_mlps from NxDI convert.
                # Scales get set later via replace_weights from checkpoint.
                # During tracing, we need placeholder scales of the right shape.
                gate_up_scale = None
                down_scale = None
                if self.config.quantized:
                    raw_gu_scale = getattr(
                        self.expert_mlps.mlp_op.gate_up_proj, "scale", None
                    )
                    raw_dn_scale = getattr(
                        self.expert_mlps.mlp_op.down_proj, "scale", None
                    )
                    E = self.num_local_experts

                    if raw_gu_scale is not None:
                        gate_up_scale = get_data(raw_gu_scale.view(E, 2, -1))

                    if raw_dn_scale is not None:
                        down_scale = get_data(raw_dn_scale.view(E, -1))

                common_args = dict(
                    inp=get_data(hidden_states),
                    gamma=get_data(self.post_attention_layernorm.weight.unsqueeze(0)),
                    router_weights=get_data(self.router.weight_T),
                    shared_expert_gate_w=get_data(shared_experts_gate_proj_weight),
                    shared_expert_up_w=get_data(shared_experts_up_proj_weight),
                    shared_expert_down_w=get_data(shared_experts_down_proj_weight),
                    expert_gate_up_weights=get_data(
                        self.expert_mlps.mlp_op.gate_up_proj.weight.view(
                            self.num_local_experts, self.hidden_size, 2, -1
                        )
                    ),
                    expert_down_weights=get_data(
                        self.expert_mlps.mlp_op.down_proj.weight
                    ),
                    expert_gate_up_weights_scale=gate_up_scale,
                    expert_down_weights_scale=down_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    top_k=self.num_experts_per_tok,
                    router_act_fn=ROUTER_ACT_FN_MAPPING[self.router.act_fn],
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    router_mm_dtype=router_mm_dtype,
                )

                if (
                    self.expert_mlps.routed_experts_mlp_config.hidden_size_actual
                    is not None
                ):
                    common_args["hidden_actual"] = (
                        self.expert_mlps.routed_experts_mlp_config.hidden_size_actual
                    )

                total_tokens = hidden_states_shape[0] * hidden_states_shape[1]
                perc_experts_loaded = (
                    total_tokens * self.num_experts_per_tok / self.num_local_experts
                )

                kernel_call = moe_block_tkg_kernel
                is_all_expert = (
                    perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD
                )

                if kernel_call:
                    routed_experts_mlp_config = (
                        self.expert_mlps.routed_experts_mlp_config
                    )
                    kernel_activation_func_id = get_kernel_activation_func_id(
                        ACTFunc.validate(routed_experts_mlp_config.hidden_act),
                        routed_experts_mlp_config.glu_type,
                    )
                    optional_kwargs = {}
                    if routed_experts_mlp_config.gate_clamp_upper_limit is not None:
                        optional_kwargs["gate_clamp_upper_limit"] = (
                            routed_experts_mlp_config.gate_clamp_upper_limit
                        )
                    if routed_experts_mlp_config.gate_clamp_lower_limit is not None:
                        optional_kwargs["gate_clamp_lower_limit"] = (
                            routed_experts_mlp_config.gate_clamp_lower_limit
                        )
                    if routed_experts_mlp_config.up_clamp_upper_limit is not None:
                        optional_kwargs["up_clamp_upper_limit"] = (
                            routed_experts_mlp_config.up_clamp_upper_limit
                        )
                    if routed_experts_mlp_config.up_clamp_lower_limit is not None:
                        optional_kwargs["up_clamp_lower_limit"] = (
                            routed_experts_mlp_config.up_clamp_lower_limit
                        )

                    if is_all_expert:
                        optional_kwargs["rank_id"] = get_data(
                            local_ep_rank.reshape(1, 1)
                        )

                    # --- MiniMax-M2 selection_bias ---
                    # Use the router's e_score_correction_bias parameter directly.
                    # This is an nn.Parameter on self.router (RouterTopKWithBias),
                    # so it gets loaded from checkpoint via replace_weights() and
                    # is on the XLA device during tracing.
                    sel_bias = get_data(self.router.e_score_correction_bias)
                    optional_kwargs["selection_bias"] = sel_bias.unsqueeze(
                        0
                    )  # [E] -> [1, E]

                    out, router_logits = kernel_call[grid](
                        **common_args,
                        router_bias=get_data(self.router.linear_router.bias)
                        if self.router.bias
                        else None,
                        expert_gate_up_bias=get_data(
                            self.expert_mlps.mlp_op.gate_up_proj.bias.view(
                                self.num_local_experts, 2, -1
                            )
                        )
                        if routed_experts_mlp_config.bias
                        else None,
                        expert_down_bias=get_data(
                            self.expert_mlps.mlp_op.down_proj.bias
                        )
                        if routed_experts_mlp_config.bias
                        else None,
                        shared_expert_gate_bias=None,
                        shared_expert_up_bias=None,
                        shared_expert_down_bias=None,
                        router_pre_norm=not self.router.apply_act_fn_over_topk,
                        hidden_act_fn=ActFnType(kernel_activation_func_id),
                        hidden_act_scale_factor=None,
                        hidden_act_bias=None,
                        norm_topk_prob=self.config.norm_topk_prob,
                        is_all_expert=is_all_expert,
                        **optional_kwargs,
                    )

                return out.view(hidden_states_shape), router_logits.to(
                    hidden_states.dtype
                )

            return replacement_moe_fused_tkg_kernel

        # Bind the replacement method
        fused_tkg._moe_fused_tkg_kernel = types.MethodType(
            _make_replacement_method(),
            fused_tkg,
        )
        patched_count += 1
        logger.info(
            "Layer %d: Replaced _moe_fused_tkg_kernel with selection_bias version",
            layer_idx,
        )

    logger.info(
        "Patched %d MoE layers with selection_bias fused TKG kernel", patched_count
    )
    return patched_count


# Apply patches on import
_patch_blockwise_shard_hidden()
_patch_expert_fused_linear_forward()
