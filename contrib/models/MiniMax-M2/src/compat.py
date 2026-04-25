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


# Apply patches on import
_patch_blockwise_shard_hidden()
_patch_expert_fused_linear_forward()
