# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility patch for NxD 0.18 blockwise.py.

NxD 0.18 (neuronx-distributed 0.18.27753, shipped in the 20260410 DLAMI) removed
the `neuronxcc.nki._private.blockwise_mm` module, leaving `_call_shard_hidden_kernel`
as a stub that raises NotImplementedError. The kernel itself still exists in nkilib
at `nkilib.experimental.moe.forward.bwmm_shard_on_H.blockwise_mm_baseline_shard_hidden`.

This module patches blockwise.py at import time to restore the kernel call.
"""

import logging
import importlib

import torch

logger = logging.getLogger(__name__)


def _patch_blockwise_shard_hidden():
    """Patch NxD blockwise.py _call_shard_hidden_kernel if it's a stub."""
    try:
        import neuronx_distributed.modules.moe.blockwise as bw
    except ImportError:
        logger.debug(
            "neuronx_distributed.modules.moe.blockwise not available, skipping patch"
        )
        return False

    # Check if the function is a stub (raises NotImplementedError)
    try:
        # Create a minimal dummy to test -- if it raises NotImplementedError, it needs patching
        bw._call_shard_hidden_kernel(None)
    except NotImplementedError:
        pass  # Confirmed stub, proceed with patch
    except (TypeError, AttributeError):
        # Function exists and tried to run (got TypeError from None arg) -- already patched or real
        logger.debug("_call_shard_hidden_kernel appears functional, skipping patch")
        return False

    try:
        # Import the kernel from nkilib
        mod = importlib.import_module("nkilib.experimental.moe.forward.bwmm_shard_on_H")
        kernel_fn = getattr(mod, "blockwise_mm_baseline_shard_hidden")

        # Wrap with nki.jit for torchxla mode
        import nki

        wrapped_kernel = nki.jit(kernel_fn, mode="torchxla")
        bw._blockwise_mm_baseline_shard_hidden_nki_call = wrapped_kernel

        def _call_shard_hidden_kernel_patched(args):
            """Call the nkilib shard_hidden kernel for blockwise matmul.

            The nkilib kernel returns output of shape [T+1, H] directly (unlike
            the shard_on_block kernel which returns [T+1, 2, H]).
            """
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


# Apply patch on import
_patch_blockwise_shard_hidden()
