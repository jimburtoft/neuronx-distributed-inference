#!/usr/bin/env python3
"""Monkeypatch NxDI's NeuronLlamaMLP to use the NKI TKG kernel for token
generation and a manual-matmul fallback for context encoding.

The built-in mlp_isa_kernel has a hard 4096 intermediate-dim limit per core
(assertion in walrus/inline_bir_kernel/src/kernels_impl/mlp.cpp:196). This
blocks models like YuE S1 (I=11008) and S2 (I=5504) at TP=1.

The NKI TKG kernel (nki_mlp_tkg_isa_kernel) has NO dimension limit and delivers
~20% total compute reduction by fusing RMSNorm + Gate/Up + SiLU + Down in SBUF.

Strategy:
  - TKG path (B*S <= 128): NKI TKG kernel (no dim limit) -- 20% speedup
  - CTE path (B*S > 128), I <= 4096: original mlp_isa_kernel -- unchanged
  - CTE path (B*S > 128), I > 4096: manual matmul with transposed weights,
    bypassing the compiler limit. ~878ms per CTE call on S2 (vs ~1053ms for
    the split-kernel approach that was benchmarked and rejected).

Usage:
    import nki_mlp_patch  # must import BEFORE model.compile()
"""

import logging
import torch
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)

logger = logging.getLogger("Neuron")

# Maximum intermediate dimension for the built-in CTE kernel (compiler limit)
MAX_INTERMEDIATE_PER_CALL = 4096


def _patched_forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
    """Patched NeuronLlamaMLP.forward:
    - TKG path (batch_seqlen <= 128): use NKI TKG kernel (no dim limit)
    - CTE path, I <= 4096: original mlp_isa_kernel
    - CTE path, I > 4096: manual matmul fallback (bypasses compiler limit)
    """
    TKG_BS_SEQLEN_THRESHOLD = 128

    if not self.mlp_kernel_enabled:
        # No kernel -- use native path
        assert rmsnorm is None and residual is None
        return (self._native_mlp(x, adapter_ids=adapter_ids), None)

    # Determine batch*seqlen
    if self.tensor_model_parallel_group is not None:
        tp_degree = self.tensor_model_parallel_group.size()
    else:
        tp_degree = self.config.neuron_config.tp_degree

    if self.sequence_parallel_enabled:
        real_seqlen = x.shape[1] * tp_degree
    else:
        real_seqlen = x.shape[1]

    batch_seqlen = x.shape[0] * real_seqlen
    is_small_batch_seqlen = batch_seqlen <= TKG_BS_SEQLEN_THRESHOLD

    # Import the TKG kernel check
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        _trace_nki_mlp_tkg_kernel,
    )

    use_tkg_nki_kernel = (
        _trace_nki_mlp_tkg_kernel
        and is_small_batch_seqlen
        and self.mlp_tkg_nki_kernel_enabled
    )

    if use_tkg_nki_kernel:
        # Token generation: use the NKI TKG kernel (no dimension limit)
        return self._kernel_enabled_nki_mlp_tkg(
            x, rmsnorm, residual, adapter_ids=adapter_ids
        )

    # CTE path: check if intermediate dimension exceeds compiler limit
    # down_proj.weight is [I, H] (transposed from [H, I] at init)
    down_w = self.down_proj.weight.data
    intermediate_size = down_w.shape[0]

    if intermediate_size <= MAX_INTERMEDIATE_PER_CALL:
        # Original CTE kernel works fine -- no dimension issue
        return self._kernel_enabled_mlp(x, rmsnorm, residual, adapter_ids=adapter_ids)

    # --- Manual matmul CTE fallback for I > 4096 ---
    # This bypasses the closed-source mlp_isa_kernel entirely.
    # Benchmarked at ~878ms per CTE call on S2 (I=5504, bs=2).

    # Step 1: Handle residual add and RMSNorm externally
    residual_out = None
    if residual is not None:
        hidden_for_mlp = residual + x
        residual_out = hidden_for_mlp
        x_input = rmsnorm(hidden_for_mlp) if rmsnorm is not None else hidden_for_mlp
    else:
        x_input = rmsnorm(x) if rmsnorm is not None else x

    # Step 2: Handle sequence parallel gather
    if self.sequence_parallel_enabled:
        from neuronx_distributed_inference.modules.custom_calls import (
            gather_from_sequence_parallel_region,
        )

        x_input = gather_from_sequence_parallel_region(
            x_input,
            self.sequence_dimension,
            process_group=self.tensor_model_parallel_group,
        )

    # Step 3: Manual MLP computation with transposed weights
    # When mlp_kernel_enabled=True, weights are already transposed at init:
    #   gate_proj.weight: [H, I]  (transposed from [I, H])
    #   up_proj.weight:   [H, I]
    #   down_proj.weight: [I, H]  (transposed from [H, I])
    gate_w = self.gate_proj.weight.data  # [H, I]
    up_w = self.up_proj.weight.data  # [H, I]
    # down_w already set above              # [I, H]

    # SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    gate_out = torch.matmul(x_input, gate_w)  # (B, S, I)
    gate_out = torch.nn.functional.silu(gate_out)
    up_out = torch.matmul(x_input, up_w)  # (B, S, I)
    intermediate = gate_out * up_out  # (B, S, I)
    output_tensor = torch.matmul(intermediate, down_w)  # (B, S, H)

    # Step 4: All-reduce or reduce-scatter
    if self.sequence_parallel_enabled:
        from neuronx_distributed_inference.modules.custom_calls import (
            reduce_scatter_to_sequence_parallel_region,
        )

        if hasattr(self.neuron_config, "tile_cc") and self.neuron_config.tile_cc:
            from neuronx_distributed_inference.modules.custom_calls import (
                reduce_scatter_to_sequence_parallel_region_tiled,
            )

            output_tensor = reduce_scatter_to_sequence_parallel_region_tiled(
                output_tensor,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )
        else:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )
    else:
        output_tensor = reduce_from_tensor_model_parallel_region(
            output_tensor, process_group=self.tensor_model_parallel_group
        )

    return (output_tensor, residual_out)


def apply_patch():
    """Apply the monkeypatch to NeuronLlamaMLP."""
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP

    NeuronLlamaMLP.forward = _patched_forward
    logger.info(
        "NKI MLP patch applied: TKG kernel for token-gen, "
        "manual matmul for CTE with I > 4096"
    )


# Auto-apply on import
apply_patch()
