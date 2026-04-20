# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax M2 model for NXD inference - Contrib wrapper.

This module patches the system nkilib with custom MiniMax-M2 overrides before
importing the model class. The patching must occur before any nkilib modules
are loaded (which happens when attention_base.py is first imported).
"""

# Patch nkilib modules BEFORE any NxDI imports that trigger attention_base.py loading.
# This replaces 6 nkilib modules with custom versions that add:
#   - Partial RoPE support (rotary_dim < d_head)
#   - Flat QK RMSNorm (pre-head-split normalization)
#   - KV cache B=1 correctness fix
#   - Torchxla compatibility fixes
# Source: jimburtoft/nki-library branch feature/minimax-m2-attention
from nkilib_custom import patch_nkilib_modules

patch_nkilib_modules()

from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import (
    NeuronMiniMaxM2ForCausalLM,
    MiniMaxM2InferenceConfig,
    convert_minimax_m2_hf_to_neuron_state_dict,
)

__all__ = [
    "MiniMaxM2InferenceConfig",
    "NeuronMiniMaxM2ForCausalLM",
]
