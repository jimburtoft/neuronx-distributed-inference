# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom nkilib overrides for MiniMax-M2.

This package contains modified copies of nkilib modules required by MiniMax-M2.
The modifications add capabilities not yet available in the upstream nkilib release.

Source fork: jimburtoft/nki-library, branch feature/minimax-m2-attention
Base commit: d2ad3a5 (NKI Lib 2026-04-13)

Changes by commit:
  25c9f36 - Torchxla compatibility fixes (kernel_helpers, attention_cte, modular_allocator)
  7cce020 - Partial RoPE support (rotary_dim < d_head)
  894d0d9 - Flat QK RMSNorm (pre-head-split normalization for MiniMax-M2)
  daad362 - KV cache B=1 fix (_update_flat_cache DMA addressing)

Modified files:
  core/attention/attention_cte.py    - tile_size constants + psum address fallback
  core/embeddings/rope.py            - rotary_dim parameter for partial RoPE
  core/utils/kernel_helpers.py       - num_programs() None check
  core/utils/modular_allocator.py    - NKI 0.3.0 compat (inline nl.ndarray, no address=)
  experimental/transformer/attention_block_tkg.py - Flat QK RMSNorm, KV cache B=1 fix,
                                                    rotary_dim threading, signature changes
  experimental/transformer/transformer_tkg.py     - Updated callers for new mandatory params

To activate these overrides, call patch_nkilib_modules() before any nkilib
kernel compilation occurs (typically in the model's __init__.py or compat.py).
"""

import sys
import importlib
import logging

logger = logging.getLogger(__name__)

# Mapping from system nkilib module path to our custom module path
_MODULE_OVERRIDES = {
    "nkilib.core.attention.attention_cte": "nkilib_custom.core.attention.attention_cte",
    "nkilib.core.embeddings.rope": "nkilib_custom.core.embeddings.rope",
    "nkilib.core.utils.kernel_helpers": "nkilib_custom.core.utils.kernel_helpers",
    "nkilib.core.utils.modular_allocator": "nkilib_custom.core.utils.modular_allocator",
    "nkilib.experimental.transformer.attention_block_tkg": "nkilib_custom.experimental.transformer.attention_block_tkg",
    "nkilib.experimental.transformer.transformer_tkg": "nkilib_custom.experimental.transformer.transformer_tkg",
}


def patch_nkilib_modules():
    """
    Replace system nkilib modules with custom MiniMax-M2 versions in sys.modules.

    This must be called BEFORE any code imports from the affected nkilib modules.
    After patching, any ``from nkilib.experimental.transformer.attention_block_tkg import ...``
    will resolve to our custom version.

    Returns:
        int: Number of modules successfully patched.
    """
    patched = 0
    for system_path, custom_path in _MODULE_OVERRIDES.items():
        try:
            custom_mod = importlib.import_module(custom_path)
            sys.modules[system_path] = custom_mod
            patched += 1
            logger.debug(f"Patched {system_path} -> {custom_path}")
        except Exception as e:
            logger.warning(f"Failed to patch {system_path}: {e}")
    if patched:
        logger.info(
            f"Patched {patched}/{len(_MODULE_OVERRIDES)} nkilib modules with MiniMax-M2 custom versions"
        )
    return patched
