# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKI scan kernels for Nemotron-3-Nano Mamba-2 SSD.

Public kernels:
    - mamba2_ssd_chunked_fwd (nki_mamba2_ssd_chunked.py):
        The chunked variant with intra-chunk quadratic + inter-chunk recurrent
        state propagation. Preferred for production use. Bit-exact vs the
        head-grouped quadratic scan in-model (50/50 tokens at ctx=128,
        30/30 at ctx=1024). See README.md for the pattern documentation.

    - mamba2_ssd_recurrent_fwd, mamba2_ssd_recurrent_fwd_state
      (nki_mamba2_ssd_recurrent.py):
        The per-token recurrent kernel. Slower than chunked (O(1) decode is
        fine, prefill is ~23x slower per invocation) but simpler; served as
        the derivation foundation for the chunked kernel.

Python-side integration wrapper:
    - chunked_ssd_prefill_scan (chunked_ssd_wrapper.py):
        Adapts Nemotron Mamba-2 prefill inputs (batched, grouped B/C) to the
        chunked kernel's per-(batch, head) signature. Called from
        NeuronNemotronMamba2Layer._forward_prefill when USE_CHUNKED_NKI_SCAN=1.

The kernels use only "cheap ops" (dma_copy, nc_matmul, nc_transpose, tensor_copy,
tensor_scalar, tensor_tensor, activation, memset). This is the DGE-OOB-survivor
pattern discovered in Qwen3-Coder-Next. See README.md for the full pattern
description and validation history.
"""
