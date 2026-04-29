# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runtime registration of MiMo-V2-Pro into NxDI + vLLM-neuron.
#
# Contrib models are not built into the NxDI MODEL_TYPES registry or
# vLLM-neuron's architecture resolver.  This module patches both at
# import time so that ``vllm serve`` can load MiMo-V2-Pro without
# modifying any installed packages.
#
# Usage (from the bench script or any launcher):
#
#   export NXDI_CONTRIB_MIMO_V2_PRO_SRC=/path/to/this/src
#   python -c "import register_vllm" && python -m vllm.entrypoints.openai.api_server ...
#
# Or, more conveniently, as a combined launcher:
#
#   python register_vllm.py --model /path/to/model --tensor-parallel-size 64 ...
#
# All extra CLI arguments are forwarded to vLLM's ``api_server``.

import logging
import os
import sys

logger = logging.getLogger("mimo_v2_pro.register")

# ------------------------------------------------------------------
# 1. Ensure this contrib's src/ is importable
# ------------------------------------------------------------------
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ------------------------------------------------------------------
# 2. Import the Neuron model class
# ------------------------------------------------------------------
from modeling_mimo_v2 import NeuronMiMoV2ForCausalLM  # noqa: E402

# ------------------------------------------------------------------
# 3. Register in NxDI MODEL_TYPES
# ------------------------------------------------------------------
from neuronx_distributed_inference.utils.constants import MODEL_TYPES  # noqa: E402

# Register both mimo_v2 (Pro) and mimo_v2_flash (Flash) model types.
# The same NeuronMiMoV2ForCausalLM class handles both -- differences
# are in config values and state dict conversion (fused_qkv unfusing).
MODEL_TYPES["mimo_v2"] = {"causal-lm": NeuronMiMoV2ForCausalLM}
MODEL_TYPES["mimo_v2_flash"] = {"causal-lm": NeuronMiMoV2ForCausalLM}
logger.info(
    "Registered mimo_v2 and mimo_v2_flash in NxDI MODEL_TYPES (now %d models)",
    len(MODEL_TYPES),
)

# ------------------------------------------------------------------
# 4. Patch vLLM-neuron architecture resolver
# ------------------------------------------------------------------
try:
    import vllm_neuron.worker.neuronx_distributed_model_loader as _loader  # noqa: E402

    _original_get_cls = _loader._get_neuron_model_cls

    def _patched_get_neuron_model_cls(architecture: str):
        """Intercept MiMoV2* architectures and return our contrib class.

        Handles both:
        - MiMoV2ForCausalLM (Pro, model_type=mimo_v2)
        - MiMoV2FlashForCausalLM (Flash, model_type=mimo_v2_flash)
        """
        if "MiMoV2" in architecture or "mimov2" in architecture.lower():
            logger.info("Resolved %s -> NeuronMiMoV2ForCausalLM", architecture)
            return NeuronMiMoV2ForCausalLM
        return _original_get_cls(architecture)

    _loader._get_neuron_model_cls = _patched_get_neuron_model_cls
    logger.info("Patched vLLM-neuron _get_neuron_model_cls")
except ImportError:
    logger.warning("vllm_neuron not found; skipping vLLM-neuron patch")

# ------------------------------------------------------------------
# 5. Patch NxDI load_pretrained_config for trust_remote_code
# ------------------------------------------------------------------
# NxDI's load_pretrained_config calls AutoConfig.from_pretrained()
# without trust_remote_code=True.  MiMo-V2 checkpoints have an
# auto_map that requires custom code, so we patch the function to
# pass the flag.
try:
    import neuronx_distributed_inference.utils.hf_adapter as _hf_adapter  # noqa: E402
    from transformers import AutoConfig as _AutoConfig  # noqa: E402

    _orig_ac_from_pretrained = _AutoConfig.from_pretrained.__func__

    @classmethod
    def _trusted_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return _orig_ac_from_pretrained(cls, *args, **kwargs)

    _AutoConfig.from_pretrained = _trusted_from_pretrained
    logger.info("Patched AutoConfig.from_pretrained to default trust_remote_code=True")
except Exception as e:
    logger.warning("Could not patch AutoConfig.from_pretrained: %s", e)

# ------------------------------------------------------------------
# 5b. Patch AutoTokenizer.from_pretrained for trust_remote_code
# ------------------------------------------------------------------
# vLLM 0.16's tokenizer creation path also calls
# AutoTokenizer.from_pretrained without trust_remote_code=True.
# MiMo-V2's config maps to a custom config class that the
# tokenizer registry doesn't recognise, so we must force the flag.
try:
    from transformers import AutoTokenizer as _AutoTokenizer  # noqa: E402

    _orig_at_from_pretrained = _AutoTokenizer.from_pretrained.__func__

    @classmethod
    def _trusted_tokenizer_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return _orig_at_from_pretrained(cls, *args, **kwargs)

    _AutoTokenizer.from_pretrained = _trusted_tokenizer_from_pretrained
    logger.info(
        "Patched AutoTokenizer.from_pretrained to default trust_remote_code=True"
    )
except Exception as e:
    logger.warning("Could not patch AutoTokenizer.from_pretrained: %s", e)

# ------------------------------------------------------------------
# 6. vLLM ModelRegistry
# ------------------------------------------------------------------
# vLLM 0.16 may have native MiMoV2ForCausalLM or MiMoV2FlashForCausalLM
# entries in its ModelRegistry (mapped to GPU implementations).  We do
# NOT overwrite them -- the native entries allow inspect_model_cls() to
# succeed so that the architecture is resolved correctly.  vLLM-neuron's
# model loader bypasses the vLLM model class entirely and uses our NxDI
# class from step 3+4 instead.


# ------------------------------------------------------------------
# 7. Optional: launch vLLM when run as a script
# ------------------------------------------------------------------
def main():
    """Launch vLLM api_server with MiMo-V2-Pro registered."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Strip this script from argv so vLLM sees only its own flags
    sys.argv = sys.argv[1:] if len(sys.argv) > 1 else ["vllm"]

    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        make_arg_parser,
        run_server,
    )
    import uvloop

    parser = FlexibleArgumentParser(description="vLLM server with MiMo-V2-Pro support")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # vLLM 0.16 splits --model into model_tag (the user value) and model
    # (a pydantic default).  The ``vllm serve`` CLI copies model_tag ->
    # model, but we bypass that path by calling run_server() directly.
    # Replicate the same mapping here so model_config.model is correct.
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag
        logger.info("Set args.model from model_tag: %s", args.model)

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
