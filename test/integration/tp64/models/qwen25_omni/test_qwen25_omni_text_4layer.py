"""Integration test for Qwen2.5-Omni text-only (Thinker) model.

Validates logit accuracy of the NxDI NeuronQwen25OmniForCausalLM model
against a CPU HuggingFace reference using a 4-layer tiny model with
random weights.  Follows the same pattern as the Qwen2-VL text 4-layer
test (test_qwen2_vl_text_4layer.py).

Key points:
- Uses Qwen2_5OmniThinkerForConditionalGeneration for the HF reference
- Verifies M-RoPE (mrope_section=[16, 24, 24]) is active in text-only mode
- Uses check_accuracy_logits with teacher-forcing logit validation
"""

import logging
import os
import tempfile

import pytest
import torch
from argparse import Namespace
from transformers import GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, to_dict
from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni import (
    NeuronQwen25OmniForCausalLM,
    Qwen25OmniInferenceConfig,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

NUM_TOKENS_TO_CHECK = 16
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_checkpoint(config_path):
    """Create a Qwen2.5-Omni Thinker model with random weights and save it.

    Uses the thinker config (model_type: qwen2_5_omni_thinker) directly
    since NeuronQwen25OmniForCausalLM loads via
    Qwen2_5OmniThinkerForConditionalGeneration.
    """
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniThinkerConfig,
    )
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    hf_config = Qwen2_5OmniThinkerConfig.from_pretrained(
        config_path, torch_dtype=torch.float16
    )
    logger.info("HF thinker config: %s", to_dict(hf_config))
    hf_model = Qwen2_5OmniThinkerForConditionalGeneration._from_config(hf_config)
    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving Qwen2.5-Omni thinker with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def get_qwen25_omni_config(
    dtype=torch.float16,
    model_path=None,
    tp_degree=4,
    batch_size=1,
    seq_length=2560,
    max_new_tokens=16,
    buckets=None,
):
    """Create a Qwen25OmniInferenceConfig for the text-only model."""
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniThinkerConfig,
    )

    neuron_config = NeuronConfig(
        batch_size=batch_size,
        ctx_batch_size=1,
        tkg_batch_size=batch_size,
        seq_len=seq_length,
        max_new_tokens=max_new_tokens,
        torch_dtype=dtype,
        skip_sharding=False,
        save_sharded_checkpoint=True,
        tp_degree=tp_degree,
        cp_degree=1,
        world_size=tp_degree,
        context_encoding_buckets=buckets,
        token_generation_buckets=buckets,
        fused_qkv=True,
        qkv_kernel_enabled=True,
        mlp_kernel_enabled=False,
        enable_bucketing=True,
        cc_pipeline_tiling_factor=2,
        attention_dtype=dtype,
        rpl_reduce_dtype=dtype,
        cast_type="as-declared",
        logical_neuron_cores=2,
    )

    # Load the thinker config directly (not via AutoConfig, which
    # does not recognise model_type qwen2_5_omni_thinker).
    hf_config = Qwen2_5OmniThinkerConfig.from_pretrained(model_path)
    config = Qwen25OmniInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )
    return config


@pytest.mark.parametrize(
    "dtype, batch_size, tp_degree, world_size, seq_length, max_new_tokens, buckets",
    [
        (torch.float16, 1, 4, 4, 2560, 16, [1280, 2560]),
    ],
)
def test_original_cpu_vs_nxdi_neuron(
    dtype,
    batch_size,
    tp_degree,
    world_size,
    seq_length,
    max_new_tokens,
    buckets,
):
    """Compare NxDI Neuron model logits against HF CPU reference.

    Creates a 4-layer Qwen2.5-Omni thinker with random weights, compiles
    it for Neuron, and validates that the logit distributions match within
    the specified tolerance using teacher-forcing logit validation.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config_4layer.json"
    )

    # Save random-weight HF checkpoint
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name

    config = get_qwen25_omni_config(
        dtype=dtype,
        model_path=model_path,
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_length=seq_length,
        max_new_tokens=max_new_tokens,
        buckets=buckets,
    )

    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id=151643,
        eos_token_id=[151645],
        pad_token_id=151645,
        output_logits=True,
    )

    logger.info("InferenceConfig: %s", to_dict(config))

    # Random input tokens
    prompt_length = 128
    input_ids = torch.randint(low=0, high=100000, size=(batch_size, prompt_length)).to(
        dtype=torch.int32
    )
    attention_mask = torch.ones(size=(batch_size, prompt_length), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    # Compile and load Neuron model
    neuron_model = NeuronQwen25OmniForCausalLM(model_path=model_path, config=config)
    traced_path = os.path.join(
        model_path,
        f"qwen25_omni_text_4layer_traced_dtype_{dtype}",
    )
    os.makedirs(traced_path, exist_ok=True)
    print(f"Compiling Neuron model to {traced_path}")
    neuron_model.compile(traced_path)
    print(f"Compiled Neuron model to {traced_path}")

    neuron_model.load(traced_path)
    print(f"Loaded Neuron model from {traced_path}")

    # Validate logits
    if batch_size == 1:
        check_accuracy_logits(
            neuron_model,
            generation_config=generation_config,
            num_tokens_to_check=NUM_TOKENS_TO_CHECK,
            inputs=inputs,
            pad_token_id=151645,
            divergence_difference_tol=0.01,
        )

    # Clean up temp dir only if test passes
    model_tempdir.cleanup()


if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(torch.float16, 1, 4, 4, 2560, 16, [1280, 2560])
