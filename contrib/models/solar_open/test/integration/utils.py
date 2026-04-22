"""Integration test utilities for Solar Open MoE.

Uses transformers.SolarOpenForCausalLM (available since transformers 4.57+) as the
CPU reference model for logit accuracy checks against NeuronSolarOpenForCausalLM.

Public API:
- create_neuron_config(): return MoENeuronConfig for integration tests
- create_tiny_solar_open_model(): write a minimal HF checkpoint
- generate_golden_logits(): generate CPU reference logits before Neuron load
- prepare_inputs(): create random input tensors
"""

import gc
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import GenerationConfig

# Add contrib src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    OnDeviceSamplingConfig,
)


# ---------------------------------------------------------------------------
# Neuron config for integration tests
# ---------------------------------------------------------------------------


def create_neuron_config(
    tp_degree: int = 2,
    seq_len: int = 128,
    batch_size: int = 1,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> MoENeuronConfig:
    """Create MoENeuronConfig for Solar Open integration tests.

    Args:
        tp_degree: Tensor parallelism degree.
        seq_len: Maximum sequence length.
        batch_size: Batch size for inference.
        torch_dtype: Dtype for model weights.

    Returns:
        Configured MoENeuronConfig.
    """
    return MoENeuronConfig(
        tp_degree=tp_degree,
        moe_tp_degree=tp_degree,
        moe_ep_degree=1,
        batch_size=batch_size,
        ctx_batch_size=batch_size,
        tkg_batch_size=batch_size,
        seq_len=seq_len,
        max_context_length=seq_len - 8,
        torch_dtype=torch_dtype,
        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
        output_logits=True,  # Required for check_accuracy_logits_v2
        enable_bucketing=False,
        flash_decoding_enabled=False,
        fused_qkv=True,
        sequence_parallel_enabled=False,
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
    )


# ---------------------------------------------------------------------------
# Tiny random model factory (uses transformers SolarOpenForCausalLM)
# ---------------------------------------------------------------------------


def create_tiny_solar_open_model(model_dir: str) -> None:
    """Create a tiny random-weight Solar Open checkpoint using transformers.

    Calls SolarOpenForCausalLM(config).save_pretrained(model_dir) which writes:
      - config.json  (model_type="solar_open", all HF fields)
      - model.safetensors  (random weights)
      - generation_config.json

    Args:
        model_dir: Directory to write the checkpoint to.
    """
    from transformers import SolarOpenConfig, SolarOpenForCausalLM

    os.makedirs(model_dir, exist_ok=True)

    config = SolarOpenConfig(
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        moe_intermediate_size=256,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        vocab_size=1024,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
    )

    torch.manual_seed(42)
    model = SolarOpenForCausalLM(config)
    # Save in bfloat16 to match Neuron inference precision and avoid
    # fp32-vs-bf16 logit divergences in accuracy tests with random weights.
    model = model.to(torch.bfloat16)
    model.save_pretrained(model_dir)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def prepare_inputs(
    batch_size: int,
    seq_len: int,
    vocab_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare random input tensors for inference.

    Args:
        batch_size: Number of sequences in the batch.
        seq_len: Sequence length.
        vocab_size: Vocabulary size for token sampling.

    Returns:
        Tuple of (input_ids, attention_mask).
    """
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Golden logit generation (CPU-only, before Neuron model load)
# ---------------------------------------------------------------------------


def generate_golden_logits(
    model_path: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: GenerationConfig,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate reference logits from the HuggingFace CPU model.

    This function loads SolarOpenForCausalLM on CPU, generates logits using
    greedy decoding, and immediately frees the model from memory. It should
    be called BEFORE the Neuron model is compiled/loaded to avoid OOM on
    memory-constrained instances (e.g., trn2.3xlarge where the Neuron runtime
    maps 96 GB HBM into the process address space).

    Args:
        model_path: Path to the HF checkpoint directory.
        input_ids: Input token IDs [batch_size, seq_len].
        attention_mask: Attention mask [batch_size, seq_len].
        generation_config: HuggingFace GenerationConfig.
        max_new_tokens: Number of tokens to generate.

    Returns:
        Expected logits tensor [num_tokens, batch_size, vocab_size].
    """
    from transformers import SolarOpenForCausalLM

    hf_model = SolarOpenForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    hf_model.eval()

    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
        )

    expected_logits = torch.stack(outputs.scores)[:max_new_tokens, :, :]
    expected_token_ids = expected_logits.argmax(dim=2).T
    print(f"  Golden token IDs: {expected_token_ids}")
    print(f"  Golden logits shape: {expected_logits.shape}")

    # Free CPU model before Neuron runtime claims device memory
    del hf_model
    gc.collect()

    return expected_logits
