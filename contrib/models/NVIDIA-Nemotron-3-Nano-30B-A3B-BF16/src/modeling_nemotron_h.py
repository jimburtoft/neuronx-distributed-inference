# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVIDIA Nemotron-3-Nano-30B-A3B-BF16 (nemotron_h) model for NxD Inference.

NVIDIA Nemotron-3-Nano-30B-A3B is a hybrid Mamba2/Attention/MoE architecture:
- 52 layers: 23 Mamba-2, 23 MoE, 6 GQA Attention
- Pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
- Each layer has ONE block (Mamba OR MoE OR Attention), single-block-per-layer
- Single pre-norm per layer, no post-attention layernorm
- 128 routed experts + 1 shared expert, sigmoid routing, relu2 activation
- 32 Q heads / 2 KV heads (16:1 GQA), explicit head_dim=128
- tie_word_embeddings=False

Key implementation details:
- Mamba state persistence via nn.ParameterList + input_output_aliases
  (same mechanism as KV cache, following Granite4 pattern)
- Manual depthwise conv1d to avoid TEN404 NKI kernel issue on seq_len=1
- Full-sequence parallel scan for prefill (O(L) NKI or O(L^2) quadratic fallback), O(1) recurrence for decode
- Per-expert Python loop for MoE (avoids HBM allocation failure from batched BMM)
- NeuronAttentionBase for GQA with RoPE
"""

import gc
import logging
import os
import warnings
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    ModelWrapper,
    DecoderModelInstance,
)
from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    BaseParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.utils import cpu_mode

logger = logging.getLogger(__name__)


# ==============================================================================
# NKI Selective Scan Kernel for Mamba2 prefill
# ==============================================================================
# When enabled, replaces the O(L^2) quadratic parallel scan with O(L) hardware-
# accelerated scan using nisa.tensor_tensor_scan on Trainium2.
# Set USE_NKI_SCAN = False to fall back to the quadratic implementation.
# Ported from Granite4 contrib (working/Granite4/contrib/src/modeling_granite.py).

USE_NKI_SCAN = (
    False  # O(L^2) quadratic scan: NKI scan produces more instructions at long context
)

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa

    HAS_NKI = True
except ImportError:
    HAS_NKI = False
    if USE_NKI_SCAN:
        logger.warning("NKI not available, falling back to quadratic scan")
        USE_NKI_SCAN = False

if HAS_NKI and USE_NKI_SCAN:
    P_MAX = 128

    @nki.jit
    def nki_scan_kernel(
        dA_exp_t,  # (NH, SL) -- pre-transposed decay coefficients
        dBx_t,  # (NH * HD * SS, SL) -- flattened+transposed
        C_t,  # (NH * SS, SL) -- flattened+transposed
        Dx_t,  # (NH * HD, SL) -- pre-computed D*x, flattened+transposed
        x_t,  # (NH * HD, SL) -- flattened+transposed (unused, for shape)
        hd_range,  # (HD,) -- dummy tensor for head_dim
        ss_range,  # (SS,) -- dummy tensor for ssm_state_size
    ):
        """NKI O(L) selective scan using tensor_tensor_scan.

        Computes: state[h, t] = dA[h, t] * state[h, t-1] + dBx[h, t]
                  y[h, d, t] = sum_s C[h, s, t] * state[h, d, s, t] + D[h] * x[h, d, t]
        """
        NH = dA_exp_t.shape[0]
        SL = dA_exp_t.shape[1]
        HD = hd_range.shape[0]
        SS = ss_range.shape[0]

        y_out = nl.ndarray((NH * HD, SL), dtype=nl.float32, buffer=nl.shared_hbm)
        final_state_out = nl.ndarray(
            (NH * HD * SS, 1), dtype=nl.float32, buffer=nl.shared_hbm
        )

        dA_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=dA_sb, value=0.0)
        nisa.dma_copy(dst=dA_sb[0:NH, 0:SL], src=dA_exp_t[0:NH, 0:SL])

        for d in nl.affine_range(HD):
            y_acc_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=y_acc_sb, value=0.0)
            Dx_row = d * NH
            nisa.dma_copy(
                dst=y_acc_sb[0:NH, 0:SL],
                src=Dx_t[Dx_row : Dx_row + NH, 0:SL],
            )

            for s in nl.affine_range(SS):
                dBx_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=dBx_sb, value=0.0)
                dBx_row = (d * SS + s) * NH
                nisa.dma_copy(
                    dst=dBx_sb[0:NH, 0:SL],
                    src=dBx_t[dBx_row : dBx_row + NH, 0:SL],
                )

                init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=init_sb, value=0.0)

                state_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor_scan(
                    dst=state_sb[0:NH, 0:SL],
                    data0=dA_sb[0:NH, 0:SL],
                    data1=dBx_sb[0:NH, 0:SL],
                    initial=init_sb[0:NH, 0:1],
                    op0=nl.multiply,
                    op1=nl.add,
                )

                final_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=final_sb[0:NH, 0:1],
                    src=state_sb[0:NH, SL - 1 : SL],
                )
                fs_row = (d * SS + s) * NH
                nisa.dma_copy(
                    dst=final_state_out[fs_row : fs_row + NH, 0:1],
                    src=final_sb[0:NH, 0:1],
                )

                C_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=C_sb, value=0.0)
                C_row = s * NH
                nisa.dma_copy(
                    dst=C_sb[0:NH, 0:SL],
                    src=C_t[C_row : C_row + NH, 0:SL],
                )

                Cs_sb = nl.ndarray((P_MAX, SL), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=Cs_sb[0:NH, 0:SL],
                    data1=C_sb[0:NH, 0:SL],
                    data2=state_sb[0:NH, 0:SL],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=y_acc_sb[0:NH, 0:SL],
                    data1=y_acc_sb[0:NH, 0:SL],
                    data2=Cs_sb[0:NH, 0:SL],
                    op=nl.add,
                )

            y_row = d * NH
            nisa.dma_copy(
                dst=y_out[y_row : y_row + NH, 0:SL],
                src=y_acc_sb[0:NH, 0:SL],
            )

        return y_out, final_state_out


def _nki_selective_scan(
    hidden_states_ssm, dt_processed, A, B, C, D, num_heads, head_dim, ssm_state_size
):
    """NKI-accelerated selective scan for Mamba2 prefill.

    Replaces O(L^2) quadratic scan with O(L) hardware scan.
    Handles data layout transformation (PyTorch -> NKI partition-first).
    Ported from Granite4 contrib.

    Args:
        hidden_states_ssm: (batch, seq_len, num_heads, head_dim) float32
        dt_processed: (batch, seq_len, num_heads) float32
        A: (num_heads,) float32 -- negative
        B: (batch, seq_len, num_heads, ssm_state_size) float32
        C: (batch, seq_len, num_heads, ssm_state_size) float32
        D: (num_heads,) float32
        num_heads, head_dim, ssm_state_size: ints

    Returns:
        y: (batch, seq_len, num_heads, head_dim)
        final_state: (batch, num_heads, head_dim, ssm_state_size)
    """
    batch, seq_len = hidden_states_ssm.shape[:2]

    # Pre-compute on PyTorch side (traced as XLA ops)
    dA_exp = torch.exp(dt_processed * A.view(1, 1, -1))  # (B, L, H)
    dB = dt_processed.unsqueeze(-1) * B  # (B, L, H, S)
    dBx = dB.unsqueeze(3) * hidden_states_ssm.unsqueeze(-1)  # (B, L, H, D, S)

    # Transpose to NKI partition-first layout (squeeze batch=1)
    dA_exp_t = dA_exp[0].transpose(0, 1).contiguous()  # (H, L)

    dBx_0 = dBx[0]  # (L, H, D, S)
    dBx_t = (
        dBx_0.permute(0, 2, 3, 1)
        .reshape(seq_len, head_dim * ssm_state_size * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (D*S*H, L)

    C_t = (
        C[0]
        .permute(0, 2, 1)
        .reshape(seq_len, ssm_state_size * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (S*H, L)

    x_t = (
        hidden_states_ssm[0]
        .permute(0, 2, 1)
        .reshape(seq_len, head_dim * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (D*H, L)

    Dx_t = (
        (D.view(1, -1, 1) * hidden_states_ssm[0])
        .permute(0, 2, 1)
        .reshape(seq_len, head_dim * num_heads)
        .transpose(0, 1)
        .contiguous()
    )  # (D*H, L)

    hd_range = torch.zeros(head_dim, dtype=torch.float32, device=dA_exp_t.device)
    ss_range = torch.zeros(ssm_state_size, dtype=torch.float32, device=dA_exp_t.device)

    # Call NKI kernel
    y_flat, state_flat = nki_scan_kernel(
        dA_exp_t,
        dBx_t,
        C_t,
        Dx_t,
        x_t,
        hd_range,
        ss_range,
    )

    # Unpack: y_flat (D*H, L) -> (1, L, H, D)
    y = (
        y_flat.reshape(head_dim, num_heads, seq_len)
        .permute(2, 1, 0)
        .unsqueeze(0)
        .contiguous()
    )

    # state_flat (D*S*H, 1) -> (1, H, D, S)
    final_state = (
        state_flat.reshape(head_dim, ssm_state_size, num_heads)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    return y, final_state


# ==============================================================================
# Pattern decoder
# ==============================================================================

PATTERN_MAP = {"M": "mamba", "E": "moe", "*": "attention"}


def decode_block_pattern(pattern: str) -> list:
    """Convert hybrid_override_pattern string to list of block types."""
    return [PATTERN_MAP[c] for c in pattern if c in PATTERN_MAP]


# ==============================================================================
# Activation
# ==============================================================================


def relu2(x: torch.Tensor) -> torch.Tensor:
    """ReLU squared: relu(x)^2. Nemotron's MoE activation."""
    return torch.relu(x).square()


# ==============================================================================
# RMSNorm helpers
# ==============================================================================


def get_rmsnorm_cls():
    """Return appropriate RMSNorm implementation (CPU or Neuron)."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


class NemotronRMSNormGated(nn.Module):
    """Gated RMSNorm: RMSNorm(x) * SiLU(gate). Gate applied AFTER norm.

    Matches HF MambaRMSNormGated with norm_before_gate=False:
    the norm is computed on x alone, then the result is multiplied by SiLU(gate).

    CRITICAL: Uses group_size for per-group normalization. The CUDA rmsnorm_fn
    normalizes per-group (groups of group_size elements), NOT over the full dim.
    The HF PyTorch fallback has an issue where it ignores group_size and normalizes
    over the full dimension, producing garbage decode output.

    For Nemotron: intermediate_size=4096, n_groups=8, group_size=512.
    This means 8 independent RMSNorms, each over 512 elements.
    """

    def __init__(self, hidden_size, group_size=None, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size if group_size is not None else hidden_size

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Group-wise RMSNorm: reshape into groups, normalize each independently
        orig_shape = hidden_states.shape
        group_size = self.group_size
        num_groups = orig_shape[-1] // group_size

        hidden_states = hidden_states.view(*orig_shape[:-1], num_groups, group_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.view(*orig_shape)

        hidden_states = (self.weight * hidden_states).to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate)
        return hidden_states


# ==============================================================================
# Config
# ==============================================================================


class NemotronHInferenceConfig(InferenceConfig):
    """Configuration class for Nemotron-3-Nano-30B-A3B inference."""

    output_attentions = False
    output_hidden_states = False
    use_return_dict = True

    def get_required_attributes(self) -> List[str]:
        return [
            # Core model
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "tie_word_embeddings",
            "norm_eps",
            # Mamba-2
            "mamba_num_heads",
            "mamba_head_dim",
            "ssm_state_size",
            "n_groups",
            "conv_kernel",
            # MoE
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "moe_shared_expert_intermediate_size",
            "routed_scaling_factor",
            # GQA Attention
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "max_position_embeddings",
            # Layer pattern
            "hybrid_override_pattern",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig

    def add_derived_config(self):
        """Add derived attributes after HF config is loaded."""
        super().add_derived_config()

        # Decode block pattern
        if hasattr(self, "hybrid_override_pattern"):
            self.layer_types = decode_block_pattern(self.hybrid_override_pattern)
            assert len(self.layer_types) == self.num_hidden_layers, (
                f"Pattern gives {len(self.layer_types)} layers but config says "
                f"{self.num_hidden_layers}"
            )

        # Defaults for optional attributes
        if not hasattr(self, "expand"):
            self.expand = 2
        if not hasattr(self, "chunk_size"):
            self.chunk_size = 128
        if not hasattr(self, "use_conv_bias"):
            self.use_conv_bias = True
        if not hasattr(self, "mamba_proj_bias"):
            self.mamba_proj_bias = False
        if not hasattr(self, "n_shared_experts"):
            self.n_shared_experts = 1
        if not hasattr(self, "norm_topk_prob"):
            self.norm_topk_prob = True
        if not hasattr(self, "mlp_bias"):
            self.mlp_bias = False
        if not hasattr(self, "attention_bias"):
            self.attention_bias = False

        # Nemotron uses norm_eps, NxDI base class expects rms_norm_eps for attention
        if hasattr(self, "norm_eps") and not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = self.norm_eps


# ==============================================================================
# Mamba-2 Layer
# ==============================================================================


class NeuronNemotronMamba2Layer(nn.Module):
    """
    Mamba-2 layer for Nemotron, with NxDI TP support.

    TP sharding strategy (Falcon-H1 pattern): use ColumnParallelLinear with
    gather_output=True for in_proj and out_proj. All SSM parameters (dt_bias,
    A_log, D), conv weights, and norm weights are REPLICATED across TP ranks
    (full size, no sharding). Each rank computes the full SSM on all heads,
    then the out_proj gather_output=True handles the all-reduce implicitly.

    NKI O(L) selective scan for prefill (USE_NKI_SCAN=True) or
    pure PyTorch O(L^2) quadratic fallback. O(1) decode.
    Manual depthwise conv1d (TEN404 workaround).
    External state buffers for input_output_aliases persistence.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.mamba_head_dim
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.chunk_size = getattr(config, "chunk_size", 128)

        # Full dimensions — NOT divided by TP (Falcon-H1 replicated pattern)
        self.num_heads = config.mamba_num_heads  # 64
        self.n_groups = config.n_groups  # 8
        self.intermediate_size = self.num_heads * self.head_dim  # 64*64=4096
        self.groups_time_state_size = self.n_groups * self.ssm_state_size  # 8*128=1024
        self.conv_dim = self.intermediate_size + 2 * self.groups_time_state_size  # 6144

        # Projection size for in_proj: gate + xBC + dt
        projection_size = (
            self.intermediate_size + self.conv_dim + self.num_heads
        )  # 10304

        # Single ColumnParallelLinear with gather_output=True (Falcon-H1 pattern)
        # This avoids the scatter bug and produces full-size output on every rank.
        if parallel_state.model_parallel_is_initialized():
            self.in_proj = ColumnParallelLinear(
                self.hidden_size,
                projection_size,
                bias=config.mamba_proj_bias,
                gather_output=True,
            )
            self.out_proj = ColumnParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                gather_output=True,
            )
        else:
            self.in_proj = nn.Linear(
                self.hidden_size, projection_size, bias=config.mamba_proj_bias
            )
            self.out_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=False
            )

        # Manual depthwise conv1d (TEN404 workaround) — FULL conv_dim (replicated)
        self.conv_weight = nn.Parameter(
            torch.randn(self.conv_dim, self.conv_kernel_size)
        )
        if getattr(config, "use_conv_bias", True):
            self.conv_bias = nn.Parameter(torch.zeros(self.conv_dim))
        else:
            self.conv_bias = None

        # SSM parameters — FULL num_heads (replicated, not TP-sharded)
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.num_heads))

        # Gated RMSNorm with per-group normalization — FULL sizes (replicated)
        # group_size = intermediate_size / n_groups = 4096 / 8 = 512
        self.norm = NemotronRMSNormGated(
            self.intermediate_size,
            group_size=self.intermediate_size // self.n_groups
            if self.n_groups > 0
            else self.intermediate_size,
            eps=getattr(config, "norm_eps", 1e-5),
        )

        self.time_step_limit = (0.0, float("inf"))

    @staticmethod
    def get_state_shapes(config, batch_size=1):
        """Return (conv_state_shape, ssm_state_shape) for buffer allocation.

        Returns FULL sizes (replicated across TP ranks, Falcon-H1 pattern).
        Every rank holds the full Mamba state since all params are replicated.
        """
        num_heads = config.mamba_num_heads  # Full: 64
        intermediate_size = num_heads * config.mamba_head_dim  # 4096
        groups_time_state_size = config.n_groups * config.ssm_state_size  # 1024
        conv_dim = intermediate_size + 2 * groups_time_state_size  # 6144
        conv_shape = (batch_size, conv_dim, config.conv_kernel - 1)
        ssm_shape = (
            batch_size,
            num_heads,
            config.mamba_head_dim,
            config.ssm_state_size,
        )
        return conv_shape, ssm_shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mamba_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Forward pass with external state.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mamba_state: (conv_state, ssm_state) from persistence buffers

        Returns:
            output: (batch, seq_len, hidden_size)
            present_key_value: dummy (K, V) tuple for KV cache compatibility
            updated_mamba_state: (conv_state, ssm_state) for persistence
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        if mamba_state is not None:
            conv_state, ssm_state = mamba_state
        else:
            conv_state = torch.zeros(
                batch_size,
                self.conv_dim,
                self.conv_kernel_size - 1,
                device=hidden_states.device,
                dtype=dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )

        # Use the padding mask passed from NeuronNemotronModel.forward().
        # This is derived from the raw attention_mask (1=real, 0=pad),
        # which is always correct regardless of NxDI's position_id padding scheme.
        padding_mask = kwargs.get("padding_mask", None)

        # Mask hidden_states BEFORE projection (Falcon-H1 pattern)
        # This ensures padding tokens produce zero projections.
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :seq_len, None].to(
                hidden_states.dtype
            )

        # Single in_proj with gather_output=True (Falcon-H1 pattern)
        projected_states = self.in_proj(hidden_states)
        # Explicit slicing (not split) for XLA
        gate = projected_states[..., : self.intermediate_size]
        hidden_states_B_C = projected_states[
            ..., self.intermediate_size : self.intermediate_size + self.conv_dim
        ]
        dt = projected_states[..., -self.num_heads :]

        if seq_len > 1:
            output, conv_state_new, ssm_state_new = self._forward_prefill(
                hidden_states_B_C, gate, dt, batch_size, seq_len, dtype, padding_mask
            )
            # Dummy dependency for XLA aliasing preservation
            conv_state_new = conv_state_new + conv_state * 0
            ssm_state_new = ssm_state_new + ssm_state * 0
        else:
            output, conv_state_new, ssm_state_new = self._forward_decode(
                hidden_states_B_C, gate, dt, batch_size, dtype, conv_state, ssm_state
            )

        # Dummy KV cache for compatibility with attention-based generation loop
        # Must include batch dimension for BS>1 support
        dummy_k = torch.zeros(
            batch_size, 1, 1, 1, dtype=output.dtype, device=output.device
        )
        dummy_v = torch.zeros(
            batch_size, 1, 1, 1, dtype=output.dtype, device=output.device
        )

        return (output, (dummy_k, dummy_v), (conv_state_new, ssm_state_new))

    def _forward_prefill(
        self, hidden_states_B_C, gate, dt, batch_size, seq_len, dtype, padding_mask=None
    ):
        """Selective scan for prefill.

        Uses O(L) NKI hardware-accelerated scan when USE_NKI_SCAN=True,
        or O(L^2) quadratic parallel scan as fallback.

        Padding handling follows Granite4 pattern:
        1. Save conv_state from last K-1 REAL token positions (not last seq positions)
        2. Zero conv output at padding positions
        3. Zero dt at padding positions (prevents SSM from processing padding)
        4. Gather SSM state from last real token (not last seq position)
        """
        # Manual depthwise conv1d
        padded = F.pad(
            hidden_states_B_C, (0, 0, self.conv_kernel_size - 1, 0), value=0.0
        )
        hidden_states_conv = torch.zeros_like(hidden_states_B_C)
        for k in range(self.conv_kernel_size):
            hidden_states_conv = hidden_states_conv + (
                padded[:, k : k + seq_len, :]
                * self.conv_weight[:, k].unsqueeze(0).unsqueeze(0)
            )
        if self.conv_bias is not None:
            hidden_states_conv = hidden_states_conv + self.conv_bias.unsqueeze(
                0
            ).unsqueeze(0)

        # Save conv_state from last K-1 REAL token positions (Granite4 pattern)
        if padding_mask is not None and seq_len >= self.conv_kernel_size - 1:
            real_len = padding_mask[:, :seq_len].sum(dim=1, keepdim=True).long()
            K_minus_1 = self.conv_kernel_size - 1
            offsets = torch.arange(
                K_minus_1, device=hidden_states_B_C.device
            ).unsqueeze(0)
            gather_idx = (real_len - K_minus_1 + offsets).clamp(min=0)
            gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, self.conv_dim)
            conv_state_seq = torch.gather(hidden_states_B_C, 1, gather_idx_expanded)
            conv_state_new = conv_state_seq.transpose(1, 2).contiguous()
        elif seq_len >= self.conv_kernel_size - 1:
            conv_state_new = (
                hidden_states_B_C[:, -(self.conv_kernel_size - 1) :, :]
                .transpose(1, 2)
                .contiguous()
            )
        else:
            pad_len = self.conv_kernel_size - 1 - seq_len
            conv_state_new = F.pad(
                hidden_states_B_C.transpose(1, 2), (pad_len, 0), value=0.0
            ).contiguous()

        hidden_states_conv = F.silu(hidden_states_conv)

        # Zero out conv output at padding positions (Granite4 pattern)
        if padding_mask is not None:
            hidden_states_conv = hidden_states_conv * padding_mask[
                :, :seq_len, None
            ].to(hidden_states_conv.dtype)

        # Split into x, B, C
        x = hidden_states_conv[..., : self.intermediate_size]
        B = hidden_states_conv[
            ...,
            self.intermediate_size : self.intermediate_size
            + self.groups_time_state_size,
        ]
        C = hidden_states_conv[..., -self.groups_time_state_size :]

        # SSM computation in float32
        A = -torch.exp(self.A_log.float())
        dt_processed = F.softplus(dt + self.dt_bias)
        dt_processed = torch.clamp(dt_processed, self.time_step_limit[0], 1e6)

        # Zero dt at padding positions (prevents SSM from processing padding)
        if padding_mask is not None:
            dt_processed = dt_processed * padding_mask[:, :seq_len, None].to(
                dt_processed.dtype
            )

        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, self.n_groups, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)

        if USE_NKI_SCAN:
            # NKI O(L) hardware-accelerated selective scan
            y, ssm_state_new = _nki_selective_scan(
                x,
                dt_processed,
                A,
                B,
                C,
                self.D.float(),
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
            )
            # NKI path returns final state from last seq position.
            # padding_mask handling for variable-length is not yet supported
            # with NKI -- assumes padded inputs (NxDI pads to max_context_length).
        else:
            # O(L^2) quadratic parallel scan (fallback)
            dA_log = dt_processed * A.view(1, 1, -1)
            dB = dt_processed.unsqueeze(-1) * B
            dBx = dB.unsqueeze(3) * x.unsqueeze(-1)

            log_dA_cumsum = torch.cumsum(dA_log, dim=1)

            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype)
            )
            log_diff = log_dA_cumsum.unsqueeze(2) - log_dA_cumsum.unsqueeze(1)
            log_diff = log_diff.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(-1) == 0, -1e9
            )
            weights = torch.exp(log_diff)

            states = torch.einsum("btih,bihds->bthds", weights, dBx)

            # Save final SSM state from last REAL token position (Granite4 pattern)
            if padding_mask is not None:
                real_len = padding_mask[:, :seq_len].sum(dim=1, keepdim=True).long()
                last_real_idx = (real_len - 1).clamp(min=0)
                gather_idx = last_real_idx.view(batch_size, 1, 1, 1, 1).expand(
                    -1, -1, self.num_heads, self.head_dim, self.ssm_state_size
                )
                ssm_state_new = (
                    torch.gather(states, 1, gather_idx).squeeze(1).contiguous()
                )
            else:
                ssm_state_new = states[:, -1, :, :, :].contiguous()

            y = torch.einsum("blhs,blhds->blhd", C, states)
            y = y + self.D.view(1, 1, -1, 1) * x
        y = y.reshape(batch_size, seq_len, -1)

        scan_output = self.norm(y, gate)
        # out_proj: ColumnParallelLinear(gather_output=True) handles TP all-reduce
        output = self.out_proj(scan_output.to(dtype))

        return output, conv_state_new, ssm_state_new.to(dtype)

    def _forward_decode(
        self, hidden_states_B_C, gate, dt, batch_size, dtype, conv_state, ssm_state
    ):
        """O(1) recurrence for single-token decode."""
        xBC_new = hidden_states_B_C.squeeze(1)  # (B, conv_dim)
        xBC_new_t = xBC_new.unsqueeze(2)  # (B, conv_dim, 1)
        conv_input = torch.cat(
            [conv_state, xBC_new_t], dim=2
        )  # (B, conv_dim, conv_kernel)

        conv_out = (conv_input * self.conv_weight.unsqueeze(0)).sum(dim=2)
        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias

        conv_state_new = conv_input[:, :, 1:].contiguous()
        conv_out = F.silu(conv_out)

        x = conv_out[..., : self.intermediate_size]
        B = conv_out[
            ...,
            self.intermediate_size : self.intermediate_size
            + self.groups_time_state_size,
        ]
        C = conv_out[..., -self.groups_time_state_size :]

        A = -torch.exp(self.A_log.float())
        dt_processed = F.softplus(dt.squeeze(1) + self.dt_bias)
        dt_processed = torch.clamp(dt_processed, self.time_step_limit[0], 1e6)

        x = x.reshape(batch_size, self.num_heads, self.head_dim).float()
        B = B.reshape(batch_size, self.n_groups, self.ssm_state_size).float()
        C = C.reshape(batch_size, self.n_groups, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=1)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=1)

        dA = torch.exp(dt_processed * A.view(1, -1))
        dB = dt_processed.unsqueeze(-1) * B
        dBx = dB.unsqueeze(2) * x.unsqueeze(-1)
        ssm_state_new = dA.unsqueeze(-1).unsqueeze(-1) * ssm_state.float() + dBx

        y = torch.einsum("bhds,bhs->bhd", ssm_state_new, C)
        y = y + self.D.view(1, -1, 1) * x
        y = y.reshape(batch_size, -1)

        gate_squeezed = gate.squeeze(1)
        scan_output = self.norm(y, gate_squeezed)
        if len(scan_output.shape) == 2:
            scan_output = scan_output.unsqueeze(1)
        # out_proj: ColumnParallelLinear(gather_output=True) handles TP all-reduce
        output = self.out_proj(scan_output.to(dtype))

        return output, conv_state_new, ssm_state_new.to(dtype)


# ==============================================================================
# MoE Layer
# ==============================================================================


class NeuronNemotronRouter(nn.Module):
    """
    Sigmoid router with e_score_correction_bias (DeepSeek-V3 style).
    n_group=1, topk_group=1 -> group routing disabled, pure sigmoid + top-k.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        # Router weight in FP32
        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, config.hidden_size, dtype=torch.float32)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_routed_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch*seq, hidden_size)
        Returns:
            topk_indices: (batch*seq, top_k)
            topk_weights: (batch*seq, top_k)
        """
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = torch.sigmoid(router_logits)

        # Bias-corrected selection
        scores_for_choice = scores + self.e_score_correction_bias.float().unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)[1]

        # Gather actual scores
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class NeuronNemotronMoELayer(BaseParallelLinear):
    """
    MoE layer: 128 routed experts + 1 shared expert.
    Non-gated MLP: down(relu2(up(x))).

    Uses per-expert loop with masking for XLA compatibility.
    Expert weights stored as stacked tensors with TP sharding on intermediate dim.

    Inherits from BaseParallelLinear so NxDI's shard_checkpoint recognizes
    the expert parameters for TP sharding.

    TP sharding: expert_up (E, H, I) sharded on dim=2 -> (E, H, I/TP)
                 expert_down (E, I, H) sharded on dim=1 -> (E, I/TP, H)
                 shared_up via ColumnParallelLinear (gather_output=False)
                 shared_down as nn.Linear on reduced intermediate
                 all-reduce after combining expert + shared outputs
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.intermediate_size = config.moe_intermediate_size
        self.shared_intermediate_size = config.moe_shared_expert_intermediate_size

        if parallel_state.model_parallel_is_initialized():
            tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            tp_degree = 1
        self.tp_degree = tp_degree

        assert self.intermediate_size % tp_degree == 0, (
            f"intermediate_size {self.intermediate_size} must be divisible by tp_degree {tp_degree}"
        )
        assert self.shared_intermediate_size % tp_degree == 0, (
            f"shared_intermediate_size {self.shared_intermediate_size} must be "
            f"divisible by tp_degree {tp_degree}"
        )

        self.intermediate_size_tp = self.intermediate_size // tp_degree
        self.shared_intermediate_size_tp = self.shared_intermediate_size // tp_degree

        # Router
        self.gate = NeuronNemotronRouter(config)

        # Stacked expert weights at PER-TP-RANK size.
        # Full: expert_up (E, H, I), expert_down (E, I, H)
        # Per-rank: expert_up (E, H, I/TP), expert_down (E, I/TP, H)
        # NxDI's shard_checkpoint will slice full checkpoint weights to per-rank size.
        self.expert_up = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size_tp)
        )
        self.expert_down = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size_tp, self.hidden_size)
        )

        # Mark for TP sharding (use direct setattr to avoid double-init assertion)
        for param, dim in [(self.expert_up, 2), (self.expert_down, 1)]:
            param.tensor_model_parallel = True
            param.partition_dim = dim
            param.partition_stride = 1
            param.num_partitions = tp_degree

        # Shared expert with TP sharding
        # Use ColumnParallelLinear for up (auto-sharded by framework)
        # For down: use raw Parameter marked for TP sharding (same approach as routed experts)
        if parallel_state.model_parallel_is_initialized():
            # Column-parallel up: (H, I_shared) -> (H, I_shared/TP) per core
            self.shared_up = ColumnParallelLinear(
                self.hidden_size,
                self.shared_intermediate_size,
                bias=getattr(config, "mlp_bias", False),
                gather_output=False,  # keep sharded for row-parallel down
            )
        else:
            self.shared_up = nn.Linear(
                self.hidden_size,
                self.shared_intermediate_size,
                bias=getattr(config, "mlp_bias", False),
            )

        # Shared down: Parameter at per-TP-rank size (I_shared/TP, H)
        # Stored transposed for x @ w pattern. Full weight is (I_shared, H),
        # each TP rank holds (I_shared/TP, H). Sharding slices dim=0.
        self.shared_down_weight = nn.Parameter(
            torch.empty(self.shared_intermediate_size_tp, self.hidden_size)
        )
        self.shared_down_weight.tensor_model_parallel = True
        self.shared_down_weight.partition_dim = 0
        self.shared_down_weight.partition_stride = 1
        self.shared_down_weight.num_partitions = tp_degree

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Forward pass matching decoder layer interface.

        Returns:
            output: (batch, seq_len, hidden_size)
            present_key_value: dummy (K, V) tuple
            updated_mamba_state: None (MoE has no recurrent state)
        """
        orig_shape = hidden_states.shape
        dtype = hidden_states.dtype
        residuals = hidden_states

        # Flatten
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        batch_tokens = hidden_flat.shape[0]

        # Route
        topk_indices, topk_weights = self.gate(hidden_flat)

        output = torch.zeros(
            batch_tokens,
            self.hidden_size,
            device=hidden_flat.device,
            dtype=dtype,
        )

        # Dual dispatch strategy based on sequence length:
        #
        # DECODE (batch_tokens <= batch_size, i.e. small T):
        #   Sparse dispatch — only load the 6 selected experts' weights.
        #   Uses index_select + bmm for each of the top-K expert slots.
        #   Cuts MoE weight traffic by ~21x (6 vs 128 expert loads).
        #
        # PREFILL (batch_tokens = seq_len, i.e. large T):
        #   Dense dispatch — loop over all 128 experts with affinity masking.
        #   The index_select + bmm approach explodes the HLO graph at T=128
        #   because each of 128 tokens may select different experts per slot,
        #   creating massive gather operations. The dense loop with simple
        #   matmul + mask is cheaper to compile and equally fast (prefill is
        #   compute-bound, not bandwidth-bound).
        #
        # NxDI traces CE and TG as separate graphs. batch_tokens is a static
        # shape at trace time, so this if/else is resolved during tracing
        # and only the relevant branch is compiled into each NEFF.

        if batch_tokens <= 2:
            # SPARSE DECODE PATH: only 6 experts loaded
            for k in range(self.gate.top_k):
                expert_idx = topk_indices[:, k]  # (T,)
                weight = topk_weights[:, k].unsqueeze(-1)  # (T, 1)

                # Gather selected expert weights
                w_up = torch.index_select(self.expert_up, 0, expert_idx)  # (T, H, I/TP)
                w_down = torch.index_select(
                    self.expert_down, 0, expert_idx
                )  # (T, I/TP, H)

                # Batched matmul
                intermediate = torch.bmm(hidden_flat.unsqueeze(1), w_up).squeeze(
                    1
                )  # (T, I/TP)
                intermediate = relu2(intermediate)

                expert_out = torch.bmm(intermediate.unsqueeze(1), w_down).squeeze(
                    1
                )  # (T, H) partial sum

                output = output + expert_out * weight

        else:
            # DENSE PREFILL PATH: loop over all 128 experts with masking
            expert_affinities = torch.zeros(
                batch_tokens,
                self.num_experts,
                device=hidden_flat.device,
                dtype=topk_weights.dtype,
            )
            expert_affinities.scatter_(1, topk_indices, topk_weights)

            for e in range(self.num_experts):
                w_up = self.expert_up[e]  # (H, I/TP)
                w_down = self.expert_down[e]  # (I/TP, H)
                intermediate = relu2(hidden_flat @ w_up)  # (T, I/TP)
                expert_out = intermediate @ w_down  # (T, H) partial sum
                affinity = expert_affinities[:, e].unsqueeze(-1)  # (T, 1)
                output = output + expert_out * affinity

        # Shared expert (also produces partial sums due to TP sharding)
        shared_intermediate = relu2(
            self.shared_up(residuals.reshape(-1, self.hidden_size))
        )
        shared_output = (
            shared_intermediate @ self.shared_down_weight
        )  # (T, H) partial sum

        # Combine routed + shared (both are partial sums)
        result = output + shared_output

        # All-reduce across TP to get full result
        if self.tp_degree > 1:
            result = reduce_from_tensor_model_parallel_region(result)

        result = result.reshape(orig_shape).to(dtype)

        # Dummy KV cache for compatibility
        # Must include batch dimension for BS>1 support
        batch_size = orig_shape[0]
        dummy_k = torch.zeros(
            batch_size, 1, 1, 1, dtype=result.dtype, device=result.device
        )
        dummy_v = torch.zeros(
            batch_size, 1, 1, 1, dtype=result.dtype, device=result.device
        )

        return (result, (dummy_k, dummy_v), None)


# ==============================================================================
# GQA Attention Layer
# ==============================================================================


class NeuronNemotronAttention(NeuronAttentionBase):
    """
    GQA attention for Nemotron using NxDI's NeuronAttentionBase.

    32 Q heads, 2 KV heads (16:1 ratio), explicit head_dim=128.
    RoPE with theta=10000.
    """

    def __init__(self, config: NemotronHInferenceConfig, layer_idx: int):
        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            base=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,  # MUST be explicit 128, not hidden_size//num_heads=84
            rotary_emb=rotary_emb,
            rms_norm_eps=getattr(
                config, "rms_norm_eps", getattr(config, "norm_eps", 1e-5)
            ),
            use_qk_norm=False,
            qkv_bias=getattr(config, "attention_bias", False),
        )

        self.layer_idx = layer_idx

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronNemotronAttention must be initialized in a distributed env."
            )


# ==============================================================================
# Decoder Layer (unified wrapper)
# ==============================================================================


class NeuronNemotronDecoderLayer(nn.Module):
    """
    Single decoder layer. Contains ONE of: Mamba-2, MoE, or GQA Attention.
    Each layer has a pre-norm (RMSNorm) and residual connection.

    Returns the standard 6-tuple:
        (hidden_states, present_key_value, cos_cache, sin_cache, None, updated_mamba_state)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=getattr(config, "norm_eps", 1e-5),
        )

        if self.layer_type == "mamba":
            self.mixer = NeuronNemotronMamba2Layer(config, layer_idx)
        elif self.layer_type == "moe":
            self.mixer = NeuronNemotronMoELayer(config, layer_idx)
        elif self.layer_type == "attention":
            self.self_attn = NeuronNemotronAttention(config, layer_idx)
        else:
            raise ValueError(f"Unknown block type: {self.layer_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mamba_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Returns 6-tuple matching Granite4 decoder layer interface:
            (hidden_states, present_key_value, cos_cache, sin_cache, None, updated_mamba_state)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        updated_mamba_state = None
        cos_cache, sin_cache = None, None

        if self.layer_type == "attention":
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )
        elif self.layer_type == "mamba":
            hidden_states, present_key_value, updated_mamba_state = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                mamba_state=mamba_state,
                **kwargs,
            )
        elif self.layer_type == "moe":
            hidden_states, present_key_value, _ = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        return (
            hidden_states,
            present_key_value,
            cos_cache,
            sin_cache,
            None,
            updated_mamba_state,
        )


# ==============================================================================
# Model Wrapper and DecoderModelInstance
# ==============================================================================


class NemotronDecoderModelInstance(DecoderModelInstance):
    """
    Extends DecoderModelInstance to alias Mamba state parameters.

    After calling super().get() which aliases the KV cache parameters,
    we add Mamba state parameters (conv_state, ssm_state for each Mamba layer)
    to the input_output_aliases dict.

    Output indices must match NeuronNemotronModel.forward() output order:
        [res, K0, V0, K1, V1, ..., conv_state_0, ssm_state_0, ...]
    """

    def get(self, bucket_rank, **kwargs):
        self.module, self.input_output_aliases = super().get(bucket_rank, **kwargs)

        past_key_values = self.module.kv_mgr.past_key_values
        mamba_states = self.module.mamba_states

        # Count where Mamba state outputs start
        num_output_from_trace = 1  # logits/tokens
        if getattr(self.module, "neuron_config", None) and getattr(
            self.module.neuron_config, "output_logits", False
        ):
            num_output_from_trace = 2
        num_output_from_trace += len(past_key_values)

        for i in range(len(mamba_states)):
            self.input_output_aliases[mamba_states[i]] = num_output_from_trace + i

        logger.info(
            f"NemotronDecoderModelInstance: aliased {len(past_key_values)} KV cache "
            f"entries and {len(mamba_states)} Mamba state entries "
            f"(Mamba starts at output index {num_output_from_trace})"
        )

        return self.module, self.input_output_aliases


class NemotronModelWrapper(ModelWrapper):
    """Custom ModelWrapper that returns NemotronDecoderModelInstance.

    Also post-processes compiler args to raise the HLO verifier instruction
    limit and disable strict verification.  The Mamba-2 quadratic scan
    generates ~6.86M HLO instructions at max_context_length>=4224, exceeding
    the default 5M verifier check (NCC_EVRF007).  Setting verify-hlo=false
    bypasses this verifier-only check; the backend's own limit is already
    raised to 15M via --internal-max-instruction-limit=15000000.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Raise the instruction count limit everywhere to handle the Mamba-2
        # quadratic scan's ~6.86M instructions at max_context_length>=4224.
        # Three checks must be bypassed:
        #   1. HLO verifier (NCC_EVRF007) -- via --tiled-inst-limit in hlo2tensorizer
        #   2. Backend limit (NCC_EBVF030) -- via --internal-max-instruction-limit
        #   3. Backend verifier -- via --internal-backend-options --max-instruction-limit
        if hasattr(self, "compiler_args") and self.compiler_args:
            self.compiler_args = self.compiler_args.replace(
                "--verify-hlo=true", "--verify-hlo=false --tiled-inst-limit=15000000"
            )

    def get_model_instance(self):
        return NemotronDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


# ==============================================================================
# Model Body
# ==============================================================================


class NeuronNemotronModel(NeuronBaseModel):
    """
    NeuronNemotronModel -- traced model body for Nemotron-3-Nano-30B.

    Overrides forward() and get_model_output() to handle:
    - Heterogeneous layers (Mamba/MoE/Attention in single-block-per-layer)
    - Mamba state persistence alongside KV cache
    - Per-layer routing of inputs (only attention layers need KV cache)
    """

    def setup_attr_for_model(self, config: NemotronHInferenceConfig):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: NemotronHInferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        self.layers = nn.ModuleList(
            [
                NeuronNemotronDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = get_rmsnorm_cls()(
            self.hidden_size,
            eps=getattr(config, "norm_eps", 1e-5),
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )

        # Mamba state persistence buffers
        self._mamba_layer_indices = [
            i
            for i in range(config.num_hidden_layers)
            if config.layer_types[i] == "mamba"
        ]
        batch_size = config.neuron_config.batch_size
        conv_shape, ssm_shape = NeuronNemotronMamba2Layer.get_state_shapes(
            config, batch_size
        )
        dtype = config.neuron_config.torch_dtype

        self.mamba_states = nn.ParameterList()
        for _ in self._mamba_layer_indices:
            self.mamba_states.append(
                nn.Parameter(torch.zeros(conv_shape, dtype=dtype), requires_grad=False)
            )
            self.mamba_states.append(
                nn.Parameter(torch.zeros(ssm_shape, dtype=dtype), requires_grad=False)
            )

        logger.info(
            f"Initialized Mamba state persistence: {len(self._mamba_layer_indices)} layers, "
            f"{len(self.mamba_states)} buffers"
        )

    def _get_mamba_states(self):
        """Get Mamba states as list of (conv_state, ssm_state) tuples."""
        states = []
        for i in range(0, len(self.mamba_states), 2):
            states.append((self.mamba_states[i], self.mamba_states[i + 1]))
        return states

    def _build_mamba_state_map(self):
        """Map layer_idx -> mamba_idx for Mamba layers."""
        return {
            layer_idx: mamba_idx
            for mamba_idx, layer_idx in enumerate(self._mamba_layer_indices)
        }

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        update_cache: bool = False,
        is_for_context_encoding: bool = False,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        windowed_context_encoding_window_idx: int = -1,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Thread Mamba state through decoder layers alongside KV cache.
        Returns: (hidden_states, next_decoder_cache, updated_mamba_state_list)
        """
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][1].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = inputs_embeds

        if self.sequence_parallel_enabled:
            self.validate_sequence_parallel(seq_length)
        hidden_states = self.process_sequence_parallel_hidden_states(
            inputs_embeds, seq_length, kwargs.get("active_block_table", None)
        )

        next_decoder_cache = ()
        cos_cache = None
        sin_cache = None
        cache_size = self.n_positions

        if not is_for_context_encoding or windowed_context_encoding_window_idx >= 1:
            past_key_values = self.kv_mgr.get_cache(
                seq_ids=seq_ids,
                seq_len=cache_size,
                is_for_context_encoding=is_for_context_encoding,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        mamba_states = self._get_mamba_states()
        mamba_state_map = self._build_mamba_state_map()
        updated_mamba_states = [None] * len(self._mamba_layer_indices)

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            mamba_state = None
            if idx in mamba_state_map:
                mamba_state = mamba_states[mamba_state_map[idx]]

            layer_outputs = decoder_layer(
                hidden_states,
                seq_ids=seq_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                adapter_ids=adapter_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                kv_mgr=self.kv_mgr,
                idx=idx,
                is_for_context_encoding=is_for_context_encoding,
                seq_len=cache_size,
                local_mask=local_attn_mask,
                padding_mask=padding_mask,
                mamba_state=mamba_state,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:4]
            layer_mamba_state = layer_outputs[5]

            if idx in mamba_state_map and layer_mamba_state is not None:
                updated_mamba_states[mamba_state_map[idx]] = layer_mamba_state

        if update_cache:
            next_decoder_cache = self.kv_mgr.update_cache(
                is_for_context_encoding=is_for_context_encoding,
                seq_ids=seq_ids,
                position_ids=position_ids,
                new_key_values=next_decoder_cache,
                seq_len=cache_size,
                windowed_context_encoding_window_idx=windowed_context_encoding_window_idx,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return (hidden_states, next_decoder_cache, updated_mamba_states)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        tile_q_indices=None,
        tile_block_tables=None,
        tile_masks=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        active_mask=None,
        rotary_position_id=None,
        vision_embeddings=None,
        vision_mask=None,
    ):
        """
        Traced forward -- appends Mamba state tensors to output list.
        Output: [res, K0, V0, ..., conv_state_0, ssm_state_0, ...]
        """
        prev_hidden = self.set_none_if_empty(prev_hidden)
        adapter_ids = self.set_none_if_empty(adapter_ids)
        accepted_indices = self.set_none_if_empty(accepted_indices)
        current_length = self.set_none_if_empty(current_length)
        medusa_mask = self.set_none_if_empty(medusa_mask)
        scatter_index = self.set_none_if_empty(scatter_index)
        slot_mapping = self.set_none_if_empty(slot_mapping)
        active_block_table = self.set_none_if_empty(active_block_table)
        num_queries = self.set_none_if_empty(num_queries)
        computed_context_lens = self.set_none_if_empty(computed_context_lens)
        tile_q_indices = self.set_none_if_empty(tile_q_indices)
        tile_block_tables = self.set_none_if_empty(tile_block_tables)
        tile_masks = self.set_none_if_empty(tile_masks)
        inputs_embeds = self.set_none_if_empty(inputs_embeds)
        kv_cache = self.set_none_if_empty(kv_cache)
        active_mask = self.set_none_if_empty(active_mask)
        rotary_position_id = self.set_none_if_empty(rotary_position_id)
        vision_embeddings = self.set_none_if_empty(vision_embeddings)
        vision_mask = self.set_none_if_empty(vision_mask)

        is_for_context_encoding = self._is_context_encoding(input_ids)

        if attention_mask is not None and is_for_context_encoding:
            padding_mask = attention_mask.float()
        else:
            padding_mask = None

        attn_mask = self.create_attn_mask(
            attention_mask,
            is_for_context_encoding,
            False,
            position_ids=position_ids,
        )

        hidden_states, updated_kv_cache, updated_mamba_states = self.get_model_output(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            is_for_context_encoding=is_for_context_encoding,
            scatter_index=slot_mapping
            if getattr(self, "is_block_kv_layout", False)
            else scatter_index,
            kvcache_buffer=kv_cache,
            update_cache=True,
            padding_mask=padding_mask,
        )

        batch_size = input_ids.shape[0]
        if not self.sliced_hidden:
            if not (
                position_ids.shape[-1] == getattr(self, "speculation_length", 0)
                or position_ids.shape[-1] == 1
            ):
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            from neuronx_distributed_inference.models.model_base import (
                mask_padded_logits,
            )

            logits = mask_padded_logits(
                logits, rank_id, world_size, pad_size=self.lm_head.pad_size
            )

        if self.on_device_sampling:
            res = self._sample_on_device(
                logits, sampling_params, False, is_for_context_encoding
            )
        else:
            res = logits

        outputs = [res]
        if getattr(self.neuron_config, "output_logits", False):
            from neuronx_distributed_inference.models.model_base import (
                _gather_along_dim,
                get_tp_group,
            )

            gathered_logits = _gather_along_dim(
                logits,
                partition_dim=2,
                process_group=get_tp_group(self.config),
            )
            outputs += [gathered_logits]
        outputs += updated_kv_cache

        # Append Mamba states for aliasing
        for conv_state, ssm_state in updated_mamba_states:
            outputs.append(conv_state)
            outputs.append(ssm_state)

        return outputs


# ==============================================================================
# CausalLM Wrapper
# ==============================================================================


class NeuronNemotronForCausalLM(NeuronBaseForCausalLM):
    """Top-level causal LM class for Nemotron-3-Nano-30B."""

    _model_cls = NeuronNemotronModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @classmethod
    def get_config_cls(cls):
        return NemotronHInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: NemotronHInferenceConfig
    ) -> dict:
        return _convert_nemotron_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        args = (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--model-type transformer -O1 "
            "--auto-cast=none "
            "--internal-max-instruction-limit=15000000 "
            "--internal-backend-options='--max-instruction-limit=15000000'"
        )

        # Scratchpad page size: controls HBM scratchpad allocation granularity.
        # Must be paired with NEURON_SCRATCHPAD_PAGE_SIZE env var at runtime.
        if getattr(self.neuron_config, "scratchpad_page_size", None):
            args += (
                f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"
            )

        # Spill-reload DGE: reduces DMA ring memory for long context.
        if getattr(self.neuron_config, "enable_spill_reload_dge", False):
            args += " --internal-enable-dge-levels spill_reload"

        return args

    def get_model_wrapper_cls(self):
        return NemotronModelWrapper

    def shard_weights(
        self, compiled_model_path, debug=False, pre_shard_weights_hook=None
    ):
        """
        Override to guard against multi-process weight loading OOM.

        The full Nemotron model is ~59 GB BF16. If accidentally run with
        torchrun (multi-process), each process would independently load
        the full model, totaling ~240 GB which exceeds 128 GB RAM.

        This override gates on RANK==0 so only one process loads and shards.
        For normal single-process usage (the intended pattern), this is a no-op.
        """
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            import time

            logger.info(f"Rank {rank}: skipping weight sharding (rank 0 only)")
            barrier_file = os.path.join(compiled_model_path, ".shard_complete")
            timeout = 3600
            elapsed = 0
            while not os.path.exists(barrier_file):
                time.sleep(5)
                elapsed += 5
                if elapsed >= timeout:
                    raise RuntimeError(f"Rank {rank}: timed out waiting for sharding")
            logger.info(f"Rank {rank}: sharding complete (waited {elapsed}s)")
            return

        logger.info("Rank 0: starting weight sharding")
        super().shard_weights(compiled_model_path, debug, pre_shard_weights_hook)
        barrier_file = os.path.join(compiled_model_path, ".shard_complete")
        with open(barrier_file, "w") as f:
            f.write("done")
        logger.info("Rank 0: weight sharding complete")

    def _copy_past_key_values(self, outputs):
        """Also copy Mamba states for CPU debugging path."""
        n_mamba_entries = len(self.context_encoding_model.model.mamba_states)

        if n_mamba_entries > 0:
            super()._copy_past_key_values(outputs[:-n_mamba_entries])

            mamba_outputs = outputs[-n_mamba_entries:]
            for i, state_tensor in enumerate(mamba_outputs):
                self.token_generation_model.model.mamba_states[i].data = state_tensor
                self.context_encoding_model.model.mamba_states[i].data = state_tensor
        else:
            super()._copy_past_key_values(outputs)


# ==============================================================================
# State Dict Conversion
# ==============================================================================


def _convert_mamba_conv_weights(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert HF conv1d.weight (3D) to our conv_weight (2D) parameter names."""
    converted = {}
    for key, tensor in state_dict.items():
        if "conv1d.weight" in key:
            new_key = key.replace("conv1d.weight", "conv_weight")
            converted[new_key] = tensor.squeeze(1)
        elif "conv1d.bias" in key:
            new_key = key.replace("conv1d.bias", "conv_bias")
            converted[new_key] = tensor
        else:
            converted[key] = tensor
    return converted


def _split_mamba_projections(
    state_dict: Dict[str, Any], config: NemotronHInferenceConfig
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """No-op: Falcon-H1 pattern uses single ColumnParallelLinear for in_proj/out_proj.

    With gather_output=True, the ColumnParallelLinear's own preshard_hook handles
    TP sharding of the projection weights. No manual splitting or transposing needed.

    Returns:
        (unmodified state_dict, empty dict)
    """
    return state_dict, {}


def _remap_hf_key(key: str, config) -> Optional[str]:
    """Remap a single HF weight key to NxDI format. Returns None to skip."""
    # Skip individual expert weights (they get stacked separately)
    if ".mixer.experts." in key:
        return None

    # Skip shared expert keys -- handled separately in main conversion function
    if ".mixer.shared_experts." in key:
        return None

    # Mamba in_proj and out_proj: pass through (Falcon-H1 pattern).
    # ColumnParallelLinear(gather_output=True) handles TP sharding via its
    # own preshard_hook — no manual splitting or transposing needed.

    # Embeddings
    if key == "backbone.embeddings.weight":
        return "embed_tokens.weight"

    # Final norm (backbone.norm_f -> norm)
    if key == "backbone.norm_f.weight":
        return "norm.weight"

    # LM head (already correct prefix)
    if key == "lm_head.weight":
        return "lm_head.weight"

    # Layer keys
    if key.startswith("backbone.layers."):
        # Strip "backbone." prefix -> "layers.{i}.norm.weight" etc.
        layer_key = key[len("backbone.") :]

        # Determine layer index to route correctly
        parts = layer_key.split(".")
        layer_idx = int(parts[1])
        layer_type = config.layer_types[layer_idx]

        # Norm: layers.{i}.norm.weight -> layers.{i}.input_layernorm.weight
        if f"layers.{layer_idx}.norm.weight" == layer_key:
            return f"layers.{layer_idx}.input_layernorm.weight"

        if layer_type == "attention":
            # Attention: NxDI's preshard_hook in GroupQueryAttention_QKV/O
            # handles the remapping from flat keys to nested module paths.
            # We just need to replace .mixer. with .self_attn.:
            #   HF: backbone.layers.{i}.mixer.q_proj.weight
            #   Our state_dict: layers.{i}.self_attn.q_proj.weight
            #   preshard_hook: -> layers.{i}.self_attn.qkv_proj.q_proj.weight
            if ".mixer." in layer_key:
                return layer_key.replace(".mixer.", ".self_attn.")

        # Mamba and MoE: keep .mixer. prefix as-is
        return layer_key

    return None


def _convert_nemotron_hf_to_neuron_state_dict(
    state_dict: Dict[str, Any], config: NemotronHInferenceConfig
) -> Dict[str, Any]:
    """
    Convert HF Nemotron-H weights to NxDI model format.

    Memory-optimized: processes expert weights one layer at a time,
    deleting originals as they are stacked to avoid holding both
    individual and stacked versions simultaneously.

    HF format (prefix: backbone.):
        backbone.embeddings.weight
        backbone.layers.{i}.norm.weight
        backbone.layers.{i}.mixer.{...}
        backbone.norm_f.weight
        lm_head.weight

    NxDI format:
        embed_tokens.weight
        layers.{i}.input_layernorm.weight
        layers.{i}.mixer.{...}  (Mamba/MoE)
        layers.{i}.self_attn.{...}  (Attention -- preshard_hook adds nesting)
        norm.weight
        lm_head.weight
    """
    new_state_dict = {}

    # First pass: convert Mamba conv1d weight shapes
    state_dict = _convert_mamba_conv_weights(state_dict)

    # Split Mamba in_proj (no-op with Falcon-H1 pattern — returns empty dict).
    # Kept for code structure compatibility.
    state_dict, mamba_proj_entries = _split_mamba_projections(state_dict, config)
    new_state_dict.update(mamba_proj_entries)

    # Identify all expert weight keys and shared expert keys for separate handling
    expert_keys = [k for k in state_dict if ".mixer.experts." in k]
    shared_expert_keys = [k for k in state_dict if ".mixer.shared_experts." in k]

    # Second pass: collect routed expert weights for stacking
    moe_expert_weights = {}  # layer_idx -> {"up": [tensors], "down": [tensors]}

    for key in expert_keys:
        _collect_expert_weight(key, state_dict[key], moe_expert_weights, config)

    # Delete individual expert weights from state_dict BEFORE stacking
    # This frees ~53 GB before we create ~53 GB of stacked tensors
    for key in expert_keys:
        del state_dict[key]
    gc.collect()
    logger.info(f"Freed {len(expert_keys)} individual expert weights before stacking")

    # Handle shared expert weights: remap keys and transpose down_proj
    for key in shared_expert_keys:
        value = state_dict[key]
        # Extract layer index
        parts = key.split(".")
        layers_pos = parts.index("layers")
        layer_idx = int(parts[layers_pos + 1])

        if "up_proj.weight" in key:
            # ColumnParallelLinear expects (out_features, in_features) -- same as HF
            new_state_dict[f"layers.{layer_idx}.mixer.shared_up.weight"] = value
        elif "down_proj.weight" in key:
            # Our shared_down_weight is (I_shared, H) for x @ w pattern
            # HF weight is (H, I_shared) in nn.Linear convention -> transpose
            new_state_dict[f"layers.{layer_idx}.mixer.shared_down_weight"] = value.t()
        del state_dict[key]

    # Remap remaining keys (excludes expert and shared expert keys)
    for key, value in state_dict.items():
        new_key = _remap_hf_key(key, config)
        if new_key is not None:
            new_state_dict[new_key] = value

    # Third pass: stack collected expert weights one layer at a time
    # Stack, add to output, then free the individual tensors
    for layer_idx in sorted(moe_expert_weights.keys()):
        experts = moe_expert_weights[layer_idx]
        if "up" in experts and experts["up"][0] is not None:
            new_state_dict[f"layers.{layer_idx}.mixer.expert_up"] = torch.stack(
                experts["up"], dim=0
            )
            experts["up"] = None  # Free individual tensors
        if "down" in experts and experts["down"][0] is not None:
            new_state_dict[f"layers.{layer_idx}.mixer.expert_down"] = torch.stack(
                experts["down"], dim=0
            )
            experts["down"] = None  # Free individual tensors
        gc.collect()

    del moe_expert_weights
    gc.collect()

    # Add rank utility tensor
    new_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    logger.info(f"State dict conversion complete: {len(new_state_dict)} keys")
    gc.collect()
    return new_state_dict


def _collect_expert_weight(
    key: str, value: torch.Tensor, collection: dict, config
) -> None:
    """Collect and transpose an individual expert weight for later stacking."""
    # Key format: backbone.layers.{i}.mixer.experts.{e}.{up,down}_proj.weight
    # After conv conversion, backbone. prefix is still present
    parts = key.split(".")
    # Find "layers" index
    try:
        layers_pos = parts.index("layers")
        layer_idx = int(parts[layers_pos + 1])
        expert_idx = int(parts[parts.index("experts") + 1])
    except (ValueError, IndexError):
        return

    if layer_idx not in collection:
        num_experts = config.n_routed_experts
        collection[layer_idx] = {
            "up": [None] * num_experts,
            "down": [None] * num_experts,
        }

    # HF Linear weight shape: (out_features, in_features)
    # We need: (in_features, out_features) for x @ w_up pattern
    if "up_proj" in key:
        collection[layer_idx]["up"][expert_idx] = value.t()
    elif "down_proj" in key:
        collection[layer_idx]["down"][expert_idx] = value.t()
