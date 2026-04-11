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

from .configuration_minimax_m2 import MiniMaxM2Config
from .modeling_minimax_m2 import (
    MiniMaxM2InferenceConfig,
    NeuronMiniMaxM2Attention,
    NeuronMiniMaxM2DecoderLayer,
    NeuronMiniMaxM2ForCausalLM,
    NeuronMiniMaxM2Model,
)

# v2 implementation following Qwen3 MoE pattern
from .modeling_minimax_m2_v2 import (
    MiniMaxM2InferenceConfig as MiniMaxM2InferenceConfigV2,
    NeuronMiniMaxM2Attention as NeuronMiniMaxM2AttentionV2,
    NeuronMiniMaxM2DecoderLayer as NeuronMiniMaxM2DecoderLayerV2,
    NeuronMiniMaxM2ForCausalLM as NeuronMiniMaxM2ForCausalLMV2,
    NeuronMiniMaxM2Model as NeuronMiniMaxM2ModelV2,
)

# v3 implementation with SDK 2.28/2.29 fixes (RouterTopKWithBias, Neuron-native QK norm,
# direct MoE construction for correct weight loading, expert_index torch.long)
from .modeling_minimax_m2_v3 import (
    MiniMaxM2InferenceConfigV3,
    NeuronMiniMaxM2AttentionV3,
    NeuronMiniMaxM2DecoderLayerV3,
    NeuronMiniMaxM2ForCausalLMV3,
    NeuronMiniMaxM2ModelV3,
    convert_minimax_m2_hf_to_neuron_state_dict,
)

__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2InferenceConfig",
    "NeuronMiniMaxM2Attention",
    "NeuronMiniMaxM2DecoderLayer",
    "NeuronMiniMaxM2ForCausalLM",
    "NeuronMiniMaxM2Model",
    # v2 exports
    "MiniMaxM2InferenceConfigV2",
    "NeuronMiniMaxM2AttentionV2",
    "NeuronMiniMaxM2DecoderLayerV2",
    "NeuronMiniMaxM2ForCausalLMV2",
    "NeuronMiniMaxM2ModelV2",
    # v3 exports
    "MiniMaxM2InferenceConfigV3",
    "NeuronMiniMaxM2AttentionV3",
    "NeuronMiniMaxM2DecoderLayerV3",
    "NeuronMiniMaxM2ForCausalLMV3",
    "NeuronMiniMaxM2ModelV3",
    "convert_minimax_m2_hf_to_neuron_state_dict",
]
