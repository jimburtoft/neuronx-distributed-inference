#!/bin/bash
# Task 008: Apply M-RoPE fix to Henan's NxDI branch on remote instance
# Run between "before" and "after" test phases

set -e

NXDI_PATH="/opt/dlami/nvme/neuronx-distributed-inference"
FILE="$NXDI_PATH/src/neuronx_distributed_inference/models/qwen25_omni/modeling_qwen25_omni.py"

echo "Applying M-RoPE fix to text-only path..."

python3 << 'PYEOF'
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/opt/dlami/nvme/neuronx-distributed-inference/src/neuronx_distributed_inference/models/qwen25_omni/modeling_qwen25_omni.py"
with open(filepath, "r") as f:
    content = f.read()

# Edit 1: Update header comment
content = content.replace(
    "#   1. Text-only (Thinker): NeuronQwen25OmniForCausalLM\n"
    "#      - Reuses Qwen2 decoder with thinker.model.* prefix remapping",
    "#   1. Text-only (Thinker): NeuronQwen25OmniForCausalLM\n"
    "#      - Reuses Qwen2-VL text model with multimodal RoPE (mrope_section=[16,24,24])\n"
    "#      - Weight keys remapped from thinker.model.* prefix"
)

# Edit 2: Add Qwen2-VL import
content = content.replace(
    "from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (\n"
    "    NeuronQwen2ForCausalLM,\n"
    "    NeuronQwen2Model,\n"
    "    convert_state_dict_to_fused_qkv,\n"
    ")",
    "from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (\n"
    "    NeuronQwen2ForCausalLM,\n"
    "    NeuronQwen2Model,\n"
    "    convert_state_dict_to_fused_qkv,\n"
    ")\n"
    "from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (\n"
    "    NeuronQwen2VLTextForCausalLM,\n"
    "    NeuronQwen2VLTextModel,\n"
    ")"
)

# Edit 3: Add rope_scaling to _TEXT_CONFIG_ATTRS
content = content.replace(
    '    "rope_theta",\n    "rms_norm_eps",',
    '    "rope_theta",\n    "rope_scaling",\n    "rms_norm_eps",'
)

# Edit 4: Change class inheritance and docstring
content = content.replace(
    "class NeuronQwen25OmniForCausalLM(NeuronQwen2ForCausalLM):\n"
    '    """Qwen2.5-Omni Thinker text model for Causal LM on Neuron.\n'
    "\n"
    "    Reuses the Qwen2 model architecture since the Thinker's text backbone\n"
    "    is architecturally identical to Qwen2.5. The main differences are:\n"
    "      - Weight keys are prefixed with 'thinker.model.' / 'thinker.lm_head.'\n"
    "      - Non-text weights (talker, token2wav, audio_tower, visual) are discarded\n"
    '    """\n'
    "\n"
    "    _model_cls = NeuronQwen2Model",
    "class NeuronQwen25OmniForCausalLM(NeuronQwen2VLTextForCausalLM):\n"
    '    """Qwen2.5-Omni Thinker text model for Causal LM on Neuron.\n'
    "\n"
    "    Uses the Qwen2-VL text model (which has M-RoPE support) since the\n"
    "    Thinker's text backbone requires multimodal rotary position embeddings\n"
    "    with mrope_section=[16, 24, 24]. For text-only input, all three M-RoPE\n"
    "    axes receive identical position IDs.\n"
    "\n"
    "    Key differences from base Qwen2-VL text model:\n"
    "      - Weight keys are prefixed with 'thinker.model.' / 'thinker.lm_head.'\n"
    "      - Non-text weights (talker, token2wav, audio_tower, visual) are discarded\n"
    '    """\n'
    "\n"
    "    _model_cls = NeuronQwen2VLTextModel"
)

with open(filepath, "w") as f:
    f.write(content)

print("All 4 edits applied successfully.")
PYEOF

echo ""
echo "Verifying changes..."
grep -n "class NeuronQwen25OmniForCausalLM" "$FILE"
grep -n "rope_scaling" "$FILE" | head -3
grep -n "NeuronQwen2VLText" "$FILE" | head -5

echo ""
echo "M-RoPE fix applied. Now run:"
echo "  python3 /opt/dlami/nvme/test_omni_mrope.py --phase after"
