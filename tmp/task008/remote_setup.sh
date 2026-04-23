#!/bin/bash
# Task 008: Remote setup script for Qwen2.5-Omni M-RoPE validation on SDK 2.29
# Run on the trn2.3xlarge instance after SSH

set -e

echo "=== Step 1: Verify SDK 2.29 ==="
# Check which venvs are available
ls /opt/aws_neuronx_venv_*/ 2>/dev/null || echo "No pre-installed venvs found"

# Find and activate the inference venv
# SDK 2.29 may have a different venv name than 2.28
VENV_PATH=$(ls -d /opt/aws_neuronx_venv_pytorch_inference_*/  2>/dev/null | head -1)
if [ -z "$VENV_PATH" ]; then
    # Fallback: look for any neuronx venv
    VENV_PATH=$(ls -d /opt/aws_neuronx_venv_pytorch_*/  2>/dev/null | head -1)
fi
echo "Using venv: $VENV_PATH"
source ${VENV_PATH}bin/activate

pip list | grep -E "neuronx|torch|vllm" 2>/dev/null || true
neuron-ls

echo ""
echo "=== Step 2: Mount NVMe ==="
if [ ! -d /opt/dlami/nvme ]; then
    sudo mkfs.xfs /dev/nvme0n1 2>/dev/null || true
    sudo mkdir -p /opt/dlami/nvme
    sudo mount /dev/nvme0n1 /opt/dlami/nvme
    sudo chown ubuntu:ubuntu /opt/dlami/nvme
fi
df -h /opt/dlami/nvme

echo ""
echo "=== Step 3: Download Qwen2.5-Omni-7B ==="
mkdir -p /opt/dlami/nvme/models
if [ ! -d /opt/dlami/nvme/models/Qwen2.5-Omni-7B ]; then
    pip install huggingface_hub 2>/dev/null
    huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /opt/dlami/nvme/models/Qwen2.5-Omni-7B
else
    echo "Model already downloaded"
fi

echo ""
echo "=== Step 4: Install NxDI from Henan's branch ==="
cd /opt/dlami/nvme
if [ ! -d neuronx-distributed-inference ]; then
    git clone https://github.com/whn09/neuronx-distributed-inference.git
    cd neuronx-distributed-inference
    git checkout feature/qwen25-omni-support
else
    cd neuronx-distributed-inference
    git fetch origin
    git checkout feature/qwen25-omni-support
    git pull
fi

pip install -e . 2>&1 | tail -5

echo ""
echo "=== Setup complete ==="
echo "Model path: /opt/dlami/nvme/models/Qwen2.5-Omni-7B"
echo "NxDI path: /opt/dlami/nvme/neuronx-distributed-inference"
echo ""
echo "Next: Run the validation script"
echo "  python3 /opt/dlami/nvme/test_omni_mrope.py --phase before"
echo "  # Then apply M-RoPE fix and rerun with --phase after"
