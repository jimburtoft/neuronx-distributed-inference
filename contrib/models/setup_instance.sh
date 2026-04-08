#!/bin/bash
# Setup script for sweeper validation on inf2.xlarge
# Run: bash setup.sh

set -e

echo "========================================"
echo "Sweeper Validation Instance Setup"
echo "========================================"

# Activate pre-installed venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
echo "Activated venv: $(which python)"
echo "Python: $(python --version)"

# Clone the fork repo
echo ""
echo "Cloning fork repo..."
cd /home/ubuntu
if [ ! -d "neuronx-distributed-inference" ]; then
    git clone https://github.com/jimburtoft/neuronx-distributed-inference.git
fi
cd neuronx-distributed-inference

# Install NxDI in dev mode
echo ""
echo "Installing NxDI (this may take a few minutes on first use)..."
pip install -e . 2>&1 | tail -5

# Fetch both branches
echo ""
echo "Fetching sweeper branches..."
git fetch origin sweeper/fix-qwen25-vl-3b
git fetch origin sweeper/fix-qwen25-omni-mrope

# Create model directories
mkdir -p /home/ubuntu/models
mkdir -p /home/ubuntu/neuron_models

# Download Qwen2.5-VL-3B-Instruct
echo ""
echo "Downloading Qwen2.5-VL-3B-Instruct..."
if [ ! -d "/home/ubuntu/models/Qwen2.5-VL-3B-Instruct" ]; then
    pip install huggingface_hub 2>/dev/null
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    local_dir='/home/ubuntu/models/Qwen2.5-VL-3B-Instruct',
    ignore_patterns=['*.bin', '*.gguf'],
)
print('VL-3B download complete.')
"
else
    echo "VL-3B already downloaded."
fi

# Download Qwen2.5-Omni-7B (thinker weights only -- this is a large model)
echo ""
echo "Downloading Qwen2.5-Omni-7B..."
if [ ! -d "/home/ubuntu/models/Qwen2.5-Omni-7B" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-Omni-7B',
    local_dir='/home/ubuntu/models/Qwen2.5-Omni-7B',
    ignore_patterns=['*.bin', '*.gguf'],
)
print('Omni-7B download complete.')
"
else
    echo "Omni-7B already downloaded."
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "To validate VL-3B:"
echo "  cd /home/ubuntu/neuronx-distributed-inference"
echo "  git checkout sweeper/fix-qwen25-vl-3b"
echo "  cd contrib/models/Qwen2.5-VL-3B-Instruct"
echo "  python validate.py --model-path /home/ubuntu/models/Qwen2.5-VL-3B-Instruct"
echo ""
echo "To validate Omni:"
echo "  cd /home/ubuntu/neuronx-distributed-inference"
echo "  git checkout sweeper/fix-qwen25-omni-mrope"
echo "  cd contrib/models/Qwen2.5-Omni-7B"
echo "  python validate.py --model-path /home/ubuntu/models/Qwen2.5-Omni-7B"
echo "========================================"
