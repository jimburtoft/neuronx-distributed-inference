#!/bin/bash
# MiniMax M2.5 BF16 - 8K benchmark
# Input: 7400, Output: 300, Concurrency: 4/8/16/32/64
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
python3 /tmp/bench_serving.py \
    --model /opt/dlami/nvme/models/MiniMax-M2.5-BF16 \
    --input-lens 7400 \
    --output-tokens 300 \
    --concurrencies 4 8 16 32 64 \
    --random-range-ratio 0.1 \
    --stream \
    --output-dir /tmp/minimax_m25_results \
    --label m25_bf16_8k
