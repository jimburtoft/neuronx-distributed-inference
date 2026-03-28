#!/bin/bash
# MiniMax M2.5 BF16 - 32K benchmark
# Input: 29700, Output: 300, Concurrency: 4/8/16
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
python3 /tmp/bench_serving.py \
    --model /opt/dlami/nvme/models/MiniMax-M2.5-BF16 \
    --input-lens 29700 \
    --output-tokens 300 \
    --concurrencies 4 8 16 \
    --random-range-ratio 0.1 \
    --stream \
    --output-dir /tmp/minimax_m25_results \
    --label m25_bf16_32k
