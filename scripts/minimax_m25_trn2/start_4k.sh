#!/bin/bash
# MiniMax M2.5 BF16 - 4K config
# max-model-len=4400, max-num-seqs=64, batch_size=64
# context_encoding=[4096], token_generation=[4400]
# Input: 3700, Output: 300

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
cd /home/ubuntu/vllm-neuron
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy llm --retry_failed_compilation"

python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/models/MiniMax-M2.5-BF16" \
    --tokenizer "/opt/dlami/nvme/models/MiniMax-M2.5-BF16" \
    --tensor-parallel-size 64 \
    --max-model-len 4400 \
    --max-num-seqs 64 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --trust-remote-code \
    --additional-config '{
        "override_neuron_config": {
            "sequence_parallel_enabled": true,
            "logical_nc_config": 2,
            "fused_qkv": true,
            "is_continuous_batching": true,
            "batch_size": 64,
            "ctx_batch_size": 1,
            "tkg_batch_size": 64,
            "max_context_length": 4400,
            "seq_len": 4400,
            "async_mode": true,
            "flash_decoding_enabled": false,
            "enable_bucketing": true,
            "context_encoding_buckets": [4096],
            "token_generation_buckets": [4400],
            "use_index_calc_kernel": false,
            "moe_mask_padded_tokens": true,
            "qkv_kernel_enabled": true,
            "qkv_nki_kernel_enabled": true,
            "qkv_cte_nki_kernel_fuse_rope": true,
            "attn_kernel_enabled": true,
            "strided_context_parallel_kernel_enabled": false,
            "normalize_top_k_affinities": true,
            "router_config": {
                "act_fn": "sigmoid",
                "dtype": "float32"
            },
            "glu_mlp": true,
            "save_sharded_checkpoint": true,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": false,
                "skip_dma_token": true
            },
            "disable_numeric_cc_token": true,
            "scratchpad_page_size": 1024,
            "moe_tp_degree": 64,
            "moe_ep_degree": 1
        }
    }'
