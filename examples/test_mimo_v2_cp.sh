#!/bin/bash
# =============================================================================
# MiMo-V2-Flash Context Parallelism Test Script
# =============================================================================
# 目标：测试 MiMo-V2-Flash 在 Trainium2 上的 CP 支持
# 对标 Qwen3-235B-A22B 的配置风格：
#   Qwen3: tp=64, cp=16, moe_tp=2, moe_ep=32
#   MiMo:  tp=64, cp=2,  moe_tp=2, moe_ep=32 (先用 cp=2 验证，再提升)
#
# 用法：
#   bash examples/test_mimo_v2_cp.sh [cp_degree] [moe_tp_degree] [moe_ep_degree]
#
# 示例：
#   bash examples/test_mimo_v2_cp.sh 2 2 32     # 保守测试
#   bash examples/test_mimo_v2_cp.sh 16 2 32    # 对标 Qwen3
#   bash examples/test_mimo_v2_cp.sh 1 1 64     # baseline（无CP，纯EP）
# =============================================================================

set -euo pipefail

# === 参数 ===
CP_DEGREE=${1:-2}
MOE_TP=${2:-2}
MOE_EP=${3:-32}
MODEL_PATH="${MODEL_PATH:-/opt/dlami/nvme/models/MiMo-V2-Flash-BF16}"
PORT=8000
MAX_MODEL_LEN=1024
MAX_NUM_SEQS=32
BATCH_SIZE=32

# === 验证 ===
if [ $((MOE_TP * MOE_EP)) -ne 64 ]; then
    echo "ERROR: moe_tp_degree($MOE_TP) * moe_ep_degree($MOE_EP) must equal 64"
    exit 1
fi
if [ $((64 % CP_DEGREE)) -ne 0 ]; then
    echo "ERROR: tp_degree(64) must be divisible by cp_degree($CP_DEGREE)"
    exit 1
fi

TAG="cp${CP_DEGREE}_mtp${MOE_TP}_mep${MOE_EP}"
LOG="/tmp/vllm_mimo_${TAG}.log"
RESULT="/tmp/mimo_test_${TAG}_result.txt"

echo "=============================================="
echo "MiMo-V2-Flash CP Test"
echo "=============================================="
echo "Config: cp_degree=$CP_DEGREE, moe_tp=$MOE_TP, moe_ep=$MOE_EP"
echo "Model:  $MODEL_PATH"
echo "Log:    $LOG"
echo "Result: $RESULT"
echo "Time:   $(date)"
echo "=============================================="

# === 环境 ===
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export VLLM_RPC_TIMEOUT=600000
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export PYTHONPATH=/home/ubuntu/neuronx-distributed-inference/src:${PYTHONPATH:-}

# === 清理旧进程 ===
echo "[$(date +%H:%M:%S)] Cleaning up old processes..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# === 启动 vLLM 服务 ===
echo "[$(date +%H:%M:%S)] Starting vLLM server..."

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --tensor-parallel-size 64 \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port $PORT \
    --trust-remote-code \
    --additional-config "{
        \"override_neuron_config\": {
            \"tp_degree\": 64,
            \"cp_degree\": ${CP_DEGREE},
            \"moe_tp_degree\": ${MOE_TP},
            \"moe_ep_degree\": ${MOE_EP},
            \"batch_size\": ${BATCH_SIZE},
            \"ctx_batch_size\": 1,
            \"tkg_batch_size\": ${BATCH_SIZE},
            \"max_context_length\": ${MAX_MODEL_LEN},
            \"seq_len\": ${MAX_MODEL_LEN},
            \"is_continuous_batching\": true,
            \"fused_qkv\": false,
            \"on_device_sampling_config\": {
                \"do_sample\": true,
                \"temperature\": 0.6,
                \"top_k\": 20,
                \"top_p\": 0.95
            },
            \"enable_bucketing\": true,
            \"context_encoding_buckets\": [${MAX_MODEL_LEN}],
            \"token_generation_buckets\": [${MAX_MODEL_LEN}],
            \"flash_decoding_enabled\": false,
            \"logical_nc_config\": 2,
            \"sequence_parallel_enabled\": true,
            \"qkv_kernel_enabled\": false,
            \"qkv_nki_kernel_enabled\": false,
            \"qkv_cte_nki_kernel_fuse_rope\": false,
            \"attn_kernel_enabled\": true,
            \"strided_context_parallel_kernel_enabled\": false,
            \"async_mode\": true,
            \"glu_mlp\": true,
            \"normalize_top_k_affinities\": true,
            \"router_config\": {
                \"act_fn\": \"sigmoid\",
                \"dtype\": \"float32\"
            },
            \"use_index_calc_kernel\": true,
            \"moe_mask_padded_tokens\": true,
            \"blockwise_matmul_config\": {
                \"use_shard_on_intermediate_dynamic_while\": true,
                \"skip_dma_token\": true
            },
            \"disable_numeric_cc_token\": true,
            \"scratchpad_page_size\": 1024
        }
    }" > "$LOG" 2>&1 &

SERVER_PID=$!
echo "[$(date +%H:%M:%S)] Server PID: $SERVER_PID"

# === 等待服务就绪 ===
echo "[$(date +%H:%M:%S)] Waiting for server (compilation + weight loading)..."
echo "  This may take 10-60+ minutes depending on cache..."
MAX_WAIT=7200
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] Server READY after ${WAITED}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] ERROR: Server process died!"
        echo ""
        echo "=== Error from log ==="
        grep -E "Error|Exception|Traceback|FAILED|NotImplementedError" "$LOG" | \
            grep -v "FutureWarning\|DeprecationWarning\|UserWarning\|enable_spmd_rank\|import_nki\|libcuda\|model of type" | \
            tail -10
        echo ""
        echo "=== Last 20 lines ==="
        tail -20 "$LOG"
        echo ""
        echo "FAILED" > "$RESULT"
        exit 1
    fi
    sleep 30
    WAITED=$((WAITED + 30))
    # 显示进度
    LAST_LINE=$(tail -1 "$LOG" 2>/dev/null | head -c 120)
    echo "  [${WAITED}s] $LAST_LINE"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not start within ${MAX_WAIT}s"
    kill $SERVER_PID 2>/dev/null
    echo "TIMEOUT" > "$RESULT"
    exit 1
fi

# === 推理测试 ===
echo ""
echo "=============================================="
echo "Inference Tests"
echo "=============================================="

run_test() {
    local test_name="$1"
    local prompt="$2"
    local max_tokens="${3:-128}"

    echo ""
    echo "--- $test_name ---"
    local START=$(date +%s%N)
    local RESP=$(curl -s --max-time 120 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_PATH\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": $max_tokens,
            \"temperature\": 0.6
        }")
    local END=$(date +%s%N)
    local ELAPSED=$(( (END - START) / 1000000 ))

    local CONTENT=$(echo "$RESP" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    c = r['choices'][0]['message']['content']
    u = r.get('usage', {})
    print(f'Content: {c[:200]}')
    print(f'Tokens: prompt={u.get(\"prompt_tokens\",\"?\")}, completion={u.get(\"completion_tokens\",\"?\")}, total={u.get(\"total_tokens\",\"?\")}')
except Exception as e:
    print(f'Parse error: {e}')
    print(f'Raw: {sys.stdin.read()[:200] if hasattr(sys.stdin, \"read\") else \"N/A\"}')
" 2>&1)

    echo "$CONTENT"
    echo "Latency: ${ELAPSED}ms"
}

run_test "Test 1: Math" "What is 25 * 37? Answer with just the number." 32
run_test "Test 2: Knowledge" "What is the capital of France? One word answer." 32
run_test "Test 3: Reasoning" "If I have 3 apples and give away 1, how many do I have? Just the number." 32
run_test "Test 4: Generation" "Write a haiku about the ocean." 64

# === 吞吐量测试 ===
echo ""
echo "=============================================="
echo "Throughput Test (concurrent requests)"
echo "=============================================="

CONCURRENT=8
echo "Sending $CONCURRENT concurrent requests..."
START_TP=$(date +%s%N)

for i in $(seq 1 $CONCURRENT); do
    curl -s --max-time 300 http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_PATH\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 50.\"}],
            \"max_tokens\": 256,
            \"temperature\": 0.6
        }" > /tmp/mimo_concurrent_${i}.json &
done
wait

END_TP=$(date +%s%N)
ELAPSED_TP=$(( (END_TP - START_TP) / 1000000 ))

TOTAL_TOKENS=0
for i in $(seq 1 $CONCURRENT); do
    T=$(python3 -c "
import json
try:
    r = json.load(open('/tmp/mimo_concurrent_${i}.json'))
    print(r.get('usage',{}).get('completion_tokens', 0))
except: print(0)
" 2>/dev/null)
    TOTAL_TOKENS=$((TOTAL_TOKENS + T))
done

if [ $ELAPSED_TP -gt 0 ]; then
    THROUGHPUT=$(python3 -c "print(f'{${TOTAL_TOKENS} / (${ELAPSED_TP} / 1000):.1f}')")
else
    THROUGHPUT="N/A"
fi

echo "Concurrent=$CONCURRENT, Total tokens=$TOTAL_TOKENS, Time=${ELAPSED_TP}ms"
echo "Throughput: $THROUGHPUT tok/s"

# === 记录内存 ===
echo ""
echo "=============================================="
echo "System Info"
echo "=============================================="
echo "RSS: $(ps -p $SERVER_PID -o rss= 2>/dev/null | awk '{printf "%.2f GB\n", $1/1024/1024}')"
free -h | head -2

# === 清理 ===
echo ""
echo "[$(date +%H:%M:%S)] Stopping server..."
kill $SERVER_PID 2>/dev/null
sleep 5
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true

# === 写结果 ===
cat > "$RESULT" << RESULT_EOF
=== MiMo-V2-Flash CP Test Result ===
Config: cp_degree=$CP_DEGREE, moe_tp=$MOE_TP, moe_ep=$MOE_EP
Batch:  $BATCH_SIZE, MaxLen: $MAX_MODEL_LEN
Status: SUCCESS
Concurrent throughput: $THROUGHPUT tok/s ($CONCURRENT requests)
Date:   $(date)
RESULT_EOF

echo ""
echo "=============================================="
echo "Test complete! Results saved to $RESULT"
echo "Full log: $LOG"
echo "=============================================="
