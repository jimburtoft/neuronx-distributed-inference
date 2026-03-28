# MiniMax M2.5 BF16 Benchmark on Trn2 (trn2.48xlarge)

## Test Configuration

- **Model**: MiniMax-M2.5 BF16 (~430B params, 256 experts, top-8 routing)
- **Hardware**: trn2.48xlarge (16x Trainium2 chips, 32 NeuronCores, 64 logical with NC=2)
- **Software**: vLLM-Neuron 0.16.0, NxDI (neuronx-distributed-inference)
- **Serving config**: TP=64, max_model_len=4400, max_num_seqs=64, batch_size=64, continuous batching
- **Neuron config**: fused_qkv=true, async_mode=true, qkv_kernel_enabled=true, attn_kernel_enabled=true
- **Buckets**: context_encoding=[4096], token_generation=[4400]
- **Benchmark config**: output=300 tokens, random_range_ratio=0.1, streaming mode
- **Date**: 2026-03-28

## Output Token Throughput (tok/s)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 8.9 | 16.4 | 28.3 | 44.6 | 62.6 |

## Time To First Token - TTFT (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 7.55 | 13.26 | 24.69 | 47.54 | 93.25 |

## Time Per Output Token - TPOT (ms)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 450.1 | 488.2 | 564.5 | 716.9 | 1,021.9 |

## P99 Latency (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 135.0 | 146.5 | 169.4 | 215.1 | 306.6 |

## Error Rate

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 0/4 | 0/8 | 0/16 | 0/32 | 0/64 |

## Trn2 vs H100 Comparison - Output Throughput (tok/s)

| Input Length | Trn2 C=4 | H100 C=4 | Ratio | Trn2 C=64 | H100 C=64 | Ratio |
|-------------|----------|----------|-------|-----------|-----------|-------|
| 3,700 (4K)  | 8.9 | 372.2 | **2.4%** | 62.6 | 1,907.2 | **3.3%** |

## Trn2 vs H100 Comparison - TTFT (seconds)

| Input Length | Trn2 C=4 | H100 C=4 | Trn2 C=64 | H100 C=64 |
|-------------|----------|----------|-----------|-----------|
| 3,700 (4K)  | 7.55 | 3.21 | 93.25 | 9.83 |

## Trn2 vs H100 Comparison - TPOT (ms)

| Input Length | Trn2 C=4 | H100 C=4 | Ratio | Trn2 C=64 | H100 C=64 | Ratio |
|-------------|----------|----------|-------|-----------|-----------|-------|
| 3,700 (4K)  | 450.1 | 10.7 | **42x** | 1,021.9 | 32.8 | **31x** |

## Key Observations

1. **Throughput extremely low**: Peak 62.6 tok/s at C=64, only 3.3% of H100 BF16 (1,907 tok/s)
2. **TPOT 31-42x worse than H100**: 450ms (Trn2) vs 10.7ms (H100) at C=4, indicating severe decode bottleneck
3. **TTFT 2-9x worse than H100**: 7.55s vs 3.21s at C=4, 93.25s vs 9.83s at C=64
4. **TPOT degrades sharply with concurrency**: 450ms (C=4) → 1,022ms (C=64), 2.3x increase
5. **No errors**: 0% error rate across all configurations
6. **Compilation overhead**: ~17 min compilation + ~18 min weight pre-sharding (first run only)
7. **Batch padding overhead**: batch_size=64 means all token generation pads to 64 sequences, contributing to poor throughput at low concurrency

## Possible Improvements

- **Expert Parallelism (EP)**: Current config uses moe_ep_degree=1 (all 256 experts on every rank). EP could reduce per-rank computation
- **Smaller batch_size for low concurrency**: batch_size=64 causes excessive padding; separate configs per concurrency range would help
- **Flash Decoding**: Currently disabled; enabling may improve token generation performance
- **Context Parallelism (CP)**: Not supported for MiniMax M2 model yet

## Raw Data

See [minimax-m25-bf16-trn2-summary.csv](minimax-m25-bf16-trn2-summary.csv) for full CSV results.
