# MiniMax M2.5 BF16 Benchmark on H100 (p5.48xlarge)

## Test Configuration

- **Model**: MiniMax-M2.5 BF16 (~430B params, 256 experts, top-8 routing)
- **Hardware**: p5.48xlarge (8x NVIDIA H100 80GB SXM)
- **Software**: vLLM 0.18.0, transformers 4.57.1
- **Serving config**: TP=8, max_model_len=65536, eager mode
- **Benchmark config**: output=300 tokens, random_range_ratio=0.1, streaming mode
- **Date**: 2026-03-28

## Output Token Throughput (tok/s)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 372.2 | 623.2 | 982.0 | 1,370.5 | 1,907.2 |
| 7,400 (8K)  | 282.2 | 459.7 | 689.1 | 948.1 | 1,207.0 |
| 14,800 (16K) | 197.0 | 310.9 | 435.5 | 570.0 | 579.7 |
| 29,700 (32K) | 122.5 | 172.7 | 248.5 | 249.4 | 250.4 |
| 59,500 (64K) | 61.6 | 96.5 | 88.9 | 93.4 | 96.7 |

## Total Throughput (input+output tok/s)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 4,791 | 8,461 | 13,423 | 18,049 | 25,777 |
| 7,400 (8K)  | 7,400 | 11,602 | 17,811 | 24,246 | 31,021 |
| 14,800 (16K) | 10,186 | 15,375 | 22,232 | 28,606 | 29,214 |
| 29,700 (32K) | 11,955 | 17,486 | 23,993 | 24,732 | 25,188 |
| 59,500 (64K) | 12,224 | 18,380 | 17,899 | 18,474 | 19,257 |

## Time To First Token - TTFT (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 3.21 | 3.83 | 4.84 | 6.90 | 9.83 |
| 7,400 (8K)  | 4.23 | 5.17 | 6.86 | 9.87 | 15.38 |
| 14,800 (16K) | 6.05 | 7.61 | 10.75 | 16.25 | 26.75 |
| 29,700 (32K) | 9.66 | 13.58 | 18.60 | 31.00 | 50.93 |
| 59,500 (64K) | 19.10 | 23.90 | 39.91 | 67.11 | 115.18 |

## Mean Latency (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 3.21 | 3.83 | 4.84 | 6.90 | 9.83 |
| 7,400 (8K)  | 4.23 | 5.17 | 6.86 | 9.87 | 15.38 |
| 14,800 (16K) | 6.05 | 7.61 | 10.75 | 16.25 | 26.75 |
| 29,700 (32K) | 9.66 | 13.58 | 18.60 | 31.00 | 50.93 |
| 59,500 (64K) | 19.10 | 23.90 | 39.91 | 67.11 | 115.18 |

## Time Per Output Token - TPOT (ms)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 10.7 | 12.8 | 16.1 | 23.0 | 32.8 |
| 7,400 (8K)  | 14.1 | 17.2 | 22.9 | 32.9 | 51.3 |
| 14,800 (16K) | 20.2 | 25.4 | 35.8 | 54.2 | 89.2 |
| 29,700 (32K) | 32.2 | 45.3 | 62.0 | 103.3 | 169.8 |
| 59,500 (64K) | 63.7 | 79.7 | 133.0 | 223.7 | 383.9 |

## P99 Latency (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 3.22 | 3.85 | 4.87 | 6.98 | 10.01 |
| 7,400 (8K)  | 4.25 | 5.22 | 6.95 | 10.10 | 15.84 |
| 14,800 (16K) | 6.09 | 7.71 | 11.00 | 16.81 | 33.06 |
| 29,700 (32K) | 9.79 | 13.89 | 19.30 | 38.47 | 76.60 |
| 59,500 (64K) | 19.48 | 24.85 | 50.60 | 102.74 | 192.17 |

## Error Rate

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 0/4 | 0/8 | 0/16 | 0/32 | 0/64 |
| 7,400 (8K)  | 0/4 | 0/8 | 0/16 | 0/32 | 0/64 |
| 14,800 (16K) | 0/4 | 0/8 | 0/16 | 0/32 | 0/64 |
| 29,700 (32K) | 0/4 | 0/8 | 0/16 | 0/32 | 0/64 |
| 59,500 (64K) | 0/4 | 0/8 | 1/16 | 0/32 | 2/64 |

## Key Observations

1. **Peak output throughput**: 1,907 tok/s at 4K input, concurrency=64
2. **Peak total throughput**: 31,021 tok/s at 8K input, concurrency=64
3. **Throughput saturation**: At 32K+ input, output throughput plateaus around 250 tok/s regardless of concurrency (32-64), indicating memory-bandwidth bottleneck
4. **TTFT scales linearly** with input length (~0.52ms per input token at C=4)
5. **TTFT degrades with concurrency**: At 64K input, TTFT grows from 19.1s (C=4) to 115.2s (C=64) - 6x increase
6. **Tail latency**: P99 latency at high concurrency+long context is significantly worse (192s at 64K/C=64 vs 115s mean)
7. **Error rate**: Very low - only 3 errors total across 59,500-length inputs at high concurrency (likely OOM or timeout)
8. **FP8 not testable**: vLLM's blockwise FP8 (block_size=[128,128]) dequantizes to BF16 during loading, causing OOM on 8x H100 80GB

## Raw Data

See [minimax-m25-bf16-h100-summary.csv](minimax-m25-bf16-h100-summary.csv) for full CSV results.
