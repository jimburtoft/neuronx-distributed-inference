# MiniMax M2.5 FP8 (Dynamic) Benchmark on H100 (p5.48xlarge)

## Test Configuration

- **Model**: MiniMax-M2.5 BF16 weights with `--quantization fp8` (dynamic FP8)
- **Hardware**: p5.48xlarge (8x NVIDIA H100 80GB SXM)
- **Software**: vLLM 0.18.0, PyTorch 2.10.0+cu126
- **Serving config**: TP=8, max_model_len=65536, eager mode, dynamic FP8 quantization
- **Note**: Dynamic FP8 loads BF16 weights (~76GB/GPU) and quantizes at runtime. Not true static FP8.
- **Benchmark config**: output=300 tokens, random_range_ratio=0.1, streaming mode
- **Date**: 2026-03-28

## Output Token Throughput (tok/s)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 45.5 | 89.3 | 175.0 | 333.9 | 562.3 |
| 7,400 (8K)  | 44.8 | 87.5 | 167.6 | 308.0 | 506.4 |
| 14,800 (16K) | 43.7 | 83.2 | 152.8 | 259.0 | 390.9 |
| 29,700 (32K) | 41.0 | 75.1 | 126.2 | 192.9 | 217.1 |
| 59,500 (64K) | 36.0 | 60.2 | 85.2 | 97.5 | 112.4 |

## Time To First Token - TTFT (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 0.27 | 0.55 | 0.75 | 1.47 | 2.60 |
| 7,400 (8K)  | 0.48 | 0.82 | 1.43 | 2.71 | 5.06 |
| 14,800 (16K) | 0.99 | 1.54 | 2.91 | 5.72 | 10.86 |
| 29,700 (32K) | 2.18 | 3.43 | 6.54 | 12.13 | 28.69 |
| 59,500 (64K) | 4.69 | 7.61 | 15.86 | 34.00 | 70.48 |

## Time Per Output Token - TPOT (ms)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 87.8 | 89.1 | 90.3 | 93.6 | 109.4 |
| 7,400 (8K)  | 88.8 | 90.3 | 93.4 | 99.8 | 117.7 |
| 14,800 (16K) | 90.8 | 94.3 | 100.8 | 114.9 | 146.4 |
| 29,700 (32K) | 96.0 | 102.9 | 118.6 | 149.1 | 227.0 |
| 59,500 (64K) | 107.9 | 125.2 | 171.1 | 257.7 | 390.4 |

## P99 Latency (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 26.39 | 26.88 | 27.41 | 28.72 | 34.09 |
| 7,400 (8K)  | 26.76 | 27.41 | 28.62 | 31.14 | 37.85 |
| 14,800 (16K) | 27.49 | 28.83 | 31.40 | 37.03 | 49.06 |
| 29,700 (32K) | 29.25 | 31.96 | 38.01 | 49.73 | 88.37 |
| 59,500 (64K) | 33.30 | 39.85 | 56.33 | 98.42 | 170.75 |

## BF16 vs FP8 (Dynamic) Comparison - Output Throughput (tok/s)

| Input Length | Metric | BF16 C=4 | FP8 C=4 | BF16 C=64 | FP8 C=64 |
|-------------|--------|----------|---------|-----------|----------|
| 3,700 (4K)  | tok/s  | 372.2 | 45.5 | 1,907.2 | 562.3 |
| 7,400 (8K)  | tok/s  | 282.2 | 44.8 | 1,207.0 | 506.4 |
| 14,800 (16K)| tok/s  | 197.0 | 43.7 | 579.7 | 390.9 |
| 29,700 (32K)| tok/s  | 122.5 | 41.0 | 250.4 | 217.1 |
| 59,500 (64K)| tok/s  | 61.6 | 36.0 | 96.7 | 112.4 |

## BF16 vs FP8 (Dynamic) Comparison - TTFT (seconds)

| Input Length | BF16 C=4 | FP8 C=4 | BF16 C=64 | FP8 C=64 |
|-------------|----------|---------|-----------|----------|
| 3,700 (4K)  | 3.21 | 0.27 | 9.83 | 2.60 |
| 7,400 (8K)  | 4.23 | 0.48 | 15.38 | 5.06 |
| 14,800 (16K)| 6.05 | 0.99 | 26.75 | 10.86 |
| 29,700 (32K)| 9.66 | 2.18 | 50.93 | 28.69 |
| 59,500 (64K)| 19.10 | 4.69 | 115.18 | 70.48 |

## Key Observations

1. **TTFT dramatically better**: FP8 TTFT is 4-12x faster than BF16 across all configurations
2. **Output throughput much worse**: FP8 throughput is 3-8x lower than BF16 at low concurrency, gap narrows at high concurrency/long context
3. **At 64K/C=64, FP8 slightly better**: 112.4 vs 96.7 tok/s - FP8 wins when BF16 is memory-constrained
4. **TPOT ~8x higher**: FP8 TPOT ~88ms vs BF16 ~11ms at 4K/C=4, indicating decode overhead
5. **No errors**: 0% error rate across all configurations
6. **This is dynamic FP8**: BF16 weights loaded (~76GB/GPU), quantized at runtime. True static FP8 checkpoint serving requires `--enable-expert-parallel` which hangs in vLLM 0.18.0
7. **Memory constraint**: BF16 weights + dynamic FP8 = ~76GB/GPU, leaving only ~4GB for KV cache per GPU (1.35M tokens total)

## Raw Data

See [minimax-m25-fp8-h100-summary.csv](minimax-m25-fp8-h100-summary.csv) for full CSV results.
