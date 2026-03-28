# MiMo-V2-Flash FP8 (Dynamic) Benchmark on H100 (p5.48xlarge)

## Test Configuration

- **Model**: MiMo-V2-Flash BF16 weights with `--quantization fp8` (dynamic FP8)
- **Hardware**: p5.48xlarge (8x NVIDIA H100 80GB SXM)
- **Software**: vLLM 0.18.0, PyTorch 2.10.0+cu126
- **Serving config**: TP=8, max_model_len=8192, gpu_memory_utilization=0.95, eager mode, dynamic FP8 quantization
- **Note**: Dynamic FP8 loads BF16 weights and quantizes at runtime. MiMo BF16 (~434GB) causes OOM on 8x H100 80GB without quantization.
- **Benchmark config**: output=300 tokens, random_range_ratio=0.1, streaming mode
- **Date**: 2026-03-28

## Output Token Throughput (tok/s)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 57.6 | 119.8 | 233.6 | 419.3 | 732.5 |

## Time To First Token - TTFT (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 0.97 | 0.44 | 0.72 | 1.23 | 2.31 |

## Time Per Output Token - TPOT (ms)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 69.2 | 66.3 | 67.7 | 74.6 | 84.1 |

## P99 Latency (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 20.8 | 20.0 | 20.5 | 22.9 | 26.2 |

## Error Rate

| Input Length | C=4 | C=8 | C=16 | C=32 | C=64 |
|-------------|------|------|-------|-------|-------|
| 3,700 (4K)  | 0/4 | 0/8 | 0/16 | 0/32 | 0/64 |

## Key Observations

1. **Peak output throughput**: 732.5 tok/s at 4K input, concurrency=64
2. **TPOT very stable**: 66-84ms across all concurrency levels, only 22% increase from C=4 to C=64
3. **TTFT excellent**: Sub-second at C=4 (0.97s), only 2.31s at C=64
4. **Linear throughput scaling**: Near-linear scaling from C=4 (57.6) to C=64 (732.5), ~12.7x for 16x concurrency increase
5. **No errors**: 0% error rate across all configurations
6. **Dynamic FP8 overhead**: Uses BF16 weights quantized at runtime, not a static FP8 checkpoint
7. **Only 4K tested**: Larger context lengths (8K/16K/32K/64K) not yet benchmarked

## Configuration Notes

- MiMo BF16 (~434GB) does not fit on 8x H100 80GB for inference (OOM even at max_model_len=2048)
- Dynamic FP8 (`--quantization fp8`) was the only way to run MiMo on this hardware
- `--enforce-eager` mode used (no CUDA graph compilation)
- `--gpu-memory-utilization 0.95` required to leave enough headroom for KV cache

## Raw Data

See [mimo-fp8-h100-summary.csv](mimo-fp8-h100-summary.csv) for full CSV results.
