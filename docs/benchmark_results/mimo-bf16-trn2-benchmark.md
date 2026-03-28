# MiMo-V2-Flash BF16 Benchmark on Trn2 (trn2.48xlarge)

## Test Configuration

- **Model**: MiMo-V2-Flash BF16 (~430B params, 128 experts, top-8 routing)
- **Hardware**: trn2.48xlarge (16x Trainium2 chips, 32 NeuronCores, 64 logical with NC=2)
- **Software**: vLLM-Neuron 0.16.0, NxDI (neuronx-distributed-inference)
- **Serving config**: TP=64, max_model_len=8192, max_num_seqs=32, batch_size=32, continuous batching
- **Neuron config**: fused_qkv=false, async_mode=true, moe_ep_degree=64, moe_tp_degree=1, attn_kernel_enabled=true
- **Buckets**: context_encoding=[4096, 8192], token_generation=[8192]
- **Benchmark config**: output=300 tokens, random_range_ratio=0.1, streaming mode
- **Date**: 2026-03-28
- **Note**: Only 4K input tested; 8K/16K/32K/64K pending

## Output Token Throughput (tok/s)

| Input Length | C=4 | C=8 | C=16 | C=32 |
|-------------|------|------|-------|-------|
| 3,700 (4K)  | 18.5 | 36.3 | 69.8 | 130.0 |

## Time To First Token - TTFT (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 |
|-------------|------|------|-------|-------|
| 3,700 (4K)  | 1.02 | 1.66 | 2.97 | 5.54 |

## Time Per Output Token - TPOT (ms)

| Input Length | C=4 | C=8 | C=16 | C=32 |
|-------------|------|------|-------|-------|
| 3,700 (4K)  | 216.0 | 220.3 | 229.1 | 246.1 |

## P99 Latency (seconds)

| Input Length | C=4 | C=8 | C=16 | C=32 |
|-------------|------|------|-------|-------|
| 3,700 (4K)  | 64.8 | 66.1 | 68.7 | 73.9 |

## Error Rate

| Input Length | C=4 | C=8 | C=16 | C=32 |
|-------------|------|------|-------|-------|
| 3,700 (4K)  | 0/4 | 0/8 | 0/16 | 0/32 |

## Trn2 BF16 vs H100 FP8 Comparison - Output Throughput (tok/s)

| Input Length | Trn2 BF16 C=4 | H100 FP8 C=4 | Ratio | Trn2 BF16 C=32 | H100 FP8 C=32 | Ratio |
|-------------|--------------|-------------|-------|----------------|--------------|-------|
| 3,700 (4K)  | 18.5 | 57.6 | **32.1%** | 130.0 | 419.3 | **31.0%** |

## Trn2 BF16 vs H100 FP8 Comparison - TTFT (seconds)

| Input Length | Trn2 BF16 C=4 | H100 FP8 C=4 | Trn2 BF16 C=32 | H100 FP8 C=32 |
|-------------|--------------|-------------|----------------|--------------|
| 3,700 (4K)  | 1.02 | 0.97 | 5.54 | 1.23 |

## Trn2 BF16 vs H100 FP8 Comparison - TPOT (ms)

| Input Length | Trn2 BF16 C=4 | H100 FP8 C=4 | Ratio | Trn2 BF16 C=32 | H100 FP8 C=32 | Ratio |
|-------------|--------------|-------------|-------|----------------|--------------|-------|
| 3,700 (4K)  | 216.0 | 69.2 | **3.1x** | 246.1 | 74.6 | **3.3x** |

## Key Observations

1. **Throughput ~31% of H100 FP8**: Trn2 BF16 achieves 130 tok/s vs H100 FP8 419 tok/s at C=32
2. **TPOT ~3x worse than H100 FP8**: 216ms vs 69ms at C=4, relatively stable across concurrency levels
3. **TTFT comparable at low concurrency**: 1.02s (Trn2) vs 0.97s (H100) at C=4, but degrades faster at high concurrency
4. **TPOT scales well**: Only 14% increase from C=4 (216ms) to C=32 (246ms), indicating good batch efficiency
5. **No errors**: 0% error rate across all configurations
6. **Note**: H100 comparison uses dynamic FP8 (not BF16) because MiMo BF16 causes OOM on 8x H100 80GB
7. **Max concurrency**: Only tested up to C=32 (batch_size=32); C=64 not tested

## Configuration Notes

- MiMo uses **Expert Parallelism** (EP=64) on Trn2, each rank handles 2 of 128 experts
- H100 uses **dynamic FP8** quantization (`--quantization fp8`), loading BF16 weights and quantizing at runtime
- MiMo BF16 (~434GB) does not fit on H100 8x80GB (640GB total, insufficient KV cache headroom)
- Fair comparison would be Trn2 BF16 vs H100 BF16, but H100 BF16 is not possible for this model size

## Raw Data

See [mimo-bf16-trn2-summary.csv](mimo-bf16-trn2-summary.csv) for full CSV results.
