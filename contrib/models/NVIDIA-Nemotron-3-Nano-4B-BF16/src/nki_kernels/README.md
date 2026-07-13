# DeltaNet-Template Pattern for Cheap-Ops-Only NKI Kernels

## Context

The Neuron compiler (SDK 2.28 through 2.31, `neuronx-cc` versions 2.20-2.26) suffers
from a Data Gather Engine (DGE) out-of-bounds runtime error when NKI kernels that
use complex indirect memory operations (scatter, dynamic indexing, Neumann-series
matmul unrolling) are compiled together with the full graph of a large MoE model.

The bug manifests at runtime, not compile time:
```
scatter/gather (indirect memory copy via vector DGE) out-of-bound access.
```

Standalone kernels pass. Full-model compilation with 128 experts and 50+ layers
triggers the bug. Documented in the consolidated ticket:
`working/Solar2/tickets/TBD-dge-oob-nki-kernel-complexity-consolidated/README.md`.

The pattern below is the ONE kernel structure that survives full-model compilation
across every SDK version tested.

## Discovered in Qwen3-Coder-Next (80B MoE, production)

Reference implementation: `working/neuronx-distributed-inference-fork/contrib/models/Qwen3-Coder-Next/src/nki_deltanet.py`.

Qwen3-Coder-Next's DeltaNet recurrent kernel is deployed at production scale (80B
MoE, 40+ layers, 128 experts) on trn2.48xlarge and does not trigger DGE OOB.
Extensive experimentation across four projects (Nemotron, Qwen3.5-35B, Solar2 250B,
MiniMax-M2 229B) shows the DeltaNet pattern is the reliable escape hatch.

## The Pattern

**Use only these NKI operations:**
- `nisa.dma_copy` -- HBM <-> SBUF transfers
- `nisa.nc_matmul` -- matmul via TensorE (partition-reducing)
- `nisa.nc_transpose` -- P<->F axis swap
- `nisa.tensor_copy` -- SBUF <-> PSUM
- `nisa.tensor_scalar` -- element-wise op with (P, 1) broadcast operand
- `nisa.tensor_tensor` -- element-wise op with same-shape operands
- `nisa.activation` -- scalar activation (exp, silu, etc.) with optional bias/scale
- `nisa.memset` -- constant fill

**Avoid:**
- `nisa.scatter` / `nisa.gather` and any DGE-based indirect access
- Dynamic indexing (`tensor[expr_involving_a_data_value]`)
- Neumann-series or multi-round matmul unrolling with mutable indices
- Cross-partition SBUF slicing (e.g., `L_local[P_MAX-1:P_MAX, :]`) -- use
  `nc_transpose` to move data to partition 0 first

**Structural template (per-token or per-chunk):**
1. Fixed-size state tile in SBUF, typically `(P_MAX=128, D)` where D is the kernel's
   free dim (head_dim, dstate, or similar 128-power).
2. Sequential Python `for t in nl.sequential_range(N)` loop over tokens or chunks.
3. Each iteration loads current inputs via `dma_copy`, computes state update via
   the cheap-op vocabulary above, and writes output via `dma_copy`.
4. Broadcasting scalars: use `nc_matmul(stationary=ones (1,P_MAX), moving=scalar (1,1))`
   to broadcast to `(P_MAX, 1)`. Then `tensor_scalar` accepts that (P_MAX, 1) as
   `operand0` with a (P_MAX, F) `dst`.
5. Cross-partition access: use `nc_transpose` to move data from partition >=1 into
   partition 0, then slice on free axis.
6. Called per (batch, head) with PyTorch outer loop -- kernel is compact, invocation
   count stays under ~500 per NEFF.

## Adaptations proven so far

| Kernel | Model | Math | Status |
|--------|-------|------|--------|
| `nki_deltanet.py` (`deltanet_recurrent_fwd`) | Qwen3-Coder-Next 80B | DeltaNet delta rule | Production |
| `nki_mamba2_ssd_recurrent.py` | Nemotron-3-Nano-30B | Mamba-2 SSD recurrent (O(S)) | Validated bit-exact vs PyTorch |
| `nki_mamba2_ssd_chunked.py` | Nemotron-3-Nano-30B | Mamba-2 SSD chunked (O(S/Q * Q^2)) | **Validated bit-exact in full model, 50/50 tokens match, 4x ctx ceiling** |

## Nemotron-specific results (SDK 2.31)

Full-model integration of `nki_mamba2_ssd_chunked.py` in Nemotron-3-Nano-30B:
- **50/50 tokens bit-identical** vs head-grouped quadratic scan at ctx=128
- **30/30 tokens bit-identical** at ctx=1024 (8 chunks -- multi-chunk state propagation correct)
- Prefill **6% faster at ctx=128, up to 41% faster at ctx=2048**
- **Max ctx on trn2.3xlarge: 8192** (up from 2048 with quadratic scan)
- Compiler ceiling at ctx=16384 is host RAM (128 GB insufficient for CE HLO),
  not device HBM

## Enabling the chunked path in Nemotron

Runtime override via environment variable:
```
USE_CHUNKED_NKI_SCAN=1 python compile_bench_231.py --ctx 4096 --seq 8192
```

Or edit the flag directly in `modeling_nemotron_h.py`:
```python
USE_CHUNKED_NKI_SCAN = os.environ.get("USE_CHUNKED_NKI_SCAN", "0") == "1"
```

## References

- Consolidated DGE OOB ticket:
  `working/Solar2/tickets/TBD-dge-oob-nki-kernel-complexity-consolidated/README.md`
- Task 016 (Nemotron blocker analysis):
  `projects/Nemotron/tasks/active/task-016-blocker-options.md`
- Reference DeltaNet kernel:
  `working/neuronx-distributed-inference-fork/contrib/models/Qwen3-Coder-Next/src/nki_deltanet.py`
- Qwen3-Coder-Next production deployment: shipping in NxDI contrib, SDK 2.30+
