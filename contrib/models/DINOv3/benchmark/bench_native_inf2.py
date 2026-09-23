"""Task 018: PyTorch Native (torch.compile backend='neuron') DINOv3 on inf2.

Closes the one untested axis in the DINOv3 inference matrix. Prior work:
  - trn2 + PyTorch Native: measured (dinov3-native project)
  - inf2 + torch_neuronx.trace(): measured (Task 021)
  - inf2 + PyTorch Native: NEVER MEASURED  <- this script

Why it matters: the dinov3-native project reported Native beating trace by up to
4.43x on trn2, but that comparison used a trace baseline measured with the
`torch_neuronx.DataParallel` num_workers=2 defect (Task 021 showed that costs
5.18x). Single-core apples-to-apples the gap was ~2.4x, and that project itself
attributed most of it to FP32->BF16 plus torch.compile fusion rather than the
compilation path. So the honest trace-vs-native delta on inf2 is unknown.

Comparison target (Task 021, inf2.xlarge, torch_neuronx.trace, FP32+autocast):
    ViT-L  1-core BS=1  ~68-71 img/s (converged), BS=8 ~70 img/s
    ViT-B  1-core       ~211 img/s
    ViT-S  1-core       ~416 img/s

Runs eager first (sanity + does Native even work on gen2), then compiled, then
accuracy vs CPU FP32, then a batch sweep.

Usage:
    python bench_native_inf2.py --models vit_l --dtype bf16 --batch-sizes 1,4,8
    python bench_native_inf2.py --models vit_s,vit_b,vit_l --skip-eager
"""

import argparse
import json
import os
import sys
import time
import traceback

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn.functional as F

HUB = {
    "vit_s": "dinov3_vits16",
    "vit_b": "dinov3_vitb16",
    "vit_l": "dinov3_vitl16",
    "vit_h_plus": "dinov3_vith16plus",
    "convnext_tiny": "dinov3_convnext_tiny",
    "convnext_base": "dinov3_convnext_base",
}


def load_model(key, seed=0):
    """Build the backbone WITHOUT going through torch.hub / hubconf.py.

    hubconf.py imports the detection-eval path, which pulls in torchvision, and
    the Beta 6 container's Python is built without _lzma so `import torchvision`
    dies with ModuleNotFoundError: No module named '_lzma'. The backbone factory
    in dinov3.hub.backbones needs neither torchvision nor transformers, so import
    it directly. Same constructor torch.hub would have called.
    """
    # Seed before construction: pretrained=False means RANDOM init, so two
    # separate load_model() calls give DIFFERENT weights. Comparing a CPU
    # reference built from one call against a device model from another produced
    # a spurious cos_sim of -0.05 that looked like a framework accuracy failure
    # but was purely this harness bug (the standalone diagnostic, which seeded
    # both, measured cos_sim 0.999996 for the same config).
    torch.manual_seed(seed)
    import dinov3.hub.backbones as B
    fn = getattr(B, HUB[key])
    m = fn(pretrained=False)
    m.eval()
    return m


def _sync():
    """Block until the device has finished.

    PyTorch Native has ASYNCHRONOUS EXECUTION ENABLED BY DEFAULT
    (steering/pytorch-native.md). Without an explicit sync the timing loop
    measures NEFF launch dispatch, not execution -- which produced an absurd
    1086 img/s @ 0.92 ms for ViT-L BS=1 (vs ~70 img/s on the trace path) and a
    garbage cos_sim of -0.0013, because the result was read before it existed.
    """
    torch.neuron.synchronize()


def bench(m, x, iters, warmup, label):
    for _ in range(warmup):
        m(x)
    _sync()
    lat = []
    for _ in range(iters):
        t = time.time()
        m(x)
        _sync()                      # per-iteration: measures real latency
        lat.append((time.time() - t) * 1000)
    # Also time the whole loop with a single trailing sync. For an async runtime
    # this is the honest THROUGHPUT number (per-iteration syncs serialize work
    # the runtime would otherwise pipeline).
    t0 = time.time()
    for _ in range(iters):
        m(x)
    _sync()
    pipelined_wall = time.time() - t0
    lat = np.array(lat)
    bs = x.shape[0]
    r = {
        "throughput_img_s": float(iters * bs / pipelined_wall),
        "synced_throughput_img_s": float(bs * 1000 / lat.mean()),
        "median_latency_ms": float(np.median(lat)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
    }
    print(f"    {label:24s} {r['throughput_img_s']:>8.1f} img/s (pipelined)  "
          f"{r['synced_throughput_img_s']:>7.1f} (synced)  "
          f"p50={r['median_latency_ms']:>7.2f}ms  p99={r['p99_latency_ms']:>7.2f}ms")
    return r


def run(key, bs, dtype_s, img_size, iters, warmup, repeats, skip_eager):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_s]
    rec = {"model": key, "batch_size": bs, "dtype": dtype_s, "img_size": img_size}
    print(f"\n{'=' * 70}\n{key}  BS={bs}  {dtype_s}\n{'=' * 70}")

    try:
        # --- CPU reference (always FP32 for a trustworthy baseline) ---
        cpu = load_model(key)
        xc = torch.randn(bs, 3, img_size, img_size)
        with torch.no_grad():
            ref = cpu(xc).float()

        dev = torch.device("privateuseone:0")
        # Same seed AND an explicit state_dict copy, so the comparison is
        # guaranteed weight-identical regardless of init-order effects.
        m = load_model(key)
        m.load_state_dict(cpu.state_dict())
        m = m.to(dtype).to(dev)
        x = xc.to(dtype).to(dev)

        # --- eager (sanity: does Native run at all on gen2?) ---
        if not skip_eager:
            with torch.no_grad():
                rec["eager"] = bench(m, x, max(5, iters // 10), 2, "eager")

        # --- compiled ---
        t0 = time.time()
        cm = torch.compile(m, backend="neuron", fullgraph=False, dynamic=False)
        with torch.no_grad():
            cm(x)  # triggers compile
        rec["compile_time_s"] = round(time.time() - t0, 1)
        print(f"    compile: {rec['compile_time_s']}s")

        with torch.no_grad():
            runs = [bench(cm, x, iters, warmup, f"compiled run{i + 1}")
                    for i in range(repeats)]
        rec["compiled_runs"] = [round(r["throughput_img_s"], 1) for r in runs]
        # Best-of-N: per steering/nki.md, compare converged numbers.
        best = max(runs, key=lambda r: r["throughput_img_s"])
        rec["compiled_best_img_s"] = round(best["throughput_img_s"], 1)
        rec["compiled_p50_ms"] = round(best["median_latency_ms"], 2)
        rec["compiled_p99_ms"] = round(best["p99_latency_ms"], 2)

        # --- accuracy vs CPU FP32 ---
        with torch.no_grad():
            got = cm(x)
            _sync()
            got = got.float().cpu()
        rec["cos_sim"] = float(F.cosine_similarity(
            ref.flatten().unsqueeze(0), got.flatten().unsqueeze(0)).item())
        rec["max_abs_diff"] = float((ref - got).abs().max().item())
        print(f"    cos_sim={rec['cos_sim']:.6f}  max_abs_diff={rec['max_abs_diff']:.2e}")

        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        print(f"    FAILED: {rec['error'][:300]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_l")
    ap.add_argument("--batch-sizes", default="1,4,8")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-eager", action="store_true")
    ap.add_argument("--out", default="results_native_inf2.json")
    args = ap.parse_args()

    env = {"dtype": args.dtype, "img_size": args.img_size,
           "torch": torch.__version__}
    try:
        import torch_neuronx
        env["torch_neuronx"] = getattr(torch_neuronx, "__version__", "?")
    except Exception as e:
        env["torch_neuronx"] = f"import failed: {e}"
    print(json.dumps(env, indent=2))

    out = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        for bs in [int(b) for b in args.batch_sizes.split(",")]:
            out.append(run(key, bs, args.dtype, args.img_size, args.iters,
                           args.warmup, args.repeats, args.skip_eager))
            with open(args.out, "w") as f:
                json.dump({"env": env, "results": out}, f, indent=2)

    print(f"\n{'=' * 74}\nPyTorch Native on inf2 -- SUMMARY ({args.dtype})\n{'=' * 74}")
    print(f"{'model':14s} {'BS':>3s} {'eager':>8s} {'compiled':>9s} {'p50 ms':>8s} "
          f"{'cos':>9s} {'compile':>8s}")
    print("-" * 68)
    for r in out:
        if r.get("status") != "ok":
            print(f"{r['model']:14s} {r['batch_size']:>3d} {'FAILED':>8s}")
            continue
        eg = r.get("eager", {}).get("throughput_img_s")
        print(f"{r['model']:14s} {r['batch_size']:>3d} "
              f"{(f'{eg:.1f}' if eg else '-'):>8s} "
              f"{r['compiled_best_img_s']:>9.1f} {r['compiled_p50_ms']:>8.2f} "
              f"{r['cos_sim']:>9.6f} {r['compile_time_s']:>7.1f}s")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
