"""Control: does torch_neuronx.trace() with BF16 weights close the gap to Native?

The Native-vs-trace decomposition on inf2 showed:
    compilation path alone (FP32 vs FP32):  0.74-0.87x  (Native is SLOWER)
    precision alone (Native BF16 vs FP32):  2.35-2.61x  (the real driver)

So the headline "Native is ~2x faster" is a PRECISION result, not a compilation
result. If that is right, tracing a BF16 model should recover most of the gain
without leaving the trace path at all -- which matters because the contrib, its
tests, and the ViT-7B TP path are all built on trace.

The contrib currently traces FP32 weights + --auto-cast=matmult. This compares:
    A) FP32 weights + --auto-cast=matmult   (what the contrib ships)
    B) BF16 weights + --auto-cast=matmult
    C) BF16 weights, --auto-cast=none       (no autocast needed if already BF16)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("DINOV3_CONTRIB_SRC", "/home/ubuntu/contrib_dinov3/src"))
sys.path.insert(0, os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3"))

import numpy as np
import torch
import torch.nn.functional as F
import torch_neuronx

from modeling_dinov3 import MODEL_REGISTRY, load_dinov3_model

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")

CONFIGS = {
    "fp32_matmult": (torch.float32, ["--auto-cast", "matmult",
                                     "--auto-cast-type", "bf16",
                                     "--model-type", "transformer"]),
    "bf16_matmult": (torch.bfloat16, ["--auto-cast", "matmult",
                                      "--auto-cast-type", "bf16",
                                      "--model-type", "transformer"]),
    "bf16_none": (torch.bfloat16, ["--auto-cast", "none",
                                   "--model-type", "transformer"]),
}


def bench(m, x, iters, warmup):
    for _ in range(warmup):
        m(x)
    lat = []
    for _ in range(iters):
        t = time.time(); m(x); lat.append((time.time() - t) * 1000)
    lat = np.array(lat)
    return (float(x.shape[0] * 1000 / lat.mean()), float(np.median(lat)),
            float(np.percentile(lat, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_l,vit_b")
    ap.add_argument("--batch-sizes", default="1,8")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results_trace_bf16.json")
    args = ap.parse_args()

    out = []
    for key in [m.strip() for m in args.models.split(",")]:
        cfg = MODEL_REGISTRY[key]
        for bs in [int(b) for b in args.batch_sizes.split(",")]:
            for cname, (dtype, cargs) in CONFIGS.items():
                rec = {"model": key, "batch_size": bs, "config": cname,
                       "dtype": str(dtype).replace("torch.", "")}
                print(f"\n=== {key} BS={bs} {cname} ===")
                try:
                    torch.manual_seed(0)
                    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
                    xc = torch.randn(bs, 3, args.img_size, args.img_size)
                    with torch.no_grad():
                        ref = cpu(xc).float()

                    m = cpu.to(dtype)
                    x = xc.to(dtype)
                    t0 = time.time()
                    mn = torch_neuronx.trace(m, x, compiler_args=cargs,
                                             inline_weights_to_neff=True)
                    rec["compile_time_s"] = round(time.time() - t0, 1)

                    got = mn(x).float()
                    rec["cos_sim"] = float(F.cosine_similarity(
                        ref.flatten().unsqueeze(0), got.flatten().unsqueeze(0)).item())

                    runs = [bench(mn, x, args.iters, args.warmup)
                            for _ in range(args.repeats)]
                    best = max(runs, key=lambda r: r[0])
                    rec["runs_img_s"] = [round(r[0], 1) for r in runs]
                    rec["best_img_s"] = round(best[0], 1)
                    rec["p50_ms"] = round(best[1], 2)
                    rec["p99_ms"] = round(best[2], 2)
                    rec["status"] = "ok"
                    print(f"  {rec['best_img_s']} img/s  p50={rec['p50_ms']}ms  "
                          f"cos={rec['cos_sim']:.6f}  compile={rec['compile_time_s']}s")
                    del mn, m, cpu
                except Exception as e:
                    rec["status"] = "failed"
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    print(f"  FAILED: {rec['error'][:200]}")
                out.append(rec)
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2)

    print(f"\n{'=' * 72}\nTRACE: FP32 vs BF16 weights (inf2, 1 core)\n{'=' * 72}")
    print(f"{'model':8s} {'BS':>3s} {'config':>14s} {'img/s':>8s} {'p50 ms':>8s} {'cos':>9s}")
    print("-" * 58)
    for r in out:
        if r.get("status") != "ok":
            print(f"{r['model']:8s} {r['batch_size']:>3d} {r['config']:>14s} {'FAILED':>8s}")
            continue
        print(f"{r['model']:8s} {r['batch_size']:>3d} {r['config']:>14s} "
              f"{r['best_img_s']:>8.1f} {r['p50_ms']:>8.2f} {r['cos_sim']:>9.6f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
