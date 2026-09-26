"""Precision-matched trace-vs-Native comparison on inf2.xlarge (2 cores).

Closes the last methodological hole. The published ratios compare
`trace FP32 + --auto-cast=matmult` against `native BF16`, which is NOT
precision-matched. On ViT-L single-core the matched BF16-vs-BF16 ratio is 1.70x,
not the published 1.90x -- and at FP32 the direction REVERSES (trace 71.8 vs
native 52.1 = 0.73x, trace wins).

Worse, trace-BF16 was only ever measured on ViT-L and ViT-B. ViT-S and ViT-H+
have no matched comparison at all.

This is the trace side: all four ViTs, FP32 and BF16 weights, batch swept, using
independent worker processes on both cores so it is directly comparable to the
Native multi-core runs.

Three configs per model:
  fp32_matmult -- what the contrib ships (FP32 weights + --auto-cast=matmult)
  bf16_matmult -- BF16 weights, autocast still on
  bf16_none    -- BF16 weights, no autocast (autocast is a no-op once BF16)

Usage:
    python bench_trace_matched.py --models vit_s,vit_b,vit_l,vit_h_plus \
        --cores 2 --batch-sizes 1,4,8
"""

import argparse
import json
import os
import subprocess
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

# --model-type=transformer for ViT; the ConvNeXt variant is not needed here.
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


def build(key, cfg_name, bs, img_size):
    dtype, cargs = CONFIGS[cfg_name]
    cfg = MODEL_REGISTRY[key]
    torch.manual_seed(0)
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    m = cpu.to(dtype)
    x = torch.randn(bs, 3, img_size, img_size).to(dtype)
    mn = torch_neuronx.trace(m, x, compiler_args=cargs,
                             inline_weights_to_neff=True)
    return mn, x


def worker_main(key, cfg_name, bs, img_size, iters, warmup, result_path):
    mn, x = build(key, cfg_name, bs, img_size)
    for _ in range(warmup):
        mn(x)
    start_at = float(os.environ["BENCH_START_AT"])
    while time.time() < start_at:
        time.sleep(0.005)
    lat = []
    t0 = time.time()
    for _ in range(iters):
        t = time.time(); mn(x); lat.append((time.time() - t) * 1000)
    wall = time.time() - t0
    lat = np.array(lat)
    with open(result_path, "w") as f:
        json.dump({"throughput_img_s": iters * bs / wall,
                   "median_latency_ms": float(np.median(lat)),
                   "p99_latency_ms": float(np.percentile(lat, 99))}, f)


def accuracy_subproc(key, cfg_name, img_size):
    """Run the accuracy check in a SUBPROCESS so the core is definitely freed.

    A loaded traced model pins its NeuronCore for the life of the process, and
    gc.collect() is not a reliable release. Doing this in-process left the core
    busy and every subsequent worker died with "The PyTorch Neuron Runtime could
    not be initialized". Process exit is the only guaranteed release.
    """
    rp = f"/tmp/acc_{key}_{cfg_name}.json"
    env = dict(os.environ)
    env["NEURON_RT_VISIBLE_CORES"] = "0"
    env["NEURON_RT_NUM_CORES"] = "1"
    p = subprocess.run(
        [sys.executable, __file__, "--_accuracy", "--model", key,
         "--config", cfg_name, "--img-size", str(img_size), "--_result", rp],
        env=env, capture_output=True)
    time.sleep(15)   # ditto -- let the runtime fully reclaim the core
    if os.path.exists(rp):
        with open(rp) as f:
            return json.load(f).get("cos_sim")
    return None


def accuracy(key, cfg_name, bs, img_size):
    """cos_sim vs a CPU FP32 reference built from identical (seeded) weights."""
    cfg = MODEL_REGISTRY[key]
    torch.manual_seed(0)
    cpu = load_dinov3_model(cfg["hub_name"], repo_dir=REPO)
    xc = torch.randn(bs, 3, img_size, img_size)
    with torch.no_grad():
        ref = cpu(xc).float()
    del cpu
    mn, _ = build(key, cfg_name, bs, img_size)
    dtype = CONFIGS[cfg_name][0]
    got = mn(xc.to(dtype)).float()
    del mn
    # A loaded traced model holds its NeuronCore. Release it and give the runtime
    # a moment before any worker fan-out, or the workers fail with
    # "The PyTorch Neuron Runtime could not be initialized".
    import gc
    gc.collect()
    time.sleep(5)
    return float(F.cosine_similarity(ref.flatten().unsqueeze(0),
                                     got.flatten().unsqueeze(0)).item())


def run_cfg(key, cfg_name, n_cores, bs, img_size, iters, warmup, tmpdir,
            _retry=False):
    os.makedirs(tmpdir, exist_ok=True)
    # Precompile once, IN A SUBPROCESS: N concurrent neuronx-cc runs contend for
    # the runtime, and a model loaded in THIS process would pin a core for the
    # whole fan-out. Process exit is the only reliable core release.
    print("      precompiling ...", end=" ", flush=True)
    t0 = time.time()
    subprocess.run(
        [sys.executable, __file__, "--_precompile", "--model", key,
         "--config", cfg_name, "--batch-size", str(bs),
         "--img-size", str(img_size)],
        env={**os.environ, "NEURON_RT_VISIBLE_CORES": "0",
             "NEURON_RT_NUM_CORES": "1"},
        capture_output=True)
    ct = time.time() - t0
    time.sleep(20)   # core release is not instantaneous after process exit
    print(f"{ct:.0f}s", flush=True)

    start_at = time.time() + 60 + (120 if key == "vit_h_plus" else 0)
    procs, paths = [], []
    for c in range(n_cores):
        rp = os.path.join(tmpdir, f"w{c}.json")
        if os.path.exists(rp):
            os.unlink(rp)
        paths.append(rp)
        env = dict(os.environ)
        env["NEURON_RT_VISIBLE_CORES"] = str(c)
        env["NEURON_RT_NUM_CORES"] = "1"
        env["BENCH_START_AT"] = str(start_at)
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--_worker", "--model", key,
             "--config", cfg_name, "--batch-size", str(bs),
             "--img-size", str(img_size), "--iters", str(iters),
             "--warmup", str(warmup), "--_result", rp],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errs = []
    for p in procs:
        _, se = p.communicate()
        if p.returncode != 0:
            errs.append(se.decode()[-500:])
    per = [json.load(open(rp)) for rp in paths if os.path.exists(rp)]
    if not per and not _retry:
        # Core-release races produce a transient "Runtime could not be
        # initialized". Wait longer and try once more before recording a failure.
        print("      retrying after 45s ...", end=" ", flush=True)
        time.sleep(45)
        return run_cfg(key, cfg_name, n_cores, bs, img_size, iters, warmup,
                       tmpdir, _retry=True)
    if not per:
        return {"status": "failed", "errors": errs[:1], "compile_time_s": round(ct, 1)}
    return {"status": "ok" if len(per) == n_cores else "partial",
            "cores_ok": len(per), "compile_time_s": round(ct, 1),
            "aggregate_img_s": round(sum(w["throughput_img_s"] for w in per), 1),
            "median_latency_ms": round(
                sum(w["median_latency_ms"] for w in per) / len(per), 2),
            "p99_latency_ms": round(max(w["p99_latency_ms"] for w in per), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vit_s,vit_b,vit_l,vit_h_plus")
    ap.add_argument("--configs", default="fp32_matmult,bf16_matmult,bf16_none")
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--batch-sizes", default="1,4,8")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results_trace_matched.json")
    ap.add_argument("--compile-only", action="store_true",
                    help="populate the NEFF cache, skip all timing")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_accuracy", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_precompile", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--_result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._precompile:
        build(args.model, args.config, args.batch_size, args.img_size)
        return

    if args._accuracy:
        cos = accuracy(args.model, args.config, 1, args.img_size)
        with open(args._result, "w") as f:
            json.dump({"cos_sim": cos}, f)
        return

    if args._worker:
        worker_main(args.model, args.config, args.batch_size, args.img_size,
                    args.iters, args.warmup, args._result)
        return

    models = [m.strip() for m in args.models.split(",")]
    cfgs = [c.strip() for c in args.configs.split(",")]
    bss = [int(b) for b in args.batch_sizes.split(",")]

    if args.compile_only:
        # Phase 1: fill the on-disk NEFF cache so the timed phase never runs
        # neuronx-cc concurrently with the benchmark (memory pressure on a 15 GB
        # host corrupts the measurements).
        for key in models:
            for cname in cfgs:
                for bs in bss:
                    print(f"compiling {key} {cname} BS={bs} ...", flush=True)
                    subprocess.run(
                        [sys.executable, __file__, "--_precompile", "--model", key,
                         "--config", cname, "--batch-size", str(bs),
                         "--img-size", str(args.img_size)],
                        env={**os.environ, "NEURON_RT_VISIBLE_CORES": "0",
                             "NEURON_RT_NUM_CORES": "1"},
                        capture_output=True)
                    time.sleep(5)
        print("compile-only phase done")
        return

    out = []
    for key in models:
        for cname in cfgs:
            # Accuracy once per (model, config) -- independent of batch size.
            try:
                cos = accuracy_subproc(key, cname, args.img_size)
            except Exception as e:
                cos = None
                print(f"    accuracy failed: {type(e).__name__}: {e}")
            for bs in bss:
                print(f"\n### {key} {cname} BS={bs} on {args.cores} core(s)", flush=True)
                r = run_cfg(key, cname, args.cores, bs, args.img_size,
                            args.iters, args.warmup,
                            f"/home/ubuntu/tm_{key}_{cname}_{bs}")
                r.update({"model": key, "config": cname, "batch_size": bs,
                          "cores": args.cores, "cos_sim": cos})
                out.append(r)
                print(f"      {r.get('aggregate_img_s', 'FAILED')} img/s  "
                      f"p50={r.get('median_latency_ms')}ms  cos={cos}", flush=True)
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=2)

    print(f"\n{'=' * 74}\ntrace on inf2.xlarge ({args.cores} cores) -- best per model/config"
          f"\n{'=' * 74}")
    print(f"{'model':10s} {'config':14s} {'best img/s':>11s} {'BS':>3s} {'cos_sim':>10s}")
    print("-" * 54)
    for key in models:
        for cname in cfgs:
            rs = [r for r in out if r["model"] == key and r["config"] == cname
                  and r.get("status") == "ok"]
            if not rs:
                print(f"{key:10s} {cname:14s} {'FAILED':>11s}")
                continue
            b = max(rs, key=lambda r: r["aggregate_img_s"])
            cs = b.get("cos_sim")
            print(f"{key:10s} {cname:14s} {b['aggregate_img_s']:>11.1f} "
                  f"{b['batch_size']:>3d} "
                  f"{(f'{cs:.6f}' if cs else '-'):>10s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
