"""Multi-core PyTorch Native on inf2: measure the per-instance number.

Task 018 measured PyTorch Native single-core on inf2 (ViT-S 949.2, ViT-B 433.7,
ViT-L 136.2 img/s) and only ESTIMATED per-instance as 2x single-core. This
measures it, and on inf2.24xlarge (12 cores) so the scaling curve is real rather
than a 2-point extrapolation.

Pattern: N independent worker processes, one per NeuronCore, pinned with
NEURON_RT_VISIBLE_CORES. This is the AWS-recommended serving architecture
(TorchServe uses one core per worker) and on the trace path it measured 12.1x on
12 cores vs 2.0x for DataParallel. A NeuronCore is single-tenant, so no
oversubscription.

Two things that bit the earlier Native work and are handled here:
  - PyTorch Native has ASYNC EXECUTION ON BY DEFAULT. Without
    torch.neuron.synchronize() the loop times launch dispatch, not execution
    (ViT-L BS=1 mismeasured at 1086 img/s / 0.92 ms).
  - dinov3's hubconf.py pulls in torchvision, which Beta 6 cannot import
    (its Python lacks _lzma). Import dinov3.hub.backbones directly instead.

Usage (inside the Beta 6 container):
    python bench_native_mp_inf2.py --model vit_l --cores 1,2,4,8,12 --batch-size 1
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.environ.get("DINOV3_REPO_DIR", "/mnt/models/dinov3")
sys.path.insert(0, REPO)

import numpy as np
import torch

HUB = {
    "vit_s": "dinov3_vits16",
    "vit_b": "dinov3_vitb16",
    "vit_l": "dinov3_vitl16",
    "vit_h_plus": "dinov3_vith16plus",
    "convnext_tiny": "dinov3_convnext_tiny",
    "convnext_base": "dinov3_convnext_base",
}

DEV = "privateuseone:0"


def load_model(key, seed=0):
    torch.manual_seed(seed)
    import dinov3.hub.backbones as B
    m = getattr(B, HUB[key])(pretrained=False)
    m.eval()
    return m


def worker_main(key, bs, dtype_s, img_size, iters, warmup, result_path):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_s]
    m = load_model(key).to(dtype).to(DEV)
    cm = torch.compile(m, backend="neuron", fullgraph=False, dynamic=False)
    x = torch.randn(bs, 3, img_size, img_size).to(dtype).to(DEV)

    with torch.no_grad():
        for _ in range(warmup):
            cm(x)
        torch.neuron.synchronize()

        # Barrier so all workers measure the same wall-clock window.
        start_at = float(os.environ["BENCH_START_AT"])
        while time.time() < start_at:
            time.sleep(0.005)

        lat = []
        for _ in range(iters):
            t = time.time()
            cm(x)
            torch.neuron.synchronize()
            lat.append((time.time() - t) * 1000)

        # Pipelined: one trailing sync. On an async runtime this is the honest
        # throughput figure -- per-iteration syncs serialize work the runtime
        # would otherwise overlap.
        t0 = time.time()
        for _ in range(iters):
            cm(x)
        torch.neuron.synchronize()
        wall = time.time() - t0

    lat = np.array(lat)
    with open(result_path, "w") as f:
        json.dump({
            "throughput_img_s": iters * bs / wall,
            "synced_throughput_img_s": float(bs * 1000 / lat.mean()),
            "median_latency_ms": float(np.median(lat)),
            "p99_latency_ms": float(np.percentile(lat, 99)),
        }, f)


def run_fanout(key, n_cores, bs, dtype_s, img_size, iters, warmup, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    # Native compiles are fast (~10-22s) but 12 concurrent ones on a shared host
    # still need slack before the timed window opens.
    start_at = time.time() + 150 + 20 * max(0, n_cores - 4)

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
             "--batch-size", str(bs), "--dtype", dtype_s,
             "--img-size", str(img_size), "--iters", str(iters),
             "--warmup", str(warmup), "--_result", rp],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))

    errs = []
    for p in procs:
        _, se = p.communicate()
        if p.returncode != 0:
            errs.append(se.decode()[-600:])

    per = [json.load(open(rp)) for rp in paths if os.path.exists(rp)]
    if not per:
        return {"status": "failed", "cores_requested": n_cores,
                "errors": errs[:1]}
    return {
        "status": "ok" if len(per) == n_cores else "partial",
        "cores_requested": n_cores,
        "cores_ok": len(per),
        "aggregate_img_s": round(sum(w["throughput_img_s"] for w in per), 1),
        "aggregate_synced_img_s": round(
            sum(w["synced_throughput_img_s"] for w in per), 1),
        "median_latency_ms": round(
            sum(w["median_latency_ms"] for w in per) / len(per), 2),
        "p99_latency_ms": round(max(w["p99_latency_ms"] for w in per), 2),
        "per_worker_img_s": [round(w["throughput_img_s"], 1) for w in per],
        "errors": errs[:1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_l")
    ap.add_argument("--cores", default="1,2,4,8,12")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_result", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        worker_main(args.model, args.batch_size, args.dtype, args.img_size,
                    args.iters, args.warmup, args._result)
        return

    out = args.out or f"results_native_mp_{args.model}_bs{args.batch_size}.json"
    core_list = [int(c) for c in args.cores.split(",")]
    recs = []
    for n in core_list:
        print(f"\n### {args.model} {args.dtype} BS={args.batch_size}: {n} core(s)",
              flush=True)
        r = run_fanout(args.model, n, args.batch_size, args.dtype, args.img_size,
                       args.iters, args.warmup,
                       f"/host/nmp_{args.model}_{args.batch_size}_{n}")
        r.update({"model": args.model, "batch_size": args.batch_size,
                  "dtype": args.dtype})
        recs.append(r)
        if r["status"] == "failed":
            print(f"    FAILED: {(r.get('errors') or [''])[0][:250]}", flush=True)
        else:
            print(f"    {r['aggregate_img_s']} img/s  p50={r['median_latency_ms']}ms"
                  + ("" if r["status"] == "ok" else f"  [{r['status']} "
                     f"{r['cores_ok']}/{n}]"), flush=True)
        with open(out, "w") as f:
            json.dump({"model": args.model, "results": recs}, f, indent=2)

    ok = [r for r in recs if r.get("status") == "ok"]
    print(f"\n{'=' * 62}\n{args.model} {args.dtype} BS={args.batch_size} "
          f"-- PyTorch Native multi-core scaling\n{'=' * 62}")
    print(f"{'cores':>6s} {'img/s':>10s} {'scaling':>9s} {'efficiency':>11s} {'p50 ms':>8s}")
    print("-" * 50)
    base = ok[0]["aggregate_img_s"] / ok[0]["cores_requested"] if ok else 1.0
    for r in ok:
        n = r["cores_requested"]
        sc = r["aggregate_img_s"] / base
        print(f"{n:>6d} {r['aggregate_img_s']:>10.1f} {sc:>8.2f}x "
              f"{100.0 * sc / n:>10.1f}% {r['median_latency_ms']:>8.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
