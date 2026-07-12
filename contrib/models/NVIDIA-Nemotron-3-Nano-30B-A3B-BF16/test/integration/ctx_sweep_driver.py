#!/usr/bin/env python3
"""Sweep context lengths with the chunked NKI SSD kernel to find the ctx ceiling on trn2.3xlarge.

For each ctx in a sweep list, compile the model with USE_CHUNKED_NKI_SCAN=1, load,
generate 30 tokens from "The capital of France is", and record TTFT / decode / TPOT.
Verifies first token is " Paris". Stops on first failure (compile error or wrong output).

Environment variables:
    NEMOTRON_MODEL_PATH       -- path to HF checkpoint (default /mnt/models/nemotron-30b)
    NEMOTRON_COMPILE_ROOT     -- where to put compiled dirs (default /mnt/models)
    NEMOTRON_BENCH_SCRIPT     -- path to compile_bench script (default: ./compile_bench_231.py
                                 relative to this file's parent directory)
    NEMOTRON_SWEEP_LIST       -- comma-separated ctx list (default 512,1024,2048,4096,8192,16384)

Usage on a trn2.3xlarge instance:
    export NEMOTRON_MODEL_PATH=/mnt/models/nemotron-30b
    python ctx_sweep_driver.py
"""

import json
import os
import subprocess
import time
from pathlib import Path


COMPILE_ROOT = os.environ.get("NEMOTRON_COMPILE_ROOT", "/mnt/models")
MODEL_PATH = os.environ.get("NEMOTRON_MODEL_PATH", "/mnt/models/nemotron-30b")
DEFAULT_BENCH_SCRIPT = str(Path(__file__).resolve().parent / "compile_bench_231.py")
BENCH_SCRIPT = os.environ.get("NEMOTRON_BENCH_SCRIPT", DEFAULT_BENCH_SCRIPT)
CONTEXTS_TO_TEST = [
    int(x) for x in os.environ.get(
        "NEMOTRON_SWEEP_LIST", "512,1024,2048,4096,8192,16384"
    ).split(",")
]
RESULT_FILE = os.environ.get(
    "NEMOTRON_SWEEP_RESULTS", f"{COMPILE_ROOT}/chunked_ctx_sweep_results.jsonl"
)


def run_ctx(ctx: int, seq: int) -> dict:
    """Run the bench script for one ctx with USE_CHUNKED_NKI_SCAN=1 in the environment."""
    logfile = f"{COMPILE_ROOT}/bench_chunked_ctx{ctx}.log"
    cmd = [
        "python", BENCH_SCRIPT,
        "--ctx", str(ctx),
        "--seq", str(seq),
        "--num-gen", "30",
    ]
    print(f"\n===== ctx={ctx} seq={seq} =====", flush=True)
    print(f"cmd: {' '.join(cmd)}", flush=True)
    env = {**os.environ, "USE_CHUNKED_NKI_SCAN": "1"}
    t0 = time.perf_counter()
    with open(logfile, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=None)
    total_s = time.perf_counter() - t0

    result = {
        "ctx": ctx, "seq": seq,
        "total_wall_s": round(total_s, 1),
        "returncode": proc.returncode,
        "compile_s": None, "load_s": None,
        "ttft_ms": None, "decode_tps": None, "tpot_ms": None,
        "first_token": None, "output_preview": None,
        "error": None,
    }

    with open(logfile) as f:
        log = f.read()

    if proc.returncode != 0:
        lines = log.strip().split("\n")
        for kw in ["HBM", "OOM", "out of memory", "OutOfMemory", "RuntimeError", "ERROR", "Error"]:
            for ln in reversed(lines[-100:]):
                if kw in ln:
                    result["error"] = ln.strip()[:300]
                    break
            if result["error"]:
                break
        if result["error"] is None:
            result["error"] = lines[-1].strip()[:300] if lines else "unknown"
        print(f"FAIL: {result['error'][:200]}", flush=True)
        return result

    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("compile_s="):
            result["compile_s"] = float(line.split("=")[1])
        elif line.startswith("load_s="):
            result["load_s"] = float(line.split("=")[1])
        elif line.startswith("TTFT="):
            result["ttft_ms"] = float(line.split("=")[1].split()[0])
        elif line.startswith("decode="):
            result["decode_tps"] = float(line.split("=")[1].split()[0])
        elif line.startswith("TPOT="):
            result["tpot_ms"] = float(line.split("=")[1].split()[0])
        elif line.startswith("first_token="):
            result["first_token"] = line.split("=", 1)[1].strip("'")
        elif line.startswith("output_preview="):
            result["output_preview"] = line.split("=", 1)[1].strip("'")

    print(f"PASS: TTFT={result['ttft_ms']}ms  decode={result['decode_tps']}tok/s  first='{result['first_token']}'", flush=True)
    return result


def cleanup_compiled(ctx: int, seq: int):
    d = Path(f"{COMPILE_ROOT}/nemotron_sdk231_hg8_ctx{ctx}_seq{seq}")
    if d.exists():
        subprocess.run(["rm", "-rf", str(d)])
        print(f"  cleaned {d}", flush=True)


def main():
    results = []
    for ctx in CONTEXTS_TO_TEST:
        seq = max(2 * ctx, 2048)
        if ctx >= 4096:
            seq = ctx * 2
        try:
            r = run_ctx(ctx, seq)
        except Exception as e:
            r = {"ctx": ctx, "seq": seq, "returncode": -1, "error": str(e)[:300]}
        results.append(r)
        with open(RESULT_FILE, "a") as f:
            f.write(json.dumps(r) + "\n")

        if r.get("returncode", -1) != 0:
            print(f"CEILING FOUND at ctx={ctx}", flush=True)
            print(f"Last error: {r.get('error')}", flush=True)
            break
        if r.get("first_token") not in [" Paris", "Paris"]:
            print(f"CORRECTNESS FAIL at ctx={ctx}: first_token={r.get('first_token')}", flush=True)
            break

        cleanup_compiled(ctx, seq)

    print("\n===== SUMMARY =====", flush=True)
    print(f"{'ctx':>6}  {'seq':>6}  {'compile_s':>10}  {'ttft_ms':>10}  {'decode_tps':>12}  {'first_tok':>12}  status", flush=True)
    for r in results:
        rc = r.get("returncode", -1)
        status = "PASS" if rc == 0 else "FAIL"
        print(f"{r['ctx']:>6}  {r['seq']:>6}  {r.get('compile_s','?'):>10}  {r.get('ttft_ms','?'):>10}  {r.get('decode_tps','?'):>12}  {r.get('first_token','?'):>12}  {status}", flush=True)


if __name__ == "__main__":
    main()
