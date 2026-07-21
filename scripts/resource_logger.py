#!/usr/bin/env python3
"""1 Hz GPU/CPU resource sampler (component C4).

Writes one CSV row per interval with, per GPU: SM utilization (%), memory-
controller utilization (%; the decode bandwidth-saturation signal), VRAM
used (MiB) and power draw (W); plus host CPU (total %, busiest core %),
RAM, and the CPU share of the vLLM server processes vs the bench client.

Timestamps are wall-clock epoch seconds so rows can be joined with the
phase logs (steps-*.jsonl "ts") and the manifest [start_ts, end_ts] slices.

Usage:
    python3 scripts/resource_logger.py --out results/raw/s1/resources.csv
Stop with SIGINT/SIGTERM (the runner does this automatically).
"""

import argparse
import csv
import shutil
import signal
import subprocess
import sys
import time

try:
    import psutil
except ImportError:  # degraded mode: GPU + loadavg only
    psutil = None

SMI = shutil.which("nvidia-smi")
GPU_QUERY = ("index,utilization.gpu,utilization.memory,"
             "memory.used,power.draw")

_stop = False


def _handle(sig, frame):
    global _stop
    _stop = True


def sample_gpus():
    """Return [[util, memutil, mem_mib, power_w], ...] or [] if no GPU."""
    if not SMI:
        return []
    try:
        out = subprocess.run(
            [SMI, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        rows = []
        for line in out.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            rows.append(p[1:5])  # drop index; order is stable
        return rows
    except Exception:
        return []


def classify_procs(tracked):
    """Update {pid: (Process, kind)} where kind is server|client|other."""
    if psutil is None:
        return {}
    seen = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            name = p.info["name"] or ""
        except Exception:
            continue
        if "bench" in cl and "serve" in cl:
            kind = "client"
        elif "vllm" in cl.lower() or "enginecore" in name.lower():
            kind = "server"
        else:
            continue
        pid = p.info["pid"]
        if pid in tracked:
            seen[pid] = tracked[pid]
        else:
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(None)  # prime the counter
                seen[pid] = (proc, kind)
            except Exception:
                pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    n_gpu = len(sample_gpus())
    if n_gpu == 0:
        print("[resource_logger] WARNING: no GPU visible; CPU columns only",
              file=sys.stderr)

    cols = ["ts", "cpu_total", "cpu_max_core", "ram_used_gb",
            "cpu_server_pct", "cpu_client_pct"]
    for i in range(n_gpu):
        cols += [f"gpu{i}_util", f"gpu{i}_memutil",
                 f"gpu{i}_mem_mib", f"gpu{i}_power_w"]

    tracked = {}
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        f.flush()
        if psutil:
            psutil.cpu_percent(percpu=True)  # prime
        while not _stop:
            t0 = time.time()
            if psutil:
                percpu = psutil.cpu_percent(percpu=True)
                cpu_total = sum(percpu) / max(len(percpu), 1)
                cpu_max = max(percpu) if percpu else 0.0
                ram = psutil.virtual_memory().used / 2**30
                tracked = classify_procs(tracked)
                srv = cli = 0.0
                for proc, kind in tracked.values():
                    try:
                        v = proc.cpu_percent(None)
                    except Exception:
                        continue
                    if kind == "server":
                        srv += v
                    else:
                        cli += v
            else:
                cpu_total = cpu_max = ram = srv = cli = -1.0
            row = [f"{t0:.3f}", f"{cpu_total:.1f}", f"{cpu_max:.1f}",
                   f"{ram:.2f}", f"{srv:.1f}", f"{cli:.1f}"]
            for g in sample_gpus():
                row += g
            w.writerow(row)
            f.flush()
            time.sleep(max(0.0, args.interval - (time.time() - t0)))
    print("[resource_logger] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
