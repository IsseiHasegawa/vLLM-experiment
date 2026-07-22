#!/usr/bin/env python3
"""Offline smoke test for the analysis pipeline.

Generates a synthetic campaign that has the *exact* schema of the real
artifacts (bench JSON field names taken from a real `vllm bench serve` result,
manifest columns, phase-log records, resource CSV), then renders all nine
figures from it. No GPU, no network, nothing written inside the repo.

    python3 scripts/smoke_test_plots.py
    python3 scripts/smoke_test_plots.py --keep      # keep the temp dir
    python3 scripts/smoke_test_plots.py --pilot     # sparse data (2 rates x 1 rep)

The synthetic numbers are plausible but invented: a saturating throughput
curve, latency that grows past the knee, tp=2 with better throughput, 0.5B far
faster than 7B. The point is to exercise the code paths and to preview the
figure layout, never to stand in for measurements.

Exit code 0 means every figure that has data rendered without an exception.
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MANIFEST_COLS = ["run_id", "session", "start_ts", "end_ts", "model", "dataset",
                 "tp", "request_rate", "rep", "command", "result_json",
                 "status", "notes"]


def build(tmp: Path, pilot: bool):
    rows = list(csv.DictReader(open(REPO / "configs" / "matrix.csv")))
    if pilot:
        keep = {"S1_r5_rep1", "S1_r20_rep1", "I1_rep1", "I2_rep1"}
        rows = [r for r in rows if r["run_id"] in keep]

    base = tmp / "results" / "raw" / "sessionSYN"
    (base / "bench").mkdir(parents=True, exist_ok=True)
    (base / "phase_logs").mkdir(parents=True, exist_ok=True)
    (tmp / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "configs" / "matrix.csv", tmp / "configs" / "matrix.csv")

    random.seed(1)
    t = 1_000_000.0
    man = open(tmp / "results" / "manifest.csv", "w", newline="")
    w = csv.DictWriter(man, fieldnames=MANIFEST_COLS)
    w.writeheader()
    res = open(base / "resources_1.csv", "w", newline="")
    rw = csv.writer(res)
    rw.writerow(["ts", "cpu_total", "cpu_max_core", "ram_used_gb",
                 "cpu_server_pct", "cpu_client_pct", "gpu0_util",
                 "gpu0_memutil", "gpu0_mem_mib", "gpu0_power_w"])
    req = open(base / "phase_logs" / "requests-1.jsonl", "w")
    req.write(json.dumps({"record": "meta", "kind": "requests"}) + "\n")
    seq = 0

    for r in rows:
        rid, rate, tp = r["run_id"], r["request_rate"], int(r["tp"])
        small = "0.5B" in r["model"]
        requested = 200.0 if rate == "inf" else float(rate)
        capacity = (40.0 if small else 9.0) * tp      # saturation point
        achieved = min(requested, capacity)
        load = min(requested / capacity, 1.0)
        ttft = (8 if small else 45) * (1 + 6 * load ** 3)
        tpot = (5 if small else 22) * (1 + 1.2 * load ** 2) / (1.7 if tp == 2 else 1)
        outlen = 128 if r["output_len"] == "" else int(r["output_len"])
        inlen = 256 if r["input_len"] == "" else int(r["input_len"])
        dur = 200 / max(achieved, 0.5)
        t0, t1 = t, t + dur
        t = t1 + 5

        j = {"date": "synthetic", "model_id": r["model"], "num_prompts": 200,
             "request_rate": requested, "duration": dur, "completed": 200,
             "failed": 0, "total_input_tokens": 200 * inlen,
             "total_output_tokens": 200 * outlen,
             "request_throughput": achieved,
             "output_throughput": achieved * outlen,
             "total_token_throughput": achieved * (inlen + outlen),
             "max_output_tokens_per_s": achieved * outlen * 1.4,
             "max_concurrent_requests": int(2 + 40 * load)}
        for m, v in (("ttft", ttft), ("tpot", tpot), ("itl", tpot),
                     ("e2el", ttft + tpot * outlen)):
            n = random.gauss(1, 0.03)
            j[f"mean_{m}_ms"] = v * n
            j[f"median_{m}_ms"] = v * n * 0.97
            j[f"std_{m}_ms"] = v * 0.1
            j[f"p50_{m}_ms"] = v * n * 0.97
            j[f"p95_{m}_ms"] = v * n * 1.35
            j[f"p99_{m}_ms"] = v * n * 1.8
        json.dump(j, open(base / "bench" / f"{rid}.json", "w"))

        w.writerow({"run_id": rid, "session": "syn", "start_ts": f"{t0:.3f}",
                    "end_ts": f"{t1:.3f}", "model": r["model"],
                    "dataset": r["dataset"], "tp": tp, "request_rate": rate,
                    "rep": r["rep"], "command": "synthetic",
                    "result_json": f"{rid}.json", "status": "ok",
                    "notes": f"achieved={achieved:.2f}"})

        for k in range(int(dur) + 1):
            rw.writerow([f"{t0 + k:.3f}", f"{20 + 60 * load:.1f}",
                         f"{40 + 55 * load:.1f}", "12.0",
                         f"{300 * load:.1f}", f"{90 * load:.1f}",
                         int(30 + 65 * load), int(45 + 50 * load),
                         int(15000 + 20000 * load), int(90 + 180 * load)])

        if r["group"] in ("I1", "I2"):
            for i in range(200):
                seq += 1
                pre = (inlen / 512) * 0.09 * (1 + load)
                dec = outlen * tpot / 1000
                q = 0.002 * load
                req.write(json.dumps({
                    "ts": t0 + dur * i / 200, "seq": seq, "pid": 1,
                    "req_id": f"{rid}-{i}", "n_prompt": inlen, "n_gen": outlen,
                    "n_cached": 0, "queued_s": q, "prefill_s": pre,
                    "decode_s": dec, "inference_s": pre + dec,
                    "e2e_s": q + pre + dec + 0.019,
                    "mean_tpot_s": dec / max(outlen - 1, 1),
                    "finish": "length"}) + "\n")

    man.close()
    res.close()
    req.close()

    # figure 5 input
    stats = {
        "sharegpt": {
            "input_lens": [max(8, int(random.lognormvariate(4.8, 0.9)))
                           for _ in range(3000)],
            "output_lens": [max(8, int(random.lognormvariate(5.0, 0.8)))
                            for _ in range(3000)],
            "summary": {"input_p50": 120, "input_p95": 520,
                        "output_p50": 150, "output_p95": 560}},
        "random": {
            "input_lens": [256] * 200, "output_lens": [128] * 200,
            "summary": {"input_p50": 256, "input_p95": 256,
                        "output_p50": 128, "output_p95": 128}},
    }
    json.dump(stats, open(tmp / "results" / "dataset_stats.json", "w"))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="vllm-smoke-"))
    try:
        n = build(tmp, args.pilot)
        print(f"synthetic campaign: {n} runs in {tmp}\n")
        out = tmp / "figures"
        rc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "plots" / "make_figures.py"),
             "--repo", str(tmp), "--outdir", str(out)]).returncode
        pngs = sorted(out.glob("*.png"))
        print(f"\n{len(pngs)} figures rendered:")
        for p in pngs:
            print(f"  {p}")
        if rc == 0 and len(pngs) == 9:
            print("\nSMOKE TEST: PASS (all 9 figures)")
        else:
            print(f"\nSMOKE TEST: CHECK (rc={rc}, {len(pngs)}/9 figures)")
        if args.keep:
            print(f"\nkept: {tmp}")
            if sys.platform == "darwin":
                subprocess.run(["open", str(out)])
        return 0 if rc == 0 else 1
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
