#!/usr/bin/env python3
"""Experiment runner for the vLLM benchmarking assignment.

Reads configs/matrix.csv, groups rows by (model, tensor_parallel), boots one
vLLM server per group (decision D3), runs `vllm bench serve` for every row,
and appends one line per run to results/manifest.csv (decision D4: runs are
later joined to phase logs by [start_ts, end_ts] wall-clock slicing).

Usage (from the repo root, venv active, server-capable machine):
    python3 scripts/run_experiments.py \
        --matrix configs/matrix.csv \
        --manifest results/manifest.csv \
        --session session1 \
        --results-dir results/raw/session1 \
        --sharegpt-path /workspace/ShareGPT_V3_unfiltered_cleaned_split.json

Common options:
    --only S1,S2       run only these groups (matrix column `group`)
    --runs id1,id2     run only these run_ids
    --dry-run          print the plan (boots, commands, skips) and exit
    --force            re-run rows already marked ok in the manifest
    --iteration-details  add --enable-logging-iteration-details (pilot only)

Behavior notes:
  * Manifest is append-only. A row is written when a run FINISHES; if the
    runner crashes mid-run, that run has no row and will be re-run on resume.
  * Rows already present with status=ok are skipped unless --force.
  * Server gets VLLM_PHASE_LOG_DIR (default: <results-dir>/phase_logs),
    --no-enable-prefix-caching and --no-async-scheduling (decision D8).
  * After boot, one small throwaway bench warms the server (decision D2);
    it is recorded with status=boot_warmup and excluded from analysis.
  * A resource logger (scripts/resource_logger.py) runs for the whole
    invocation, sampling GPU/CPU at 1 Hz into <results-dir>/.
  * Requested vs achieved rate is checked per run; a shortfall > 10% is
    flagged RATE_SHORTFALL in notes (client-saturation detection).
"""

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MANIFEST_COLS = [
    "run_id", "session", "start_ts", "end_ts", "model", "dataset", "tp",
    "request_rate", "rep", "command", "result_json", "status", "notes",
]

BENCH_FIXED = [
    "--num-warmups", "10",
    "--ignore-eos",
    "--percentile-metrics", "ttft,tpot,itl,e2el",
    "--metric-percentiles", "50,95,99",
    "--save-result",
]


def load_matrix(path, only_groups, only_runs):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if only_groups and row["group"] not in only_groups:
                continue
            if only_runs and row["run_id"] not in only_runs:
                continue
            rows.append(row)
    return rows


def load_done(manifest_path):
    done = set()
    p = Path(manifest_path)
    if not p.exists():
        return done
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add(row["run_id"])
    return done


def append_manifest(manifest_path, row_dict):
    p = Path(manifest_path)
    new_file = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if new_file:
            w.writeheader()
        w.writerow({k: row_dict.get(k, "") for k in MANIFEST_COLS})


def build_server_cmd(model, tp, iteration_details, host, port):
    cmd = [
        "vllm", "serve", model,
        "--host", host, "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
    ]
    if iteration_details:
        cmd.append("--enable-logging-iteration-details")
    return cmd


def build_bench_cmd(row, args, result_dir, result_filename):
    cmd = [
        "vllm", "bench", "serve",
        "--model", row["model"],
        "--host", args.host, "--port", str(args.port),
        "--dataset-name", row["dataset"],
        "--num-prompts", row["num_prompts"],
        "--request-rate", row["request_rate"],
        "--seed", row["seed"],
    ]
    if row["dataset"] == "sharegpt":
        if not args.sharegpt_path:
            raise SystemExit(f"{row['run_id']}: sharegpt needs --sharegpt-path")
        cmd += ["--dataset-path", args.sharegpt_path]
    elif row["dataset"] == "random":
        cmd += ["--random-input-len", row["input_len"],
                "--random-output-len", row["output_len"]]
    else:
        raise SystemExit(f"{row['run_id']}: unknown dataset {row['dataset']}")
    cmd += BENCH_FIXED
    cmd += ["--result-dir", str(result_dir), "--result-filename", result_filename]
    return cmd


def wait_ready(url, proc, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, f"server exited early (code {proc.returncode})"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True, f"ready in {time.time() - t0:.0f}s"
        except Exception:
            pass
        time.sleep(3)
    return False, f"timeout after {timeout_s}s"


def parse_result(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get("completed"), d.get("request_throughput"), d.get("duration")
    except Exception:
        return None, None, None


def stop_process_group(proc, name, grace_s):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.poll() is not None:
            print(f"[runner] {name} stopped gracefully")
            return
        time.sleep(1)
    print(f"[runner] {name} did not stop in {grace_s}s -> SIGKILL")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--sharegpt-path", default="")
    ap.add_argument("--only", default="", help="comma-separated groups")
    ap.add_argument("--runs", default="", help="comma-separated run_ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--iteration-details", action="store_true")
    ap.add_argument("--no-boot-warmup", action="store_true")
    ap.add_argument("--no-resource-logger", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ready-timeout", type=int, default=900)
    args = ap.parse_args()

    only_groups = set(filter(None, args.only.split(",")))
    only_runs = set(filter(None, args.runs.split(",")))
    rows = load_matrix(args.matrix, only_groups, only_runs)
    if not rows:
        raise SystemExit("no matrix rows selected")
    done = set() if args.force else load_done(args.manifest)

    results_dir = Path(args.results_dir)
    bench_dir = results_dir / "bench"
    phase_log_dir = os.environ.get(
        "VLLM_PHASE_LOG_DIR", str(results_dir / "phase_logs"))
    server_env = dict(os.environ)
    server_env["VLLM_PHASE_LOG_DIR"] = phase_log_dir
    server_env.setdefault("HF_HOME", "/workspace/hf_cache")

    # ---- plan: group consecutive rows by (model, tp) ----
    boots, cur_key, cur_rows = [], None, []
    for row in rows:
        key = (row["model"], row["tp"])
        if key != cur_key:
            if cur_rows:
                boots.append((cur_key, cur_rows))
            cur_key, cur_rows = key, []
        cur_rows.append(row)
    boots.append((cur_key, cur_rows))

    n_total = len(rows)
    n_skip = sum(1 for r in rows if r["run_id"] in done)
    print(f"[runner] session={args.session} rows={n_total} "
          f"(skip {n_skip} already ok) boots={len(boots)}")
    print(f"[runner] phase logs -> {phase_log_dir}")

    if args.dry_run:
        for (model, tp), group_rows in boots:
            print(f"\n=== BOOT {model} tp={tp} ===")
            print("  " + shlex.join(build_server_cmd(
                model, tp, args.iteration_details, args.host, args.port)))
            for row in group_rows:
                mark = "SKIP(ok)" if row["run_id"] in done else "RUN"
                cmd = build_bench_cmd(row, args, bench_dir,
                                      row["run_id"] + ".json")
                print(f"  [{mark}] {row['run_id']}: {shlex.join(cmd)}")
        return

    bench_dir.mkdir(parents=True, exist_ok=True)
    Path(phase_log_dir).mkdir(parents=True, exist_ok=True)

    # ---- resource logger for the whole invocation ----
    res_logger = None
    if not args.no_resource_logger:
        rl = Path(__file__).parent / "resource_logger.py"
        if rl.exists():
            out = results_dir / f"resources_{int(time.time())}.csv"
            res_logger = subprocess.Popen(
                [sys.executable, str(rl), "--out", str(out)],
                start_new_session=True)
            print(f"[runner] resource logger -> {out}")
        else:
            print("[runner] WARNING: resource_logger.py not found, skipping")

    failures = 0
    server = None
    server_log = None
    try:
        for bi, ((model, tp), group_rows) in enumerate(boots, 1):
            todo = [r for r in group_rows if r["run_id"] not in done]
            if not todo:
                print(f"[runner] boot {bi}: all rows done, skipping boot")
                continue

            scmd = build_server_cmd(model, tp, args.iteration_details,
                                    args.host, args.port)
            slog_path = results_dir / (
                f"server_{model.split('/')[-1]}_tp{tp}_{int(time.time())}.log")
            print(f"\n[runner] boot {bi}/{len(boots)}: {shlex.join(scmd)}")
            print(f"[runner] server log -> {slog_path}")
            server_log = open(slog_path, "w")
            server = subprocess.Popen(scmd, stdout=server_log,
                                      stderr=subprocess.STDOUT,
                                      env=server_env, start_new_session=True)
            ok, msg = wait_ready(
                f"http://{args.host}:{args.port}/health", server,
                args.ready_timeout)
            print(f"[runner] health: {msg}")
            if not ok:
                failures += len(todo)
                append_manifest(args.manifest, {
                    "run_id": f"BOOT{bi}", "session": args.session,
                    "start_ts": f"{time.time():.3f}", "end_ts": "",
                    "model": model, "tp": tp, "status": "boot_fail",
                    "notes": msg,
                })
                stop_process_group(server, "server", 30)
                server_log.close()
                continue

            # sanity: the un-instrumented step path must not be active
            time.sleep(2)
            server_log.flush()
            if "PHASE-INSTR: batch-queue" in open(slog_path).read():
                raise SystemExit(
                    "FATAL: batch-queue path active; step instrumentation "
                    "would be silent. Check --no-async-scheduling.")

            if not args.no_boot_warmup:
                wname = f"BOOT{bi}_{model.split('/')[-1]}_tp{tp}"
                wrow = {"run_id": wname, "model": model, "dataset": "random",
                        "input_len": "128", "output_len": "64", "tp": tp,
                        "request_rate": "inf", "rep": "0",
                        "num_prompts": "20", "seed": "42"}
                wcmd = build_bench_cmd(wrow, args, bench_dir, wname + ".json")
                t0 = time.time()
                subprocess.run(wcmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT)
                append_manifest(args.manifest, {
                    "run_id": wname, "session": args.session,
                    "start_ts": f"{t0:.3f}", "end_ts": f"{time.time():.3f}",
                    "model": model, "dataset": "random", "tp": tp,
                    "request_rate": "inf", "rep": "0",
                    "command": shlex.join(wcmd),
                    "status": "boot_warmup", "notes": "discard",
                })
                print(f"[runner] boot warmup done ({time.time()-t0:.0f}s)")

            for k, row in enumerate(todo, 1):
                rid = row["run_id"]
                rjson = bench_dir / (rid + ".json")
                cmd = build_bench_cmd(row, args, bench_dir, rid + ".json")
                blog = results_dir / f"bench_{rid}.log"
                print(f"[runner] [{k}/{len(todo)}] {rid} "
                      f"(rate={row['request_rate']}) ...", flush=True)
                t0 = time.time()
                with open(blog, "w") as bl:
                    rc = subprocess.run(cmd, stdout=bl,
                                        stderr=subprocess.STDOUT).returncode
                t1 = time.time()
                completed, ach, dur = parse_result(rjson)
                notes = []
                status = "ok"
                if rc != 0 or completed != int(row["num_prompts"]):
                    status = "fail"
                    failures += 1
                    notes.append(f"rc={rc},completed={completed}")
                if ach is not None:
                    notes.append(f"achieved={ach:.2f}")
                    rr = row["request_rate"]
                    if rr not in ("inf", "") and ach < 0.9 * float(rr):
                        notes.append("RATE_SHORTFALL")
                append_manifest(args.manifest, {
                    "run_id": rid, "session": args.session,
                    "start_ts": f"{t0:.3f}", "end_ts": f"{t1:.3f}",
                    "model": model, "dataset": row["dataset"], "tp": tp,
                    "request_rate": row["request_rate"], "rep": row["rep"],
                    "command": shlex.join(cmd),
                    "result_json": str(rjson), "status": status,
                    "notes": ";".join(notes),
                })
                print(f"[runner]   -> {status} in {t1-t0:.0f}s "
                      f"({';'.join(notes)})")
                time.sleep(2)

            time.sleep(3)  # let the periodic flusher write the tail
            stop_process_group(server, "server", 60)
            server_log.close()
            server, server_log = None, None
    except KeyboardInterrupt:
        print("\n[runner] interrupted by user")
        failures += 1
    finally:
        stop_process_group(server, "server", 30)
        if server_log:
            server_log.close()
        stop_process_group(res_logger, "resource logger", 10)

    print(f"\n[runner] DONE: failures={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
