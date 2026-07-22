#!/usr/bin/env python3
# Session 0 verification for phase instrumentation (stdlib only).
#
# Run ON THE POD while the vLLM server is still running (needed for /metrics):
#   python3 scripts/verify_session0.py \
#       --phase-log-dir /workspace/phase_logs \
#       --bench-json /workspace/results_s0/s0_bench.json \
#       --metrics-url http://localhost:8000/metrics
#
# Checks:
#   V1  Internal consistency of requests-*.jsonl
#       (prefill+decode==inference, n_cached==0, e2e sanity, ignore-eos)
#   V2  Cross-check vs vLLM's own Prometheus histograms (/metrics)
#   V3  Cross-check vs the bench client JSON (TTFT decomposition, TPOT)
#   V4  steps-*.jsonl sanity (token accounting, exec_s>0, kv_usage range)
#
# Exit code 0 = no FAIL. Output is a PASS/FAIL/INFO table; save it with tee.

import argparse
import glob
import gzip
import json
import os
import statistics as st
import sys
import urllib.request

RESULTS = []  # (status, name, detail)


def record(status, name, detail=""):
    """Append a result row and print it as ``[STATUS] name -> detail``."""
    RESULTS.append((status, name, detail))
    print(f"[{status:<4}] {name}" + (f"  ->  {detail}" if detail else ""))


def ok(name, cond, detail=""):
    """Record PASS if ``cond`` is true, otherwise FAIL."""
    record("PASS" if cond else "FAIL", name, detail)


def load_jsonl(patterns):
    """Load JSONL (and ``.jsonl.gz``) records matching the given glob patterns.

    Skips blank lines, meta records (``record=="meta"``), and logs WARN for
    malformed JSON. Returns ``(records, matched_file_paths)``.
    """
    recs, files = [], []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            files.append(path)
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        record("WARN", f"broken line in {os.path.basename(path)}")
                        continue
                    if j.get("record") == "meta":
                        continue
                    recs.append(j)
    return recs, files


def grab_metric(text, name):
    """Sum all samples of a Prometheus metric (handles labels/multiple engines)."""
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith(name) and len(line) > len(name) and line[len(name)] in " {":
            try:
                total += float(line.split()[-1])
                found = True
            except ValueError:
                pass
    return found, total


def main():
    """Run Session 0 phase-instrumentation checks (V1–V4) and exit on FAIL."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-log-dir", required=True)
    ap.add_argument("--bench-json", required=True)
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--num-measured", type=int, default=50,
                    help="num-prompts of the measured bench run (excl. warmups)")
    ap.add_argument("--expect-output-len", type=int, default=128)
    args = ap.parse_args()

    # ---------- load ----------
    reqs, rfiles = load_jsonl([
        os.path.join(args.phase_log_dir, "requests*.jsonl"),
        os.path.join(args.phase_log_dir, "requests*.jsonl.gz"),
    ])
    steps, sfiles = load_jsonl([
        os.path.join(args.phase_log_dir, "steps*.jsonl"),
        os.path.join(args.phase_log_dir, "steps*.jsonl.gz"),
    ])
    reqs.sort(key=lambda r: r.get("ts", 0))
    print(f"loaded: {len(reqs)} request records from {len(rfiles)} file(s), "
          f"{len(steps)} step records from {len(sfiles)} file(s)\n")

    ok("requests file exists", len(rfiles) > 0)
    ok("steps file exists", len(sfiles) > 0)
    if not reqs or not steps:
        record("FAIL", "no data to verify - aborting")
        sys.exit(1)

    n = args.num_measured
    ok(f"request records >= {n}", len(reqs) >= n,
       f"{len(reqs)} records (warmups/test request included)")
    last = reqs[-n:] if len(reqs) >= n else reqs

    # ---------- V1: internal consistency ----------
    bad = [r for r in reqs
           if abs(r["prefill_s"] + r["decode_s"] - r["inference_s"]) > 1e-6]
    ok("V1a prefill+decode == inference (all records)", not bad,
       f"{len(bad)} violations" if bad else "exact")

    cached = [r for r in reqs if r.get("n_cached", 0) != 0]
    ok("V1b n_cached == 0 (prefix caching disabled)", not cached,
       f"{len(cached)} records with cache hits!" if cached else "all zero")

    bad2 = [r for r in reqs
            if r["e2e_s"] + 0.005 < r["queued_s"] + r["inference_s"]]
    ok("V1c e2e >= queued+inference (5ms tol, cross-clock)", not bad2,
       f"{len(bad2)} violations" if bad2 else "")

    gens = sorted({r["n_gen"] for r in last})
    ok(f"V1d ignore-eos: all measured n_gen == {args.expect_output_len}",
       gens == [args.expect_output_len], f"observed n_gen values: {gens}")

    negs = [r for r in reqs if min(r["queued_s"], r["prefill_s"], r["decode_s"]) < 0]
    ok("V1e no negative durations", not negs)

    # ---------- V4: steps sanity ----------
    tot_gen = sum(s.get("n_gen_toks", 0) for s in steps)
    tot_ctx = sum(s.get("n_ctx_toks", 0) for s in steps)
    # A request schedules n_prompt ctx tokens and (n_gen - 1) gen tokens:
    # the first output token is produced by the last prefill chunk itself.
    exp_gen = sum(r["n_gen"] for r in reqs) - len(reqs)  # includes warmups
    exp_ctx = sum(r["n_prompt"] for r in reqs)

    def tiered(name, actual, expected):
        """Score actual vs expected as PASS / WARN (small shortfall) / FAIL."""
        diff, base = abs(actual - expected), max(expected, 1)
        detail = f"steps={actual} vs expected={expected}"
        if diff <= max(5, 0.01 * base):
            record("PASS", name, detail)
        elif actual < expected and diff <= 0.05 * base:
            record("WARN", name, detail +
                   " (small shortfall: likely tail-flush loss at shutdown; "
                   "acceptable for session 0)")
        else:
            record("FAIL", name, detail)

    tiered("V4a steps decode-token accounting matches requests",
           tot_gen, exp_gen)
    tiered("V4b steps prefill-token accounting matches requests",
           tot_ctx, exp_ctx)
    ok("V4c all exec_s > 0", all(s["exec_s"] > 0 for s in steps))
    kvs = [s["kv_usage"] for s in steps]
    ok("V4d kv_usage in [0,1] (or -1 sentinel)",
       all((0 <= k <= 1) or k == -1.0 for k in kvs),
       f"min={min(kvs):.3f} max={max(kvs):.3f}")

    # ---------- V2: Prometheus cross-check ----------
    try:
        with urllib.request.urlopen(args.metrics_url, timeout=10) as resp:
            mtext = resp.read().decode("utf-8", "replace")
    except Exception as e:
        record("FAIL", "V2 fetch /metrics (is the server still running?)", str(e))
        mtext = ""

    if mtext:
        for base, field in [("queue", "queued_s"),
                            ("prefill", "prefill_s"),
                            ("decode", "decode_s")]:
            mname = f"vllm:request_{base}_time_seconds"
            fs, msum = grab_metric(mtext, mname + "_sum")
            fc, mcnt = grab_metric(mtext, mname + "_count")
            if not (fs and fc) or mcnt == 0:
                record("WARN", f"V2 metric {mname} not found (name changed?)")
                continue
            prom_mean = msum / mcnt
            ours = st.mean(r[field] for r in reqs)
            close = abs(prom_mean - ours) <= max(0.005, 0.02 * ours)
            ok(f"V2 {base}: ours vs Prometheus mean",
               close, f"ours={ours*1000:.2f}ms prom={prom_mean*1000:.2f}ms "
                      f"(count {len(reqs)} vs {int(mcnt)})")

    # ---------- V3: bench client cross-check ----------
    try:
        with open(args.bench_json) as f:
            bench = json.load(f)
    except Exception as e:
        record("FAIL", "V3 load bench json", str(e))
        bench = {}

    if bench:
        c_ttft = bench.get("mean_ttft_ms")
        if c_ttft is None and isinstance(bench.get("ttfts"), list) and bench["ttfts"]:
            c_ttft = 1000 * st.mean(bench["ttfts"])
        if c_ttft is None:
            record("WARN", "V3 mean_ttft_ms not found in bench json")
        else:
            s_ttft = 1000 * st.mean(r["queued_s"] + r["prefill_s"] for r in last)
            eps = c_ttft - s_ttft
            ok("V3a client TTFT ~= server queued+prefill (+eps)",
               -2.0 <= eps <= 100.0,
               f"client={c_ttft:.2f}ms server={s_ttft:.2f}ms eps={eps:+.2f}ms")

        c_tpot = bench.get("mean_tpot_ms")
        if c_tpot is not None:
            s_tpot = 1000 * st.mean(r["mean_tpot_s"] for r in last)
            close = abs(c_tpot - s_tpot) <= max(0.5, 0.10 * s_tpot)
            ok("V3b client TPOT ~= server mean_tpot", close,
               f"client={c_tpot:.3f}ms server={s_tpot:.3f}ms")
        comp = bench.get("completed")
        if comp is not None:
            ok("V3c bench completed == num-measured", comp == n, f"completed={comp}")

    # ---------- human summary ----------
    print("\n--- summary of the measured run (last "
          f"{len(last)} requests, server-side) ---")
    for f_ in ["queued_s", "prefill_s", "decode_s", "e2e_s"]:
        vals = sorted(r[f_] for r in last)
        p95 = vals[max(0, int(round(0.95 * len(vals))) - 1)]
        print(f"  {f_:10s} mean={st.mean(vals)*1000:8.2f}ms  "
              f"p95={p95*1000:8.2f}ms")
    print(f"  steps: {len(steps)}  (mean exec="
          f"{st.mean(s['exec_s'] for s in steps)*1000:.2f}ms, "
          f"mean sched={st.mean(s['sched_s'] for s in steps)*1000:.3f}ms)")

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\n=== RESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAIL'} "
          f"({sum(1 for r in RESULTS if r[0]=='PASS')} pass, "
          f"{sum(1 for r in RESULTS if r[0]=='WARN')} warn) ===")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()