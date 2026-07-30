#!/usr/bin/env python3
"""Generate configs/matrix.csv — the experiment plan, as code.

The matrix is generated rather than hand-edited so that the design rationale
lives next to the design, and so that a reviewer can see exactly how every run
was chosen. Run:

    python3 scripts/make_matrix.py --out configs/matrix.csv

Design summary (see PLAN.md §3 for the reasoning):

  Rate grid (7B)     {1,2,3,4,5,6,8,inf}. The pilot measured a sustainable
                     capacity of 3.2 req/s (ShareGPT) and 4.5 req/s (random),
                     so 1-2 sit in the linear region, 3-5 bracket both knees,
                     6-8 are overload, inf is the offline maximum.
  Repetitions        3, with seeds 1/2/3. The same seed set is reused for every
                     condition, so comparisons stay paired while the error bars
                     reflect prompt sampling and arrival jitter as well as
                     system noise (revision of decision D6).
  Anchors            A1a at the start and A1b at the end of every session:
                     detects drift within a session and bridges sessions.
  C1                 Instrumentation-overhead control: the same condition run
                     with the phase logger enabled (A1a) and disabled (C1off).
  C2                 Closed-loop sweep with a fixed concurrency limit. Unlike
                     the open-loop rate sweep, every point has a well-defined
                     steady state, so it provides the latency-throughput curve
                     that overload rates cannot.
  P0                 0.5B capacity probe; the S3 grid is confirmed or extended
                     from its result before S3 runs.

Columns consumed by scripts/run_experiments.py:
  run_id group model dataset input_len output_len tp request_rate
  max_concurrency rep num_prompts seed instr
"""

import argparse
import csv

M7 = "Qwen/Qwen2.5-7B-Instruct"
M05 = "Qwen/Qwen2.5-0.5B-Instruct"

COLS = ["run_id", "group", "model", "dataset", "input_len", "output_len",
        "tp", "request_rate", "max_concurrency", "rep", "num_prompts",
        "seed", "instr"]

SEEDS = {1: 1, 2: 2, 3: 3}          # rep -> seed
RATES_7B = ["1", "2", "3", "4", "5", "6", "8", "inf"]
RATES_05B = ["1", "2", "4", "8", "12", "16", "24", "32", "inf"]
# S2 (7B/random) saturates far above its original grid; these extend it in
# session B, with r=5 repeated from session A as the overlap point.
RATES_S2B = ["5", "10", "12", "16", "20"]
CONCURRENCY = ["1", "2", "4", "8", "16", "32", "64"]

# Closed-loop runs are self-paced, so a low concurrency limit makes each run
# long. Sample counts are reduced where the limit already bounds the run.
C2_PROMPTS = {"1": "60", "2": "60", "4": "120", "8": "120",
              "16": "200", "32": "200", "64": "200"}


def row(run_id, group, model, dataset, tp, rate, rep, *, inp="", out="",
        conc="", prompts="200", instr="on"):
    return {
        "run_id": run_id, "group": group, "model": model, "dataset": dataset,
        "input_len": inp, "output_len": out, "tp": str(tp),
        "request_rate": rate, "max_concurrency": conc, "rep": str(rep),
        "num_prompts": prompts, "seed": str(SEEDS[rep]), "instr": instr,
    }


def sweep(group, model, dataset, tp, rates, *, inp="", out="", instr="on"):
    """One rate sweep: every rate x every repetition."""
    rows = []
    for rate in rates:
        tag = "inf" if rate == "inf" else rate
        for rep in (1, 2, 3):
            rows.append(row(f"{group}_r{tag}_rep{rep}", group, model, dataset,
                            tp, rate, rep, inp=inp, out=out, instr=instr))
    return rows


def build():
    rows = []

    # ---- session A: single GPU, 7B then 0.5B on the same instance ----------
    # Order matters: the runner boots one server per consecutive
    # (model, tp, instr) block, so A1a and C1off each get their own boot and
    # the instrumentation A/B stays adjacent in time.

    # A1a - anchor + instrumented arm of the C1 control
    for rep in (1, 2, 3):
        rows.append(row(f"A1a_rep{rep}", "A1a", M7, "sharegpt", 1, "5", rep))

    # C1off - same condition with the phase logger disabled
    for rep in (1, 2, 3):
        rows.append(row(f"C1off_rep{rep}", "C1off", M7, "sharegpt", 1, "5",
                        rep, instr="off"))

    # main open-loop sweeps
    rows += sweep("S1", M7, "sharegpt", 1, RATES_7B)
    rows += sweep("S2", M7, "random", 1, RATES_7B, inp="256", out="128")

    # phase-characterisation runs: prefill-heavy vs decode-heavy at equal rate
    for rep in (1, 2, 3):
        rows.append(row(f"I1_rep{rep}", "I1", M7, "random", 1, "5", rep,
                        inp="512", out="128"))
    for rep in (1, 2, 3):
        rows.append(row(f"I2_rep{rep}", "I2", M7, "random", 1, "5", rep,
                        inp="128", out="512"))

    # A1b - anchor at the end of the 7B block (drift check vs A1a)
    for rep in (1, 2, 3):
        rows.append(row(f"A1b_rep{rep}", "A1b", M7, "sharegpt", 1, "5", rep))

    # 0.5B on the same instance: capacity probe, then the sweep
    rows.append(row("P0_probe", "P0", M05, "sharegpt", 1, "inf", 1))
    rows += sweep("S3", M05, "sharegpt", 1, RATES_05B)

    # ---- session B: closed-loop control on a single GPU --------------------
    for rep in (1, 2, 3):
        rows.append(row(f"A1c_rep{rep}", "A1c", M7, "sharegpt", 1, "5", rep))

    # S2b - S2's grid stopped at ~50 % utilisation (see D13/D28), so 7B/random
    # never reached a knee. These rates extend it. r=5 repeats a session A
    # point so the two instances can be compared directly; the group is named
    # separately because it is a different instance, not more of S2.
    rows += sweep("S2b", M7, "random", 1, RATES_S2B, inp="256", out="128")
    for c in CONCURRENCY:
        for rep in (1, 2, 3):
            rows.append(row(f"C2_c{c}_rep{rep}", "C2", M7, "sharegpt", 1,
                            "inf", rep, conc=c, prompts=C2_PROMPTS[c]))

    # ---- session C: GPU count, all tp values on one multi-GPU instance -----
    # tp=1 is re-measured here so the GPU-count comparison is within-instance.
    for rep in (1, 2, 3):
        rows.append(row(f"A1d_rep{rep}", "A1d", M7, "sharegpt", 1, "5", rep))
    rows += sweep("G1", M7, "sharegpt", 1, RATES_7B)
    rows += sweep("G2", M7, "sharegpt", 2, RATES_7B)
    rows += sweep("G4", M7, "sharegpt", 4, RATES_7B)   # only if 4 GPUs secured

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/matrix.csv")
    args = ap.parse_args()

    rows = build()
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate run_id"

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r["group"] for r in rows)
    print(f"wrote {args.out}: {len(rows)} runs")
    for g in sorted(c):
        ex = next(r for r in rows if r["group"] == g)
        rates = sorted({r["request_rate"] for r in rows if r["group"] == g},
                       key=lambda x: (x == "inf", float(x) if x != "inf" else 0))
        detail = f"tp={ex['tp']} {ex['model'].split('/')[-1]:<20} {ex['dataset']:<9}"
        if ex["max_concurrency"] or g == "C2":
            conc = sorted({r["max_concurrency"] for r in rows
                           if r["group"] == g}, key=lambda x: int(x or 0))
            detail += f" concurrency={','.join(conc)}"
        else:
            detail += f" rate={','.join(rates)}"
        if ex["instr"] == "off":
            detail += "  [instrumentation OFF]"
        print(f"  {g:<6} {c[g]:>3} runs  {detail}")


if __name__ == "__main__":
    main()
