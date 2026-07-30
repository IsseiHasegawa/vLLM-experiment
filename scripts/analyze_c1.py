#!/usr/bin/env python3
"""C1 control: does the instrumentation perturb what it measures?

Compares the same serving condition run with the phase logger enabled (group
A1a) and disabled (group C1off). The two arms are adjacent in time and use the
same seeds, so a difference beyond noise would indicate that logging changes
the system under test.

    python3 scripts/analyze_c1.py --repo . [--on A1a --off C1off]

Output is a table suitable for the Methods section, plus a verdict. The test
is Welch's t (unequal variances, no normality assumption beyond the CLT) with
n=3 per arm; with so few samples the effect size matters more than the p-value,
so both are reported and the verdict keys off a practical-equivalence bound.

Interpretation: with only the environment variable removed, the disabled arm
still executes the patched code, so this measures the cost of *logging*
(record construction, buffering, disk writes, background flush). The residual
cost of the patch when disabled is one boolean test per request and per step.
"""

import argparse
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "plots"))
import common  # noqa: E402

# Metrics that would move first if logging stole time from the serving loop.
METRICS = [
    ("p50_ttft_ms", "TTFT p50", "ms"),
    ("p95_ttft_ms", "TTFT p95", "ms"),
    ("p50_tpot_ms", "TPOT p50", "ms"),
    ("p95_tpot_ms", "TPOT p95", "ms"),
    ("request_throughput", "Request throughput", "req/s"),
    ("output_throughput", "Output throughput", "tok/s"),
    ("duration", "Run duration", "s"),
]

# Practical-equivalence bound: a relative difference below this is treated as
# no effect regardless of the p-value, because it is far below run-to-run
# variation of the serving system itself.
EQUIV_PCT = 2.0

# Metrics where a larger number is better, so a negative delta means logging
# made things worse. Used to report a consistent direction across metrics.
HIGHER_IS_BETTER = {"request_throughput", "output_throughput"}


def paired(a, b):
    """Paired t statistic and two-sided p on the per-seed differences.

    The two arms use the same seeds (D14), so the workload is identical within
    a pair and the difference isolates logging. This is far sharper than the
    unpaired test: at rate 5 the seed-to-seed spread of throughput is ~10 %
    while the paired difference is a few tenths of a percent.
    """
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    d = [x - y for x, y in zip(a, b)]
    sd = st.stdev(d)
    if sd == 0:
        return 0.0, 1.0
    t = st.mean(d) / (sd / math.sqrt(len(d)))
    return t, math.erfc(abs(t) / math.sqrt(2))


def welch(a, b):
    """Welch's t statistic and two-sided p, via a normal approximation."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 0.0, 1.0
    t = (st.mean(a) - st.mean(b)) / se
    # Normal approximation to the two-sided p-value (df is small, so this is
    # anti-conservative; reported only as a rough guide alongside effect size).
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--on", default="A1a", help="group with logging enabled")
    ap.add_argument("--off", default="C1off", help="group with logging off")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = common.load_runs(repo=args.repo)
    on = common.select(runs, group=args.on)
    off = common.select(runs, group=args.off)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("C1 control - instrumentation overhead")
    emit(f"  enabled  ({args.on}):   n={len(on)}")
    emit(f"  disabled ({args.off}): n={len(off)}")
    if not on or not off:
        emit("\nFAIL: both arms are required; run groups "
             f"{args.on} and {args.off} first.")
        return 1
    emit()

    hdr = (f"{'metric':<22}{'logging on':>14}{'logging off':>14}"
           f"{'delta':>12}{'t':>8}{'p':>8}")
    emit(hdr)
    emit("-" * len(hdr))

    worst = 0.0
    flagged = []
    for key, label, unit in METRICS:
        a = [r["bench"][key] for r in on if key in r["bench"]]
        b = [r["bench"][key] for r in off if key in r["bench"]]
        if not a or not b:
            continue
        ma, mb = st.mean(a), st.mean(b)
        sa = st.stdev(a) if len(a) > 1 else 0.0
        sb = st.stdev(b) if len(b) > 1 else 0.0
        rel = 100.0 * (ma - mb) / mb if mb else float("nan")
        t, p = welch(a, b)
        emit(f"{label:<22}{ma:>9.2f}±{sa:<4.2f}{mb:>9.2f}±{sb:<4.2f}"
             f"{rel:>11.2f}%{t:>8.2f}{p:>8.3f}")
        if abs(rel) > abs(worst):
            worst = rel
        # Flag only when the difference is both statistically and practically
        # notable; either alone is uninformative at n=3.
        if abs(rel) > EQUIV_PCT and p < 0.05:
            flagged.append((label, rel, p))

    emit()
    emit(f"largest unpaired difference: {worst:+.2f}% "
         f"(practical-equivalence bound ±{EQUIV_PCT:.1f}%)")

    # ---- paired by seed: the comparison the design actually supports -------
    on_by_seed = {r.get("seed"): r["bench"] for r in on}
    off_by_seed = {r.get("seed"): r["bench"] for r in off}
    seeds = sorted(s for s in on_by_seed if s in off_by_seed)
    worst_paired = 0.0
    signs = []
    paired_flags = []
    if len(seeds) >= 2:
        emit()
        phdr = (f"{'metric (paired by seed)':<24}{'mean delta':>12}"
                f"{'per-seed range':>22}{'t':>8}{'p':>8}")
        emit(phdr)
        emit("-" * len(phdr))
        for key, label, unit in METRICS:
            a = [on_by_seed[s][key] for s in seeds if key in on_by_seed[s]]
            b = [off_by_seed[s][key] for s in seeds if key in off_by_seed[s]]
            if len(a) != len(b) or not a:
                continue
            rels = [100.0 * (x - y) / y for x, y in zip(a, b) if y]
            if not rels:
                continue
            mr = st.mean(rels)
            t, p = paired(a, b)
            emit(f"{label:<24}{mr:>11.2f}%"
                 f"{f'{min(rels):+.2f}% .. {max(rels):+.2f}%':>22}"
                 f"{t:>8.2f}{p:>8.3f}")
            if abs(mr) > abs(worst_paired):
                worst_paired = mr
            # Normalise so that "positive" always means "logging made it
            # worse", i.e. flip the throughput metrics.
            signs.append(1 if mr * (-1 if key in HIGHER_IS_BETTER else 1) > 0
                         else -1)
            if abs(mr) > EQUIV_PCT and p < 0.05:
                paired_flags.append((label, mr, p))
        emit()
        emit(f"largest paired difference:   {worst_paired:+.2f}%  "
             f"(n={len(seeds)} seed pairs)")
        pos = sum(1 for s in signs if s > 0)
        emit(f"direction agreement: {max(pos, len(signs) - pos)}/{len(signs)} "
             "metrics move in the same direction (logging slower / lower "
             "throughput). A consistent direction is evidence of a real "
             "effect even where each metric alone is inside the bound.")

    emit()
    # The paired test is the sharper one and is preferred when the arms pair
    # up by seed; the unpaired numbers stay in the output for completeness.
    use, src = (paired_flags, "paired") if len(seeds) >= 2 else (flagged, "unpaired")
    if use:
        emit(f"VERDICT: instrumentation effect detected ({src}) on "
             + ", ".join(f"{l} ({r:+.2f}%, p={p:.3f})" for l, r, p in use))
        emit("  -> report the effect; treat instrumented latencies as an "
             "upper bound and instrumented throughput as a lower bound.")
        rc = 1
    else:
        emit(f"VERDICT: no instrumentation effect above ±{EQUIV_PCT:.1f}% "
             f"({src} test).")
        rc = 0
    if len(seeds) >= 2:
        emit(f"  Bound for the report: every metric within "
             f"{abs(worst_paired):.2f}% (paired, n={len(seeds)}).")
        emit("  With n=3 the unpaired p-values have almost no power; the "
             "paired bound is the claim this control supports.")

    # Per-request overhead implied by the logging volume, for the Methods text.
    recs = []
    for r in on:
        recs += common.phase_records(r, "requests")
    steps = []
    for r in on:
        steps += common.phase_records(r, "steps")
    if recs:
        emit()
        emit(f"logging volume in the enabled arm: {len(recs)} request records, "
             f"{len(steps)} step records over {len(on)} runs")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
