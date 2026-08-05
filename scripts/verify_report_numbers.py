#!/usr/bin/env python3
"""Check every number claimed in report/main.md against the measured data.

The claims below are transcribed from the report by hand and carry the section
they appear in. Each is recomputed from results/ through the same loader the
figure scripts use (scripts/plots/common.py), so a disagreement means the prose
drifted from the data: the figures were regenerated from the loader, the text
was not.

    python scripts/verify_report_numbers.py                 # all claims
    python scripts/verify_report_numbers.py --section 5.2   # one section
    python scripts/verify_report_numbers.py --discover      # print data schema

Aggregation follows the figures: a per-record field is averaged within each run
first, then across the repetitions of that point (Figure 12: "step times are
averaged over the measured section of each run"). resource_rows() is already
scoped to the measured window, so resource means need no extra handling.

Lines marked INFO are computed and printed but not judged -- they surface
quantities the report states without a directly comparable definition.
"""

import argparse
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "plots"))
sys.path.insert(0, HERE)

import common  # noqa: E402


class Missing(Exception):
    """A record set or column the claim needs is absent from this dataset."""


# ---------------------------------------------------------------- data access

def _as_list(R):
    return list(R.values()) if isinstance(R, dict) else list(R)


def runs_for(R, group, rate=None, conc=None):
    rs = common.select(R, group=group)
    if rate is not None:
        rs = [r for r in rs if str(r.get("request_rate")) == str(rate)]
    if conc is not None:
        rs = [r for r in rs if str(r.get("max_concurrency")) == str(conc)]
    if not rs:
        raise Missing(f"no runs: group={group} rate={rate} conc={conc}")
    return rs


def bench(R, group, rate=None, field=None, conc=None):
    vals = []
    for r in runs_for(R, group, rate, conc):
        b = r.get("bench") or {}
        if field not in b:
            raise Missing(f"bench[{field!r}] absent")
        vals.append(b[field])
    return st.mean(vals)


def rec_mean(R, group, rate, kind, field, scale=1.0, filt=None):
    """Per-run mean of a phase-log field, then mean across repetitions."""
    per_run = []
    for run in runs_for(R, group, rate):
        recs = common.phase_records(run, kind)
        if not recs:
            raise Missing(f"no {kind} records")
        if field not in recs[0]:
            raise Missing(f"{kind}[{field!r}] absent")
        vals = [rec[field] * scale for rec in recs
                if filt is None or filt(rec)]
        if vals:
            per_run.append(st.mean(vals))
    if not per_run:
        raise Missing(f"{kind}[{field!r}]: nothing passed the filter")
    return st.mean(per_run)


def rec_max(R, group, rate, kind, field, filt=None):
    vals = []
    for run in runs_for(R, group, rate):
        for rec in common.phase_records(run, kind):
            if (filt is None or filt(rec)) and field in rec:
                vals.append(rec[field])
    if not vals:
        raise Missing(f"{kind}[{field!r}]: no records")
    return max(vals)


def res_mean(R, group, rate, col, gpu=None, conc=None):
    """Per-run mean of a 1 Hz resource column, then across repetitions.

    gpu=0..3 selects one device; gpu='mean' averages only the devices actually
    loaded (VRAM above 1 GiB), which is how a tp=2 run keeps GPU2 and GPU3 --
    holding 3 MiB each -- out of the average.
    """
    per_run = []
    for run in runs_for(R, group, rate, conc):
        rows = common.resource_rows(run)
        if not rows:
            continue
        if gpu is None:
            v = common.mean_of(rows, col)
        elif gpu == "mean":
            idx = [i for i in common.gpu_columns(rows)
                   if (common.mean_of(rows, f"gpu{i}_mem_mib", 0) or 0) > 1024]
            if not idx:
                continue
            v = st.mean([common.mean_of(rows, col.format(i)) for i in idx])
        else:
            v = common.mean_of(rows, col.format(gpu))
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        per_run.append(v)
    if not per_run:
        raise Missing(f"resource column {col!r}: no usable samples")
    return st.mean(per_run)


def pct_change(R, g_from, g_to, rate, field):
    return (bench(R, g_to, rate, field) / bench(R, g_from, rate, field)
            - 1.0) * 100.0


def phase_pct_change(R, g_from, g_to, rate, field):
    a = rec_mean(R, g_from, rate, "requests", field)
    b = rec_mean(R, g_to, rate, "requests", field)
    return (b / a - 1.0) * 100.0


def seed_matched(R, g_from, g_to, field, finite_only=False):
    """Per-(rate, rep) percent changes, the pairing the report's t-tests use.

    Section 4.4 averages over 23 pairs -- 8 rates x 3 repetitions, less the one
    containing the failed G1 run -- not over a single rate. Comparing group
    means at rate 8 alone gives a different number (-3.1 % against -0.3 %) that
    is one of the individual points the report describes as fluctuating by up
    to +/-9 % with inconsistent sign.
    """
    def index(g):
        out = {}
        for r in common.select(R, group=g):
            out[(str(r.get("request_rate")), str(r.get("rep")))] = r
        return out

    A, B = index(g_from), index(g_to)
    diffs = []
    for key, a in A.items():
        if finite_only and key[0] == "inf":
            continue
        b = B.get(key)
        if b is None:
            continue
        va = (a.get("bench") or {}).get(field)
        vb = (b.get("bench") or {}).get(field)
        if va in (None, 0) or vb is None:
            continue
        diffs.append((vb / va - 1.0) * 100.0)
    if len(diffs) < 2:
        raise Missing(f"seed-matched {g_from}->{g_to} on {field!r}: "
                      f"{len(diffs)} pairs")
    return diffs


def sm_mean(R, g_from, g_to, field, finite_only=False):
    return st.mean(seed_matched(R, g_from, g_to, field, finite_only))


def sm_t(R, g_from, g_to, field, finite_only=False):
    d = seed_matched(R, g_from, g_to, field, finite_only)
    return st.mean(d) / (st.stdev(d) / math.sqrt(len(d)))


def sm_n(R, g_from, g_to, field, finite_only=False):
    return float(len(seed_matched(R, g_from, g_to, field, finite_only)))


def seed_matched_at(R, g_from, g_to, field, rate):
    """Seed-matched percent changes at one arrival rate only."""
    def index(g):
        return {str(r.get("rep")): r
                for r in common.select(R, group=g)
                if str(r.get("request_rate")) == str(rate)}

    A, B = index(g_from), index(g_to)
    out = []
    for rep, a in A.items():
        b = B.get(rep)
        if b is None:
            continue
        va = (a.get("bench") or {}).get(field)
        vb = (b.get("bench") or {}).get(field)
        if va in (None, 0) or vb is None:
            continue
        out.append((vb / va - 1.0) * 100.0)
    if len(out) < 2:
        raise Missing(f"rate {rate}: {len(out)} pairs")
    return out


def t_at(R, g_from, g_to, field, rate):
    d = seed_matched_at(R, g_from, g_to, field, rate)
    return st.mean(d) / (st.stdev(d) / math.sqrt(len(d)))


# Section 4.4 writes "consistently significant across individual rates in the
# range of t = -2.8 to -14.3". With three seed pairs a paired t-test has two
# degrees of freedom, where the two-sided 5 % critical value is 4.303 -- so a
# rate at |t| = 2.8 (p = 0.107) is not significant. These count how many of the
# seven finite rates actually clear the threshold.
T_CRIT_DF2 = 4.303


def n_significant(R, g_from, g_to, field):
    n = 0
    for rate in (1, 2, 3, 4, 5, 6, 8):
        try:
            if abs(t_at(R, g_from, g_to, field, rate)) > T_CRIT_DF2:
                n += 1
        except Missing:
            continue
    return float(n)


def t_range(R, g_from, g_to, field, which):
    ts = []
    for rate in (1, 2, 3, 4, 5, 6, 8):
        try:
            ts.append(t_at(R, g_from, g_to, field, rate))
        except Missing:
            continue
    return min(ts, key=abs) if which == "min" else max(ts, key=abs)


# chunked prefill is always on, so a step is decode-only exactly when it
# carried no context tokens (section 3.1)
DECODE_ONLY = lambda s: s.get("n_ctx_toks", 0) == 0        # noqa: E731
CARRIES_PREFILL = lambda s: s.get("n_ctx_toks", 0) > 0     # noqa: E731


# --------------------------------------------------------------------- claims

class Claim:
    def __init__(self, sec, label, claimed, fn, tol, unit):
        self.sec, self.label, self.claimed = sec, label, claimed
        self.fn, self.tol, self.unit = fn, tol, unit


CLAIMS = []


def add(sec, label, claimed, fn, tol=0.02, unit=""):
    CLAIMS.append(Claim(sec, label, claimed, fn, tol, unit))


def info(sec, label, fn, unit=""):
    CLAIMS.append(Claim(sec, label, None, fn, 0, unit))


# --- 3.1 instrumentation -----------------------------------------------------
add("3.1", "n_cached max over S1 rate 5", 0,
    lambda R: rec_max(R, "S1", 5, "requests", "n_cached"), tol=0.0)
info("3.1", "request records, S1 rate 5, warm-up removed",
     lambda R: sum(len(common.phase_records(r, "requests"))
                   for r in runs_for(R, "S1", 5)))

# --- 3.5 anchors and the C1 control ------------------------------------------
for g, v in [("A1a", 3.195), ("A1b", 3.205), ("A1c", 3.174), ("A1d", 3.187)]:
    add("3.5", f"anchor {g} throughput", v,
        lambda R, g=g: bench(R, g, 5, "request_throughput"), tol=0.01,
        unit="req/s")
add("3.5", "C1 cost, TTFT p50 (%)", 2.97,
    lambda R: pct_change(R, "C1off", "A1a", 5, "p50_ttft_ms"), tol=0.35)
add("3.5", "C1 cost, throughput (%)", -0.26,
    lambda R: pct_change(R, "C1off", "A1a", 5, "request_throughput"), tol=1.0)

# --- 4.1 arrival rate (S1) ----------------------------------------------------
for rate, f, v in [(1, "p50_ttft_ms", 76), (8, "p50_ttft_ms", 120),
                   (1, "p95_ttft_ms", 151), (8, "p95_ttft_ms", 245),
                   (1, "p50_tpot_ms", 33.0), (8, "p50_tpot_ms", 44.2),
                   (1, "p95_tpot_ms", 34.8), (8, "p95_tpot_ms", 57.8),
                   (1, "p50_itl_ms", 32.4), (8, "p50_itl_ms", 35.6),
                   (1, "p95_itl_ms", 34.3), (8, "p95_itl_ms", 95.0),
                   ("inf", "p50_ttft_ms", 3553),
                   ("inf", "p95_ttft_ms", 6767)]:
    add("4.1", f"S1 {f} @ rate {rate}", v,
        lambda R, r=rate, f=f: bench(R, "S1", r, f), unit="ms")
add("4.1", "S1 request throughput @ rate 8", 3.54,
    lambda R: bench(R, "S1", 8, "request_throughput"), unit="req/s")
add("4.1", "S1 output throughput @ rate 1", 181,
    lambda R: bench(R, "S1", 1, "output_throughput"), unit="tok/s")
add("4.1", "S1 output throughput @ rate 8", 675,
    lambda R: bench(R, "S1", 8, "output_throughput"), unit="tok/s")
add("4.1", "queue dwell @ rate 8", 0.021,
    lambda R: rec_mean(R, "S1", 8, "requests", "queued_s", 1000), tol=0.25,
    unit="ms")
add("4.1", "queue dwell max, finite rates", 0.08,
    lambda R: max(rec_max(R, "S1", r, "requests", "queued_s") * 1000
                  for r in (1, 2, 3, 4, 5, 6, 8)), tol=0.30, unit="ms")
for rate, v in [(1, 6.0), (4, 18.8), (8, 23.0)]:
    add("4.1", f"decode-only batch @ rate {rate}", v,
        lambda R, r=rate: rec_mean(R, "S1", r, "steps", "n_gen_reqs",
                                   filt=DECODE_ONLY), tol=0.05)
add("4.1", "decode-only batch max @ rate 8", 76,
    lambda R: rec_max(R, "S1", 8, "steps", "n_gen_reqs", filt=DECODE_ONLY),
    tol=0.05)

# --- 4.2 phase composition ----------------------------------------------------
add("4.2", "e2e mean, S1 rate 5", 7434,
    lambda R: rec_mean(R, "S1", 5, "requests", "e2e_s", 1000), unit="ms")
add("4.2", "prefill mean, S1 rate 5", 68.8,
    lambda R: rec_mean(R, "S1", 5, "requests", "prefill_s", 1000), unit="ms")
add("4.2", "decode mean, S1 rate 5", 7334,
    lambda R: rec_mean(R, "S1", 5, "requests", "decode_s", 1000), unit="ms")
add("4.2", "client TTFT mean, S1 rate 5", 111.6,
    lambda R: bench(R, "S1", 5, "mean_ttft_ms"), unit="ms")
# the report gives e2e = 7434 while queued+prefill+decode = 7403.7; this prints
# the 30 ms gap so it gets explained or corrected rather than left open
info("4.2", "e2e minus (queued+prefill+decode), S1 rate 5",
     lambda R: (rec_mean(R, "S1", 5, "requests", "e2e_s", 1000)
                - rec_mean(R, "S1", 5, "requests", "queued_s", 1000)
                - rec_mean(R, "S1", 5, "requests", "prefill_s", 1000)
                - rec_mean(R, "S1", 5, "requests", "decode_s", 1000)),
     unit="ms")
for g, v in [("I1", 126), ("I2", 70)]:
    add("4.2", f"{g} prefill mean", v,
        lambda R, g=g: rec_mean(R, g, 5, "requests", "prefill_s", 1000),
        unit="ms")
for g, v in [("I1", 0.95), ("I2", 0.02)]:
    add("4.2", f"{g} queue dwell", v,
        lambda R, g=g: rec_mean(R, g, 5, "requests", "queued_s", 1000),
        tol=0.25, unit="ms")
for g, f, v in [("I1", "total_token_throughput", 2891),
                ("I2", "total_token_throughput", 2144),
                ("I1", "output_throughput", 578),
                ("I2", "output_throughput", 1715)]:
    add("4.2", f"{g} {f}", v,
        lambda R, g=g, f=f: bench(R, g, 5, f), unit="tok/s")
add("4.2", "0.5B prefill mean @ rate 1", 17.2,
    lambda R: rec_mean(R, "S3", 1, "requests", "prefill_s", 1000), tol=0.05,
    unit="ms")
add("4.2", "0.5B prefill mean @ rate 32", 14.4,
    lambda R: rec_mean(R, "S3", 32, "requests", "prefill_s", 1000), tol=0.05,
    unit="ms")
add("4.2", "0.5B client TTFT mean @ rate 1", 33.1,
    lambda R: bench(R, "S3", 1, "mean_ttft_ms"), tol=0.05, unit="ms")
add("4.2", "0.5B client TTFT mean @ rate 32", 49.2,
    lambda R: bench(R, "S3", 32, "mean_ttft_ms"), tol=0.05, unit="ms")

# --- 4.3 dataset and model size -----------------------------------------------
add("4.3", "random TTFT p95 @ rate 1", 109,
    lambda R: bench(R, "S2", 1, "p95_ttft_ms"), unit="ms")
add("4.3", "random TTFT p95 @ rate 8", 211,
    lambda R: bench(R, "S2", 8, "p95_ttft_ms"), unit="ms")
add("4.3", "random output throughput @ rate 5", 580,
    lambda R: bench(R, "S2", 5, "output_throughput"), unit="tok/s")
add("4.3", "random output throughput @ rate 8", 874,
    lambda R: bench(R, "S2", 8, "output_throughput"), unit="tok/s")
add("4.3", "ShareGPT output throughput @ rate 5", 612,
    lambda R: bench(R, "S1", 5, "output_throughput"), unit="tok/s")
add("4.3", "S2b request throughput @ rate 20", 12.10,
    lambda R: bench(R, "S2b", 20, "request_throughput"), unit="req/s")
add("4.3", "S2b output throughput @ rate 20", 1549,
    lambda R: bench(R, "S2b", 20, "output_throughput"), unit="tok/s")
add("4.3", "S2b TPOT p95 @ rate 12", 85.1,
    lambda R: bench(R, "S2b", 12, "p95_tpot_ms"), unit="ms")
add("4.3", "0.5B TTFT p95 @ rate 1", 52,
    lambda R: bench(R, "S3", 1, "p95_ttft_ms"), unit="ms")
add("4.3", "0.5B TPOT p95 @ rate 1", 6.1,
    lambda R: bench(R, "S3", 1, "p95_tpot_ms"), unit="ms")
add("4.3", "0.5B TTFT p95 @ rate 32", 78,
    lambda R: bench(R, "S3", 32, "p95_ttft_ms"), unit="ms")
add("4.3", "0.5B output throughput @ rate 32", 3371,
    lambda R: bench(R, "S3", 32, "output_throughput"), unit="tok/s")
add("4.3", "0.5B request throughput @ rate 32", 17.66,
    lambda R: bench(R, "S3", 32, "request_throughput"), unit="req/s")
add("4.3", "P0 offline probe throughput", 17.3,
    lambda R: bench(R, "P0", "inf", "request_throughput"), tol=0.03,
    unit="req/s")

# --- 4.4 GPU count and parallelism strategy ------------------------------------
for g, rate, v in [("G2", 8, 32.5), ("G4", 8, 52.0),
                   ("G2", "inf", 44.0), ("G4", "inf", 55.4)]:
    add("4.4", f"{g} throughput gain @ rate {rate} (%)", v,
        lambda R, g=g, r=rate: pct_change(R, "G1", g, r,
                                          "request_throughput"), tol=0.06)
for g, v in [("G1", 61.4), ("G2", 43.6), ("G4", 36.9)]:
    add("4.4", f"{g} TPOT p95 @ rate 8", v,
        lambda R, g=g: bench(R, g, 8, "p95_tpot_ms"), unit="ms")
for g, v in [("G1", 248), ("G2", 227), ("G4", 213)]:
    add("4.4", f"{g} TTFT p95 @ rate 8", v,
        lambda R, g=g: bench(R, g, 8, "p95_ttft_ms"), unit="ms")
for g, f, v in [("G2", "prefill_s", -13.2), ("G2", "decode_s", -28.5),
                ("G4", "prefill_s", -16.9), ("G4", "decode_s", -42.5),
                ("P1", "prefill_s", -10.4), ("P1", "decode_s", -1.7)]:
    add("4.4", f"{g} {f[:-2]} change @ rate 5 (%)", v,
        lambda R, g=g, f=f: phase_pct_change(R, "G1", g, 5, f),
        tol=0.15 if abs(v) > 5 else 1.0)
add("4.4", "pp=2 throughput, seed-matched mean (%)", -0.3,
    lambda R: sm_mean(R, "G1", "P1", "request_throughput"), tol=1.0)
add("4.4", "pp=2 throughput, seed-matched t", -0.43,
    lambda R: sm_t(R, "G1", "P1", "request_throughput"), tol=0.35)
add("4.4", "pp=2 throughput, pair count", 23,
    lambda R: sm_n(R, "G1", "P1", "request_throughput"), tol=0.0)
add("4.4", "pp=2 TTFT p95, seed-matched mean, finite (%)", -12.0,
    lambda R: sm_mean(R, "G1", "P1", "p95_ttft_ms", True), tol=0.10)
add("4.4", "pp=2 TTFT p95, seed-matched t", -12.1,
    lambda R: sm_t(R, "G1", "P1", "p95_ttft_ms", True), tol=0.25)
add("4.4", "pp=2 TTFT p95, pair count, finite", 20,
    lambda R: sm_n(R, "G1", "P1", "p95_ttft_ms", True), tol=0.0)
add("4.4", "pp=2 mean TTFT, seed-matched mean, finite (%)", -11.9,
    lambda R: sm_mean(R, "G1", "P1", "mean_ttft_ms", True), tol=0.10)
add("4.4", "tp=2 TTFT p95, seed-matched mean, finite (%)", -10.0,
    lambda R: sm_mean(R, "G1", "G2", "p95_ttft_ms", True), tol=0.10)
add("4.4", "tp=2 TTFT p95, seed-matched t", -9.8,
    lambda R: sm_t(R, "G1", "G2", "p95_ttft_ms", True), tol=0.25)
info("4.4", "tp=2 throughput, seed-matched: negative pairs (report: 0)",
     lambda R: float(sum(1 for d in seed_matched(R, "G1", "G2",
                                                 "request_throughput")
                         if d <= 0)))
info("4.4", "pp=2 throughput, largest |pair| (%), report: up to 9",
     lambda R: max(abs(d) for d in seed_matched(R, "G1", "P1",
                                                "request_throughput")))
info("4.4", "pp=2 throughput at rate 8 alone (%)",
     lambda R: pct_change(R, "G1", "P1", 8, "request_throughput"))
info("4.4", "pp=2 TTFT p95: smallest |t| over rates (report: 2.8)",
     lambda R: t_range(R, "G1", "P1", "p95_ttft_ms", "min"))
info("4.4", "pp=2 TTFT p95: largest |t| over rates (report: 14.3)",
     lambda R: t_range(R, "G1", "P1", "p95_ttft_ms", "max"))
info("4.4", "pp=2 TTFT p95: rates significant at df=2 (|t|>4.303), of 7",
     lambda R: n_significant(R, "G1", "P1", "p95_ttft_ms"))

# --- 4.5 closed loop ------------------------------------------------------------
for conc, v in [(1, 31.9), (64, 711.3)]:
    add("4.5", f"C2 output throughput @ concurrency {conc}", v,
        lambda R, c=conc: bench(R, "C2", None, "output_throughput", conc=c),
        unit="tok/s")
add("4.5", "C2x output throughput @ concurrency 128", 717.4,
    lambda R: bench(R, "C2x", None, "output_throughput", conc=128),
    unit="tok/s")
add("4.5", "C2 E2EL p95 @ concurrency 1", 19.2,
    lambda R: bench(R, "C2", None, "p95_e2el_ms", conc=1) / 1000, tol=0.05,
    unit="s")
add("4.5", "C2 E2EL p95 @ concurrency 16", 23.0,
    lambda R: bench(R, "C2", None, "p95_e2el_ms", conc=16) / 1000, tol=0.05,
    unit="s")
add("4.5", "C2x E2EL p95 @ concurrency 128", 27.8,
    lambda R: bench(R, "C2x", None, "p95_e2el_ms", conc=128) / 1000, tol=0.05,
    unit="s")
add("4.5", "C2 request throughput @ concurrency 64", 3.73,
    lambda R: bench(R, "C2", None, "request_throughput", conc=64), tol=0.03,
    unit="req/s")

# --- 5.1 KV cache ----------------------------------------------------------------
for g, v in [("G1", 2.13), ("G2", 0.77), ("G4", 0.33)]:
    add("5.1", f"{g} KV usage @ rate 8 (%)", v,
        lambda R, g=g: rec_mean(R, g, 8, "steps", "kv_usage", 100), tol=0.10)
add("5.1", "G1 KV usage peak @ rate 8 (%)", 6.43,
    lambda R: rec_max(R, "G1", 8, "steps", "kv_usage") * 100, tol=0.10)
for g, v in [("G1", 22.9), ("G2", 22.3), ("G4", 21.7)]:
    add("5.1", f"{g} decode batch @ rate 8", v,
        lambda R, g=g: rec_mean(R, g, 8, "steps", "n_gen_reqs",
                                filt=DECODE_ONLY), tol=0.05)

# --- 5.2 step times (a stale set of these sits in an HTML comment) ---------------
for g, v in [("G1", 32.27), ("G2", 22.27), ("G4", 17.95)]:
    add("5.2", f"{g} decode-only step @ rate 8", v,
        lambda R, g=g: rec_mean(R, g, 8, "steps", "exec_s", 1000,
                                filt=DECODE_ONLY), unit="ms")
for g, v in [("G1", 72.34), ("G2", 63.27), ("G4", 57.33)]:
    add("5.2", f"{g} prefill-carrying step @ rate 8", v,
        lambda R, g=g: rec_mean(R, g, 8, "steps", "exec_s", 1000,
                                filt=CARRIES_PREFILL), unit="ms")
add("5.2", "G1 decode-only step @ rate 1", 31.48,
    lambda R: rec_mean(R, "G1", 1, "steps", "exec_s", 1000, filt=DECODE_ONLY),
    unit="ms")
add("5.2", "G4 decode-only step @ rate 1", 12.71,
    lambda R: rec_mean(R, "G4", 1, "steps", "exec_s", 1000, filt=DECODE_ONLY),
    unit="ms")

# --- 5.3 resource counters, Session C (G1..G4) -----------------------------------
add("5.3", "G1 SM util @ rate 8 (%)", 87.9,
    lambda R: res_mean(R, "G1", 8, "gpu{}_util", gpu=0), tol=0.05)
add("5.3", "G4 SM util @ rate 8, gpu0 (%)", 82.1,
    lambda R: res_mean(R, "G4", 8, "gpu{}_util", gpu=0), tol=0.05)
add("5.3", "G1 memory controller @ rate 8 (%)", 93.6,
    lambda R: res_mean(R, "G1", 8, "gpu{}_memutil", gpu=0), tol=0.05)
add("5.3", "G2 memory controller @ rate 8, gpu0 (%)", 64.3,
    lambda R: res_mean(R, "G2", 8, "gpu{}_memutil", gpu=0), tol=0.05)
add("5.3", "G4 memory controller @ rate 8, gpu0 (%)", 39.5,
    lambda R: res_mean(R, "G4", 8, "gpu{}_memutil", gpu=0), tol=0.05)
info("5.3", "G4 memory controller @ rate 8, 4-GPU mean (%)",
     lambda R: st.mean([res_mean(R, "G4", 8, "gpu{}_memutil", gpu=i)
                        for i in range(4)]))
add("5.3", "G1 GPU power @ rate 8 (W)", 285,
    lambda R: res_mean(R, "G1", 8, "gpu{}_power_w", gpu=0), tol=0.05)
add("5.3", "G4 GPU power @ rate 8, gpu0 (W)", 187,
    lambda R: res_mean(R, "G4", 8, "gpu{}_power_w", gpu=0), tol=0.05)

# --- 5.4 topology -----------------------------------------------------------------
for i, v in enumerate([39.5, 39.3, 39.7, 39.4]):
    add("5.4", f"G4 memory controller gpu{i} @ rate 8 (%)", v,
        lambda R, i=i: res_mean(R, "G4", 8, "gpu{}_memutil", gpu=i), tol=0.05)
add("5.4", "G2 VRAM gpu0 (MiB)", 43831,
    lambda R: res_mean(R, "G2", 8, "gpu{}_mem_mib", gpu=0), tol=0.05)
add("5.4", "G2 VRAM gpu2 (MiB)", 3,
    lambda R: res_mean(R, "G2", 8, "gpu{}_mem_mib", gpu=2), tol=1.0)
add("5.4", "G4 VRAM gpu0 (MiB)", 44115,
    lambda R: res_mean(R, "G4", 8, "gpu{}_mem_mib", gpu=0), tol=0.05)

# --- 5.5 CPU and framework-bound, Session A (S1 / S3) -----------------------------
# 5.3 quotes the same nominal condition from G1; if both sets pass, the report
# is silently comparing two sessions and should say so
add("5.5", "S1 SM util @ rate 8 (%)", 88.8,
    lambda R: res_mean(R, "S1", 8, "gpu{}_util", gpu=0), tol=0.05)
add("5.5", "S1 memory controller @ rate 8 (%)", 94.2,
    lambda R: res_mean(R, "S1", 8, "gpu{}_memutil", gpu=0), tol=0.05)
add("5.5", "S1 GPU power @ rate 8 (W)", 273,
    lambda R: res_mean(R, "S1", 8, "gpu{}_power_w", gpu=0), tol=0.05)
add("5.5", "S3 SM util @ rate 8 (%)", 54.9,
    lambda R: res_mean(R, "S3", 8, "gpu{}_util", gpu=0), tol=0.05)
add("5.5", "S3 memory controller @ rate 8 (%)", 39.3,
    lambda R: res_mean(R, "S3", 8, "gpu{}_memutil", gpu=0), tol=0.05)
add("5.5", "S3 SM util @ rate 32 (%)", 53.3,
    lambda R: res_mean(R, "S3", 32, "gpu{}_util", gpu=0), tol=0.05)
add("5.5", "S3 memory controller @ rate 32 (%)", 38.8,
    lambda R: res_mean(R, "S3", 32, "gpu{}_memutil", gpu=0), tol=0.05)
for rate, v in [(1, 28.1), (6, 40.5), (8, 39.4)]:
    add("5.5", f"S1 server CPU @ rate {rate} (%)", v,
        lambda R, r=rate: res_mean(R, "S1", r, "cpu_server_pct"), tol=0.05)
for rate, v in [(1, 65.3), (8, 120.8), (24, 144.4), (32, 141.2)]:
    add("5.5", f"S3 server CPU @ rate {rate} (%)", v,
        lambda R, r=rate: res_mean(R, "S3", r, "cpu_server_pct"), tol=0.05)
add("5.5", "S3 client CPU @ rate 1 (%)", 3.9,
    lambda R: res_mean(R, "S3", 1, "cpu_client_pct"), tol=0.10)
add("5.5", "S3 client CPU @ rate 32 (%)", 40.6,
    lambda R: res_mean(R, "S3", 32, "cpu_client_pct"), tol=0.10)
info("5.5", "S1 system-wide CPU @ rate 8 (%), report says 5-8",
     lambda R: res_mean(R, "S1", 8, "cpu_total"))
info("5.5", "S3 system-wide CPU @ rate 32 (%), report says 5-8",
     lambda R: res_mean(R, "S3", 32, "cpu_total"))
add("5.5", "S1 sched time @ rate 8", 0.801,
    lambda R: rec_mean(R, "S1", 8, "steps", "sched_s", 1000), tol=0.05,
    unit="ms")
add("5.5", "S1 exec time @ rate 8", 36.17,
    lambda R: rec_mean(R, "S1", 8, "steps", "exec_s", 1000), tol=0.05,
    unit="ms")
add("5.5", "S3 sched time @ rate 8", 0.303,
    lambda R: rec_mean(R, "S3", 8, "steps", "sched_s", 1000), tol=0.05,
    unit="ms")
add("5.5", "S3 exec time @ rate 8", 5.29,
    lambda R: rec_mean(R, "S3", 8, "steps", "exec_s", 1000), tol=0.05,
    unit="ms")
add("5.5", "S3 sched time @ rate 32", 0.549,
    lambda R: rec_mean(R, "S3", 32, "steps", "sched_s", 1000), tol=0.05,
    unit="ms")
add("5.5", "S3 exec time @ rate 32", 6.18,
    lambda R: rec_mean(R, "S3", 32, "steps", "exec_s", 1000), tol=0.05,
    unit="ms")

# --- 6 the scheduler-time asymmetry stated but not measured ----------------------
add("6", "G1 sched time @ rate 8", 0.848,
    lambda R: rec_mean(R, "G1", 8, "steps", "sched_s", 1000), tol=0.05,
    unit="ms")
add("6", "G2 sched time @ rate 8", 0.22,
    lambda R: rec_mean(R, "G2", 8, "steps", "sched_s", 1000), tol=0.20,
    unit="ms")


# ----------------------------------------------------------------- discovery

def discover(R):
    import inspect

    print("=== common.py exports ===")
    print("  " + ", ".join(sorted(n for n in dir(common)
                                  if not n.startswith("_"))))
    print("\n=== signatures ===")
    for name in ("load_runs", "select", "aggregate", "phase_records",
                 "resource_rows", "gpu_columns", "mean_of",
                 "measurement_window", "describe", "rate_key"):
        fn = getattr(common, name, None)
        if fn is None:
            continue
        try:
            print(f"  {name}{inspect.signature(fn)}")
        except (TypeError, ValueError):
            print(f"  {name}: {type(fn).__name__}")
        doc = (inspect.getdoc(fn) or "").split("\n")[0]
        if doc:
            print(f"      {doc}")

    runs = _as_list(R)
    print(f"\n=== container ===\n  type={type(R).__name__}  n={len(runs)}")
    run = runs[0]
    print("\n=== run keys ===\n  " + ", ".join(sorted(map(str, run.keys()))))
    print("\n=== run['bench'] keys ===\n  "
          + ", ".join(sorted(map(str, (run.get("bench") or {}).keys()))))
    for kind in ("requests", "steps"):
        recs = common.phase_records(run, kind)
        print(f"\n=== {kind} keys ({len(recs)} records) ===\n  "
              + ", ".join(sorted(map(str, recs[0].keys()))))
    for r in runs[:20]:
        rows = common.resource_rows(r)
        if rows:
            print(f"\n=== resource_rows keys, group {r.get('group')} "
                  f"({len(rows)} rows) ===\n  "
                  + ", ".join(sorted(map(str, rows[0].keys()))))
            print(f"  gpu_columns -> {common.gpu_columns(rows)}")
            break


# --------------------------------------------------------------------- runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--results-dir")
    ap.add_argument("--section", help="e.g. 5.2")
    ap.add_argument("--discover", action="store_true")
    a = ap.parse_args()

    kw = {}
    if a.repo:
        kw["repo"] = a.repo
    if a.results_dir:
        kw["results_dirs"] = a.results_dir
    R = common.load_runs(**kw)

    if a.discover:
        discover(R)
        return 0

    claims = [c for c in CLAIMS if not a.section or c.sec == a.section]
    if not claims:
        print(f"no claims in section {a.section!r}")
        return 2
    width = max(len(c.label) for c in claims)
    ok = bad = skipped = 0
    fails = []
    current = None

    for c in claims:
        if c.sec != current:
            current = c.sec
            print(f"\n--- section {current} " + "-" * max(0, width - 10))
        try:
            got = c.fn(R)
        except Missing as e:
            print(f"SKIP {c.label:<{width}}  {e}")
            skipped += 1
            continue
        except Exception as e:                              # noqa: BLE001
            print(f"SKIP {c.label:<{width}}  {type(e).__name__}: {e}")
            skipped += 1
            continue
        if c.claimed is None:
            print(f"INFO {c.label:<{width}}  {'':>18}"
                  f"  data={got:>10.3f} {c.unit}")
            continue
        off = abs(got - c.claimed) / (abs(c.claimed) or 1.0)
        if off <= c.tol:
            ok += 1
            mark = "OK  "
        else:
            bad += 1
            mark = "FAIL"
            fails.append((c, got, off))
        print(f"{mark} {c.label:<{width}}  report={c.claimed:>10.3f}"
              f"  data={got:>10.3f} {c.unit:<6} ({off*100:5.1f}%)")

    print(f"\n{ok} OK   {bad} FAIL   {skipped} SKIP")
    if fails:
        print("\n=== mismatches, largest first ===")
        for c, got, off in sorted(fails, key=lambda x: -x[2]):
            print(f"  {c.sec:<4} {c.label:<{width}} "
                  f"report={c.claimed:.3f}  data={got:.3f}  ({off*100:.1f}%)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
