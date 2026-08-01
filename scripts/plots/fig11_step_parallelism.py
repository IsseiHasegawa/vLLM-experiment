"""Figure 11: why tensor parallelism helps, measured at step granularity.

This is the figure that connects the assignment's two hardest requirements -
"measure the time and resource usage during the prefill phase and the decode
phase" and "enable and evaluate parallel processing options" - using the step
log, which no off-the-shelf vLLM output provides.

Left panel   - mean model-execution time per engine step at rate 8, split into
               decode-only steps (n_ctx_toks == 0) and steps that carry prefill
               work (n_ctx_toks > 0). Chunked prefill is on by default (D23), so
               "prefill" here means "this step also processed context tokens",
               not a contiguous prefill interval. The two phases respond very
               differently to sharding: decode is memory-bandwidth bound and
               gains most, prefill is already compute-dense and gains least.

Middle panel - the same decode-only steps binned by batch size. The speedup
               shrinks as the batch grows: at small batch almost all of the step
               is spent streaming weights, which sharding divides by N; at large
               batch the weights are amortised over more tokens while the
               all-reduce volume grows with the batch.

Right panel  - decode step time and the resulting throughput gain against
               arrival rate. Steps are ~1.7x faster at rate 1 yet throughput
               rises only 3.5%: below saturation the arrival rate sets
               throughput, and a faster step only buys idle time. The two curves
               converge only once the system is saturated.

Data: steps-*.jsonl.gz sliced by the manifest windows of G1/G2/G4, plus
achieved throughput from the manifest notes.
"""

import statistics as st

import matplotlib.pyplot as plt

from common import C, MARKERS, SERIES, phase_records, save, select

GROUPS = ("G1", "G2", "G4")
LABELS = {"G1": "tp=1", "G2": "tp=2", "G4": "tp=4"}


def _steps(runs, group, rate=None):
    """Step records for a group, optionally restricted to one arrival rate."""
    out = []
    for run in select(runs, group=group):
        if rate is not None and str(run.get("request_rate")) != str(rate):
            continue
        out += phase_records(run, "steps")
    return out


def _throughput(runs, group, rate):
    v = [r["bench"]["request_throughput"] for r in select(runs, group=group)
         if str(r.get("request_rate")) == str(rate)]
    return st.mean(v) if v else None


def fig11(runs, outdir, rate="8"):
    have = [g for g in GROUPS if select(runs, group=g)]
    if len(have) < 2:
        print("  SKIP fig11: need at least two of G1/G2/G4")
        return None

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # ---- left: decode-only vs prefill-carrying steps ---------------------
    width = 0.35
    xs = range(len(have))
    dec, pre = [], []
    for g in have:
        s = _steps(runs, g, rate)
        d = [x["exec_s"] * 1000 for x in s if x.get("n_ctx_toks", 0) == 0]
        p = [x["exec_s"] * 1000 for x in s if x.get("n_ctx_toks", 0) > 0]
        dec.append(st.mean(d) if d else 0.0)
        pre.append(st.mean(p) if p else 0.0)
    a1.bar([x - width / 2 for x in xs], dec, width, color=C["blue"],
           label="decode-only steps")
    a1.bar([x + width / 2 for x in xs], pre, width, color=C["orange"],
           label="steps carrying prefill")
    for i, (d, p) in enumerate(zip(dec, pre)):
        if dec[0]:
            a1.annotate(f"{dec[0]/d:.2f}x", (i - width / 2, d), ha="center",
                        va="bottom", fontsize=7, color=C["grey"])
        if pre[0]:
            a1.annotate(f"{pre[0]/p:.2f}x", (i + width / 2, p), ha="center",
                        va="bottom", fontsize=7, color=C["grey"])
    a1.set_xticks(list(xs))
    a1.set_xticklabels([LABELS[g] for g in have])
    a1.set_ylabel("Model execution time per step (ms)")
    a1.set_title(f"Step time by phase (rate {rate})")
    a1.legend(fontsize=7, frameon=False)

    # ---- middle: decode-only steps binned by batch size ------------------
    bins = [(4, 8), (8, 16), (16, 32), (32, 64)]
    for i, g in enumerate(have):
        s = [x for x in _steps(runs, g, rate) if x.get("n_ctx_toks", 0) == 0]
        xs2, ys2 = [], []
        for lo, hi in bins:
            v = [x["exec_s"] * 1000 for x in s
                 if lo <= x.get("n_gen_toks", 0) < hi]
            if len(v) >= 30:
                xs2.append(f"{lo}-{hi}")
                ys2.append(st.mean(v))
        if xs2:
            a2.plot(xs2, ys2, color=SERIES[i], marker=MARKERS[i],
                    label=LABELS[g])
    a2.set_xlabel("Decode batch size (requests in step)")
    a2.set_ylabel("Step time (ms)")
    a2.set_title("Decode step time vs batch size")
    a2.legend(fontsize=7, frameon=False)

    # ---- right: step time and throughput gain vs arrival rate ------------
    rates = ["1", "2", "3", "4", "5", "6", "8", "inf"]
    for i, g in enumerate(have):
        xs3, ys3 = [], []
        for r in rates:
            s = [x for x in _steps(runs, g, r) if x.get("n_ctx_toks", 0) == 0]
            if s:
                xs3.append(r)
                ys3.append(st.mean(x["exec_s"] * 1000 for x in s))
        if xs3:
            a3.plot(xs3, ys3, color=SERIES[i], marker=MARKERS[i],
                    label=f"{LABELS[g]} step time")
    a3.set_xlabel("Request rate (req/s)")
    a3.set_ylabel("Decode step time (ms)")
    a3.set_title("Step speedup does not become\nthroughput until saturation")

    a4 = a3.twinx()
    base = "G1"
    for i, g in enumerate(have):
        if g == base:
            continue
        xs4, ys4 = [], []
        for r in rates:
            t0 = _throughput(runs, base, r)
            t1 = _throughput(runs, g, r)
            if t0 and t1:
                xs4.append(r)
                ys4.append(100 * (t1 / t0 - 1))
        if xs4:
            a4.plot(xs4, ys4, color=SERIES[have.index(g)], linestyle=":",
                    marker=MARKERS[have.index(g)], markerfacecolor="white",
                    label=f"{LABELS[g]} throughput gain")
    a4.set_ylabel("Throughput gain over tp=1 (%)")
    h1, l1 = a3.get_legend_handles_labels()
    h2, l2 = a4.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc="center left")

    fig.suptitle("Where tensor parallelism helps: step-level decomposition "
                 "(Qwen2.5-7B, ShareGPT, one 4x A40 instance)")
    return save(fig, "fig11_step_parallelism", outdir)


ALL = {11: fig11}
