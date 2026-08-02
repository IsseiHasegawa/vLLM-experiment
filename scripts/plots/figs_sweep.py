"""Figures 1-4, 6, 7, 10: request-rate sweeps, overlays and the closed loop.

  fig01  rate -> TTFT (p50, p95)                       S1
  fig02  rate -> TPOT and ITL (p50, p95)               S1
  fig03  rate -> throughput (requests/s, tokens/s)     S1
  fig04  rate -> TTFT p95 and throughput, ShareGPT vs random   S1 vs S2
  fig06  rate -> TTFT p95, TPOT p95 and throughput, 7B vs 0.5B S1 vs S3
  fig07  rate -> TTFT p95, TPOT p95 and throughput, 1/2/4 GPUs G1/G2/G4 (+P1)
  fig10  closed-loop latency vs throughput                     C2 (+C2x)

Error bars are the standard deviation over the 3 repetitions of each point.
Repetitions use seeds 1/2/3 (D14), so they include prompt sampling and arrival
jitter, not system noise alone.
"""

import matplotlib.pyplot as plt

import statistics as st

from common import (C, INF_LABEL, MARKERS, SERIES, aggregate, annotate_inf,
                    log_latency, phase_records, plot_series, save, select,
                    xpos)

# Panels for these fields get a log y-axis; see common.log_latency for why.
# Throughput stays linear: it spans well under one order of magnitude, and the
# ShareGPT/random crossing in figure 4 and the tp ordering in figure 7 are both
# easiest to read on a linear scale.
LOG_FIELDS = {"p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
              "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
              "p50_itl_ms", "p95_itl_ms", "p99_itl_ms",
              "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
              "p50_e2el_ms", "p95_e2el_ms", "mean_e2el_ms"}


def _panel(ax, runs, group, field, label, color, marker, ls="-", inf_x=None,
           set_ticks=True):
    pts = aggregate(select(runs, group=group), field)
    plot_series(ax, pts, label, color, marker, ls, inf_x=inf_x,
                set_ticks=set_ticks)
    annotate_inf(ax, pts, inf_x)
    return pts


def _rate_axis(ax, groups_pts, inf_x):
    """One tick set covering every group's rate grid, shared 'inf' at the end.

    Overlays can mix grids - figure 6 puts 7B (1-8 req/s) against 0.5B
    (1-32 req/s) - so the union spans 1 to 32 and is unreadable on a linear
    axis, where 1, 2 and 4 collide while 24 and 32 sit far apart. A log axis
    spaces them evenly. Only a power-of-two subset is labelled; the remaining
    rates get unlabelled minor ticks so the points are still locatable.
    """
    finite = sorted({float(p[0]) for pts in groups_pts for p in pts
                     if p[0] != "inf"})
    if not finite:
        return
    # Only switch to log when the grids really are far apart. Figure 6 mixes
    # 1-8 (7B) with 1-32 (0.5B) and needs it; figures 4 and 7 stay on 1-8,
    # where a linear axis can label every rate including 3, 5 and 6.
    wide = max(finite) / min(finite) >= 16
    if wide:
        ax.set_xscale("log")
    major = [r for r in finite if r in (1, 2, 4, 8, 16, 32)] if wide else finite
    ax.set_xticks(major + [inf_x])
    ax.set_xticklabels([f"{r:g}" for r in major] + [INF_LABEL])
    minor = [r for r in finite if r not in major]
    if minor:
        ax.set_xticks(minor, minor=True)
        ax.set_xticklabels([], minor=True)


def fig01(runs, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _panel(ax, runs, "S1", "p50_ttft_ms", "p50", SERIES[0], MARKERS[0])
    _panel(ax, runs, "S1", "p95_ttft_ms", "p95", SERIES[1], MARKERS[1], "--")
    log_latency(ax)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("Time to first token vs arrival rate\nQwen2.5-7B, ShareGPT, 1 GPU")
    ax.legend(title="percentile")
    return save(fig, "fig01_ttft_vs_rate", outdir)


def fig02(runs, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    _panel(a1, runs, "S1", "p50_tpot_ms", "p50", SERIES[0], MARKERS[0])
    _panel(a1, runs, "S1", "p95_tpot_ms", "p95", SERIES[1], MARKERS[1], "--")
    a1.set_ylabel("TPOT (ms/token, log scale)")
    a1.set_title("Time per output token")
    _panel(a2, runs, "S1", "p50_itl_ms", "p50", SERIES[0], MARKERS[0])
    _panel(a2, runs, "S1", "p95_itl_ms", "p95", SERIES[1], MARKERS[1], "--")
    a2.set_ylabel("ITL (ms, log scale)")
    a2.set_title("Inter-token latency")
    for a in (a1, a2):
        log_latency(a)
        a.set_xlabel("Request rate (req/s)")
        a.legend(title="percentile")
    fig.suptitle("Decode-side latency vs arrival rate "
                 "(Qwen2.5-7B, ShareGPT, 1 GPU)")
    fig.tight_layout()
    return save(fig, "fig02_decode_latency_vs_rate", outdir)


def _arrival_window_rate(runs, group):
    """Completions inside the arrival window, divided by that window.

    The client's own `request_throughput` is completed / (measurement
    duration), and the measurement duration runs until the *last* request
    finishes. That drain is 12 s at rate 1 and 32 s at rate 8 here, so the
    reported rate is biased low at every point, including points where the
    server is comfortably keeping up (0.95 at an offered rate of 1). Reading a
    capacity off that curve therefore reads a definitional artefact (D24).

    This alternative counts requests that completed at or before the last
    arrival, over the arrival span. It removes the drain but introduces the
    opposite bias: requests still in flight when arrivals stop are never
    counted, which costs roughly rate x latency requests and grows with load.
    Neither series is unbiased. Plotting both is the honest statement: the
    achieved rate is definition-sensitive, and on this system - where vLLM
    admits every arrival into the running batch, so backlog appears as batch
    growth rather than as a queue - achieved throughput is a weak saturation
    detector. The saturation argument rests on latency (figure 1) and on the
    closed-loop curve (figure 10).
    """
    out = []
    by_rate = {}
    for run in select(runs, group=group):
        rate = str(run.get("request_rate"))
        if rate == "inf":
            continue
        recs = phase_records(run, "requests")
        if not recs:
            continue
        arrivals = [r["arrival_ts"] for r in recs if "arrival_ts" in r]
        if not arrivals:
            continue
        t0, t1 = min(arrivals), max(arrivals)
        span = t1 - t0
        if span <= 0:
            continue
        done = sum(1 for r in recs if r["ts"] <= t1)
        by_rate.setdefault(rate, []).append(done / span)
    for rate, vals in by_rate.items():
        m = st.mean(vals)
        e = st.stdev(vals) if len(vals) > 1 else 0.0
        out.append((rate, m, e, len(vals)))
    out.sort(key=lambda p: float(p[0]))
    return out


def fig03(runs, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.6))
    pts = _panel(a1, runs, "S1", "request_throughput",
                 "achieved: completed / measured duration",
                 SERIES[0], MARKERS[0])
    # Second definition of the same quantity, drain excluded; see
    # _arrival_window_rate for why both are shown and neither is unbiased.
    aw = _arrival_window_rate(runs, "S1")
    if aw:
        xs_aw, _ = xpos(pts)
        xmap = {p[0]: x for p, x in zip(pts, xs_aw)}
        xs2 = [xmap[p[0]] for p in aw if p[0] in xmap]
        a1.errorbar(xs2, [p[1] for p in aw], yerr=[p[2] for p in aw],
                    color=SERIES[1], marker=MARKERS[1], linestyle="--",
                    markerfacecolor="white",
                    label="achieved: completed / arrival window")
    # Reference line: achieved == requested. Departure marks saturation.
    finite = [(float(p[0]), p[1]) for p in pts if p[0] != "inf"]
    if finite:
        xs = [x for x, _ in finite]
        a1.plot(xs, xs, color=C["grey"], lw=0.8, ls=":", label="requested")
    a1.set_ylabel("Request throughput (req/s)")
    a1.set_title("Achieved vs requested rate")
    a1.legend(fontsize=7)
    _panel(a2, runs, "S1", "output_throughput", "output tokens",
           SERIES[2], MARKERS[2])
    _panel(a2, runs, "S1", "total_token_throughput", "total tokens",
           SERIES[3], MARKERS[3], "--")
    a2.set_ylabel("Token throughput (tok/s)")
    a2.set_title("Token throughput")
    a2.legend()
    for a in (a1, a2):
        a.set_xlabel("Request rate (req/s)")
    fig.suptitle("Throughput and saturation (Qwen2.5-7B, ShareGPT, 1 GPU)")
    fig.tight_layout()
    return save(fig, "fig03_throughput_vs_rate", outdir)


def _overlay(runs, outdir, groups, names, title, name, fields=None,
             log_fields=()):
    """Two-panel overlay: p95 TTFT and output throughput for each group."""
    fields = fields or [("p95_ttft_ms", "TTFT p95 (ms)"),
                        ("output_throughput", "Output throughput (tok/s)")]
    fig, axes = plt.subplots(1, len(fields), figsize=(4.4 * len(fields), 3.6))
    if len(fields) == 1:
        axes = [axes]
    # One offline position for the whole figure, derived from the widest grid.
    probe = [aggregate(select(runs, group=g), fields[0][0]) for g in groups]
    fin_all = [float(p[0]) for pts in probe for p in pts if p[0] != "inf"]
    inf_x = max(fin_all) * 1.35 if fin_all else 1.0
    for ax, (field, ylabel) in zip(axes, fields):
        pts_all = []
        for i, (g, nm) in enumerate(zip(groups, names)):
            pts_all.append(_panel(ax, runs, g, field, nm, SERIES[i],
                                  MARKERS[i], "-" if i == 0 else "--",
                                  inf_x=inf_x, set_ticks=False))
        _rate_axis(ax, pts_all, inf_x)
        if field in LOG_FIELDS or field in log_fields:
            log_latency(ax)
            ylabel = ylabel.replace(")", ", log scale)")
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return save(fig, name, outdir)


def fig04(runs, outdir):
    return _overlay(runs, outdir, ["S1", "S2"],
                    ["ShareGPT (conversational)", "random (256/128 fixed)"],
                    "Effect of the input workload (Qwen2.5-7B, 1 GPU)",
                    "fig04_dataset_comparison")


def fig06(runs, outdir):
    # The throughput panel needs a log axis here, unlike in figure 4. The two
    # models' throughputs span 181-3783 tok/s (21x): on a linear axis the 7B
    # curve occupies 13 % of the panel and its saturation near 675 tok/s - half
    # of what this figure is about - is unreadable. Figure 4 keeps a linear
    # throughput axis because its two series overlap and cross near rate 5,
    # which a log axis would flatten. The choice is made per figure rather than
    # by a threshold: a cutoff that separated 14x from 21x would be arbitrary.
    return _overlay(runs, outdir, ["S1", "S3"],
                    ["Qwen2.5-7B", "Qwen2.5-0.5B"],
                    "Effect of model size (ShareGPT, 1 GPU)",
                    "fig06_model_comparison",
                    fields=[("p95_ttft_ms", "TTFT p95 (ms)"),
                            ("p95_tpot_ms", "TPOT p95 (ms/token)"),
                            ("output_throughput", "Output throughput (tok/s)")],
                    log_fields=("output_throughput",))


def fig07(runs, outdir):
    """GPU count. G1/G2/G4 are all measured on the same multi-GPU instance,
    so the only difference between the series is the parallelism setting."""
    # P1 (pp=2) uses the same two GPUs as G2 (tp=2), so the pair isolates the
    # communication pattern - all-reduce at every layer versus one activation
    # hand-off per stage boundary - rather than the device count.
    groups = [g for g in ("G1", "G2", "G4", "P1") if select(runs, group=g)]
    names = {"G1": "1 GPU (tp=1)", "G2": "2 GPUs (tp=2)",
             "G4": "4 GPUs (tp=4)", "P1": "2 GPUs (pp=2)"}
    if not groups:
        print("  SKIP fig07: no G1/G2/G4/P1 runs")
        return None
    return _overlay(runs, outdir, groups, [names[g] for g in groups],
                    "Effect of GPU count and parallelism strategy "
                    "(Qwen2.5-7B, ShareGPT, single instance)",
                    "fig07_gpu_count_comparison",
                    fields=[("p95_ttft_ms", "TTFT p95 (ms)"),
                            ("p95_tpot_ms", "TPOT p95 (ms/token)"),
                            ("output_throughput", "Output throughput (tok/s)")])


def fig10(runs, outdir):
    """Closed-loop latency-throughput curve.

    Each point fixes the number of in-flight requests, so unlike an open-loop
    rate above capacity it has a well-defined steady state. The knee of this
    curve is the defensible statement of the latency/throughput trade-off.
    """
    c2 = select(runs, group="C2")
    if not c2:
        print("  SKIP fig10: no C2 runs")
        return None
    # C2x (c=128) ran on the session C instance, not the session B one, so it
    # is drawn as a separate open-marker point rather than silently appended
    # to the C2 curve; A1d vs A1c is the anchor that justifies showing both
    # on one axis.
    c2x = select(runs, group="C2x")
    import statistics as st

    def curve(rs):
        by = {}
        for r in rs:
            by.setdefault(r["max_concurrency"], []).append(r)
        xs, ys, exs, eys, labels = [], [], [], [], []
        for c in sorted(by, key=lambda x: int(x)):
            thr = [r["bench"]["output_throughput"] for r in by[c]]
            lat = [r["bench"]["p95_e2el_ms"] / 1000 for r in by[c]]
            xs.append(st.mean(thr))
            ys.append(st.mean(lat))
            exs.append(st.stdev(thr) if len(thr) > 1 else 0.0)
            eys.append(st.stdev(lat) if len(lat) > 1 else 0.0)
            labels.append(c)
        return xs, ys, exs, eys, labels

    xs, ys, exs, eys, labels = curve(c2)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    # Error bars on both axes. The two are anti-correlated and the pair is a
    # result in itself: at concurrency 1 the throughput spread is 0.4 % but the
    # p95 latency spread is 20.6 %, and at 128 it is 12.9 % against 6.8 %. With
    # one request in flight the aggregate rate is just 1 / mean latency over 60
    # requests and is very stable, while p95 is decided by whichever long
    # ShareGPT completion happened to land in that small sample. Saturated, the
    # run's duration is set by the tail so throughput becomes the noisy axis,
    # while p95 over 200 requests is well determined. Drawing only yerr, as the
    # first version did, showed the low-concurrency noise and hid the
    # high-concurrency noise entirely.
    ebar = dict(elinewidth=1.0, capsize=2.5, alpha=0.45, zorder=1)
    ax.errorbar(xs, ys, yerr=eys, xerr=exs, color=SERIES[0], fmt="none", **ebar)
    ax.plot(xs, ys, color=SERIES[0], marker=MARKERS[0], zorder=3,
            label="C2 (session B instance)" if c2x else None)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, -10),
                    fontsize=7, color=C["grey"], zorder=4)
    if c2x:
        xx, xy, xex, xey, xl = curve(c2x)
        ax.errorbar(xx, xy, yerr=xey, xerr=xex, color=SERIES[0], fmt="none",
                    **ebar)
        ax.plot(xx, xy, color=SERIES[0], marker=MARKERS[0], mfc="white",
                linestyle="none", zorder=3,
                label="C2x (session C instance)")
        for x, y, l in zip(xx, xy, xl):
            ax.annotate(l, (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=C["grey"], zorder=4)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    # Start the throughput axis near the data rather than at 0: the leftmost
    # point is 32 tok/s and the empty strip below it carries no information.
    lo = min(xs + (xx if c2x else []))
    hi = max(xs + (xx if c2x else []))
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.10 * (hi - lo))
    ax.set_xlabel("Output throughput (tok/s)")
    ax.set_ylabel("End-to-end latency p95 (s)")
    ax.set_title("Closed-loop latency vs throughput\n"
                 "(Qwen2.5-7B, ShareGPT, 1 GPU; labels = concurrency limit)")
    return save(fig, "fig10_closed_loop_tradeoff", outdir)


ALL = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 6: fig06, 7: fig07, 10: fig10}
