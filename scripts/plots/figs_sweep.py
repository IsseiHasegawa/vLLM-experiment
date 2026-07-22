"""Figures 1-4, 6, 7: request-rate sweeps and their overlays.

  fig01  rate -> TTFT (p50, p95)                       S1
  fig02  rate -> TPOT and ITL (p50, p95)               S1
  fig03  rate -> throughput (requests/s, tokens/s)     S1
  fig04  rate -> TTFT p95 and throughput, ShareGPT vs random   S1 vs S2
  fig06  rate -> TTFT p95, TPOT p95 and throughput, 7B vs 0.5B S1 vs S3
  fig07  rate -> throughput and TTFT p95, 1 GPU vs tp=2        S1 vs S4

Error bars are the standard deviation over the 3 repetitions of each point
(seed is fixed, so they reflect system noise only; decision D6).
"""

import matplotlib.pyplot as plt

from common import (C, MARKERS, SERIES, aggregate, annotate_inf, plot_series,
                    save, select)


def _panel(ax, runs, group, field, label, color, marker, ls="-"):
    pts = aggregate(select(runs, group=group), field)
    plot_series(ax, pts, label, color, marker, ls)
    annotate_inf(ax, pts)
    return pts


def fig01(runs, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _panel(ax, runs, "S1", "p50_ttft_ms", "p50", SERIES[0], MARKERS[0])
    _panel(ax, runs, "S1", "p95_ttft_ms", "p95", SERIES[1], MARKERS[1], "--")
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to first token vs arrival rate\nQwen2.5-7B, ShareGPT, 1 GPU")
    ax.legend(title="percentile")
    return save(fig, "fig01_ttft_vs_rate", outdir)


def fig02(runs, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    _panel(a1, runs, "S1", "p50_tpot_ms", "p50", SERIES[0], MARKERS[0])
    _panel(a1, runs, "S1", "p95_tpot_ms", "p95", SERIES[1], MARKERS[1], "--")
    a1.set_ylabel("TPOT (ms/token)")
    a1.set_title("Time per output token")
    _panel(a2, runs, "S1", "p50_itl_ms", "p50", SERIES[0], MARKERS[0])
    _panel(a2, runs, "S1", "p95_itl_ms", "p95", SERIES[1], MARKERS[1], "--")
    a2.set_ylabel("ITL (ms)")
    a2.set_title("Inter-token latency")
    for a in (a1, a2):
        a.set_xlabel("Request rate (req/s)")
        a.legend(title="percentile")
    fig.suptitle("Decode-side latency vs arrival rate (Qwen2.5-7B, ShareGPT, 1 GPU)",
                 y=1.02)
    return save(fig, "fig02_decode_latency_vs_rate", outdir)


def fig03(runs, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    pts = _panel(a1, runs, "S1", "request_throughput", "achieved",
                 SERIES[0], MARKERS[0])
    # Reference line: achieved == requested. Departure marks saturation.
    finite = [(float(p[0]), p[1]) for p in pts if p[0] != "inf"]
    if finite:
        xs = [x for x, _ in finite]
        a1.plot(xs, xs, color=C["grey"], lw=0.8, ls=":", label="requested")
    a1.set_ylabel("Request throughput (req/s)")
    a1.set_title("Achieved vs requested rate")
    a1.legend()
    _panel(a2, runs, "S1", "output_throughput", "output tokens",
           SERIES[2], MARKERS[2])
    _panel(a2, runs, "S1", "total_token_throughput", "total tokens",
           SERIES[3], MARKERS[3], "--")
    a2.set_ylabel("Token throughput (tok/s)")
    a2.set_title("Token throughput")
    a2.legend()
    for a in (a1, a2):
        a.set_xlabel("Request rate (req/s)")
    fig.suptitle("Throughput and saturation (Qwen2.5-7B, ShareGPT, 1 GPU)", y=1.02)
    return save(fig, "fig03_throughput_vs_rate", outdir)


def _overlay(runs, outdir, groups, names, title, name, fields=None):
    """Two-panel overlay: p95 TTFT and output throughput for each group."""
    fields = fields or [("p95_ttft_ms", "TTFT p95 (ms)"),
                        ("output_throughput", "Output throughput (tok/s)")]
    fig, axes = plt.subplots(1, len(fields), figsize=(4.2 * len(fields), 3.4))
    if len(fields) == 1:
        axes = [axes]
    for ax, (field, ylabel) in zip(axes, fields):
        for i, (g, nm) in enumerate(zip(groups, names)):
            _panel(ax, runs, g, field, nm, SERIES[i], MARKERS[i],
                   "-" if i == 0 else "--")
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(title, y=1.02)
    return save(fig, name, outdir)


def fig04(runs, outdir):
    return _overlay(runs, outdir, ["S1", "S2"],
                    ["ShareGPT (conversational)", "random (256/128 fixed)"],
                    "Effect of the input workload (Qwen2.5-7B, 1 GPU)",
                    "fig04_dataset_comparison")


def fig06(runs, outdir):
    return _overlay(runs, outdir, ["S1", "S3"],
                    ["Qwen2.5-7B", "Qwen2.5-0.5B"],
                    "Effect of model size (ShareGPT, 1 GPU)",
                    "fig06_model_comparison",
                    fields=[("p95_ttft_ms", "TTFT p95 (ms)"),
                            ("p95_tpot_ms", "TPOT p95 (ms/token)"),
                            ("output_throughput", "Output throughput (tok/s)")])


def fig07(runs, outdir):
    return _overlay(runs, outdir, ["S1", "S4"],
                    ["1 GPU", "2 GPUs (tensor parallel)"],
                    "Effect of GPU count and tensor parallelism "
                    "(Qwen2.5-7B, ShareGPT)",
                    "fig07_gpu_count_comparison")


ALL = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 6: fig06, 7: fig07}
