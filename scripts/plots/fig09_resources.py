"""Figure 9: resource utilization vs arrival rate.

Left  - GPU SM utilization and memory-controller utilization. The gap between
        them is the compute-bound / bandwidth-bound signal: decode-dominated
        load drives the memory controller far harder than the SMs.
Right - host CPU (total and the busiest core) plus the server/client split.
        This covers the assignment's "document CPU performance" item and shows
        whether the co-located benchmark client became the limiter at high rate.

Data: resources*.csv (1 Hz) sliced by the manifest window of each run, averaged
over the run and then over repetitions.
"""

import statistics as st

import matplotlib.pyplot as plt

from common import (C, MARKERS, SERIES, INF_LABEL, gpu_columns, mean_of,
                    rate_key, resource_rows, save, select, xpos)


def _by_rate(runs, group, col, gpu_agg=False):
    """[(rate, mean, stdev, n)] of a resources column, averaged per run."""
    buckets = {}
    for run in select(runs, group=group):
        rows = resource_rows(run)
        if not rows:
            continue
        if gpu_agg:
            gpus = gpu_columns(rows)
            if not gpus:
                continue
            vals = [mean_of(rows, f"{g}_{col}") for g in gpus]
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            v = st.mean(vals)  # mean across GPUs of the per-run mean
        else:
            v = mean_of(rows, col)
            if v != v:
                continue
        buckets.setdefault(str(run["request_rate"]), []).append(v)
    out = []
    for rate in sorted(buckets, key=rate_key):
        vals = buckets[rate]
        out.append((rate, st.mean(vals),
                    st.stdev(vals) if len(vals) > 1 else 0.0, len(vals)))
    return out


def _draw(ax, pts, label, color, marker, ls="-"):
    if not pts:
        return False
    xs, labels = xpos(pts)
    ax.errorbar(xs, [p[1] for p in pts], yerr=[p[2] for p in pts],
                label=label, color=color, marker=marker, linestyle=ls)
    if any(l == INF_LABEL for l in labels):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
    return True


def fig09(runs, outdir, group="S1"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    any_data = False

    # ---- GPU ---------------------------------------------------------------
    any_data |= _draw(a1, _by_rate(runs, group, "util", gpu_agg=True),
                      "SM utilization", SERIES[0], MARKERS[0])
    any_data |= _draw(a1, _by_rate(runs, group, "memutil", gpu_agg=True),
                      "memory-controller utilization", SERIES[3], MARKERS[3],
                      "--")
    a1.set_xlabel("Request rate (req/s)")
    a1.set_ylabel("Utilization (%)")
    a1.set_ylim(0, 105)
    a1.set_title("GPU: compute vs memory bandwidth")
    a1.legend()

    # ---- CPU ---------------------------------------------------------------
    _draw(a2, _by_rate(runs, group, "cpu_total"), "host CPU (mean of cores)",
          SERIES[1], MARKERS[1])
    _draw(a2, _by_rate(runs, group, "cpu_max_core"), "busiest core",
          SERIES[2], MARKERS[2], "--")
    _draw(a2, _by_rate(runs, group, "cpu_client_pct"),
          "benchmark client (% of one core)", SERIES[4], MARKERS[4], ":")
    a2.set_xlabel("Request rate (req/s)")
    a2.set_ylabel("CPU (%)")
    a2.set_title("Host CPU")
    a2.legend()

    if not any_data:
        for a in (a1, a2):
            a.clear()
            a.text(0.5, 0.5, "no resource samples found\n"
                   "(resource_logger.py must run during the session)",
                   ha="center", va="center", transform=a.transAxes,
                   color=C["grey"])
            a.set_axis_off()

    fig.suptitle("Resource utilization vs arrival rate "
                 "(Qwen2.5-7B, ShareGPT, 1 GPU)", y=1.02)
    return save(fig, "fig09_resources_vs_rate", outdir)


ALL = {9: fig09}
