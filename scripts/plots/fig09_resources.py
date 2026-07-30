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
    # Only the per-process counters are used. cpu_total and cpu_max_core are
    # host-wide (psutil sees all 96 cores of the physical machine, not the
    # container's 9 vCPUs) and are dominated by co-tenants: cpu_total is flat
    # at 5-8 % regardless of load and cpu_max_core sits near 100 % even at
    # idle. Plotting them would show two meaningless flat lines and hide the
    # server-side trend that the bottleneck analysis rests on. See D33.
    _draw(a2, _by_rate(runs, group, "cpu_server_pct"),
          "vLLM server (% of one core)", SERIES[1], MARKERS[1])
    _draw(a2, _by_rate(runs, group, "cpu_client_pct"),
          "benchmark client (% of one core)", SERIES[4], MARKERS[4], ":")
    # The container has 9 vCPUs, i.e. a 900 % ceiling. Stating it in text
    # rather than drawing the line keeps the axis readable: the server sits
    # near 40 % on 7B and 144 % on 0.5B, so a line at 900 would flatten both.
    a2.annotate("ceiling: 9 vCPU = 900 %", xy=(0.02, 0.93),
                xycoords="axes fraction", fontsize=7, color=C["grey"])
    a2.set_xlabel("Request rate (req/s)")
    a2.set_ylabel("CPU (% of one core)")
    a2.set_title("Process CPU (per-process, container-attributable)")
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
