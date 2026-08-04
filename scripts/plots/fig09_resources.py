"""Figure 9: resource utilization vs arrival rate, 7B against 0.5B.

Left  - GPU SM utilization and memory-controller utilization for both models.
        On 7B the two sit near 90 % with the memory controller the harder
        pressed of the pair, which is the bandwidth-bound signal. On 0.5B both
        fall to roughly half that and flatten, so no GPU resource is the limit.
Right - per-process CPU for the vLLM server and the co-located benchmark
        client. The 7B server settles below half a core while the 0.5B server
        passes a full core and keeps climbing.

The two models share both panels because the claim in section 5.5 is about the
limiting resource *changing* with model size, not about either model alone:
moving from 7B to 0.5B pushes the GPU curves down and the CPU curve up in the
same figure. The two rate grids differ (7B 1-8 req/s, 0.5B 1-32), so the x axis
is the union of the two on a log scale with a shared offline point; see
common.rate_axis.

Data: resources*.csv (1 Hz) sliced by the manifest window of each run, averaged
over the run and then over repetitions.
"""

import statistics as st

import matplotlib.pyplot as plt

from common import (C, MARKERS, SERIES, annotate_inf, gpu_columns, mean_of,
                    plot_series, rate_axis, rate_key, resource_rows, save,
                    select)

# Model -> colour, matching figure 6 so the same model keeps the same colour
# across the report. Metric is encoded by line style and marker instead, which
# leaves the four curves per panel separable on both channels.
MODELS = (("S1", "7B", SERIES[0]), ("S3", "0.5B", SERIES[1]))
GPU_METRICS = (("util", "SM", MARKERS[0], "-"),
               ("memutil", "memory controller", MARKERS[1], "--"))
CPU_METRICS = (("cpu_server_pct", "vLLM server", MARKERS[0], "-"),
               ("cpu_client_pct", "benchmark client", MARKERS[1], "--"))


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


def _panel(ax, runs, metrics, inf_x, gpu_agg=False):
    """Draw every (model, metric) pair on one panel. True if anything drew."""
    drawn = []
    for group, model, colour in MODELS:
        for col, metric, marker, ls in metrics:
            pts = _by_rate(runs, group, col, gpu_agg=gpu_agg)
            if not pts:
                continue  # missing metric: leave the gap, do not interpolate
            # set_ticks=False because rate_axis below owns the tick set; a
            # per-series call would leave only the last grid on the axis.
            plot_series(ax, pts, f"{model}: {metric}", colour, marker, ls,
                        inf_x=inf_x, set_ticks=False)
            drawn.append(pts)
    if not drawn:
        return False
    # annotate_inf reads the offline point off the end of the series, so it
    # needs one series that actually has one rather than the concatenation.
    with_inf = next((p for p in drawn if any(q[0] == "inf" for q in p)), None)
    if with_inf:
        annotate_inf(ax, with_inf, inf_x)
    rate_axis(ax, drawn, inf_x)
    return True


def fig09(runs, outdir, models=MODELS):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    # One offline position for the whole figure, derived from the widest grid,
    # so both models' 'inf' points land on the same x. Without this the 0.5B
    # offline point would sit one of its own steps past 32 while the 7B one sat
    # one step past 8, and the latter would read as a finite rate near 9.
    probe = [_by_rate(runs, g, "util", gpu_agg=True) for g, _, _ in models]
    finite = [float(p[0]) for pts in probe for p in pts if p[0] != "inf"]
    inf_x = max(finite) * 1.35 if finite else 1.0

    # ---- GPU ---------------------------------------------------------------
    any_data = _panel(a1, runs, GPU_METRICS, inf_x, gpu_agg=True)
    a1.set_xlabel("Request rate (req/s)")
    a1.set_ylabel("Utilization (%)")
    a1.set_ylim(0, 105)
    a1.set_title("GPU: compute vs memory bandwidth")
    a1.legend(fontsize=7.5)

    # ---- CPU ---------------------------------------------------------------
    # Only the per-process counters are used. cpu_total and cpu_max_core are
    # host-wide (psutil sees all 96 cores of the physical machine, not the
    # container's 9 vCPUs) and are dominated by co-tenants: cpu_total is flat
    # at 5-8 % regardless of load and cpu_max_core sits near 100 % even at
    # idle. Plotting them would show two meaningless flat lines and hide the
    # server-side trend that the bottleneck analysis rests on. See D33.
    any_data |= _panel(a2, runs, CPU_METRICS, inf_x)
    # The container has 9 vCPUs, i.e. a 900 % ceiling, which is left out of the
    # panel: a line 6x above the highest curve would flatten everything below
    # it. The earlier version stated the headroom as a text annotation, but
    # that was computed for a single group and would now report one model's
    # peak while showing two. The per-model peaks (39.6 % on 7B, 144.4 % on
    # 0.5B, both against 900 %) are stated in section 5.5 instead.
    # Subtract ~2-3 % from the server series for the logger's own cost (D36).
    a2.set_xlabel("Request rate (req/s)")
    a2.set_ylabel("CPU (% of one core)")
    a2.set_title("Process CPU (per-process, container-attributable)")
    a2.legend(fontsize=7.5)

    if not any_data:
        for a in (a1, a2):
            a.clear()
            a.text(0.5, 0.5, "no resource samples found\n"
                   "(resource_logger.py must run during the session)",
                   ha="center", va="center", transform=a.transAxes,
                   color=C["grey"])
            a.set_axis_off()

    fig.suptitle("Resource utilization vs arrival rate "
                 "(Qwen2.5-7B vs Qwen2.5-0.5B, ShareGPT, 1 GPU)", y=1.02)
    return save(fig, "fig09_resources_vs_rate", outdir)


ALL = {9: fig09}
