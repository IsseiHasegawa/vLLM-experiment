"""Figure 5: input/output token-length distributions of the two datasets.

This documents the datasets required by the assignment ("choose and document at
least two different datasets"). ShareGPT is real conversational traffic with a
heavy-tailed length distribution; `random` is synthetic and fixed-length, which
is exactly why the two produce different scheduling behaviour.

Two distributions are drawn per dataset (decision D26):

  nominal  - what is in the source file, from results/dataset_stats.json
             (scripts/dataset_stats.py; needs a tokenizer, so it runs on the pod)
  realised - the prompt/output lengths actually served, taken from the phase
             logs of the executed runs

They differ substantially for ShareGPT: the harness admits only prompts of about
1024 tokens or fewer, so the source file's 66 076-token tail is never served.
Plotting the nominal distribution alone would overstate the offered workload by
more than an order of magnitude in the tail, so both are shown and the caption
must say which is which.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from common import C, load_runs, phase_records, save, select


# Which executed group represents each dataset's realised distribution.
REALISED_FROM = {"sharegpt": "S1", "random": "S2"}


def _realised(repo="."):
    """{dataset: {"input_lens": [...], "output_lens": [...]}} from phase logs."""
    out = {}
    try:
        runs = load_runs(repo=repo)
    except Exception as e:  # pragma: no cover - missing results is normal
        print(f"  NOTE fig05: no realised lengths ({e})")
        return out
    for name, group in REALISED_FROM.items():
        recs = []
        for run in select(runs, group=group):
            recs += phase_records(run, "requests")
        if recs:
            out[name] = {"input_lens": [r["n_prompt"] for r in recs],
                         "output_lens": [r["n_gen"] for r in recs],
                         "n_runs": len(select(runs, group=group))}
    return out


def fig05(stats_path="results/dataset_stats.json", outdir="figures", repo="."):
    p = Path(stats_path)
    if not p.exists():
        print(f"  SKIP fig05: {p} not found (run scripts/dataset_stats.py first)")
        return None
    stats = json.loads(p.read_text())
    real = _realised(repo)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    colors = {"sharegpt": C["blue"], "random": C["orange"]}

    # Cumulative distributions on a log x-axis, not density histograms. A
    # histogram cannot show these two workloads on one pair of axes: `random`
    # is a fixed length, so its density is a delta spike that owns the entire
    # y-axis and flattens the ShareGPT curve to invisibility (in the first
    # version of this figure only one of the four series was actually
    # visible), while the nominal ShareGPT tail at 66 076 tokens stretches the
    # x-axis until the realised data - the part that was actually served -
    # occupies 1.5 % of the panel. A CDF has a bounded y-axis, renders a fixed
    # length as a clean step, and on a log x-axis carries four orders of
    # magnitude at once.
    def cdf(ax, vals, color, ls, label, lw=1.6, alpha=1.0, z=2):
        vals = sorted(v for v in vals if v > 0)
        if not vals:
            return
        n = len(vals)
        ax.step(vals, [(i + 1) / n for i in range(n)], where="post",
                color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                zorder=z, label=label)

    for name, d in stats.items():
        col = colors.get(name, C["green"])
        for ax, key in ((a1, "input_lens"), (a2, "output_lens")):
            cdf(ax, d.get(key) or [], col, "-", f"{name} nominal (source file)")
        rv = real.get(name)
        if rv:
            for ax, key in ((a1, "input_lens"), (a2, "output_lens")):
                cdf(ax, rv[key], col, "--", f"{name} realised (served)",
                    lw=3.2, alpha=0.45, z=1)

    for ax, t in ((a1, "Input (prompt) length"), (a2, "Output length")):
        ax.set_xscale("log")
        ax.set_xlabel("tokens (log scale)")
        ax.set_ylabel("fraction of requests <= x")
        ax.set_ylim(0, 1.02)
        ax.set_title(t)
        ax.grid(True, which="minor", alpha=0.12)
        for level in (0.5, 0.95):
            ax.axhline(level, color=C["grey"], lw=0.6, ls=":", alpha=0.7)
        ax.legend(fontsize=7, loc="upper left", frameon=False)

    # The harness admits prompts of roughly 1024 tokens or fewer (D26). This is
    # where the nominal and realised ShareGPT curves separate, and it is why
    # plotting the nominal distribution alone overstates the served tail by 65x.
    a1.axvline(1024, color=C["grey"], lw=0.8, ls="-.", alpha=0.8)
    # Label the cutoff with rotated text beside the line rather than with a
    # leader arrow: an arrow from any free area to the separation point at
    # (1024, ~0.96) has to cross the panel diagonally and clips the legend.
    # Below y~0.9 the region right of x~300 is empty, so the label sits there.
    a1.text(1180, 0.06, "harness admission cutoff ~1024 tokens",
            rotation=90, va="bottom", ha="left", fontsize=7, color=C["grey"])
    # Only the prompt length is filtered; on the output side nominal and
    # realised lie on top of each other, which is itself the statement. Placed
    # bottom-left, clear of the legend in the upper-left corner.
    a2.annotate("no filtering on the output side:\n"
                "nominal and realised coincide",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7,
                color=C["grey"])

    # A short numeric summary is more useful in a paper than the shape alone.
    def q(v, f):
        v = sorted(v)
        return v[int(f * (len(v) - 1))] if v else "?"

    lines = []
    for name, d in stats.items():
        st_ = d.get("summary", {})
        rv = real.get(name)
        if st_:
            lines.append(
                f"{name:9s} nominal   in p50 {st_.get('input_p50','?'):>5} "
                f"p95 {st_.get('input_p95','?'):>5} max {st_.get('input_max','?'):>6}"
                f"  |  out p50 {st_.get('output_p50','?'):>5} "
                f"p95 {st_.get('output_p95','?'):>5}")
        if rv:
            i, o = rv["input_lens"], rv["output_lens"]
            lines.append(
                f"{name:9s} realised  in p50 {q(i,.5):>5} p95 {q(i,.95):>5} "
                f"max {max(i):>6}  |  out p50 {q(o,.5):>5} p95 {q(o,.95):>5} "
                f"max {max(o):>6}")
    if lines:
        fig.subplots_adjust(bottom=0.34)
        fig.text(0.5, 0.015, "\n".join(lines), ha="center", va="bottom",
                 fontsize=7, family="monospace", color=C["grey"])

    fig.suptitle("Token-length distributions of the two workloads: source file vs actually served")
    return save(fig, "fig05_dataset_distributions", outdir)


ALL = {5: fig05}
