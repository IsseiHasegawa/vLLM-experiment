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

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    colors = {"sharegpt": C["blue"], "random": C["orange"]}
    for name, d in stats.items():
        col = colors.get(name, C["green"])
        for ax, key in ((a1, "input_lens"), (a2, "output_lens")):
            vals = d.get(key) or []
            if not vals:
                continue
            ax.hist(vals, bins=60, histtype="step", linewidth=1.5,
                    color=col, label=f"{name} nominal (n={len(vals)})",
                    density=True)
        rv = real.get(name)
        if rv:
            for ax, key in ((a1, "input_lens"), (a2, "output_lens")):
                ax.hist(rv[key], bins=60, histtype="stepfilled", alpha=0.35,
                        color=col, density=True,
                        label=f"{name} realised (n={len(rv[key])})")
    for ax, t in ((a1, "Input (prompt) length"), (a2, "Output length")):
        ax.set_xlabel("tokens")
        ax.set_ylabel("density")
        ax.set_title(t)
        ax.legend()

    # A short numeric summary is more useful in a paper than the shape alone.
    def q(v, f):
        v = sorted(v)
        return v[int(f * (len(v) - 1))] if v else "?"

    lines = []
    for name, d in stats.items():
        st_ = d.get("summary", {})
        if st_:
            lines.append(f"{name} nominal: input p50 {st_.get('input_p50', '?')}, "
                         f"p95 {st_.get('input_p95', '?')}, max "
                         f"{st_.get('input_max', '?')}; output p50 "
                         f"{st_.get('output_p50', '?')}, p95 "
                         f"{st_.get('output_p95', '?')}")
        rv = real.get(name)
        if rv:
            i, o = rv["input_lens"], rv["output_lens"]
            lines.append(f"{name} realised: input p50 {q(i, .5)}, p95 {q(i, .95)}, "
                         f"max {max(i)}; output p50 {q(o, .5)}, p95 {q(o, .95)}")
    if lines:
        fig.text(0.5, -0.12, "\n".join(lines), ha="center", fontsize=8,
                 color=C["grey"])

    fig.suptitle("Token-length distributions of the two workloads", y=1.02)
    return save(fig, "fig05_dataset_distributions", outdir)


ALL = {5: fig05}
