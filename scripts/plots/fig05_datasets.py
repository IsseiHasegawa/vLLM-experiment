"""Figure 5: input/output token-length distributions of the two datasets.

This documents the datasets required by the assignment ("choose and document at
least two different datasets"). ShareGPT is real conversational traffic with a
heavy-tailed length distribution; `random` is synthetic and fixed-length, which
is exactly why the two produce different scheduling behaviour.

Input: results/dataset_stats.json, produced by scripts/dataset_stats.py (which
needs a tokenizer, so it runs wherever transformers is installed - the pod is
fine). This script itself needs only matplotlib.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from common import C, save


def fig05(stats_path="results/dataset_stats.json", outdir="figures"):
    p = Path(stats_path)
    if not p.exists():
        print(f"  SKIP fig05: {p} not found (run scripts/dataset_stats.py first)")
        return None
    stats = json.loads(p.read_text())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    colors = {"sharegpt": C["blue"], "random": C["orange"]}
    for name, d in stats.items():
        col = colors.get(name, C["green"])
        for ax, key in ((a1, "input_lens"), (a2, "output_lens")):
            vals = d.get(key) or []
            if not vals:
                continue
            ax.hist(vals, bins=60, histtype="step", linewidth=1.5,
                    color=col, label=f"{name} (n={len(vals)})", density=True)
    for ax, t in ((a1, "Input (prompt) length"), (a2, "Output length")):
        ax.set_xlabel("tokens")
        ax.set_ylabel("density")
        ax.set_title(t)
        ax.legend()

    # A short numeric summary is more useful in a paper than the shape alone.
    lines = []
    for name, d in stats.items():
        s = d.get("summary", {})
        if s:
            lines.append(f"{name}: input median {s.get('input_p50', '?')}, "
                         f"p95 {s.get('input_p95', '?')}; output median "
                         f"{s.get('output_p50', '?')}, p95 {s.get('output_p95', '?')}")
    if lines:
        fig.text(0.5, -0.12, "\n".join(lines), ha="center", fontsize=8,
                 color=C["grey"])

    fig.suptitle("Token-length distributions of the two workloads", y=1.02)
    return save(fig, "fig05_dataset_distributions", outdir)


ALL = {5: fig05}
