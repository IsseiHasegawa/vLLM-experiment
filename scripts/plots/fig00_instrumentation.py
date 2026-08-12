#!/usr/bin/env python3
"""Figure R1 / paper Fig 1 replacement: instrumentation diagram, flat style."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures" / "fig00_instrumentation.png"
BLACK, GRAY = "black", "dimgray"

fig, ax = plt.subplots(figsize=(8.0, 3.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)
ax.set_axis_off()

def rect(x, y, w, h, lw=1.0):
    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                           edgecolor=BLACK, linewidth=lw))

rect(3.1, 0.3, 3.8, 3.7, lw=1.2)
ax.text(5.0, 3.62, "Instrumented vLLM (fork)", ha="center", va="center",
        fontsize=10, fontweight="bold")
ax.text(5.0, 3.28, "instrumentation branch, 3 files", ha="center",
        va="center", fontsize=8, color=GRAY)

rect(3.35, 1.75, 3.3, 1.15)
ax.text(5.0, 2.55, "requests.jsonl (request axis)", ha="center",
        va="center", fontsize=9)
ax.text(5.0, 2.12, "queued / prefill / decode / e2e", ha="center",
        va="center", fontsize=8, color=GRAY)

rect(3.35, 0.5, 3.3, 1.15)
ax.text(5.0, 1.3, "steps.jsonl (step axis)", ha="center",
        va="center", fontsize=9)
ax.text(5.0, 0.87, "sched / exec / KV usage / batch", ha="center",
        va="center", fontsize=8, color=GRAY)

rect(0.25, 1.55, 2.3, 1.2)
ax.text(1.4, 2.35, "Benchmark client", ha="center", va="center", fontsize=9)
ax.text(1.4, 1.95, "controls arrival rate", ha="center", va="center",
        fontsize=8, color=GRAY)

rect(7.45, 1.55, 2.3, 1.2)
ax.text(8.6, 2.35, "Resource logger", ha="center", va="center", fontsize=9)
ax.text(8.6, 1.95, "external, 1 Hz GPU + CPU", ha="center", va="center",
        fontsize=8, color=GRAY)

ax.annotate("", xy=(3.1, 2.15), xytext=(2.55, 2.15),
            arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
ax.annotate("", xy=(6.9, 2.15), xytext=(7.45, 2.15),
            arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0,
                            linestyle=(0, (4, 3))))

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")