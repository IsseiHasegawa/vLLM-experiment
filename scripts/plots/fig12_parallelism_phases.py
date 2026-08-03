"""Figure 12: which phase each parallelism strategy actually speeds up.

Figure 7 shows that tensor and pipeline parallelism improve different metrics -
TP moves throughput and TPOT, PP moves TTFT and almost nothing else. This
figure says why, using the per-request phase records that only the custom
instrumentation provides.

Left panel   - mean prefill and decode time per request at rate 5, one bar pair
               per configuration, each labelled with its change against tp=1.
               The two phases respond to the two strategies in opposite ways.
Right panel  - the same two phases as a percentage change against tp=1 across
               the whole rate grid, so the pattern can be seen to hold under
               load rather than at one operating point.

The mechanism the panels support: a decode step produces one token per request,
and under pipeline parallelism that token must cross stage 0 and then stage 1
in sequence, so there is no parallelism *within* a step and decode time barely
moves. Tensor parallelism splits every layer, so both GPUs work on the same
token at once and the aggregate memory bandwidth - which is what decode is
bound by - doubles. Prefill is different: it processes hundreds of tokens at a
time, so pipeline stages have work to overlap and PP does help there. Because
decode is ~98% of a request (figure 8), only the strategy that speeds up decode
moves throughput.

P1 (pp=2) was measured on a session D instance whose GPU0-GPU1 pair is PXB on
one NUMA node, the same interconnect class session C used for G2 (tp=2); see
results/raw/sessionD/SESSION_NOTE.txt.

Data: requests-*.jsonl(.gz) sliced by the manifest window (D4, D28).
"""

import statistics as st

import matplotlib.pyplot as plt

from common import C, phase_records, save, select

GROUPS = [("G1", "tp=1"), ("G2", "tp=2"), ("G4", "tp=4"), ("P1", "pp=2")]
RATES = ["1", "2", "3", "4", "5", "6", "8"]


def _phases(runs, group, rate=None):
    """(mean prefill ms, mean decode ms, n) over the requests of a group."""
    p, d = [], []
    for run in select(runs, group=group):
        if rate is not None and str(run.get("request_rate")) != str(rate):
            continue
        for r in phase_records(run, "requests"):
            p.append(r["prefill_s"] * 1000)
            d.append(r["decode_s"] * 1000)
    if not p:
        return None
    return st.mean(p), st.mean(d), len(p)


def fig12(runs, outdir, rate="5"):
    have = [(g, lab) for g, lab in GROUPS if select(runs, group=g)]
    if len(have) < 2:
        print("  SKIP fig12: need at least two of G1/G2/G4/P1")
        return None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # ---- left: prefill and decode side by side, one pair per config --------
    labels, pre, dec = [], [], []
    for g, lab in have:
        r = _phases(runs, g, rate)
        if r is None:
            continue
        labels.append(lab)
        pre.append(r[0])
        dec.append(r[1])
    if not labels:
        print(f"  SKIP fig12: no phase records at rate {rate}")
        plt.close(fig)
        return None
    x = range(len(labels))
    width = 0.36
    a1.bar([i - width / 2 for i in x], pre, width, color=C["blue"],
           label="prefill")
    a1.bar([i + width / 2 for i in x], dec, width, color=C["orange"],
           label="decode")
    # Decode is two orders of magnitude larger than prefill, so a linear axis
    # would hide the prefill bars entirely (the mistake figure 8's first
    # version made). Log keeps both visible and makes the ratios readable.
    a1.set_yscale("log")
    for i, (p, d) in enumerate(zip(pre, dec)):
        a1.annotate("baseline" if i == 0 else f"{100 * (p / pre[0] - 1):+.0f}%",
                    (i - width / 2, p), ha="center", va="bottom", fontsize=7,
                    color=C["grey"])
        a1.annotate("baseline" if i == 0 else f"{100 * (d / dec[0] - 1):+.0f}%",
                    (i + width / 2, d), ha="center", va="bottom", fontsize=7,
                    color=C["grey"])
    a1.set_xticks(list(x))
    a1.set_xticklabels(labels)
    a1.set_ylabel("Mean time per request (ms, log scale)")
    a1.set_title(f"Prefill and decode by strategy (rate {rate})")
    # Headroom above the tallest bar so the legend does not sit on it, and
    # the legend itself goes below the axes for the same reason it does on
    # figures 8 and 11: the bar tops and the value labels own the upper edge.
    a1.set_ylim(top=max(dec) * 3)
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=2)

    # ---- right: percentage change against tp=1 across the rate grid --------
    base = {r: _phases(runs, "G1", r) for r in RATES}
    styles = {"tp=2": ("-", "s"), "tp=4": ("-", "^"), "pp=2": ("--", "D")}
    colors = {"tp=2": C["orange"], "tp=4": C["green"], "pp=2": C["red"]}
    for g, lab in have:
        if lab == "tp=1":
            continue
        xs, yp, yd = [], [], []
        for r in RATES:
            cur, ref = _phases(runs, g, r), base.get(r)
            if cur is None or ref is None:
                continue
            xs.append(r)
            yp.append(100 * (cur[0] / ref[0] - 1))
            yd.append(100 * (cur[1] / ref[1] - 1))
        if not xs:
            continue
        ls, mk = styles.get(lab, ("-", "o"))
        a2.plot(xs, yp, color=colors.get(lab, C["grey"]), marker=mk,
                linestyle=ls, markerfacecolor="white",
                label=f"{lab} prefill")
        a2.plot(xs, yd, color=colors.get(lab, C["grey"]), marker=mk,
                linestyle=ls, label=f"{lab} decode")
    a2.axhline(0, color=C["grey"], lw=0.8, ls=":")
    a2.set_xlabel("Request rate (req/s)")
    a2.set_ylabel("Change vs tp=1 (%)")
    a2.set_title("Hollow = prefill, filled = decode")
    # Six entries in the lower-left corner crossed the tp=4 prefill line,
    # which runs at -58 % there. Below the axes instead, in three columns.
    a2.legend(fontsize=6, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))

    fig.suptitle("Which phase each parallelism strategy speeds up "
                 "(Qwen2.5-7B, ShareGPT)")
    fig.tight_layout()
    return save(fig, "fig12_parallelism_phases", outdir)


ALL = {12: fig12}
