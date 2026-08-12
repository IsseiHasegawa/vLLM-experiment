"""Figure 8: where a request's time actually goes.

Left panel  - stacked mean time per request (queued / prefill / decode) for the
              prefill-heavy (I1: 512 in, 128 out) and decode-heavy (I2: 128 in,
              512 out) instrumentation runs. This is the assignment's core
              "time spent in the prefill phase and the decode phase" figure and
              it can only be drawn from the custom instrumentation.
Right panel - decomposition of client-observed TTFT into server-side queueing,
              server-side prefill, and the remainder (HTTP, serialization,
              tokenization, streaming). Session 0 showed the remainder was 62%
              of TTFT for a 0.5B model, so this panel is a result in itself.

Data: requests-*.jsonl(.gz) sliced by the manifest window (decision D4), plus
mean_ttft_ms from the corresponding bench JSON.
"""

import statistics as st

import matplotlib.pyplot as plt

from common import C, phase_records, save, select


def _phase_means(runs, group):
    """Mean queued/prefill/decode over every request of every rep in a group."""
    q, p, d, n = [], [], [], 0
    for run in select(runs, group=group):
        for r in phase_records(run, "requests"):
            q.append(r["queued_s"] * 1000)
            p.append(r["prefill_s"] * 1000)
            d.append(r["decode_s"] * 1000)
            n += 1
    if not n:
        return None
    return st.mean(q), st.mean(p), st.mean(d), n


def _ttft_split(runs, group):
    """(client TTFT, server queued, server prefill) in ms, averaged over reps."""
    client, q, p = [], [], []
    for run in select(runs, group=group):
        v = run["bench"].get("mean_ttft_ms")
        if v is None:
            continue
        recs = phase_records(run, "requests")
        if not recs:
            continue
        client.append(v)
        q.append(st.mean(r["queued_s"] for r in recs) * 1000)
        p.append(st.mean(r["prefill_s"] for r in recs) * 1000)
    if not client:
        return None
    return st.mean(client), st.mean(q), st.mean(p)


def fig08(runs, outdir, groups=("I1", "I2"),
          labels=("prefill-heavy\n512 in / 128 out",
                  "decode-heavy\n128 in / 512 out")):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.4))

    # ---- left: stacked phase breakdown -------------------------------------
    names, queued, prefill, decode = [], [], [], []
    for g, lab in zip(groups, labels):
        m = _phase_means(runs, g)
        if m is None:
            continue
        names.append(f"{lab}\n(n={m[3]})")
        queued.append(m[0])
        prefill.append(m[1])
        decode.append(m[2])
    if names:
        x = range(len(names))
        b1 = a1.bar(x, queued, 0.5, label="queued", color=C["grey"])
        b2 = a1.bar(x, prefill, 0.5, bottom=queued, label="prefill",
                    color=C["blue"])
        bot = [q + p for q, p in zip(queued, prefill)]
        b3 = a1.bar(x, decode, 0.5, bottom=bot, label="decode", color=C["orange"])
        a1.set_xticks(list(x))
        a1.set_xticklabels(names)
        a1.set_ylabel("Mean time per request (ms)")
        a1.set_title("Per-request phase breakdown")
        # Below the axes: the annotations for the invisible segments sit above
        # the bars, which is where an in-axes legend would have to go.
        a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7,
                  frameon=False, ncol=3)
        top = max(q + pp + d for q, pp, d in zip(queued, prefill, decode))
        a1.set_ylim(0, top * 1.32)
        # Decode is 98-100 % of the request, so the queued and prefill segments
        # cannot be seen at all: three colours are in the legend and one is
        # visible. That *is* the result - even the "prefill-heavy" shape (4x
        # the input, a quarter of the output) spends 98 % of the request in
        # decode - but a bar the reader cannot see does not report a number, so
        # each segment is labelled with its absolute value as well.
        for i in range(len(names)):
            total = queued[i] + prefill[i] + decode[i]
            a1.text(i, bot[i] + decode[i] / 2,
                    f"decode\n{decode[i]:,.0f} ms\n{100 * decode[i] / total:.1f}%",
                    ha="center", va="center", color="white", fontsize=8)
            a1.annotate(f"queued {queued[i]:.1f} ms ({100*queued[i]/total:.2f}%)\n"
                        f"prefill {prefill[i]:.0f} ms ({100*prefill[i]/total:.2f}%)",
                        xy=(i, bot[i]), xytext=(i, top * 1.10),
                        ha="center", va="bottom", fontsize=7, color=C["grey"],
                        arrowprops=dict(arrowstyle="->", color=C["grey"],
                                        lw=0.7))
    else:
        a1.text(0.5, 0.5, "no I1/I2 phase records", ha="center",
                transform=a1.transAxes, color=C["grey"])
        a1.set_axis_off()

    # ---- right: TTFT decomposition -----------------------------------------
    names2, qs, ps, eps = [], [], [], []
    for g, lab in zip(groups, labels):
        s = _ttft_split(runs, g)
        if s is None:
            continue
        client, q, p = s
        names2.append(lab)
        qs.append(q)
        ps.append(p)
        eps.append(max(client - q - p, 0.0))
    if names2:
        x = range(len(names2))
        a2.bar(x, qs, 0.5, label="server: queued", color=C["grey"])
        a2.bar(x, ps, 0.5, bottom=qs, label="server: prefill", color=C["blue"])
        bot = [a + b for a, b in zip(qs, ps)]
        a2.bar(x, eps, 0.5, bottom=bot,
               label="outside prefill compute (HTTP, serialization, "
                     "tokenization)", color=C["red"])
        for i in range(len(names2)):
            total = qs[i] + ps[i] + eps[i]
            if total:
                a2.text(i, bot[i] + eps[i] / 2, f"{100 * eps[i] / total:.0f}%",
                        ha="center", va="center", color="white", fontsize=8)
        a2.set_xticks(list(x))
        a2.set_xticklabels(names2)
        a2.set_ylabel("Client-observed TTFT (ms)")
        a2.set_title("What TTFT is made of")
        # Below the axes, not inside it: the three-line label for the residual
        # sat on top of the prefill-heavy bar and collided with its own
        # percentage annotation.
        a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=7, frameon=False, ncol=1)
        a2.set_ylim(0, max(q + pp + e for q, pp, e in zip(qs, ps, eps)) * 1.12)
    else:
        a2.text(0.5, 0.5, "no TTFT decomposition data", ha="center",
                transform=a2.transAxes, color=C["grey"])
        a2.set_axis_off()

    fig.suptitle("Phase-level timing from the instrumentation "
                 "(Qwen2.5-7B, 1 GPU, rate 5)")
    fig.tight_layout()
    return save(fig, "fig08_phase_breakdown", outdir)

def fig13(runs, outdir, groups=("I1", "I2"),
          labels=("prefill-heavy\n512 in / 128 out",
                  "decode-heavy\n128 in / 512 out")):
    """Right panel of fig08 as a standalone figure, for the short report."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    names, qs, ps, eps = [], [], [], []
    for g, lab in zip(groups, labels):
        s = _ttft_split(runs, g)
        if s is None:
            continue
        client, q, p = s
        names.append(lab)
        qs.append(q)
        ps.append(p)
        eps.append(max(client - q - p, 0.0))

    if not names:
        ax.text(0.5, 0.5, "no TTFT decomposition data", ha="center",
                transform=ax.transAxes, color=C["grey"])
        ax.set_axis_off()
        fig.tight_layout()
        return save(fig, "fig13_ttft_decomposition", outdir)

    x = range(len(names))
    ax.bar(x, qs, 0.5, label="server: queued", color=C["grey"])
    ax.bar(x, ps, 0.5, bottom=qs, label="server: prefill", color=C["blue"])
    bot = [a + b for a, b in zip(qs, ps)]
    ax.bar(x, eps, 0.5, bottom=bot,
           label="outside the server-recorded interval", color=C["red"])
    for i in range(len(names)):
        total = qs[i] + ps[i] + eps[i]
        if total:
            ax.text(i, bot[i] + eps[i] / 2, f"{100 * eps[i] / total:.0f}%",
                    ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Client-observed TTFT (ms)")
    ax.set_title("What the client's wait is made of\n"
                 "(Qwen2.5-7B, 1 GPU, rate 5)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              fontsize=8, frameon=False, ncol=1)
    ax.set_ylim(0, max(q + pp + e for q, pp, e in zip(qs, ps, eps)) * 1.12)

    fig.tight_layout()
    return save(fig, "fig13_ttft_decomposition", outdir)

ALL = {8: fig08, 13: fig13}
