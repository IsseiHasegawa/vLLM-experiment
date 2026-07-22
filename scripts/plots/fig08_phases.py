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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))

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
        a1.legend()
        # Percentage labels make the prefill/decode shift explicit.
        for i in range(len(names)):
            total = queued[i] + prefill[i] + decode[i]
            for val, base, col in ((prefill[i], queued[i], "prefill"),
                                   (decode[i], bot[i], "decode")):
                if val / total > 0.06:
                    a1.text(i, base + val / 2, f"{100 * val / total:.0f}%",
                            ha="center", va="center", color="white", fontsize=8)
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
        a2.bar(x, eps, 0.5, bottom=bot, label="outside prefill compute\n"
               "(HTTP, serialization, tokenization)", color=C["red"])
        for i in range(len(names2)):
            total = qs[i] + ps[i] + eps[i]
            if total:
                a2.text(i, bot[i] + eps[i] / 2, f"{100 * eps[i] / total:.0f}%",
                        ha="center", va="center", color="white", fontsize=8)
        a2.set_xticks(list(x))
        a2.set_xticklabels(names2)
        a2.set_ylabel("Client-observed TTFT (ms)")
        a2.set_title("What TTFT is made of")
        a2.legend(loc="upper left")
    else:
        a2.text(0.5, 0.5, "no TTFT decomposition data", ha="center",
                transform=a2.transAxes, color=C["grey"])
        a2.set_axis_off()

    fig.suptitle("Phase-level timing from the instrumentation "
                 "(Qwen2.5-7B, 1 GPU, rate 5)", y=1.02)
    return save(fig, "fig08_phase_breakdown", outdir)


ALL = {8: fig08}
