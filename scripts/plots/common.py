"""Shared loading, aggregation and styling for the figure scripts.

Dependencies: matplotlib only (plus the standard library). No pandas, so this
runs on a bare Python install.

Data model
----------
A *run* is one row of configs/matrix.csv that has been executed. Its identity is
`run_id`. Three sources are joined on it:

  configs/matrix.csv        the intended configuration (model, dataset, tp, rate, rep)
  results/manifest.csv      what actually happened (start_ts, end_ts, status, notes)
  <results>/bench/<id>.json the client-side metrics produced by `vllm bench serve`

Per-request and per-step records (phase logs) and 1 Hz resource samples carry no
run_id; they are attributed offline by slicing on wall-clock `ts` against
[start_ts, end_ts] from the manifest (decision D4).

Only manifest rows with status == "ok" are used. Boot warmups and failures are
ignored by construction.
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (LogLocator, NullFormatter,  # noqa: E402
                               ScalarFormatter)

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------

# Colour-blind safe (Okabe-Ito). Series order is deterministic so the same
# condition keeps the same colour across every figure.
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#666666",
}
SERIES = [C["blue"], C["orange"], C["green"], C["red"], C["purple"]]
MARKERS = ["o", "s", "^", "D", "v"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "errorbar.capsize": 2.5,
})

# Rate "inf" means "send all requests at once" (offline / max-throughput point).
# It is drawn at a fixed offset to the right of the largest finite rate and
# relabelled, so the finite part of the axis stays linear and readable.
INF_LABEL = "inf"


def save(fig, name: str, outdir="figures") -> Path:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    p = Path(outdir) / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_runs(repo=".", results_dirs=None, matrix="configs/matrix.csv",
              manifest="results/manifest.csv"):
    """Return {run_id: run_dict} for every completed run found on disk.

    run_dict = matrix config + manifest timing + all bench JSON fields.
    `results_dirs` defaults to every results/raw/* directory.
    """
    repo = Path(repo)
    cfg = {}
    with open(repo / matrix, newline="") as f:
        for row in csv.DictReader(f):
            cfg[row["run_id"]] = row

    # Every manifest*.csv is read, so the pilot campaign and the main
    # campaign can coexist without one hiding the other. run_ids are NOT
    # unique across campaigns (the pilot and session A share I1_rep1,
    # I2_rep1, S1_r5_rep1, S2_r5_rep1), so rows are keyed by
    # (run_id, campaign directory) recovered from result_json. Keying on
    # run_id alone let a pilot row supply the timing window for a session A
    # bench file, which silently emptied every phase-log slice for those ids.
    man = {}

    mpaths = sorted(glob.glob(str(repo / "results" / "manifest*.csv")))
    mpath = repo / manifest
    if str(mpath) not in mpaths and mpath.exists():
        mpaths.append(str(mpath))
    for mp in mpaths:
        with open(mp, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                rj = row.get("result_json") or ""
                camp = Path(rj).parent.parent.name if rj else ""
                man[(row["run_id"], camp)] = row


    if results_dirs is None:
        results_dirs = sorted(glob.glob(str(repo / "results" / "raw" / "*")))

    runs = {}
    for d in results_dirs:
        for jf in sorted(glob.glob(str(Path(d) / "bench" / "*.json"))):
            rid = Path(jf).stem
            if rid not in cfg:
                continue  # boot warmups etc.
            try:
                with open(jf) as f:
                    bench = json.load(f)
            except Exception as e:
                print(f"  WARN unreadable {jf}: {e}")
                continue
            r = dict(cfg[rid])
            r["bench"] = bench
            r["results_dir"] = d
            mrow = man.get((rid, Path(d).name))
            if mrow is None:
                # `result_json` does not always carry the campaign directory
                # (a bare filename gives an empty campaign key), so fall back
                # to the run_id when that is unambiguous. When several
                # campaigns claim the same run_id and none of them names this
                # directory, skip rather than guess: attaching the wrong
                # timing window is what D30 was about.
                cands = [v for k, v in man.items() if k[0] == rid]
                if len(cands) == 1:
                    mrow = cands[0]
                elif cands:
                    print(f"  WARN {rid}: {len(cands)} manifest rows, none for "
                          f"{Path(d).name}; skipped to avoid mis-attribution")
            if mrow is not None:
                r["start_ts"] = _num(mrow["start_ts"])
                r["end_ts"] = _num(mrow["end_ts"])
                r["notes"] = mrow.get("notes", "")
                r["command"] = mrow.get("command", "")
            elif mpaths and man:
                # Manifest exists but this run is not marked ok here -> skip.
                continue
            if rid in runs:
                print(f"  NOTE run_id {rid} found in "
                      f"{Path(runs[rid]['results_dir']).name} and "
                      f"{Path(d).name}; using {Path(d).name}")
            runs[rid] = r
    return runs


def select(runs, **kw):
    """select(runs, group="S1") -> list of runs matching all key=value pairs."""
    out = []
    for r in runs.values():
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            out.append(r)
    return out


# --------------------------------------------------------------------------
# aggregation over repetitions
# --------------------------------------------------------------------------

def rate_key(rate):
    """Sort key: finite rates ascending, 'inf' last."""
    return (1, 0.0) if str(rate) == "inf" else (0, float(rate))


def aggregate(runs_list, field, source="bench"):
    """Aggregate `field` over repetitions.

    Returns [(rate_str, mean, stdev, n), ...] sorted by rate.
    `source="bench"` reads run["bench"][field]; otherwise run[field].
    """
    buckets = {}
    for r in runs_list:
        v = r["bench"].get(field) if source == "bench" else r.get(field)
        v = _num(v)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        buckets.setdefault(str(r["request_rate"]), []).append(v)
    out = []
    for rate in sorted(buckets, key=rate_key):
        vals = buckets[rate]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        out.append((rate, st.mean(vals), sd, len(vals)))
    return out


def xpos(points, inf_x=None):
    """Map rate labels to x positions; 'inf' goes one step right of the max.

    `inf_x` lets the caller pin the offline point to a shared position. Two
    series with different rate grids (7B runs 1-8, 0.5B runs 1-32) would
    otherwise each place 'inf' one of *their* steps past *their* maximum, so
    the same condition would land at two different x positions in one panel
    and the smaller model's offline point would read as a finite rate.
    """
    finite = [float(p[0]) for p in points if p[0] != "inf"]
    if not finite:
        return [0.0 if inf_x is None else inf_x], ["inf"]
    if inf_x is None:
        step = (max(finite) - min(finite)) / max(len(finite) - 1, 1) or 1.0
        inf_x = max(finite) + step
    xs, labels = [], []
    for p in points:
        if p[0] == "inf":
            xs.append(inf_x)
            labels.append(INF_LABEL)
        else:
            xs.append(float(p[0]))
            labels.append(p[0])
    return xs, labels


def plot_series(ax, points, label, color, marker, ls="-", inf_x=None,
                set_ticks=True):
    """Draw one aggregated series with error bars (stdev over repetitions).

    `set_ticks=False` when several series share a panel: the caller then owns
    the tick set, because a per-series call would leave the last series' grid
    on the axis and hide the other series' rates.
    """
    if not points:
        return
    xs, labels = xpos(points, inf_x)
    ys = [p[1] for p in points]
    es = [p[2] for p in points]

    # The offline point is drawn detached from the finite-rate line. It is not
    # the next step of the same sweep: 'inf' sends all requests at once, so it
    # is a different arrival process, and for ShareGPT it is not even a stable
    # measurement (seed spread 41 %, D25). Joining it with a line invites two
    # misreadings. First, that the curve passed through the intervening x
    # values - in figure 6 the 7B grid stops at 8 while the shared axis runs to
    # 32, so a connecting line would sweep across rates 16 and 24 that were
    # never run for that model. Second, that the offline point is simply "the
    # curve continued", which is the reading D25 rules out.
    n_fin = sum(1 for l in labels if l != INF_LABEL)
    if n_fin:
        ax.errorbar(xs[:n_fin], ys[:n_fin], yerr=es[:n_fin], label=label,
                    color=color, marker=marker, linestyle=ls)
    if n_fin < len(xs):
        ax.errorbar(xs[n_fin:], ys[n_fin:], yerr=es[n_fin:],
                    label=None if n_fin else label, color=color,
                    marker=marker, linestyle="none")
    has_inf = any(l == INF_LABEL for l in labels)
    if has_inf and set_ticks:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
    return xs


def annotate_inf(ax, points, inf_x=None):
    """Mark the offline ('inf') point so it is not read as a finite rate."""
    if any(p[0] == "inf" for p in points):
        xs, _ = xpos(points, inf_x)
        ax.axvline(xs[-1], color=C["grey"], lw=0.6, ls=":", alpha=0.7)


def rate_axis(ax, groups_pts, inf_x):
    """One tick set covering every group's rate grid, shared 'inf' at the end.

    Overlays can mix grids - figures 6 and 9 both put 7B (1-8 req/s) against
    0.5B (1-32 req/s) - so the union spans 1 to 32 and is unreadable on a
    linear axis, where 1, 2 and 4 collide while 24 and 32 sit far apart. A log
    axis spaces them evenly. Only a power-of-two subset is labelled; the
    remaining rates get unlabelled minor ticks so the points are still
    locatable.
    """
    finite = sorted({float(p[0]) for pts in groups_pts for p in pts
                     if p[0] != "inf"})
    if not finite:
        return
    # Only switch to log when the grids really are far apart. Figures 6 and 9
    # mix 1-8 (7B) with 1-32 (0.5B) and need it; figures 4 and 7 stay on 1-8,
    # where a linear axis can label every rate including 3, 5 and 6.
    wide = max(finite) / min(finite) >= 16
    if wide:
        ax.set_xscale("log")
    major = [r for r in finite if r in (1, 2, 4, 8, 16, 32)] if wide else finite
    ax.set_xticks(major + [inf_x])
    ax.set_xticklabels([f"{r:g}" for r in major] + [INF_LABEL])
    minor = [r for r in finite if r not in major]
    if minor:
        ax.set_xticks(minor, minor=True)
        ax.set_xticklabels([], minor=True)


def log_latency(ax):
    """Put a latency axis on a log scale, with readable (non-exponent) ticks.

    Latency in these sweeps spans two orders of magnitude once the offline
    point shares the axis: on 7B/ShareGPT the p95 TTFT is 151 ms at rate 1,
    245 ms at rate 8 and 6767 ms at rate inf. On a linear axis the entire
    finite-rate range - the part that answers "how does the arrival rate
    affect latency" - is pressed into the bottom 3 % of the panel and reads as
    a flat line, so the figure appears to claim that latency is constant up to
    capacity and then explodes. It is not: p95 TTFT rises 62 % across the
    finite grid, and the ShareGPT/random gap at rate 8 (245 vs 211 ms) is a
    result of its own. A log axis raises the finite-rate share of the panel
    from 3 % to 24 %, keeps the offline point visible, and separates p50 from
    p95 (a factor of ~2) which the linear axis also hides.

    Major ticks at 1, 2 and 5 per decade so a two-decade range still carries
    enough labels to read values off the axis.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                          numticks=12))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="minor", axis="y", alpha=0.12)


# --------------------------------------------------------------------------
# phase logs and resources
# --------------------------------------------------------------------------

_JSONL_CACHE: dict = {}


def load_jsonl(patterns):
    """Load JSONL / JSONL.gz records, dropping meta headers.

    Results are cached per pattern tuple: the phase logs are read once per
    figure run instead of once per call, which matters because the window
    helpers below need the request log to locate the measured section.
    """
    key = tuple(patterns)
    if key in _JSONL_CACHE:
        return _JSONL_CACHE[key]
    recs = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if j.get("record") == "meta":
                        continue
                    recs.append(j)
    _JSONL_CACHE[key] = recs
    return recs


def _raw_phase(run, kind):
    """Every phase-log record on disk for this results directory."""
    d = Path(run["results_dir"])
    pats = [str(d / "phase_logs" / f"{kind}*.jsonl"),
            str(d / "phase_logs" / f"{kind}*.jsonl.gz"),
            str(d / f"{kind}*.jsonl"), str(d / f"{kind}*.jsonl.gz")]
    return load_jsonl(pats)


def _num_warmups(run, default=30):
    """Warm-up request count for this run, read back from the command."""
    m = re.search(r"--num-warmups\s+(\d+)", run.get("command", "") or "")
    return int(m.group(1)) if m else default


_WINDOW_CACHE: dict = {}


def measurement_window(run):
    """(t0, t1) covering only the *measured* section of a run.

    The manifest window is the lifetime of the `vllm bench serve` process, of
    which roughly 100 s is interpreter start-up, tokenizer load and warm-up
    with the GPU near idle (60 % of session A's wall clock; see D21). Averaging
    anything over that window understates utilisation, and the error grows with
    the request rate because the measured section shrinks while start-up does
    not, so utilisation appears to *fall* under load.

    The measured section is recovered from the request log: records inside the
    manifest window, ordered by completion, with the first `--num-warmups`
    dropped. Runs without a phase log (the C1off arm) fall back to
    `end_ts - duration`.
    """
    rid = run.get("run_id", "")
    key = (rid, run.get("results_dir", ""))
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]

    t0, t1 = run.get("start_ts"), run.get("end_ts")
    win = (t0, t1)
    if t0 is not None and t1 is not None:
        recs = [r for r in _raw_phase(run, "requests")
                if t0 <= r.get("ts", 0) <= t1 + 1.0]
        recs.sort(key=lambda r: r.get("ts", 0))
        recs = recs[_num_warmups(run):]
        if recs:
            win = (min(r.get("arrival_ts", r["ts"]) for r in recs),
                   max(r["ts"] for r in recs))
        else:
            dur = run.get("bench", {}).get("duration")
            if dur:
                win = (t1 - dur, t1)
    _WINDOW_CACHE[key] = win
    return win


def phase_records(run, kind="requests"):
    """Per-request (or per-step) records belonging to one run, warm-up removed.

    Implements decision D4: slice the phase log on wall-clock ts using the
    manifest window. `requests` records are timestamped at completion, so the
    window is the manifest window and the leading `--num-warmups` records are
    dropped by completion order; this yields exactly `num_prompts` records.
    `steps` records carry no request identity, so they are sliced on the
    measured window returned by measurement_window().
    """
    recs = _raw_phase(run, kind)
    t0, t1 = run.get("start_ts"), run.get("end_ts")
    if t0 is None or t1 is None:
        return recs
    if kind == "requests":
        recs = [r for r in recs if t0 <= r.get("ts", 0) <= t1 + 1.0]
        recs.sort(key=lambda r: r.get("ts", 0))
        return recs[_num_warmups(run):]
    m0, m1 = measurement_window(run)
    return [r for r in recs if m0 <= r.get("ts", 0) <= m1]


def resource_rows(run):
    """1 Hz resource samples inside this run's *measured* window.

    Not the manifest window: see measurement_window() for why that would
    understate every utilisation figure, worst at the highest rates.
    """
    d = Path(run["results_dir"])
    rows = []
    for p in sorted(glob.glob(str(d / "resources*.csv"))):
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    t0, t1 = measurement_window(run)
    if t0 is None or t1 is None:
        return rows
    return [r for r in rows if t0 <= _num(r.get("ts"), 0) <= t1]


def gpu_columns(rows):
    """Return the GPU index list present in a resources CSV."""
    if not rows:
        return []
    idx = set()
    for k in rows[0]:
        if k.startswith("gpu") and "_" in k:
            idx.add(k.split("_")[0])
    return sorted(idx)


def mean_of(rows, col, default=float("nan")):
    vals = [_num(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None and v >= 0]
    return st.mean(vals) if vals else default


# --------------------------------------------------------------------------
# reporting helper
# --------------------------------------------------------------------------

def describe(runs):
    """Print what was found, so a missing group is noticed before plotting."""
    if not runs:
        print("  (no runs found)")
        return
    by = {}
    for r in runs.values():
        by.setdefault(r["group"], []).append(r)
    for g in sorted(by):
        rates = sorted({x["request_rate"] for x in by[g]}, key=rate_key)
        reps = sorted({x["rep"] for x in by[g]})
        print(f"  {g}: {len(by[g]):3d} runs | rates {','.join(rates)} "
              f"| reps {','.join(reps)}")
