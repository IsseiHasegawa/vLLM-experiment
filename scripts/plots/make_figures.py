#!/usr/bin/env python3
"""Build the report figures.

    python3 scripts/plots/make_figures.py                # all available
    python3 scripts/plots/make_figures.py --figures 1,3,8
    python3 scripts/plots/make_figures.py --repo . --outdir figures

Figures whose data is missing are skipped with a message rather than crashing,
so this can be run at any point during the campaign (including the pilot, where
only a couple of rates exist).
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import fig05_datasets  # noqa: E402
import fig08_phases  # noqa: E402
import fig09_resources  # noqa: E402
import figs_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--figures", default="all")
    ap.add_argument("--results-dir", action="append", default=None,
                    help="restrict to specific results/raw/* dirs")
    args = ap.parse_args()

    want = (set(range(1, 10)) if args.figures == "all"
            else {int(x) for x in args.figures.split(",") if x.strip()})

    print("Loading runs ...")
    runs = common.load_runs(repo=args.repo, results_dirs=args.results_dir)
    print(f"  {len(runs)} completed runs")
    common.describe(runs)
    print()

    registry = {}
    registry.update(figs_sweep.ALL)
    registry.update(fig08_phases.ALL)
    registry.update(fig09_resources.ALL)

    made, skipped, failed = [], [], []
    for n in sorted(want):
        if n == 5:
            print("fig05 ...")
            try:
                p = fig05_datasets.fig05(
                    os.path.join(args.repo, "results", "dataset_stats.json"),
                    args.outdir)
                (made if p else skipped).append(5)
            except Exception:
                traceback.print_exc()
                failed.append(5)
            continue
        fn = registry.get(n)
        if fn is None:
            continue
        print(f"fig{n:02d} ...")
        try:
            p = fn(runs, args.outdir)
            (made if p else skipped).append(n)
        except Exception:
            traceback.print_exc()
            failed.append(n)

    print(f"\nmade {sorted(made)}  skipped {sorted(skipped)}  failed {sorted(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
