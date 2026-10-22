#!/usr/bin/env python3
"""Real-trace re-validation runner for RUSA-NIDS (CICIDS2017 / CSE-CIC-IDS2018).

This is the script referenced in Section 6 (Threats to Validity) as what
still needs to run before the synthetic-only numbers in the paper can be
called deployment-validated. It reuses the exact nested train/val
discipline from scripts/calibrate_eta_urgency.py -- only the generator
changes (TraceWorkloadGenerator instead of FlowWorkloadGenerator), per
real_trace.py's drop-in design.

Fold strategy -- day-files as folds, not random splits:
CICIDS2017 ships as one CSV per capture day, each day dominated by a
different attack family (Mon=benign only; Tue=brute force; Wed=DoS/
Heartbleed; Thu=web attack/infiltration; Fri=botnet/portscan/DDoS).
CSE-CIC-IDS2018 is organized the same way across more days/attack types.
Folding by day (rather than a random row-level split) is the correct
choice here: a random split leaks near-duplicate flows from the same
attack burst across train/val, which would overstate how well a
calibrated (eta, slack) generalizes to an attack family it has never
seen -- exactly the kind of leakage Section 5.1 already commits this
paper to avoiding.

Usage (once you have the CSVs -- see --help on where to get them):
    python scripts/run_real_trace_experiment.py \
        --data-dir data/CICIDS2017 \
        --risk-col detector_prob \
        --out results/

If --risk-col is omitted, every fold runs with the label-derived
pseudo-risk fallback (real_trace.py docstring) -- useful to smoke-test
that the pipeline runs end-to-end on real traffic shapes/timing, but NOT
a result to put in the paper. Score your CSVs with the actual detector
and add that column before reporting numbers.
"""

from __future__ import annotations

import argparse
import zlib
import csv as csv_module
import glob
import itertools
import logging
import os
import statistics
from pathlib import Path

from rusa_nids.real_trace import TraceWorkloadGenerator
from rusa_nids.schedulers import RusaNidsPolicy
from rusa_nids.schedulers.rusa import RusaWeights
from rusa_nids.simulate import run_simulation

logging.disable(logging.WARNING)  # keep the pseudo-risk warning but drop per-row noise

ETA_GRID = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
SLACK_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
N_WORKERS = 2


def discover_day_files(data_dir: str) -> dict:
    """Group CSVs under data_dir by filename prefix up to the first '-' or
    '_' (CICIDS2017/CSE-CIC-IDS2018 day-file naming: "Wednesday-workingHours...csv").
    One group = one fold.
    """
    folds: dict = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = Path(path).stem
        key = name.split("-")[0].split("_")[0]
        folds.setdefault(key, []).append(path)
    return folds


def objective(eta: float, slack: float, gen: TraceWorkloadGenerator, seed: int) -> int:
    """Re-simulate against an ALREADY-LOADED generator. The CSV is parsed
    once per day (load_day_generator, below); this just rewinds the
    cursor (``reset()``) and re-runs the event loop with a fresh policy --
    the only way a 78-point grid x 7-8 folds finishes in human time
    instead of re-parsing a half-million-row day-file 500+ times.
    """
    weights = RusaWeights(eta=eta, urgency_slack_s=slack)
    policy = RusaNidsPolicy(weights=weights)
    gen.reset()
    res = run_simulation(policy=policy, n_workers=N_WORKERS, n_flows=len(gen), generator=gen, seed=seed)
    return res.dropped_deadline


def load_day_generator(csv_paths, risk_col, uncertainty_col, max_rows, seed: int) -> TraceWorkloadGenerator:
    return TraceWorkloadGenerator(
        csv_paths=csv_paths,
        risk_col=risk_col,
        uncertainty_col=uncertainty_col,
        max_rows=max_rows,
        seed=seed,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory of CICFlowMeter day-file CSVs.")
    p.add_argument("--risk-col", default=None, help="Column holding your detector's calibrated attack probability. Omit only for a pipeline smoke-test (see module docstring).")
    p.add_argument("--uncertainty-col", default=None)
    p.add_argument("--max-rows-per-day", type=int, default=4000, help="Reservoir-sample each day-file down to this many flows before simulating -- the full files (hundreds of thousands of rows each) make a 78-point grid x N-fold sweep computationally infeasible otherwise. Sampling is uniform over the whole file (Algorithm R in real_trace.py), not a head truncation.")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    folds = discover_day_files(args.data_dir)
    if len(folds) < 2:
        raise SystemExit(
            f"Found {len(folds)} day-file group(s) under {args.data_dir!r} -- "
            "need at least 2 (one held out, the rest for grid selection). "
            "Check --data-dir points at the extracted day-file CSVs."
        )

    grid = list(itertools.product(ETA_GRID, SLACK_GRID))
    fold_names = sorted(folds)

    print(f"Loading {len(fold_names)} day-file(s), up to {args.max_rows_per_day} flows each "
          f"(reservoir-sampled)...")
    generators = {}
    for name in fold_names:
        seed = zlib.crc32(name.encode()) % (2**31)
        gen = load_day_generator(folds[name], args.risk_col, args.uncertainty_col, args.max_rows_per_day, seed)
        generators[name] = gen
        print(f"  {name}: {len(gen)} flows loaded from {len(folds[name])} file(s)")

    cache = {}
    for name in fold_names:
        seed = zlib.crc32(name.encode()) % (2**31)
        for eta, slack in grid:
            cache[(name, eta, slack)] = objective(eta, slack, generators[name], seed=seed)

    fold_rows = []
    for held_out in fold_names:
        train_names = [n for n in fold_names if n != held_out]
        best_point, best_mean = None, float("inf")
        for eta, slack in grid:
            mean_dropped = statistics.mean(cache[(n, eta, slack)] for n in train_names)
            if mean_dropped < best_mean:
                best_mean, best_point = mean_dropped, (eta, slack)
        eta_sel, slack_sel = best_point
        held_out_dropped = cache[(held_out, eta_sel, slack_sel)]
        fold_rows.append({
            "held_out_day": held_out,
            "selected_eta": eta_sel,
            "selected_slack": slack_sel,
            "train_mean_dropped_deadline": round(best_mean, 2),
            "held_out_dropped_deadline": held_out_dropped,
        })

    out_path = os.path.join(args.out, "real_trace_nested_cv.csv")
    with open(out_path, "w", newline="") as f:
        w = csv_module.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    print(f"Folds (day-files): {fold_names}")
    for r in fold_rows:
        print(r)
    print(f"\nWrote {out_path}")
    if args.risk_col is None:
        print(
            "\nWARNING: ran with the label-derived pseudo-risk fallback "
            "(--risk-col not given). These numbers are a pipeline "
            "smoke-test only -- do not cite them in the paper."
        )


if __name__ == "__main__":
    main()
