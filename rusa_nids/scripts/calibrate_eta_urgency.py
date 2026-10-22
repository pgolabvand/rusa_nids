#!/usr/bin/env python3
"""Option B calibration framework for RUSA-NIDS's (eta, urgency_slack_s) pair.

Context (see Methodology tab, "Scale-consistency fix", 2026-10-19): after
normalizing the risk x urgency term (`ru_hat`), eta has a REAL effect on
scheduling decisions for the first time -- but the paper's hand-picked
default (eta=1.5) was shown by a single-seed grid search to underperform
eta=0 in the one regime tested (workers=2). A single-seed sweep is not a
calibration: it has no notion of a train/validation split, so there is
no way to tell "the best eta for this seed" from "the best eta in
general". Section 5.1 already commits the paper to calibrating weights
"offline, on train/validation data only, never on the evaluation split" --
this script is that discipline, implemented.

What this is NOT: a replacement for calibrating on real CICIDS2017 /
CSE-CIC-IDS2018 traffic. There is no "train/validation split of reality"
to fall back on here -- only independent seeds of the same synthetic
generator, which share its cost/deadline distributions. The folds below
isolate seed-to-seed variance (which the paper's own single-seed number
hid completely), but they will NOT catch anything that depends on the
synthetic generator's distributional assumptions being wrong. Swap
`FlowWorkloadGenerator` for a real trace reader and this script's nested
train/val loop is otherwise unchanged -- that is the point of writing it
this way instead of as a one-off sweep script.

Method: nested cross-validation over K independent seeds.
  for each outer fold k (held out):
      train_seeds = all seeds except k
      for each (eta, slack) in the grid:
          score(eta, slack) = mean over train_seeds of objective(seed)
      best_(eta, slack) for this fold = argmin score
      held_out_performance = objective(best_(eta, slack), seed=k)
  report: the K selected (eta, slack) pairs (do they agree?), and the K
  held-out objective values (the honest, un-leaked generalization
  estimate -- this is the number that belongs in the paper, not the
  single best-fold number).

Objective (lower is better): dropped_deadline count. This is the
decision-relevant metric for a NIDS/NIPS scheduler (a dropped detection
deadline is a miss), not latency or throughput, which the paper already
treats as secondary (Section 5, Metrics).

Usage:
    python scripts/calibrate_eta_urgency.py --out results/
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import statistics
from dataclasses import replace

from rusa_nids.schedulers import RusaNidsPolicy
from rusa_nids.schedulers.rusa import RusaWeights
from rusa_nids.simulate import FlowWorkloadGenerator, run_simulation

logging.disable(logging.CRITICAL)

ETA_GRID = [0.0, 0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
SLACK_GRID = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]  # independent folds -- NOT real data splits
N_WORKERS = 2  # the one regime where the paper's own numbers showed a margin
N_FLOWS = 3000


def objective(eta: float, slack: float, seed: int) -> int:
    """dropped_deadline count for one (eta, slack, seed) point. Lower is better."""
    weights = RusaWeights(eta=eta, urgency_slack_s=slack)
    policy = RusaNidsPolicy(weights=weights)
    gen = FlowWorkloadGenerator(seed=seed)
    res = run_simulation(policy=policy, n_workers=N_WORKERS, n_flows=N_FLOWS,
                          generator=gen, seed=seed)
    return res.dropped_deadline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = list(itertools.product(ETA_GRID, SLACK_GRID))

    # Cache every (eta, slack, seed) evaluation once -- the grid is walked
    # once per outer fold, so without this the script is O(K) slower for
    # no benefit (the grid and seeds are fixed in advance).
    cache: dict[tuple[float, float, int], int] = {}
    for (eta, slack), seed in itertools.product(grid, SEEDS):
        cache[(eta, slack, seed)] = objective(eta, slack, seed)

    fold_rows = []
    chosen = []
    for held_out in SEEDS:
        train_seeds = [s for s in SEEDS if s != held_out]
        best_point, best_mean = None, float("inf")
        for eta, slack in grid:
            mean_dropped = statistics.mean(cache[(eta, slack, s)] for s in train_seeds)
            if mean_dropped < best_mean:
                best_mean, best_point = mean_dropped, (eta, slack)
        eta_sel, slack_sel = best_point
        held_out_dropped = cache[(eta_sel, slack_sel, held_out)]
        chosen.append(best_point)
        fold_rows.append({
            "held_out_seed": held_out,
            "selected_eta": eta_sel,
            "selected_slack": slack_sel,
            "train_mean_dropped_deadline": round(best_mean, 2),
            "held_out_dropped_deadline": held_out_dropped,
        })

    with open(os.path.join(args.out, "calibration_nested_cv.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    # Baselines for comparison, evaluated on the SAME held-out seeds (never
    # used to pick anything): the paper's original hand-picked default, and
    # the no-risk-term baseline, so the report shows calibration's actual lift.
    baseline_rows = []
    for name, (eta_b, slack_b) in [("paper_default (eta=1.5, slack=0.25)", (1.5, 0.25)),
                                     ("no_risk_term (eta=0)", (0.0, 0.25))]:
        vals = [cache[(eta_b, slack_b, s)] for s in SEEDS]
        baseline_rows.append({"config": name, "mean_dropped_deadline": round(statistics.mean(vals), 2),
                               "stdev": round(statistics.pstdev(vals), 2)})
    held_out_vals = [r["held_out_dropped_deadline"] for r in fold_rows]
    baseline_rows.append({
        "config": "nested_cv_selected (per-fold eta/slack, held-out eval)",
        "mean_dropped_deadline": round(statistics.mean(held_out_vals), 2),
        "stdev": round(statistics.pstdev(held_out_vals), 2),
    })
    with open(os.path.join(args.out, "calibration_baseline_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        w.writeheader()
        for r in baseline_rows:
            w.writerow(r)

    print("== Nested CV fold results (results/calibration_nested_cv.csv) ==")
    for r in fold_rows:
        print(r)
    print("\n== Baseline comparison, held-out-honest (results/calibration_baseline_comparison.csv) ==")
    for r in baseline_rows:
        print(r)

    eta_votes = sorted(set(c[0] for c in chosen))
    slack_votes = sorted(set(c[1] for c in chosen))
    print(f"\nSelected eta values across {len(SEEDS)} folds: {eta_votes}")
    print(f"Selected urgency_slack_s values across {len(SEEDS)} folds: {slack_votes}")
    if len(eta_votes) > 1 or len(slack_votes) > 1:
        print("WARNING: fold selections do not agree -- the synthetic generator's "
              "seed-to-seed variance is large relative to the grid spacing at "
              "workers=2/n_flows=3000. Report the held-out distribution, not a "
              "single number, and do not present one fold's winner as 'the' "
              "calibrated value without this caveat.")


if __name__ == "__main__":
    main()
