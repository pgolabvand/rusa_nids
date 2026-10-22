#!/usr/bin/env python3
"""Dump the full (day, eta, slack) -> dropped_deadline cache for the
forest-scored 15k real-trace data, then show, for each held-out day, the
top-5 (eta, slack) pairs by training-fold mean (the same selection logic
run_real_trace_experiment.py uses), so we can see WHY a given pair won --
including near-ties -- rather than only the single argmin.

Reuses the exact same objective()/generator-reuse pattern as
run_real_trace_experiment.py; same grids, same seeds.
"""
from __future__ import annotations

import csv
import itertools
import logging
import statistics
import sys
import os

sys.path.insert(0, os.getcwd())
logging.disable(logging.WARNING)

from scripts.run_real_trace_experiment import (
    ETA_GRID, SLACK_GRID, N_WORKERS, discover_day_files, objective, load_day_generator,
)

DATA_DIR = "../datasets/scored_15k_forest"
MAX_ROWS = 15000


def main():
    folds = discover_day_files(DATA_DIR)
    fold_names = sorted(folds)
    grid = list(itertools.product(ETA_GRID, SLACK_GRID))

    generators = {}
    for name in fold_names:
        seed = zlib.crc32(name.encode()) % (2**31)
        generators[name] = load_day_generator(folds[name], "detector_prob", "detector_uncertainty", MAX_ROWS, seed)

    cache = {}
    for name in fold_names:
        seed = zlib.crc32(name.encode()) % (2**31)
        for eta, slack in grid:
            cache[(name, eta, slack)] = objective(eta, slack, generators[name], seed=seed)

    with open("results/real_trace_15k_forest_full_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "eta", "slack", "dropped_deadline"])
        for name in fold_names:
            for eta, slack in grid:
                w.writerow([name, eta, slack, cache[(name, eta, slack)]])
    print("Wrote results/real_trace_15k_forest_full_grid.csv")

    for held_out in fold_names:
        train_names = [n for n in fold_names if n != held_out]
        scored = []
        for eta, slack in grid:
            mean_dropped = statistics.mean(cache[(n, eta, slack)] for n in train_names)
            scored.append((mean_dropped, eta, slack))
        scored.sort(key=lambda t: t[0])
        print(f"\nHeld out {held_out} -- training set {train_names} -- top 5 (eta,slack) by training mean:")
        for mean_dropped, eta, slack in scored[:5]:
            per_fold = {n: cache[(n, eta, slack)] for n in train_names}
            print(f"  eta={eta} slack={slack}  train_mean={mean_dropped:.2f}  per_fold={per_fold}")


if __name__ == "__main__":
    main()
