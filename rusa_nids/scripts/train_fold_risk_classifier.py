#!/usr/bin/env python3
"""Leave-one-day-out risk classifier for the CICIDS2017/CSE-CIC-IDS2018
real-trace pipeline (produces the ``detector_prob`` column that
run_real_trace_experiment.py's ``--risk-col`` expects).

Why this exists, and why it is NOT the thesis/first-paper CNN-XGBoost
model: that model classifies DGA-generated DOMAIN NAMES character-by-
character (see /areas/phd-thesis-botnet-dga.md) -- a completely different
input space from CICFlowMeter's per-flow numeric feature vectors
(duration, packet counts, byte rates, flag counts...). There is no way to
"run" a character-level domain-name model on tabular flow records; they
don't share a feature space. Reusing it here would be a category error,
not a shortcut, so this script trains a separate, purpose-built baseline
instead: a plain logistic regression over standard CICFlowMeter numeric
features, in NumPy only (no scikit-learn dependency -- consistent with
rusa_nids's own dependencies=[] stance, and avoids a pip install on a
machine where that install was observed to hang).

This IS the correct way to get a non-circular ``risk_col`` for the real
trace pipeline, and it already respects the no-leakage discipline the
rest of this project insists on: for each held-out day, the classifier
is trained ONLY on the other days (never on the day it then scores), so
"risk" is never just a reflection of that day's own ground truth. That
is what makes the resulting probability a legitimate stand-in for "what
a deployed detector would have said" rather than the circular
label-as-risk fallback real_trace.py warns about.

What this is NOT: a validated intrusion detector. It is an 11-feature
logistic regression with no hyperparameter search, no cross-validated
accuracy reporting of its OWN, and no claim of detection-quality
competitiveness. Report it as exactly that in the paper -- "a baseline
logistic-regression risk model, used only to supply a non-circular risk
signal for the scheduler calibration experiment, not evaluated as a
detector in its own right." If you want the real thesis classifier's
signal here eventually, it would need to be retrained/retargeted on
flow-level features first; that is a separate, larger piece of work, not
something this script or a future tweak to it can shortcut.

Usage:
    python scripts/train_fold_risk_classifier.py \
        --data-dir "../datasets/TrafficLabelling " \
        --out data/scored \
        --rows-per-day 4000

Writes one CSV per day under --out, each with ``detector_prob`` and
``detector_uncertainty`` columns added, ready for:
    python scripts/run_real_trace_experiment.py --data-dir data/scored \
        --risk-col detector_prob --uncertainty-col detector_uncertainty
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from rusa_nids.simple_forest import RandomForest

FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Packet Length Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Average Packet Size",
    "Down/Up Ratio",
]


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


def discover_day_files(data_dir: str) -> Dict[str, List[str]]:
    folds: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        key = Path(path).stem.split("-")[0].split("_")[0]
        folds.setdefault(key, []).append(path)
    return folds


def load_feature_matrix(csv_paths: Sequence[str], max_rows: int, seed: int):
    """Reservoir-sample rows and extract (X, y, raw_rows, fieldnames).
    y = 1 for any non-BENIGN label, 0 for BENIGN.
    """
    rng = random.Random(seed)
    reservoir: List[dict] = []
    seen = 0
    fieldnames = None
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = reader.fieldnames
            norm_map = {_normalize(k): k for k in reader.fieldnames}
            label_key = norm_map.get("label")
            for row in reader:
                if max_rows is None:
                    reservoir.append(row)
                elif seen < max_rows:
                    reservoir.append(row)
                else:
                    j = rng.randint(0, seen)
                    if j < max_rows:
                        reservoir[j] = row
                seen += 1
    norm_map = {_normalize(k): k for k in fieldnames} if fieldnames else {}
    label_key = norm_map.get("label")
    feat_keys = [norm_map.get(_normalize(c)) for c in FEATURE_COLUMNS]
    missing = [c for c, k in zip(FEATURE_COLUMNS, feat_keys) if k is None]
    if missing:
        raise ValueError(f"CSV missing expected feature column(s): {missing}")

    X = np.zeros((len(reservoir), len(FEATURE_COLUMNS)), dtype=np.float64)
    y = np.zeros(len(reservoir), dtype=np.float64)
    for i, row in enumerate(reservoir):
        for j, key in enumerate(feat_keys):
            try:
                v = float(row[key])
            except (ValueError, TypeError):
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            X[i, j] = v
        label = (row.get(label_key) or "").strip().upper()
        y[i] = 0.0 if label == "BENIGN" else 1.0
    return X, y, reservoir, fieldnames


def standardize(X: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 300, l2: float = 1e-3):
    """Plain batch gradient descent logistic regression -- no sklearn.
    Fine at this scale (a few thousand rows x 17 features); not meant to
    be a competitive detector (see module docstring).
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(epochs):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / n + l2 * np.r_[w[:-1], 0.0]
        w -= lr * grad
    return w


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows-per-day", type=int, default=4000, help="Reservoir sample size, both for training (pooled other-days) and for the scored output per held-out day.")
    p.add_argument("--model", choices=["forest", "logreg"], default="forest", help="forest: NumPy random forest (default, nonlinear, no sklearn needed -- see simple_forest.py). logreg: the original plain logistic-regression baseline, kept for comparison.")
    p.add_argument("--n-trees", type=int, default=60)
    p.add_argument("--max-depth", type=int, default=7)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    folds = discover_day_files(args.data_dir)
    fold_names = sorted(folds)
    if len(fold_names) < 2:
        raise SystemExit(f"Need >=2 day-file groups under {args.data_dir!r}, found {len(fold_names)}.")

    print(f"Days found: {fold_names}")
    report_rows = []
    for held_out in fold_names:
        train_days = [d for d in fold_names if d != held_out]
        train_paths = [pth for d in train_days for pth in folds[d]]
        seed = zlib.crc32(held_out.encode()) % (2**31)

        X_train, y_train, _, _ = load_feature_matrix(train_paths, max_rows=args.rows_per_day, seed=seed)
        X_held, y_held, held_rows, held_fieldnames = load_feature_matrix(
            folds[held_out], max_rows=args.rows_per_day, seed=seed
        )

        if args.model == "forest":
            rf = RandomForest(n_trees=args.n_trees, max_depth=args.max_depth, seed=seed)
            rf.fit(X_train, y_train)
            probs = rf.predict_proba(X_held)
        else:
            Xs_train, mean, std = standardize(X_train)
            w = train_logistic_regression(Xs_train, y_train)
            Xs_held, _, _ = standardize(X_held, mean, std)
            probs = predict_proba(Xs_held, w)

        # Simple held-out sanity check (NOT a claim of detector quality --
        # see module docstring): accuracy at a 0.5 threshold.
        preds = (probs >= 0.5).astype(float)
        acc = float((preds == y_held).mean()) if len(y_held) else float("nan")
        report_rows.append({"held_out_day": held_out, "model": args.model, "train_rows": len(y_train), "held_out_rows": len(y_held), "threshold_0.5_accuracy": round(acc, 4)})
        print(f"  {held_out}: trained on {len(y_train)} rows from {train_days}, "
              f"scored {len(y_held)} held-out rows, 0.5-threshold accuracy={acc:.3f} "
              f"(model={args.model}; sanity check only, not a detector-quality claim)")

        out_fieldnames = list(held_fieldnames) + ["detector_prob", "detector_uncertainty"]
        out_path = os.path.join(args.out, f"{held_out}-scored.csv")
        with open(out_path, "w", newline="") as f:
            w_csv = csv.DictWriter(f, fieldnames=out_fieldnames)
            w_csv.writeheader()
            for row, prob in zip(held_rows, probs):
                out_row = dict(row)
                out_row["detector_prob"] = f"{prob:.6f}"
                out_row["detector_uncertainty"] = f"{1.0 - abs(2.0 * prob - 1.0):.6f}"
                w_csv.writerow(out_row)
        print(f"    wrote {out_path}")

    os.makedirs(os.path.join(args.out, "_reports"), exist_ok=True)
    report_path = os.path.join(args.out, "_reports", "classifier_sanity_report.csv")
    with open(report_path, "w", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w_csv.writeheader()
        for r in report_rows:
            w_csv.writerow(r)
    print(f"\nWrote {report_path}")
    print(
        "\nNext: python scripts/run_real_trace_experiment.py "
        f"--data-dir {args.out} --risk-col detector_prob "
        "--uncertainty-col detector_uncertainty --out results/"
    )


if __name__ == "__main__":
    main()
