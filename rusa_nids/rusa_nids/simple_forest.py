"""A small, dependency-free random-forest classifier (NumPy only).

Upgrade path for scripts/train_fold_risk_classifier.py: the first pass at
a real-trace risk signal used plain logistic regression (held-out
accuracy 0.59-0.94, no tuning) and produced an eta=0.0 nested-CV result
that contradicted the synthetic calibration's eta=3.0 -- but a linear
model is a weak, easily-confounded way to test whether that
contradiction is real or just an artifact of a noisy risk signal. This
module is a CART-based random forest (bootstrap bagging + random feature
subsets at each split, the standard Breiman recipe), written from scratch
in NumPy so the project keeps its ``dependencies = []`` stance -- a
scikit-learn install was observed to hang on the target machine, and a
hand-rolled forest this size (a few thousand training rows, 17 features)
is not performance-sensitive enough to need it.

This is still not the thesis's CNN-XGBoost model (different feature
space entirely -- see train_fold_risk_classifier.py's own docstring) and
still not a claim of production detector quality. It is a meaningfully
stronger nonlinear baseline than logistic regression, which is the
relevant comparison for the eta=0.0 question: if eta stays at 0 with a
forest too, that is much better evidence the risk term itself is the
issue (not just "logistic regression is too weak"); if eta becomes
nonzero, the earlier eta=0.0 result was a classifier-noise artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _gini_from_counts(n1: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Gini impurity for binary labels given counts of class-1 (n1) out
    of total (n), vectorized over many candidate split points at once.
    Returns 0 impurity (not NaN) where n == 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = np.where(n > 0, n1 / np.maximum(n, 1), 0.0)
    return 1.0 - p1**2 - (1.0 - p1) ** 2


def _best_split_for_feature(x_col: np.ndarray, y: np.ndarray):
    """Best threshold on one feature column, by weighted Gini reduction.
    Returns (threshold, weighted_gini) or (None, None) if the column is
    constant (no valid split). O(n log n) via a single sort, using
    cumulative class-1 counts so every candidate split's Gini is O(1).
    """
    order = np.argsort(x_col, kind="mergesort")
    xs = x_col[order]
    ys = y[order]
    n = len(xs)
    if n < 2 or xs[0] == xs[-1]:
        return None, None

    cum1 = np.cumsum(ys)              # class-1 count in [0..i] (left side if split after i)
    total1 = cum1[-1]
    left_n = np.arange(1, n)          # sizes 1..n-1 for the left partition
    right_n = n - left_n
    left_n1 = cum1[:-1]
    right_n1 = total1 - left_n1

    # Only evaluate split points where the threshold actually separates
    # distinct values (xs[i] != xs[i+1]); duplicate-value "splits" aren't
    # realizable thresholds for new data.
    valid = xs[:-1] != xs[1:]
    if not valid.any():
        return None, None

    gini_left = _gini_from_counts(left_n1, left_n)
    gini_right = _gini_from_counts(right_n1, right_n)
    weighted = (left_n * gini_left + right_n * gini_right) / n
    weighted = np.where(valid, weighted, np.inf)

    best_i = int(np.argmin(weighted))
    if not np.isfinite(weighted[best_i]):
        return None, None
    threshold = (xs[best_i] + xs[best_i + 1]) / 2.0
    return float(threshold), float(weighted[best_i])


@dataclass
class _Node:
    is_leaf: bool
    proba: float = 0.0
    feature: int = -1
    threshold: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class DecisionTree:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 20, min_samples_leaf: int = 10,
                 max_features: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.root: Optional[_Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n, d = X.shape
        proba = float(y.mean()) if n else 0.0
        if (depth >= self.max_depth or n < self.min_samples_split or
                len(np.unique(y)) < 2):
            return _Node(is_leaf=True, proba=proba)

        max_feat = self.max_features or max(1, int(np.sqrt(d)))
        feat_idx = self.rng.choice(d, size=min(max_feat, d), replace=False)

        best_feature, best_threshold, best_score = -1, 0.0, np.inf
        for f in feat_idx:
            thr, score = _best_split_for_feature(X[:, f], y)
            if thr is None:
                continue
            if score < best_score:
                best_feature, best_threshold, best_score = int(f), thr, score

        if best_feature == -1:
            return _Node(is_leaf=True, proba=proba)

        left_mask = X[:, best_feature] <= best_threshold
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return _Node(is_leaf=True, proba=proba)

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(is_leaf=False, feature=best_feature, threshold=best_threshold, left=left, right=right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row, self.root) for row in X])

    def _predict_one(self, row: np.ndarray, node: _Node) -> float:
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.proba


class RandomForest:
    """Bootstrap-bagged ``DecisionTree``s with random feature subsets per
    split (standard Breiman random forest), predicting the mean of the
    trees' leaf class-1 fractions.
    """

    def __init__(self, n_trees: int = 40, max_depth: int = 6, min_samples_split: int = 20,
                 min_samples_leaf: int = 10, max_features: Optional[int] = None, seed: int = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees: list[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self.trees = []
        for i in range(self.n_trees):
            tree_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            boot_idx = tree_rng.integers(0, n, size=n)  # bootstrap sample, with replacement
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=tree_rng,
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("RandomForest.fit() must be called before predict_proba()")
        preds = np.stack([t.predict_proba(X) for t in self.trees], axis=0)
        return preds.mean(axis=0)
