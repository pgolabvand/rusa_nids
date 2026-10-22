"""RUSA-NIDS: Risk-, Uncertainty-, and System-Aware NIDS Scheduler.

Implements the scoring function from Section 5.2 of the research plan:

    S(i,j) = alpha * q_hat_j + beta * e_hat_ij + gamma * u_hat_i * e_hat_ij
             + delta * v_hat_j - eta * ru_hat_ij

Lower score wins. Normalization (q_hat, e_hat, v_hat, ru_hat) is done
relative to the feasible-candidate set at decision time, matching the
"normalize on train/validation, never on the evaluation split" instruction
for *learned* components -- here there is nothing learned in the scorer
itself, but the same discipline applies to any coefficient calibration
(`RusaWeights`) done offline before evaluation.

Scale-consistency fix (found while running E6, 2026-10-19): the original
formula subtracted the raw product ``eta * r_i * g(...)``, bounded in
[0, eta], while q_hat/e_hat/v_hat are each min-max normalized to [0, 1]
and summed with weights of order 1 (alpha+beta+gamma+delta ~= 3). Measured
on the synthetic simulator, that gap meant eta had to be pushed to
~100-200 before the risk x urgency term ever changed a routing decision --
two orders of magnitude above the paper's hand-picked default of 1.5, so
every E1/E2/E6 result run with that default was, in effect, measuring
Resource-Aware. ``ru_hat`` below min-max normalizes ``r_i * g(...)``
across the same candidate set as the other three terms, putting eta on
the same footing as alpha/beta/gamma/delta -- the fix is a renormalization,
not a reweighting, so it does not silently change what "high risk" means,
only how it competes with the other terms. Set
``RusaWeights(legacy_unnormalized_eta=True)`` to reproduce the original
(unnormalized, effectively inert at eta=1.5) behavior for the ablation
writeup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from rusa_nids.models import Flow, Worker
from rusa_nids.schedulers.base import SchedulingPolicy


@dataclass(frozen=True)
class RusaWeights:
    """Calibrated offline on train/validation data only (Section 5.1)."""

    alpha: float = 1.0   # queue delay
    beta: float = 1.0    # execution cost
    gamma: float = 0.5   # uncertainty x execution cost (expensive re-checks cost more)
    delta: float = 0.5   # resource pressure
    eta: float = 1.5     # risk x urgency (dominant term for high-risk, near-deadline flows)
    urgency_slack_s: float = 0.25  # g() saturates when slack this large (seconds)
    legacy_unnormalized_eta: bool = False  # reproduce the pre-fix (effectively inert) scorer


def _urgency(deadline_s: float, estimated_latency_s: float, slack_s: float) -> float:
    """g(d_i, q_hat_j): urgency in [0, 1]. 1 = latency estimate is at or
    past the deadline (maximally urgent to avoid), 0 = plenty of slack.

    Monotone, bounded, deadline-relative -- deliberately simple so the
    paper's ablation (E6) can later swap in a sharper (e.g. exponential)
    urgency function without touching the rest of the scorer.
    """
    if deadline_s <= 0:
        return 1.0
    slack = deadline_s - estimated_latency_s
    if slack <= 0:
        return 1.0
    return max(0.0, 1.0 - min(1.0, slack / max(slack_s, 1e-9)))


def _minmax_normalize(values: Sequence[float]) -> Sequence[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class RusaNidsPolicy(SchedulingPolicy):
    """The proposed algorithm. Stateless given a feasible candidate set --
    see schedulers/base.py for why that matters for testability."""

    name = "rusa_nids"

    def __init__(self, weights: Optional[RusaWeights] = None):
        self.weights = weights or RusaWeights()

    def _candidate_raw_features(self, flow: Flow, worker: Worker):
        exec_cost = worker.estimate_execution_time(flow)
        est_latency = worker.queue_delay_s + exec_cost
        return worker.queue_delay_s, exec_cost, worker.resource_pressure, est_latency

    def select_worker(
        self, flow: Flow, feasible_workers: Sequence[Worker]
    ) -> Optional[Worker]:
        if not feasible_workers:
            return None
        if len(feasible_workers) == 1:
            return feasible_workers[0]

        w = self.weights
        raw = [self._candidate_raw_features(flow, wk) for wk in feasible_workers]
        queue_delays = [r[0] for r in raw]
        exec_costs = [r[1] for r in raw]
        pressures = [r[2] for r in raw]
        urgencies = [
            _urgency(flow.deadline_s, est_latency, w.urgency_slack_s)
            for (_, _, _, est_latency) in raw
        ]

        q_hat = _minmax_normalize(queue_delays)
        e_hat = _minmax_normalize(exec_costs)
        v_hat = _minmax_normalize(pressures)
        if w.legacy_unnormalized_eta:
            # Pre-fix behavior: raw r_i * urgency, bounded [0, 1] but not
            # rescaled to the candidate set's actual spread -- kept only so
            # the ablation writeup can reproduce the diagnosed bug.
            ru_hat = [flow.risk * u for u in urgencies]
        else:
            ru_hat = _minmax_normalize([flow.risk * u for u in urgencies])

        best_worker: Optional[Worker] = None
        best_score = float("inf")
        for idx, worker in enumerate(feasible_workers):
            score = (
                w.alpha * q_hat[idx]
                + w.beta * e_hat[idx]
                + w.gamma * flow.uncertainty * e_hat[idx]
                + w.delta * v_hat[idx]
                - w.eta * ru_hat[idx]
            )
            if score < best_score:
                best_score = score
                best_worker = worker
        return best_worker

    def score(self, flow: Flow, worker: Worker) -> float:
        # Single-candidate score for audit logging (re-derives normalized
        # features against this one worker only -- degenerate normalization,
        # fine for logging, NOT used for ranking).
        queue_delay, exec_cost, pressure, est_latency = self._candidate_raw_features(
            flow, worker
        )
        w = self.weights
        urgency = _urgency(flow.deadline_s, est_latency, w.urgency_slack_s)
        return (
            w.alpha * queue_delay
            + w.beta * exec_cost
            + w.gamma * flow.uncertainty * exec_cost
            + w.delta * pressure
            - w.eta * flow.risk * urgency
        )
