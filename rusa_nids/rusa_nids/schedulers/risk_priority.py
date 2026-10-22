from __future__ import annotations

from typing import Optional, Sequence

from rusa_nids.models import Flow, Worker
from rusa_nids.schedulers.base import SchedulingPolicy


class RiskPriorityPolicy(SchedulingPolicy):
    """Security-first baseline: among feasible workers, prefers the one
    with lowest queue delay, but scales the preference strength by flow
    risk (high-risk flows pay more attention to latency; low-risk flows
    are placed almost arbitrarily). Ignores resource pressure and
    detection uncertainty entirely -- isolates "does risk-awareness alone
    explain RUSA-NIDS's gains" for ablation E6."""

    name = "risk_priority"

    def score(self, flow: Flow, worker: Worker) -> float:
        risk_weight = 0.2 + 0.8 * flow.risk  # higher risk -> latency matters more
        return risk_weight * worker.queue_delay_s

    def select_worker(
        self, flow: Flow, feasible_workers: Sequence[Worker]
    ) -> Optional[Worker]:
        if not feasible_workers:
            return None
        return min(feasible_workers, key=lambda w: self.score(flow, w))
