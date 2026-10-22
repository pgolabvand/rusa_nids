from __future__ import annotations

from typing import Optional, Sequence

from rusa_nids.models import Flow, Worker
from rusa_nids.schedulers.base import SchedulingPolicy


class ResourceAwarePolicy(SchedulingPolicy):
    """Picks the feasible worker minimizing a blend of queue delay and
    resource pressure (CPU/memory), but still ignores flow-level risk and
    detection uncertainty. This isolates "does resource-awareness alone
    explain RUSA-NIDS's gains" for the ablation in E6."""

    name = "resource_aware"

    def __init__(self, queue_weight: float = 0.6, pressure_weight: float = 0.4):
        self.queue_weight = queue_weight
        self.pressure_weight = pressure_weight

    def score(self, flow: Flow, worker: Worker) -> float:
        return (
            self.queue_weight * worker.queue_delay_s
            + self.pressure_weight * worker.resource_pressure
        )

    def select_worker(
        self, flow: Flow, feasible_workers: Sequence[Worker]
    ) -> Optional[Worker]:
        if not feasible_workers:
            return None
        return min(feasible_workers, key=lambda w: self.score(flow, w))
