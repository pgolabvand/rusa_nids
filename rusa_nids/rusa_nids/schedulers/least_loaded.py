from __future__ import annotations

from typing import Optional, Sequence

from rusa_nids.models import Flow, Worker
from rusa_nids.schedulers.base import SchedulingPolicy


class LeastLoadedPolicy(SchedulingPolicy):
    """Picks the feasible worker with the lowest queue delay. Standard
    load-based baseline (Section 6, Stage 2) -- aware of load, blind to
    flow risk, model uncertainty, and resource-type heterogeneity (CPU vs
    memory pressure)."""

    name = "least_loaded"

    def select_worker(
        self, flow: Flow, feasible_workers: Sequence[Worker]
    ) -> Optional[Worker]:
        if not feasible_workers:
            return None
        return min(feasible_workers, key=lambda w: w.queue_delay_s)

    def score(self, flow: Flow, worker: Worker) -> float:
        return worker.queue_delay_s
