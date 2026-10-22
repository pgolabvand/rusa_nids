from __future__ import annotations

import itertools
from typing import Dict, Optional, Sequence

from rusa_nids.models import Flow, Worker
from rusa_nids.schedulers.base import SchedulingPolicy


class RoundRobinPolicy(SchedulingPolicy):
    """Naive baseline: cycles through workers in a fixed global order,
    skipping any not in the feasible set for this flow. Ignores load,
    risk, and uncertainty entirely -- this is the floor baseline in the
    experiment matrix (Section 6, Stage 2)."""

    name = "round_robin"

    def __init__(self, worker_ids_in_order: Sequence[str]):
        self._order = list(worker_ids_in_order)
        self._cursor = itertools.cycle(self._order)
        # keep a lock-free single counter; ClusterState serializes calls
        self._next_index = 0

    def select_worker(
        self, flow: Flow, feasible_workers: Sequence[Worker]
    ) -> Optional[Worker]:
        if not feasible_workers:
            return None
        feasible_ids: Dict[str, Worker] = {w.worker_id: w for w in feasible_workers}
        n = len(self._order)
        for _ in range(n):
            candidate_id = self._order[self._next_index]
            self._next_index = (self._next_index + 1) % n
            if candidate_id in feasible_ids:
                return feasible_ids[candidate_id]
        return None
