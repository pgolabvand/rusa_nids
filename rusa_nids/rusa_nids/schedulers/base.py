"""Scheduler interface. Every baseline and RUSA-NIDS implement this so the
experiment harness (simulate.py) can swap policies without touching
ClusterState.

Contract
--------
``select_worker`` must be a *pure function* of (flow, feasible_workers):
no side effects, no mutation of Worker objects. ClusterState is the only
place allowed to mutate state, and it does so atomically after a policy
returns its choice -- this is what makes the "no flow silently dropped"
and "atomic allocation" guarantees in Section 5.3 of the research plan
enforceable and testable in isolation from any policy's internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from rusa_nids.models import Flow, Worker


class SchedulingPolicy(ABC):
    """Base class for all scheduling policies under comparison."""

    name: str = "base"

    @abstractmethod
    def select_worker(
        self, flow: Flow, feasible_workers: Sequence[Worker]
    ) -> Optional[Worker]:
        """Return the chosen worker, or None if the policy declines to
        place this flow among the given *already feasible* candidates
        (feasibility filtering -- capacity + deadline -- is done by
        ClusterState before this is called; see cluster.py).
        """
        raise NotImplementedError

    def score(self, flow: Flow, worker: Worker) -> float:
        """Optional: policies that rank by a score can expose it here for
        ScheduleDecision.score (audit log). Default: NaN sentinel, meaning
        "not score-based" (e.g. Round Robin)."""
        return float("nan")
