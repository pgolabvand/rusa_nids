"""ClusterState: the only place allowed to mutate Worker/Flow scheduling
state. Everything the research plan's Section 5.3 flagged as missing from
the pseudocode lives here, not in a policy:

  - atomic task allocation       -> single RLock around read-decide-commit
  - queue management             -> _pending_queue + retry_pending()
  - starvation prevention        -> EDF-ish re-ordering on each retry pass
                                     + bounded max_requeue_attempts (never
                                     silent: a flow that exhausts its
                                     budget is logged as DROPPED_*, not
                                     discarded quietly)
  - decision logging             -> self.decisions (ScheduleDecision log),
                                     append-only, meant to back an audit
                                     trail / Postgres sink in deployment
  - timeout / backpressure       -> max_queue_size -> explicit
                                     DROPPED_QUEUE_FULL, never a silent drop
  - worker failure behavior      -> mark_worker_down() re-queues orphaned
                                     in-flight flows at the FRONT of the
                                     queue (NIPS note: don't lose events)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from rusa_nids.models import Flow, FlowOutcomeReason, ScheduleDecision, ScheduleOutcome, Worker
from rusa_nids.schedulers.base import SchedulingPolicy

logger = logging.getLogger("rusa_nids.cluster")


class ClusterState:
    def __init__(
        self,
        workers: List[Worker],
        policy: SchedulingPolicy,
        max_queue_size: int = 10_000,
        max_requeue_attempts: int = 5,
        clock: Optional[callable] = None,
    ):
        self.workers: Dict[str, Worker] = {w.worker_id: w for w in workers}
        self.policy = policy
        self.max_queue_size = max_queue_size
        self.max_requeue_attempts = max_requeue_attempts
        self._clock = clock or time.monotonic

        self._pending_queue: Deque[Flow] = deque()
        self._in_flight: Dict[int, Tuple[str, Flow]] = {}
        self.decisions: List[ScheduleDecision] = []

        try:
            import threading

            self._lock = threading.RLock()
        except ImportError:  # pragma: no cover - threading always available on CPython
            self._lock = _NullLock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def submit(self, flow: Flow) -> ScheduleOutcome:
        """Attempt to schedule a newly arrived flow. If no worker is
        feasible right now, the flow is queued (subject to backpressure)
        rather than dropped."""
        with self._lock:
            outcome = self._try_schedule(flow)
            if (
                not outcome.scheduled
                and outcome.decision.outcome == FlowOutcomeReason.REQUEUED_NO_FEASIBLE_WORKER
            ):
                if len(self._pending_queue) >= self.max_queue_size:
                    decision = self._log_decision(
                        flow, None, None, FlowOutcomeReason.DROPPED_QUEUE_FULL
                    )
                    logger.warning(
                        "backpressure: dropping flow %s, queue at capacity (%d)",
                        flow.flow_id,
                        self.max_queue_size,
                    )
                    return ScheduleOutcome(decision, False)
                self._pending_queue.append(flow)
            return outcome

    def retry_pending(self) -> int:
        """Re-attempt queued flows. Call this once per simulation tick /
        scheduling loop iteration. Returns the number newly scheduled.

        Starvation prevention: on every pass, pending flows are reordered
        by remaining deadline slack (earliest-deadline-first), so a flow
        that keeps losing the capacity race to newer, laxer-deadline flows
        does not wait indefinitely behind them.
        """
        with self._lock:
            now = self._clock()
            ordered = sorted(
                self._pending_queue, key=lambda f: f.remaining_deadline_s(now)
            )
            still_pending: Deque[Flow] = deque()
            newly_scheduled = 0
            for flow in ordered:
                outcome = self._try_schedule(flow)
                if outcome.scheduled:
                    newly_scheduled += 1
                elif outcome.decision.outcome == FlowOutcomeReason.REQUEUED_NO_FEASIBLE_WORKER:
                    still_pending.append(flow)
                # DROPPED_DEADLINE_EXCEEDED: already logged inside _try_schedule, not re-queued
            self._pending_queue = still_pending
            return newly_scheduled

    def complete(self, flow_id: int, actual_cost_s: Optional[float] = None) -> None:
        """Release a worker's capacity once a flow finishes processing.
        ``actual_cost_s`` lets the simulator report the *real* execution
        time if it differs from the estimate used at scheduling time
        (useful for E6/uncertainty ablations)."""
        with self._lock:
            entry = self._in_flight.pop(flow_id, None)
            if entry is None:
                logger.debug("complete() called for unknown/already-completed flow %s", flow_id)
                return
            worker_id, flow = entry
            worker = self.workers.get(worker_id)
            if worker is None:
                return
            cost = actual_cost_s if actual_cost_s is not None else worker.estimate_execution_time(flow)
            worker.in_flight_cost = max(0.0, worker.in_flight_cost - cost)
            worker.queue_depth = max(0, worker.queue_depth - 1)

    def mark_worker_down(self, worker_id: str) -> int:
        """Fault injection hook for E5. Orphaned in-flight flows are
        requeued at the FRONT of the pending queue (priority re-insertion)
        rather than dropped -- this is the "don't lose alerts" guarantee
        from Section 5.4. Returns the number of flows requeued."""
        with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None:
                raise KeyError(f"unknown worker_id {worker_id!r}")
            worker.is_alive = False

            orphaned_ids = [
                fid for fid, (wid, _f) in self._in_flight.items() if wid == worker_id
            ]
            orphaned_flows = []
            for fid in orphaned_ids:
                _wid, flow = self._in_flight.pop(fid)
                flow.requeue_count = 0  # fresh budget: not this flow's fault
                orphaned_flows.append(flow)

            # front-insert, most urgent first, to minimize added latency
            now = self._clock()
            orphaned_flows.sort(key=lambda f: f.remaining_deadline_s(now))
            for flow in reversed(orphaned_flows):
                self._pending_queue.appendleft(flow)

            worker.in_flight_cost = 0.0
            worker.queue_depth = 0
            logger.warning(
                "worker %s marked down; %d in-flight flow(s) requeued",
                worker_id,
                len(orphaned_flows),
            )
            return len(orphaned_flows)

    def mark_worker_up(
        self, worker_id: str, cpu_available: float = 1.0, mem_available: float = 1.0
    ) -> None:
        with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None:
                raise KeyError(f"unknown worker_id {worker_id!r}")
            worker.is_alive = True
            worker.cpu_available = cpu_available
            worker.mem_available = mem_available
            worker.last_heartbeat = self._clock()

    @property
    def pending_count(self) -> int:
        return len(self._pending_queue)

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    def assigned_worker_id(self, flow_id: int) -> Optional[str]:
        """Public accessor for the worker currently holding a flow, if
        any -- avoids callers (e.g. the simulator) reaching into the
        private ``_in_flight`` map."""
        with self._lock:
            entry = self._in_flight.get(flow_id)
            return entry[0] if entry else None

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _feasible_workers(self, flow: Flow, now: float) -> List[Worker]:
        feasible = []
        remaining_deadline = flow.remaining_deadline_s(now)
        for worker in self.workers.values():
            if not worker.is_alive:
                continue
            if not worker.has_capacity(flow.estimated_cost_s):
                continue
            estimated_latency = worker.queue_delay_s + worker.estimate_execution_time(flow)
            if estimated_latency > remaining_deadline:
                continue
            feasible.append(worker)
        return feasible

    def _try_schedule(self, flow: Flow) -> ScheduleOutcome:
        now = self._clock()
        feasible = self._feasible_workers(flow, now)

        if not feasible:
            return self._handle_infeasible(flow)

        worker = self.policy.select_worker(flow, feasible)
        if worker is None:
            # Policy explicitly declined despite feasible candidates
            # (e.g. a future policy with its own admission control).
            return self._handle_infeasible(flow)

        cost = worker.estimate_execution_time(flow)
        est_latency = worker.queue_delay_s + cost  # computed BEFORE mutating state
        score = self.policy.score(flow, worker)

        # --- atomic commit: both lines happen under the held lock ---
        worker.in_flight_cost += cost
        worker.queue_depth += 1
        self._in_flight[flow.flow_id] = (worker.worker_id, flow)
        # --------------------------------------------------------------

        decision = self._log_decision(
            flow,
            worker.worker_id,
            score,
            FlowOutcomeReason.SCHEDULED,
            estimated_latency_s=est_latency,
        )
        return ScheduleOutcome(decision, True)

    def _handle_infeasible(self, flow: Flow) -> ScheduleOutcome:
        flow.requeue_count += 1
        if flow.requeue_count > self.max_requeue_attempts:
            decision = self._log_decision(
                flow, None, None, FlowOutcomeReason.DROPPED_DEADLINE_EXCEEDED
            )
            logger.warning(
                "flow %s dropped after %d requeue attempts (deadline %.3fs)",
                flow.flow_id,
                flow.requeue_count,
                flow.deadline_s,
            )
            return ScheduleOutcome(decision, False)
        decision = self._log_decision(
            flow, None, None, FlowOutcomeReason.REQUEUED_NO_FEASIBLE_WORKER
        )
        return ScheduleOutcome(decision, False)

    def _log_decision(
        self,
        flow: Flow,
        worker_id: Optional[str],
        score: Optional[float],
        outcome: FlowOutcomeReason,
        estimated_latency_s: Optional[float] = None,
    ) -> ScheduleDecision:
        decision = ScheduleDecision(
            flow_id=flow.flow_id,
            worker_id=worker_id,
            score=score,
            decided_at=self._clock(),
            outcome=outcome,
            estimated_latency_s=estimated_latency_s,
            policy_name=self.policy.name,
        )
        self.decisions.append(decision)
        return decision


class _NullLock:
    """Fallback no-op lock; threading is always available on CPython so
    this only exists to keep type checkers happy about the attribute."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
