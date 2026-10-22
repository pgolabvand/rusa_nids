"""Discrete-event simulation harness for experiments E1-E4 (Section 6,
Stage 3 of the research plan): worker-count scaling, input intensity,
load balance, and (via mark_worker_down) fault tolerance for E5.

This is intentionally a *simulator*, not a live Ray deployment -- it lets
you validate scheduling-policy behavior (latency/throughput/fairness)
cheaply and deterministically (seeded) before paying for a real cluster
run. Swap ``ClusterState`` + a live worker pool in for the "real" E1-E7
numbers reported in the paper; keep this for CI-style regression tests on
the scheduler logic itself.

Event model
-----------
Two event types on a min-heap keyed by simulated time:
  - ARRIVAL:   a new flow enters the system -> ClusterState.submit()
  - COMPLETE:  a worker finishes a flow -> ClusterState.complete()

Between events we call ``ClusterState.retry_pending()`` so queued flows
get a chance to drain onto newly-freed capacity without waiting for the
next arrival.
"""

from __future__ import annotations

import heapq
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from rusa_nids.cluster import ClusterState
from rusa_nids.models import Flow, FlowOutcomeReason, Worker
from rusa_nids.schedulers.base import SchedulingPolicy

ARRIVAL = "arrival"
COMPLETE = "complete"


@dataclass
class SimResult:
    policy_name: str
    completed: int
    dropped_deadline: int
    dropped_queue_full: int
    throughput_flows_per_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    per_worker_completed: Dict[str, int] = field(default_factory=dict)


class FlowWorkloadGenerator:
    """Synthetic workload generator. Replace with a CICFlowMeter-derived
    trace reader for real E1-E7 runs; this exists so policy logic can be
    exercised and unit-tested without a dataset dependency.
    """

    def __init__(
        self,
        seed: int = 0,
        mean_interarrival_s: float = 0.01,
        mean_cost_s: float = 0.02,
        cost_cv: float = 0.4,
        attack_fraction: float = 0.1,
        mean_deadline_s: float = 0.5,
    ):
        self._rng = random.Random(seed)
        self.mean_interarrival_s = mean_interarrival_s
        self.mean_cost_s = mean_cost_s
        self.cost_cv = cost_cv
        self.attack_fraction = attack_fraction
        self.mean_deadline_s = mean_deadline_s

    def next_interarrival(self) -> float:
        return self._rng.expovariate(1.0 / self.mean_interarrival_s)

    def make_flow(self, arrival_time: float) -> Flow:
        is_attack = self._rng.random() < self.attack_fraction
        cost = max(
            1e-4, self._rng.gauss(self.mean_cost_s, self.mean_cost_s * self.cost_cv)
        )
        risk = self._rng.betavariate(5, 1.5) if is_attack else self._rng.betavariate(1.2, 6)
        uncertainty = self._rng.betavariate(2, 2)
        deadline = max(cost * 1.5, self._rng.expovariate(1.0 / self.mean_deadline_s))
        flow = Flow(
            risk=min(1.0, risk),
            uncertainty=min(1.0, uncertainty),
            deadline_s=deadline,
            estimated_cost_s=cost,
            arrival_time=arrival_time,
            attack_label="attack" if is_attack else "benign",
        )
        return flow


def make_worker_pool(
    n_workers: int, capacity: float = 1.0, seed: int = 0
) -> List[Worker]:
    rng = random.Random(seed)
    workers = []
    for i in range(n_workers):
        workers.append(
            Worker(
                worker_id=f"w{i}",
                capacity=capacity,
                cpu_available=rng.uniform(0.6, 1.0),
                mem_available=rng.uniform(0.6, 1.0),
            )
        )
    return workers


def run_simulation(
    policy: SchedulingPolicy,
    n_workers: int,
    n_flows: int,
    generator: Optional[FlowWorkloadGenerator] = None,
    worker_capacity: float = 1.0,
    seed: int = 0,
    worker_fault_at_flow: Optional[int] = None,
    worker_fault_id: Optional[str] = None,
) -> SimResult:
    """Run one E1/E2/E4-style trial. ``worker_fault_at_flow`` + ``worker_fault_id``
    implement the E5 fault-injection point: when that many flows have
    arrived, the named worker is marked down mid-simulation.
    """
    gen = generator or FlowWorkloadGenerator(seed=seed)
    workers = make_worker_pool(n_workers, capacity=worker_capacity, seed=seed)
    _clock_box = [0.0]  # mutable box; ClusterState calls clock() to read live sim time
    cluster = ClusterState(workers, policy, clock=lambda: _clock_box[0])

    event_heap: List[Tuple[float, int, str, int]] = []  # (time, seq, kind, flow_id)
    seq_counter = 0

    t = 0.0
    arrivals_emitted = 0
    pending_flow_lookup: Dict[int, Flow] = {}

    def schedule_event(time_: float, kind: str, flow_id: int):
        nonlocal seq_counter
        heapq.heappush(event_heap, (time_, seq_counter, kind, flow_id))
        seq_counter += 1

    # seed first arrival
    t += gen.next_interarrival()
    first_flow = gen.make_flow(t)
    pending_flow_lookup[first_flow.flow_id] = first_flow
    schedule_event(t, ARRIVAL, first_flow.flow_id)
    arrivals_emitted = 1

    completed_latencies: List[float] = []
    per_worker_completed: Dict[str, int] = {w.worker_id: 0 for w in workers}
    fault_triggered = False

    while event_heap:
        time_, _seq, kind, flow_id = heapq.heappop(event_heap)
        _clock_box[0] = time_

        if kind == ARRIVAL:
            flow = pending_flow_lookup[flow_id]
            outcome = cluster.submit(flow)
            if outcome.scheduled:
                worker_id = outcome.decision.worker_id
                exec_time = cluster.workers[worker_id].estimate_execution_time(flow)
                schedule_event(time_ + exec_time, COMPLETE, flow_id)

            if (
                worker_fault_at_flow is not None
                and not fault_triggered
                and arrivals_emitted >= worker_fault_at_flow
            ):
                cluster.mark_worker_down(worker_fault_id or workers[0].worker_id)
                fault_triggered = True

            if arrivals_emitted < n_flows:
                next_t = time_ + gen.next_interarrival()
                next_flow = gen.make_flow(next_t)
                pending_flow_lookup[next_flow.flow_id] = next_flow
                schedule_event(next_t, ARRIVAL, next_flow.flow_id)
                arrivals_emitted += 1

            # drain queue opportunistically
            cluster.retry_pending()

        elif kind == COMPLETE:
            flow = pending_flow_lookup.pop(flow_id)
            worker_id = cluster.assigned_worker_id(flow_id)
            cluster.complete(flow_id)
            latency = time_ - flow.arrival_time
            completed_latencies.append(latency)
            if worker_id:
                per_worker_completed[worker_id] = per_worker_completed.get(worker_id, 0) + 1
            cluster.retry_pending()

    dropped_deadline = sum(
        1 for d in cluster.decisions if d.outcome == FlowOutcomeReason.DROPPED_DEADLINE_EXCEEDED
    )
    dropped_queue_full = sum(
        1 for d in cluster.decisions if d.outcome == FlowOutcomeReason.DROPPED_QUEUE_FULL
    )
    total_time = _clock_box[0] if _clock_box[0] > 0 else 1e-9
    throughput = len(completed_latencies) / total_time

    def pctl(data: List[float], p: float) -> float:
        if not data:
            return float("nan")
        s = sorted(data)
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]

    return SimResult(
        policy_name=policy.name,
        completed=len(completed_latencies),
        dropped_deadline=dropped_deadline,
        dropped_queue_full=dropped_queue_full,
        throughput_flows_per_s=throughput,
        latency_p50_s=pctl(completed_latencies, 0.50),
        latency_p95_s=pctl(completed_latencies, 0.95),
        latency_p99_s=pctl(completed_latencies, 0.99),
        per_worker_completed=per_worker_completed,
    )
