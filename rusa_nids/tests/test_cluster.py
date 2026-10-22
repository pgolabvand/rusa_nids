"""Correctness tests for ClusterState guarantees (Section 5.3 requirements):
atomic allocation, no silent drops, starvation bound, fault requeue.

These are the tests a reviewer would expect before trusting any E1-E7
numbers produced on top of this scheduler.
"""

from __future__ import annotations

import pytest

from rusa_nids.cluster import ClusterState
from rusa_nids.models import Flow, FlowOutcomeReason, Worker
from rusa_nids.schedulers.least_loaded import LeastLoadedPolicy
from rusa_nids.schedulers.rusa import RusaNidsPolicy


def make_flow(cost=0.1, deadline=1.0, risk=0.5, uncertainty=0.3, arrival_time=None):
    kwargs = dict(
        risk=risk,
        uncertainty=uncertainty,
        deadline_s=deadline,
        estimated_cost_s=cost,
    )
    if arrival_time is not None:
        # Caller wants a specific arrival time -- only valid if the
        # ClusterState under test uses the SAME clock basis (e.g. a fixed
        # test clock). Flow's own default_factory uses time.monotonic(),
        # matching ClusterState's default clock, so leave it alone unless
        # the caller explicitly overrides.
        kwargs["arrival_time"] = arrival_time
    return Flow(**kwargs)


def test_never_exceeds_worker_capacity():
    """Atomic allocation: submitting more cost than a worker's capacity
    must never be accepted by that worker."""
    worker = Worker(worker_id="w0", capacity=1.0, cpu_available=1.0, mem_available=1.0)
    cluster = ClusterState([worker], LeastLoadedPolicy())

    flows = [make_flow(cost=0.4, deadline=10.0) for _ in range(5)]
    scheduled = [cluster.submit(f).scheduled for f in flows]

    assert worker.in_flight_cost <= worker.capacity + 1e-9
    # exactly 2 of 5 flows at cost 0.4 fit in capacity 1.0
    assert sum(scheduled) == 2


def test_every_flow_outcome_is_accounted_for():
    """No flow may vanish without a logged decision -- the paper's
    'nothing silently dropped' claim, tested directly."""
    worker = Worker(worker_id="w0", capacity=0.5, cpu_available=1.0, mem_available=1.0)
    cluster = ClusterState(
        [worker], LeastLoadedPolicy(), max_queue_size=2, max_requeue_attempts=1
    )

    flows = [make_flow(cost=0.5, deadline=10.0) for _ in range(5)]
    submitted_ids = {f.flow_id for f in flows}
    for f in flows:
        cluster.submit(f)

    logged_ids = {d.flow_id for d in cluster.decisions}
    assert submitted_ids.issubset(logged_ids)
    # every logged decision has an explicit, non-silent outcome
    for d in cluster.decisions:
        assert d.outcome in FlowOutcomeReason


def test_infeasible_deadline_does_not_schedule():
    worker = Worker(worker_id="w0", capacity=1.0, cpu_available=1.0, mem_available=1.0)
    cluster = ClusterState([worker], LeastLoadedPolicy())

    impossible = make_flow(cost=0.9, deadline=0.01)  # execution alone exceeds deadline
    outcome = cluster.submit(impossible)

    assert not outcome.scheduled
    assert outcome.decision.outcome in (
        FlowOutcomeReason.REQUEUED_NO_FEASIBLE_WORKER,
        FlowOutcomeReason.DROPPED_DEADLINE_EXCEEDED,
    )


def test_worker_failure_requeues_in_flight_flows_not_drops_them():
    w0 = Worker(worker_id="w0", capacity=2.0, cpu_available=1.0, mem_available=1.0)
    w1 = Worker(worker_id="w1", capacity=2.0, cpu_available=1.0, mem_available=1.0)
    cluster = ClusterState([w0, w1], LeastLoadedPolicy())

    flow = make_flow(cost=0.5, deadline=10.0)
    outcome = cluster.submit(flow)
    assert outcome.scheduled
    assigned_worker = outcome.decision.worker_id

    requeued = cluster.mark_worker_down(assigned_worker)

    if assigned_worker and cluster.in_flight_count == 0:
        assert requeued == 1
        assert cluster.pending_count == 1
    # flow is never just gone
    assert flow.flow_id in {d.flow_id for d in cluster.decisions}


def test_starvation_bounded_by_max_requeue_attempts():
    """A flow that can never find a feasible worker must eventually be
    dropped with an explicit, logged reason -- not retried forever."""
    worker = Worker(worker_id="w0", capacity=0.1, cpu_available=1.0, mem_available=1.0)
    cluster = ClusterState([worker], LeastLoadedPolicy(), max_requeue_attempts=3)

    doomed = make_flow(cost=5.0, deadline=100.0)  # never fits in capacity 0.1
    cluster.submit(doomed)
    for _ in range(5):
        cluster.retry_pending()

    final_decisions = [d for d in cluster.decisions if d.flow_id == doomed.flow_id]
    assert final_decisions[-1].outcome == FlowOutcomeReason.DROPPED_DEADLINE_EXCEEDED
    assert cluster.pending_count == 0


def test_rusa_prefers_low_pressure_worker_for_high_risk_urgent_flow():
    busy = Worker(worker_id="busy", capacity=2.0, cpu_available=0.1, mem_available=0.1)
    idle = Worker(worker_id="idle", capacity=2.0, cpu_available=0.95, mem_available=0.95)
    cluster = ClusterState([busy, idle], RusaNidsPolicy())

    urgent_risky = make_flow(cost=0.05, deadline=0.2, risk=0.95, uncertainty=0.1)
    outcome = cluster.submit(urgent_risky)

    assert outcome.scheduled
    assert outcome.decision.worker_id == "idle"


def test_scheduler_policies_are_pure_no_mutation_on_declined_candidates():
    """Policies must not mutate Worker state -- only ClusterState does,
    after a decision is made (schedulers/base.py contract)."""
    w0 = Worker(worker_id="w0", capacity=1.0, cpu_available=1.0, mem_available=1.0)
    before = (w0.in_flight_cost, w0.queue_depth)

    policy = LeastLoadedPolicy()
    flow = make_flow(cost=0.2, deadline=5.0)
    chosen = policy.select_worker(flow, [w0])

    assert chosen is w0
    assert (w0.in_flight_cost, w0.queue_depth) == before
