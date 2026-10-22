"""Core data model for the RUSA-NIDS scheduling simulation.

Design notes
------------
- All domain objects are dataclasses with explicit units documented inline,
  so that normalization (Section 5.1 of the research plan) has a single,
  auditable source of truth.
- ``Flow`` and ``Worker`` are mutable, but all mutation that affects
  scheduling correctness (queue state, capacity) goes through
  ``ClusterState``, which owns a lock. Nothing in this module is
  thread-safe on its own -- see cluster.py.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


_flow_id_counter = itertools.count(1)


class FlowOutcomeReason(Enum):
    SCHEDULED = auto()
    REQUEUED_NO_FEASIBLE_WORKER = auto()
    DROPPED_DEADLINE_EXCEEDED = auto()   # only after max_requeue_attempts
    DROPPED_QUEUE_FULL = auto()          # explicit backpressure, never silent


@dataclass
class Flow:
    """A unit of scheduling work: one network flow (or flow batch) awaiting
    detection processing.

    Time fields are in seconds, relative to ``arrival_time`` (monotonic
    clock). Risk and uncertainty are expected in [0, 1]; callers are
    responsible for producing calibrated values (Section 5.1: normalize on
    train/validation data only, never on the evaluation split).
    """

    risk: float                      # r_i in [0, 1], higher = more dangerous
    uncertainty: float                # u_i in [0, 1], higher = more ambiguous
    deadline_s: float                 # d_i: max tolerable end-to-end latency
    estimated_cost_s: float           # baseline execution cost estimate (unscored by node)
    arrival_time: float = field(default_factory=time.monotonic)
    flow_id: int = field(default_factory=lambda: next(_flow_id_counter))
    attack_label: Optional[str] = None  # ground truth, offline eval only -- never fed to scheduler
    requeue_count: int = 0

    def age_s(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        return now - self.arrival_time

    def remaining_deadline_s(self, now: Optional[float] = None) -> float:
        return self.deadline_s - self.age_s(now)


@dataclass
class Worker:
    """A scheduling target: one cluster node / logical worker process.

    ``capacity`` and resource fields are abstract units (e.g. "concurrent
    flow-equivalents"); map them to real cgroup/Ray resource quantities in
    the deployment layer, not here, so the scheduler logic stays testable
    without a live cluster.
    """

    worker_id: str
    capacity: float                  # c_j: max concurrent cost-units
    cpu_available: float              # 0..1 fraction free
    mem_available: float              # 0..1 fraction free
    in_flight_cost: float = 0.0       # sum of estimated_cost_s of assigned, unfinished flows
    queue_depth: int = 0              # number of flows waiting at this worker
    is_alive: bool = True
    last_heartbeat: float = field(default_factory=time.monotonic)

    @property
    def queue_delay_s(self) -> float:
        """Estimated wait before a newly assigned flow starts executing.

        Simple M/1-ish proxy: in-flight cost divided by effective capacity.
        Replace with a measured EWMA from telemetry in the real deployment
        (Prometheus histogram quantiles), per Section 6/Prometheus plan.
        """
        if self.capacity <= 0:
            return float("inf")
        return self.in_flight_cost / self.capacity

    @property
    def resource_pressure(self) -> float:
        """v_j: composite resource pressure in [0, 1], 0 = idle, 1 = saturated."""
        cpu_pressure = 1.0 - max(0.0, min(1.0, self.cpu_available))
        mem_pressure = 1.0 - max(0.0, min(1.0, self.mem_available))
        return max(cpu_pressure, mem_pressure)

    def has_capacity(self, cost: float) -> bool:
        return self.is_alive and (self.in_flight_cost + cost) <= self.capacity

    def estimate_execution_time(self, flow: Flow) -> float:
        """e_ij: node-specific execution cost estimate for this flow.

        Baseline: flow's own estimate scaled by this node's resource
        pressure (a saturated node runs the same workload slower). A
        production version should use a learned per-(worker, flow-class)
        cost model -- see Section 5.2 caveat on e_ij in the research plan.
        """
        slowdown = 1.0 + self.resource_pressure
        return flow.estimated_cost_s * slowdown


@dataclass
class ScheduleDecision:
    """An audit record for one scheduling decision -- required for the
    paper's reproducibility and fault-handling claims (Section 5.3)."""

    flow_id: int
    worker_id: Optional[str]
    score: Optional[float]
    decided_at: float
    outcome: FlowOutcomeReason
    estimated_latency_s: Optional[float] = None
    policy_name: str = ""


@dataclass
class ScheduleOutcome:
    """Batch result returned by ClusterState.schedule_flow for one flow."""

    decision: ScheduleDecision
    scheduled: bool
