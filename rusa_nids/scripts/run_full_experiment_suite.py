#!/usr/bin/env python3
"""Runs E1, E2, E4, E5, E6 against the synthetic simulator (NOT real
CICIDS2017/CSE-CIC-IDS2018 traffic -- see README's "What this is NOT").

This exists to turn the paper's Experimental Design section from a plan
into numbers the draft can cite today, with every result explicitly
labeled synthetic-simulator-only. Swap FlowWorkloadGenerator for a real
trace reader to get the numbers that actually go in the submission.

Usage:
    python scripts/run_full_experiment_suite.py --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import asdict

from rusa_nids.schedulers import (
    LeastLoadedPolicy,
    ResourceAwarePolicy,
    RiskPriorityPolicy,
    RoundRobinPolicy,
    RusaNidsPolicy,
)
from rusa_nids.schedulers.rusa import RusaWeights
from rusa_nids.simulate import FlowWorkloadGenerator, run_simulation


def build_policies(n_workers: int):
    worker_ids = [f"w{i}" for i in range(n_workers)]
    return [
        RoundRobinPolicy(worker_ids),
        LeastLoadedPolicy(),
        ResourceAwarePolicy(),
        RiskPriorityPolicy(),
        RusaNidsPolicy(),
    ]


def coefficient_of_variation(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan")
    mean = statistics.mean(vals)
    if mean == 0:
        return float("nan")
    if len(vals) < 2:
        return 0.0
    return statistics.pstdev(vals) / mean


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_e1(out_dir, flows, seed, workers):
    """E1 -- worker-count scaling."""
    rows = []
    for n_workers in workers:
        for policy in build_policies(n_workers):
            gen = FlowWorkloadGenerator(seed=seed)
            res = run_simulation(policy=policy, n_workers=n_workers, n_flows=flows,
                                  generator=gen, seed=seed)
            rows.append({
                "workers": n_workers, "policy": res.policy_name,
                "completed": res.completed, "dropped_deadline": res.dropped_deadline,
                "dropped_queue_full": res.dropped_queue_full,
                "throughput_fps": round(res.throughput_flows_per_s, 2),
                "p50_ms": round(res.latency_p50_s * 1000, 2),
                "p95_ms": round(res.latency_p95_s * 1000, 2),
                "p99_ms": round(res.latency_p99_s * 1000, 2),
            })
    write_csv(os.path.join(out_dir, "e1_worker_scaling.csv"), rows,
              list(rows[0].keys()))
    return rows


def run_e2(out_dir, n_workers, flows, seed, interarrival_values):
    """E2 -- input intensity, worker count fixed at E1's knee (4, from the
    README's observed capacity-constrained regime)."""
    rows = []
    for mean_ia in interarrival_values:
        for policy in build_policies(n_workers):
            gen = FlowWorkloadGenerator(seed=seed, mean_interarrival_s=mean_ia)
            res = run_simulation(policy=policy, n_workers=n_workers, n_flows=flows,
                                  generator=gen, seed=seed)
            # implied offered load: mean service demand / mean interarrival,
            # service demand ~ mean_cost_s * (1 + avg resource_pressure) / n_workers
            offered_load = (gen.mean_cost_s / mean_ia) / n_workers
            rows.append({
                "mean_interarrival_s": mean_ia,
                "offered_load_per_worker": round(offered_load, 3),
                "policy": res.policy_name,
                "completed": res.completed,
                "dropped_deadline": res.dropped_deadline,
                "dropped_queue_full": res.dropped_queue_full,
                "throughput_fps": round(res.throughput_flows_per_s, 2),
                "p95_ms": round(res.latency_p95_s * 1000, 2),
            })
    write_csv(os.path.join(out_dir, "e2_input_intensity.csv"), rows,
              list(rows[0].keys()))
    return rows


def run_e4(out_dir, n_workers, flows, seed):
    """E4 -- load balance across a heterogeneous worker pool. Metric: CV of
    per-worker completed-flow counts (lower = more balanced)."""
    rows = []
    for policy in build_policies(n_workers):
        gen = FlowWorkloadGenerator(seed=seed)
        res = run_simulation(policy=policy, n_workers=n_workers, n_flows=flows,
                              generator=gen, seed=seed)
        counts = list(res.per_worker_completed.values())
        rows.append({
            "policy": res.policy_name,
            "per_worker_completed": json.dumps(res.per_worker_completed),
            "cv_completed": round(coefficient_of_variation(counts), 4),
            "min_completed": min(counts),
            "max_completed": max(counts),
        })
    write_csv(os.path.join(out_dir, "e4_load_balance.csv"), rows,
              list(rows[0].keys()))
    return rows


def run_e5(out_dir, n_workers, flows, seed, fault_points):
    """E5 -- fault injection. For each fault timing, compare against a
    no-fault run on the same seed/policy to get flows lost vs
    requeued-and-completed and the added p95 latency."""
    rows = []
    for fault_at in fault_points:
        for policy in build_policies(n_workers):
            gen_base = FlowWorkloadGenerator(seed=seed)
            baseline = run_simulation(policy=policy, n_workers=n_workers, n_flows=flows,
                                       generator=gen_base, seed=seed)
            gen_fault = FlowWorkloadGenerator(seed=seed)
            faulted = run_simulation(policy=policy, n_workers=n_workers, n_flows=flows,
                                      generator=gen_fault, seed=seed,
                                      worker_fault_at_flow=fault_at,
                                      worker_fault_id="w0")
            rows.append({
                "fault_at_flow": fault_at,
                "policy": policy.name,
                "completed_baseline": baseline.completed,
                "completed_faulted": faulted.completed,
                "completed_delta": faulted.completed - baseline.completed,
                "dropped_deadline_baseline": baseline.dropped_deadline,
                "dropped_deadline_faulted": faulted.dropped_deadline,
                "dropped_deadline_delta": faulted.dropped_deadline - baseline.dropped_deadline,
                "p95_ms_baseline": round(baseline.latency_p95_s * 1000, 2),
                "p95_ms_faulted": round(faulted.latency_p95_s * 1000, 2),
                "p95_ms_delta": round((faulted.latency_p95_s - baseline.latency_p95_s) * 1000, 2),
            })
    write_csv(os.path.join(out_dir, "e5_fault_injection.csv"), rows,
              list(rows[0].keys()))
    return rows


def run_e6(out_dir, n_workers_list, flows, seed):
    """E6 -- ablation. resource_aware (gamma=eta=0) and risk_priority
    (no uncertainty/resource coupling) are already independent baselines,
    so this reuses E1's table and adds a weight-sensitivity sweep on
    RUSA-NIDS's eta (risk x urgency weight) at the capacity-constrained
    worker count, to characterize sensitivity rather than assume the
    hand-picked defaults generalize."""
    rows = []
    n_workers = n_workers_list[0]
    for eta in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        weights = RusaWeights(eta=eta)
        policy = RusaNidsPolicy(weights=weights)
        gen = FlowWorkloadGenerator(seed=seed)
        res = run_simulation(policy=policy, n_workers=n_workers, n_flows=flows,
                              generator=gen, seed=seed)
        rows.append({
            "workers": n_workers,
            "eta": eta,
            "completed": res.completed,
            "dropped_deadline": res.dropped_deadline,
            "p95_ms": round(res.latency_p95_s * 1000, 2),
        })
    write_csv(os.path.join(out_dir, "e6_eta_sensitivity.csv"), rows,
              list(rows[0].keys()))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    p.add_argument("--flows", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("== E1: worker-count scaling ==")
    e1 = run_e1(args.out, args.flows, args.seed, workers=[1, 2, 4, 8, 16])
    for r in e1:
        print(r)

    print("\n== E2: input intensity (workers=4, the capacity-constrained knee) ==")
    e2 = run_e2(args.out, n_workers=4, flows=args.flows, seed=args.seed,
                interarrival_values=[0.03, 0.02, 0.015, 0.01, 0.007, 0.005])
    for r in e2:
        print(r)

    print("\n== E4: load balance (heterogeneous pool, workers=8) ==")
    e4 = run_e4(args.out, n_workers=8, flows=args.flows, seed=args.seed)
    for r in e4:
        print(r)

    print("\n== E5: fault injection (workers=4, w0 killed mid-run) ==")
    e5 = run_e5(args.out, n_workers=4, flows=args.flows, seed=args.seed,
                fault_points=[500, 1500, 2500])
    for r in e5:
        print(r)

    print("\n== E6: eta sensitivity (workers=4) ==")
    e6 = run_e6(args.out, n_workers_list=[4], flows=args.flows, seed=args.seed)
    for r in e6:
        print(r)

    print(f"\nAll CSVs written to {args.out}/")


if __name__ == "__main__":
    main()
