#!/usr/bin/env python3
"""Runnable E1/E2-style comparison across all baselines + RUSA-NIDS.

This is the first thing to run after `pip install -e .` to sanity-check
that the scheduler logic produces sane, differentiated numbers before
spending time on CICIDS2017 ingestion. It does NOT use real traffic data
-- see rusa_nids/simulate.py's FlowWorkloadGenerator docstring.

Usage:
    python scripts/run_baselines_comparison.py
    python scripts/run_baselines_comparison.py --workers 1 2 4 8 --flows 3000
"""

from __future__ import annotations

import argparse
import logging

from rusa_nids.schedulers import (
    LeastLoadedPolicy,
    ResourceAwarePolicy,
    RiskPriorityPolicy,
    RoundRobinPolicy,
    RusaNidsPolicy,
)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--flows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-flow drop/backpressure warnings from the scheduler audit log.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)

    header = (
        f"{'workers':>7} {'policy':>14} {'completed':>9} {'drop_dl':>7} "
        f"{'drop_q':>6} {'thrpt/s':>8} {'p50_ms':>7} {'p95_ms':>7} {'p99_ms':>7}"
    )
    print(header)
    print("-" * len(header))

    for n_workers in args.workers:
        for policy in build_policies(n_workers):
            gen = FlowWorkloadGenerator(seed=args.seed)
            result = run_simulation(
                policy=policy,
                n_workers=n_workers,
                n_flows=args.flows,
                generator=gen,
                seed=args.seed,
            )
            print(
                f"{n_workers:>7} {result.policy_name:>14} {result.completed:>9} "
                f"{result.dropped_deadline:>7} {result.dropped_queue_full:>6} "
                f"{result.throughput_flows_per_s:>8.1f} "
                f"{result.latency_p50_s * 1000:>7.2f} "
                f"{result.latency_p95_s * 1000:>7.2f} "
                f"{result.latency_p99_s * 1000:>7.2f}"
            )


if __name__ == "__main__":
    main()
