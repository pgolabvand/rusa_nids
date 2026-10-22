# RUSA-NIDS prototype

Research prototype backing the Cluster Computing submission on
risk-/uncertainty-/resource-aware scheduling for cluster-based NIDS/NIPS.
This is the scheduling-logic layer only — see "What this is not" below.

## Layout

```
rusa_nids/
  models.py              Flow, Worker, ScheduleDecision (data model)
  schedulers/
    base.py               SchedulingPolicy interface (pure functions, no mutation)
    round_robin.py         baseline
    least_loaded.py        baseline
    resource_aware.py      baseline (ablation: resource-only)
    risk_priority.py       baseline (ablation: risk-only)
    rusa.py                 RUSA-NIDS (Section 5.2 scoring function)
  cluster.py              ClusterState: atomic allocation, queue mgmt,
                           starvation prevention, decision log, fault handling
  simulate.py              discrete-event simulator + synthetic workload generator
tests/test_cluster.py     correctness tests for every Section 5.3 guarantee
scripts/run_baselines_comparison.py   runnable E1-style sweep
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
python scripts/run_baselines_comparison.py --workers 1 2 4 8 --flows 2000
```

## Mapping to the research plan

| Research plan item | Where it lives |
|---|---|
| Decision variables (r_i, u_i, d_i, q_j, c_j, m_j, e_ij), Sec 5.1 | `models.py` (`Flow`, `Worker`) |
| Scoring function S(i,j), Sec 5.2 | `schedulers/rusa.py` |
| Atomic allocation, queue mgmt, starvation prevention, decision logging, timeout/backpressure, worker-failure behavior — explicitly flagged as missing from the paper's pseudocode, Sec 5.3 | `cluster.py` |
| Baselines (Round Robin, Least Loaded, Resource-aware, Risk-priority), Sec 6 Stage 2 | `schedulers/*.py` |
| E1 (worker scaling), E2 (input intensity), E4 (load balance) | `simulate.py::run_simulation` + `scripts/run_baselines_comparison.py` |
| E5 (fault injection) | `ClusterState.mark_worker_down` + `run_simulation(worker_fault_at_flow=...)` |
| E6 (ablation) | Run `resource_aware` and `risk_priority` as isolated-feature baselines; both are already implemented as separate policies, not RUSA-NIDS with terms zeroed, so they are honest ablations, not a post-hoc approximation |

## What this is NOT (read before citing results from it)

- **Not a live cluster.** `simulate.py` is a single-process discrete-event
  simulator with a synthetic Poisson/Beta workload generator. It validates
  *scheduler logic* (correctness, relative ordering of policies) cheaply
  and deterministically. The paper's actual E1–E7 numbers need this same
  `ClusterState` + policies driving real Ray workers against CICIDS2017/
  CSE-CIC-IDS2018 flow features — swap the generator and the completion
  callback for a real inference call; the scheduler/cluster code does not
  need to change.
- **`estimate_execution_time` is a placeholder cost model**
  (`Worker.estimate_execution_time` in `models.py`): flow cost scaled by
  resource pressure. This is the `e_ij` term the research plan explicitly
  flags as doing unstated work — before trusting E1/E2 numbers, replace it
  with a profiled or learned per-(worker, flow-class) cost model and
  report its own prediction error, per the earlier review.
- **Detection quality (Macro-F1, Recall, FPR) is out of scope here.**
  This package schedules; it does not classify. Wire in the actual
  detector so `Flow.risk`/`Flow.uncertainty` come from real model outputs,
  not the synthetic Beta-distribution stand-in in
  `FlowWorkloadGenerator.make_flow`.

## An early, honest finding from this prototype

Running the synthetic sweep (`scripts/run_baselines_comparison.py`) shows
RUSA-NIDS beating every baseline on completed-flows / dropped-deadline
**only in the capacity-constrained regime** (2–4 workers here). Once
workers ≥ 8 (i.e., utilization comfortably below 100%), RUSA-NIDS and
`resource_aware` converge to identical results, and `risk_priority`
converges to `least_loaded`. That is expected given how the synthetic
generator correlates risk and timing (it currently does not, deliberately,
to avoid baking in the result) — but it is also a direct warning for E2:
**if CICIDS2017's attack flows are not systematically more time-sensitive
than benign ones, the risk/urgency term may show near-zero benefit over
pure resource-aware scheduling at realistic utilization.** Decide now,
before running the real E1–E7 battery, whether that correlation should be
engineered into the deadline-assignment policy (e.g., attacks get tighter
SLAs by design, which is defensible operationally) or left to emerge from
the data — and report whichever you choose explicitly, since it changes
what the ablation (E6) is actually measuring.
