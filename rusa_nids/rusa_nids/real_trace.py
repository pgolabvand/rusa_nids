"""Real-trace workload adapter for CICIDS2017 / CSE-CIC-IDS2018 (Section 6
Stage 4 -- real-data re-validation, flagged in the paper's Threats to
Validity as the key outstanding gap before Q1 submission).

Design goal: drop-in replacement for ``FlowWorkloadGenerator`` in
simulate.py. ``run_simulation`` only calls ``next_interarrival()`` and
``make_flow(arrival_time)`` on its generator, so everything downstream
(``ClusterState``, ``RusaNidsPolicy``, the nested-CV calibration script)
is unchanged -- swapping the generator is the only integration point, by
construction (see simulate.py's own docstring).

No third-party dependency (no pandas): this project pins
``dependencies = []`` in pyproject.toml deliberately, and CICFlowMeter CSVs
are read perfectly well with the stdlib ``csv`` module at the row counts
these experiments need (a few hundred thousand rows per day-file, not the
full multi-GB corpus). If you later need to stream the *entire* raw
CICIDS2017/CSE-CIC-IDS2018 corpus without holding it in memory, swap
``_load_rows`` for a chunked reader -- the generator's public interface
does not change.

Three domain-modeling choices are NOT in the CSV and must be made
explicit (each is a paper-level assumption, not an engineering shortcut --
say so in Section 6/Threats to Validity when you report numbers from this
path):

1. estimated_cost_s (processing cost): CICFlowMeter's own "Flow Duration"
   is network flow duration, not NIDS processing cost. We rescale it
   (min-max, per load) into [min_cost_s, max_cost_s] -- the same
   normalize-don't-assume-units discipline the paper already applies to
   ru_hat (Methodology, "Scale-consistency fix"). Document min_cost_s/
   max_cost_s in the paper if you report from this path.
2. deadline_s: a detection deadline is an *operational SLA policy*, not a
   property of a captured packet trace -- CICIDS2017 doesn't and can't
   carry one. ``mean_deadline_s`` is a parameter you choose to match your
   deployment's actual freshness requirement, exactly as in the synthetic
   generator.
3. risk / uncertainty: the *correct* path is to run your trained detector
   (the CNN-XGBoost classifier from the thesis/first-paper work, or
   whatever production classifier this NIDS fronts) over each flow and
   feed its calibrated probability in via ``risk_col``/``uncertainty_col``
   -- that is what Section 5.1's "risk comes from a calibrated classifier"
   assumption actually requires. Label-derived pseudo-risk
   (``risk_col=None``) is provided ONLY as a fast sanity-check fallback;
   it is circular (it bakes ground truth into "risk") and MUST NOT be
   presented as a real-trace risk signal in the paper. ``used_label_proxy``
   on the generator records which path was taken so results carry the
   provenance.
"""

from __future__ import annotations

import csv
import glob
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

from rusa_nids.models import Flow

logger = logging.getLogger(__name__)

# CICFlowMeter's CSV header varies by release/tool version (leading spaces,
# casing, and the exact label column name all differ between CICIDS2017's
# original release and CSE-CIC-IDS2018's). Resolve by normalized-name
# aliasing instead of hardcoding one exact header string.
_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "timestamp": ("timestamp",),
    "duration": ("flowduration", "duration"),
    "label": ("label",),
}

# CICIDS2017 timestamps look like "5/7/2017 8:42" or "5/7/2017 8:42:12";
# CSE-CIC-IDS2018 day-files sometimes use "DD/MM/YYYY HH:MM:SS". Try both.
_TIMESTAMP_FORMATS = (
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
)


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


def _resolve_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    normalized = {_normalize(f): f for f in fieldnames}
    resolved: Dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                resolved[canonical] = normalized[alias]
                break
    missing = [c for c in ("duration", "label") if c not in resolved]
    if missing:
        raise ValueError(
            f"Could not find required column(s) {missing} in CSV header "
            f"{list(fieldnames)!r}. CICFlowMeter output expected -- check "
            "the file is a day-file export, not already-aggregated stats."
        )
    return resolved


def _parse_timestamp(raw: str) -> Optional[float]:
    raw = raw.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt).timestamp()
        except ValueError:
            continue
    return None


@dataclass
class _RawFlowRow:
    duration_raw: float
    label: str
    risk: Optional[float]
    uncertainty: Optional[float]
    timestamp: Optional[float]


def _load_rows(
    csv_paths: Sequence[str],
    risk_col: Optional[str] = None,
    uncertainty_col: Optional[str] = None,
    max_rows: Optional[int] = None,
    seed: int = 0,
) -> List[_RawFlowRow]:
    """Stream every day-file once. When ``max_rows`` caps the sample, use
    reservoir sampling (Algorithm R) instead of taking the first N rows --
    CICIDS2017/CSE-CIC-IDS2018 day-files are capture-ordered, so a head
    truncation systematically under-samples whatever attack phase starts
    later in the day (e.g. Friday's DDoS block sits well after the file's
    start) and would silently bias any (eta, slack) selection made on it.
    """
    rng = random.Random(seed)
    reservoir: List[_RawFlowRow] = []
    seen = 0
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            cols = _resolve_columns(reader.fieldnames)
            norm_fields = {_normalize(k): k for k in reader.fieldnames}
            risk_key = norm_fields.get(_normalize(risk_col)) if risk_col else None
            unc_key = norm_fields.get(_normalize(uncertainty_col)) if uncertainty_col else None
            for row in reader:
                try:
                    duration_raw = float(row[cols["duration"]])
                except (KeyError, ValueError):
                    continue
                ts = _parse_timestamp(row[cols["timestamp"]]) if "timestamp" in cols else None
                risk_val = None
                if risk_key is not None:
                    try:
                        risk_val = float(row[risk_key])
                    except (ValueError, KeyError):
                        risk_val = None
                unc_val = None
                if unc_key is not None:
                    try:
                        unc_val = float(row[unc_key])
                    except (ValueError, KeyError):
                        unc_val = None
                item = _RawFlowRow(
                    duration_raw=duration_raw,
                    label=row.get(cols["label"], "").strip(),
                    risk=risk_val,
                    uncertainty=unc_val,
                    timestamp=ts,
                )
                if max_rows is None:
                    reservoir.append(item)
                elif seen < max_rows:
                    reservoir.append(item)
                else:
                    j = rng.randint(0, seen)
                    if j < max_rows:
                        reservoir[j] = item
                seen += 1
    return reservoir


class TraceWorkloadGenerator:
    """Drop-in replacement for ``FlowWorkloadGenerator`` backed by one or
    more CICFlowMeter-derived CSV files (CICIDS2017 day-files, or
    CSE-CIC-IDS2018's per-day exports).

    Usage mirrors the synthetic generator exactly::

        gen = TraceWorkloadGenerator(
            csv_paths=glob.glob("data/CICIDS2017/Wednesday-*.csv"),
            risk_col=None,  # or "detector_prob" once you've scored the CSV
        )
        res = run_simulation(policy=policy, n_workers=2, n_flows=len(gen),
                              generator=gen, seed=1)

    Rows are sorted by their parsed timestamp where available (falling
    back to file order, which is CICFlowMeter's own capture order --
    already chronological within a day-file). Interarrival times are the
    *real* deltas between consecutive flows in the trace, compressed or
    stretched by ``time_scale`` if you want to replay a day-file faster
    than wall-clock (default 1.0 = use the trace's own relative timing).
    """

    def __init__(
        self,
        csv_paths: Iterable[str],
        risk_col: Optional[str] = None,
        uncertainty_col: Optional[str] = None,
        min_cost_s: float = 1e-4,
        max_cost_s: float = 0.2,
        mean_deadline_s: float = 0.5,
        time_scale: float = 1.0,
        max_rows: Optional[int] = None,
        attack_label_values: Sequence[str] = ("BENIGN",),
        seed: int = 0,
    ):
        paths = sorted({p for pattern in csv_paths for p in glob.glob(pattern)} or set(csv_paths))
        if not paths:
            raise FileNotFoundError(
                f"No files matched csv_paths={list(csv_paths)!r}. Point this "
                "at the CICFlowMeter CSV day-files (see Related Work / "
                "Experimental Setup for where to obtain them)."
            )
        self._seed = seed
        self._rng = random.Random(seed)
        self.mean_deadline_s = mean_deadline_s
        self.time_scale = time_scale
        self.used_label_proxy = risk_col is None
        if self.used_label_proxy:
            logger.warning(
                "TraceWorkloadGenerator: risk_col not given -- falling back "
                "to a label-derived pseudo-risk. This is circular (ground "
                "truth feeding 'risk') and is a sanity-check fallback ONLY. "
                "Do not report numbers from this path as real-trace risk "
                "calibration in the paper -- score the CSV with the actual "
                "detector first and pass risk_col."
            )

        raw_rows = _load_rows(paths, risk_col=risk_col, uncertainty_col=uncertainty_col, max_rows=max_rows, seed=seed)
        if not raw_rows:
            raise ValueError(f"Loaded 0 usable rows from {paths!r}.")

        # Chronological order where timestamps parsed; else keep file/capture order.
        if any(r.timestamp is not None for r in raw_rows):
            raw_rows.sort(key=lambda r: (r.timestamp is None, r.timestamp or 0.0))

        durations = [r.duration_raw for r in raw_rows]
        lo, hi = min(durations), max(durations)
        span = (hi - lo) or 1.0

        def rescale_cost(d: float) -> float:
            frac = (d - lo) / span
            return min_cost_s + frac * (max_cost_s - min_cost_s)

        self._flows_fifo: List[Dict] = []
        for r in raw_rows:
            is_attack = r.label.upper() not in {v.upper() for v in attack_label_values}
            if r.risk is not None:
                risk = max(0.0, min(1.0, r.risk))
            else:
                # Same-shaped pseudo-risk as the synthetic generator's
                # default attack/benign priors -- see module docstring
                # caveat; this is a fallback, not a measurement.
                risk = self._rng.betavariate(5, 1.5) if is_attack else self._rng.betavariate(1.2, 6)
            if r.uncertainty is not None:
                uncertainty = max(0.0, min(1.0, r.uncertainty))
            else:
                uncertainty = self._rng.betavariate(2, 2)
            self._flows_fifo.append(
                {
                    "cost": max(1e-4, rescale_cost(r.duration_raw)),
                    "risk": risk,
                    "uncertainty": uncertainty,
                    "attack_label": "attack" if is_attack else "benign",
                    "timestamp": r.timestamp,
                }
            )
        self._cursor = 0
        self._prev_ts = self._flows_fifo[0]["timestamp"]

    def reset(self) -> None:
        """Rewind to the first flow without re-reading/re-parsing the CSV
        files. Use this across a parameter sweep (same loaded trace, many
        (eta, slack) pairs) instead of constructing a new generator each
        time -- re-parsing a 500k-row day-file per grid point is the
        difference between a sweep finishing in seconds and not finishing
        at all.

        CRITICAL: must also re-seed ``self._rng`` to its original state.
        ``make_flow`` draws ``self._rng.expovariate(...)`` once per flow to
        synthesize ``deadline_s`` (CICIDS2017 carries no real deadline --
        see module docstring point 2), so without re-seeding here, every
        subsequent sweep point after the first replays against a
        *different, silently-drifting* set of synthetic deadlines instead
        of the same workload -- which breaks the entire comparability of a
        grid search that calls reset() between points (exactly what
        scripts/run_real_trace_experiment.py's objective() does, once per
        (eta, slack) grid point). Found and fixed as part of the same
        reproducibility audit that caught the seed-hash() bug (Section
        5.6.1) -- see that discussion for context.
        """
        self._cursor = 0
        self._prev_ts = self._flows_fifo[0]["timestamp"] if self._flows_fifo else None
        self._rng = random.Random(self._seed)

    def __len__(self) -> int:
        return len(self._flows_fifo)

    def next_interarrival(self) -> float:
        if self._cursor >= len(self._flows_fifo):
            return 0.0
        nxt = self._flows_fifo[self._cursor]
        ts = nxt["timestamp"]
        if ts is not None and self._prev_ts is not None:
            delta = max(0.0, (ts - self._prev_ts) * self.time_scale)
        else:
            delta = 1e-3  # no usable timestamps: treat as closely-spaced arrivals
        self._prev_ts = ts if ts is not None else self._prev_ts
        return delta

    def make_flow(self, arrival_time: float) -> Flow:
        row = self._flows_fifo[self._cursor]
        self._cursor += 1
        deadline = max(
            row["cost"] * 1.5, self._rng.expovariate(1.0 / self.mean_deadline_s)
        )
        return Flow(
            risk=row["risk"],
            uncertainty=row["uncertainty"],
            deadline_s=deadline,
            estimated_cost_s=row["cost"],
            arrival_time=arrival_time,
            attack_label=row["attack_label"],
        )
