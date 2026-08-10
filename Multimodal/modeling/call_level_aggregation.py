"""Call-level aggregation of segment-level vishing probabilities.

Track A.3 of the Paper 1 IEEE Access revision. Reviewer 2 asked for
empirical evaluation of the EMA and running-maximum streaming
aggregation rules the paper proposed in §3.3.2 / Algorithm 1 but never
implemented (paper main.tex L309 self-admits this is "future work").

This module implements both rules **exactly as the paper formulates
them**, with no extensions:

    EMA           : S_i = α · S_{i-1} + (1 - α) · p_i,   S_0 = 0    (main.tex L313)
    Running-Max   : S_i = max(S_{i-1}, p_i),             S_0 = 0    (main.tex L320)
    Alert         : flag the call when S_i ≥ τ                       (main.tex L327)
    Final label   : ŷ = 1[S_N ≥ τ]                                   (main.tex L353)

The paper's α convention: higher α means smoother / more conservative
(old state dominates). The running-max is global from call start —
there is no sliding window.

This is pure post-processing (no PyTorch, no GPU). It consumes a
predictions JSONL produced by run_segment_inference.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class CallResult:
    call_id: str
    label: int
    source: str
    n_segments: int
    duration: float                       # last segment's end time, seconds
    final_score: float                    # S_N
    final_decision: int                   # 1 if final_score ≥ τ else 0
    alerted: bool                         # True if S crossed τ at any segment
    alert_time: Optional[float]           # end_time of first alerting segment (None if never)
    alert_segment_index: Optional[int]    # 0-based index of first alerting segment

    def as_dict(self) -> dict:
        return asdict(self)


def load_predictions(path: str | Path) -> dict[str, list[dict]]:
    """Load a segment-predictions JSONL and group by call_id.

    Within each call, segments are sorted by `start` (ascending).
    Sorting is defensive — manifests appear time-ordered, but we don't
    rely on that for correctness.
    """
    by_call: dict[str, list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_call[row["call_id"]].append(row)

    for call_id in by_call:
        by_call[call_id].sort(key=lambda s: s["start"])
        # Sanity: monotone non-decreasing start times
        starts = [s["start"] for s in by_call[call_id]]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], (
                f"Non-monotone start times in {call_id}: {starts[i - 1]} -> {starts[i]}"
            )
    return dict(by_call)


def _finalize(call_id, segments, S, alerted, alert_idx, tau) -> CallResult:
    last = segments[-1]
    return CallResult(
        call_id=call_id,
        label=int(last["label"]),
        source=str(last.get("source", "unknown")),
        n_segments=len(segments),
        duration=float(max(s["end"] for s in segments)),
        final_score=float(S),
        final_decision=int(S >= tau),
        alerted=bool(alerted),
        alert_time=(float(segments[alert_idx]["end"]) if alert_idx is not None else None),
        alert_segment_index=alert_idx,
    )


def aggregate_ema(call_id: str, segments: list[dict], alpha: float, tau: float) -> CallResult:
    """Paper formulation: S_i = α·S_{i-1} + (1-α)·p_i, S_0 = 0."""
    assert segments, f"Empty call {call_id}"
    assert 0.0 <= alpha < 1.0, f"alpha out of range: {alpha}"
    S = 0.0
    alerted = False
    alert_idx: Optional[int] = None
    for i, seg in enumerate(segments):
        p = float(seg["prob"])
        S = alpha * S + (1.0 - alpha) * p
        if not alerted and S >= tau:
            alerted = True
            alert_idx = i
    return _finalize(call_id, segments, S, alerted, alert_idx, tau)


def aggregate_running_max(call_id: str, segments: list[dict], tau: float) -> CallResult:
    """Paper formulation: S_i = max(S_{i-1}, p_i), S_0 = 0. Global, not windowed."""
    assert segments, f"Empty call {call_id}"
    S = 0.0
    alerted = False
    alert_idx: Optional[int] = None
    for i, seg in enumerate(segments):
        p = float(seg["prob"])
        S = max(S, p)
        if not alerted and S >= tau:
            alerted = True
            alert_idx = i
    return _finalize(call_id, segments, S, alerted, alert_idx, tau)


def aggregate_majority_vote(call_id: str, segments: list[dict], tau: float = 0.5) -> CallResult:
    """Baseline: a call is positive iff mean(p_i) ≥ τ.

    Not part of the paper's proposal; included as a reference baseline
    for the comparison table in §4.
    """
    assert segments, f"Empty call {call_id}"
    probs = [float(s["prob"]) for s in segments]
    S = sum(probs) / len(probs)
    decision = int(S >= tau)
    # Majority vote has no streaming-alert semantics; we report it
    # as a final-decision-only baseline.
    return CallResult(
        call_id=call_id,
        label=int(segments[-1]["label"]),
        source=str(segments[-1].get("source", "unknown")),
        n_segments=len(segments),
        duration=float(max(s["end"] for s in segments)),
        final_score=float(S),
        final_decision=decision,
        alerted=bool(decision),
        alert_time=(float(max(s["end"] for s in segments)) if decision else None),
        alert_segment_index=(len(segments) - 1 if decision else None),
    )


def aggregate_all(
    by_call: dict[str, list[dict]],
    method: str,
    *,
    alpha: Optional[float] = None,
    tau: float = 0.5,
) -> list[CallResult]:
    if method == "ema":
        if alpha is None:
            raise ValueError("EMA requires alpha")
        return [aggregate_ema(c, segs, alpha, tau) for c, segs in by_call.items()]
    if method == "running_max":
        return [aggregate_running_max(c, segs, tau) for c, segs in by_call.items()]
    if method == "majority":
        return [aggregate_majority_vote(c, segs, tau) for c, segs in by_call.items()]
    raise ValueError(f"Unknown method: {method}")


# --- Self-test (run as script) -----------------------------------------------

def _selftest() -> None:
    """Sanity-check the aggregation rules against analytic expectations."""
    # All-zero probs: S should stay 0 forever
    zero_segs = [{"prob": 0.0, "start": i, "end": i + 1, "label": 0} for i in range(5)]
    r = aggregate_ema("z", zero_segs, alpha=0.5, tau=0.5)
    assert r.final_score == 0.0, r
    assert not r.alerted, r
    r = aggregate_running_max("z", zero_segs, tau=0.5)
    assert r.final_score == 0.0, r

    # All-one probs: EMA → 1 (geometric series limit), running-max → 1 immediately
    one_segs = [{"prob": 1.0, "start": i, "end": i + 1, "label": 1} for i in range(20)]
    r = aggregate_ema("o", one_segs, alpha=0.5, tau=0.5)
    assert abs(r.final_score - 1.0) < 1e-6, r.final_score
    assert r.alerted and r.alert_segment_index == 0, r  # 0.5*0 + 0.5*1 = 0.5 ≥ 0.5 on first seg
    r = aggregate_running_max("o", one_segs, tau=0.5)
    assert r.final_score == 1.0, r
    assert r.alert_segment_index == 0, r  # max(0, 1) = 1 ≥ 0.5 immediately

    # Conservative EMA (α=0.9) does NOT alert on a single 1.0 → S_1 = 0.1
    r = aggregate_ema("c", [{"prob": 1.0, "start": 0, "end": 1, "label": 1}], alpha=0.9, tau=0.5)
    assert r.final_score == pytest_approx(0.1), r.final_score
    assert not r.alerted, r

    # Running-max monotonicity: S only goes up
    mixed = [
        {"prob": 0.3, "start": 0, "end": 1, "label": 1},
        {"prob": 0.7, "start": 1, "end": 2, "label": 1},
        {"prob": 0.2, "start": 2, "end": 3, "label": 1},
        {"prob": 0.5, "start": 3, "end": 4, "label": 1},
    ]
    r = aggregate_running_max("m", mixed, tau=0.5)
    assert r.final_score == 0.7, r
    assert r.alert_segment_index == 1, r

    print("✅ self-test passed")


def pytest_approx(expected, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - expected) < tol
        def __repr__(self):
            return f"~{expected}"
    return _A()


if __name__ == "__main__":
    _selftest()
