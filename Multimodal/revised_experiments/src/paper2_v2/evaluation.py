"""Call-primary Work 8 aggregation, metrics, and validation-only selection."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import (
    ArtifactRef,
    MatrixConfig,
    SelectionPolicy,
    SelectionSeal,
)


@dataclass(frozen=True)
class SegmentPrediction:
    segment_id: str
    call_id: str
    group_id: str
    label: int
    probability: float
    start: float
    end: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.segment_id or not self.call_id or not self.group_id:
            raise ValueError("prediction identifiers must be nonempty")
        if self.label not in {0, 1}:
            raise ValueError("label must be zero or one")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be finite and within [0,1]")
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.end < self.start:
            raise ValueError("segment times must be finite with end >= start")


@dataclass(frozen=True)
class CallPrediction:
    call_id: str
    group_id: str
    label: int
    probability: float
    n_segments: int
    start: float
    end: float
    source: str

    def decision(self, threshold: float) -> int:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0,1]")
        return int(self.probability >= threshold)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def aggregate_mean_probability(
    predictions: Iterable[SegmentPrediction],
) -> list[CallPrediction]:
    """Aggregate complete calls using the preregistered arithmetic mean.

    This is intentionally named ``mean_probability`` rather than the
    historical and inaccurate ``majority_vote`` label.
    """

    by_call: dict[str, list[SegmentPrediction]] = defaultdict(list)
    seen_segments: set[str] = set()
    for prediction in predictions:
        if prediction.segment_id in seen_segments:
            raise ValueError(f"duplicate segment prediction {prediction.segment_id!r}")
        seen_segments.add(prediction.segment_id)
        by_call[prediction.call_id].append(prediction)
    if not by_call:
        raise ValueError("cannot aggregate an empty prediction collection")

    calls: list[CallPrediction] = []
    for call_id, rows in sorted(by_call.items()):
        labels = {row.label for row in rows}
        groups = {row.group_id for row in rows}
        sources = {row.source for row in rows}
        if len(labels) != 1 or len(groups) != 1 or len(sources) != 1:
            raise ValueError(
                f"call {call_id!r} has inconsistent label/group/source metadata"
            )
        calls.append(
            CallPrediction(
                call_id=call_id,
                group_id=next(iter(groups)),
                label=next(iter(labels)),
                probability=float(np.mean([row.probability for row in rows], dtype=np.float64)),
                n_segments=len(rows),
                start=min(row.start for row in rows),
                end=max(row.end for row in rows),
                source=next(iter(sources)),
            )
        )
    return calls


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def call_level_nll(calls: Sequence[CallPrediction], *, epsilon: float = 1e-7) -> float:
    if not calls:
        raise ValueError("call NLL requires at least one call")
    labels = np.asarray([call.label for call in calls], dtype=np.float64)
    probabilities = np.clip(
        np.asarray([call.probability for call in calls], dtype=np.float64),
        epsilon,
        1.0 - epsilon,
    )
    losses = -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
    return float(losses.mean())


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    if bins <= 0:
        raise ValueError("ECE bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        if not np.any(selected):
            continue
        confidence = float(probabilities[selected].mean())
        frequency = float(labels[selected].mean())
        ece += float(selected.sum() / total) * abs(confidence - frequency)
    return float(ece)


def compute_call_metrics(
    calls: Sequence[CallPrediction], *, threshold: float
) -> dict[str, float | int | None]:
    if not calls:
        raise ValueError("metrics require at least one call")
    labels = np.asarray([call.label for call in calls], dtype=np.int64)
    probabilities = np.asarray([call.probability for call in calls], dtype=np.float64)
    decisions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((labels == 1) & (decisions == 1)))
    tn = int(np.sum((labels == 0) & (decisions == 0)))
    fp = int(np.sum((labels == 0) & (decisions == 1)))
    fn = int(np.sum((labels == 1) & (decisions == 0)))
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    negative_predictive_value = _safe_ratio(tn, tn + fn)
    positive_f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _safe_ratio(2 * tn, 2 * tn + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_ratio(tp * tn - fp * fn, denominator)
    has_both_labels = len(set(labels.tolist())) == 2
    return {
        "n_calls": len(calls),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": negative_predictive_value,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "macro_f1": float((positive_f1 + negative_f1) / 2.0),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "mcc": mcc,
        "fpr": float(1.0 - specificity),
        "fnr": float(1.0 - sensitivity),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_labels else None,
        "pr_auc": (
            float(average_precision_score(labels, probabilities)) if has_both_labels else None
        ),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": call_level_nll(calls),
        "ece_10": _expected_calibration_error(labels, probabilities, bins=10),
    }


def threshold_grid(policy: SelectionPolicy) -> tuple[float, ...]:
    start = Decimal(str(policy.threshold_start))
    end = Decimal(str(policy.threshold_end))
    step = Decimal(str(policy.threshold_step))
    values: list[float] = []
    current = start
    while current <= end:
        values.append(float(current))
        current += step
    return tuple(values)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metrics: dict[str, float | int | None]
    evaluated_thresholds: int


def select_validation_threshold(
    calls: Sequence[CallPrediction], policy: SelectionPolicy
) -> ThresholdSelection:
    """Select a threshold using validation calls only and deterministic ties."""

    candidates = []
    for threshold in threshold_grid(policy):
        metrics = compute_call_metrics(calls, threshold=threshold)
        key = (
            float(metrics["macro_f1"]),
            -float(metrics["fpr"]),
            float(metrics["sensitivity"]),
            -abs(threshold - 0.5),
            threshold,
        )
        candidates.append((key, threshold, metrics))
    _, threshold, metrics = max(candidates, key=lambda row: row[0])
    return ThresholdSelection(threshold, metrics, len(candidates))


@dataclass(frozen=True)
class EpochValidation:
    epoch: int
    validation_call_nll: float
    checkpoint: ArtifactRef
    validation_predictions_sha256: str

    def __post_init__(self) -> None:
        if self.epoch <= 0:
            raise ValueError("epoch must be positive")
        if not math.isfinite(self.validation_call_nll) or self.validation_call_nll < 0:
            raise ValueError("validation_call_nll must be finite and nonnegative")
        from .config import _check_sha256  # keep one canonical hash convention

        _check_sha256(self.validation_predictions_sha256, "validation_predictions_sha256")


def select_checkpoint(epochs: Sequence[EpochValidation]) -> EpochValidation:
    """Minimum validation call NLL, with the earlier epoch as exact tie-break."""

    if not epochs:
        raise ValueError("checkpoint selection requires validation epochs")
    epoch_numbers = [item.epoch for item in epochs]
    if len(set(epoch_numbers)) != len(epoch_numbers):
        raise ValueError("duplicate validation epoch")
    return min(epochs, key=lambda item: (item.validation_call_nll, item.epoch))


def make_selection_seal(
    config: MatrixConfig,
    *,
    condition: str,
    seed: int,
    partition_repeat: int,
    outer_fold: int,
    selected_checkpoint: EpochValidation,
    threshold_selection: ThresholdSelection,
) -> SelectionSeal:
    """Seal all validation choices before an outer-evaluation path can be requested."""

    config.condition(condition)
    if seed not in config.seeds:
        raise ValueError("seed is not preregistered")
    config.selection_view(partition_repeat, outer_fold)
    return SelectionSeal(
        protocol_id=config.protocol_id,
        partition_repeat=partition_repeat,
        outer_fold=outer_fold,
        condition=condition,
        seed=seed,
        checkpoint=selected_checkpoint.checkpoint,
        validation_predictions_sha256=selected_checkpoint.validation_predictions_sha256,
        threshold=threshold_selection.threshold,
    )
