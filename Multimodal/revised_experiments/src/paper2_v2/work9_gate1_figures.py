"""Precommitted post-outer figures for the Work 9 Gate-1 analysis.

This module is deliberately isolated from the frozen Work 9 runner and the
frozen Gate-1 analysis.  It can run only after Gate-1 has sealed a canonical
report directory.  It verifies that directory, the report schema, the report's
literal outer-ledger-index hash, and every call-ledger hash before rendering.

The renderer never opens checkpoints, segment ledgers, audio, transcripts, or
Phase-A validation ledgers.  Condition, fold/seed, confusion, and inferential
figures come from the canonical Gate-1 outputs.  Threshold-free ROC, PR,
calibration, and probability-distribution diagnostics use only the immutable
complete-call ledgers referenced by the Gate-1-validated outer index.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


FIGURE_SPEC_RELATIVE = Path(
    "revised_experiments/configs/work9_gate1_figure_spec_v1.json"
)
FIGURE_SCRIPT_RELATIVE = Path(
    "revised_experiments/scripts/generate_work9_gate1_figures.py"
)
FIGURE_SOURCE_RELATIVE = Path(
    "revised_experiments/src/paper2_v2/work9_gate1_figures.py"
)
FIGURE_TEST_RELATIVE = Path(
    "revised_experiments/tests/test_work9_gate1_figures.py"
)
REPORT_SCHEMA_RELATIVE = Path(
    "revised_experiments/registry/schemas/work9_gate1_report.schema.json"
)
PROTOCOL_ID = "paper2-v2-mfcc-matched-within-corpus-v1"
RUN_FAMILY = "primary_matrix"
SPEC_SCHEMA = "paper2-v2-work9-gate1-figure-spec-1"
MANIFEST_SCHEMA = "paper2-v2-work9-gate1-figure-manifest-1"
REPORT_SCHEMA = "paper2-v2-work9-gate1-report-1"
OUTER_INDEX_SCHEMA = "paper2-v2-work9-outer-ledger-index-1"
CONDITION_METRICS_SCHEMA = "paper2-v2-work9-gate1-condition-metrics-1"
COMPARISON_SCHEMA = "paper2-v2-work9-gate1-primary-comparison-1"
CALL_SCHEMA = "paper2-v2-work9-call-prediction-1"

CONDITIONS = (
    "T",
    "A-mean",
    "A-query",
    "A-temp",
    "AT-concat",
    "AT-gate",
    "AT-cross",
    "AT-cross-randtrain",
)
SEEDS = (17, 29, 43, 71, 101)
FOLDS = (0, 1, 2, 3, 4)
COMPARISONS = (
    "AT-cross_minus_A-query",
    "AT-cross_minus_AT-concat",
    "AT-cross_minus_AT-gate",
)
REPORTED_METRICS = (
    "macro_f1",
    "positive_f1",
    "sensitivity",
    "specificity",
    "precision",
    "negative_predictive_value",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
    "brier",
    "nll",
    "ece_10",
)
FORMATS = ("svg", "pdf", "png")
CLAIM_LABEL = (
    "Within-corpus outer-development evaluation — complete-call predictions"
)
DESCRIPTIVE_LABEL = (
    "Five neural seeds; seed/fold displays and curve envelopes are descriptive, not inferential"
)

COLORS = {
    "T": "#7656A8",
    "A-mean": "#2E86AB",
    "A-query": "#1976A3",
    "A-temp": "#155B82",
    "AT-concat": "#E38B2C",
    "AT-gate": "#D26428",
    "AT-cross": "#B63A5B",
    "AT-cross-randtrain": "#777777",
}
COMPARISON_COLORS = ("#B63A5B", "#D26428", "#E38B2C")

CALL_ROW_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "protocol_id",
        "run_family",
        "condition",
        "partition_repeat",
        "outer_fold",
        "seed",
        "partition",
        "call_id",
        "group_id",
        "label",
        "probability",
        "hard_prediction",
        "threshold",
        "n_segments",
        "aggregation_rule",
        "source",
        "start",
        "end",
        "duration_seconds",
        "checkpoint_sha256",
        "manifest_sha256",
    }
)


class Gate1FigureError(RuntimeError):
    """Raised when a provenance, schema, or figure-policy invariant fails."""


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    serialized_path: str
    sha256: str
    bytes: int
    row_count: int | None = None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "path": self.serialized_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class CallPrediction:
    condition: str
    outer_fold: int
    seed: int
    call_id: str
    group_id: str
    label: int
    probability: float
    hard_prediction: int
    threshold: float


@dataclass(frozen=True)
class Gate1FigureData:
    condition_records: tuple[Mapping[str, Any], ...]
    comparisons: tuple[Mapping[str, Any], ...]
    calls: tuple[CallPrediction, ...]
    gate1_hash_manifest: ArtifactReference | None = None
    gate1_artifacts: tuple[ArtifactReference, ...] = ()
    outer_index: ArtifactReference | None = None
    outer_call_ledgers: tuple[ArtifactReference, ...] = ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Gate1FigureError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: Any, context: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gate1FigureError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (lower is not None and result < lower):
        raise Gate1FigureError(f"{context} is non-finite or below its minimum")
    return result


def _safe_project_path(raw: str | Path, project_root: Path, context: str) -> Path:
    candidate = Path(raw)
    root = project_root.resolve()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Gate1FigureError(f"{context} escapes the project root") from error
    if path.is_symlink() or not path.is_file():
        raise Gate1FigureError(f"{context} is missing, symlinked, or not a file")
    return path


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise Gate1FigureError(f"file is outside project root: {path}") from error


def _file_reference(
    path: Path,
    project_root: Path,
    *,
    serialized_path: str | None = None,
    row_count: int | None = None,
) -> ArtifactReference:
    if not path.is_file() or path.is_symlink():
        raise Gate1FigureError(f"required regular file is missing or symlinked: {path}")
    return ArtifactReference(
        path=path.resolve(),
        serialized_path=serialized_path or _relative(path, project_root),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        row_count=row_count,
    )


def _artifact_reference(
    raw: Mapping[str, Any], project_root: Path, context: str
) -> ArtifactReference:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "bytes", "row_count"}:
        raise Gate1FigureError(f"{context} is not a canonical artifact reference")
    serialized = raw["path"]
    if not isinstance(serialized, str):
        raise Gate1FigureError(f"{context}.path must be a string")
    relative = Path(serialized)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise Gate1FigureError(f"{context}.path must be normalized and project-relative")
    path = _safe_project_path(relative, project_root, context)
    expected_sha = _require_sha256(raw["sha256"], f"{context}.sha256")
    if sha256_file(path) != expected_sha:
        raise Gate1FigureError(f"{context} SHA-256 mismatch")
    expected_bytes = raw["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise Gate1FigureError(f"{context}.bytes must be an integer")
    if path.stat().st_size != expected_bytes:
        raise Gate1FigureError(f"{context} byte-count mismatch")
    row_count = raw["row_count"]
    if row_count is not None and (
        isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0
    ):
        raise Gate1FigureError(f"{context}.row_count is invalid")
    return ArtifactReference(path, serialized, expected_sha, expected_bytes, row_count)


def _read_json(path: Path, context: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate1FigureError(f"cannot read {context}: {path}") from error


def _read_jsonl(path: Path, context: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Gate1FigureError(f"{context} contains a blank line")
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise Gate1FigureError(f"{context} row {row_number} is not an object")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate1FigureError(f"cannot read {context}: {path}") from error
    return rows


def load_figure_spec(project_root: str | Path) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    spec_path = root / FIGURE_SPEC_RELATIVE
    spec = _read_json(spec_path, "frozen Gate-1 figure specification")
    if not isinstance(spec, Mapping):
        raise Gate1FigureError("figure specification must be an object")
    if (
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("status") != "frozen_before_work9_outer_outcomes"
        or spec.get("protocol_id") != PROTOCOL_ID
    ):
        raise Gate1FigureError("figure specification identity/status mismatch")
    for section in ("upstream_bindings", "implementation_bindings"):
        rows = spec.get(section)
        if not isinstance(rows, list) or not rows:
            raise Gate1FigureError(f"figure specification {section} is empty")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {"role", "path", "sha256"}:
                raise Gate1FigureError(f"{section}[{index}] is malformed")
            path = _safe_project_path(str(row["path"]), root, f"{section}[{index}]")
            expected = _require_sha256(row["sha256"], f"{section}[{index}].sha256")
            if sha256_file(path) != expected:
                raise Gate1FigureError(f"frozen binding changed: {row['role']}")
    figures = spec.get("figures")
    if not isinstance(figures, list) or [row.get("figure_id") for row in figures] != list(
        expected_figure_ids()
    ):
        raise Gate1FigureError("frozen figure family/order changed")
    if spec.get("data_access", {}).get("checkpoints_permitted") is not False:
        raise Gate1FigureError("figure spec must forbid checkpoint access")
    if spec.get("data_access", {}).get("outer_segment_ledgers_permitted") is not False:
        raise Gate1FigureError("figure spec must forbid segment-ledger access")
    return spec


def _parse_hash_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for row_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise Gate1FigureError(f"Gate-1 hashes manifest row {row_number} is malformed")
        digest = _require_sha256(parts[0], f"Gate-1 hashes row {row_number}")
        name = parts[1]
        if not name or Path(name).is_absolute() or ".." in Path(name).parts or name in entries:
            raise Gate1FigureError(f"Gate-1 hashes row {row_number} has an unsafe name")
        entries[name] = digest
    expected = {
        "REPORT.md",
        "READ_ONLY_NOTICE.txt",
        "condition_metrics.jsonl",
        "primary_comparisons.jsonl",
        "protocol.json",
        "report.json",
        "stability.json",
        "validation_audit.json",
    }
    if set(entries) != expected:
        raise Gate1FigureError("Gate-1 hashes manifest file set changed")
    return entries


def _validate_condition_records(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (condition, seed, fold)
        for condition in CONDITIONS
        for seed in SEEDS
        for fold in (None, *FOLDS)
    }
    observed: set[tuple[str, int, int | None]] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != CONDITION_METRICS_SCHEMA:
            raise Gate1FigureError(f"condition metric row {index} has wrong schema")
        condition = row.get("condition")
        seed = row.get("seed")
        fold = row.get("outer_fold")
        coordinate = (condition, seed, fold)
        if coordinate not in expected or coordinate in observed:
            raise Gate1FigureError(f"unexpected/duplicate condition metric coordinate {coordinate}")
        observed.add(coordinate)  # type: ignore[arg-type]
        expected_scope = "pooled_five_outer_folds_per_seed" if fold is None else "outer_fold_per_seed"
        if row.get("scope") != expected_scope:
            raise Gate1FigureError(f"condition metric row {coordinate} has wrong scope")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise Gate1FigureError(f"condition metric row {coordinate} lacks metrics")
        if set(metrics) != {"n_calls", "confusion", *REPORTED_METRICS}:
            raise Gate1FigureError(f"condition metric row {coordinate} metric keys changed")
    if observed != expected:
        raise Gate1FigureError("condition metrics do not contain the exact 240 records")


def _validate_comparisons(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 3 or tuple(row.get("comparison_id") for row in rows) != COMPARISONS:
        raise Gate1FigureError("primary comparison family/order changed")
    for index, row in enumerate(rows):
        if row.get("schema_version") != COMPARISON_SCHEMA or row.get("comparison_index") != index:
            raise Gate1FigureError(f"primary comparison {index} schema/index mismatch")
        bootstrap = row.get("bootstrap")
        multiplicity = row.get("multiplicity")
        if not isinstance(bootstrap, Mapping) or not isinstance(multiplicity, Mapping):
            raise Gate1FigureError(f"primary comparison {index} lacks inference objects")
        if (
            bootstrap.get("method")
            != "paired_relation_group_cluster_keep_all_calls_segments_seeds"
            or bootstrap.get("replicates") != 10000
            or bootstrap.get("confidence_level") != 0.95
            or multiplicity.get("method") != "holm"
            or multiplicity.get("family_size") != 3
            or multiplicity.get("familywise_alpha") != 0.05
        ):
            raise Gate1FigureError(f"primary comparison {index} inference policy changed")


def _ece_10(labels: np.ndarray, probabilities: np.ndarray) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        selected = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == 9
            else (probabilities >= lower) & (probabilities < upper)
        )
        if np.any(selected):
            result += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return float(result)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else float(numerator / denominator)


def compute_call_metrics(calls: Sequence[CallPrediction]) -> dict[str, Any]:
    if not calls:
        raise Gate1FigureError("metric recomputation requires calls")
    labels = np.asarray([call.label for call in calls], dtype=np.int64)
    probabilities = np.asarray([call.probability for call in calls], dtype=np.float64)
    predictions = np.asarray([call.hard_prediction for call in calls], dtype=np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    positive_f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _ratio(2 * tn, 2 * tn + fp + fn)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    both_labels = len(set(labels.tolist())) == 2
    return {
        "n_calls": len(calls),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "macro_f1": float((positive_f1 + negative_f1) / 2.0),
        "positive_f1": positive_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": npv,
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "mcc": _ratio(tp * tn - fp * fn, mcc_denominator),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_labels else None,
        "pr_auc": float(average_precision_score(labels, probabilities)) if both_labels else None,
        "fpr": float(1.0 - specificity),
        "fnr": float(1.0 - sensitivity),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))),
        "ece_10": _ece_10(labels, probabilities),
    }


def _assert_metric_match(actual: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    if actual["n_calls"] != expected.get("n_calls") or actual["confusion"] != expected.get("confusion"):
        raise Gate1FigureError(f"{context} count/confusion mismatch")
    for metric in REPORTED_METRICS:
        left, right = actual[metric], expected.get(metric)
        if left is None or right is None:
            if left is not None or right is not None:
                raise Gate1FigureError(f"{context}.{metric} definedness mismatch")
        elif not math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-12):
            raise Gate1FigureError(f"{context}.{metric} mismatch")


def _load_outer_calls(
    outer_index_path: Path,
    project_root: Path,
) -> tuple[tuple[CallPrediction, ...], tuple[ArtifactReference, ...]]:
    outer = _read_json(outer_index_path, "frozen outer-ledger index")
    if not isinstance(outer, Mapping) or (
        outer.get("schema_version") != OUTER_INDEX_SCHEMA
        or outer.get("protocol_id") != PROTOCOL_ID
        or outer.get("run_family") != RUN_FAMILY
        or outer.get("status") != "frozen_complete"
        or outer.get("expected_coordinate_count") != 200
    ):
        raise Gate1FigureError("outer-ledger index identity/status mismatch")
    entries = outer.get("entries")
    if not isinstance(entries, list) or len(entries) != 200:
        raise Gate1FigureError("outer-ledger index must have exactly 200 entries")
    expected = {
        (condition, fold, seed)
        for condition in CONDITIONS
        for fold in FOLDS
        for seed in SEEDS
    }
    observed: set[tuple[str, int, int]] = set()
    all_calls: list[CallPrediction] = []
    ledger_refs: list[ArtifactReference] = []
    universe: dict[tuple[int, str], tuple[int, str]] = {}
    coordinate_call_sets: dict[tuple[str, int, int], set[str]] = {}
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise Gate1FigureError(f"outer index entry {entry_index} is not an object")
        coordinate = (entry.get("condition"), entry.get("outer_fold"), entry.get("seed"))
        if entry.get("partition_repeat") != 0 or coordinate not in expected or coordinate in observed:
            raise Gate1FigureError(f"unexpected/duplicate outer coordinate {coordinate}")
        observed.add(coordinate)  # type: ignore[arg-type]
        ref = _artifact_reference(
            entry.get("outer_eval_call_ledger"),
            project_root,
            f"outer call ledger {coordinate}",
        )
        rows = _read_jsonl(ref.path, f"outer call ledger {coordinate}")
        if ref.row_count != len(rows):
            raise Gate1FigureError(f"outer call ledger {coordinate} row-count mismatch")
        ledger_refs.append(ref)
        call_ids: set[str] = set()
        for row_number, row in enumerate(rows, start=1):
            context = f"outer call ledger {coordinate} row {row_number}"
            if set(row) != CALL_ROW_KEYS or row.get("schema_version") != CALL_SCHEMA:
                raise Gate1FigureError(f"{context} schema/keys mismatch")
            if (
                row.get("condition") != coordinate[0]
                or row.get("outer_fold") != coordinate[1]
                or row.get("seed") != coordinate[2]
                or row.get("partition_repeat") != 0
                or row.get("partition") != "outer_eval"
                or row.get("protocol_id") != PROTOCOL_ID
                or row.get("run_family") != RUN_FAMILY
                or row.get("aggregation_rule") != "mean_probability"
            ):
                raise Gate1FigureError(f"{context} coordinate/protocol mismatch")
            call_id = row.get("call_id")
            group_id = row.get("group_id")
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                raise Gate1FigureError(f"{context} call_id is invalid/duplicate")
            if not isinstance(group_id, str) or not group_id:
                raise Gate1FigureError(f"{context} group_id is invalid")
            call_ids.add(call_id)
            label = row.get("label")
            hard = row.get("hard_prediction")
            if label not in (0, 1) or hard not in (0, 1):
                raise Gate1FigureError(f"{context} label/prediction is not binary")
            probability = _finite(row.get("probability"), f"{context}.probability", lower=0.0)
            threshold = _finite(row.get("threshold"), f"{context}.threshold", lower=0.0)
            if probability > 1.0 or threshold > 1.0 or hard != int(probability >= threshold):
                raise Gate1FigureError(f"{context} probability/threshold/prediction mismatch")
            identity = (coordinate[1], call_id)
            metadata = (int(label), group_id)
            if identity in universe and universe[identity] != metadata:
                raise Gate1FigureError(f"{context} label/group metadata changed across models")
            universe[identity] = metadata
            all_calls.append(
                CallPrediction(
                    condition=str(coordinate[0]),
                    outer_fold=int(coordinate[1]),
                    seed=int(coordinate[2]),
                    call_id=call_id,
                    group_id=group_id,
                    label=int(label),
                    probability=probability,
                    hard_prediction=int(hard),
                    threshold=threshold,
                )
            )
        coordinate_call_sets[coordinate] = call_ids  # type: ignore[index]
    if observed != expected:
        raise Gate1FigureError("outer call-ledger coordinate grid is incomplete")
    for fold in FOLDS:
        reference = coordinate_call_sets[(CONDITIONS[0], fold, SEEDS[0])]
        for condition in CONDITIONS:
            for seed in SEEDS:
                if coordinate_call_sets[(condition, fold, seed)] != reference:
                    raise Gate1FigureError(f"call universe differs in outer fold {fold}")
    return tuple(all_calls), tuple(ledger_refs)


def load_gate1_figure_data(
    project_root: str | Path,
    *,
    gate1_directory: str | Path,
    expected_gate1_hash_manifest_sha256: str,
) -> Gate1FigureData:
    root = Path(project_root).resolve()
    load_figure_spec(root)
    gate1 = Path(gate1_directory).resolve()
    try:
        gate1.relative_to(root)
    except ValueError as error:
        raise Gate1FigureError("Gate-1 directory escapes the project root") from error
    if not gate1.is_dir() or gate1.is_symlink():
        raise Gate1FigureError("Gate-1 directory is missing or symlinked")
    hashes_path = gate1 / "hashes.sha256"
    expected_hash = _require_sha256(
        expected_gate1_hash_manifest_sha256,
        "expected Gate-1 hashes.sha256 digest",
    )
    if sha256_file(hashes_path) != expected_hash:
        raise Gate1FigureError("Gate-1 hashes.sha256 literal digest mismatch")
    hash_entries = _parse_hash_manifest(hashes_path)
    artifacts: list[ArtifactReference] = []
    for name, digest in sorted(hash_entries.items()):
        path = _safe_project_path(gate1 / name, root, f"Gate-1 artifact {name}")
        if sha256_file(path) != digest:
            raise Gate1FigureError(f"Gate-1 artifact hash mismatch: {name}")
        artifacts.append(_file_reference(path, root))

    report_path = gate1 / "report.json"
    report = _read_json(report_path, "Gate-1 report")
    if not isinstance(report, Mapping) or (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "pass"
        or report.get("within_corpus_only") is not True
        or report.get("protocol_id") != PROTOCOL_ID
    ):
        raise Gate1FigureError("Gate-1 report identity/status mismatch")
    try:
        import jsonschema

        schema = _read_json(root / REPORT_SCHEMA_RELATIVE, "Gate-1 report schema")
        jsonschema.Draft202012Validator(schema).validate(report)
    except ImportError as error:
        raise Gate1FigureError("jsonschema is required for Gate-1 figure generation") from error
    except jsonschema.ValidationError as error:
        raise Gate1FigureError(f"Gate-1 report schema validation failed: {error.message}") from error

    condition_rows = _read_jsonl(gate1 / "condition_metrics.jsonl", "Gate-1 condition metrics")
    comparison_rows = _read_jsonl(gate1 / "primary_comparisons.jsonl", "Gate-1 comparisons")
    _validate_condition_records(condition_rows)
    _validate_comparisons(comparison_rows)
    if report.get("primary_comparisons") != comparison_rows:
        raise Gate1FigureError("report and comparison JSONL differ")

    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("outer_ledger_index"), Mapping):
        raise Gate1FigureError("Gate-1 report lacks its outer-ledger index reference")
    outer_input = inputs["outer_ledger_index"]
    if set(outer_input) != {"path", "sha256"}:
        raise Gate1FigureError("Gate-1 report outer-index reference changed")
    outer_path = _safe_project_path(str(outer_input["path"]), root, "Gate-1 outer-ledger index")
    outer_sha = _require_sha256(outer_input["sha256"], "Gate-1 report outer-index SHA")
    if sha256_file(outer_path) != outer_sha:
        raise Gate1FigureError("Gate-1 report outer-index SHA mismatch")
    calls, ledger_refs = _load_outer_calls(outer_path, root)

    record_lookup = {
        (str(row["condition"]), int(row["seed"]), row["outer_fold"]): row["metrics"]
        for row in condition_rows
    }
    calls_lookup: dict[tuple[str, int, int], list[CallPrediction]] = defaultdict(list)
    for call in calls:
        calls_lookup[(call.condition, call.seed, call.outer_fold)].append(call)
    for condition in CONDITIONS:
        for seed in SEEDS:
            pooled: list[CallPrediction] = []
            for fold in FOLDS:
                coordinate_calls = calls_lookup[(condition, seed, fold)]
                pooled.extend(coordinate_calls)
                _assert_metric_match(
                    compute_call_metrics(coordinate_calls),
                    record_lookup[(condition, seed, fold)],
                    f"condition metric {(condition, seed, fold)}",
                )
            _assert_metric_match(
                compute_call_metrics(pooled),
                record_lookup[(condition, seed, None)],
                f"condition metric {(condition, seed, None)}",
            )

    return Gate1FigureData(
        condition_records=tuple(condition_rows),
        comparisons=tuple(comparison_rows),
        calls=calls,
        gate1_hash_manifest=_file_reference(hashes_path, root),
        gate1_artifacts=tuple(artifacts),
        outer_index=_file_reference(outer_path, root),
        outer_call_ledgers=ledger_refs,
    )


def expected_figure_ids() -> tuple[str, ...]:
    return (
        "outer_condition_macro_f1",
        "outer_condition_metric_profile",
        "gate1_primary_effects",
        "gate1_seed_stability",
        "gate1_fold_stability",
        "outer_macro_f1_fold_seed",
        "outer_confusion_matrices",
        "outer_roc_curves",
        "outer_pr_curves",
        "outer_calibration_curves",
        "outer_probability_distributions",
    )


def _condition_lookup(data: Gate1FigureData) -> dict[tuple[str, int, int | None], Mapping[str, Any]]:
    return {
        (str(row["condition"]), int(row["seed"]), row["outer_fold"]): row["metrics"]
        for row in data.condition_records
    }


def _calls_lookup(data: Gate1FigureData) -> dict[tuple[str, int], list[CallPrediction]]:
    result: dict[tuple[str, int], list[CallPrediction]] = defaultdict(list)
    for call in data.calls:
        result[(call.condition, call.seed)].append(call)
    return result


def _footer(fig: plt.Figure, *, inferential: bool = False) -> None:
    fig.text(0.01, 0.036, CLAIM_LABEL, fontsize=8.3, color="#333333")
    second = (
        "Only the displayed 95% CIs are relation-group clustered; Holm p-values control the frozen three-test family"
        if inferential
        else DESCRIPTIVE_LABEL
    )
    fig.text(0.01, 0.012, second, fontsize=8.0, color="#555555")


def _save_figure(
    fig: plt.Figure, base: Path, formats: Sequence[str]
) -> tuple[Path, ...]:
    outputs: list[Path] = []
    for extension in formats:
        path = base.with_suffix(f".{extension}")
        metadata: dict[str, Any] = {"Creator": "Paper2 Work9 Gate1 figure pipeline v1"}
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        outputs.append(path)
    plt.close(fig)
    return tuple(outputs)


def _condition_macro_f1(data: Gate1FigureData) -> plt.Figure:
    lookup = _condition_lookup(data)
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for index, condition in enumerate(CONDITIONS):
        values = np.asarray([lookup[(condition, seed, None)]["macro_f1"] for seed in SEEDS], dtype=float)
        offsets = np.linspace(-0.12, 0.12, len(SEEDS))
        ax.scatter(values, index + offsets, s=28, color=COLORS[condition], alpha=0.55, edgecolor="none")
        ax.scatter([values.mean()], [index], marker="D", s=70, color=COLORS[condition], edgecolor="black", linewidth=0.6, zorder=3)
        ax.plot([values.min(), values.max()], [index, index], color=COLORS[condition], linewidth=2, alpha=0.75, zorder=1)
    ax.set_yticks(range(len(CONDITIONS)), CONDITIONS)
    ax.invert_yaxis()
    ax.set_xlim(-0.01, 1.01)
    ax.set_xlabel("Call-level macro-F1")
    ax.set_title("Outer-development call macro-F1 across five frozen seeds", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.text(0.99, 0.965, "diamonds: seed mean; bars: seed min–max", transform=ax.transAxes, ha="right", fontsize=8.5)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def _metric_profile(data: Gate1FigureData) -> plt.Figure:
    lookup = _condition_lookup(data)
    panels = (
        ("macro_f1", "Macro-F1", "higher ↑"),
        ("sensitivity", "Sensitivity", "higher ↑"),
        ("specificity", "Specificity", "higher ↑"),
        ("roc_auc", "ROC AUC", "higher ↑"),
        ("pr_auc", "Average precision (AP)", "higher ↑"),
        ("brier", "Brier score", "lower ↓"),
        ("nll", "Negative log loss", "lower ↓"),
        ("ece_10", "ECE (10 bins)", "lower ↓"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.8), sharex=True)
    x = np.arange(len(CONDITIONS))
    for ax, (metric, title, direction) in zip(axes.flat, panels, strict=True):
        for index, condition in enumerate(CONDITIONS):
            values = np.asarray(
                [lookup[(condition, seed, None)][metric] for seed in SEEDS if lookup[(condition, seed, None)][metric] is not None],
                dtype=float,
            )
            if len(values):
                ax.plot([index, index], [values.min(), values.max()], color=COLORS[condition], linewidth=2)
                ax.scatter(index, values.mean(), marker="D", s=38, color=COLORS[condition], edgecolor="black", linewidth=0.45, zorder=3)
        ax.set_title(f"{title} ({direction})", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x, CONDITIONS, rotation=55, ha="right", fontsize=7.5)
        if metric not in ("nll",):
            ax.set_ylim(bottom=0.0)
    fig.suptitle("Outer-development condition profile: seed mean and min–max", x=0.02, ha="left", fontweight="bold", fontsize=14)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def _primary_effects(data: Gate1FigureData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    labels = (
        "vs A-query\n(incremental test)",
        "vs AT-concat\n(mechanism 1/2)",
        "vs AT-gate\n(mechanism 2/2)",
    )
    y = np.arange(3)
    for index, row in enumerate(data.comparisons):
        observed = float(row["observed_effect"])
        lower = float(row["bootstrap"]["ci_lower"])
        upper = float(row["bootstrap"]["ci_upper"])
        ax.errorbar(
            observed,
            index,
            xerr=[[observed - lower], [upper - observed]],
            fmt="D",
            markersize=7,
            capsize=5,
            color=COMPARISON_COLORS[index],
            ecolor=COMPARISON_COLORS[index],
            linewidth=2.2,
        )
        holm_p = float(row["multiplicity"]["adjusted_p_value"])
        ax.text(
            0.99,
            index,
            f"Holm p={holm_p:.4g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=9,
        )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.axhline(0.5, color="#777777", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired macro-F1 difference (AT-cross − comparator)")
    ax.set_title("Preregistered Gate-1 effects with 95% relation-group clustered bootstrap CIs", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    _footer(fig, inferential=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _seed_stability(data: Gate1FigureData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for index, row in enumerate(data.comparisons):
        values = [float(row["seed_stability"]["per_seed_differences"][str(seed)]) for seed in SEEDS]
        ax.plot(SEEDS, values, marker="o", linewidth=2, color=COMPARISON_COLORS[index], label=row["comparison_id"].replace("AT-cross_minus_", "vs "))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(SEEDS)
    ax.set_xlabel("Frozen neural seed")
    ax.set_ylabel("Paired macro-F1 difference")
    ax.set_title("Gate-1 effect stability across seeds", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=8.5)
    ax.grid(alpha=0.2)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _fold_stability(data: Gate1FigureData) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharex=True, sharey=True)
    image = None
    for comparison_index, (ax, row) in enumerate(zip(axes, data.comparisons, strict=True)):
        fold_data = row["fold_stability"]["per_fold_per_seed_differences"]
        matrix = np.asarray([fold_data[str(fold)] for fold in FOLDS], dtype=float)
        maximum = max(0.02, float(np.max(np.abs(matrix))))
        image = ax.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-maximum,
            vmax=maximum,
            aspect="auto",
        )
        for fold in FOLDS:
            for seed_index in range(5):
                value = matrix[fold, seed_index]
                ax.text(
                    seed_index,
                    fold,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.62 * maximum else "black",
                )
        ax.set_xticks(range(5), SEEDS, rotation=35)
        ax.set_yticks(range(5), FOLDS)
        ax.set_xlabel("Seed")
        ax.set_title(row["comparison_id"].replace("AT-cross_minus_", "vs "), fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Outer fold")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.055, label="Macro-F1 difference")
    fig.suptitle("Gate-1 effect stability across folds and seeds", x=0.02, ha="left", fontweight="bold", fontsize=14)
    _footer(fig)
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.2, top=0.86, wspace=0.25)
    return fig


def _macro_f1_heatmaps(data: Gate1FigureData) -> plt.Figure:
    lookup = _condition_lookup(data)
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.7), constrained_layout=False)
    image = None
    for ax, condition in zip(axes.flat, CONDITIONS, strict=True):
        matrix = np.asarray(
            [[lookup[(condition, seed, fold)]["macro_f1"] for seed in SEEDS] for fold in FOLDS],
            dtype=float,
        )
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        for fold in range(5):
            for seed_index in range(5):
                color = "white" if matrix[fold, seed_index] < 0.48 else "black"
                ax.text(seed_index, fold, f"{matrix[fold, seed_index]:.2f}", ha="center", va="center", fontsize=7, color=color)
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.set_xticks(range(5), SEEDS, rotation=35)
        ax.set_yticks(range(5), FOLDS)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Outer fold")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.05, label="Macro-F1")
    fig.suptitle("Call-level macro-F1 by outer fold and seed", x=0.02, ha="left", fontweight="bold", fontsize=14)
    _footer(fig)
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.14, top=0.9, wspace=0.28, hspace=0.38)
    return fig


def _confusion_matrices(data: Gate1FigureData) -> plt.Figure:
    lookup = _condition_lookup(data)
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.4))
    image = None
    for ax, condition in zip(axes.flat, CONDITIONS, strict=True):
        matrices: list[np.ndarray] = []
        for seed in SEEDS:
            confusion = lookup[(condition, seed, None)]["confusion"]
            matrix = np.asarray([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]], dtype=float)
            denominators = matrix.sum(axis=1, keepdims=True)
            matrices.append(np.divide(matrix, denominators, out=np.zeros_like(matrix), where=denominators != 0))
        mean_matrix = np.mean(matrices, axis=0)
        image = ax.imshow(mean_matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, f"{100 * mean_matrix[row, column]:.1f}%", ha="center", va="center", color="white" if mean_matrix[row, column] > 0.55 else "black", fontweight="bold")
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.set_xticks((0, 1), ("Pred 0", "Pred 1"))
        ax.set_yticks((0, 1), ("True 0", "True 1"))
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.05, label="Mean row-normalized proportion")
    fig.suptitle("Mean true-label-normalized confusion matrices across seeds (fold-specific sealed thresholds)", x=0.02, ha="left", fontweight="bold", fontsize=14)
    _footer(fig)
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.14, top=0.9, wspace=0.3, hspace=0.4)
    return fig


def _curve_summary(data: Gate1FigureData, *, kind: str) -> plt.Figure:
    calls = _calls_lookup(data)
    grid = np.linspace(0.0, 1.0, 201)
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    for condition in CONDITIONS:
        interpolated: list[np.ndarray] = []
        for seed in SEEDS:
            seed_calls = calls[(condition, seed)]
            labels = np.asarray([call.label for call in seed_calls], dtype=int)
            probabilities = np.asarray([call.probability for call in seed_calls], dtype=float)
            if len(np.unique(labels)) != 2:
                raise Gate1FigureError(f"{kind} curve undefined for {condition}, seed {seed}")
            if kind == "roc":
                x, y, _ = roc_curve(labels, probabilities)
            else:
                precision, recall, _ = precision_recall_curve(labels, probabilities)
                x, y = recall[::-1], precision[::-1]
                unique_x, unique_indices = np.unique(x, return_index=True)
                x, y = unique_x, y[unique_indices]
            interpolated.append(np.interp(grid, x, y))
        curves = np.vstack(interpolated)
        mean = curves.mean(axis=0)
        ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=COLORS[condition], alpha=0.07)
        ax.plot(grid, mean, color=COLORS[condition], linewidth=2, label=condition)
    if kind == "roc":
        ax.plot((0, 1), (0, 1), color="black", linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        title = "Outer-development ROC curves: mean and seed min–max envelope"
    else:
        prevalence = np.mean(
            [
                call.label
                for call in calls[(CONDITIONS[0], SEEDS[0])]
            ]
        )
        ax.axhline(prevalence, color="black", linestyle="--", linewidth=1, label="prevalence")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        title = "Outer-development precision–recall curves: mean and seed min–max envelope"
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _calibration_curves(data: Gate1FigureData) -> plt.Figure:
    calls = _calls_lookup(data)
    edges = np.linspace(0.0, 1.0, 11)
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.7), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, CONDITIONS, strict=True):
        seed_bin_x: list[list[float | None]] = []
        seed_bin_y: list[list[float | None]] = []
        for seed in SEEDS:
            seed_calls = calls[(condition, seed)]
            probabilities = np.asarray([call.probability for call in seed_calls], dtype=float)
            labels = np.asarray([call.label for call in seed_calls], dtype=int)
            indices = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), 9)
            per_bin_x: list[float | None] = []
            per_bin_y: list[float | None] = []
            for bin_index in range(10):
                selected = indices == bin_index
                if np.any(selected):
                    per_bin_x.append(float(probabilities[selected].mean()))
                    per_bin_y.append(float(labels[selected].mean()))
                else:
                    per_bin_x.append(None)
                    per_bin_y.append(None)
            seed_bin_x.append(per_bin_x)
            seed_bin_y.append(per_bin_y)
            seed_x = [value for value in per_bin_x if value is not None]
            seed_y = [
                per_bin_y[index]
                for index, value in enumerate(per_bin_x)
                if value is not None
            ]
            ax.plot(seed_x, seed_y, color=COLORS[condition], alpha=0.16, linewidth=1)
        x, y = [], []
        for bin_index in range(10):
            xs = [row[bin_index] for row in seed_bin_x if row[bin_index] is not None]
            ys = [row[bin_index] for row in seed_bin_y if row[bin_index] is not None]
            if xs:
                x.append(float(np.mean(xs)))
                y.append(float(np.mean(ys)))
        ax.plot(x, y, marker="o", color=COLORS[condition], linewidth=2)
        ax.plot((0, 1), (0, 1), color="black", linestyle="--", linewidth=0.8)
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.grid(alpha=0.16)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive fraction")
    fig.suptitle("Fixed 10-bin calibration diagnostics (five seed traces and unweighted bin-wise seed mean)", x=0.02, ha="left", fontweight="bold", fontsize=14)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def _probability_distributions(data: Gate1FigureData) -> plt.Figure:
    calls = _calls_lookup(data)
    bins = np.linspace(0.0, 1.0, 21)
    centers = (bins[:-1] + bins[1:]) / 2.0
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.7), sharex=True)
    for ax, condition in zip(axes.flat, CONDITIONS, strict=True):
        for label, color, name in ((0, "#2E86AB", "true 0"), (1, "#C23B5A", "true 1")):
            seed_densities = []
            for seed in SEEDS:
                values = np.asarray(
                    [
                        call.probability
                        for call in calls[(condition, seed)]
                        if call.label == label
                    ],
                    dtype=float,
                )
                density, _ = np.histogram(values, bins=bins, density=True)
                seed_densities.append(density)
                ax.step(
                    centers,
                    density,
                    where="mid",
                    color=color,
                    linewidth=0.8,
                    alpha=0.12,
                )
            densities = np.vstack(seed_densities)
            mean_density = densities.mean(axis=0)
            ax.step(centers, mean_density, where="mid", color=color, linewidth=1.8, label=name)
            ax.fill_between(
                centers,
                densities.min(axis=0),
                densities.max(axis=0),
                step="mid",
                color=color,
                alpha=0.08,
            )
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Predicted positive probability")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Threshold-free call-probability distributions by true label", x=0.02, ha="left", fontweight="bold", fontsize=14)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def render_figures(
    data: Gate1FigureData,
    output_directory: Path,
    *,
    formats: Sequence[str] = FORMATS,
) -> dict[str, tuple[Path, ...]]:
    if any(extension not in FORMATS for extension in formats) or not formats:
        raise Gate1FigureError("unsupported/empty export format family")
    _validate_condition_records(data.condition_records)
    _validate_comparisons(data.comparisons)
    output_directory.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "paper2-work9-gate1-v1",
        }
    )
    builders = {
        "outer_condition_macro_f1": _condition_macro_f1,
        "outer_condition_metric_profile": _metric_profile,
        "gate1_primary_effects": _primary_effects,
        "gate1_seed_stability": _seed_stability,
        "gate1_fold_stability": _fold_stability,
        "outer_macro_f1_fold_seed": _macro_f1_heatmaps,
        "outer_confusion_matrices": _confusion_matrices,
        "outer_roc_curves": lambda bundle: _curve_summary(bundle, kind="roc"),
        "outer_pr_curves": lambda bundle: _curve_summary(bundle, kind="pr"),
        "outer_calibration_curves": _calibration_curves,
        "outer_probability_distributions": _probability_distributions,
    }
    results: dict[str, tuple[Path, ...]] = {}
    for figure_id in expected_figure_ids():
        results[figure_id] = _save_figure(
            builders[figure_id](data), output_directory / figure_id, formats
        )
    return results


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("matplotlib", "numpy", "scikit-learn", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def generate_gate1_figures(
    project_root: str | Path,
    *,
    gate1_directory: str | Path,
    expected_gate1_hash_manifest_sha256: str,
    output_directory: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    spec = load_figure_spec(root)
    data = load_gate1_figure_data(
        root,
        gate1_directory=gate1_directory,
        expected_gate1_hash_manifest_sha256=expected_gate1_hash_manifest_sha256,
    )
    output = Path(output_directory).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise Gate1FigureError("figure output must remain inside project root") from error
    if output.exists():
        raise FileExistsError(f"figure output already exists and is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        rendered = render_figures(data, staging / "figures")
        spec_ref = _file_reference(root / FIGURE_SPEC_RELATIVE, root)
        source_ref = _file_reference(root / FIGURE_SOURCE_RELATIVE, root)
        script_ref = _file_reference(root / FIGURE_SCRIPT_RELATIVE, root)
        figure_specs = {str(row["figure_id"]): row for row in spec["figures"]}
        figure_rows = []
        for figure_id in expected_figure_ids():
            files = [_file_reference(path, staging) for path in rendered[figure_id]]
            figure_rows.append(
                {
                    "figure_id": figure_id,
                    "title": figure_specs[figure_id]["title"],
                    "analysis_role": figure_specs[figure_id]["analysis_role"],
                    "files": [reference.as_mapping() for reference in files],
                }
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "sealed_complete",
            "protocol_id": PROTOCOL_ID,
            "generation_id": "work9_gate1_figures_v1",
            "figure_spec": spec_ref.as_mapping(),
            "implementation": {
                "source": source_ref.as_mapping(),
                "script": script_ref.as_mapping(),
            },
            "inputs": {
                "gate1_hash_manifest": data.gate1_hash_manifest.as_mapping(),
                "gate1_artifacts": [reference.as_mapping() for reference in data.gate1_artifacts],
                "outer_ledger_index": data.outer_index.as_mapping(),
                "outer_call_ledgers": [reference.as_mapping() for reference in data.outer_call_ledgers],
            },
            "data_access_declaration": {
                "gate1_outputs_used": True,
                "outer_complete_call_ledgers_used": True,
                "outer_segment_ledgers_used": False,
                "checkpoints_used": False,
                "phase_a_validation_ledgers_used": False,
                "raw_audio_used": False,
                "transcripts_used": False,
                "generated_only_after_gate1_seal": True,
            },
            "claim_boundary": {
                "within_corpus_only": True,
                "independent_unit": "call",
                "practical_utility_claim_permitted": False,
                "seed_fold_curves_descriptive_only": True,
                "only_inferential_figure": "gate1_primary_effects",
            },
            "environment": _environment(),
            "figure_count": len(figure_rows),
            "file_count": sum(len(row["files"]) for row in figure_rows),
            "figures": figure_rows,
        }
        (staging / "figure_manifest.json").write_bytes(_json_bytes(manifest))
        staging.rename(output)
        for path in sorted(output.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        output.chmod(0o555)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = output / "figure_manifest.json"
    verify_figure_manifest(root, manifest_path)
    return manifest_path


def verify_figure_manifest(
    project_root: str | Path, manifest_path: str | Path
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    spec = load_figure_spec(root)
    path = Path(manifest_path).resolve()
    manifest = _read_json(path, "Gate-1 figure manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "sealed_complete"
        or manifest.get("protocol_id") != PROTOCOL_ID
    ):
        raise Gate1FigureError("Gate-1 figure manifest identity/status mismatch")
    if manifest.get("figure_count") != len(expected_figure_ids()) or manifest.get("file_count") != len(expected_figure_ids()) * len(FORMATS):
        raise Gate1FigureError("Gate-1 figure/file count mismatch")
    if [row.get("figure_id") for row in manifest.get("figures", [])] != list(expected_figure_ids()):
        raise Gate1FigureError("Gate-1 manifest figure family/order changed")
    declarations = manifest.get("data_access_declaration", {})
    required_declarations = {
        "gate1_outputs_used": True,
        "outer_complete_call_ledgers_used": True,
        "outer_segment_ledgers_used": False,
        "checkpoints_used": False,
        "phase_a_validation_ledgers_used": False,
        "raw_audio_used": False,
        "transcripts_used": False,
        "generated_only_after_gate1_seal": True,
    }
    if declarations != required_declarations:
        raise Gate1FigureError("Gate-1 manifest access declaration changed")
    spec_ref = manifest.get("figure_spec")
    if not isinstance(spec_ref, Mapping) or spec_ref.get("sha256") != sha256_file(root / FIGURE_SPEC_RELATIVE):
        raise Gate1FigureError("Gate-1 manifest figure-spec binding mismatch")
    implementation = manifest.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {"source", "script"}:
        raise Gate1FigureError("Gate-1 figure implementation bindings are malformed")
    for name, expected_path in (
        ("source", FIGURE_SOURCE_RELATIVE),
        ("script", FIGURE_SCRIPT_RELATIVE),
    ):
        reference = implementation[name]
        if (
            not isinstance(reference, Mapping)
            or reference.get("path") != expected_path.as_posix()
            or reference.get("sha256") != sha256_file(root / expected_path)
        ):
            raise Gate1FigureError(f"Gate-1 figure {name} binding mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "gate1_hash_manifest",
        "gate1_artifacts",
        "outer_ledger_index",
        "outer_call_ledgers",
    }:
        raise Gate1FigureError("Gate-1 figure input bindings are malformed")
    _artifact_reference(inputs["gate1_hash_manifest"], root, "manifest Gate-1 hashes")
    _artifact_reference(inputs["outer_ledger_index"], root, "manifest outer index")
    gate1_artifacts = inputs["gate1_artifacts"]
    call_ledgers = inputs["outer_call_ledgers"]
    if not isinstance(gate1_artifacts, list) or len(gate1_artifacts) != 8:
        raise Gate1FigureError("manifest must bind exactly eight canonical Gate-1 artifacts")
    if not isinstance(call_ledgers, list) or len(call_ledgers) != 200:
        raise Gate1FigureError("manifest must bind exactly 200 outer call ledgers")
    for index, reference in enumerate(gate1_artifacts):
        _artifact_reference(reference, root, f"manifest Gate-1 artifact {index}")
    for index, reference in enumerate(call_ledgers):
        _artifact_reference(reference, root, f"manifest outer call ledger {index}")
    if manifest.get("claim_boundary") != {
        "within_corpus_only": True,
        "independent_unit": "call",
        "practical_utility_claim_permitted": False,
        "seed_fold_curves_descriptive_only": True,
        "only_inferential_figure": "gate1_primary_effects",
    }:
        raise Gate1FigureError("Gate-1 figure claim boundary changed")
    for row in manifest["figures"]:
        spec_row = next(item for item in spec["figures"] if item["figure_id"] == row["figure_id"])
        if row.get("title") != spec_row["title"] or row.get("analysis_role") != spec_row["analysis_role"]:
            raise Gate1FigureError(f"figure metadata changed for {row.get('figure_id')}")
        files = row.get("files")
        if not isinstance(files, list) or len(files) != 3:
            raise Gate1FigureError(f"figure file set changed for {row.get('figure_id')}")
        suffixes = set()
        for ref in files:
            if not isinstance(ref, Mapping):
                raise Gate1FigureError("figure file reference is malformed")
            file_path = (path.parent / str(ref["path"])).resolve()
            try:
                file_path.relative_to(path.parent)
            except ValueError as error:
                raise Gate1FigureError("figure file escapes bundle") from error
            if not file_path.is_file() or file_path.is_symlink():
                raise Gate1FigureError("figure file is missing/symlinked")
            if sha256_file(file_path) != ref.get("sha256") or file_path.stat().st_size != ref.get("bytes"):
                raise Gate1FigureError("figure file hash/size mismatch")
            suffixes.add(file_path.suffix.lstrip("."))
        if suffixes != set(FORMATS):
            raise Gate1FigureError(f"figure formats changed for {row.get('figure_id')}")
    return manifest


__all__ = [
    "CLAIM_LABEL",
    "COMPARISONS",
    "CONDITIONS",
    "CallPrediction",
    "Gate1FigureData",
    "Gate1FigureError",
    "compute_call_metrics",
    "expected_figure_ids",
    "generate_gate1_figures",
    "load_figure_spec",
    "load_gate1_figure_data",
    "render_figures",
    "sha256_file",
    "verify_figure_manifest",
]
