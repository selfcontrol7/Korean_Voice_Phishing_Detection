"""Post-run Gate-1 analysis for immutable Work 9 execution ledgers.

This module never trains or selects a model.  It validates the complete frozen
Work 9 primary-matrix run family (whose design was frozen in Work 8),
recomputes complete-call mean probabilities from each outer-evaluation segment
ledger, and then performs the preregistered clustered paired inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .work9_acceptance import verify_environment_lock_mapping
from .config import (
    PRIMARY_COMPARISONS,
    REPORTED_CALL_METRICS,
    THRESHOLD_TIE_BREAK,
    MatrixConfig,
    load_matrix_config,
    sha256_file,
)
from .pairing import PairingItem, PairingPlan
from .work9_types import (
    ACCEPTANCE_RECEIPT_RELATIVE,
    ACCEPTANCE_SCHEMA_VERSION,
    EXPECTED_SELECTION_COUNT,
    EXPERIMENT_CARD_RELATIVE,
    GATE1_ANALYSIS_PROTOCOL_RELATIVE,
    GATE1_ANALYSIS_PROTOCOL_SHA256,
    GATE1_POLICY_RELATIVE,
    SUPERSEDED_ACCEPTANCE_GENERATIONS,
    WORK8_CONFIG_SHA256,
    WORK8_CONFIG_RELATIVE,
    WORK8_RECEIPT_RELATIVE,
    WORK8_RECEIPT_SHA256,
    Work9Coordinates,
)


INDEX_SCHEMA_VERSION = "paper2-v2-work9-outer-ledger-index-1"
SELECTION_INDEX_SCHEMA_VERSION = "paper2-v2-work9-selection-index-1"
SEGMENT_SCHEMA_VERSION = "paper2-v2-work9-segment-prediction-1"
CALL_SCHEMA_VERSION = "paper2-v2-work9-call-prediction-1"
SEGMENT_SCORE_SCHEMA_VERSION = "paper2-v2-work9-validation-segment-score-1"
CALL_SCORE_SCHEMA_VERSION = "paper2-v2-work9-validation-call-score-1"
SELECTION_SEAL_SCHEMA_VERSION = "paper2-v2-work9-selection-seal-1"
EPOCH_TRACE_SCHEMA_VERSION = "paper2-v2-work9-epoch-trace-1"
THRESHOLD_TRACE_SCHEMA_VERSION = "paper2-v2-work9-threshold-trace-1"
REPORT_SCHEMA_VERSION = "paper2-v2-work9-gate1-report-1"
GATE1_DECISION_POLICY_SHA256 = (
    "a6697bebd0df3d6dca99bcfd82dd07bf874f2cf7923fae5d07d0b1201444880b"
)
CANONICAL_SEEDS = (17, 29, 43, 71, 101)

OUTER_INDEX_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_family",
        "status",
        "selection_index",
        "gate1_analysis_protocol",
        "expected_coordinate_count",
        "entries",
    }
)
OUTER_ENTRY_KEYS = frozenset(
    {
        "run_id",
        "condition",
        "partition_repeat",
        "outer_fold",
        "seed",
        "threshold",
        "checkpoint_sha256",
        "outer_eval_segment_ledger",
        "outer_eval_call_ledger",
        "outer_eval_metrics",
        "run_record",
    }
)
SELECTION_INDEX_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_family",
        "status",
        "expected_coordinate_count",
        "config",
        "work8_receipt",
        "runner_acceptance",
        "experiment_card",
        "gate1_policy",
        "gate1_analysis_protocol",
        "entries",
    }
)
SELECTION_ENTRY_KEYS = frozenset(
    {
        "run_id",
        "condition",
        "partition_repeat",
        "outer_fold",
        "seed",
        "selection_seal",
        "selection_run_record",
        "epoch_trace",
        "threshold_trace",
    }
)
SEGMENT_ROW_KEYS = frozenset(
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
        "segment_id",
        "call_id",
        "group_id",
        "label",
        "probability",
        "hard_prediction",
        "source",
        "start",
        "end",
        "duration_seconds",
        "threshold",
        "checkpoint_sha256",
        "manifest_sha256",
    }
)
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
SEGMENT_SCORE_ROW_KEYS = (
    SEGMENT_ROW_KEYS - {"hard_prediction", "threshold"}
) | {"epoch"}
CALL_SCORE_ROW_KEYS = (CALL_ROW_KEYS - {"hard_prediction", "threshold"}) | {
    "epoch"
}
SELECTION_SEAL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "protocol_id",
        "run_family",
        "condition",
        "partition_repeat",
        "outer_fold",
        "seed",
        "config_sha256",
        "work8_receipt_sha256",
        "runner_acceptance_sha256",
        "experiment_card_sha256",
        "gate1_policy_sha256",
        "gate1_analysis_protocol_sha256",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "feature_store_metadata_sha256",
        "feature_store_index_sha256",
        "feature_store_vectors_sha256",
        "pairing_plan_sha256",
        "pairing_plan_artifact",
        "checkpoint",
        "selected_epoch",
        "validation_call_nll",
        "validation_segment_ledger",
        "validation_call_ledger",
        "epoch_trace",
        "threshold_trace",
        "threshold",
    }
)
EPOCH_TRACE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "protocol_id",
        "run_family",
        "condition",
        "partition_repeat",
        "outer_fold",
        "seed",
        "monitor",
        "tie_break",
        "epochs",
        "selected_epoch",
    }
)
EPOCH_RECORD_KEYS = frozenset(
    {
        "epoch",
        "validation_call_nll",
        "train_reported_loss",
        "diagnostic_threshold",
        "checkpoint",
        "validation_segment_scores",
        "validation_call_scores",
    }
)
THRESHOLD_TRACE_KEYS = frozenset(
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
        "validation_call_ledger",
        "threshold_grid",
        "threshold_tie_break",
        "candidates",
        "selected_threshold",
    }
)
OUTER_METRICS_KEYS = frozenset(
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
        "aggregation_rule",
        "threshold",
        "checkpoint_sha256",
        "outer_eval_manifest_sha256",
        "metrics",
        "interpretation",
    }
)
RUN_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "protocol_id",
        "run_family",
        "condition",
        "partition_repeat",
        "outer_fold",
        "seed",
        "phase",
        "status",
        "lineage",
        "data",
        "model",
        "randomness",
        "training",
        "selection",
        "artifacts",
        "compute",
        "failure",
    }
)


class Gate1AnalysisError(RuntimeError):
    """Raised when an immutable input or preregistered invariant fails."""


def _strict_keys(mapping: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = set(mapping)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise Gate1AnalysisError(
            f"{context} keys invalid; missing={missing}, unexpected={unexpected}"
        )


def _require_sha256(value: Any, context: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise Gate1AnalysisError(f"{context} must be a lowercase SHA-256")
    return digest


def _finite_float(value: Any, context: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool):
        raise Gate1AnalysisError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Gate1AnalysisError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise Gate1AnalysisError(f"{context} must be finite")
    if lower is not None and result < lower:
        raise Gate1AnalysisError(f"{context} must be >= {lower}")
    return result


def _probability(value: Any, context: str) -> float:
    result = _finite_float(value, context)
    if not 0.0 <= result <= 1.0:
        raise Gate1AnalysisError(f"{context} must be within [0,1]")
    return result


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Gate1AnalysisError(f"{context} must be a positive integer")
    result = value
    if result <= 0:
        raise Gate1AnalysisError(f"{context} must be a positive integer")
    return result


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Gate1AnalysisError(f"{context} must be a nonnegative integer")
    return value


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate1AnalysisError(f"cannot read {context}: {path}") from error
    if not isinstance(payload, Mapping):
        raise Gate1AnalysisError(f"{context} must be a JSON object")
    return payload


def _read_jsonl(path: Path, context: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Gate1AnalysisError(
                        f"{context} contains a blank line at {path}:{line_number}"
                    )
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise Gate1AnalysisError(
                        f"{context} row {line_number} must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate1AnalysisError(f"cannot read {context}: {path}") from error
    if not rows:
        raise Gate1AnalysisError(f"{context} is empty: {path}")
    return rows


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                raise Gate1AnalysisError(f"immutable JSONL contains a blank line: {path}")
            count += 1
    return count


@dataclass(frozen=True)
class ImmutableRef:
    path: Path
    sha256: str
    bytes: int
    row_count: int | None
    serialized_path: str

    @classmethod
    def from_mapping(
        cls,
        raw: Any,
        *,
        project_root: Path,
        context: str,
        ledger: bool,
    ) -> "ImmutableRef":
        if not isinstance(raw, Mapping):
            raise Gate1AnalysisError(f"{context} must be an artifact reference")
        _strict_keys(
            raw, frozenset({"path", "sha256", "bytes", "row_count"}), context
        )
        serialized = str(raw["path"])
        posix = PurePosixPath(serialized)
        if not serialized or posix.is_absolute() or ".." in posix.parts:
            raise Gate1AnalysisError(f"{context}.path must be a normalized relative path")
        if posix.as_posix() != serialized:
            raise Gate1AnalysisError(f"{context}.path must use normalized POSIX separators")
        candidate = project_root / Path(*posix.parts)
        cursor = project_root
        for part in posix.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise Gate1AnalysisError(f"{context}.path traverses a symlink")
        path = candidate.resolve()
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise Gate1AnalysisError(f"{context}.path escapes project root") from error
        if not path.is_file():
            raise Gate1AnalysisError(f"{context}.path is not a regular immutable file: {path}")
        expected_hash = _require_sha256(raw["sha256"], f"{context}.sha256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise Gate1AnalysisError(
                f"{context} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        byte_count = _positive_int(raw["bytes"], f"{context}.bytes")
        actual_bytes = path.stat().st_size
        if actual_bytes != byte_count:
            raise Gate1AnalysisError(
                f"{context} byte-count mismatch: expected {byte_count}, got {actual_bytes}"
            )
        row_count_raw = raw["row_count"]
        if ledger:
            row_count = _positive_int(row_count_raw, f"{context}.row_count")
            actual_rows = _count_jsonl_rows(path)
            if actual_rows != row_count:
                raise Gate1AnalysisError(
                    f"{context} row-count mismatch: expected {row_count}, got {actual_rows}"
                )
        else:
            if row_count_raw is not None:
                raise Gate1AnalysisError(f"{context}.row_count must be null for non-ledgers")
            row_count = None
        return cls(path, expected_hash, byte_count, row_count, serialized)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "path": self.serialized_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class ManifestSegment:
    segment_id: str
    call_id: str
    group_id: str
    label: int
    source: str
    start: float
    end: float


@dataclass(frozen=True)
class Gate1Call:
    condition: str
    partition_repeat: int
    outer_fold: int
    seed: int
    call_id: str
    group_id: str
    label: int
    probability: float
    hard_prediction: int
    threshold: float
    n_segments: int
    source: str
    start: float
    end: float


@dataclass(frozen=True)
class ValidatedLedgerPair:
    calls: tuple[Gate1Call, ...]
    segment_probabilities: Mapping[str, float]
    threshold: float | None


@dataclass(frozen=True)
class ValidatedGate1Input:
    calls_by_coordinate: Mapping[tuple[str, int, int, int], tuple[Gate1Call, ...]]
    outer_ledger_index_sha256: str
    selection_index_sha256: str
    selection_index_path: Path
    matrix_config_sha256: str
    audit: Mapping[str, Any]


def _label(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise Gate1AnalysisError(f"{context} must be binary 0/1")
    return int(value)


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise Gate1AnalysisError(f"{context} must be nonempty")
    return value


def _manifest_segments(
    path: Path,
    *,
    partition_repeat: int,
    outer_fold: int,
    manifest_partition: str,
) -> dict[str, ManifestSegment]:
    rows = _read_jsonl(
        path,
        f"{manifest_partition} manifest repeat={partition_repeat} fold={outer_fold}",
    )
    result: dict[str, ManifestSegment] = {}
    calls: dict[str, tuple[str, int, str]] = {}
    for line_number, row in enumerate(rows, start=1):
        context = f"{path}:{line_number}"
        if row.get("partition") != manifest_partition:
            raise Gate1AnalysisError(f"{context} is not {manifest_partition}")
        if int(row.get("partition_repeat", -1)) != partition_repeat:
            raise Gate1AnalysisError(f"{context} has wrong partition_repeat")
        if int(row.get("outer_fold", -1)) != outer_fold:
            raise Gate1AnalysisError(f"{context} has wrong outer_fold")
        segment_id = _nonempty(row.get("segment_id", ""), f"{context}.segment_id")
        if segment_id in result:
            raise Gate1AnalysisError(f"duplicate manifest segment_id {segment_id!r}")
        call_id = _nonempty(row.get("call_id", ""), f"{context}.call_id")
        group_id = _nonempty(row.get("group_id", ""), f"{context}.group_id")
        label = _label(row.get("label"), f"{context}.label")
        source = _nonempty(row.get("source", ""), f"{context}.source")
        start = _finite_float(row.get("start"), f"{context}.start", lower=0.0)
        end = _finite_float(row.get("end"), f"{context}.end", lower=0.0)
        if end < start:
            raise Gate1AnalysisError(f"{context} has end before start")
        identity = (group_id, label, source)
        if call_id in calls and calls[call_id] != identity:
            raise Gate1AnalysisError(f"manifest call {call_id!r} has inconsistent metadata")
        calls[call_id] = identity
        result[segment_id] = ManifestSegment(
            segment_id, call_id, group_id, label, source, start, end
        )
    return result


def _coordinate_values(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        _nonempty(row["condition"], "coordinate.condition"),
        _nonnegative_int(row["partition_repeat"], "coordinate.partition_repeat"),
        _nonnegative_int(row["outer_fold"], "coordinate.outer_fold"),
        _nonnegative_int(row["seed"], "coordinate.seed"),
    )


def _validate_coordinate_fields(
    row: Mapping[str, Any],
    *,
    context: str,
    run_id: str,
    protocol_id: str,
    coordinate: tuple[str, int, int, int],
    partition: str | None = None,
) -> None:
    condition, partition_repeat, outer_fold, seed = coordinate
    expected: dict[str, Any] = {
        "run_id": run_id,
        "protocol_id": protocol_id,
        "condition": condition,
        "partition_repeat": partition_repeat,
        "outer_fold": outer_fold,
        "seed": seed,
    }
    if partition is not None:
        expected["partition"] = partition
    mismatches = {
        key: {"expected": wanted, "actual": row.get(key)}
        for key, wanted in expected.items()
        if row.get(key) != wanted
    }
    if mismatches:
        raise Gate1AnalysisError(f"{context} coordinate mismatch: {mismatches}")


def _validate_selection_seal(
    seal_ref: ImmutableRef,
    *,
    entry: Mapping[str, Any],
    refs: Mapping[str, ImmutableRef],
    config: MatrixConfig,
    matrix_config_sha256: str,
    selection_inputs: Mapping[str, ImmutableRef],
    train_manifest_sha256: str,
    validation_manifest_sha256: str,
    expected_pairing_plan_sha256: str | None,
) -> Mapping[str, Any]:
    seal = _read_json(seal_ref.path, "selection seal")
    _strict_keys(seal, SELECTION_SEAL_KEYS, "selection seal")
    if seal["schema_version"] != SELECTION_SEAL_SCHEMA_VERSION:
        raise Gate1AnalysisError("selection seal schema version mismatch")
    if seal["run_family"] != "primary_matrix":
        raise Gate1AnalysisError("selection seal is not in the primary matrix")
    coordinate = _coordinate_values(entry)
    _validate_coordinate_fields(
        seal,
        context="selection seal",
        run_id=str(entry["run_id"]),
        protocol_id=config.protocol_id,
        coordinate=coordinate,
    )
    scalar_expected = {
        "selected_epoch": int(entry["selected_epoch"]),
        "validation_call_nll": float(entry["validation_call_nll"]),
        "threshold": float(entry["threshold"]),
    }
    for key, wanted in scalar_expected.items():
        actual = float(seal[key]) if key != "selected_epoch" else int(seal[key])
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-15):
            raise Gate1AnalysisError(f"selection seal {key} does not match indexed selection")
    for name in (
        "checkpoint",
        "validation_segment_ledger",
        "validation_call_ledger",
        "epoch_trace",
        "threshold_trace",
    ):
        if seal[name] != refs[name].as_mapping():
            raise Gate1AnalysisError(f"selection seal does not bind {name}")
    expected_hashes = {
        "config_sha256": matrix_config_sha256,
        "work8_receipt_sha256": selection_inputs["work8_receipt"].sha256,
        "runner_acceptance_sha256": selection_inputs["runner_acceptance"].sha256,
        "experiment_card_sha256": selection_inputs["experiment_card"].sha256,
        "gate1_policy_sha256": GATE1_DECISION_POLICY_SHA256,
        "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
        "train_manifest_sha256": train_manifest_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
        "feature_store_metadata_sha256": config.text_store.metadata.sha256,
        "feature_store_index_sha256": config.text_store.index_sha256,
        "feature_store_vectors_sha256": config.text_store.vectors_sha256,
    }
    for key, wanted in expected_hashes.items():
        if seal[key] != wanted:
            raise Gate1AnalysisError(f"selection seal lineage mismatch for {key}")
    pairing_hash = seal["pairing_plan_sha256"]
    if pairing_hash != expected_pairing_plan_sha256:
        raise Gate1AnalysisError("selection seal pairing plan differs from frozen Work 8 receipt")
    pairing_artifact = seal["pairing_plan_artifact"]
    if expected_pairing_plan_sha256 is None:
        if pairing_artifact is not None or "pairing_plan" in refs:
            raise Gate1AnalysisError("non-random condition unexpectedly binds a pairing plan")
    elif (
        "pairing_plan" not in refs
        or pairing_artifact != refs["pairing_plan"].as_mapping()
    ):
        raise Gate1AnalysisError("selection seal does not bind its pairing-plan artifact")
    return seal


def _tree_contains_mapping(value: Any, wanted: Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        if dict(value) == dict(wanted):
            return True
        return any(_tree_contains_mapping(item, wanted) for item in value.values())
    if isinstance(value, list):
        return any(_tree_contains_mapping(item, wanted) for item in value)
    return False


def _validate_run_record(
    record_ref: ImmutableRef,
    *,
    entry: Mapping[str, Any],
    config: MatrixConfig,
    phase: str,
    status: str,
    required_artifact_refs: Sequence[ImmutableRef],
    matrix_config_sha256: str,
    selection_inputs: Mapping[str, ImmutableRef],
    train_manifest_sha256: str,
    validation_manifest_sha256: str,
    outer_eval_manifest_sha256: str | None,
    pairing_plan_sha256: str | None,
) -> None:
    record = _read_json(record_ref.path, f"{phase} run record")
    _strict_keys(record, RUN_RECORD_KEYS, f"{phase} run record")
    if record["schema_version"] != "paper2-v2-work9-run-record-1":
        raise Gate1AnalysisError(f"{phase} run record schema mismatch")
    if record["run_family"] != "primary_matrix":
        raise Gate1AnalysisError(f"{phase} run record is not primary-matrix")
    _validate_coordinate_fields(
        record,
        context=f"{phase} run record",
        run_id=str(entry["run_id"]),
        protocol_id=config.protocol_id,
        coordinate=_coordinate_values(entry),
    )
    if record["phase"] != phase or record["status"] != status:
        raise Gate1AnalysisError(f"{phase} run record has wrong phase/status")
    if record["failure"] is not None:
        raise Gate1AnalysisError(f"{phase} run record contains a failure")
    lineage = record["lineage"]
    wanted_lineage = {
        "config_sha256": matrix_config_sha256,
        "work8_receipt_sha256": selection_inputs["work8_receipt"].sha256,
        "runner_acceptance_sha256": selection_inputs["runner_acceptance"].sha256,
        "experiment_card_sha256": selection_inputs["experiment_card"].sha256,
        "gate1_policy_sha256": GATE1_DECISION_POLICY_SHA256,
        "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
    }
    if lineage != wanted_lineage:
        raise Gate1AnalysisError(f"{phase} run record lineage hashes differ")
    data = record["data"]
    if not isinstance(data, Mapping):
        raise Gate1AnalysisError(f"{phase} run record data must be an object")
    data_expected = {
        "train_manifest_sha256": train_manifest_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
        "outer_eval_manifest_sha256": outer_eval_manifest_sha256,
        "grouping_policy_id": config.grouping_policy_id,
    }
    if any(data.get(key) != value for key, value in data_expected.items()):
        raise Gate1AnalysisError(f"{phase} run record manifest/grouping lineage differs")
    feature_ids = data.get("feature_store_ids")
    wanted_feature_ids = {
        "text_feature_set_id": config.text_store.feature_set_id,
        "metadata_sha256": config.text_store.metadata.sha256,
        "index_sha256": config.text_store.index_sha256,
        "vectors_sha256": config.text_store.vectors_sha256,
    }
    if feature_ids != wanted_feature_ids:
        raise Gate1AnalysisError(f"{phase} run record feature-store lineage differs")
    selection = record["selection"]
    if not isinstance(selection, Mapping):
        raise Gate1AnalysisError(f"{phase} run record selection must be an object")
    for key in ("selected_epoch", "validation_call_nll", "threshold"):
        actual = selection.get(key)
        wanted = entry[key]
        if key == "selected_epoch":
            if _positive_int(actual, f"{phase} run record.{key}") != wanted:
                raise Gate1AnalysisError(f"{phase} run record {key} differs")
        elif not math.isclose(
            _finite_float(actual, f"{phase} run record.{key}"),
            float(wanted),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise Gate1AnalysisError(f"{phase} run record {key} differs")
    if phase == "selection":
        training = record["training"]
        if not isinstance(training, Mapping) or training.get("pairing_plan_sha256") != pairing_plan_sha256:
            raise Gate1AnalysisError("selection run record pairing-plan SHA differs")
    artifacts = record["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise Gate1AnalysisError(f"{phase} run record artifacts must be an object")
    for artifact in required_artifact_refs:
        if not _tree_contains_mapping(artifacts, artifact.as_mapping()):
            raise Gate1AnalysisError(
                f"{phase} run record does not bind {artifact.serialized_path}"
            )


def _pairing_plan_digests(
    receipt_ref: ImmutableRef,
    *,
    config: MatrixConfig,
) -> dict[tuple[int, int, int], str]:
    """Recover the 25 frozen random-train plan digests from the Work 8 receipt."""

    receipt = _read_json(receipt_ref.path, "Work 8 preregistration receipt")
    if receipt.get("protocol_id") != config.protocol_id:
        raise Gate1AnalysisError("Work 8 receipt protocol differs from matrix config")
    rows = receipt.get("random_train_pairing_plans")
    if not isinstance(rows, list):
        raise Gate1AnalysisError("Work 8 receipt has no random-train pairing plan list")
    expected = {
        (fold.partition_repeat, fold.outer_fold, seed)
        for fold in config.folds
        for seed in config.seeds
    }
    result: dict[tuple[int, int, int], str] = {}
    for index, row in enumerate(rows):
        context = f"Work 8 pairing plan [{index}]"
        if not isinstance(row, Mapping):
            raise Gate1AnalysisError(f"{context} must be an object")
        if row.get("condition") != "AT-cross-randtrain" or row.get("split_role") != "train":
            raise Gate1AnalysisError(f"{context} has wrong condition/split role")
        key = (
            _nonnegative_int(row.get("partition_repeat"), f"{context}.partition_repeat"),
            _nonnegative_int(row.get("outer_fold"), f"{context}.outer_fold"),
            _nonnegative_int(row.get("seed"), f"{context}.seed"),
        )
        if key not in expected or key in result:
            raise Gate1AnalysisError(f"{context} has unexpected or duplicate coordinate {key}")
        fold = config.fold(key[0], key[1])
        if row.get("train_manifest_sha256") != fold.train_manifest.sha256:
            raise Gate1AnalysisError(f"{context} train manifest hash mismatch")
        result[key] = _require_sha256(row.get("plan_sha256"), f"{context}.plan_sha256")
    if set(result) != expected:
        raise Gate1AnalysisError("Work 8 receipt pairing plans are incomplete")
    if receipt.get("pairing_plan_count") != len(expected):
        raise Gate1AnalysisError("Work 8 receipt pairing_plan_count is incorrect")
    return result


def _validate_selection_index_lineage_inputs(
    refs: Mapping[str, ImmutableRef],
    *,
    project_root: Path,
    config: MatrixConfig,
    matrix_config_sha256: str,
) -> None:
    expected_paths = {
        "config": WORK8_CONFIG_RELATIVE,
        "work8_receipt": WORK8_RECEIPT_RELATIVE,
        "runner_acceptance": ACCEPTANCE_RECEIPT_RELATIVE,
        "experiment_card": EXPERIMENT_CARD_RELATIVE,
        "gate1_policy": GATE1_POLICY_RELATIVE,
        "gate1_analysis_protocol": GATE1_ANALYSIS_PROTOCOL_RELATIVE,
    }
    for name, relative in expected_paths.items():
        if refs[name].path != (project_root / relative).resolve():
            raise Gate1AnalysisError(
                f"selection index {name} is not the canonical frozen artifact"
            )
    if refs["config"].sha256 != matrix_config_sha256:
        raise Gate1AnalysisError("selection index config hash mismatch")
    if refs["work8_receipt"].sha256 != WORK8_RECEIPT_SHA256:
        raise Gate1AnalysisError("selection index Work 8 receipt hash mismatch")
    if refs["gate1_policy"].sha256 != GATE1_DECISION_POLICY_SHA256:
        raise Gate1AnalysisError("selection index Gate-1 policy hash mismatch")
    if refs["gate1_analysis_protocol"].sha256 != GATE1_ANALYSIS_PROTOCOL_SHA256:
        raise Gate1AnalysisError("selection index Gate-1 analysis protocol hash mismatch")

    receipt = _read_json(refs["work8_receipt"].path, "Work 8 receipt")
    receipt_safety = receipt.get("safety")
    if (
        receipt.get("status") != "pass"
        or receipt.get("protocol_id") != config.protocol_id
        or not isinstance(receipt_safety, Mapping)
        or receipt_safety.get("training_launched") is not False
    ):
        raise Gate1AnalysisError("selection index does not bind the passing Work 8 receipt")

    card = _read_json(refs["experiment_card"].path, "Work 9 experiment card")
    card_config = card.get("work8_config")
    card_receipt = card.get("work8_receipt")
    card_policy = card.get("decision_policy")
    if (
        card.get("schema_version") != "paper2-v2-work9-experiment-card-1"
        or card.get("status") != "frozen_before_work9_model_outcomes"
        or card.get("protocol_id") != config.protocol_id
        or card.get("run_family") != "primary_matrix"
        or not isinstance(card_config, Mapping)
        or card_config.get("sha256") != WORK8_CONFIG_SHA256
        or not isinstance(card_receipt, Mapping)
        or card_receipt.get("sha256") != WORK8_RECEIPT_SHA256
        or not isinstance(card_policy, Mapping)
        or card_policy.get("sha256") != GATE1_DECISION_POLICY_SHA256
    ):
        raise Gate1AnalysisError("Work 9 experiment-card lineage/status mismatch")

    acceptance = _read_json(refs["runner_acceptance"].path, "Work 9 runner acceptance")
    expected_acceptance_lineage = {
        "work8_config_sha256": WORK8_CONFIG_SHA256,
        "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
        "experiment_card_sha256": refs["experiment_card"].sha256,
        "gate1_policy_sha256": GATE1_DECISION_POLICY_SHA256,
        "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
    }
    if (
        acceptance.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION
        or acceptance.get("status") != "accepted_after_two_cuda_smokes"
        or any(
            acceptance.get(key) != value
            for key, value in expected_acceptance_lineage.items()
        )
    ):
        raise Gate1AnalysisError("Work 9 runner-acceptance lineage/status mismatch")
    _validate_acceptance_supersession(acceptance, project_root=project_root)
    accepted_analysis = acceptance.get("gate1_analysis_protocol")
    if not isinstance(accepted_analysis, Mapping):
        raise Gate1AnalysisError("runner acceptance lacks Gate-1 analysis protocol ref")
    if (
        accepted_analysis.get("path") != GATE1_ANALYSIS_PROTOCOL_RELATIVE.as_posix()
        or accepted_analysis.get("sha256") != GATE1_ANALYSIS_PROTOCOL_SHA256
    ):
        raise Gate1AnalysisError("runner acceptance binds another analysis protocol")
    smokes = acceptance.get("cuda_smokes")
    if not isinstance(smokes, list) or len(smokes) != 2:
        raise Gate1AnalysisError("runner acceptance does not contain exactly two CUDA smokes")
    if any(not isinstance(item, Mapping) for item in smokes):
        raise Gate1AnalysisError("runner acceptance CUDA smokes must be objects")
    semantic_hashes = {item.get("semantic_output_sha256") for item in smokes}
    environment_sha = _validate_acceptance_environment(acceptance)
    if (
        any(item.get("device_type") != "cuda" for item in smokes)
        or any(item.get("cublas_workspace_config") != ":4096:8" for item in smokes)
        or any(item.get("environment_lock_sha256") != environment_sha for item in smokes)
        or len(semantic_hashes) != 1
        or None in semantic_hashes
    ):
        raise Gate1AnalysisError("runner acceptance CUDA-smoke reproducibility mismatch")


def _validate_acceptance_environment(acceptance: Mapping[str, Any]) -> str:
    """Self-hash and cross-bind the accepted environment evidence."""

    try:
        environment_sha = verify_environment_lock_mapping(
            acceptance.get("environment_lock")
        )
    except Exception as error:
        raise Gate1AnalysisError("runner acceptance environment lock is invalid") from error
    smokes = acceptance.get("cuda_smokes")
    if not isinstance(smokes, list) or len(smokes) != 2 or any(
        not isinstance(item, Mapping)
        or item.get("environment_lock_sha256") != environment_sha
        for item in smokes
    ):
        raise Gate1AnalysisError("runner acceptance smoke environment lineage mismatch")
    unit_tests = acceptance.get("unit_tests")
    if not isinstance(unit_tests, Mapping) or unit_tests.get(
        "environment_lock_sha256"
    ) != environment_sha:
        raise Gate1AnalysisError("runner acceptance unit-test environment lineage mismatch")
    return environment_sha


def _validate_acceptance_supersession(
    acceptance: Mapping[str, Any], *, project_root: Path
) -> None:
    """Reopen and verify every preserved failed acceptance generation."""

    supersedes = acceptance.get("supersedes")
    if not isinstance(supersedes, Mapping) or set(supersedes) != {
        "current_acceptance_generation",
        "generations",
    }:
        raise Gate1AnalysisError("runner acceptance supersession keys changed")
    if supersedes.get("current_acceptance_generation") != "work9_runner_acceptance_v3":
        raise Gate1AnalysisError("runner acceptance generation is not v3")
    generations = supersedes.get("generations")
    if not isinstance(generations, list) or len(generations) != len(
        SUPERSEDED_ACCEPTANCE_GENERATIONS
    ):
        raise Gate1AnalysisError("runner acceptance supersession history is incomplete")
    for raw, expected in zip(
        generations, SUPERSEDED_ACCEPTANCE_GENERATIONS, strict=True
    ):
        if not isinstance(raw, Mapping):
            raise Gate1AnalysisError("runner acceptance generation must be an object")
        _strict_keys(
            raw,
            frozenset({"acceptance_generation", "draft", "failure", "reason"}),
            "runner acceptance generation",
        )
        generation, draft_relative, draft_sha, failure_relative, failure_sha, reason = expected
        if raw["acceptance_generation"] != generation or raw["reason"] != reason:
            raise Gate1AnalysisError("runner acceptance generation identity changed")
        superseded_draft = ImmutableRef.from_mapping(
            raw["draft"],
            project_root=project_root,
            context=f"superseded {generation} DRAFT",
            ledger=False,
        )
        superseded_failure = ImmutableRef.from_mapping(
            raw["failure"],
            project_root=project_root,
            context=f"superseded {generation} failure",
            ledger=False,
        )
        if (
            superseded_draft.path != (project_root / draft_relative).resolve()
            or superseded_draft.sha256 != draft_sha
            or superseded_failure.path != (project_root / failure_relative).resolve()
            or superseded_failure.sha256 != failure_sha
        ):
            raise Gate1AnalysisError(
                "runner acceptance supersession evidence mismatch"
            )
        if (
            superseded_draft.path.stat().st_mode
            | superseded_failure.path.stat().st_mode
        ) & 0o222:
            raise Gate1AnalysisError(
                "runner acceptance supersession evidence is writable"
            )


PAIRING_RECORD_KEYS = frozenset(
    {
        "seed",
        "split_role",
        "receiver_segment_id",
        "receiver_call_id",
        "receiver_group_id",
        "donor_segment_id",
        "donor_call_id",
        "donor_group_id",
    }
)


def _validate_pairing_plan_artifact(
    plan_ref: ImmutableRef,
    *,
    seed: int,
    expected_segments: Mapping[str, ManifestSegment],
    expected_semantic_sha256: str,
) -> None:
    """Reopen a frozen train-pairing ledger and recompute its semantic plan hash."""

    if plan_ref.path.name != "pairing_plan.jsonl":
        raise Gate1AnalysisError("pairing-plan artifact must be named pairing_plan.jsonl")
    rows = _read_jsonl(plan_ref.path, "random-train pairing plan")
    if len(rows) != len(expected_segments):
        raise Gate1AnalysisError("pairing-plan receiver count differs from train manifest")
    donors: dict[str, str] = {}
    for index, row in enumerate(rows):
        context = f"pairing plan row {index + 1}"
        _strict_keys(row, PAIRING_RECORD_KEYS, context)
        if _nonnegative_int(row["seed"], f"{context}.seed") != seed:
            raise Gate1AnalysisError(f"{context} seed mismatch")
        if row["split_role"] != "train":
            raise Gate1AnalysisError(f"{context} is not train-only")
        receiver_id = _nonempty(
            row["receiver_segment_id"], f"{context}.receiver_segment_id"
        )
        donor_id = _nonempty(row["donor_segment_id"], f"{context}.donor_segment_id")
        if receiver_id in donors:
            raise Gate1AnalysisError(f"pairing plan repeats receiver {receiver_id!r}")
        if receiver_id not in expected_segments or donor_id not in expected_segments:
            raise Gate1AnalysisError(f"{context} leaves the frozen train universe")
        receiver = expected_segments[receiver_id]
        donor = expected_segments[donor_id]
        actual_identity = (
            row["receiver_call_id"],
            row["receiver_group_id"],
            row["donor_call_id"],
            row["donor_group_id"],
        )
        wanted_identity = (
            receiver.call_id,
            receiver.group_id,
            donor.call_id,
            donor.group_id,
        )
        if actual_identity != wanted_identity:
            raise Gate1AnalysisError(f"{context} call/group identity mismatch")
        donors[receiver_id] = donor_id
    if list(donors) != sorted(donors):
        raise Gate1AnalysisError("pairing-plan records are not sorted by receiver")
    try:
        plan = PairingPlan(
            seed=seed,
            split_role="train",
            donor_by_receiver=donors,
            item_by_segment={
                segment_id: PairingItem(
                    segment_id=segment_id,
                    call_id=segment.call_id,
                    group_id=segment.group_id,
                )
                for segment_id, segment in expected_segments.items()
            },
        )
    except ValueError as error:
        raise Gate1AnalysisError(f"pairing-plan semantic invariants fail: {error}") from error
    if plan.sha256 != expected_semantic_sha256:
        raise Gate1AnalysisError(
            "pairing-plan semantic SHA differs from the frozen Work 8 receipt"
        )
    if rows != plan.as_records():
        raise Gate1AnalysisError("pairing-plan ledger differs from canonical records")


def _validate_prediction_ledgers(
    *,
    entry: Mapping[str, Any],
    refs: Mapping[str, ImmutableRef],
    config: MatrixConfig,
    expected_segments: Mapping[str, ManifestSegment],
    partition: str,
    manifest_sha256: str,
    segment_ref_name: str,
    call_ref_name: str,
    checkpoint_sha256: str,
    expected_threshold: float | None,
    expected_segment_probabilities: Mapping[str, float] | None = None,
) -> ValidatedLedgerPair:
    coordinate = _coordinate_values(entry)
    run_id = str(entry["run_id"])
    threshold = (
        None
        if expected_threshold is None
        else _probability(expected_threshold, f"{partition} expected threshold")
    )
    segment_rows = _read_jsonl(
        refs[segment_ref_name].path, f"{partition} segment ledger"
    )
    by_segment: dict[str, Mapping[str, Any]] = {}
    by_call: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for line_number, row in enumerate(segment_rows, start=1):
        context = f"{partition} segment ledger row {line_number}"
        _strict_keys(row, SEGMENT_ROW_KEYS, context)
        if row["schema_version"] != SEGMENT_SCHEMA_VERSION:
            raise Gate1AnalysisError(f"{context} schema version mismatch")
        _validate_coordinate_fields(
            row,
            context=context,
            run_id=run_id,
            protocol_id=config.protocol_id,
            coordinate=coordinate,
            partition=partition,
        )
        if row["run_family"] != "primary_matrix":
            raise Gate1AnalysisError(f"{context} is not a primary-matrix prediction")
        row_threshold = _probability(row["threshold"], f"{context}.threshold")
        if threshold is None:
            threshold = row_threshold
        elif not math.isclose(row_threshold, threshold, rel_tol=0.0, abs_tol=1e-15):
            raise Gate1AnalysisError(f"{context} threshold differs from sealed selection")
        if row["checkpoint_sha256"] != checkpoint_sha256:
            raise Gate1AnalysisError(f"{context} checkpoint hash mismatch")
        if row["manifest_sha256"] != manifest_sha256:
            raise Gate1AnalysisError(f"{context} manifest hash mismatch")
        segment_id = _nonempty(row["segment_id"], f"{context}.segment_id")
        if segment_id in by_segment:
            raise Gate1AnalysisError(f"duplicate prediction segment_id {segment_id!r}")
        if segment_id not in expected_segments:
            raise Gate1AnalysisError(f"prediction has unknown segment_id {segment_id!r}")
        expected = expected_segments[segment_id]
        actual_identity = (
            str(row["call_id"]),
            str(row["group_id"]),
            _label(row["label"], f"{context}.label"),
            str(row["source"]),
        )
        wanted_identity = (
            expected.call_id,
            expected.group_id,
            expected.label,
            expected.source,
        )
        if actual_identity != wanted_identity:
            raise Gate1AnalysisError(
                f"{context} ID/label/group/source mismatch for {segment_id!r}"
            )
        start = _finite_float(row["start"], f"{context}.start", lower=0.0)
        end = _finite_float(row["end"], f"{context}.end", lower=0.0)
        if not math.isclose(start, expected.start, rel_tol=0.0, abs_tol=1e-12):
            raise Gate1AnalysisError(f"{context} start mismatch")
        if not math.isclose(end, expected.end, rel_tol=0.0, abs_tol=1e-12):
            raise Gate1AnalysisError(f"{context} end mismatch")
        segment_probability = _probability(
            row["probability"], f"{context}.probability"
        )
        if expected_segment_probabilities is not None:
            if segment_id not in expected_segment_probabilities or not math.isclose(
                segment_probability,
                float(expected_segment_probabilities[segment_id]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise Gate1AnalysisError(
                    f"{context} probability differs from selected epoch score ledger"
                )
        segment_prediction = _label(
            row["hard_prediction"], f"{context}.hard_prediction"
        )
        assert threshold is not None
        if segment_prediction != int(segment_probability >= threshold):
            raise Gate1AnalysisError(f"{context} hard prediction is inconsistent")
        duration = _finite_float(
            row["duration_seconds"], f"{context}.duration_seconds", lower=0.0
        )
        if not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-12):
            raise Gate1AnalysisError(f"{context} duration differs from end-start")
        by_segment[segment_id] = row
        by_call[expected.call_id].append(row)
    missing_segments = sorted(set(expected_segments) - set(by_segment))
    if missing_segments:
        raise Gate1AnalysisError(
            f"{partition} segment ledger is incomplete; missing {missing_segments[:5]}"
        )
    if threshold is None:
        raise Gate1AnalysisError(f"{partition} segment ledger has no threshold")

    call_rows = _read_jsonl(refs[call_ref_name].path, f"{partition} call ledger")
    expected_call_ids = set(by_call)
    seen_calls: set[str] = set()
    calls: list[Gate1Call] = []
    for line_number, row in enumerate(call_rows, start=1):
        context = f"{partition} call ledger row {line_number}"
        _strict_keys(row, CALL_ROW_KEYS, context)
        if row["schema_version"] != CALL_SCHEMA_VERSION:
            raise Gate1AnalysisError(f"{context} schema version mismatch")
        _validate_coordinate_fields(
            row,
            context=context,
            run_id=run_id,
            protocol_id=config.protocol_id,
            coordinate=coordinate,
            partition=partition,
        )
        if row["run_family"] != "primary_matrix":
            raise Gate1AnalysisError(f"{context} is not a primary-matrix prediction")
        if row["checkpoint_sha256"] != checkpoint_sha256:
            raise Gate1AnalysisError(f"{context} checkpoint hash mismatch")
        if row["manifest_sha256"] != manifest_sha256:
            raise Gate1AnalysisError(f"{context} manifest hash mismatch")
        call_id = _nonempty(row["call_id"], f"{context}.call_id")
        if call_id in seen_calls:
            raise Gate1AnalysisError(f"duplicate call prediction {call_id!r}")
        if call_id not in by_call:
            raise Gate1AnalysisError(f"call ledger has unknown call_id {call_id!r}")
        seen_calls.add(call_id)
        segments = by_call[call_id]
        expected_first = expected_segments[str(segments[0]["segment_id"])]
        if (
            str(row["group_id"]) != expected_first.group_id
            or _label(row["label"], f"{context}.label") != expected_first.label
            or str(row["source"]) != expected_first.source
        ):
            raise Gate1AnalysisError(f"{context} call ID/label/group/source mismatch")
        probabilities = np.asarray(
            [_probability(item["probability"], f"{context}.segment_probability") for item in segments],
            dtype=np.float64,
        )
        recomputed = float(probabilities.mean(dtype=np.float64))
        call_probability = _probability(row["probability"], f"{context}.probability")
        if not math.isclose(call_probability, recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise Gate1AnalysisError(
                f"{context} is not the complete-call arithmetic mean: "
                f"reported={call_probability}, recomputed={recomputed}"
            )
        call_threshold = _probability(row["threshold"], f"{context}.threshold")
        if not math.isclose(call_threshold, threshold, rel_tol=0.0, abs_tol=1e-15):
            raise Gate1AnalysisError(f"{context} threshold differs from selection seal/index")
        hard_prediction = _label(
            row["hard_prediction"], f"{context}.hard_prediction"
        )
        if hard_prediction != int(call_probability >= threshold):
            raise Gate1AnalysisError(f"{context} hard prediction is inconsistent")
        n_segments = _positive_int(row["n_segments"], f"{context}.n_segments")
        if n_segments != len(segments):
            raise Gate1AnalysisError(f"{context} n_segments mismatch")
        if row["aggregation_rule"] != "mean_probability":
            raise Gate1AnalysisError(f"{context} changed call aggregation rule")
        start = _finite_float(row["start"], f"{context}.start", lower=0.0)
        end = _finite_float(row["end"], f"{context}.end", lower=0.0)
        wanted_start = min(float(item["start"]) for item in segments)
        wanted_end = max(float(item["end"]) for item in segments)
        if not math.isclose(start, wanted_start, rel_tol=0.0, abs_tol=1e-12):
            raise Gate1AnalysisError(f"{context} call start mismatch")
        if not math.isclose(end, wanted_end, rel_tol=0.0, abs_tol=1e-12):
            raise Gate1AnalysisError(f"{context} call end mismatch")
        duration = _finite_float(
            row["duration_seconds"], f"{context}.duration_seconds", lower=0.0
        )
        if not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-12):
            raise Gate1AnalysisError(f"{context} duration differs from end-start")
        calls.append(
            Gate1Call(
                condition=coordinate[0],
                partition_repeat=coordinate[1],
                outer_fold=coordinate[2],
                seed=coordinate[3],
                call_id=call_id,
                group_id=expected_first.group_id,
                label=expected_first.label,
                probability=recomputed,
                hard_prediction=hard_prediction,
                threshold=threshold,
                n_segments=n_segments,
                source=expected_first.source,
                start=start,
                end=end,
            )
        )
    missing_calls = sorted(expected_call_ids - seen_calls)
    if missing_calls:
        raise Gate1AnalysisError(
            f"{partition} call ledger is incomplete; missing {missing_calls[:5]}"
        )
    return ValidatedLedgerPair(
        calls=tuple(sorted(calls, key=lambda item: item.call_id)),
        segment_probabilities={
            segment_id: float(row["probability"])
            for segment_id, row in sorted(by_segment.items())
        },
        threshold=threshold,
    )


def _validate_probability_score_ledgers(
    *,
    entry: Mapping[str, Any],
    refs: Mapping[str, ImmutableRef],
    config: MatrixConfig,
    expected_segments: Mapping[str, ManifestSegment],
    manifest_sha256: str,
    checkpoint_sha256: str,
    expected_epoch: int,
) -> ValidatedLedgerPair:
    """Validate one epoch's threshold-free validation probability ledgers."""

    coordinate = _coordinate_values(entry)
    run_id = str(entry["run_id"])
    segment_rows = _read_jsonl(
        refs["validation_segment_scores"].path, "validation segment score ledger"
    )
    by_segment: dict[str, Mapping[str, Any]] = {}
    by_call: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for line_number, row in enumerate(segment_rows, start=1):
        context = f"validation segment score row {line_number}"
        _strict_keys(row, SEGMENT_SCORE_ROW_KEYS, context)
        if row["schema_version"] != SEGMENT_SCORE_SCHEMA_VERSION:
            raise Gate1AnalysisError(f"{context} schema version mismatch")
        if _positive_int(row["epoch"], f"{context}.epoch") != expected_epoch:
            raise Gate1AnalysisError(f"{context} epoch differs from epoch trace")
        _validate_coordinate_fields(
            row,
            context=context,
            run_id=run_id,
            protocol_id=config.protocol_id,
            coordinate=coordinate,
            partition="validation",
        )
        if row["run_family"] != "primary_matrix":
            raise Gate1AnalysisError(f"{context} is not a primary-matrix score")
        if row["checkpoint_sha256"] != checkpoint_sha256:
            raise Gate1AnalysisError(f"{context} checkpoint hash mismatch")
        if row["manifest_sha256"] != manifest_sha256:
            raise Gate1AnalysisError(f"{context} manifest hash mismatch")
        segment_id = _nonempty(row["segment_id"], f"{context}.segment_id")
        if segment_id in by_segment:
            raise Gate1AnalysisError(f"duplicate validation score segment {segment_id!r}")
        if segment_id not in expected_segments:
            raise Gate1AnalysisError(f"unknown validation score segment {segment_id!r}")
        expected = expected_segments[segment_id]
        if (
            str(row["call_id"]) != expected.call_id
            or str(row["group_id"]) != expected.group_id
            or _label(row["label"], f"{context}.label") != expected.label
            or str(row["source"]) != expected.source
        ):
            raise Gate1AnalysisError(f"{context} ID/label/group/source mismatch")
        start = _finite_float(row["start"], f"{context}.start", lower=0.0)
        end = _finite_float(row["end"], f"{context}.end", lower=0.0)
        duration = _finite_float(
            row["duration_seconds"], f"{context}.duration_seconds", lower=0.0
        )
        if (
            not math.isclose(start, expected.start, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(end, expected.end, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise Gate1AnalysisError(f"{context} timing mismatch")
        _probability(row["probability"], f"{context}.probability")
        by_segment[segment_id] = row
        by_call[expected.call_id].append(row)
    missing_segments = sorted(set(expected_segments) - set(by_segment))
    if missing_segments:
        raise Gate1AnalysisError(
            f"validation score ledger is incomplete; missing {missing_segments[:5]}"
        )

    call_rows = _read_jsonl(
        refs["validation_call_scores"].path, "validation call score ledger"
    )
    seen_calls: set[str] = set()
    calls: list[Gate1Call] = []
    for line_number, row in enumerate(call_rows, start=1):
        context = f"validation call score row {line_number}"
        _strict_keys(row, CALL_SCORE_ROW_KEYS, context)
        if row["schema_version"] != CALL_SCORE_SCHEMA_VERSION:
            raise Gate1AnalysisError(f"{context} schema version mismatch")
        if _positive_int(row["epoch"], f"{context}.epoch") != expected_epoch:
            raise Gate1AnalysisError(f"{context} epoch differs from epoch trace")
        _validate_coordinate_fields(
            row,
            context=context,
            run_id=run_id,
            protocol_id=config.protocol_id,
            coordinate=coordinate,
            partition="validation",
        )
        if row["run_family"] != "primary_matrix":
            raise Gate1AnalysisError(f"{context} is not a primary-matrix score")
        if row["checkpoint_sha256"] != checkpoint_sha256:
            raise Gate1AnalysisError(f"{context} checkpoint hash mismatch")
        if row["manifest_sha256"] != manifest_sha256:
            raise Gate1AnalysisError(f"{context} manifest hash mismatch")
        call_id = _nonempty(row["call_id"], f"{context}.call_id")
        if call_id in seen_calls or call_id not in by_call:
            raise Gate1AnalysisError(f"duplicate or unknown validation score call {call_id!r}")
        seen_calls.add(call_id)
        segments = by_call[call_id]
        expected_first = expected_segments[str(segments[0]["segment_id"])]
        if (
            str(row["group_id"]) != expected_first.group_id
            or _label(row["label"], f"{context}.label") != expected_first.label
            or str(row["source"]) != expected_first.source
        ):
            raise Gate1AnalysisError(f"{context} ID/label/group/source mismatch")
        recomputed = float(
            np.mean(
                [_probability(item["probability"], f"{context}.segment_probability") for item in segments],
                dtype=np.float64,
            )
        )
        probability = _probability(row["probability"], f"{context}.probability")
        if not math.isclose(probability, recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise Gate1AnalysisError(f"{context} is not the complete-call arithmetic mean")
        n_segments = _positive_int(row["n_segments"], f"{context}.n_segments")
        if n_segments != len(segments) or row["aggregation_rule"] != "mean_probability":
            raise Gate1AnalysisError(f"{context} aggregation metadata mismatch")
        start = _finite_float(row["start"], f"{context}.start", lower=0.0)
        end = _finite_float(row["end"], f"{context}.end", lower=0.0)
        duration = _finite_float(
            row["duration_seconds"], f"{context}.duration_seconds", lower=0.0
        )
        if (
            not math.isclose(start, min(float(item["start"]) for item in segments), rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(end, max(float(item["end"]) for item in segments), rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise Gate1AnalysisError(f"{context} timing mismatch")
        calls.append(
            Gate1Call(
                condition=coordinate[0],
                partition_repeat=coordinate[1],
                outer_fold=coordinate[2],
                seed=coordinate[3],
                call_id=call_id,
                group_id=expected_first.group_id,
                label=expected_first.label,
                probability=recomputed,
                hard_prediction=int(recomputed >= 0.5),
                threshold=0.5,
                n_segments=n_segments,
                source=expected_first.source,
                start=start,
                end=end,
            )
        )
    missing_calls = sorted(set(by_call) - seen_calls)
    if missing_calls:
        raise Gate1AnalysisError(
            f"validation call score ledger is incomplete; missing {missing_calls[:5]}"
        )
    return ValidatedLedgerPair(
        calls=tuple(sorted(calls, key=lambda item: item.call_id)),
        segment_probabilities={
            segment_id: float(row["probability"])
            for segment_id, row in sorted(by_segment.items())
        },
        threshold=None,
    )


def _validate_epoch_trace(
    *,
    project_root: Path,
    trace_ref: ImmutableRef,
    entry: Mapping[str, Any],
    refs: Mapping[str, ImmutableRef],
    config: MatrixConfig,
    expected_segments: Mapping[str, ManifestSegment],
    validation_manifest_sha256: str,
) -> tuple[ValidatedLedgerPair, dict[str, Any]]:
    trace = _read_json(trace_ref.path, "epoch trace")
    _strict_keys(trace, EPOCH_TRACE_KEYS, "epoch trace")
    if trace["schema_version"] != EPOCH_TRACE_SCHEMA_VERSION:
        raise Gate1AnalysisError("epoch trace schema version mismatch")
    if trace["run_family"] != "primary_matrix":
        raise Gate1AnalysisError("epoch trace is not primary-matrix")
    _validate_coordinate_fields(
        trace,
        context="epoch trace",
        run_id=str(entry["run_id"]),
        protocol_id=config.protocol_id,
        coordinate=_coordinate_values(entry),
    )
    if trace["monitor"] != "validation_call_nll":
        raise Gate1AnalysisError("epoch trace changed checkpoint monitor")
    if trace["tie_break"] != "earlier_epoch":
        raise Gate1AnalysisError("epoch trace changed checkpoint tie-break")
    epochs_raw = trace["epochs"]
    if not isinstance(epochs_raw, list) or not epochs_raw:
        raise Gate1AnalysisError("epoch trace must contain at least one epoch")
    epoch_records: list[
        tuple[int, float, ValidatedLedgerPair, dict[str, ImmutableRef]]
    ] = []
    seen_epochs: set[int] = set()
    epoch_paths: set[str] = set()
    for index, raw in enumerate(epochs_raw):
        context = f"epoch trace epochs[{index}]"
        if not isinstance(raw, Mapping):
            raise Gate1AnalysisError(f"{context} must be an object")
        _strict_keys(raw, EPOCH_RECORD_KEYS, context)
        epoch = _positive_int(raw["epoch"], f"{context}.epoch")
        if epoch in seen_epochs:
            raise Gate1AnalysisError(f"epoch trace repeats epoch {epoch}")
        seen_epochs.add(epoch)
        reported_nll = _finite_float(
            raw["validation_call_nll"], f"{context}.validation_call_nll", lower=0.0
        )
        _finite_float(raw["train_reported_loss"], f"{context}.train_reported_loss")
        diagnostic_threshold = _probability(
            raw["diagnostic_threshold"], f"{context}.diagnostic_threshold"
        )
        epoch_refs = {
            "checkpoint": ImmutableRef.from_mapping(
                raw["checkpoint"],
                project_root=project_root,
                context=f"{context}.checkpoint",
                ledger=False,
            ),
            "validation_segment_scores": ImmutableRef.from_mapping(
                raw["validation_segment_scores"],
                project_root=project_root,
                context=f"{context}.validation_segment_scores",
                ledger=True,
            ),
            "validation_call_scores": ImmutableRef.from_mapping(
                raw["validation_call_scores"],
                project_root=project_root,
                context=f"{context}.validation_call_scores",
                ledger=True,
            ),
        }
        for name in ("checkpoint", "validation_segment_scores", "validation_call_scores"):
            path = epoch_refs[name].serialized_path
            if path in epoch_paths:
                raise Gate1AnalysisError(f"epoch trace reuses artifact path {path!r}")
            epoch_paths.add(path)
        epoch_pair = _validate_probability_score_ledgers(
            entry=entry,
            refs=epoch_refs,
            config=config,
            expected_segments=expected_segments,
            manifest_sha256=validation_manifest_sha256,
            checkpoint_sha256=epoch_refs["checkpoint"].sha256,
            expected_epoch=epoch,
        )
        recomputed_nll = _call_nll(epoch_pair.calls)
        if not math.isclose(recomputed_nll, reported_nll, rel_tol=0.0, abs_tol=1e-12):
            raise Gate1AnalysisError(
                f"{context} validation call NLL mismatch: "
                f"reported={reported_nll}, recomputed={recomputed_nll}"
            )
        _, recomputed_diagnostic_threshold = recompute_threshold_candidates(
            epoch_pair.calls, config
        )
        if not math.isclose(
            diagnostic_threshold,
            recomputed_diagnostic_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Gate1AnalysisError(f"{context} diagnostic threshold is incorrect")
        epoch_records.append((epoch, recomputed_nll, epoch_pair, epoch_refs))
    ordered_epochs = sorted(seen_epochs)
    if ordered_epochs != list(range(1, max(ordered_epochs) + 1)):
        raise Gate1AnalysisError("epoch trace must contain consecutive epochs starting at one")
    selected = min(epoch_records, key=lambda item: (item[1], item[0]))
    selected_epoch, selected_nll, selected_pair, selected_refs = selected
    if int(trace["selected_epoch"]) != selected_epoch:
        raise Gate1AnalysisError("epoch trace selected epoch is not min-NLL/earlier-epoch")
    if int(entry["selected_epoch"]) != selected_epoch:
        raise Gate1AnalysisError("indexed selection epoch is not min-NLL/earlier-epoch")
    if not math.isclose(
        float(entry["validation_call_nll"]), selected_nll, rel_tol=0.0, abs_tol=1e-12
    ):
        raise Gate1AnalysisError("indexed selection validation call NLL is incorrect")
    if selected_refs["checkpoint"].as_mapping() != refs["checkpoint"].as_mapping():
        raise Gate1AnalysisError("selected epoch does not bind indexed checkpoint")
    return selected_pair, {
        "epochs_recomputed": len(epoch_records),
        "selected_epoch": selected_epoch,
        "selected_validation_call_nll": selected_nll,
        "epoch_artifact_files_verified": 3 * len(epoch_records),
        "minimum_nll_then_earlier_epoch_verified": True,
    }


def _validate_threshold_trace(
    *,
    trace_ref: ImmutableRef,
    entry: Mapping[str, Any],
    refs: Mapping[str, ImmutableRef],
    config: MatrixConfig,
    selected_validation_calls: Sequence[Gate1Call],
) -> dict[str, Any]:
    trace = _read_json(trace_ref.path, "threshold trace")
    _strict_keys(trace, THRESHOLD_TRACE_KEYS, "threshold trace")
    if trace["schema_version"] != THRESHOLD_TRACE_SCHEMA_VERSION:
        raise Gate1AnalysisError("threshold trace schema version mismatch")
    if trace["run_family"] != "primary_matrix":
        raise Gate1AnalysisError("threshold trace is not primary-matrix")
    _validate_coordinate_fields(
        trace,
        context="threshold trace",
        run_id=str(entry["run_id"]),
        protocol_id=config.protocol_id,
        coordinate=_coordinate_values(entry),
        partition="validation",
    )
    if trace["validation_call_ledger"] != refs["validation_call_ledger"].as_mapping():
        raise Gate1AnalysisError(
            "threshold trace does not bind selected validation call ledger"
        )
    grid = trace["threshold_grid"]
    wanted_grid = {
        "start": config.selection.threshold_start,
        "end": config.selection.threshold_end,
        "step": config.selection.threshold_step,
        "count": 99,
    }
    _assert_float_tree_equal(grid, wanted_grid, context="threshold_trace.threshold_grid")
    if trace["threshold_tie_break"] != list(config.selection.threshold_tie_break):
        raise Gate1AnalysisError("threshold trace tie-break differs from frozen policy")
    candidates, selected_threshold = recompute_threshold_candidates(
        selected_validation_calls, config
    )
    _assert_float_tree_equal(
        trace["candidates"], candidates, context="threshold_trace.candidates"
    )
    if not math.isclose(
        float(trace["selected_threshold"]),
        selected_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise Gate1AnalysisError("threshold trace selected threshold is incorrect")
    if not math.isclose(
        float(entry["threshold"]), selected_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise Gate1AnalysisError("indexed selection threshold differs from validation winner")
    if any(
        not math.isclose(call.threshold, selected_threshold, rel_tol=0.0, abs_tol=1e-12)
        for call in selected_validation_calls
    ):
        raise Gate1AnalysisError("selected validation call ledger uses the wrong threshold")
    return {
        "threshold_candidates_recomputed": len(candidates),
        "selected_threshold": selected_threshold,
        "frozen_tie_break_verified": True,
        "threshold_selected_from_validation_only": True,
    }


def _validate_outer_metrics_artifact(
    metrics_ref: ImmutableRef,
    *,
    entry: Mapping[str, Any],
    config: MatrixConfig,
    calls: Sequence[Gate1Call],
    checkpoint_sha256: str,
    outer_manifest_sha256: str,
) -> None:
    payload = _read_json(metrics_ref.path, "outer-evaluation metrics")
    _strict_keys(payload, OUTER_METRICS_KEYS, "outer-evaluation metrics")
    if payload["schema_version"] != "paper2-v2-work9-outer-metrics-1":
        raise Gate1AnalysisError("outer-evaluation metrics schema version mismatch")
    _validate_coordinate_fields(
        payload,
        context="outer-evaluation metrics",
        run_id=str(entry["run_id"]),
        protocol_id=config.protocol_id,
        coordinate=_coordinate_values(entry),
        partition="outer_eval",
    )
    if (
        payload["run_family"] != "primary_matrix"
        or payload["aggregation_rule"] != "mean_probability"
        or payload["checkpoint_sha256"] != checkpoint_sha256
        or payload["outer_eval_manifest_sha256"] != outer_manifest_sha256
        or payload["interpretation"]
        != "within_corpus_source_confounded_outer_development_evaluation"
    ):
        raise Gate1AnalysisError("outer-evaluation metrics lineage/policy mismatch")
    threshold = _probability(payload["threshold"], "outer-evaluation metrics.threshold")
    if not math.isclose(threshold, float(entry["threshold"]), rel_tol=0.0, abs_tol=1e-15):
        raise Gate1AnalysisError("outer-evaluation metrics threshold mismatch")
    recomputed = compute_declared_call_metrics(calls)
    confusion = recomputed["confusion"]
    negative_f1 = _safe_ratio(
        2 * confusion["tn"],
        2 * confusion["tn"] + confusion["fp"] + confusion["fn"],
    )
    expected = {
        "n_calls": recomputed["n_calls"],
        "threshold": threshold,
        "tp": confusion["tp"],
        "tn": confusion["tn"],
        "fp": confusion["fp"],
        "fn": confusion["fn"],
        "sensitivity": recomputed["sensitivity"],
        "specificity": recomputed["specificity"],
        "precision": recomputed["precision"],
        "negative_predictive_value": recomputed["negative_predictive_value"],
        "positive_f1": recomputed["positive_f1"],
        "negative_f1": negative_f1,
        "macro_f1": recomputed["macro_f1"],
        "balanced_accuracy": recomputed["balanced_accuracy"],
        "mcc": recomputed["mcc"],
        "fpr": recomputed["fpr"],
        "fnr": recomputed["fnr"],
        "roc_auc": recomputed["roc_auc"],
        "pr_auc": recomputed["pr_auc"],
        "brier": recomputed["brier"],
        "nll": recomputed["nll"],
        "ece_10": recomputed["ece_10"],
    }
    _assert_float_tree_equal(
        payload["metrics"], expected, context="outer-evaluation metrics.metrics"
    )


def validate_gate1_inputs(
    *,
    project_root: Path,
    matrix_config_path: Path,
    outer_ledger_index_path: Path,
    expected_outer_ledger_index_sha256: str,
    config: MatrixConfig | None = None,
) -> tuple[MatrixConfig, ValidatedGate1Input]:
    """Validate canonical Work 9 indexes and every selection/outer ledger row."""

    root = project_root.resolve()
    matrix_path = matrix_config_path.resolve()
    outer_index_path = outer_ledger_index_path.resolve()
    matrix_sha256 = sha256_file(matrix_path)
    expected_index_hash = _require_sha256(
        expected_outer_ledger_index_sha256,
        "expected_outer_ledger_index_sha256",
    )
    actual_index_hash = sha256_file(outer_index_path)
    if actual_index_hash != expected_index_hash:
        raise Gate1AnalysisError(
            "outer ledger index hash mismatch: "
            f"expected {expected_index_hash}, got {actual_index_hash}"
        )
    if config is None:
        config = load_matrix_config(matrix_path, project_root=root)
    if matrix_sha256 != WORK8_CONFIG_SHA256:
        raise Gate1AnalysisError(
            "Gate-1 requires the canonical frozen Work 8 matrix config SHA-256"
        )
    if tuple(config.seeds) != CANONICAL_SEEDS:
        raise Gate1AnalysisError("Gate-1 requires exactly the five frozen neural seeds")

    expected_coordinates = {
        (condition, fold.partition_repeat, fold.outer_fold, seed)
        for condition in config.conditions
        for fold in config.folds
        for seed in config.seeds
    }
    expected_count = len(expected_coordinates)
    if expected_count != EXPECTED_SELECTION_COUNT:
        raise Gate1AnalysisError("Gate-1 requires exactly 200 primary-matrix coordinates")

    outer_index = _read_json(outer_index_path, "Work 9 outer ledger index")
    _strict_keys(outer_index, OUTER_INDEX_KEYS, "Work 9 outer ledger index")
    if outer_index["schema_version"] != INDEX_SCHEMA_VERSION:
        raise Gate1AnalysisError("outer ledger index schema version mismatch")
    if (
        outer_index["protocol_id"] != config.protocol_id
        or outer_index["run_family"] != "primary_matrix"
        or outer_index["status"] != "frozen_complete"
    ):
        raise Gate1AnalysisError("outer ledger index protocol/family/status mismatch")
    if outer_index["expected_coordinate_count"] != expected_count:
        raise Gate1AnalysisError("outer ledger index expected_coordinate_count mismatch")
    outer_rows = outer_index["entries"]
    if not isinstance(outer_rows, list) or len(outer_rows) != expected_count:
        raise Gate1AnalysisError(
            f"outer ledger index must contain exactly {expected_count} entries, "
            f"found {len(outer_rows) if isinstance(outer_rows, list) else 'non-list'}"
        )

    outer_analysis_protocol_ref = ImmutableRef.from_mapping(
        outer_index["gate1_analysis_protocol"],
        project_root=root,
        context="outer ledger index.gate1_analysis_protocol",
        ledger=False,
    )
    if (
        outer_analysis_protocol_ref.path
        != (root / GATE1_ANALYSIS_PROTOCOL_RELATIVE).resolve()
        or outer_analysis_protocol_ref.sha256 != GATE1_ANALYSIS_PROTOCOL_SHA256
    ):
        raise Gate1AnalysisError("outer ledger index binds another analysis protocol")

    selection_index_ref = ImmutableRef.from_mapping(
        outer_index["selection_index"],
        project_root=root,
        context="outer ledger index.selection_index",
        ledger=False,
    )
    selection_index = _read_json(selection_index_ref.path, "Work 9 selection index")
    _strict_keys(selection_index, SELECTION_INDEX_KEYS, "Work 9 selection index")
    if selection_index["schema_version"] != SELECTION_INDEX_SCHEMA_VERSION:
        raise Gate1AnalysisError("selection index schema version mismatch")
    if (
        selection_index["protocol_id"] != config.protocol_id
        or selection_index["run_family"] != "primary_matrix"
        or selection_index["status"] != "frozen_complete"
    ):
        raise Gate1AnalysisError("selection index protocol/family/status mismatch")
    if selection_index["expected_coordinate_count"] != expected_count:
        raise Gate1AnalysisError("selection index expected_coordinate_count mismatch")
    selection_rows = selection_index["entries"]
    if not isinstance(selection_rows, list) or len(selection_rows) != expected_count:
        raise Gate1AnalysisError(
            f"selection index must contain exactly {expected_count} entries, "
            f"found {len(selection_rows) if isinstance(selection_rows, list) else 'non-list'}"
        )

    selection_inputs = {
        name: ImmutableRef.from_mapping(
            selection_index[name],
            project_root=root,
            context=f"selection index.{name}",
            ledger=False,
        )
        for name in (
            "config",
            "work8_receipt",
            "runner_acceptance",
            "experiment_card",
            "gate1_policy",
            "gate1_analysis_protocol",
        )
    }
    if (
        selection_index["gate1_analysis_protocol"]
        != outer_analysis_protocol_ref.as_mapping()
    ):
        raise Gate1AnalysisError(
            "outer and selection indexes bind different analysis protocols"
        )
    if selection_inputs["config"].path != matrix_path:
        raise Gate1AnalysisError("selection index does not bind the supplied matrix config")
    _validate_selection_index_lineage_inputs(
        selection_inputs,
        project_root=root,
        config=config,
        matrix_config_sha256=matrix_sha256,
    )
    pairing_plan_sha256 = _pairing_plan_digests(
        selection_inputs["work8_receipt"], config=config
    )

    def index_entry_map(
        rows: Sequence[Any],
        *,
        keys: frozenset[str],
        context: str,
    ) -> dict[tuple[str, int, int, int], Mapping[str, Any]]:
        result: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
        run_ids: set[str] = set()
        for row_number, row in enumerate(rows, start=1):
            row_context = f"{context} entry {row_number}"
            if not isinstance(row, Mapping):
                raise Gate1AnalysisError(f"{row_context} must be an object")
            _strict_keys(row, keys, row_context)
            coordinate = _coordinate_values(row)
            if coordinate not in expected_coordinates:
                raise Gate1AnalysisError(
                    f"{row_context} has unexpected coordinate {coordinate}"
                )
            if coordinate in result:
                raise Gate1AnalysisError(f"{context} repeats coordinate {coordinate}")
            run_id = _nonempty(row["run_id"], f"{row_context}.run_id")
            canonical_run_id = Work9Coordinates(
                protocol_id=config.protocol_id,
                run_family="primary_matrix",
                condition=coordinate[0],
                partition_repeat=coordinate[1],
                outer_fold=coordinate[2],
                seed=coordinate[3],
            ).run_id
            if run_id != canonical_run_id:
                raise Gate1AnalysisError(
                    f"{row_context} run_id is not canonical for its coordinate"
                )
            if run_id in run_ids:
                raise Gate1AnalysisError(f"{context} repeats run_id {run_id!r}")
            run_ids.add(run_id)
            result[coordinate] = row
        missing = sorted(expected_coordinates - set(result))
        if missing:
            raise Gate1AnalysisError(f"{context} is incomplete: {missing[:5]}")
        return result

    outer_by_coordinate = index_entry_map(
        outer_rows, keys=OUTER_ENTRY_KEYS, context="outer ledger index"
    )
    selection_by_coordinate = index_entry_map(
        selection_rows, keys=SELECTION_ENTRY_KEYS, context="selection index"
    )

    ledger_paths: set[str] = set()
    manifest_cache: dict[tuple[str, int, int], dict[str, ManifestSegment]] = {}
    calls_by_coordinate: dict[
        tuple[str, int, int, int], tuple[Gate1Call, ...]
    ] = {}
    verified_artifact_paths = {
        selection_index_ref.serialized_path,
        outer_analysis_protocol_ref.serialized_path,
        *(artifact.serialized_path for artifact in selection_inputs.values()),
    }
    recomputed_epochs = 0
    recomputed_threshold_candidates = 0
    for coordinate in sorted(expected_coordinates):
        outer_entry = outer_by_coordinate[coordinate]
        selection_entry = selection_by_coordinate[coordinate]
        if outer_entry["run_id"] != selection_entry["run_id"]:
            raise Gate1AnalysisError(f"Work 9 indexes disagree on run_id for {coordinate}")
        condition, partition_repeat, outer_fold, seed = coordinate
        fold = config.fold(partition_repeat, outer_fold)

        selection_seal_ref = ImmutableRef.from_mapping(
            selection_entry["selection_seal"],
            project_root=root,
            context=f"selection seal {coordinate}",
            ledger=False,
        )
        raw_seal = _read_json(selection_seal_ref.path, f"selection seal {coordinate}")
        _strict_keys(raw_seal, SELECTION_SEAL_KEYS, f"selection seal {coordinate}")
        entry = {
            "run_id": outer_entry["run_id"],
            "condition": condition,
            "partition_repeat": partition_repeat,
            "outer_fold": outer_fold,
            "seed": seed,
            "selected_epoch": raw_seal["selected_epoch"],
            "validation_call_nll": raw_seal["validation_call_nll"],
            "threshold": raw_seal["threshold"],
        }
        _positive_int(entry["selected_epoch"], f"selection seal {coordinate}.selected_epoch")
        _finite_float(
            entry["validation_call_nll"],
            f"selection seal {coordinate}.validation_call_nll",
            lower=0.0,
        )
        threshold = _probability(entry["threshold"], f"selection seal {coordinate}.threshold")
        if not math.isclose(
            _probability(outer_entry["threshold"], f"outer index {coordinate}.threshold"),
            threshold,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise Gate1AnalysisError(f"outer and selection indexes disagree on threshold {coordinate}")

        refs = {
            "checkpoint": ImmutableRef.from_mapping(
                raw_seal["checkpoint"],
                project_root=root,
                context=f"checkpoint {coordinate}",
                ledger=False,
            ),
            "validation_segment_ledger": ImmutableRef.from_mapping(
                raw_seal["validation_segment_ledger"],
                project_root=root,
                context=f"validation segment ledger {coordinate}",
                ledger=True,
            ),
            "validation_call_ledger": ImmutableRef.from_mapping(
                raw_seal["validation_call_ledger"],
                project_root=root,
                context=f"validation call ledger {coordinate}",
                ledger=True,
            ),
            "epoch_trace": ImmutableRef.from_mapping(
                selection_entry["epoch_trace"],
                project_root=root,
                context=f"epoch trace {coordinate}",
                ledger=False,
            ),
            "threshold_trace": ImmutableRef.from_mapping(
                selection_entry["threshold_trace"],
                project_root=root,
                context=f"threshold trace {coordinate}",
                ledger=False,
            ),
            "outer_segment_ledger": ImmutableRef.from_mapping(
                outer_entry["outer_eval_segment_ledger"],
                project_root=root,
                context=f"outer segment ledger {coordinate}",
                ledger=True,
            ),
            "outer_call_ledger": ImmutableRef.from_mapping(
                outer_entry["outer_eval_call_ledger"],
                project_root=root,
                context=f"outer call ledger {coordinate}",
                ledger=True,
            ),
            "selection_seal": selection_seal_ref,
            "selection_run_record": ImmutableRef.from_mapping(
                selection_entry["selection_run_record"],
                project_root=root,
                context=f"selection run record {coordinate}",
                ledger=False,
            ),
            "outer_metrics": ImmutableRef.from_mapping(
                outer_entry["outer_eval_metrics"],
                project_root=root,
                context=f"outer metrics {coordinate}",
                ledger=False,
            ),
            "outer_run_record": ImmutableRef.from_mapping(
                outer_entry["run_record"],
                project_root=root,
                context=f"outer run record {coordinate}",
                ledger=False,
            ),
        }
        raw_pairing_artifact = raw_seal["pairing_plan_artifact"]
        if raw_pairing_artifact is not None:
            refs["pairing_plan"] = ImmutableRef.from_mapping(
                raw_pairing_artifact,
                project_root=root,
                context=f"pairing plan {coordinate}",
                ledger=True,
            )
        if raw_seal["epoch_trace"] != refs["epoch_trace"].as_mapping():
            raise Gate1AnalysisError(f"selection index/seal disagree on epoch trace {coordinate}")
        if raw_seal["threshold_trace"] != refs["threshold_trace"].as_mapping():
            raise Gate1AnalysisError(
                f"selection index/seal disagree on threshold trace {coordinate}"
            )
        if outer_entry["checkpoint_sha256"] != refs["checkpoint"].sha256:
            raise Gate1AnalysisError(f"outer index checkpoint hash mismatch for {coordinate}")

        for name in (
            "validation_segment_ledger",
            "validation_call_ledger",
            "outer_segment_ledger",
            "outer_call_ledger",
            *(("pairing_plan",) if "pairing_plan" in refs else ()),
        ):
            path_key = refs[name].serialized_path
            if path_key in ledger_paths:
                raise Gate1AnalysisError(
                    f"immutable ledger path reused across primary runs: {path_key}"
                )
            ledger_paths.add(path_key)
        expected_pairing = (
            pairing_plan_sha256[(partition_repeat, outer_fold, seed)]
            if condition == "AT-cross-randtrain"
            else None
        )
        _validate_selection_seal(
            refs["selection_seal"],
            entry=entry,
            refs=refs,
            config=config,
            matrix_config_sha256=matrix_sha256,
            selection_inputs=selection_inputs,
            train_manifest_sha256=fold.train_manifest.sha256,
            validation_manifest_sha256=fold.val_manifest.sha256,
            expected_pairing_plan_sha256=expected_pairing,
        )
        _validate_run_record(
            refs["selection_run_record"],
            entry=entry,
            config=config,
            phase="selection",
            status="selection_complete",
            required_artifact_refs=(
                refs["selection_seal"],
                refs["checkpoint"],
                refs["validation_segment_ledger"],
                refs["validation_call_ledger"],
                refs["epoch_trace"],
                refs["threshold_trace"],
                *((refs["pairing_plan"],) if "pairing_plan" in refs else ()),
            ),
            matrix_config_sha256=matrix_sha256,
            selection_inputs=selection_inputs,
            train_manifest_sha256=fold.train_manifest.sha256,
            validation_manifest_sha256=fold.val_manifest.sha256,
            outer_eval_manifest_sha256=None,
            pairing_plan_sha256=expected_pairing,
        )
        _validate_run_record(
            refs["outer_run_record"],
            entry=entry,
            config=config,
            phase="outer_eval",
            status="outer_complete",
            required_artifact_refs=(
                refs["selection_seal"],
                refs["checkpoint"],
                refs["outer_segment_ledger"],
                refs["outer_call_ledger"],
                refs["outer_metrics"],
            ),
            matrix_config_sha256=matrix_sha256,
            selection_inputs=selection_inputs,
            train_manifest_sha256=fold.train_manifest.sha256,
            validation_manifest_sha256=fold.val_manifest.sha256,
            outer_eval_manifest_sha256=fold.outer_eval_manifest.sha256,
            pairing_plan_sha256=expected_pairing,
        )
        outer_key = ("outer_eval", partition_repeat, outer_fold)
        if outer_key not in manifest_cache:
            manifest_cache[outer_key] = _manifest_segments(
                fold.outer_eval_manifest.path,
                partition_repeat=partition_repeat,
                outer_fold=outer_fold,
                manifest_partition="outer_eval",
            )
        val_key = ("val", partition_repeat, outer_fold)
        if val_key not in manifest_cache:
            manifest_cache[val_key] = _manifest_segments(
                fold.val_manifest.path,
                partition_repeat=partition_repeat,
                outer_fold=outer_fold,
                manifest_partition="val",
            )
        train_key = ("train", partition_repeat, outer_fold)
        if train_key not in manifest_cache:
            manifest_cache[train_key] = _manifest_segments(
                fold.train_manifest.path,
                partition_repeat=partition_repeat,
                outer_fold=outer_fold,
                manifest_partition="train",
            )
        if expected_pairing is not None:
            _validate_pairing_plan_artifact(
                refs["pairing_plan"],
                seed=seed,
                expected_segments=manifest_cache[train_key],
                expected_semantic_sha256=expected_pairing,
            )
        selected_epoch_pair, epoch_audit = _validate_epoch_trace(
            project_root=root,
            trace_ref=refs["epoch_trace"],
            entry=entry,
            refs=refs,
            config=config,
            expected_segments=manifest_cache[val_key],
            validation_manifest_sha256=fold.val_manifest.sha256,
        )
        selected_validation_pair = _validate_prediction_ledgers(
            entry=entry,
            refs=refs,
            config=config,
            expected_segments=manifest_cache[val_key],
            partition="validation",
            manifest_sha256=fold.val_manifest.sha256,
            segment_ref_name="validation_segment_ledger",
            call_ref_name="validation_call_ledger",
            checkpoint_sha256=refs["checkpoint"].sha256,
            expected_threshold=float(entry["threshold"]),
            expected_segment_probabilities=selected_epoch_pair.segment_probabilities,
        )
        if not math.isclose(
            _call_nll(selected_validation_pair.calls),
            float(entry["validation_call_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Gate1AnalysisError(
                f"selected validation ledger NLL differs for run {coordinate}"
            )
        threshold_audit = _validate_threshold_trace(
            trace_ref=refs["threshold_trace"],
            entry=entry,
            refs=refs,
            config=config,
            selected_validation_calls=selected_validation_pair.calls,
        )
        outer_pair = _validate_prediction_ledgers(
            entry=entry,
            refs=refs,
            config=config,
            expected_segments=manifest_cache[outer_key],
            partition="outer_eval",
            manifest_sha256=fold.outer_eval_manifest.sha256,
            segment_ref_name="outer_segment_ledger",
            call_ref_name="outer_call_ledger",
            checkpoint_sha256=refs["checkpoint"].sha256,
            expected_threshold=float(entry["threshold"]),
        )
        _validate_outer_metrics_artifact(
            refs["outer_metrics"],
            entry=entry,
            config=config,
            calls=outer_pair.calls,
            checkpoint_sha256=refs["checkpoint"].sha256,
            outer_manifest_sha256=fold.outer_eval_manifest.sha256,
        )
        calls_by_coordinate[coordinate] = outer_pair.calls
        verified_artifact_paths.update(ref.serialized_path for ref in refs.values())
        recomputed_epochs += int(epoch_audit["epochs_recomputed"])
        recomputed_threshold_candidates += int(
            threshold_audit["threshold_candidates_recomputed"]
        )

    canonical_identity: dict[str, tuple[str, int, int, str]] | None = None
    expected_call_count: int | None = None
    for condition in config.conditions:
        for seed in config.seeds:
            pooled = [
                call
                for fold in config.folds
                for call in calls_by_coordinate[
                    (condition, fold.partition_repeat, fold.outer_fold, seed)
                ]
            ]
            call_counts = Counter(call.call_id for call in pooled)
            repeated = sorted(call_id for call_id, count in call_counts.items() if count != 1)
            if repeated:
                raise Gate1AnalysisError(
                    f"condition={condition} seed={seed} does not outer-score each call once: "
                    f"{repeated[:5]}"
                )
            identity = {
                call.call_id: (
                    call.group_id,
                    call.label,
                    call.outer_fold,
                    call.source,
                )
                for call in pooled
            }
            if canonical_identity is None:
                canonical_identity = identity
                expected_call_count = len(identity)
            elif identity != canonical_identity:
                raise Gate1AnalysisError(
                    f"call/label/group/fold identity differs for condition={condition}, seed={seed}"
                )
    assert canonical_identity is not None and expected_call_count is not None
    labels = {label for _, label, _, _ in canonical_identity.values()}
    if labels != {0, 1}:
        raise Gate1AnalysisError("primary pooled call set must contain both labels")
    groups = {group for group, _, _, _ in canonical_identity.values()}
    audit = {
        "status": "pass",
        "primary_run_count": len(outer_rows),
        "expected_primary_run_count": len(expected_coordinates),
        "conditions": len(config.conditions),
        "outer_folds": len(config.folds),
        "seeds": len(config.seeds),
        "eligible_call_count": expected_call_count,
        "relation_group_count": len(groups),
        "verified_artifact_files": len(verified_artifact_paths) + 3 * recomputed_epochs,
        "unique_ledger_files": len(ledger_paths),
        "all_coordinates_complete": True,
        "all_artifact_hashes_verified": True,
        "canonical_outer_and_selection_indexes_joined": True,
        "frozen_gate1_analysis_protocol_lineage_verified": True,
        "runner_acceptance_and_experiment_card_lineage_verified": True,
        "selection_run_records_verified": True,
        "outer_run_records_verified": True,
        "random_train_pairing_hashes_match_work8_receipt": True,
        "random_train_pairing_artifacts_semantically_recomputed": True,
        "random_train_pairing_artifacts_recomputed": len(pairing_plan_sha256),
        "all_manifest_segments_present_once": True,
        "all_call_means_recomputed": True,
        "all_outer_metrics_recomputed": True,
        "all_epoch_validation_call_means_and_nll_recomputed": True,
        "epoch_validation_ledgers_recomputed": recomputed_epochs,
        "threshold_candidates_recomputed": recomputed_threshold_candidates,
        "minimum_validation_nll_then_earlier_epoch_verified": True,
        "validation_threshold_grid_and_tie_break_verified": True,
        "all_call_ids_labels_groups_match_frozen_manifests": True,
        "outer_call_scored_once_per_condition_seed": True,
        "capacity_family_rows_accepted": 0,
    }
    return config, ValidatedGate1Input(
        calls_by_coordinate=calls_by_coordinate,
        outer_ledger_index_sha256=actual_index_hash,
        selection_index_sha256=selection_index_ref.sha256,
        selection_index_path=selection_index_ref.path,
        matrix_config_sha256=matrix_sha256,
        audit=audit,
    )


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> Any:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.zeros(np.broadcast_shapes(numerator_array.shape, denominator_array.shape))
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array != 0)
    return float(result) if result.ndim == 0 else result


def _ece_10(labels: np.ndarray, probabilities: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    result = 0.0
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        selected = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == 9
            else (probabilities >= lower) & (probabilities < upper)
        )
        if not np.any(selected):
            continue
        result += float(selected.mean()) * abs(
            float(probabilities[selected].mean()) - float(labels[selected].mean())
        )
    return float(result)


def compute_declared_call_metrics(calls: Sequence[Gate1Call]) -> dict[str, Any]:
    """Compute the exact locked call metric family using per-fold sealed decisions."""

    if not calls:
        raise Gate1AnalysisError("call metrics require at least one complete call")
    labels = np.asarray([call.label for call in calls], dtype=np.int64)
    probabilities = np.asarray([call.probability for call in calls], dtype=np.float64)
    predictions = np.asarray([call.hard_prediction for call in calls], dtype=np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    positive_f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _safe_ratio(2 * tn, 2 * tn + fp + fn)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_ratio(tp * tn - fp * fn, mcc_denominator)
    both_labels = len(set(labels.tolist())) == 2
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    metrics = {
        "macro_f1": float((positive_f1 + negative_f1) / 2.0),
        "positive_f1": float(positive_f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "negative_predictive_value": float(npv),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_labels else None,
        "pr_auc": float(average_precision_score(labels, probabilities)) if both_labels else None,
        "fpr": float(1.0 - specificity),
        "fnr": float(1.0 - sensitivity),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(
            np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped)))
        ),
        "ece_10": _ece_10(labels, probabilities),
    }
    if tuple(metrics) != REPORTED_CALL_METRICS:
        raise Gate1AnalysisError("internal call metric order differs from frozen policy")
    return {
        "n_calls": len(calls),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        **metrics,
    }


def _call_nll(calls: Sequence[Gate1Call]) -> float:
    probabilities = np.clip(
        np.asarray([call.probability for call in calls], dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    labels = np.asarray([call.label for call in calls], dtype=np.float64)
    return float(
        np.mean(
            -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
        )
    )


def recompute_threshold_candidates(
    calls: Sequence[Gate1Call], config: MatrixConfig
) -> tuple[list[dict[str, Any]], float]:
    """Recompute all 99 validation candidates and the frozen deterministic winner."""

    start = config.selection.threshold_start
    end = config.selection.threshold_end
    step = config.selection.threshold_step
    count = int(round((end - start) / step)) + 1
    thresholds = [round(start + index * step, 12) for index in range(count)]
    if len(thresholds) != 99 or thresholds[0] != 0.01 or thresholds[-1] != 0.99:
        raise Gate1AnalysisError("frozen threshold grid did not produce exactly 0.01..0.99")
    candidates: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, float, float, float, float], float]] = []
    for threshold in thresholds:
        thresholded = tuple(
            replace(
                call,
                hard_prediction=int(call.probability >= threshold),
                threshold=threshold,
            )
            for call in calls
        )
        metrics = compute_declared_call_metrics(thresholded)
        key = (
            float(metrics["macro_f1"]),
            -float(metrics["fpr"]),
            float(metrics["sensitivity"]),
            -abs(threshold - 0.5),
            threshold,
        )
        ranked.append((key, threshold))
        candidates.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "selection_key": list(key),
            }
        )
    _, selected = max(ranked, key=lambda item: item[0])
    return candidates, selected


def _assert_float_tree_equal(
    actual: Any, expected: Any, *, context: str, tolerance: float = 1e-12
) -> None:
    """Strictly compare nested trace data with a small serialization tolerance."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise Gate1AnalysisError(f"{context} object keys differ")
        for key in expected:
            _assert_float_tree_equal(
                actual[key], expected[key], context=f"{context}.{key}", tolerance=tolerance
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise Gate1AnalysisError(f"{context} list length differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_float_tree_equal(
                actual_item,
                expected_item,
                context=f"{context}[{index}]",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError) as error:
            raise Gate1AnalysisError(f"{context} is not numeric") from error
        if not math.isclose(actual_float, expected, rel_tol=0.0, abs_tol=tolerance):
            raise Gate1AnalysisError(
                f"{context} differs: expected {expected}, got {actual_float}"
            )
        return
    if actual != expected:
        raise Gate1AnalysisError(f"{context} differs: expected {expected!r}, got {actual!r}")


def _metric_summary(values: Sequence[float | None]) -> dict[str, Any]:
    defined = np.asarray([float(value) for value in values if value is not None])
    if len(defined) == 0:
        return {
            "defined_count": 0,
            "mean": None,
            "sample_standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "defined_count": int(len(defined)),
        "mean": float(defined.mean()),
        "sample_standard_deviation": (
            float(defined.std(ddof=1)) if len(defined) > 1 else 0.0
        ),
        "minimum": float(defined.min()),
        "maximum": float(defined.max()),
    }


def build_condition_metrics(
    calls_by_coordinate: Mapping[tuple[str, int, int, int], tuple[Gate1Call, ...]],
    config: MatrixConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int, int | None], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int, int | None], dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for condition in config.conditions:
        for seed in config.seeds:
            pooled = tuple(
                call
                for fold in config.folds
                for call in calls_by_coordinate[
                    (condition, fold.partition_repeat, fold.outer_fold, seed)
                ]
            )
            metrics = compute_declared_call_metrics(pooled)
            lookup[(condition, seed, None)] = metrics
            records.append(
                {
                    "schema_version": "paper2-v2-work9-gate1-condition-metrics-1",
                    "scope": "pooled_five_outer_folds_per_seed",
                    "condition": condition,
                    "seed": seed,
                    "outer_fold": None,
                    "metrics": metrics,
                }
            )
            for fold in config.folds:
                fold_calls = calls_by_coordinate[
                    (condition, fold.partition_repeat, fold.outer_fold, seed)
                ]
                fold_metrics = compute_declared_call_metrics(fold_calls)
                lookup[(condition, seed, fold.outer_fold)] = fold_metrics
                records.append(
                    {
                        "schema_version": "paper2-v2-work9-gate1-condition-metrics-1",
                        "scope": "outer_fold_per_seed",
                        "condition": condition,
                        "seed": seed,
                        "outer_fold": fold.outer_fold,
                        "metrics": fold_metrics,
                    }
                )
        metric_seed_summary = {
            metric: _metric_summary(
                [lookup[(condition, seed, None)][metric] for seed in config.seeds]
            )
            for metric in REPORTED_CALL_METRICS
        }
        per_fold_mean = {
            str(fold.outer_fold): {
                metric: _metric_summary(
                    [
                        lookup[(condition, seed, fold.outer_fold)][metric]
                        for seed in config.seeds
                    ]
                )["mean"]
                for metric in REPORTED_CALL_METRICS
            }
            for fold in config.folds
        }
        fold_stability = {
            metric: _metric_summary(
                [per_fold_mean[str(fold.outer_fold)][metric] for fold in config.folds]
            )
            for metric in REPORTED_CALL_METRICS
        }
        summaries[condition] = {
            "combined_estimator": "mean_of_per_seed_metrics_after_pooling_all_five_outer_folds",
            "metrics_across_seeds": metric_seed_summary,
            "per_outer_fold_mean_across_seeds": per_fold_mean,
            "metrics_across_outer_fold_means": fold_stability,
        }
    records.sort(
        key=lambda row: (
            config.conditions.index(str(row["condition"])),
            int(row["seed"]),
            -1 if row["outer_fold"] is None else int(row["outer_fold"]),
        )
    )
    return records, summaries, lookup


def _macro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    tn, fp, fn, tp = (counts[..., index] for index in range(4))
    positive = _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)
    negative = _safe_ratio(2.0 * tn, 2.0 * tn + fp + fn)
    return (positive + negative) / 2.0


def _cluster_confusion_contributions(
    calls_by_coordinate: Mapping[tuple[str, int, int, int], tuple[Gate1Call, ...]],
    *,
    condition: str,
    seeds: Sequence[int],
    folds: Sequence[Any],
    group_ids: Sequence[str],
) -> np.ndarray:
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    contributions = np.zeros((len(seeds), len(group_ids), 4), dtype=np.int64)
    for seed in seeds:
        for fold in folds:
            for call in calls_by_coordinate[
                (condition, fold.partition_repeat, fold.outer_fold, seed)
            ]:
                if call.label == 0 and call.hard_prediction == 0:
                    cell = 0
                elif call.label == 0 and call.hard_prediction == 1:
                    cell = 1
                elif call.label == 1 and call.hard_prediction == 0:
                    cell = 2
                else:
                    cell = 3
                contributions[seed_index[seed], group_index[call.group_id], cell] += 1
    return contributions


def _bootstrap_effects(
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if replicates <= 0:
        raise Gate1AnalysisError("bootstrap replicates must be positive")
    groups = primary.shape[1]
    rng = np.random.default_rng(seed)
    effects = np.empty(replicates, dtype=np.float64)
    probabilities = np.full(groups, 1.0 / groups, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        multiplicities = rng.multinomial(groups, probabilities, size=size)
        primary_counts = np.einsum("rg,sgk->rsk", multiplicities, primary)
        comparator_counts = np.einsum("rg,sgk->rsk", multiplicities, comparator)
        effects[start : start + size] = (
            _macro_f1_from_counts(primary_counts)
            - _macro_f1_from_counts(comparator_counts)
        ).mean(axis=1)
    return effects


def _permutation_effects(
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if replicates <= 0:
        raise Gate1AnalysisError("permutation replicates must be positive")
    groups = primary.shape[1]
    rng = np.random.default_rng(seed)
    effects = np.empty(replicates, dtype=np.float64)
    primary_total = primary.sum(axis=1)
    comparator_total = comparator.sum(axis=1)
    delta = comparator - primary
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        swap = rng.integers(0, 2, size=(size, groups), dtype=np.int8)
        transferred = np.einsum("rg,sgk->rsk", swap, delta)
        permuted_primary = primary_total[None, :, :] + transferred
        permuted_comparator = comparator_total[None, :, :] - transferred
        effects[start : start + size] = (
            _macro_f1_from_counts(permuted_primary)
            - _macro_f1_from_counts(permuted_comparator)
        ).mean(axis=1)
    return effects


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original hypothesis order."""

    if not p_values:
        return []
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise Gate1AnalysisError("Holm inputs must be finite p-values")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    adjusted = [0.0] * len(values)
    running = 0.0
    hypotheses = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (hypotheses - rank) * values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def analyze_primary_comparisons(
    *,
    calls_by_coordinate: Mapping[tuple[str, int, int, int], tuple[Gate1Call, ...]],
    config: MatrixConfig,
    metric_lookup: Mapping[tuple[str, int, int | None], Mapping[str, Any]],
    bootstrap_replicates: int | None = None,
    permutation_replicates: int | None = None,
) -> list[dict[str, Any]]:
    """Run paired relation-group bootstrap and cluster-swap permutation tests."""

    bootstrap_n = (
        config.statistics.bootstrap_replicates
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    permutation_n = (
        config.statistics.permutation_replicates
        if permutation_replicates is None
        else permutation_replicates
    )
    group_ids = sorted(
        {
            call.group_id
            for calls in calls_by_coordinate.values()
            for call in calls
        }
    )
    contribution_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for comparison_index, (primary_name, comparator_name) in enumerate(
        config.statistics.primary_comparisons
    ):
        for condition in (primary_name, comparator_name):
            if condition not in contribution_cache:
                contribution_cache[condition] = _cluster_confusion_contributions(
                    calls_by_coordinate,
                    condition=condition,
                    seeds=config.seeds,
                    folds=config.folds,
                    group_ids=group_ids,
                )
        primary = contribution_cache[primary_name]
        comparator = contribution_cache[comparator_name]
        seed_differences = [
            float(metric_lookup[(primary_name, seed, None)]["macro_f1"])
            - float(metric_lookup[(comparator_name, seed, None)]["macro_f1"])
            for seed in config.seeds
        ]
        observed = float(np.mean(seed_differences))
        bootstrap = _bootstrap_effects(
            primary,
            comparator,
            replicates=bootstrap_n,
            seed=config.statistics.bootstrap_seed,
        )
        alpha = 1.0 - config.statistics.confidence_level
        confidence_interval = np.quantile(
            bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
        )
        permutation = _permutation_effects(
            primary,
            comparator,
            replicates=permutation_n,
            seed=config.statistics.permutation_seed,
        )
        raw_p = float(
            (1 + np.count_nonzero(np.abs(permutation) >= abs(observed) - 1e-15))
            / (permutation_n + 1)
        )
        seed_array = np.asarray(seed_differences, dtype=np.float64)
        seed_sd = float(seed_array.std(ddof=1)) if len(seed_array) > 1 else 0.0
        paired_dz = None if math.isclose(seed_sd, 0.0) else float(observed / seed_sd)
        per_fold_seed_differences = {
            str(fold.outer_fold): [
                float(metric_lookup[(primary_name, seed, fold.outer_fold)]["macro_f1"])
                - float(metric_lookup[(comparator_name, seed, fold.outer_fold)]["macro_f1"])
                for seed in config.seeds
            ]
            for fold in config.folds
        }
        per_fold_means = {
            fold: float(np.mean(values))
            for fold, values in per_fold_seed_differences.items()
        }
        rows.append(
            {
                "schema_version": "paper2-v2-work9-gate1-primary-comparison-1",
                "comparison_index": comparison_index,
                "comparison_id": f"{primary_name}_minus_{comparator_name}",
                "primary_condition": primary_name,
                "comparator_condition": comparator_name,
                "metric": "macro_f1",
                "estimand": config.statistics.primary_estimand,
                "observed_effect": observed,
                "effect_direction": "primary_minus_comparator",
                "effect_sizes": {
                    "macro_f1_difference": observed,
                    "absolute_macro_f1_difference_magnitude": abs(observed),
                    "paired_seed_cohens_dz": paired_dz,
                    "positive_seed_fraction": float(np.mean(seed_array > 0.0)),
                    "nonnegative_seed_fraction": float(np.mean(seed_array >= 0.0)),
                },
                "seed_stability": {
                    "per_seed_differences": {
                        str(seed): value
                        for seed, value in zip(config.seeds, seed_differences, strict=True)
                    },
                    "sample_standard_deviation": seed_sd,
                    "minimum": float(seed_array.min()),
                    "maximum": float(seed_array.max()),
                },
                "fold_stability": {
                    "per_fold_per_seed_differences": per_fold_seed_differences,
                    "per_fold_mean_across_seeds": per_fold_means,
                    "mean_summary": _metric_summary(list(per_fold_means.values())),
                },
                "relation_group_clusters": len(group_ids),
                "call_count": len(
                    {
                        call.call_id
                        for calls in calls_by_coordinate.values()
                        for call in calls
                    }
                ),
                "seed_count": len(config.seeds),
                "bootstrap": {
                    "method": config.statistics.bootstrap_method,
                    "replicates": bootstrap_n,
                    "seed": config.statistics.bootstrap_seed,
                    "interval_method": "percentile_linear_quantile",
                    "confidence_level": config.statistics.confidence_level,
                    "ci_lower": float(confidence_interval[0]),
                    "ci_upper": float(confidence_interval[1]),
                },
                "permutation": {
                    "method": config.statistics.permutation_method,
                    "replicates": permutation_n,
                    "seed": config.statistics.permutation_seed,
                    "alternative": "two_sided_absolute_effect",
                    "finite_sample_correction": "plus_one_numerator_and_denominator",
                    "raw_p_value": raw_p,
                },
                "multiplicity": {
                    "method": config.statistics.multiplicity_method,
                    "family_size": len(config.statistics.primary_comparisons),
                    "adjusted_p_value": None,
                    "familywise_alpha": config.statistics.familywise_alpha,
                    "reject_after_holm": None,
                },
                "practical_utility_claim": "forbidden_not_made",
            }
        )
    adjusted = holm_adjust([row["permutation"]["raw_p_value"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        row["multiplicity"]["adjusted_p_value"] = adjusted_p
        row["multiplicity"]["reject_after_holm"] = bool(
            adjusted_p < config.statistics.familywise_alpha
        )
    return rows


def load_gate1_decision_policy(path: Path) -> Mapping[str, Any]:
    actual_hash = sha256_file(path)
    if actual_hash != GATE1_DECISION_POLICY_SHA256:
        raise Gate1AnalysisError(
            "Gate-1 decision policy hash mismatch: "
            f"expected {GATE1_DECISION_POLICY_SHA256}, got {actual_hash}"
        )
    policy = _read_json(path, "frozen Gate-1 decision policy")
    expected_top_keys = frozenset(
        {
            "alpha",
            "cross_mechanism_superiority",
            "gate2_continuation",
            "practical_utility",
            "schema_version",
            "stability",
            "statistical_incremental_text_conditioning",
            "status",
        }
    )
    _strict_keys(policy, expected_top_keys, "frozen Gate-1 decision policy")
    if policy["schema_version"] != "paper2-v2-work9-gate1-decision-policy-1":
        raise Gate1AnalysisError("Gate-1 decision policy schema mismatch")
    if policy["status"] != "frozen_before_work9_model_outcomes":
        raise Gate1AnalysisError("Gate-1 decision policy was not frozen before outcomes")
    if not math.isclose(float(policy["alpha"]), 0.05, rel_tol=0.0, abs_tol=1e-15):
        raise Gate1AnalysisError("Gate-1 decision alpha changed")
    incremental = policy["statistical_incremental_text_conditioning"]
    mechanism = policy["cross_mechanism_superiority"]
    if not isinstance(incremental, Mapping) or not isinstance(mechanism, Mapping):
        raise Gate1AnalysisError("Gate-1 decision families must be objects")
    if incremental.get("comparison") != "AT-cross_minus_A-query":
        raise Gate1AnalysisError("Gate-1 incremental comparison changed")
    if mechanism.get("all_comparisons_required") != [
        "AT-cross_minus_AT-concat",
        "AT-cross_minus_AT-gate",
    ]:
        raise Gate1AnalysisError("Gate-1 mechanism comparison family changed")
    expected_requirements = {
        "clustered_bootstrap_ci_lower_strictly_greater_than": 0.0,
        "holm_adjusted_permutation_p_strictly_less_than": 0.05,
        "observed_macro_f1_difference_strictly_greater_than": 0.0,
    }
    if incremental.get("requirements") != expected_requirements:
        raise Gate1AnalysisError("Gate-1 incremental criteria changed")
    if mechanism.get("per_comparison_requirements") != expected_requirements:
        raise Gate1AnalysisError("Gate-1 mechanism criteria changed")
    practical = policy["practical_utility"]
    if not isinstance(practical, Mapping) or practical.get("claim_authorized") is not False:
        raise Gate1AnalysisError("Gate-1 policy improperly authorizes practical utility")
    return policy


def load_gate1_analysis_protocol(path: Path) -> Mapping[str, Any]:
    actual_hash = sha256_file(path)
    if actual_hash != GATE1_ANALYSIS_PROTOCOL_SHA256:
        raise Gate1AnalysisError(
            "Gate-1 analysis protocol hash mismatch: "
            f"expected {GATE1_ANALYSIS_PROTOCOL_SHA256}, got {actual_hash}"
        )
    protocol = _read_json(path, "frozen Gate-1 analysis protocol")
    _strict_keys(
        protocol,
        frozenset(
            {
                "bootstrap",
                "call_primary",
                "coordinate_contract",
                "decision_policy_sha256",
                "effect_sizes",
                "estimand",
                "fold_seed_stability",
                "multiplicity",
                "permutation",
                "practical_utility",
                "protocol_id",
                "resampling_cluster",
                "run_family",
                "schema_version",
                "status",
                "work8_config_sha256",
                "work8_receipt_sha256",
            }
        ),
        "frozen Gate-1 analysis protocol",
    )
    expected_scalars = {
        "schema_version": "paper2-v2-work9-gate1-analysis-protocol-1",
        "status": "frozen_before_work9_outer_outcomes",
        "run_family": "primary_matrix",
        "work8_config_sha256": WORK8_CONFIG_SHA256,
        "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
        "decision_policy_sha256": GATE1_DECISION_POLICY_SHA256,
        "estimand": "mean_across_seeds_of_paired_call_macro_f1_difference",
    }
    if any(protocol.get(key) != value for key, value in expected_scalars.items()):
        raise Gate1AnalysisError("Gate-1 analysis protocol lineage/estimand changed")
    bootstrap = protocol["bootstrap"]
    permutation = protocol["permutation"]
    multiplicity = protocol["multiplicity"]
    if not all(isinstance(item, Mapping) for item in (bootstrap, permutation, multiplicity)):
        raise Gate1AnalysisError("Gate-1 resampling policies must be objects")
    if (
        bootstrap.get("replicates") != 10000
        or bootstrap.get("seed") != 20260809
        or bootstrap.get("interval") != "percentile"
        or bootstrap.get("quantile_method") != "linear"
        or bootstrap.get("confidence_level") != 0.95
        or permutation.get("replicates") != 10000
        or permutation.get("seed") != 20260810
        or permutation.get("alternative") != "two_sided_absolute_effect"
        or permutation.get("finite_sample_correction")
        != "plus_one_numerator_and_denominator"
        or multiplicity.get("method") != "holm"
        or multiplicity.get("familywise_alpha") != 0.05
        or multiplicity.get("rejection_rule")
        != "adjusted_p_strictly_less_than_alpha"
    ):
        raise Gate1AnalysisError("Gate-1 frozen resampling/multiplicity choices changed")
    return protocol


def classify_gate1_continuation(
    comparisons: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only the frozen pre-outcome statistical continuation rules."""

    by_id = {str(row["comparison_id"]): row for row in comparisons}

    def criteria(comparison_id: str) -> dict[str, Any]:
        if comparison_id not in by_id:
            raise Gate1AnalysisError(f"missing Gate-1 comparison {comparison_id}")
        row = by_id[comparison_id]
        observed = float(row["observed_effect"])
        ci_lower = float(row["bootstrap"]["ci_lower"])
        holm_p = float(row["multiplicity"]["adjusted_p_value"])
        checks = {
            "observed_effect_strictly_positive": observed > 0.0,
            "clustered_bootstrap_ci_lower_strictly_positive": ci_lower > 0.0,
            "holm_adjusted_permutation_p_strictly_below_0_05": holm_p < 0.05,
        }
        return {
            "comparison_id": comparison_id,
            "observed_effect": observed,
            "clustered_bootstrap_ci_lower": ci_lower,
            "holm_adjusted_permutation_p": holm_p,
            "checks": checks,
            "all_requirements_met": all(checks.values()),
        }

    incremental_check = criteria("AT-cross_minus_A-query")
    mechanism_checks = [
        criteria("AT-cross_minus_AT-concat"),
        criteria("AT-cross_minus_AT-gate"),
    ]

    def status(checks: Sequence[Mapping[str, Any]]) -> str:
        if all(bool(item["all_requirements_met"]) for item in checks):
            return "supported"
        if any(float(item["observed_effect"]) <= 0.0 for item in checks):
            return "unsupported"
        return "inconclusive"

    incremental_status = status([incremental_check])
    mechanism_status = status(mechanism_checks)
    return {
        "schema_version": "paper2-v2-work9-gate1-continuation-classification-1",
        "decision_policy_sha256": GATE1_DECISION_POLICY_SHA256,
        "classification_is_statistical_not_practical": True,
        "statistical_incremental_text_conditioning": {
            "status": incremental_status,
            "comparison": incremental_check,
        },
        "cross_mechanism_superiority": {
            "status": mechanism_status,
            "all_comparisons_required": mechanism_checks,
        },
        "gate2_cross_specific_pairing_controls": (
            "proceed"
            if incremental_status == "supported"
            else "do_not_proceed"
        ),
        "cross_mechanism_superiority_claim": (
            "statistically_supported"
            if mechanism_status == "supported"
            else f"{mechanism_status}_no_claim"
        ),
        "seed_and_fold_signs_role": (
            "descriptive_only_not_additional_post_hoc_pass_rules"
        ),
        "practical_utility_claim": "forbidden_not_made",
        "policy_snapshot": dict(policy),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                _json_ready(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _validate_report_schema(report: Mapping[str, Any], *, project_root: Path) -> None:
    try:
        import jsonschema
    except ImportError as error:
        raise Gate1AnalysisError("jsonschema is required to seal a Gate-1 report") from error

    try:
        schema = json.loads(
            (
                project_root
                / "revised_experiments/registry/schemas/work9_gate1_report.schema.json"
            ).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(schema).validate(_json_ready(report))
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as error:
        raise Gate1AnalysisError(f"Gate-1 report schema validation failed: {error}") from error


def _human_report(
    *,
    config: MatrixConfig,
    audit: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
) -> str:
    lines = [
        "# Work 9 Gate-1 post-run analysis",
        "",
        "Status: sealed statistical analysis of immutable outer-evaluation ledgers.",
        "No model training, checkpoint selection, threshold selection, or practical-utility claim is performed here.",
        "",
        "## Validation",
        "",
        f"- Completed primary runs: {audit['primary_run_count']}",
        f"- Eligible complete calls: {audit['eligible_call_count']}",
        f"- Relation-group clusters: {audit['relation_group_count']}",
        "- Every call probability was recomputed as the arithmetic mean of every manifest segment.",
        "- All ledger, manifest, selection-seal, checkpoint, run-record, and canonical-index hashes were checked.",
        "",
        "## Preregistered primary comparisons",
        "",
        "| Comparison | Mean macro-F1 difference | 95% clustered bootstrap CI | Raw permutation p | Holm p | Reject |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in comparisons:
        bootstrap = row["bootstrap"]
        permutation = row["permutation"]
        multiplicity = row["multiplicity"]
        lines.append(
            "| {comparison} | {effect:.6f} | [{lower:.6f}, {upper:.6f}] | "
            "{raw:.6g} | {adjusted:.6g} | {reject} |".format(
                comparison=row["comparison_id"],
                effect=row["observed_effect"],
                lower=bootstrap["ci_lower"],
                upper=bootstrap["ci_upper"],
                raw=permutation["raw_p_value"],
                adjusted=multiplicity["adjusted_p_value"],
                reject="yes" if multiplicity["reject_after_holm"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Frozen continuation classification",
            "",
            "- Statistical incremental text conditioning: "
            + str(classification["statistical_incremental_text_conditioning"]["status"]),
            "- Cross-mechanism superiority: "
            + str(classification["cross_mechanism_superiority"]["status"]),
            "- Gate-2 cross-specific pairing controls: "
            + str(classification["gate2_cross_specific_pairing_controls"]),
            "- Seed and fold signs are descriptive and are not additional pass rules.",
            "",
            "## Claim boundary",
            "",
            "The frozen policy has no cost function and no smallest practically meaningful effect.",
            "Accordingly, this report describes statistical effects and uncertainty only; it makes no clinical, operational, or practical-utility claim.",
            "All results remain within-corpus outer-development evaluation.",
            "",
            "Frozen analysis protocol: "
            f"`{GATE1_ANALYSIS_PROTOCOL_RELATIVE.as_posix()}` "
            f"(SHA-256 `{GATE1_ANALYSIS_PROTOCOL_SHA256}`).",
            f"Protocol: `{config.protocol_id}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _hash_manifest(directory: Path, names: Sequence[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}\n" for name in sorted(names)]
    (directory / "hashes.sha256").write_text("".join(lines), encoding="utf-8")


def run_gate1_analysis(
    *,
    project_root: Path,
    matrix_config_path: Path,
    outer_ledger_index_path: Path,
    expected_outer_ledger_index_sha256: str,
    decision_policy_path: Path,
    analysis_protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate, analyze, and atomically seal a new Gate-1 report directory."""

    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Gate-1 output already exists and is immutable: {output}")
    decision_policy = load_gate1_decision_policy(decision_policy_path.resolve())
    analysis_protocol_path = analysis_protocol_path.resolve()
    expected_analysis_protocol_path = (
        project_root.resolve() / GATE1_ANALYSIS_PROTOCOL_RELATIVE
    ).resolve()
    if analysis_protocol_path != expected_analysis_protocol_path:
        raise Gate1AnalysisError("Gate-1 analysis protocol must use the canonical path")
    analysis_protocol = load_gate1_analysis_protocol(analysis_protocol_path)
    config, validated = validate_gate1_inputs(
        project_root=project_root,
        matrix_config_path=matrix_config_path,
        outer_ledger_index_path=outer_ledger_index_path,
        expected_outer_ledger_index_sha256=expected_outer_ledger_index_sha256,
    )
    condition_records, condition_summaries, metric_lookup = build_condition_metrics(
        validated.calls_by_coordinate, config
    )
    comparisons = analyze_primary_comparisons(
        calls_by_coordinate=validated.calls_by_coordinate,
        config=config,
        metric_lookup=metric_lookup,
    )
    continuation = classify_gate1_continuation(comparisons, decision_policy)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass",
        "analysis_role": "post_run_gate1_immutable_ledger_statistics",
        "protocol_id": config.protocol_id,
        "within_corpus_only": True,
        "inputs": {
            "matrix_config": {
                "path": str(matrix_config_path.resolve()),
                "sha256": validated.matrix_config_sha256,
            },
            "outer_ledger_index": {
                "path": str(outer_ledger_index_path.resolve()),
                "sha256": validated.outer_ledger_index_sha256,
            },
            "selection_index": {
                "path": str(validated.selection_index_path),
                "sha256": validated.selection_index_sha256,
            },
            "gate1_decision_policy": {
                "path": str(decision_policy_path.resolve()),
                "sha256": GATE1_DECISION_POLICY_SHA256,
            },
            "gate1_analysis_protocol": {
                "path": str(analysis_protocol_path),
                "sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
            },
        },
        "validation": dict(validated.audit),
        "call_primary_policy": {
            "independent_unit": "call",
            "complete_call_aggregation": "mean_probability_over_all_manifest_segments",
            "outer_folds_combined_before_per_seed_metric": 5,
            "seeds_combined_by": "arithmetic_mean_of_five_per_seed_metrics",
            "reported_call_metrics": list(REPORTED_CALL_METRICS),
            "segment_metrics_role": config.statistics.segment_metrics_role,
        },
        "condition_summaries": condition_summaries,
        "primary_comparisons": comparisons,
        "gate1_continuation_classification": continuation,
        "multiplicity": {
            "method": "holm",
            "familywise_alpha": config.statistics.familywise_alpha,
            "family": [list(pair) for pair in PRIMARY_COMPARISONS],
        },
        "practical_utility": {
            "frozen_rule": config.statistics.practical_claim_rule,
            "cost_policy_available": False,
            "smallest_practically_meaningful_effect": None,
            "claim_permitted": False,
            "claim_made": False,
        },
    }
    stability = {
        "schema_version": "paper2-v2-work9-gate1-stability-1",
        "condition_fold_seed_stability": condition_summaries,
        "primary_comparison_fold_seed_stability": {
            row["comparison_id"]: {
                "seed_stability": row["seed_stability"],
                "fold_stability": row["fold_stability"],
            }
            for row in comparisons
        },
    }
    protocol = {
        "schema_version": "paper2-v2-work9-gate1-protocol-1",
        "artifact_status": "sealed",
        "protocol_id": config.protocol_id,
        "analysis_is_training_free": True,
        "matrix_config_sha256": validated.matrix_config_sha256,
        "outer_ledger_index_sha256": validated.outer_ledger_index_sha256,
        "selection_index_sha256": validated.selection_index_sha256,
        "gate1_decision_policy_sha256": GATE1_DECISION_POLICY_SHA256,
        "gate1_analysis_protocol": {
            "path": GATE1_ANALYSIS_PROTOCOL_RELATIVE.as_posix(),
            "sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
            "schema_version": analysis_protocol["schema_version"],
            "status": analysis_protocol["status"],
        },
        "primary_estimand": config.statistics.primary_estimand,
        "primary_comparisons": [list(pair) for pair in config.statistics.primary_comparisons],
        "reported_call_metrics": list(config.statistics.reported_call_metrics),
        "bootstrap": {
            "method": config.statistics.bootstrap_method,
            "replicates": config.statistics.bootstrap_replicates,
            "seed": config.statistics.bootstrap_seed,
            "confidence_level": config.statistics.confidence_level,
            "interval": "percentile_linear_quantile",
        },
        "permutation": {
            "method": config.statistics.permutation_method,
            "replicates": config.statistics.permutation_replicates,
            "seed": config.statistics.permutation_seed,
            "alternative": "two_sided_absolute_effect",
        },
        "multiplicity": {
            "method": config.statistics.multiplicity_method,
            "familywise_alpha": config.statistics.familywise_alpha,
        },
        "practical_claim_rule": config.statistics.practical_claim_rule,
    }
    _validate_report_schema(report, project_root=project_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "validation_audit.json", validated.audit)
        _write_jsonl(staging / "condition_metrics.jsonl", condition_records)
        _write_jsonl(staging / "primary_comparisons.jsonl", comparisons)
        _write_json(staging / "stability.json", stability)
        _write_json(staging / "report.json", report)
        (staging / "REPORT.md").write_text(
            _human_report(
                config=config,
                audit=validated.audit,
                comparisons=comparisons,
                classification=continuation,
            ),
            encoding="utf-8",
        )
        (staging / "READ_ONLY_NOTICE.txt").write_text(
            "Paper 2 Work 9 Gate-1 immutable-ledger analysis. Do not edit in place; "
            "build a new output directory and retain the original checksums.\n",
            encoding="utf-8",
        )
        names = [
            "protocol.json",
            "validation_audit.json",
            "condition_metrics.jsonl",
            "primary_comparisons.jsonl",
            "stability.json",
            "report.json",
            "REPORT.md",
            "READ_ONLY_NOTICE.txt",
        ]
        _hash_manifest(staging, names)
        staging.rename(output)
        for artifact in output.iterdir():
            if artifact.is_file():
                artifact.chmod(0o444)
        output.chmod(0o555)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "pass",
        "output_dir": str(output),
        "report": report,
        "hash_manifest_sha256": sha256_file(output / "hashes.sha256"),
    }


__all__ = [
    "CALL_SCHEMA_VERSION",
    "GATE1_ANALYSIS_PROTOCOL_SHA256",
    "Gate1AnalysisError",
    "Gate1Call",
    "INDEX_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SEGMENT_SCHEMA_VERSION",
    "analyze_primary_comparisons",
    "build_condition_metrics",
    "compute_declared_call_metrics",
    "holm_adjust",
    "load_gate1_analysis_protocol",
    "run_gate1_analysis",
    "validate_gate1_inputs",
]
