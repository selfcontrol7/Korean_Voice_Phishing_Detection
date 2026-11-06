"""Fail-closed analysis for the Work 9 capacity-matched sensitivity.

This module is intentionally separate from the completed Gate-1 validator.  It
normalizes the *hybrid* capacity family (150 newly fitted coordinates and 50
immutable primary anchors), recomputes call-level metrics, and runs exactly the
three secondary comparisons frozen before capacity outcomes existed.  It has
no continuation-decision API and cannot change or reopen Gate 1 or Gate 2.

The filesystem loader validates every native row against the schema belonging
to its source family before creating a common in-memory representation.  Pure
analysis functions accept that already-validated representation so their
scientific behaviour can be tested entirely with synthetic outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import work9_gate1 as scientific_core
from .config import REPORTED_CALL_METRICS, sha256_file
from .evaluation import SegmentPrediction, aggregate_mean_probability
from .work9_capacity_registry import (
    _reject_symlink_components,
    validate_isolated_relative_root,
)
from .work9_capacity_types import (
    ACCEPTANCE_RECEIPT_RELATIVE,
    ANCHOR_CONDITIONS,
    ANCHOR_POLICY_RELATIVE,
    ANCHOR_POLICY_SHA256,
    CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    EXECUTION_ADDENDUM_RELATIVE,
    EXECUTION_ADDENDUM_SHA256,
    EXPERIMENT_CARD_RELATIVE,
    EXPERIMENT_CARD_SHA256,
    FIGURE_SPEC_RELATIVE,
    FIGURE_SPEC_SHA256,
    GATE1_HASH_MANIFEST_RELATIVE,
    GATE1_HASH_MANIFEST_SHA256,
    NEW_FIT_CONDITIONS,
    PRIMARY_OUTER_INDEX_SHA256,
    PRIMARY_OUTER_INDEX_RELATIVE,
    PRIMARY_RUNNER_ACCEPTANCE_RELATIVE,
    PRIMARY_RUNNER_ACCEPTANCE_SHA256,
    PRIMARY_SELECTION_INDEX_SHA256,
    PRIMARY_SELECTION_INDEX_RELATIVE,
    REGISTRY_RELATIVE,
    RUN_FAMILY,
    TEST_SANDBOX_ROOT_RELATIVE,
    WORK8_CONFIG_RELATIVE,
    WORK8_CONFIG_SHA256,
    WORK8_RECEIPT_RELATIVE,
    WORK8_RECEIPT_SHA256,
)


ANALYZER_VERSION = "paper2-v2-work9-capacity-analyzer-1"
REPORT_SCHEMA_VERSION = "paper2-v2-work9-capacity-analysis-report-1"
CONDITION_METRIC_SCHEMA_VERSION = (
    "paper2-v2-work9-capacity-condition-metrics-1"
)
COMPARISON_SCHEMA_VERSION = "paper2-v2-work9-capacity-secondary-comparison-1"
PROTOCOL_ID = "paper2-v2-mfcc-matched-within-corpus-v1"
PRIMARY_RUN_FAMILY = "primary_matrix"
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
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (17, 29, 43, 71, 101)
COMPARISONS = (
    ("AT-cross", "A-query"),
    ("AT-cross", "AT-concat"),
    ("AT-cross", "AT-gate"),
)
FIGURE_IDS = (
    "capacity_parameter_match",
    "capacity_condition_macro_f1",
    "capacity_primary_effects",
    "primary_vs_capacity_condition_change",
    "capacity_fold_seed_stability",
    "capacity_metric_profile",
    "capacity_compute_profile",
)
EVIDENCE_LABEL = (
    "Capacity-matched secondary within-corpus sensitivity; cannot reopen Gate 1 or Gate 2"
)
COMPLETED_GATE1_REPORT_SHA256 = (
    "e52c8d4fa07d9cbd77d36c1901220c5e7c286251f95ec75911209e1dcb77ed72"
)
COMPLETED_GATE1_REPORT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_gate1_v2/report.json"
)
ANALYSIS_REPORT_SCHEMA_RELATIVE = Path(
    "revised_experiments/registry/schemas/work9_capacity/"
    "work9_capacity_analysis_report.schema.json"
)
ANALYSIS_OUTPUT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_analysis_v1"
)


class CapacityAnalysisError(RuntimeError):
    """Raised when hybrid lineage or the frozen secondary contract drifts."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    materialized = list(rows)
    if not materialized:
        raise CapacityAnalysisError("refusing to serialize an empty analysis ledger")
    return (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for row in materialized
        )
        + "\n"
    ).encode("utf-8")


def _freeze_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            item.chmod(0o444)
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir()), reverse=True
    ):
        directory.chmod(0o555)
    path.chmod(0o555)


def _restore_tree_for_cleanup(path: Path) -> None:
    if not path.exists():
        return
    path.chmod(0o755)
    for item in path.rglob("*"):
        item.chmod(0o755 if item.is_dir() else 0o644)


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityAnalysisError(f"cannot read {context}: {path}") from error
    if not isinstance(value, Mapping):
        raise CapacityAnalysisError(f"{context} must be a JSON object")
    return value


def _read_jsonl(path: Path, context: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise CapacityAnalysisError(
                        f"{context} row {line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityAnalysisError(f"cannot read {context}: {path}") from error
    if not rows:
        raise CapacityAnalysisError(f"{context} is empty")
    return rows


def _safe_project_path(project_root: Path, serialized: Any, context: str) -> Path:
    if not isinstance(serialized, str) or not serialized:
        raise CapacityAnalysisError(f"{context}.path must be a nonempty string")
    relative = Path(serialized)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise CapacityAnalysisError(f"{context}.path must be project-relative and safe")
    root = project_root.resolve()
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CapacityAnalysisError(f"{context}.path contains a symlink component")
    resolved = (root / relative).resolve()
    if resolved == root or root not in resolved.parents:
        raise CapacityAnalysisError(f"{context}.path escapes the project root")
    return resolved


@dataclass(frozen=True)
class ImmutableReference:
    path: Path
    serialized_path: str
    sha256: str
    bytes: int
    row_count: int | None

    @classmethod
    def load(
        cls,
        value: Any,
        *,
        project_root: Path,
        context: str,
        ledger: bool,
    ) -> "ImmutableReference":
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "bytes",
            "row_count",
        }:
            raise CapacityAnalysisError(f"{context} is not an exact artifact reference")
        path = _safe_project_path(project_root, value["path"], context)
        digest = value["sha256"]
        byte_count = value["bytes"]
        row_count = value["row_count"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CapacityAnalysisError(f"{context}.sha256 is invalid")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise CapacityAnalysisError(f"{context}.bytes is invalid")
        if ledger:
            if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
                raise CapacityAnalysisError(f"{context}.row_count is invalid")
        elif row_count is not None:
            raise CapacityAnalysisError(f"{context}.row_count must be null")
        if not path.is_file() or path.stat().st_size != byte_count:
            raise CapacityAnalysisError(f"{context} file/byte count mismatch")
        if sha256_file(path) != digest:
            raise CapacityAnalysisError(f"{context} SHA-256 mismatch")
        if ledger:
            with path.open("r", encoding="utf-8") as handle:
                actual_rows = sum(1 for line in handle if line.strip())
            if actual_rows != row_count:
                raise CapacityAnalysisError(f"{context} row count mismatch")
        return cls(path, str(value["path"]), digest, byte_count, row_count)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "path": self.serialized_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "row_count": self.row_count,
        }


def _validate_schema(
    payload: Mapping[str, Any], *, schema_path: Path, context: str
) -> None:
    try:
        import jsonschema
    except ImportError as error:  # pragma: no cover - environment guard
        raise CapacityAnalysisError("jsonschema is required") from error
    schema = _read_json(schema_path, f"{context} schema")
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        detail = "; ".join(
            f"{list(error.path)}: {error.message}" for error in errors[:10]
        )
        raise CapacityAnalysisError(f"{context} schema validation failed: {detail}")


def _validate_jsonl_schema(
    rows: Sequence[Mapping[str, Any]], *, schema_path: Path, context: str
) -> None:
    if not rows:
        raise CapacityAnalysisError(f"{context} is empty")
    for index, row in enumerate(rows):
        _validate_schema(row, schema_path=schema_path, context=f"{context}[{index}]")


def _verify_frozen_file(project_root: Path, relative: Path, expected: str) -> Path:
    path = project_root.resolve() / relative
    if not path.is_file() or sha256_file(path) != expected:
        raise CapacityAnalysisError(f"frozen input changed: {relative.as_posix()}")
    return path


@dataclass(frozen=True)
class FrozenCapacityPolicies:
    analysis_protocol: Mapping[str, Any]
    figure_spec: Mapping[str, Any]
    experiment_card: Mapping[str, Any]
    execution_addendum: Mapping[str, Any]


def load_frozen_capacity_policies(project_root: str | Path) -> FrozenCapacityPolicies:
    """Load and semantically verify every pre-outcome capacity policy."""

    root = Path(project_root).resolve()
    _verify_frozen_file(root, ANCHOR_POLICY_RELATIVE, ANCHOR_POLICY_SHA256)
    analysis = _read_json(
        _verify_frozen_file(
            root,
            CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
            CAPACITY_ANALYSIS_PROTOCOL_SHA256,
        ),
        "capacity analysis protocol",
    )
    figure = _read_json(
        _verify_frozen_file(root, FIGURE_SPEC_RELATIVE, FIGURE_SPEC_SHA256),
        "capacity figure specification",
    )
    card = _read_json(
        _verify_frozen_file(root, EXPERIMENT_CARD_RELATIVE, EXPERIMENT_CARD_SHA256),
        "capacity experiment card",
    )
    addendum = _read_json(
        _verify_frozen_file(
            root, EXECUTION_ADDENDUM_RELATIVE, EXECUTION_ADDENDUM_SHA256
        ),
        "capacity execution-lineage addendum",
    )

    coordinate = analysis.get("coordinate_contract")
    multiplicity = analysis.get("multiplicity")
    bootstrap = analysis.get("bootstrap")
    permutation = analysis.get("permutation")
    boundary = analysis.get("claim_boundary")
    completed = analysis.get("completed_gate1")
    if not all(
        isinstance(value, Mapping)
        for value in (coordinate, multiplicity, bootstrap, permutation, boundary, completed)
    ):
        raise CapacityAnalysisError("capacity analysis protocol sections are malformed")
    if (
        analysis.get("status") != "frozen_before_capacity_model_outcomes"
        or analysis.get("protocol_id") != PROTOCOL_ID
        or analysis.get("run_family") != RUN_FAMILY
        or tuple(coordinate.get("conditions", ())) != CONDITIONS
        or tuple(coordinate.get("outer_folds", ())) != FOLDS
        or tuple(coordinate.get("seeds", ())) != SEEDS
        or coordinate.get("partition_repeat") != 0
        or coordinate.get("expected_coordinate_count") != 200
        or coordinate.get("new_coordinate_count") != 150
        or coordinate.get("anchor_coordinate_count") != 50
    ):
        raise CapacityAnalysisError("frozen hybrid coordinate contract changed")
    frozen_family = tuple(tuple(pair) for pair in multiplicity.get("family", ()))
    if (
        frozen_family != COMPARISONS
        or multiplicity.get("method") != "holm"
        or multiplicity.get("familywise_alpha") != 0.05
        or bootstrap.get("replicates") != 10000
        or bootstrap.get("seed") != 20260809
        or bootstrap.get("confidence_level") != 0.95
        or bootstrap.get("quantile_method") != "linear"
        or permutation.get("replicates") != 10000
        or permutation.get("seed") != 20260810
        or permutation.get("alternative") != "two_sided_absolute_effect"
    ):
        raise CapacityAnalysisError("frozen three-comparison inference contract changed")
    if (
        boundary.get("can_change_completed_gate1_decision") is not False
        or boundary.get("can_reopen_gate2") is not False
        or boundary.get("external_generalization_authorized") is not False
        or boundary.get("practical_utility_authorized") is not False
        or completed.get("gate2_decision") != "do_not_proceed"
        or completed.get("hash_manifest_sha256") != GATE1_HASH_MANIFEST_SHA256
        or completed.get("report_sha256") != COMPLETED_GATE1_REPORT_SHA256
    ):
        raise CapacityAnalysisError("Gate 1/Gate 2 immutable boundary changed")

    figure_rows = figure.get("figures")
    if (
        figure.get("status") != "frozen_before_capacity_model_outcomes"
        or figure.get("protocol_id") != PROTOCOL_ID
        or figure.get("run_family") != RUN_FAMILY
        or figure.get("evidence_label") != EVIDENCE_LABEL
        or not isinstance(figure_rows, list)
        or tuple(row.get("figure_id") for row in figure_rows) != FIGURE_IDS
        or tuple(figure.get("formats", ())) != ("svg", "pdf", "png_300dpi")
    ):
        raise CapacityAnalysisError("frozen seven-figure contract changed")
    inferential_ids = tuple(
        row["figure_id"] for row in figure_rows if row.get("inferential") is True
    )
    if inferential_ids != ("capacity_primary_effects",):
        raise CapacityAnalysisError("figure inferential/descriptive roles changed")

    hybrid = addendum.get("hybrid_contract")
    publication = addendum.get("analysis_publication_gate")
    addendum_completed = addendum.get("completed_gate1")
    if not all(isinstance(value, Mapping) for value in (hybrid, publication, addendum_completed)):
        raise CapacityAnalysisError("execution addendum analysis boundary is malformed")
    mixed = publication.get("mixed_input_families")
    if (
        hybrid.get("logical_coordinate_count") != 200
        or hybrid.get("new_coordinate_count") != 150
        or hybrid.get("anchor_coordinate_count") != 50
        or hybrid.get("new_run_family") != RUN_FAMILY
        or hybrid.get("anchor_run_family") != PRIMARY_RUN_FAMILY
        or mixed != {RUN_FAMILY: 150, PRIMARY_RUN_FAMILY: 50}
        or publication.get(
            "capacity_outcomes_must_not_be_opened_before_analyzer_renderer_source_tests_and_acceptance_are_frozen"
        )
        is not True
        or addendum_completed.get("may_be_reopened") is not False
        or addendum_completed.get("gate2_decision") != "do_not_proceed"
    ):
        raise CapacityAnalysisError("execution addendum hybrid/publication boundary changed")
    return FrozenCapacityPolicies(analysis, figure, card, addendum)


def expected_coordinate_roles() -> dict[tuple[str, int, int, int], tuple[str, str]]:
    result: dict[tuple[str, int, int, int], tuple[str, str]] = {}
    for condition in CONDITIONS:
        role_family = (
            ("new_capacity_fit", RUN_FAMILY)
            if condition in NEW_FIT_CONDITIONS
            else ("immutable_primary_anchor", PRIMARY_RUN_FAMILY)
        )
        for fold in FOLDS:
            for seed in SEEDS:
                result[(condition, 0, fold, seed)] = role_family
    return result


def _entry_coordinate(entry: Mapping[str, Any], context: str) -> tuple[str, int, int, int]:
    try:
        key = (
            str(entry["condition"]),
            int(entry["partition_repeat"]),
            int(entry["outer_fold"]),
            int(entry["seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CapacityAnalysisError(f"{context} coordinate is malformed") from error
    return key


def expected_run_id(
    coordinate: tuple[str, int, int, int], source_run_family: str
) -> str:
    condition, repeat, fold, seed = coordinate
    return f"{source_run_family}__{condition}__r{repeat:02d}-o{fold:02d}-s{seed}"


def verify_exact_hybrid_entries(
    entries: Any, *, context: str
) -> dict[tuple[str, int, int, int], Mapping[str, Any]]:
    """Require the exact unique 150/50 logical coordinate and family split."""

    if not isinstance(entries, list) or len(entries) != 200:
        raise CapacityAnalysisError(f"{context} must contain exactly 200 entries")
    expected = expected_coordinate_roles()
    indexed: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise CapacityAnalysisError(f"{context}[{index}] is not an object")
        coordinate = _entry_coordinate(entry, f"{context}[{index}]")
        if coordinate in indexed:
            raise CapacityAnalysisError(f"{context} has duplicate coordinate {coordinate}")
        if coordinate not in expected:
            raise CapacityAnalysisError(f"{context} has unexpected coordinate {coordinate}")
        role, family = expected[coordinate]
        if entry.get("entry_role") != role or entry.get("source_run_family") != family:
            raise CapacityAnalysisError(
                f"{context} role/family mismatch at {coordinate}: expected {role}/{family}"
            )
        if entry.get("run_id") != expected_run_id(coordinate, family):
            raise CapacityAnalysisError(f"{context} run_id mismatch at {coordinate}")
        indexed[coordinate] = entry
    if set(indexed) != set(expected):
        raise CapacityAnalysisError(f"{context} does not equal the frozen coordinate set")
    counts = Counter(entry["source_run_family"] for entry in indexed.values())
    if counts != Counter({RUN_FAMILY: 150, PRIMARY_RUN_FAMILY: 50}):
        raise CapacityAnalysisError(f"{context} native family cardinality changed")
    return indexed


@dataclass(frozen=True)
class NormalizedCoordinate:
    run_id: str
    condition: str
    partition_repeat: int
    outer_fold: int
    seed: int
    entry_role: str
    source_run_family: str
    native_segment_schema_version: str
    native_call_schema_version: str
    native_run_record_schema_version: str
    calls: tuple[scientific_core.Gate1Call, ...]
    trainable_parameters: int
    wall_time_seconds: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (
            self.condition,
            self.partition_repeat,
            self.outer_fold,
            self.seed,
        )


@dataclass(frozen=True)
class ValidatedHybridInput:
    coordinates: tuple[NormalizedCoordinate, ...]
    selection_index: Mapping[str, Any]
    outer_index: Mapping[str, Any]
    selection_index_reference: Mapping[str, Any]
    outer_index_reference: Mapping[str, Any]
    normalization_audit: Mapping[str, Any]

    @property
    def calls_by_coordinate(
        self,
    ) -> dict[tuple[str, int, int, int], tuple[scientific_core.Gate1Call, ...]]:
        return {coordinate.key: coordinate.calls for coordinate in self.coordinates}


def _semantic_call_inventory(
    calls: Sequence[scientific_core.Gate1Call],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                call.call_id,
                call.group_id,
                call.label,
                call.n_segments,
                call.source,
                call.start,
                call.end,
            )
            for call in calls
        )
    )


def verify_normalized_hybrid_contract(
    coordinates: Sequence[NormalizedCoordinate],
) -> Mapping[str, Any]:
    """Recheck exact roles, schemas, uniqueness, and paired call inventory."""

    if len(coordinates) != 200:
        raise CapacityAnalysisError("normalized hybrid table must contain 200 coordinates")
    expected = expected_coordinate_roles()
    indexed: dict[tuple[str, int, int, int], NormalizedCoordinate] = {}
    native_schema_counts: Counter[tuple[str, str, str, str]] = Counter()
    for coordinate in coordinates:
        if coordinate.key in indexed:
            raise CapacityAnalysisError(f"duplicate normalized coordinate {coordinate.key}")
        if coordinate.key not in expected:
            raise CapacityAnalysisError(f"unexpected normalized coordinate {coordinate.key}")
        role, family = expected[coordinate.key]
        if (coordinate.entry_role, coordinate.source_run_family) != (role, family):
            raise CapacityAnalysisError(f"normalized role/family mismatch at {coordinate.key}")
        if coordinate.run_id != expected_run_id(coordinate.key, family):
            raise CapacityAnalysisError(f"normalized run_id mismatch at {coordinate.key}")
        expected_call_schema = (
            "paper2-v2-work9-capacity-call-prediction-1"
            if family == RUN_FAMILY
            else "paper2-v2-work9-call-prediction-1"
        )
        expected_record_schema = (
            "paper2-v2-work9-capacity-run-record-1"
            if family == RUN_FAMILY
            else "paper2-v2-work9-run-record-1"
        )
        expected_segment_schema = (
            "paper2-v2-work9-capacity-segment-prediction-1"
            if family == RUN_FAMILY
            else "paper2-v2-work9-segment-prediction-1"
        )
        if (
            coordinate.native_segment_schema_version != expected_segment_schema
            or coordinate.native_call_schema_version != expected_call_schema
            or coordinate.native_run_record_schema_version != expected_record_schema
        ):
            raise CapacityAnalysisError(f"native schema family mismatch at {coordinate.key}")
        if not coordinate.calls:
            raise CapacityAnalysisError(f"normalized coordinate has no calls: {coordinate.key}")
        if len({call.call_id for call in coordinate.calls}) != len(coordinate.calls):
            raise CapacityAnalysisError(f"duplicate call in coordinate {coordinate.key}")
        for call in coordinate.calls:
            if (
                call.condition,
                call.partition_repeat,
                call.outer_fold,
                call.seed,
            ) != coordinate.key:
                raise CapacityAnalysisError(f"normalized call coordinate mismatch at {coordinate.key}")
            if call.hard_prediction != int(call.probability >= call.threshold):
                raise CapacityAnalysisError(f"sealed decision mismatch at {coordinate.key}")
        indexed[coordinate.key] = coordinate
        native_schema_counts[
            (family, expected_segment_schema, expected_call_schema, expected_record_schema)
        ] += 1
    if set(indexed) != set(expected):
        raise CapacityAnalysisError("normalized coordinate set is incomplete")

    # Every condition/seed must describe the same calls within each held-out
    # fold.  This is the pairing assumption needed by the frozen cluster tests.
    for fold in FOLDS:
        reference = _semantic_call_inventory(indexed[(CONDITIONS[0], 0, fold, SEEDS[0])].calls)
        for condition in CONDITIONS:
            for seed in SEEDS:
                actual = _semantic_call_inventory(indexed[(condition, 0, fold, seed)].calls)
                if actual != reference:
                    raise CapacityAnalysisError(
                        f"paired call inventory differs in outer fold {fold}: {condition}/{seed}"
                    )
    expected_counts = {
        (
            RUN_FAMILY,
            "paper2-v2-work9-capacity-segment-prediction-1",
            "paper2-v2-work9-capacity-call-prediction-1",
            "paper2-v2-work9-capacity-run-record-1",
        ): 150,
        (
            PRIMARY_RUN_FAMILY,
            "paper2-v2-work9-segment-prediction-1",
            "paper2-v2-work9-call-prediction-1",
            "paper2-v2-work9-run-record-1",
        ): 50,
    }
    if dict(native_schema_counts) != expected_counts:
        raise CapacityAnalysisError("normalized native schema counts differ from 150/50")
    return {
        "expected_coordinate_count": 200,
        "normalized_coordinate_count": 200,
        "new_capacity_coordinate_count": 150,
        "immutable_primary_anchor_coordinate_count": 50,
        "unique_logical_coordinates": True,
        "exact_condition_fold_seed_set": True,
        "paired_call_inventory_equal_within_each_outer_fold": True,
        "native_schema_counts": [
            {
                "run_family": family,
                "segment_schema_version": segment_schema,
                "call_schema_version": call_schema,
                "run_record_schema_version": record_schema,
                "coordinate_count": count,
            }
            for (family, segment_schema, call_schema, record_schema), count in sorted(
                native_schema_counts.items()
            )
        ],
    }


def _artifact_mapping_equal(left: Any, right: Any) -> bool:
    return isinstance(left, Mapping) and isinstance(right, Mapping) and dict(left) == dict(right)


def _coordinate_semantics(
    payload: Mapping[str, Any],
    coordinate: tuple[str, int, int, int],
    *,
    run_id: str,
    run_family: str,
    context: str,
) -> None:
    condition, repeat, fold, seed = coordinate
    expected = {
        "run_id": run_id,
        "protocol_id": PROTOCOL_ID,
        "run_family": run_family,
        "condition": condition,
        "partition_repeat": repeat,
        "outer_fold": fold,
        "seed": seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise CapacityAnalysisError(f"{context}.{key} differs from hybrid index")


def _verify_complete_call_aggregation(
    segment_predictions: Sequence[SegmentPrediction],
    call_rows: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    """Recompute every complete-call row from its full segment inventory."""

    aggregated = {
        call.call_id: call for call in aggregate_mean_probability(segment_predictions)
    }
    if len(aggregated) != len(call_rows):
        raise CapacityAnalysisError(f"{context} segment/call cardinality mismatch")
    seen: set[str] = set()
    for row in call_rows:
        call_id = str(row.get("call_id"))
        if call_id in seen or call_id not in aggregated:
            raise CapacityAnalysisError(f"{context} call inventory mismatch: {call_id}")
        seen.add(call_id)
        recomputed = aggregated[call_id]
        expected = {
            "group_id": recomputed.group_id,
            "label": recomputed.label,
            "source": recomputed.source,
            "n_segments": recomputed.n_segments,
            "start": recomputed.start,
            "end": recomputed.end,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise CapacityAnalysisError(
                f"{context} call metadata does not recompute: {call_id}"
            )
        if not math.isclose(
            float(row["probability"]),
            recomputed.probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise CapacityAnalysisError(
                f"{context} call mean probability does not recompute: {call_id}"
            )
        if not math.isclose(
            float(row["duration_seconds"]),
            recomputed.end - recomputed.start,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CapacityAnalysisError(
                f"{context} call duration does not recompute: {call_id}"
            )
    return aggregated


def load_validated_hybrid_input(
    project_root: str | Path,
    *,
    selection_index_path: str | Path,
    expected_selection_index_sha256: str,
    outer_index_path: str | Path,
    expected_outer_index_sha256: str,
) -> ValidatedHybridInput:
    """Validate both hybrid indexes and normalize their two native families.

    This is the only production entry point that opens capacity outcome files.
    Acceptance must freeze this source and its tests before invoking it.
    """

    root = Path(project_root).resolve()
    policies = load_frozen_capacity_policies(root)
    lexical_paths: list[tuple[Path, str]] = []
    for supplied, context in (
        (selection_index_path, "hybrid selection index"),
        (outer_index_path, "hybrid outer index"),
    ):
        raw = Path(supplied)
        if raw == Path(".") or ".." in raw.parts:
            raise CapacityAnalysisError(f"{context} path is lexically unsafe")
        lexical = raw if raw.is_absolute() else root / raw
        try:
            _reject_symlink_components(lexical, stop=root)
        except ValueError as error:
            raise CapacityAnalysisError(
                f"{context} path contains a symlink component"
            ) from error
        lexical_paths.append((lexical, context))
    selection_path = lexical_paths[0][0].resolve()
    outer_path = lexical_paths[1][0].resolve()
    canonical_matrix = (root / REGISTRY_RELATIVE / "_matrix").resolve()
    if selection_path != canonical_matrix / "selection_index.json":
        raise CapacityAnalysisError("hybrid selection index must use the canonical capacity registry path")
    if outer_path != canonical_matrix / "outer_ledger_index.json":
        raise CapacityAnalysisError("hybrid outer index must use the canonical capacity registry path")
    for path, expected, context in (
        (selection_path, expected_selection_index_sha256, "hybrid selection index"),
        (outer_path, expected_outer_index_sha256, "hybrid outer index"),
    ):
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise CapacityAnalysisError(f"{context} must be a regular project file")
        if sha256_file(path) != expected:
            raise CapacityAnalysisError(f"{context} SHA-256 differs from release")
    selection = _read_json(selection_path, "hybrid selection index")
    outer = _read_json(outer_path, "hybrid outer index")
    _validate_schema(
        selection,
        schema_path=root
        / "revised_experiments/registry/schemas/work9_capacity/work9_selection_index.schema.json",
        context="hybrid selection index",
    )
    _validate_schema(
        outer,
        schema_path=root
        / "revised_experiments/registry/schemas/work9_capacity/work9_outer_ledger_index.schema.json",
        context="hybrid outer index",
    )
    if (
        selection.get("protocol_id") != PROTOCOL_ID
        or outer.get("protocol_id") != PROTOCOL_ID
        or selection.get("run_family") != RUN_FAMILY
        or outer.get("run_family") != RUN_FAMILY
        or selection.get("status") != "frozen_complete"
        or outer.get("status") != "frozen_complete"
    ):
        raise CapacityAnalysisError("hybrid index identity/status mismatch")
    selection_entries = verify_exact_hybrid_entries(
        selection.get("entries"), context="hybrid selection index entries"
    )
    outer_entries = verify_exact_hybrid_entries(
        outer.get("entries"), context="hybrid outer index entries"
    )
    if any(
        selection_entries[key].get("run_id") != outer_entries[key].get("run_id")
        for key in selection_entries
    ):
        raise CapacityAnalysisError("selection and outer indexes bind different run IDs")

    bound_selection = ImmutableReference.load(
        outer.get("selection_index"),
        project_root=root,
        context="outer index selection-index binding",
        ledger=False,
    )
    if bound_selection.path != selection_path or bound_selection.sha256 != expected_selection_index_sha256:
        raise CapacityAnalysisError("outer index does not bind the supplied selection index")
    for container, key, expected_path, expected_hash, context in (
        (selection, "config", WORK8_CONFIG_RELATIVE, WORK8_CONFIG_SHA256, "selection Work-8 config"),
        (selection, "work8_receipt", WORK8_RECEIPT_RELATIVE, WORK8_RECEIPT_SHA256, "selection Work-8 receipt"),
        (selection, "experiment_card", EXPERIMENT_CARD_RELATIVE, EXPERIMENT_CARD_SHA256, "selection experiment card"),
        (selection, "anchor_policy", ANCHOR_POLICY_RELATIVE, ANCHOR_POLICY_SHA256, "selection anchor policy"),
        (selection, "capacity_analysis_protocol", CAPACITY_ANALYSIS_PROTOCOL_RELATIVE, CAPACITY_ANALYSIS_PROTOCOL_SHA256, "selection analysis protocol"),
        (outer, "capacity_analysis_protocol", CAPACITY_ANALYSIS_PROTOCOL_RELATIVE, CAPACITY_ANALYSIS_PROTOCOL_SHA256, "outer analysis protocol"),
        (selection, "figure_spec", FIGURE_SPEC_RELATIVE, FIGURE_SPEC_SHA256, "selection figure spec"),
        (outer, "figure_spec", FIGURE_SPEC_RELATIVE, FIGURE_SPEC_SHA256, "outer figure spec"),
        (selection, "execution_addendum", EXECUTION_ADDENDUM_RELATIVE, EXECUTION_ADDENDUM_SHA256, "selection execution addendum"),
        (outer, "execution_addendum", EXECUTION_ADDENDUM_RELATIVE, EXECUTION_ADDENDUM_SHA256, "outer execution addendum"),
        (selection, "primary_runner_acceptance", PRIMARY_RUNNER_ACCEPTANCE_RELATIVE, PRIMARY_RUNNER_ACCEPTANCE_SHA256, "selection primary runner acceptance"),
        (outer, "primary_runner_acceptance", PRIMARY_RUNNER_ACCEPTANCE_RELATIVE, PRIMARY_RUNNER_ACCEPTANCE_SHA256, "outer primary runner acceptance"),
        (selection, "completed_gate1_hash_manifest", GATE1_HASH_MANIFEST_RELATIVE, GATE1_HASH_MANIFEST_SHA256, "selection completed Gate-1 manifest"),
        (outer, "completed_gate1_hash_manifest", GATE1_HASH_MANIFEST_RELATIVE, GATE1_HASH_MANIFEST_SHA256, "outer completed Gate-1 manifest"),
        (selection, "primary_selection_index", PRIMARY_SELECTION_INDEX_RELATIVE, PRIMARY_SELECTION_INDEX_SHA256, "primary selection index"),
        (outer, "primary_outer_index", PRIMARY_OUTER_INDEX_RELATIVE, PRIMARY_OUTER_INDEX_SHA256, "primary outer index"),
    ):
        reference = ImmutableReference.load(
            container.get(key), project_root=root, context=context, ledger=False
        )
        if reference.sha256 != expected_hash:
            raise CapacityAnalysisError(f"{context} does not match frozen lineage")
        if reference.serialized_path != expected_path.as_posix():
            raise CapacityAnalysisError(f"{context} path does not match frozen lineage")

    selection_acceptance = ImmutableReference.load(
        selection.get("runner_acceptance"),
        project_root=root,
        context="selection capacity runner acceptance",
        ledger=False,
    )
    outer_acceptance = ImmutableReference.load(
        outer.get("runner_acceptance"),
        project_root=root,
        context="outer capacity runner acceptance",
        ledger=False,
    )
    if (
        selection_acceptance.serialized_path != ACCEPTANCE_RECEIPT_RELATIVE.as_posix()
        or outer_acceptance.as_mapping() != selection_acceptance.as_mapping()
    ):
        raise CapacityAnalysisError("hybrid indexes do not bind one canonical runner acceptance")

    expected_parameters = policies.experiment_card.get("parameter_counts")
    if not isinstance(expected_parameters, Mapping):
        raise CapacityAnalysisError("experiment card parameter counts are malformed")
    normalized: list[NormalizedCoordinate] = []
    for coordinate in expected_coordinate_roles():
        entry = outer_entries[coordinate]
        role, family = expected_coordinate_roles()[coordinate]
        run_id = str(entry.get("run_id"))
        call_ref = ImmutableReference.load(
            entry.get("outer_eval_call_ledger"),
            project_root=root,
            context=f"outer call ledger {coordinate}",
            ledger=True,
        )
        # Hash every indexed outer artifact, including the segment ledger and
        # metrics, before reading the analysis subset.
        segment_ref = ImmutableReference.load(
            entry.get("outer_eval_segment_ledger"),
            project_root=root,
            context=f"outer segment ledger {coordinate}",
            ledger=True,
        )
        ImmutableReference.load(
            entry.get("outer_eval_metrics"),
            project_root=root,
            context=f"outer metrics {coordinate}",
            ledger=False,
        )
        record_ref = ImmutableReference.load(
            entry.get("run_record"),
            project_root=root,
            context=f"outer run record {coordinate}",
            ledger=False,
        )
        schema_root = "work9_capacity" if family == RUN_FAMILY else "work9"
        call_schema = root / (
            f"revised_experiments/registry/schemas/{schema_root}/"
            "work9_call_prediction.schema.json"
        )
        segment_schema = root / (
            f"revised_experiments/registry/schemas/{schema_root}/"
            "work9_segment_prediction.schema.json"
        )
        record_schema = root / (
            f"revised_experiments/registry/schemas/{schema_root}/"
            "work9_run_record.schema.json"
        )
        segment_rows = _read_jsonl(
            segment_ref.path, f"native segment ledger {coordinate}"
        )
        call_rows = _read_jsonl(call_ref.path, f"native call ledger {coordinate}")
        record = _read_json(record_ref.path, f"native run record {coordinate}")
        _validate_jsonl_schema(
            segment_rows,
            schema_path=segment_schema,
            context=f"native segment ledger {coordinate}",
        )
        _validate_jsonl_schema(
            call_rows, schema_path=call_schema, context=f"native call ledger {coordinate}"
        )
        _validate_schema(
            record, schema_path=record_schema, context=f"native run record {coordinate}"
        )
        _coordinate_semantics(
            record,
            coordinate,
            run_id=run_id,
            run_family=family,
            context=f"native run record {coordinate}",
        )
        if record.get("phase") != "outer_eval" or record.get("status") != "outer_complete":
            raise CapacityAnalysisError(f"native run record is not outer-complete: {coordinate}")
        artifacts = record.get("artifacts")
        if (
            not isinstance(artifacts, Mapping)
            or not _artifact_mapping_equal(
                artifacts.get("outer_eval_calls"), entry.get("outer_eval_call_ledger")
            )
            or not _artifact_mapping_equal(
                artifacts.get("outer_eval_segments"),
                entry.get("outer_eval_segment_ledger"),
            )
        ):
            raise CapacityAnalysisError(
                f"run record does not bind indexed prediction ledgers: {coordinate}"
            )
        record_selection = record.get("selection")
        if (
            not isinstance(record_selection, Mapping)
            or float(record_selection.get("threshold", -1.0))
            != float(entry.get("threshold", -2.0))
        ):
            raise CapacityAnalysisError(f"run record threshold differs from outer index: {coordinate}")
        segment_predictions: list[SegmentPrediction] = []
        segment_ids: set[str] = set()
        for row_index, row in enumerate(segment_rows):
            _coordinate_semantics(
                row,
                coordinate,
                run_id=run_id,
                run_family=family,
                context=f"native segment row {coordinate}/{row_index}",
            )
            if row.get("partition") != "outer_eval":
                raise CapacityAnalysisError(
                    f"non-outer segment in outer ledger: {coordinate}"
                )
            if (
                float(row.get("threshold")) != float(entry.get("threshold"))
                or row.get("checkpoint_sha256") != entry.get("checkpoint_sha256")
            ):
                raise CapacityAnalysisError(
                    f"segment lineage differs from outer index: {coordinate}"
                )
            segment_id = str(row["segment_id"])
            if segment_id in segment_ids:
                raise CapacityAnalysisError(
                    f"duplicate segment in native ledger: {coordinate}/{segment_id}"
                )
            segment_ids.add(segment_id)
            segment_predictions.append(
                SegmentPrediction(
                    segment_id=segment_id,
                    call_id=str(row["call_id"]),
                    group_id=str(row["group_id"]),
                    label=int(row["label"]),
                    probability=float(row["probability"]),
                    source=str(row["source"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                )
            )
        aggregated = _verify_complete_call_aggregation(
            segment_predictions,
            call_rows,
            context=f"native prediction ledgers {coordinate}",
        )
        calls: list[scientific_core.Gate1Call] = []
        for row_index, row in enumerate(call_rows):
            _coordinate_semantics(
                row,
                coordinate,
                run_id=run_id,
                run_family=family,
                context=f"native call row {coordinate}/{row_index}",
            )
            if row.get("partition") != "outer_eval":
                raise CapacityAnalysisError(f"non-outer call in outer ledger: {coordinate}")
            if (
                float(row.get("threshold")) != float(entry.get("threshold"))
                or row.get("checkpoint_sha256") != entry.get("checkpoint_sha256")
            ):
                raise CapacityAnalysisError(f"call lineage differs from outer index: {coordinate}")
            call_id = str(row["call_id"])
            recomputed = aggregated.get(call_id)
            if recomputed is None:
                raise CapacityAnalysisError(
                    f"call has no complete segment inventory: {coordinate}/{call_id}"
                )
            expected_call_values = {
                "group_id": recomputed.group_id,
                "label": recomputed.label,
                "source": recomputed.source,
                "n_segments": recomputed.n_segments,
                "start": recomputed.start,
                "end": recomputed.end,
            }
            if any(row.get(key) != value for key, value in expected_call_values.items()):
                raise CapacityAnalysisError(
                    f"call metadata does not recompute from segments: {coordinate}/{call_id}"
                )
            if not math.isclose(
                float(row["probability"]),
                recomputed.probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise CapacityAnalysisError(
                    f"call mean probability does not recompute from all segments: {coordinate}/{call_id}"
                )
            if not math.isclose(
                float(row["duration_seconds"]),
                recomputed.end - recomputed.start,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise CapacityAnalysisError(
                    f"call duration does not recompute from segments: {coordinate}/{call_id}"
                )
            calls.append(
                scientific_core.Gate1Call(
                    condition=coordinate[0],
                    partition_repeat=coordinate[1],
                    outer_fold=coordinate[2],
                    seed=coordinate[3],
                    call_id=call_id,
                    group_id=str(row["group_id"]),
                    label=int(row["label"]),
                    probability=float(row["probability"]),
                    hard_prediction=int(row["hard_prediction"]),
                    threshold=float(row["threshold"]),
                    n_segments=int(row["n_segments"]),
                    source=str(row["source"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                )
            )
        model = record.get("model")
        compute = record.get("compute")
        if not isinstance(model, Mapping) or not isinstance(compute, Mapping):
            raise CapacityAnalysisError(f"run model/compute record is malformed: {coordinate}")
        parameters = model.get("trainable_parameters")
        if (
            isinstance(parameters, bool)
            or not isinstance(parameters, int)
            or parameters != int(expected_parameters[coordinate[0]])
        ):
            raise CapacityAnalysisError(f"trainable parameter count changed: {coordinate}")
        try:
            wall = float(compute["wall_time_seconds"])
            allocated = int(compute["peak_cuda_allocated_bytes"])
            reserved = int(compute["peak_cuda_reserved_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise CapacityAnalysisError(f"run compute record is malformed: {coordinate}") from error
        if not math.isfinite(wall) or wall < 0 or allocated < 0 or reserved < 0:
            raise CapacityAnalysisError(f"run compute values are invalid: {coordinate}")
        normalized.append(
            NormalizedCoordinate(
                run_id=run_id,
                condition=coordinate[0],
                partition_repeat=coordinate[1],
                outer_fold=coordinate[2],
                seed=coordinate[3],
                entry_role=role,
                source_run_family=family,
                native_segment_schema_version=str(segment_rows[0]["schema_version"]),
                native_call_schema_version=str(call_rows[0]["schema_version"]),
                native_run_record_schema_version=str(record["schema_version"]),
                calls=tuple(calls),
                trainable_parameters=parameters,
                wall_time_seconds=wall,
                peak_cuda_allocated_bytes=allocated,
                peak_cuda_reserved_bytes=reserved,
            )
        )
    audit = verify_normalized_hybrid_contract(normalized)
    return ValidatedHybridInput(
        coordinates=tuple(normalized),
        selection_index=selection,
        outer_index=outer,
        selection_index_reference={
            "path": selection_path.relative_to(root).as_posix(),
            "sha256": expected_selection_index_sha256,
            "bytes": selection_path.stat().st_size,
            "row_count": None,
        },
        outer_index_reference={
            "path": outer_path.relative_to(root).as_posix(),
            "sha256": expected_outer_index_sha256,
            "bytes": outer_path.stat().st_size,
            "row_count": None,
        },
        normalization_audit=audit,
    )


@dataclass(frozen=True)
class _Fold:
    partition_repeat: int
    outer_fold: int


@dataclass(frozen=True)
class _Statistics:
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    permutation_replicates: int
    permutation_seed: int
    familywise_alpha: float
    bootstrap_method: str
    permutation_method: str
    multiplicity_method: str
    primary_comparisons: tuple[tuple[str, str], ...]
    primary_estimand: str


@dataclass(frozen=True)
class _AnalysisConfig:
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: tuple[_Fold, ...]
    statistics: _Statistics


def _analysis_config(
    policies: FrozenCapacityPolicies,
    *,
    bootstrap_replicates: int | None = None,
    permutation_replicates: int | None = None,
) -> _AnalysisConfig:
    protocol = policies.analysis_protocol
    bootstrap = protocol["bootstrap"]
    permutation = protocol["permutation"]
    multiplicity = protocol["multiplicity"]
    return _AnalysisConfig(
        conditions=CONDITIONS,
        seeds=SEEDS,
        folds=tuple(_Fold(0, fold) for fold in FOLDS),
        statistics=_Statistics(
            bootstrap_replicates=(
                int(bootstrap["replicates"])
                if bootstrap_replicates is None
                else bootstrap_replicates
            ),
            bootstrap_seed=int(bootstrap["seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
            permutation_replicates=(
                int(permutation["replicates"])
                if permutation_replicates is None
                else permutation_replicates
            ),
            permutation_seed=int(permutation["seed"]),
            familywise_alpha=float(multiplicity["familywise_alpha"]),
            bootstrap_method=str(bootstrap["method"]),
            permutation_method=str(permutation["method"]),
            multiplicity_method=str(multiplicity["method"]),
            primary_comparisons=COMPARISONS,
            primary_estimand=str(protocol["estimand"]),
        ),
    )


def load_completed_gate1_baseline(
    project_root: str | Path, policies: FrozenCapacityPolicies
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Read only the sealed descriptive primary baseline; make no decision."""

    root = Path(project_root).resolve()
    manifest_path = _verify_frozen_file(
        root, GATE1_HASH_MANIFEST_RELATIVE, GATE1_HASH_MANIFEST_SHA256
    )
    report_path = _verify_frozen_file(
        root, COMPLETED_GATE1_REPORT_RELATIVE, COMPLETED_GATE1_REPORT_SHA256
    )
    manifest_rows = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise CapacityAnalysisError("completed Gate-1 hash manifest is malformed")
        manifest_rows[parts[1].strip()] = parts[0]
    if manifest_rows.get("report.json") != COMPLETED_GATE1_REPORT_SHA256:
        raise CapacityAnalysisError("completed Gate-1 manifest does not bind report.json")
    report = _read_json(report_path, "completed Gate-1 report")
    summaries = report.get("condition_summaries")
    if not isinstance(summaries, Mapping) or set(summaries) != set(CONDITIONS):
        raise CapacityAnalysisError("completed Gate-1 condition baseline is incomplete")
    # The only decision fact copied forward is the already completed outcome.
    frozen_completed = policies.analysis_protocol["completed_gate1"]
    baseline_lineage = {
        "hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
        "report_sha256": COMPLETED_GATE1_REPORT_SHA256,
        "gate2_decision": str(frozen_completed["gate2_decision"]),
        "may_be_reopened": False,
        "role": "sealed_descriptive_primary_baseline_only",
    }
    return summaries, baseline_lineage


def _metric_mean(summary: Mapping[str, Any], metric: str) -> float | None:
    metrics = summary.get("metrics_across_seeds")
    row = metrics.get(metric) if isinstance(metrics, Mapping) else None
    value = row.get("mean") if isinstance(row, Mapping) else None
    return None if value is None else float(value)


def _summarize_compute(coordinates: Sequence[NormalizedCoordinate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        selected = [row for row in coordinates if row.condition == condition]
        if len(selected) != 25:
            raise CapacityAnalysisError(f"compute profile lacks 25 rows for {condition}")
        parameters = {row.trainable_parameters for row in selected}
        if len(parameters) != 1:
            raise CapacityAnalysisError(f"parameter count varies within {condition}")
        wall = np.asarray([row.wall_time_seconds for row in selected], dtype=float)
        allocated = np.asarray(
            [row.peak_cuda_allocated_bytes for row in selected], dtype=float
        )
        reserved = np.asarray(
            [row.peak_cuda_reserved_bytes for row in selected], dtype=float
        )
        rows.append(
            {
                "condition": condition,
                "coordinate_count": 25,
                "trainable_parameters": next(iter(parameters)),
                "wall_time_seconds": {
                    "mean": float(wall.mean()),
                    "minimum": float(wall.min()),
                    "maximum": float(wall.max()),
                },
                "peak_cuda_allocated_bytes": {
                    "mean": float(allocated.mean()),
                    "minimum": int(allocated.min()),
                    "maximum": int(allocated.max()),
                },
                "peak_cuda_reserved_bytes": {
                    "mean": float(reserved.mean()),
                    "minimum": int(reserved.min()),
                    "maximum": int(reserved.max()),
                },
                "scope_exclusions": [
                    "asr",
                    "encoder_extraction",
                    "mobile",
                    "real_time_deployment",
                ],
            }
        )
    return rows


def verify_report_claim_boundary(report: Mapping[str, Any]) -> None:
    """Reject a report that relabels this sensitivity as a new gate."""

    if report.get("evidence_label") != EVIDENCE_LABEL:
        raise CapacityAnalysisError("capacity evidence label changed")
    boundary = report.get("completed_gate1")
    if boundary != {
        "hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
        "report_sha256": COMPLETED_GATE1_REPORT_SHA256,
        "gate2_decision": "do_not_proceed",
        "may_be_reopened": False,
        "role": "sealed_descriptive_primary_baseline_only",
    }:
        raise CapacityAnalysisError("report attempts to reopen or relabel Gate 1/Gate 2")
    comparisons = report.get("secondary_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise CapacityAnalysisError("report must contain exactly three comparisons")
    actual = tuple(
        (row.get("primary_condition"), row.get("comparator_condition"))
        for row in comparisons
        if isinstance(row, Mapping)
    )
    if actual != COMPARISONS:
        raise CapacityAnalysisError("report comparison family differs from frozen three")
    forbidden_keys = {
        "new_gate1_decision",
        "gate1_pass_fail",
        "gate2_reopening",
        "continuation_decision",
        "deployment_claim",
        "external_generalization_claim",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys.intersection(value):
                raise CapacityAnalysisError("report contains a forbidden gate/claim field")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report)


def analyze_validated_hybrid(
    validated: ValidatedHybridInput,
    *,
    policies: FrozenCapacityPolicies,
    primary_condition_summaries: Mapping[str, Any],
    primary_baseline_lineage: Mapping[str, Any],
    synthetic_test_mode: bool = False,
    bootstrap_replicates: int | None = None,
    permutation_replicates: int | None = None,
) -> dict[str, Any]:
    """Run the frozen secondary analysis on validated normalized coordinates."""

    if (bootstrap_replicates is not None or permutation_replicates is not None) and not synthetic_test_mode:
        raise CapacityAnalysisError("replicate overrides are restricted to synthetic tests")
    if synthetic_test_mode:
        for value in (bootstrap_replicates, permutation_replicates):
            if value is not None and value <= 0:
                raise CapacityAnalysisError("synthetic replicate overrides must be positive")
    audit = verify_normalized_hybrid_contract(validated.coordinates)
    config = _analysis_config(
        policies,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )
    calls = validated.calls_by_coordinate
    condition_records, condition_summaries, metric_lookup = (
        scientific_core.build_condition_metrics(calls, config)  # type: ignore[arg-type]
    )
    for row in condition_records:
        row["schema_version"] = CONDITION_METRIC_SCHEMA_VERSION
        row["analysis_role"] = "capacity_matched_secondary_sensitivity"
    comparisons = scientific_core.analyze_primary_comparisons(  # type: ignore[arg-type]
        calls_by_coordinate=calls,
        config=config,
        metric_lookup=metric_lookup,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )
    if tuple(
        (row["primary_condition"], row["comparator_condition"])
        for row in comparisons
    ) != COMPARISONS:
        raise CapacityAnalysisError("scientific core emitted a non-frozen comparison")
    for row in comparisons:
        row["schema_version"] = COMPARISON_SCHEMA_VERSION
        row["analysis_role"] = "secondary_capacity_sensitivity_not_gate1"
        row["can_change_completed_gate1"] = False
        row["can_reopen_gate2"] = False

    if set(primary_condition_summaries) != set(CONDITIONS):
        raise CapacityAnalysisError("primary descriptive baseline must cover all conditions")
    changes: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        current = condition_summaries[condition]
        baseline = primary_condition_summaries[condition]
        metric_changes: dict[str, Any] = {}
        for metric in REPORTED_CALL_METRICS:
            current_mean = _metric_mean(current, metric)
            primary_mean = _metric_mean(baseline, metric)
            metric_changes[metric] = {
                "primary_h256_mean": primary_mean,
                "capacity_family_mean": current_mean,
                "capacity_minus_primary": (
                    None
                    if current_mean is None or primary_mean is None
                    else current_mean - primary_mean
                ),
            }
            if (
                not synthetic_test_mode
                and condition in ANCHOR_CONDITIONS
                and current_mean is not None
                and primary_mean is not None
                and not math.isclose(
                    current_mean, primary_mean, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise CapacityAnalysisError(
                    f"immutable primary anchor differs from sealed Gate-1 baseline: {condition}/{metric}"
                )
        changes.append(
            {
                "condition": condition,
                "role": "descriptive_only_no_formal_hypothesis_test",
                "metrics": metric_changes,
            }
        )

    parameter_counts = policies.experiment_card["parameter_counts"]
    hidden_dims = policies.experiment_card["hidden_dims"]
    target = int(policies.experiment_card["capacity_contract"]["target_parameter_count"])
    parameter_profile = [
        {
            "condition": condition,
            "hidden_dim": int(hidden_dims[condition]),
            "trainable_parameters": int(parameter_counts[condition]),
            "target_parameters": target,
            "absolute_gap": int(parameter_counts[condition]) - target,
            "relative_absolute_gap": abs(int(parameter_counts[condition]) - target) / target,
        }
        for condition in CONDITIONS
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "status": (
            "synthetic_contract_test" if synthetic_test_mode else "secondary_sensitivity_complete"
        ),
        "protocol_id": PROTOCOL_ID,
        "run_family": RUN_FAMILY,
        "analysis_role": "preregistered_secondary_parameter_budget_sensitivity_not_a_second_primary_test",
        "evidence_label": EVIDENCE_LABEL,
        "completed_gate1": dict(primary_baseline_lineage),
        "lineage": {
            "selection_index": dict(validated.selection_index_reference),
            "outer_index": dict(validated.outer_index_reference),
            "analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "figure_spec_sha256": FIGURE_SPEC_SHA256,
            "experiment_card_sha256": EXPERIMENT_CARD_SHA256,
            "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
        },
        "normalization_audit": dict(audit),
        "inference": {
            "role": "secondary_sensitivity_inference",
            "estimand": policies.analysis_protocol["estimand"],
            "comparison_family": [list(pair) for pair in COMPARISONS],
            "bootstrap": dict(policies.analysis_protocol["bootstrap"]),
            "permutation": dict(policies.analysis_protocol["permutation"]),
            "multiplicity": dict(policies.analysis_protocol["multiplicity"]),
            "synthetic_replicate_override": (
                {
                    "bootstrap": config.statistics.bootstrap_replicates,
                    "permutation": config.statistics.permutation_replicates,
                }
                if synthetic_test_mode
                else None
            ),
        },
        "condition_metrics": condition_records,
        "condition_summaries": condition_summaries,
        "secondary_comparisons": comparisons,
        "descriptive_primary_vs_capacity": changes,
        "parameter_profile": parameter_profile,
        "compute_profile": _summarize_compute(validated.coordinates),
        "claim_boundary": {
            "can_change_completed_gate1_decision": False,
            "can_reopen_gate2": False,
            "external_generalization_authorized": False,
            "practical_utility_authorized": False,
            "deployment_or_mobile_claim_authorized": False,
            "source_scope": "within_corpus_source_confounded",
        },
    }
    verify_report_claim_boundary(report)
    return report


def _validate_analysis_report_schema(report: Mapping[str, Any], project_root: Path) -> None:
    _validate_schema(
        report,
        schema_path=project_root / ANALYSIS_REPORT_SCHEMA_RELATIVE,
        context="capacity analysis report",
    )
    verify_report_claim_boundary(report)


def write_analysis_bundle_once(
    project_root: str | Path,
    *,
    output_directory: str | Path,
    report: Mapping[str, Any],
    synthetic_test_mode: bool = False,
) -> Path:
    """Publish one immutable analysis bundle without modifying prior evidence."""

    root = Path(project_root).resolve()
    raw_output = Path(output_directory)
    expected_status = (
        "synthetic_contract_test" if synthetic_test_mode else "secondary_sensitivity_complete"
    )
    if report.get("status") != expected_status:
        raise CapacityAnalysisError("analysis output mode differs from report status")
    try:
        if synthetic_test_mode:
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=TEST_SANDBOX_ROOT_RELATIVE,
                context="synthetic capacity analysis output",
                require_new=True,
            )
        else:
            if raw_output != ANALYSIS_OUTPUT_RELATIVE:
                raise CapacityAnalysisError(
                    f"production analysis output is fixed at {ANALYSIS_OUTPUT_RELATIVE}"
                )
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=Path("revised_experiments/artifacts"),
                context="production capacity analysis output",
                require_new=True,
            )
    except (ValueError, FileExistsError) as error:
        raise CapacityAnalysisError(str(error)) from error
    _validate_analysis_report_schema(report, root)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        files = {
            "report.json": _canonical_json_bytes(report),
            "condition_metrics.jsonl": _canonical_jsonl_bytes(report["condition_metrics"]),
            "secondary_comparisons.jsonl": _canonical_jsonl_bytes(
                report["secondary_comparisons"]
            ),
            "normalization_audit.json": _canonical_json_bytes(
                report["normalization_audit"]
            ),
        }
        for name, payload in files.items():
            (staging / name).write_bytes(payload)
        hashes = "".join(
            f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
            for name, payload in sorted(files.items())
        )
        (staging / "hashes.sha256").write_text(hashes, encoding="utf-8")
        (staging / "READ_ONLY_NOTICE.txt").write_text(
            "Secondary capacity-matched within-corpus sensitivity. This bundle cannot reopen Gate 1 or Gate 2.\n",
            encoding="utf-8",
        )
        _freeze_tree(staging)
        os.rename(staging, destination)
    except BaseException:
        _restore_tree_for_cleanup(staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "report.json"


def run_capacity_analysis(
    project_root: str | Path,
    *,
    selection_index_path: str | Path,
    expected_selection_index_sha256: str,
    outer_index_path: str | Path,
    expected_outer_index_sha256: str,
    output_directory: str | Path,
) -> Path:
    """Production orchestration with no test overrides and no gate decision."""

    root = Path(project_root).resolve()
    policies = load_frozen_capacity_policies(root)
    validated = load_validated_hybrid_input(
        root,
        selection_index_path=selection_index_path,
        expected_selection_index_sha256=expected_selection_index_sha256,
        outer_index_path=outer_index_path,
        expected_outer_index_sha256=expected_outer_index_sha256,
    )
    baseline, lineage = load_completed_gate1_baseline(root, policies)
    report = analyze_validated_hybrid(
        validated,
        policies=policies,
        primary_condition_summaries=baseline,
        primary_baseline_lineage=lineage,
    )
    return write_analysis_bundle_once(
        root, output_directory=output_directory, report=report,
        synthetic_test_mode=False,
    )


__all__ = [
    "ANALYZER_VERSION",
    "COMPARISONS",
    "CONDITIONS",
    "CapacityAnalysisError",
    "EVIDENCE_LABEL",
    "FIGURE_IDS",
    "NormalizedCoordinate",
    "REPORT_SCHEMA_VERSION",
    "SEEDS",
    "ValidatedHybridInput",
    "analyze_validated_hybrid",
    "expected_coordinate_roles",
    "load_completed_gate1_baseline",
    "load_frozen_capacity_policies",
    "load_validated_hybrid_input",
    "run_capacity_analysis",
    "verify_exact_hybrid_entries",
    "verify_normalized_hybrid_contract",
    "verify_report_claim_boundary",
    "write_analysis_bundle_once",
]
