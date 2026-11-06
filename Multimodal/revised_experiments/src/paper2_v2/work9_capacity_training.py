"""Deterministic two-phase coordinator for matched-capacity sensitivity.

The production entry point must establish ``PYTHONHASHSEED`` and, on CUDA,
``CUBLAS_WORKSPACE_CONFIG`` before importing this module.  This module never
opens an outer-evaluation manifest during fit/selection.
"""

from __future__ import annotations

import io
import json
import math
import os
import platform
import random
import resource
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .conditions import ConditionId, build_condition
from .config import (
    ArtifactRef,
    MatrixConfig,
    ProtocolValidationError,
    SelectionSeal,
    load_matrix_config,
    sha256_file,
)
from .data import collate_samples
from .evaluation import (
    CallPrediction,
    SegmentPrediction,
    aggregate_mean_probability,
    call_level_nll,
    compute_call_metrics,
    select_validation_threshold,
    threshold_grid,
)
from .pairing import PairingPlan, pairing_plan_for_split
from .text_features import IndexedTextStore
from .work9_data import CACHE_POLICY, Work9FeatureDataset
from .work9_capacity_registry import (
    CapacityRegistry,
    CapacityRunWorkspace,
    _reject_symlink_components,
    validate_isolated_relative_root,
)
from .work9_capacity_types import (
    ACCEPTANCE_RECEIPT_RELATIVE,
    ACCEPTANCE_SCHEMA_VERSION,
    ANCHOR_CONDITIONS,
    CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
    CUDA_SMOKE_SEMANTIC_SCHEMA_VERSION,
    DRY_RECEIPT_RELATIVE,
    EXECUTION_ADDENDUM_RELATIVE,
    EXECUTION_ADDENDUM_SHA256,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_CUDA_VERSION,
    EXPECTED_CUDNN_VERSION,
    EXPECTED_DEVICE_NAME,
    EXPECTED_TORCH_VERSION,
    EXPERIMENT_CARD_RELATIVE,
    EXPERIMENT_CARD_SHA256,
    FIGURE_SPEC_RELATIVE,
    FIGURE_SPEC_SHA256,
    GATE1_HASH_MANIFEST_RELATIVE,
    GATE1_HASH_MANIFEST_SHA256,
    NEW_FIT_CONDITIONS,
    PRIMARY_OUTER_INDEX_RELATIVE,
    PRIMARY_OUTER_INDEX_SHA256,
    PRIMARY_ENVIRONMENT_LOCK_SHA256,
    PRIMARY_RUNNER_ACCEPTANCE_RELATIVE,
    PRIMARY_RUNNER_ACCEPTANCE_SHA256,
    PRIMARY_SELECTION_INDEX_RELATIVE,
    PRIMARY_SELECTION_INDEX_SHA256,
    REGISTRY_RELATIVE,
    CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    ANCHOR_POLICY_RELATIVE,
    ANCHOR_POLICY_SHA256,
    RUN_FAMILY,
    SMOKE_EVIDENCE_ROOT_RELATIVE,
    WORK8_CONFIG_RELATIVE,
    WORK8_CONFIG_SHA256,
    WORK8_RECEIPT_RELATIVE,
    WORK8_RECEIPT_SHA256,
    CapacityCoordinates,
    CapacitySelectionSeal,
    artifact_record,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_selection_seal,
    sha256_bytes,
)


SCHEMA_ROOT_RELATIVE = Path("revised_experiments/registry/schemas/work9_capacity")
SEGMENT_SCORE_SCHEMA = "work9_validation_segment_score.schema.json"
CALL_SCORE_SCHEMA = "work9_validation_call_score.schema.json"
SEGMENT_PREDICTION_SCHEMA = "work9_segment_prediction.schema.json"
CALL_PREDICTION_SCHEMA = "work9_call_prediction.schema.json"
EPOCH_TRACE_SCHEMA = "work9_epoch_trace.schema.json"
THRESHOLD_TRACE_SCHEMA = "work9_threshold_trace.schema.json"
SELECTION_SEAL_SCHEMA = "work9_selection_seal.schema.json"
RUN_RECORD_SCHEMA = "work9_run_record.schema.json"
SELECTION_ARTIFACT_INDEX_SCHEMA = "work9_selection_artifact_index.schema.json"
RUN_ARTIFACT_INDEX_SCHEMA = "work9_run_artifact_index.schema.json"

OPTIMIZER_POLICY = {
    "name": "AdamW",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "amsgrad": False,
    "maximize": False,
    "foreach": False,
    "capturable": False,
    "differentiable": False,
    "fused": False,
}
SCHEDULER_POLICY = {
    "name": "ReduceLROnPlateau",
    "mode": "min",
    "threshold": 1e-4,
    "threshold_mode": "rel",
    "cooldown": 0,
    "eps": 1e-8,
}
DETERMINISM_POLICY = {
    "mixed_precision": False,
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "allow_tf32": False,
    "float32_matmul_precision": "highest",
    "dataloader_workers": 0,
}


class CapacityTrainingError(RuntimeError):
    """Fail-closed coordinator error."""


class CapacityConfigView:
    """Read-only Work 8 view that changes only the frozen per-condition width."""

    def __init__(self, base: MatrixConfig) -> None:
        self._base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def instantiate_condition(self, condition: str) -> Any:
        if condition not in NEW_FIT_CONDITIONS:
            raise CapacityTrainingError(
                "anchor conditions are immutable references and cannot be newly fit"
            )
        return self._instantiate_capacity_width(condition)

    def instantiate_smoke_condition(self, condition: str) -> Any:
        if condition not in NEW_FIT_CONDITIONS + ANCHOR_CONDITIONS:
            raise CapacityTrainingError("smoke condition is outside frozen capacity matrix")
        return self._instantiate_capacity_width(condition)

    def _instantiate_capacity_width(self, condition: str) -> Any:
        hidden_dim = self._base.capacity_sensitivity.matched_hidden_dims[condition]
        built = build_condition(
            condition,
            hidden_dim=hidden_dim,
            num_heads=self._base.model.num_heads,
            dropout=self._base.model.dropout,
            text_dim=self._base.model.text_dim,
            text_path_key="__indexed_text_store__",
        )
        expected = self._base.capacity_sensitivity.matched_parameter_counts[condition]
        if built.total_parameters != expected or built.trainable_parameters != expected:
            raise CapacityTrainingError(
                f"capacity factory drift for {condition}: expected {expected}, "
                f"got total={built.total_parameters}, trainable={built.trainable_parameters}"
            )
        return built


def _schema_path(project_root: Path, name: str) -> Path:
    return project_root / SCHEMA_ROOT_RELATIVE / name


@lru_cache(maxsize=None)
def _compiled_schema_validator(project_root_text: str, schema_name: str) -> Any:
    try:
        import jsonschema
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("jsonschema is required for Work 9") from error
    project_root = Path(project_root_text)
    schema = json.loads(
        _schema_path(project_root, schema_name).read_text(encoding="utf-8")
    )
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )


def validate_json_schema(
    payload: Mapping[str, Any], *, project_root: Path, schema_name: str
) -> None:
    validator = _compiled_schema_validator(str(project_root.resolve()), schema_name)
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        messages = [f"{list(error.path)}: {error.message}" for error in errors[:10]]
        raise CapacityTrainingError(
            f"{schema_name} validation failed: {'; '.join(messages)}"
        )


def validate_jsonl_schema(
    rows: Sequence[Mapping[str, Any]], *, project_root: Path, schema_name: str
) -> None:
    if not rows:
        raise CapacityTrainingError(f"{schema_name} ledger is empty")
    validator = _compiled_schema_validator(str(project_root.resolve()), schema_name)
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
        if errors:
            messages = [f"{list(error.path)}: {error.message}" for error in errors[:10]]
            raise CapacityTrainingError(
                f"ledger row {index} fails {schema_name}: {'; '.join(messages)}"
            )


@dataclass(frozen=True)
class FrozenWork9Inputs:
    project_root: Path
    config: CapacityConfigView
    work8_receipt: Mapping[str, Any]
    runner_acceptance: Mapping[str, Any] | None
    runner_acceptance_sha256: str | None
    experiment_card_sha256: str


def _verify_exact_file(path: Path, expected_sha256: str, context: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{context} missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise CapacityTrainingError(
            f"{context} hash mismatch: expected {expected_sha256}, got {actual}"
        )


def _verify_ref_mapping(
    raw: Mapping[str, Any], project_root: Path, *, context: str
) -> Path:
    if not {"path", "sha256"} <= set(raw):
        raise CapacityTrainingError(f"{context} is missing path/sha256")
    allowed_shapes = (
        {"path", "sha256"},
        {"path", "sha256", "bytes"},
        {"path", "sha256", "bytes", "row_count"},
    )
    if set(raw) not in allowed_shapes:
        raise CapacityTrainingError(f"{context} has unexpected reference keys")
    raw_path = Path(str(raw["path"]))
    if raw_path.is_absolute() or raw_path == Path(".") or ".." in raw_path.parts:
        raise CapacityTrainingError(f"{context} path must be safe and project-relative")
    root = project_root.resolve()
    lexical_path = root / raw_path
    try:
        _reject_symlink_components(lexical_path, stop=root)
    except ValueError as error:
        raise CapacityTrainingError(f"{context} path contains a symlink") from error
    path = lexical_path.resolve()
    if root not in path.parents:
        raise CapacityTrainingError(f"{context} path escaped project root")
    _verify_exact_file(path, str(raw["sha256"]), context)
    if "bytes" in raw and int(raw["bytes"]) != path.stat().st_size:
        raise CapacityTrainingError(f"{context} byte count mismatch")
    return path


def verify_primary_selection_anchor_lineage(
    project_root: str | Path,
) -> dict[str, Any]:
    """Phase-A-safe verification of only primary selection anchor artifacts."""

    root = Path(project_root).resolve()
    selection_path = root / PRIMARY_SELECTION_INDEX_RELATIVE
    gate1_hash_path = root / GATE1_HASH_MANIFEST_RELATIVE
    work8_receipt_path = root / WORK8_RECEIPT_RELATIVE
    _verify_exact_file(
        selection_path, PRIMARY_SELECTION_INDEX_SHA256, "primary selection index"
    )
    _verify_exact_file(
        gate1_hash_path, GATE1_HASH_MANIFEST_SHA256, "completed Gate 1 hash manifest"
    )
    _verify_exact_file(
        work8_receipt_path, WORK8_RECEIPT_SHA256, "frozen Work 8 receipt"
    )
    for path in (selection_path, gate1_hash_path):
        if path.stat().st_mode & 0o222:
            raise CapacityTrainingError(f"immutable primary anchor is writable: {path}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "frozen_complete"
        or selection.get("run_family") != "primary_matrix"
        or selection.get("expected_coordinate_count") != 200
        or len(selection.get("entries", [])) != 200
    ):
        raise CapacityTrainingError("primary selection index is not frozen-complete")
    selection_keys = {
        (str(row["condition"]), int(row["partition_repeat"]), int(row["outer_fold"]), int(row["seed"]))
        for row in selection["entries"]
    }
    if len(selection_keys) != 200:
        raise CapacityTrainingError("primary selection coordinate set is not unique")
    expected_anchor_keys = {
        (condition, 0, fold, seed)
        for condition in ANCHOR_CONDITIONS
        for fold in range(5)
        for seed in (17, 29, 43, 71, 101)
    }
    if not expected_anchor_keys <= selection_keys or len(expected_anchor_keys) != 50:
        raise CapacityTrainingError("primary selection index lacks the 50 anchors")

    selected_by_key = {
        (str(row["condition"]), int(row["partition_repeat"]), int(row["outer_fold"]), int(row["seed"])): row
        for row in selection["entries"]
        if str(row["condition"]) in ANCHOR_CONDITIONS
    }
    work8_receipt = json.loads(work8_receipt_path.read_text(encoding="utf-8"))
    planned = {
        (int(row["outer_fold"]), int(row["seed"])): row
        for row in work8_receipt.get("random_train_pairing_plans", [])
    }
    if len(planned) != 25:
        raise CapacityTrainingError("Work 8 does not contain the exact 25 pairing plans")
    verified_pairing_plans = 0
    for key in sorted(expected_anchor_keys):
        selected = selected_by_key[key]
        for field in (
            "selection_seal",
            "selection_run_record",
            "epoch_trace",
            "threshold_trace",
        ):
            _verify_ref_mapping(
                selected[field], root, context=f"primary anchor {key} {field}"
            )
        seal_path = root / str(selected["selection_seal"]["path"])
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        record_path = root / str(selected["selection_run_record"]["path"])
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            seal.get("run_id") != selected.get("run_id")
            or seal.get("condition") != key[0]
            or seal.get("partition_repeat") != key[1]
            or seal.get("outer_fold") != key[2]
            or seal.get("seed") != key[3]
            or record.get("run_id") != selected.get("run_id")
            or record.get("status") != "selection_complete"
            or record.get("model", {}).get("total_parameters") != 530689
            or record.get("model", {}).get("trainable_parameters") != 530689
        ):
            raise CapacityTrainingError(f"primary anchor semantic lineage differs at {key}")
        if key[0] == "AT-cross":
            if (
                seal.get("pairing_plan_sha256") is not None
                or seal.get("pairing_plan_artifact") is not None
            ):
                raise CapacityTrainingError("correct-pair anchor unexpectedly binds random plan")
            continue
        plan_record = planned[(key[2], key[3])]
        plan_ref = seal.get("pairing_plan_artifact")
        if not isinstance(plan_ref, Mapping):
            raise CapacityTrainingError(f"random-pair anchor lacks plan artifact at {key}")
        plan_path = _verify_ref_mapping(
            plan_ref, root, context=f"primary random-pair anchor {key} pairing plan"
        )
        with plan_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if (
            len(rows) != int(plan_ref.get("row_count", -1))
            or len(rows) != int(plan_record.get("receiver_count", -1))
            or seal.get("train_manifest_sha256")
            != plan_record.get("train_manifest_sha256")
        ):
            raise CapacityTrainingError(f"random-pair plan inventory differs at {key}")
        receivers = [str(row.get("receiver_segment_id", "")) for row in rows]
        donors = [str(row.get("donor_segment_id", "")) for row in rows]
        if (
            len(set(receivers)) != len(rows)
            or set(receivers) != set(donors)
            or any(
                int(row.get("seed", -1)) != key[3]
                or row.get("split_role") != "train"
                or row.get("receiver_segment_id") == row.get("donor_segment_id")
                or row.get("receiver_call_id") == row.get("donor_call_id")
                or row.get("receiver_group_id") == row.get("donor_group_id")
                for row in rows
            )
        ):
            raise CapacityTrainingError(f"random-pair plan semantics differ at {key}")
        semantic_payload = {
            "seed": key[3],
            "split_role": "train",
            "pairs": [
                {
                    "receiver_segment_id": receiver,
                    "donor_segment_id": donor,
                }
                for receiver, donor in sorted(zip(receivers, donors, strict=True))
            ],
        }
        semantic_sha = sha256_bytes(
            json.dumps(
                semantic_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if (
            semantic_sha != seal.get("pairing_plan_sha256")
            or semantic_sha != plan_record.get("plan_sha256")
        ):
            raise CapacityTrainingError(f"random-pair semantic SHA differs at {key}")
        verified_pairing_plans += 1
    return {
        "status": "pass",
        "phase": "selection_only",
        "anchor_conditions": list(ANCHOR_CONDITIONS),
        "anchor_coordinate_count": 50,
        "primary_selection_index_sha256": PRIMARY_SELECTION_INDEX_SHA256,
        "gate1_hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
        "primary_outer_index_opened": False,
        "primary_outer_artifacts_opened": False,
        "random_pairing_plans_semantically_verified": verified_pairing_plans,
    }


def _verify_runner_acceptance_environment(acceptance: Mapping[str, Any]) -> str:
    """Recompute and bind the accepted, smoke, and unit-test environment."""

    from .work9_capacity_acceptance import (
        environment_lock,
        verify_environment_lock_mapping,
    )

    embedded = acceptance.get("environment_lock")
    try:
        embedded_sha = verify_environment_lock_mapping(embedded)
    except Exception as error:
        raise CapacityTrainingError("runner acceptance environment lock is invalid") from error
    current = environment_lock()
    if embedded != current:
        raise CapacityTrainingError("current environment differs from runner acceptance")
    smokes = acceptance.get("cuda_smokes")
    if not isinstance(smokes, list) or len(smokes) != 2 or any(
        not isinstance(item, Mapping)
        or item.get("environment_lock_sha256") != embedded_sha
        for item in smokes
    ):
        raise CapacityTrainingError("CUDA smokes do not bind accepted environment")
    expected_device = {
        "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "torch_version": EXPECTED_TORCH_VERSION,
        "cuda_version": EXPECTED_CUDA_VERSION,
        "cudnn_version": EXPECTED_CUDNN_VERSION,
    }
    for index, smoke in enumerate(smokes):
        if any(smoke.get(key) != value for key, value in expected_device.items()):
            raise CapacityTrainingError(
                f"CUDA smoke {index} differs from primary-anchor device"
            )
    unit_tests = acceptance.get("unit_tests")
    if not isinstance(unit_tests, Mapping) or unit_tests.get(
        "environment_lock_sha256"
    ) != embedded_sha:
        raise CapacityTrainingError("unit tests do not bind accepted environment")
    return embedded_sha


def verify_frozen_work9_inputs(
    project_root: str | Path,
    *,
    require_runner_acceptance: bool,
) -> FrozenWork9Inputs:
    """Verify Work 8, the experiment card, and the Work 9 source freeze.

    ``require_runner_acceptance=False`` is permitted only for validation and
    smoke.  Production fit selection always passes ``True``.
    """

    root = Path(project_root).resolve()
    config_path = root / WORK8_CONFIG_RELATIVE
    receipt_path = root / WORK8_RECEIPT_RELATIVE
    anchor_path = root / ANCHOR_POLICY_RELATIVE
    analysis_path = root / CAPACITY_ANALYSIS_PROTOCOL_RELATIVE
    card_path = root / EXPERIMENT_CARD_RELATIVE
    figure_path = root / FIGURE_SPEC_RELATIVE
    addendum_path = root / EXECUTION_ADDENDUM_RELATIVE
    primary_acceptance_path = root / PRIMARY_RUNNER_ACCEPTANCE_RELATIVE
    _verify_exact_file(config_path, WORK8_CONFIG_SHA256, "frozen Work 8 config")
    _verify_exact_file(receipt_path, WORK8_RECEIPT_SHA256, "frozen Work 8 receipt")
    _verify_exact_file(anchor_path, ANCHOR_POLICY_SHA256, "capacity anchor policy")
    _verify_exact_file(
        analysis_path,
        CAPACITY_ANALYSIS_PROTOCOL_SHA256,
        "capacity analysis protocol",
    )
    _verify_exact_file(card_path, EXPERIMENT_CARD_SHA256, "capacity experiment card")
    _verify_exact_file(figure_path, FIGURE_SPEC_SHA256, "capacity figure spec")
    _verify_exact_file(
        addendum_path, EXECUTION_ADDENDUM_SHA256, "capacity execution addendum"
    )
    _verify_exact_file(
        primary_acceptance_path,
        PRIMARY_RUNNER_ACCEPTANCE_SHA256,
        "primary runner acceptance",
    )
    for path in (
        anchor_path,
        analysis_path,
        card_path,
        figure_path,
        addendum_path,
        primary_acceptance_path,
    ):
        if path.stat().st_mode & 0o222:
            raise CapacityTrainingError(f"frozen capacity policy is writable: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass" or receipt.get("safety", {}).get(
        "training_launched"
    ) is not False:
        raise CapacityTrainingError("Work 8 receipt is not the passing no-training receipt")
    for index, source in enumerate(receipt.get("source_contracts", [])):
        path = _verify_ref_mapping(source, root, context=f"Work 8 source {index}")
        if int(source.get("bytes", -1)) != path.stat().st_size:
            raise CapacityTrainingError("Work 8 source byte count mismatch")

    # Parsing is intentionally outer-blind. Work 8 already sealed and hashed
    # the full tree; Work 9 Phase A re-verifies only non-outer inputs here.
    base_config = load_matrix_config(
        config_path,
        project_root=root,
        verify_files=False,
        verify_manifest_isolation=False,
    )
    for artifact in (
        base_config.gate0_receipt,
        base_config.exclusion_inventory,
        base_config.fold_protocol_artifact,
        base_config.fold_assignments_artifact,
        base_config.fold_checksums_artifact,
    ):
        artifact.verify()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("status") != "frozen_before_capacity_model_outcomes":
        raise CapacityTrainingError("capacity experiment card is not frozen")
    if card.get("protocol_id") != base_config.protocol_id or card.get("run_family") != RUN_FAMILY:
        raise CapacityTrainingError("experiment card coordinates differ from Work 8")
    if card.get("source_label_limitation", "").find("AIHub") < 0:
        raise CapacityTrainingError("experiment card source limitation is malformed")
    if card.get("work8_config", {}).get("sha256") != WORK8_CONFIG_SHA256:
        raise CapacityTrainingError("experiment card does not bind the Work 8 config")
    if card.get("work8_receipt", {}).get("sha256") != WORK8_RECEIPT_SHA256:
        raise CapacityTrainingError("experiment card does not bind the Work 8 receipt")
    if card.get("anchor_policy", {}).get("sha256") != ANCHOR_POLICY_SHA256:
        raise CapacityTrainingError("experiment card does not bind anchor policy")
    if card.get("analysis_protocol", {}).get("sha256") != CAPACITY_ANALYSIS_PROTOCOL_SHA256:
        raise CapacityTrainingError("experiment card does not bind analysis protocol")
    if card.get("figure_spec", {}).get("sha256") != FIGURE_SPEC_SHA256:
        raise CapacityTrainingError("experiment card does not bind figure spec")
    card_sha = sha256_file(card_path)
    if card_sha != EXPERIMENT_CARD_SHA256:
        raise CapacityTrainingError("capacity experiment-card constant drifted")
    if dict(card.get("hidden_dims", {})) != dict(
        base_config.capacity_sensitivity.matched_hidden_dims
    ):
        raise CapacityTrainingError("capacity hidden-width contract drifted")
    if dict(card.get("parameter_counts", {})) != dict(
        base_config.capacity_sensitivity.matched_parameter_counts
    ):
        raise CapacityTrainingError("capacity parameter-count contract drifted")
    primary_acceptance = json.loads(
        primary_acceptance_path.read_text(encoding="utf-8")
    )
    if primary_acceptance.get("environment_lock", {}).get(
        "environment_lock_sha256"
    ) != PRIMARY_ENVIRONMENT_LOCK_SHA256:
        raise CapacityTrainingError("primary accepted environment lock changed")
    from .work9_capacity_acceptance import environment_lock

    if environment_lock().get("environment_lock_sha256") != PRIMARY_ENVIRONMENT_LOCK_SHA256:
        raise CapacityTrainingError("current environment differs from primary anchors")
    anchor_audit = verify_primary_selection_anchor_lineage(root)
    if anchor_audit["anchor_coordinate_count"] != 50:
        raise CapacityTrainingError("capacity anchor audit is incomplete")
    config = CapacityConfigView(base_config)

    acceptance_path = root / ACCEPTANCE_RECEIPT_RELATIVE
    acceptance: Mapping[str, Any] | None = None
    acceptance_sha: str | None = None
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        acceptance_sha = sha256_file(acceptance_path)
        if acceptance.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION:
            raise CapacityTrainingError("runner acceptance schema is unsupported")
        if acceptance.get("status") != "accepted_after_two_cuda_smokes":
            raise CapacityTrainingError("runner acceptance is not final")
        expected_lineage = {
            "work8_config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "experiment_card_sha256": card_sha,
            "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
            "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
            "figure_spec_sha256": FIGURE_SPEC_SHA256,
            "primary_selection_index_sha256": PRIMARY_SELECTION_INDEX_SHA256,
            "primary_outer_index_sha256": PRIMARY_OUTER_INDEX_SHA256,
            "gate1_hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
        }
        mismatches = {
            key: (wanted, acceptance.get(key))
            for key, wanted in expected_lineage.items()
            if acceptance.get(key) != wanted
        }
        if mismatches:
            raise CapacityTrainingError(f"runner acceptance lineage mismatch: {mismatches}")
        accepted_analysis_path = _verify_ref_mapping(
            acceptance.get("capacity_analysis_protocol", {}),
            root,
            context="capacity analysis protocol",
        )
        if accepted_analysis_path != analysis_path.resolve():
            raise CapacityTrainingError("runner acceptance points to another analysis protocol")
        accepted_dry_path = _verify_ref_mapping(
            acceptance.get("dry_run_receipt", {}),
            root,
            context="capacity no-training dry receipt",
        )
        if accepted_dry_path != (root / DRY_RECEIPT_RELATIVE).resolve():
            raise CapacityTrainingError("runner acceptance points to another dry receipt")
        dry_receipt = json.loads(accepted_dry_path.read_text(encoding="utf-8"))
        if dry_receipt.get("status") != "pass" or dry_receipt.get("safety", {}).get(
            "training_launched"
        ) is not False:
            raise CapacityTrainingError("capacity dry receipt is not no-training evidence")
        from .work9_capacity_acceptance import (
            verify_dry_receipt_payload_against_current,
        )

        try:
            verify_dry_receipt_payload_against_current(root, dry_receipt)
        except Exception as error:
            raise CapacityTrainingError(
                "capacity dry receipt no longer matches current runner code"
            ) from error
        for index, source in enumerate(acceptance.get("source_contracts", [])):
            _verify_ref_mapping(source, root, context=f"Work 9 accepted source {index}")
        from .work9_capacity_acceptance import bundle_digests, source_contracts

        if acceptance.get("source_contracts") != source_contracts(root):
            raise CapacityTrainingError("runner acceptance source inventory is incomplete")
        if acceptance.get("bundle_digests") != bundle_digests(root):
            raise CapacityTrainingError("runner source/schema/test bundle changed after acceptance")
        _verify_runner_acceptance_environment(acceptance)
        smokes = acceptance.get("cuda_smokes", [])
        if len(smokes) != 2 or any(item.get("device_type") != "cuda" for item in smokes):
            raise CapacityTrainingError("runner acceptance requires exactly two CUDA smokes")
        if {item.get("condition") for item in smokes} != {
            "A-mean", "AT-cross-randtrain"
        }:
            raise CapacityTrainingError("CUDA smokes do not cover both frozen branches")
        if any(
            item.get("cublas_workspace_config") != ":4096:8" for item in smokes
        ):
            raise CapacityTrainingError("CUDA smoke receipt lacks deterministic cuBLAS config")
        for index, item in enumerate(smokes):
            _verify_ref_mapping(
                item.get("receipt", {}), root, context=f"CUDA smoke receipt {index}"
            )
            _verify_ref_mapping(
                item.get("semantic_output", {}),
                root,
                context=f"CUDA smoke semantic output {index}",
            )
        unit_tests = acceptance.get("unit_tests", {})
        if unit_tests.get("status") != "pass" or int(
            unit_tests.get("tests_passed", 0)
        ) <= 0:
            raise CapacityTrainingError("runner acceptance has no passing unit-test receipt")
        _verify_ref_mapping(
            unit_tests.get("receipt", {}), root, context="unit-test receipt"
        )
        _verify_ref_mapping(unit_tests.get("log", {}), root, context="unit-test log")
    elif require_runner_acceptance:
        raise CapacityTrainingError(
            "production is locked: final runner acceptance requires two real CUDA smokes"
        )

    return FrozenWork9Inputs(root, config, receipt, acceptance, acceptance_sha, card_sha)


def validate_capacity_no_training(project_root: str | Path) -> dict[str, Any]:
    """Run the complete outer-blind capacity preflight without optimizer steps."""

    frozen = verify_frozen_work9_inputs(
        project_root, require_runner_acceptance=False
    )
    frozen.config.text_store.verify()
    capacity_rows: list[dict[str, Any]] = []
    for condition in frozen.config.conditions:
        built = frozen.config.instantiate_smoke_condition(condition)
        hidden_dim = frozen.config.capacity_sensitivity.matched_hidden_dims[condition]
        expected = frozen.config.capacity_sensitivity.matched_parameter_counts[condition]
        if built.total_parameters != expected or hidden_dim % frozen.config.model.num_heads:
            raise CapacityTrainingError(f"capacity preflight failed for {condition}")
        capacity_rows.append(
            {
                "condition": condition,
                "hidden_dim": hidden_dim,
                "parameter_count": expected,
                "entry_role": (
                    "new_capacity_fit"
                    if condition in NEW_FIT_CONDITIONS
                    else "immutable_primary_anchor"
                ),
            }
        )
    pairing_plans = frozen.work8_receipt.get("random_train_pairing_plans", [])
    if len(pairing_plans) != 25:
        raise CapacityTrainingError("Work 8 pairing-plan inventory is not exactly 25")
    anchor_audit = verify_primary_selection_anchor_lineage(frozen.project_root)
    logical_keys = {
        (condition, 0, fold, seed)
        for condition in frozen.config.conditions
        for fold in range(5)
        for seed in frozen.config.seeds
    }
    new_keys = {key for key in logical_keys if key[0] in NEW_FIT_CONDITIONS}
    anchor_keys = logical_keys - new_keys
    if (len(logical_keys), len(new_keys), len(anchor_keys)) != (200, 150, 50):
        raise CapacityTrainingError("hybrid coordinate construction drifted")
    primary_registry = frozen.project_root / "revised_experiments/artifacts/work9_runs_v1"
    primary_selection = primary_registry / "_matrix/selection_index.json"
    return {
        "schema_version": "paper2-v2-work9-capacity-no-training-receipt-1",
        "status": "pass",
        "protocol_id": frozen.config.protocol_id,
        "run_family": RUN_FAMILY,
        "lineage": {
            "work8_config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "experiment_card_sha256": EXPERIMENT_CARD_SHA256,
            "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
            "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
            "figure_spec_sha256": FIGURE_SPEC_SHA256,
            "primary_selection_index_sha256": sha256_file(primary_selection),
            # This is a preregistered expected identity only.  Phase A does
            # not stat, open, or hash the primary outer index.
            "expected_primary_outer_index_sha256_after_selection_freeze": (
                PRIMARY_OUTER_INDEX_SHA256
            ),
            "gate1_hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
        },
        "capacity_models": capacity_rows,
        "hybrid_coordinate_contract": {
            "logical_coordinates": 200,
            "new_fit_coordinates": 150,
            "immutable_primary_anchors": 50,
        },
        "fold_count": len(frozen.config.folds),
        "seed_count": len(frozen.config.seeds),
        "pairing_plan_count": len(pairing_plans),
        "anchor_audit": anchor_audit,
        "safety": {
            "training_launched": False,
            "optimizer_steps": 0,
            "checkpoints_written": 0,
            "new_capacity_outer_outcomes_accessed": False,
            "primary_outer_index_opened_or_hashed": False,
            "primary_outer_artifacts_opened_or_hashed": False,
            "primary_selection_registry_read_only_verified": True,
            "production_capacity_registry_accessed": False,
            "primary_registry_modified": False,
            "gate1_reopened": False,
            "gate2_reopened": False,
        },
    }


def require_process_determinism(seed: int, device: str | torch.device) -> None:
    """Fail before CUDA work if process-level determinism was not pre-established."""

    declared = str(device)
    expected_hash_seed = str(seed)
    if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
        raise CapacityTrainingError(
            f"PYTHONHASHSEED must equal coordinate seed {expected_hash_seed} before process start"
        )
    if declared.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise CapacityTrainingError(
            "CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch"
        )


def verify_cuda_device_identity(device: str | torch.device) -> dict[str, Any]:
    """Require the exact environment/GPU used by the immutable primary anchors."""

    value = torch.device(device)
    if value.type != "cuda" or not torch.cuda.is_available():
        raise CapacityTrainingError("capacity acceptance/production requires CUDA")
    torch.cuda.set_device(value)
    identity = {
        "device_name": torch.cuda.get_device_name(value),
        "compute_capability": list(torch.cuda.get_device_capability(value)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    expected = {
        "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "torch_version": EXPECTED_TORCH_VERSION,
        "cuda_version": EXPECTED_CUDA_VERSION,
        "cudnn_version": EXPECTED_CUDNN_VERSION,
    }
    if identity != expected:
        raise CapacityTrainingError(
            f"capacity device differs from immutable primary-anchor device: "
            f"expected {expected}, got {identity}"
        )
    return identity


def reset_cuda_peak_memory_stats(device: str | torch.device) -> None:
    """Select a CUDA device before resetting its allocator telemetry.

    PyTorch 2.6 may reject ``reset_peak_memory_stats(cuda_device)`` before a
    CUDA context exists.  Explicit device selection establishes the required
    context and also makes the telemetry target unambiguous.
    """

    device_value = torch.device(device)
    if device_value.type != "cuda":
        return
    torch.cuda.set_device(device_value)
    torch.cuda.reset_peak_memory_stats(device_value)


def seed_everything(seed: int, device: str | torch.device) -> torch.Generator:
    require_process_determinism(seed, device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise CapacityTrainingError("CUDA requested but unavailable")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def equal_call_batch_loss(
    logits: Tensor,
    labels: Tensor,
    sample_weights: Tensor,
    *,
    segment_count: int,
    call_count: int,
) -> tuple[Tensor, Tensor]:
    """Exact call-equal objective under uniform shuffled segment minibatches."""

    if logits.shape != labels.shape or logits.shape != sample_weights.shape:
        raise ValueError("logits, labels, and sample_weights must share [B]")
    if segment_count <= 0 or call_count <= 0:
        raise ValueError("global train segment/call counts must be positive")
    element_losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    batch_loss = (sample_weights * element_losses).mean() * (
        float(segment_count) / float(call_count)
    )
    return batch_loss, element_losses


def remaining_epoch_numbers(
    *, start_epoch: int, epoch_budget: int, stale_epochs: int, patience: int
) -> tuple[int, ...]:
    """Pure crash-resume guard for the persisted early-stop state."""

    if min(start_epoch, epoch_budget, patience) <= 0 or stale_epochs < 0:
        raise ValueError("epoch-resume inputs are invalid")
    if stale_epochs >= patience or start_epoch > epoch_budget:
        return ()
    return tuple(range(start_epoch, epoch_budget + 1))


def _build_optimizer(model: nn.Module, config: MatrixConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=tuple(OPTIMIZER_POLICY["betas"]),
        eps=OPTIMIZER_POLICY["eps"],
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _build_scheduler(optimizer: AdamW, config: MatrixConfig) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        threshold=SCHEDULER_POLICY["threshold"],
        threshold_mode="rel",
        cooldown=0,
        min_lr=config.training.minimum_learning_rate,
        eps=SCHEDULER_POLICY["eps"],
    )


def _model_bytes(model: nn.Module) -> bytes:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in sorted(model.state_dict().items())
    }
    return save_safetensors(state, metadata={"format": "paper2-v2-work9-capacity-safetensors-1"})


def _load_model_bytes(model: nn.Module, payload: bytes) -> None:
    state = load_safetensors(payload)
    model.load_state_dict(state, strict=True)


def _dataloader(
    dataset: Work9FeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda rows: collate_samples(rows, dataset.input_spec),
    )


def _coordinate_fields(coordinates: CapacityCoordinates, *, partition: str) -> dict[str, Any]:
    return {
        "run_id": coordinates.run_id,
        "protocol_id": coordinates.protocol_id,
        "run_family": coordinates.run_family,
        "partition": partition,
        "partition_repeat": coordinates.partition_repeat,
        "outer_fold": coordinates.outer_fold,
        "seed": coordinates.seed,
        "condition": coordinates.condition,
    }


def _infer_probabilities(
    model: nn.Module,
    dataset: Work9FeatureDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> list[SegmentPrediction]:
    model.eval()
    probability_by_segment: dict[str, float] = {}
    loader = _dataloader(dataset, batch_size=batch_size, shuffle=False, generator=None)
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch.to(device))
            probabilities = torch.sigmoid(logits).detach().cpu().to(torch.float64)
            if probabilities.shape != (batch.batch_size,) or not torch.isfinite(
                probabilities
            ).all():
                raise CapacityTrainingError("inference produced invalid probabilities")
            for segment_id, probability in zip(
                batch.segment_ids, probabilities.tolist(), strict=True
            ):
                if segment_id in probability_by_segment:
                    raise CapacityTrainingError("duplicate segment during inference")
                probability_by_segment[segment_id] = float(probability)
    if set(probability_by_segment) != set(dataset.segment_ids):
        raise CapacityTrainingError("inference did not cover the complete manifest")
    predictions = []
    for row in dataset.rows:
        predictions.append(
            SegmentPrediction(
                segment_id=str(row["segment_id"]),
                call_id=str(row["call_id"]),
                group_id=str(row["group_id"]),
                source=str(row.get("source", "unknown")),
                label=int(row["label"]),
                probability=probability_by_segment[str(row["segment_id"])],
                start=float(row["start"]),
                end=float(row["end"]),
            )
        )
    return predictions


def _segment_score_rows(
    predictions: Sequence[SegmentPrediction],
    *,
    coordinates: CapacityCoordinates,
    epoch: int,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "paper2-v2-work9-capacity-validation-segment-score-1",
            **_coordinate_fields(coordinates, partition="validation"),
            "epoch": epoch,
            "segment_id": row.segment_id,
            "call_id": row.call_id,
            "group_id": row.group_id,
            "source": row.source,
            "label": row.label,
            "probability": row.probability,
            "start": row.start,
            "end": row.end,
            "duration_seconds": row.end - row.start,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
        }
        for row in sorted(predictions, key=lambda item: item.segment_id)
    ]


def _call_score_rows(
    calls: Sequence[CallPrediction],
    *,
    coordinates: CapacityCoordinates,
    epoch: int,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "paper2-v2-work9-capacity-validation-call-score-1",
            **_coordinate_fields(coordinates, partition="validation"),
            "epoch": epoch,
            "call_id": row.call_id,
            "group_id": row.group_id,
            "source": row.source,
            "label": row.label,
            "probability": row.probability,
            "n_segments": row.n_segments,
            "start": row.start,
            "end": row.end,
            "duration_seconds": row.end - row.start,
            "aggregation_rule": "mean_probability",
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
        }
        for row in sorted(calls, key=lambda item: item.call_id)
    ]


def _prediction_rows(
    predictions: Sequence[SegmentPrediction],
    calls: Sequence[CallPrediction],
    *,
    coordinates: CapacityCoordinates,
    partition: str,
    threshold: float,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = _coordinate_fields(coordinates, partition=partition)
    segments = [
        {
            "schema_version": "paper2-v2-work9-capacity-segment-prediction-1",
            **fields,
            "segment_id": row.segment_id,
            "call_id": row.call_id,
            "group_id": row.group_id,
            "source": row.source,
            "label": row.label,
            "probability": row.probability,
            "hard_prediction": int(row.probability >= threshold),
            "start": row.start,
            "end": row.end,
            "duration_seconds": row.end - row.start,
            "threshold": threshold,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
        }
        for row in sorted(predictions, key=lambda item: item.segment_id)
    ]
    call_rows = [
        {
            "schema_version": "paper2-v2-work9-capacity-call-prediction-1",
            **fields,
            "call_id": row.call_id,
            "group_id": row.group_id,
            "source": row.source,
            "label": row.label,
            "probability": row.probability,
            "hard_prediction": int(row.probability >= threshold),
            "n_segments": row.n_segments,
            "start": row.start,
            "end": row.end,
            "duration_seconds": row.end - row.start,
            "aggregation_rule": "mean_probability",
            "threshold": threshold,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
        }
        for row in sorted(calls, key=lambda item: item.call_id)
    ]
    return segments, call_rows


def threshold_trace_payload(
    calls: Sequence[CallPrediction],
    *,
    config: MatrixConfig,
    coordinates: CapacityCoordinates,
    validation_call_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = []
    for threshold in threshold_grid(config.selection):
        raw_metrics = compute_call_metrics(calls, threshold=threshold)
        confusion = {
            key: int(raw_metrics[key]) for key in ("tn", "fp", "fn", "tp")
        }
        metrics = {
            "n_calls": int(raw_metrics["n_calls"]),
            "confusion": confusion,
            **{name: raw_metrics[name] for name in config.statistics.reported_call_metrics},
        }
        key = [
            float(raw_metrics["macro_f1"]),
            -float(raw_metrics["fpr"]),
            float(raw_metrics["sensitivity"]),
            -abs(threshold - 0.5),
            threshold,
        ]
        candidates.append(
            {"threshold": threshold, "metrics": metrics, "selection_key": key}
        )
    selected = max(candidates, key=lambda item: tuple(item["selection_key"]))
    return {
        "schema_version": "paper2-v2-work9-capacity-threshold-trace-1",
        **_coordinate_fields(coordinates, partition="validation"),
        "validation_call_ledger": dict(validation_call_ledger),
        "threshold_grid": {"start": 0.01, "end": 0.99, "step": 0.01, "count": 99},
        "threshold_tie_break": list(config.selection.threshold_tie_break),
        "candidates": candidates,
        "selected_threshold": selected["threshold"],
    }


def _ref_for_bytes(
    project_root: Path,
    path: Path,
    payload: bytes,
    *,
    row_count: int | None,
) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "row_count": row_count,
    }


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _expected_pairing_sha(
    receipt: Mapping[str, Any], coordinates: CapacityCoordinates, train_sha256: str
) -> str:
    matches = [
        row
        for row in receipt.get("random_train_pairing_plans", [])
        if row.get("condition") == coordinates.condition
        and int(row.get("partition_repeat", -1)) == coordinates.partition_repeat
        and int(row.get("outer_fold", -1)) == coordinates.outer_fold
        and int(row.get("seed", -1)) == coordinates.seed
        and row.get("train_manifest_sha256") == train_sha256
    ]
    if len(matches) != 1:
        raise CapacityTrainingError("Work 8 receipt has no unique pairing plan for coordinate")
    return str(matches[0]["plan_sha256"])


def _build_pairing(
    coordinates: CapacityCoordinates,
    train_rows: Sequence[Mapping[str, Any]],
    *,
    receipt: Mapping[str, Any],
    train_sha256: str,
) -> PairingPlan | None:
    plan = pairing_plan_for_split(
        coordinates.condition,
        split_role="train",
        rows=train_rows,
        seed=coordinates.seed,
    )
    if coordinates.condition == ConditionId.AUDIO_TEXT_CROSS_RANDOM_TRAIN.value:
        if plan is None:
            raise CapacityTrainingError("random-train condition did not produce a plan")
        expected = _expected_pairing_sha(receipt, coordinates, train_sha256)
        if plan.sha256 != expected:
            raise CapacityTrainingError(
                f"regenerated random-pairing digest differs: {plan.sha256} != {expected}"
            )
    elif plan is not None:
        raise CapacityTrainingError("non-random condition unexpectedly received pairing plan")
    return plan


def _training_state_bytes(
    *,
    epoch: int,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    best_nll: float,
    best_epoch: int,
    stale_epochs: int,
    generator: torch.Generator,
) -> bytes:
    payload = {
        "schema_version": "paper2-v2-work9-capacity-training-state-1",
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_nll": best_nll,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "dataloader_generator_state": generator.get_state(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


@dataclass(frozen=True)
class FitSelectionResult:
    workspace: CapacityRunWorkspace
    seal: CapacitySelectionSeal
    selection_seal_path: Path
    selection_run_record_path: Path


def _verify_completed_selection_run(
    frozen: FrozenWork9Inputs,
    workspace: CapacityRunWorkspace,
) -> FitSelectionResult | None:
    seal_path = workspace.path / "selection/selection_seal.json"
    record_path = workspace.path / "selection_run_record.json"
    index_path = workspace.path / "selection_artifact_index.json"
    marker_path = workspace.path / "SELECTION_COMPLETE.json"
    paths = (seal_path, record_path, index_path, marker_path)
    if not all(path.is_file() for path in paths):
        return None
    seal_raw = json.loads(seal_path.read_text(encoding="utf-8"))
    validate_json_schema(
        seal_raw, project_root=frozen.project_root, schema_name=SELECTION_SEAL_SCHEMA
    )
    seal = load_selection_seal(seal_path, frozen.project_root)
    if seal.coordinates != workspace.coordinates:
        raise CapacityTrainingError("completed selection seal coordinate mismatch")
    seal.verify(workspace.path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    validate_json_schema(
        record, project_root=frozen.project_root, schema_name=RUN_RECORD_SCHEMA
    )
    if record.get("status") != "selection_complete":
        raise CapacityTrainingError("selection run record is not complete")
    verify_capacity_selection_seal_against_frozen(
        frozen, seal, run_root=workspace.path, record=record
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("run_id") != workspace.coordinates.run_id
        or marker.get("selection_seal_sha256") != sha256_file(seal_path)
        or marker.get("outer_eval_accessed") is not False
    ):
        raise CapacityTrainingError("selection completion marker lineage mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    validate_json_schema(
        index,
        project_root=frozen.project_root,
        schema_name=SELECTION_ARTIFACT_INDEX_SCHEMA,
    )
    if index.get("run_id") != workspace.coordinates.run_id:
        raise CapacityTrainingError("selection artifact index run_id mismatch")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CapacityTrainingError("selection artifact inventory is malformed")
    if _load_ref(artifacts["selection_seal"], frozen.project_root, ledger=False) != seal_path:
        raise CapacityTrainingError("selection artifact index points to another seal")
    if _load_ref(
        artifacts["selection_run_record"], frozen.project_root, ledger=False
    ) != record_path:
        raise CapacityTrainingError("selection artifact index points to another run record")
    for key, ledger in (
        ("checkpoint", False),
        ("validation_segment_ledger", True),
        ("validation_call_ledger", True),
        ("epoch_trace", False),
        ("threshold_trace", False),
    ):
        _load_ref(artifacts[key], frozen.project_root, ledger=ledger)
    if seal.pairing_plan_artifact is not None:
        _load_ref(artifacts["pairing_plan"], frozen.project_root, ledger=True)
    return FitSelectionResult(workspace, seal, seal_path, record_path)


def fit_select_coordinate(
    project_root: str | Path,
    coordinates: CapacityCoordinates,
    *,
    device: str,
    require_runner_acceptance: bool = True,
    registry: CapacityRegistry | None = None,
    epoch_budget: int | None = None,
) -> FitSelectionResult:
    """Fit and seal one coordinate using train/validation only.

    ``epoch_budget`` is a test/smoke hook and is rejected when final runner
    acceptance is required.  Production always consumes the frozen budget.
    """

    frozen = verify_frozen_work9_inputs(
        project_root, require_runner_acceptance=require_runner_acceptance
    )
    config = frozen.config
    if coordinates.protocol_id != config.protocol_id or coordinates.seed not in config.seeds:
        raise CapacityTrainingError("coordinate is outside frozen Work 8 config")
    if epoch_budget is not None and require_runner_acceptance:
        raise CapacityTrainingError("production fit cannot override the frozen epoch budget")
    if require_runner_acceptance:
        verify_cuda_device_identity(device)
    total_epochs = config.training.epoch_budget if epoch_budget is None else int(epoch_budget)
    if total_epochs <= 0:
        raise ValueError("epoch budget must be positive")
    generator = seed_everything(coordinates.seed, device)
    device_value = torch.device(device)
    invocation_started = time.monotonic()
    reset_cuda_peak_memory_stats(device_value)
    registry = registry or CapacityRegistry(frozen.project_root)
    canonical_registry = (frozen.project_root / REGISTRY_RELATIVE).resolve()
    if require_runner_acceptance:
        if coordinates.condition not in NEW_FIT_CONDITIONS:
            raise CapacityTrainingError("immutable primary anchors cannot be retrained")
        if registry.root != canonical_registry:
            raise CapacityTrainingError("production capacity fit requires canonical registry")
    elif coordinates.condition in ANCHOR_CONDITIONS:
        if epoch_budget is None or registry.root == canonical_registry:
            raise CapacityTrainingError(
                "anchor models are allowed only in bounded non-production smoke registries"
            )
    workspace = registry.reserve(coordinates)
    with workspace.lock():
        completed = _verify_completed_selection_run(frozen, workspace)
        if completed is not None:
            return completed
        selection = config.selection_view(
            coordinates.partition_repeat, coordinates.outer_fold
        )
        # SelectionView is deliberately outer-blind.
        if hasattr(selection, "outer_eval_manifest"):
            raise CapacityTrainingError("selection view exposed outer_eval")
        train_rows = _load_manifest_rows(selection.train_manifest.path)
        plan = _build_pairing(
            coordinates,
            train_rows,
            receipt=frozen.work8_receipt,
            train_sha256=selection.train_manifest.sha256,
        )
        built = (
            config.instantiate_smoke_condition(coordinates.condition)
            if coordinates.condition in ANCHOR_CONDITIONS
            else config.instantiate_condition(coordinates.condition)
        )
        store = (
            IndexedTextStore(config.text_store.metadata.path.parent, verify_hashes=True)
            if "text" in built.input_spec.global_features
            else None
        )
        train_data = Work9FeatureDataset(
            selection.train_manifest.path,
            input_spec=built.input_spec,
            project_root=frozen.project_root,
            text_store=store,
            pairing_plan=plan,
            cache_mfcc=True,
        )
        validation_data = Work9FeatureDataset(
            selection.val_manifest.path,
            input_spec=built.input_spec,
            project_root=frozen.project_root,
            text_store=store,
            pairing_plan=None,
            cache_mfcc=True,
        )
        plan_ref = None
        if plan is not None:
            plan_ref = workspace.write_jsonl_or_verify(
                "pairing_plan.jsonl", plan.as_records()
            )
            if plan_ref.sha256 != plan.sha256:
                # PairingPlan.sha256 hashes its compact semantic payload, while
                # the ledger has richer audit columns. Bind both explicitly.
                workspace.write_json_or_verify(
                    "pairing_plan_semantic.json",
                    {
                        "schema_version": "paper2-v2-work9-capacity-pairing-semantic-1",
                        "label_policy": "unconstrained_global_random",
                        "donor_scope": "same_fold_train_manifest",
                        "plan_sha256": plan.sha256,
                        "ledger_sha256": plan_ref.sha256,
                    },
                )
        run_plan = {
            "schema_version": "paper2-v2-work9-capacity-run-plan-1",
            **asdict(coordinates),
            "run_id": coordinates.run_id,
            "phase": "fit_select",
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": (
                frozen.runner_acceptance_sha256 or "0" * 64
            ),
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
            "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
            "train_manifest_sha256": selection.train_manifest.sha256,
            "validation_manifest_sha256": selection.val_manifest.sha256,
            "outer_eval_accessed": False,
            "pairing_plan_sha256": None if plan is None else plan.sha256,
            "pairing_plan_artifact": (
                None
                if plan_ref is None
                else artifact_record(
                    plan_ref.path,
                    frozen.project_root,
                    row_count=len(plan.as_records()),
                )
            ),
            "pairing_label_policy": (
                None if plan is None else "unconstrained_global_random"
            ),
            "training": {
                "batch_size": config.training.batch_size,
                "epoch_budget": total_epochs,
                "loss_objective": "mean((N_segments/C_calls)*inverse_call_length*BCE)",
                "optimizer": {
                    **OPTIMIZER_POLICY,
                    "learning_rate": config.training.learning_rate,
                    "weight_decay": config.training.weight_decay,
                },
                "scheduler": {
                    **SCHEDULER_POLICY,
                    "factor": config.training.scheduler_factor,
                    "patience": config.training.scheduler_patience,
                    "minimum_learning_rate": config.training.minimum_learning_rate,
                },
            },
            "determinism": {
                **DETERMINISM_POLICY,
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            },
            "mfcc_cache": {
                "policy": CACHE_POLICY,
                "train_sha256": train_data.cache_content_sha256,
                "validation_sha256": validation_data.cache_content_sha256,
                "bytes": train_data.cache_bytes + validation_data.cache_bytes,
            },
        }
        workspace.write_json_or_verify("run_plan.json", run_plan)

        model = built.model.to(device_value)
        optimizer = _build_optimizer(model, config)
        scheduler = _build_scheduler(optimizer, config)
        epoch_entries: list[dict[str, Any]] = []
        best_nll = math.inf
        best_epoch = 0
        stale_epochs = 0
        start_epoch = 1
        accumulated_wall_seconds = 0.0
        peak_cuda_allocated_bytes = 0
        peak_cuda_reserved_bytes = 0
        existing = sorted(workspace.path.glob("epochs/epoch_*/epoch_record.json"))
        if existing:
            records = [json.loads(path.read_text(encoding="utf-8")) for path in existing]
            epoch_numbers = [int(row["epoch"]) for row in records]
            if epoch_numbers != list(range(1, max(epoch_numbers) + 1)):
                raise CapacityTrainingError("epoch resume history is noncontiguous")
            for record in records:
                trace = record.get("trace_entry", {})
                if int(trace.get("epoch", -1)) != int(record["epoch"]):
                    raise CapacityTrainingError("epoch record/trace epoch mismatch")
                checkpoint_path = _load_ref(
                    trace["checkpoint"], frozen.project_root, ledger=False
                )
                segment_score_path = _load_ref(
                    trace["validation_segment_scores"],
                    frozen.project_root,
                    ledger=True,
                )
                call_score_path = _load_ref(
                    trace["validation_call_scores"],
                    frozen.project_root,
                    ledger=True,
                )
                if workspace.path not in checkpoint_path.parents or any(
                    workspace.path not in path.parents
                    for path in (segment_score_path, call_score_path)
                ):
                    raise CapacityTrainingError("resumed epoch artifact escaped reserved run")
                segment_score_rows = _load_manifest_rows(segment_score_path)
                call_score_rows = _load_manifest_rows(call_score_path)
                validate_jsonl_schema(
                    segment_score_rows,
                    project_root=frozen.project_root,
                    schema_name=SEGMENT_SCORE_SCHEMA,
                )
                validate_jsonl_schema(
                    call_score_rows,
                    project_root=frozen.project_root,
                    schema_name=CALL_SCORE_SCHEMA,
                )
            latest = records[-1]
            model_path = workspace.path / latest["checkpoint"]["path_within_run"]
            state_path = workspace.path / latest["training_state"]["path_within_run"]
            _verify_exact_file(model_path, latest["checkpoint"]["sha256"], "epoch checkpoint")
            _verify_exact_file(state_path, latest["training_state"]["sha256"], "training state")
            _load_model_bytes(model, model_path.read_bytes())
            state = torch.load(io.BytesIO(state_path.read_bytes()), map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best_nll = float(state["best_nll"])
            best_epoch = int(state["best_epoch"])
            stale_epochs = int(state["stale_epochs"])
            random.setstate(state["python_random_state"])
            np.random.set_state(state["numpy_random_state"])
            torch.set_rng_state(state["torch_cpu_rng_state"])
            if torch.cuda.is_available() and state["torch_cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])
            generator.set_state(state["dataloader_generator_state"])
            epoch_entries.extend(row["trace_entry"] for row in records)
            accumulated_wall_seconds = sum(
                float(row.get("wall_time_seconds", 0.0)) for row in records
            )
            peak_cuda_allocated_bytes = max(
                (int(row.get("peak_cuda_allocated_bytes", 0)) for row in records),
                default=0,
            )
            peak_cuda_reserved_bytes = max(
                (int(row.get("peak_cuda_reserved_bytes", 0)) for row in records),
                default=0,
            )
            start_epoch = int(latest["epoch"]) + 1

        remaining_epochs: Iterable[int] = remaining_epoch_numbers(
            start_epoch=start_epoch,
            epoch_budget=total_epochs,
            stale_epochs=stale_epochs,
            patience=config.training.early_stop_patience,
        )
        for epoch in remaining_epochs:
            epoch_started = time.monotonic()
            model.train()
            seen: set[str] = set()
            weighted_loss_sum = 0.0
            train_loader = _dataloader(
                train_data,
                batch_size=config.training.batch_size,
                shuffle=True,
                generator=generator,
            )
            for batch in train_loader:
                duplicates = seen.intersection(batch.segment_ids)
                if duplicates:
                    raise CapacityTrainingError("training repeated a segment in one epoch")
                seen.update(batch.segment_ids)
                batch_device = batch.to(device_value)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_device)
                batch_loss, element_losses = equal_call_batch_loss(
                    logits,
                    batch_device.labels,
                    batch_device.sample_weights,
                    segment_count=train_data.segment_count,
                    call_count=train_data.call_count,
                )
                if not torch.isfinite(batch_loss):
                    raise CapacityTrainingError("non-finite training loss")
                batch_loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise CapacityTrainingError("non-finite gradient")
                optimizer.step()
                weighted_loss_sum += math.fsum(
                    (batch_device.sample_weights * element_losses)
                    .detach()
                    .to(torch.float64)
                    .cpu()
                    .tolist()
                )
            if seen != set(train_data.segment_ids):
                raise CapacityTrainingError("training epoch did not visit every segment exactly once")
            train_reported_loss = weighted_loss_sum / train_data.call_count

            checkpoint_bytes = _model_bytes(model)
            checkpoint_sha = sha256_bytes(checkpoint_bytes)
            validation_predictions = _infer_probabilities(
                model,
                validation_data,
                batch_size=config.training.batch_size,
                device=device_value,
            )
            validation_calls = aggregate_mean_probability(validation_predictions)
            validation_nll = call_level_nll(validation_calls)
            diagnostic_threshold = select_validation_threshold(
                validation_calls, config.selection
            ).threshold
            segment_scores = _segment_score_rows(
                validation_predictions,
                coordinates=coordinates,
                epoch=epoch,
                checkpoint_sha256=checkpoint_sha,
                manifest_sha256=selection.val_manifest.sha256,
            )
            call_scores = _call_score_rows(
                validation_calls,
                coordinates=coordinates,
                epoch=epoch,
                checkpoint_sha256=checkpoint_sha,
                manifest_sha256=selection.val_manifest.sha256,
            )
            validate_jsonl_schema(
                segment_scores,
                project_root=frozen.project_root,
                schema_name=SEGMENT_SCORE_SCHEMA,
            )
            validate_jsonl_schema(
                call_scores,
                project_root=frozen.project_root,
                schema_name=CALL_SCORE_SCHEMA,
            )
            if validation_nll < best_nll:
                best_nll = validation_nll
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            scheduler.step(validation_nll)
            state_bytes = _training_state_bytes(
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                best_nll=best_nll,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                generator=generator,
            )
            epoch_dir = workspace.path / "epochs" / f"epoch_{epoch:03d}"
            segment_bytes = canonical_jsonl_bytes(segment_scores)
            call_bytes = canonical_jsonl_bytes(call_scores)
            trace_entry = {
                "epoch": epoch,
                "validation_call_nll": validation_nll,
                "train_reported_loss": train_reported_loss,
                "diagnostic_threshold": diagnostic_threshold,
                "checkpoint": _ref_for_bytes(
                    frozen.project_root,
                    epoch_dir / "model.safetensors",
                    checkpoint_bytes,
                    row_count=None,
                ),
                "validation_segment_scores": _ref_for_bytes(
                    frozen.project_root,
                    epoch_dir / "validation_segment_scores.jsonl",
                    segment_bytes,
                    row_count=len(segment_scores),
                ),
                "validation_call_scores": _ref_for_bytes(
                    frozen.project_root,
                    epoch_dir / "validation_call_scores.jsonl",
                    call_bytes,
                    row_count=len(call_scores),
                ),
            }
            relative_epoch = f"epochs/epoch_{epoch:03d}"
            epoch_record = {
                "schema_version": "paper2-v2-work9-capacity-epoch-record-1",
                "epoch": epoch,
                "checkpoint": {
                    "path_within_run": f"{relative_epoch}/model.safetensors",
                    "sha256": checkpoint_sha,
                },
                "training_state": {
                    "path_within_run": f"{relative_epoch}/training_state.pt",
                    "sha256": sha256_bytes(state_bytes),
                },
                "trace_entry": trace_entry,
                "wall_time_seconds": time.monotonic() - epoch_started,
                "peak_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device_value)
                    if device_value.type == "cuda"
                    else 0
                ),
                "peak_cuda_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device_value)
                    if device_value.type == "cuda"
                    else 0
                ),
            }
            workspace.publish_directory_once(
                relative_epoch,
                {
                    "model.safetensors": checkpoint_bytes,
                    "training_state.pt": state_bytes,
                    "validation_segment_scores.jsonl": segment_bytes,
                    "validation_call_scores.jsonl": call_bytes,
                    "epoch_record.json": canonical_json_bytes(epoch_record),
                },
            )
            epoch_entries.append(trace_entry)
            accumulated_wall_seconds += float(epoch_record["wall_time_seconds"])
            peak_cuda_allocated_bytes = max(
                peak_cuda_allocated_bytes,
                int(epoch_record["peak_cuda_allocated_bytes"]),
            )
            peak_cuda_reserved_bytes = max(
                peak_cuda_reserved_bytes,
                int(epoch_record["peak_cuda_reserved_bytes"]),
            )
            if stale_epochs >= config.training.early_stop_patience:
                break

        if not epoch_entries:
            raise CapacityTrainingError("fit selection produced no complete epochs")
        selected_entry = min(
            epoch_entries,
            key=lambda row: (float(row["validation_call_nll"]), int(row["epoch"])),
        )
        selected_epoch = int(selected_entry["epoch"])
        selected_checkpoint_path = frozen.project_root / selected_entry["checkpoint"]["path"]
        selected_checkpoint_bytes = selected_checkpoint_path.read_bytes()
        selected_checkpoint_sha = sha256_bytes(selected_checkpoint_bytes)
        _load_model_bytes(model, selected_checkpoint_bytes)
        selected_predictions = _infer_probabilities(
            model,
            validation_data,
            batch_size=config.training.batch_size,
            device=device_value,
        )
        selected_calls = aggregate_mean_probability(selected_predictions)
        selected_nll = call_level_nll(selected_calls)
        if not math.isclose(
            selected_nll,
            float(selected_entry["validation_call_nll"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise CapacityTrainingError("selected checkpoint validation NLL did not reproduce")
        threshold_selection = select_validation_threshold(selected_calls, config.selection)
        threshold = threshold_selection.threshold
        selected_segment_rows, selected_call_rows = _prediction_rows(
            selected_predictions,
            selected_calls,
            coordinates=coordinates,
            partition="validation",
            threshold=threshold,
            checkpoint_sha256=selected_checkpoint_sha,
            manifest_sha256=selection.val_manifest.sha256,
        )
        validate_jsonl_schema(
            selected_segment_rows,
            project_root=frozen.project_root,
            schema_name=SEGMENT_PREDICTION_SCHEMA,
        )
        validate_jsonl_schema(
            selected_call_rows,
            project_root=frozen.project_root,
            schema_name=CALL_PREDICTION_SCHEMA,
        )
        selection_dir = workspace.path / "selection"
        validation_segment_bytes = canonical_jsonl_bytes(selected_segment_rows)
        validation_call_bytes = canonical_jsonl_bytes(selected_call_rows)
        validation_call_ref = _ref_for_bytes(
            frozen.project_root,
            selection_dir / "validation_calls.jsonl",
            validation_call_bytes,
            row_count=len(selected_call_rows),
        )
        epoch_trace = {
            "schema_version": "paper2-v2-work9-capacity-epoch-trace-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "monitor": "validation_call_nll",
            "tie_break": "earlier_epoch",
            "epochs": epoch_entries,
            "selected_epoch": selected_epoch,
        }
        threshold_trace = threshold_trace_payload(
            selected_calls,
            config=config,
            coordinates=coordinates,
            validation_call_ledger=validation_call_ref,
        )
        validate_json_schema(
            epoch_trace, project_root=frozen.project_root, schema_name=EPOCH_TRACE_SCHEMA
        )
        validate_json_schema(
            threshold_trace,
            project_root=frozen.project_root,
            schema_name=THRESHOLD_TRACE_SCHEMA,
        )
        epoch_trace_bytes = canonical_json_bytes(epoch_trace)
        threshold_trace_bytes = canonical_json_bytes(threshold_trace)
        selection_refs = {
            "checkpoint": _ref_for_bytes(
                frozen.project_root,
                selection_dir / "checkpoint.safetensors",
                selected_checkpoint_bytes,
                row_count=None,
            ),
            "validation_segment_ledger": _ref_for_bytes(
                frozen.project_root,
                selection_dir / "validation_segments.jsonl",
                validation_segment_bytes,
                row_count=len(selected_segment_rows),
            ),
            "validation_call_ledger": validation_call_ref,
            "epoch_trace": _ref_for_bytes(
                frozen.project_root,
                selection_dir / "epoch_trace.json",
                epoch_trace_bytes,
                row_count=None,
            ),
            "threshold_trace": _ref_for_bytes(
                frozen.project_root,
                selection_dir / "threshold_trace.json",
                threshold_trace_bytes,
                row_count=None,
            ),
        }
        seal_mapping = {
            "schema_version": "paper2-v2-work9-capacity-selection-seal-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": (
                frozen.runner_acceptance_sha256 or "0" * 64
            ),
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
            "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
            "train_manifest_sha256": selection.train_manifest.sha256,
            "validation_manifest_sha256": selection.val_manifest.sha256,
            "feature_store_metadata_sha256": config.text_store.metadata.sha256,
            "feature_store_index_sha256": config.text_store.index_sha256,
            "feature_store_vectors_sha256": config.text_store.vectors_sha256,
            "pairing_plan_sha256": None if plan is None else plan.sha256,
            "pairing_plan_artifact": (
                None
                if plan_ref is None
                else artifact_record(
                    plan_ref.path,
                    frozen.project_root,
                    row_count=len(plan.as_records()),
                )
            ),
            "selected_epoch": selected_epoch,
            "validation_call_nll": selected_nll,
            "threshold": threshold,
            **selection_refs,
        }
        if frozen.runner_acceptance_sha256 is None:
            # Smoke/test selections cannot masquerade as production seals.
            seal_mapping["runner_acceptance_sha256"] = "0" * 64
        validate_json_schema(
            seal_mapping,
            project_root=frozen.project_root,
            schema_name=SELECTION_SEAL_SCHEMA,
        )
        selection_files = {
            "checkpoint.safetensors": selected_checkpoint_bytes,
            "validation_segments.jsonl": validation_segment_bytes,
            "validation_calls.jsonl": validation_call_bytes,
            "epoch_trace.json": epoch_trace_bytes,
            "threshold_trace.json": threshold_trace_bytes,
            "selection_seal.json": canonical_json_bytes(seal_mapping),
        }
        workspace.publish_directory_once("selection", selection_files)
        seal_path = selection_dir / "selection_seal.json"
        seal = load_selection_seal(seal_path, frozen.project_root)
        seal.verify(workspace.path)
        selection_record_path = workspace.path / "selection_run_record.json"
        if selection_record_path.exists():
            run_record = json.loads(selection_record_path.read_text(encoding="utf-8"))
            validate_json_schema(
                run_record,
                project_root=frozen.project_root,
                schema_name=RUN_RECORD_SCHEMA,
            )
            if (
                run_record.get("run_id") != coordinates.run_id
                or run_record.get("status") != "selection_complete"
                or run_record.get("lineage", {}).get("config_sha256")
                != WORK8_CONFIG_SHA256
            ):
                raise CapacityTrainingError("existing selection run record lineage differs")
            run_record_ref = ArtifactRef(
                selection_record_path.resolve(), sha256_file(selection_record_path)
            )
        else:
            run_record = _selection_run_record(
                frozen=frozen,
                coordinates=coordinates,
                workspace=workspace,
                built=built,
                train_data=train_data,
                validation_data=validation_data,
                selection=selection,
                plan=plan,
                plan_ref=plan_ref,
                seal=seal,
                total_epochs=len(epoch_entries),
                device=device_value,
                wall_time_seconds=accumulated_wall_seconds,
                invocation_wall_time_seconds=time.monotonic() - invocation_started,
                peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
                peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
            )
            validate_json_schema(
                run_record,
                project_root=frozen.project_root,
                schema_name=RUN_RECORD_SCHEMA,
            )
            run_record_ref = workspace.write_json_once(
                "selection_run_record.json", run_record
            )
        verify_capacity_selection_seal_against_frozen(
            frozen, seal, run_root=workspace.path, record=run_record
        )
        inventory = {
            "schema_version": "paper2-v2-work9-capacity-selection-artifact-index-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "artifacts": {
                "selection_seal": artifact_record(seal_path, frozen.project_root),
                "selection_run_record": artifact_record(
                    run_record_ref.path, frozen.project_root
                ),
                **selection_refs,
            },
        }
        if plan_ref is not None:
            inventory["artifacts"]["pairing_plan"] = artifact_record(
                plan_ref.path,
                frozen.project_root,
                row_count=len(plan.as_records()),
            )
        validate_json_schema(
            inventory,
            project_root=frozen.project_root,
            schema_name=SELECTION_ARTIFACT_INDEX_SCHEMA,
        )
        workspace.write_json_or_verify("selection_artifact_index.json", inventory)
        workspace.write_json_or_verify(
            "SELECTION_COMPLETE.json",
            {
                "schema_version": "paper2-v2-work9-capacity-selection-complete-1",
                "run_id": coordinates.run_id,
                **asdict(coordinates),
                "selection_seal_sha256": sha256_file(seal_path),
                "outer_eval_accessed": False,
            },
        )
        return FitSelectionResult(
            workspace, seal, seal_path, run_record_ref.path
        )


def _selection_run_record(
    *,
    frozen: FrozenWork9Inputs,
    coordinates: CapacityCoordinates,
    workspace: CapacityRunWorkspace,
    built: Any,
    train_data: Work9FeatureDataset,
    validation_data: Work9FeatureDataset,
    selection: Any,
    plan: PairingPlan | None,
    plan_ref: Any,
    seal: CapacitySelectionSeal,
    total_epochs: int,
    device: torch.device,
    wall_time_seconds: float,
    invocation_wall_time_seconds: float,
    peak_cuda_allocated_bytes: int,
    peak_cuda_reserved_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": "paper2-v2-work9-capacity-run-record-1",
        "run_id": coordinates.run_id,
        **asdict(coordinates),
        "phase": "selection",
        "status": "selection_complete",
        "lineage": {
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": (
                frozen.runner_acceptance_sha256 or "0" * 64
            ),
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
            "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
        },
        "data": {
            "train_manifest_sha256": selection.train_manifest.sha256,
            "validation_manifest_sha256": selection.val_manifest.sha256,
            "outer_eval_manifest_sha256": None,
            "grouping_policy_id": frozen.config.grouping_policy_id,
            "feature_store_ids": {
                "text_feature_set_id": frozen.config.text_store.feature_set_id,
                "metadata_sha256": frozen.config.text_store.metadata.sha256,
                "index_sha256": frozen.config.text_store.index_sha256,
                "vectors_sha256": frozen.config.text_store.vectors_sha256,
            },
            "mfcc_cache_policy": CACHE_POLICY,
            "mfcc_cache_sha256": {
                "train": train_data.cache_content_sha256,
                "validation": validation_data.cache_content_sha256,
            },
        },
        "model": {
            "architecture": built.architecture,
            "total_parameters": built.total_parameters,
            "trainable_parameters": built.trainable_parameters,
        },
        "randomness": {
            "seed": coordinates.seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            **DETERMINISM_POLICY,
        },
        "training": {
            "epochs_completed": total_epochs,
            "optimizer": {
                **OPTIMIZER_POLICY,
                "learning_rate": frozen.config.training.learning_rate,
                "weight_decay": frozen.config.training.weight_decay,
            },
            "scheduler": {
                **SCHEDULER_POLICY,
                "factor": frozen.config.training.scheduler_factor,
                "patience": frozen.config.training.scheduler_patience,
                "minimum_learning_rate": frozen.config.training.minimum_learning_rate,
            },
            "loss_objective": "call_equal_fixed_global_scale_v1",
            "pairing_plan_sha256": None if plan is None else plan.sha256,
        },
        "selection": {
            "selected_epoch": seal.selected_epoch,
            "validation_call_nll": seal.validation_call_nll,
            "threshold": seal.threshold,
            "checkpoint_tie_break": "earlier_epoch",
            "threshold_candidates": 99,
        },
        "artifacts": {
            "selection_seal": artifact_record(
                workspace.path / "selection/selection_seal.json", frozen.project_root
            ),
            "checkpoint": artifact_record(seal.checkpoint.path, frozen.project_root),
            "validation_segments": artifact_record(
                seal.validation_segment_ledger.path,
                frozen.project_root,
                row_count=len(validation_data),
            ),
            "validation_calls": artifact_record(
                seal.validation_call_ledger.path,
                frozen.project_root,
                row_count=validation_data.call_count,
            ),
            "epoch_trace": artifact_record(seal.epoch_trace.path, frozen.project_root),
            "threshold_trace": artifact_record(
                seal.threshold_trace.path, frozen.project_root
            ),
            "pairing_plan": (
                None
                if plan_ref is None
                else artifact_record(
                    plan_ref.path,
                    frozen.project_root,
                    row_count=len(plan.as_records()),
                )
            ),
        },
        "compute": {
            "device": str(device),
            "device_identity": (
                verify_cuda_device_identity(device)
                if device.type == "cuda"
                else None
            ),
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "wall_time_seconds": wall_time_seconds,
            "invocation_wall_time_seconds": invocation_wall_time_seconds,
            "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "failure": None,
    }


def as_legacy_selection_seal(seal: CapacitySelectionSeal) -> SelectionSeal:
    """Narrow adapter used only after the global Work 9 registry claim."""

    return SelectionSeal(
        protocol_id=seal.coordinates.protocol_id,
        partition_repeat=seal.coordinates.partition_repeat,
        outer_fold=seal.coordinates.outer_fold,
        condition=seal.coordinates.condition,
        seed=seal.coordinates.seed,
        checkpoint=seal.checkpoint,
        validation_predictions_sha256=seal.validation_segment_ledger.sha256,
        threshold=seal.threshold,
    )


def _load_ref(raw: Mapping[str, Any], project_root: Path, *, ledger: bool) -> Path:
    expected = {"path", "sha256", "bytes", "row_count"}
    if set(raw) != expected:
        raise CapacityTrainingError("matrix artifact reference has unexpected keys")
    raw_path = Path(str(raw["path"]))
    if raw_path.is_absolute() or raw_path == Path(".") or ".." in raw_path.parts:
        raise CapacityTrainingError("matrix artifact path must be project-relative and safe")
    root = project_root.resolve()
    lexical_path = root / raw_path
    try:
        _reject_symlink_components(lexical_path, stop=root)
    except ValueError as error:
        raise CapacityTrainingError("matrix artifact path contains a symlink") from error
    path = lexical_path.resolve()
    if root not in path.parents:
        raise CapacityTrainingError("matrix artifact escaped project root")
    _verify_exact_file(path, str(raw["sha256"]), "matrix artifact")
    if path.stat().st_size != int(raw["bytes"]):
        raise CapacityTrainingError("matrix artifact byte count mismatch")
    if ledger:
        if not isinstance(raw["row_count"], int) or isinstance(raw["row_count"], bool):
            raise CapacityTrainingError("ledger artifact row_count is not an integer")
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != raw["row_count"]:
            raise CapacityTrainingError("ledger artifact row_count mismatch")
    elif raw["row_count"] is not None:
        raise CapacityTrainingError("non-ledger matrix artifact has row_count")
    return path


def expected_hybrid_coordinate_contract(
    protocol_id: str,
    seeds: Sequence[int] = (17, 29, 43, 71, 101),
) -> dict[tuple[str, int, int, int], tuple[str, str, str]]:
    """Return the exact preregistered 200-coordinate role/family contract."""

    if tuple(int(seed) for seed in seeds) != (17, 29, 43, 71, 101):
        raise CapacityTrainingError("hybrid sensitivity seeds differ from preregistration")
    result: dict[tuple[str, int, int, int], tuple[str, str, str]] = {}
    for condition in NEW_FIT_CONDITIONS + ANCHOR_CONDITIONS:
        role = (
            "new_capacity_fit"
            if condition in NEW_FIT_CONDITIONS
            else "immutable_primary_anchor"
        )
        family = RUN_FAMILY if condition in NEW_FIT_CONDITIONS else "primary_matrix"
        for outer_fold in range(5):
            for seed in seeds:
                key = (condition, 0, outer_fold, int(seed))
                run_id = (
                    f"{family}__{condition}__r00-o{outer_fold:02d}-s{int(seed)}"
                )
                result[key] = (role, family, run_id)
    if len(result) != 200:
        raise CapacityTrainingError("internal hybrid coordinate contract is not 200")
    return result


def verify_exact_hybrid_entries(
    entries: Any,
    *,
    protocol_id: str,
    seeds: Sequence[int] = (17, 29, 43, 71, 101),
    context: str,
) -> dict[str, int]:
    """Fail unless entries are the unique exact 150-new + 50-anchor matrix."""

    if not isinstance(entries, list) or len(entries) != 200:
        raise CapacityTrainingError(f"{context} must contain exactly 200 entries")
    expected = expected_hybrid_coordinate_contract(protocol_id, seeds)
    observed: dict[tuple[str, int, int, int], Mapping[str, Any]] = {}
    for position, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise CapacityTrainingError(f"{context} entry {position} is not an object")
        try:
            key = (
                str(raw["condition"]),
                int(raw["partition_repeat"]),
                int(raw["outer_fold"]),
                int(raw["seed"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CapacityTrainingError(
                f"{context} entry {position} has invalid coordinates"
            ) from error
        if key in observed:
            raise CapacityTrainingError(f"{context} contains duplicate coordinate {key}")
        observed[key] = raw
        wanted = expected.get(key)
        if wanted is None:
            raise CapacityTrainingError(f"{context} contains unexpected coordinate {key}")
        role, family, run_id = wanted
        actual = (
            raw.get("entry_role"),
            raw.get("source_run_family"),
            raw.get("run_id"),
        )
        if actual != wanted:
            raise CapacityTrainingError(
                f"{context} role/family/run_id mismatch at {key}: "
                f"expected {wanted}, got {actual}"
            )
    missing = set(expected) - set(observed)
    if missing:
        raise CapacityTrainingError(f"{context} is missing coordinates: {sorted(missing)[:3]}")
    new_count = sum(row.get("entry_role") == "new_capacity_fit" for row in entries)
    anchor_count = sum(
        row.get("entry_role") == "immutable_primary_anchor" for row in entries
    )
    if (new_count, anchor_count) != (150, 50):
        raise CapacityTrainingError(f"{context} does not satisfy 150-new/50-anchor roles")
    return {"logical": 200, "new": new_count, "anchors": anchor_count}


def _expected_selection_lineage(frozen: FrozenWork9Inputs) -> dict[str, str]:
    if frozen.runner_acceptance_sha256 is None:
        runner_sha = "0" * 64
    else:
        runner_sha = frozen.runner_acceptance_sha256
    return {
        "config_sha256": WORK8_CONFIG_SHA256,
        "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
        "runner_acceptance_sha256": runner_sha,
        "experiment_card_sha256": frozen.experiment_card_sha256,
        "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
        "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
        "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
        "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
        "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
    }


def verify_capacity_selection_seal_against_frozen(
    frozen: FrozenWork9Inputs,
    seal: CapacitySelectionSeal,
    *,
    run_root: Path,
    record: Mapping[str, Any] | None = None,
) -> None:
    """Strict semantic verifier used at resume, freeze, and outer release.

    JSON Schema establishes shape.  This routine re-derives every scientific
    lineage value, manifest, feature-store identity, model width/count, and
    artifact location from the frozen protocol.
    """

    coordinates = seal.coordinates
    if coordinates.protocol_id != frozen.config.protocol_id:
        raise CapacityTrainingError("selection seal protocol differs from frozen Work 8")
    if coordinates.seed not in frozen.config.seeds:
        raise CapacityTrainingError("selection seal seed differs from frozen Work 8")
    if coordinates.condition not in NEW_FIT_CONDITIONS + ANCHOR_CONDITIONS:
        raise CapacityTrainingError("selection seal condition is outside hybrid matrix")
    expected_lineage = _expected_selection_lineage(frozen)
    mismatches = {
        name: (wanted, getattr(seal, name))
        for name, wanted in expected_lineage.items()
        if getattr(seal, name) != wanted
    }
    if mismatches:
        raise CapacityTrainingError(f"selection seal frozen lineage mismatch: {mismatches}")
    selection = frozen.config.selection_view(
        coordinates.partition_repeat, coordinates.outer_fold
    )
    if (
        seal.train_manifest_sha256 != selection.train_manifest.sha256
        or seal.validation_manifest_sha256 != selection.val_manifest.sha256
    ):
        raise CapacityTrainingError("selection seal train/validation manifest lineage differs")
    expected_store = (
        frozen.config.text_store.metadata.sha256,
        frozen.config.text_store.index_sha256,
        frozen.config.text_store.vectors_sha256,
    )
    actual_store = (
        seal.feature_store_metadata_sha256,
        seal.feature_store_index_sha256,
        seal.feature_store_vectors_sha256,
    )
    if actual_store != expected_store:
        raise CapacityTrainingError("selection seal text-store lineage differs")
    if coordinates.condition != "AT-cross-randtrain" and (
        seal.pairing_plan_sha256 is not None or seal.pairing_plan_artifact is not None
    ):
        raise CapacityTrainingError("non-random selection seal binds a pairing plan")
    seal.verify(run_root)
    expected_names = {
        seal.checkpoint.path: "selection/checkpoint.safetensors",
        seal.validation_segment_ledger.path: "selection/validation_segments.jsonl",
        seal.validation_call_ledger.path: "selection/validation_calls.jsonl",
        seal.epoch_trace.path: "selection/epoch_trace.json",
        seal.threshold_trace.path: "selection/threshold_trace.json",
    }
    for path, relative in expected_names.items():
        if path.resolve() != (run_root / relative).resolve():
            raise CapacityTrainingError(f"selection seal artifact path drift: {relative}")
    if record is None:
        return
    if record.get("run_id") != coordinates.run_id or record.get("status") != "selection_complete":
        raise CapacityTrainingError("selection run record identity/status differs")
    if record.get("phase") != "selection":
        raise CapacityTrainingError("selection run record phase differs")
    record_lineage = record.get("lineage")
    if not isinstance(record_lineage, Mapping) or any(
        record_lineage.get(name) != wanted for name, wanted in expected_lineage.items()
    ):
        raise CapacityTrainingError("selection run record frozen lineage differs")
    data = record.get("data", {})
    if (
        data.get("train_manifest_sha256") != seal.train_manifest_sha256
        or data.get("validation_manifest_sha256") != seal.validation_manifest_sha256
        or data.get("outer_eval_manifest_sha256") is not None
        or data.get("grouping_policy_id") != frozen.config.grouping_policy_id
    ):
        raise CapacityTrainingError("selection run record data lineage differs")
    store = data.get("feature_store_ids", {})
    if (
        store.get("text_feature_set_id") != frozen.config.text_store.feature_set_id
        or store.get("metadata_sha256") != seal.feature_store_metadata_sha256
        or store.get("index_sha256") != seal.feature_store_index_sha256
        or store.get("vectors_sha256") != seal.feature_store_vectors_sha256
        or data.get("mfcc_cache_policy") != CACHE_POLICY
    ):
        raise CapacityTrainingError("selection run record feature lineage differs")
    built = frozen.config.instantiate_smoke_condition(coordinates.condition)
    model = record.get("model", {})
    if (
        model.get("architecture") != built.architecture
        or model.get("total_parameters") != built.total_parameters
        or model.get("trainable_parameters") != built.trainable_parameters
    ):
        raise CapacityTrainingError("selection run record model contract differs")
    selection_record = record.get("selection", {})
    if (
        selection_record.get("selected_epoch") != seal.selected_epoch
        or float(selection_record.get("validation_call_nll", -1.0))
        != seal.validation_call_nll
        or float(selection_record.get("threshold", -1.0)) != seal.threshold
    ):
        raise CapacityTrainingError("selection run record selection values differ from seal")
    training = record.get("training", {})
    optimizer = training.get("optimizer", {})
    scheduler = training.get("scheduler", {})
    if (
        training.get("loss_objective") != "call_equal_fixed_global_scale_v1"
        or optimizer.get("name") != "AdamW"
        or optimizer.get("learning_rate") != frozen.config.training.learning_rate
        or optimizer.get("weight_decay") != frozen.config.training.weight_decay
        or scheduler.get("name") != "ReduceLROnPlateau"
        or scheduler.get("factor") != frozen.config.training.scheduler_factor
        or scheduler.get("patience") != frozen.config.training.scheduler_patience
        or scheduler.get("minimum_learning_rate")
        != frozen.config.training.minimum_learning_rate
    ):
        raise CapacityTrainingError("selection run record training contract differs")
    artifacts = record.get("artifacts", {})
    expected_artifacts = {
        "selection_seal": (run_root / "selection/selection_seal.json").resolve(),
        "checkpoint": seal.checkpoint.path.resolve(),
        "validation_segments": seal.validation_segment_ledger.path.resolve(),
        "validation_calls": seal.validation_call_ledger.path.resolve(),
        "epoch_trace": seal.epoch_trace.path.resolve(),
        "threshold_trace": seal.threshold_trace.path.resolve(),
    }
    for name, expected_path in expected_artifacts.items():
        raw = artifacts.get(name)
        if not isinstance(raw, Mapping):
            raise CapacityTrainingError(f"selection run record lacks {name}")
        if _load_ref(
            raw,
            frozen.project_root,
            ledger=name in {"validation_segments", "validation_calls"},
        ) != expected_path:
            raise CapacityTrainingError(f"selection run record {name} path differs")
    compute = record.get("compute", {})
    if compute.get("primary_environment_lock_sha256") != PRIMARY_ENVIRONMENT_LOCK_SHA256:
        raise CapacityTrainingError("selection run record environment identity differs")
    if str(compute.get("device", "")).startswith("cuda"):
        expected_device = {
            "device_name": EXPECTED_DEVICE_NAME,
            "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
            "torch_version": EXPECTED_TORCH_VERSION,
            "cuda_version": EXPECTED_CUDA_VERSION,
            "cudnn_version": EXPECTED_CUDNN_VERSION,
        }
        if compute.get("device_identity") != expected_device:
            raise CapacityTrainingError("selection run record GPU identity differs")


def verify_primary_outer_anchor_lineage(
    project_root: str | Path,
    *,
    hybrid_selection_index_path: str | Path,
) -> dict[str, Any]:
    """Verify primary outer anchors only after hybrid selection is frozen."""

    root = Path(project_root).resolve()
    selection_path = Path(hybrid_selection_index_path).resolve()
    canonical_selection = (root / REGISTRY_RELATIVE / "_matrix/selection_index.json").resolve()
    if selection_path != canonical_selection or not selection_path.is_file():
        raise CapacityTrainingError(
            "primary outer anchors cannot be opened before canonical hybrid selection freeze"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_json_schema(
        selection,
        project_root=root,
        schema_name="work9_selection_index.schema.json",
    )
    verify_exact_hybrid_entries(
        selection.get("entries"),
        protocol_id=str(selection.get("protocol_id")),
        context="hybrid selection index before primary outer access",
    )
    outer_path = root / PRIMARY_OUTER_INDEX_RELATIVE
    _verify_exact_file(outer_path, PRIMARY_OUTER_INDEX_SHA256, "primary outer index")
    if outer_path.stat().st_mode & 0o222:
        raise CapacityTrainingError("immutable primary outer index is writable")
    outer = json.loads(outer_path.read_text(encoding="utf-8"))
    if (
        outer.get("status") != "frozen_complete"
        or outer.get("run_family") != "primary_matrix"
        or len(outer.get("entries", [])) != 200
    ):
        raise CapacityTrainingError("primary outer index is not frozen-complete")
    expected_keys = {
        (condition, 0, fold, seed)
        for condition in ANCHOR_CONDITIONS
        for fold in range(5)
        for seed in (17, 29, 43, 71, 101)
    }
    rows = {
        (
            str(row["condition"]),
            int(row["partition_repeat"]),
            int(row["outer_fold"]),
            int(row["seed"]),
        ): row
        for row in outer["entries"]
        if str(row["condition"]) in ANCHOR_CONDITIONS
    }
    if set(rows) != expected_keys:
        raise CapacityTrainingError("primary outer anchor coordinate set is not exact")
    for key, row in rows.items():
        wanted_run_id = f"primary_matrix__{key[0]}__r00-o{key[2]:02d}-s{key[3]}"
        if row.get("run_id") != wanted_run_id:
            raise CapacityTrainingError(f"primary outer anchor run_id differs at {key}")
        for field, ledger in (
            ("outer_eval_segment_ledger", True),
            ("outer_eval_call_ledger", True),
            ("outer_eval_metrics", False),
            ("run_record", False),
        ):
            _load_ref(row[field], root, ledger=ledger)
    return {
        "status": "pass",
        "phase": "post_hybrid_selection_freeze_outer_anchor_verification",
        "anchor_coordinate_count": 50,
        "hybrid_selection_index_sha256": sha256_file(selection_path),
        "primary_outer_index_sha256": PRIMARY_OUTER_INDEX_SHA256,
        "primary_outer_artifacts_verified": True,
    }


def freeze_complete_selection_index(
    project_root: str | Path,
    *,
    registry: CapacityRegistry | None = None,
) -> Path:
    """Freeze the 150 new seals plus 50 immutable primary anchors."""

    frozen = verify_frozen_work9_inputs(project_root, require_runner_acceptance=True)
    assert frozen.runner_acceptance_sha256 is not None
    registry = registry or CapacityRegistry(frozen.project_root)
    expected_new = tuple(
        CapacityCoordinates(
            frozen.config.protocol_id, RUN_FAMILY, condition, 0, outer_fold, seed
        )
        for condition in NEW_FIT_CONDITIONS
        for outer_fold in range(5)
        for seed in frozen.config.seeds
    )
    if len(expected_new) != 150:
        raise CapacityTrainingError("capacity family must contain exactly 150 new selections")
    new_entries: dict[tuple[str, int, int], dict[str, Any]] = {}
    for coordinates in expected_new:
        run_root = registry.root / coordinates.run_id
        marker = run_root / "SELECTION_COMPLETE.json"
        seal_path = run_root / "selection/selection_seal.json"
        record_path = run_root / "selection_run_record.json"
        artifact_index_path = run_root / "selection_artifact_index.json"
        for path in (marker, seal_path, record_path, artifact_index_path):
            if not path.is_file():
                raise CapacityTrainingError(
                    f"selection index incomplete at {coordinates.run_id}: {path.name} missing"
                )
        seal_raw = json.loads(seal_path.read_text(encoding="utf-8"))
        validate_json_schema(
            seal_raw,
            project_root=frozen.project_root,
            schema_name=SELECTION_SEAL_SCHEMA,
        )
        seal = load_selection_seal(seal_path, frozen.project_root)
        if seal.coordinates != coordinates:
            raise CapacityTrainingError("selection seal coordinate mismatch")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        validate_json_schema(
            record, project_root=frozen.project_root, schema_name=RUN_RECORD_SCHEMA
        )
        if record.get("status") != "selection_complete":
            raise CapacityTrainingError("selection run record is not complete")
        verify_capacity_selection_seal_against_frozen(
            frozen, seal, run_root=run_root, record=record
        )
        marker_raw = json.loads(marker.read_text(encoding="utf-8"))
        if marker_raw.get("selection_seal_sha256") != sha256_file(seal_path):
            raise CapacityTrainingError("selection completion marker does not bind seal")
        new_entries[(coordinates.condition, coordinates.outer_fold, coordinates.seed)] = {
                "run_id": coordinates.run_id,
                "condition": coordinates.condition,
                "partition_repeat": coordinates.partition_repeat,
                "outer_fold": coordinates.outer_fold,
                "seed": coordinates.seed,
                "entry_role": "new_capacity_fit",
                "source_run_family": RUN_FAMILY,
                "selection_seal": artifact_record(seal_path, frozen.project_root),
                "selection_run_record": artifact_record(
                    record_path, frozen.project_root
                ),
                "epoch_trace": artifact_record(
                    seal.epoch_trace.path, frozen.project_root
                ),
                "threshold_trace": artifact_record(
                    seal.threshold_trace.path, frozen.project_root
                ),
            }

    primary_selection_path = frozen.project_root / PRIMARY_SELECTION_INDEX_RELATIVE
    primary_selection = json.loads(primary_selection_path.read_text(encoding="utf-8"))
    anchor_entries = {
        (str(row["condition"]), int(row["outer_fold"]), int(row["seed"])): row
        for row in primary_selection["entries"]
        if str(row["condition"]) in ANCHOR_CONDITIONS
    }
    if len(anchor_entries) != 50:
        raise CapacityTrainingError("primary selection index has incomplete anchor inventory")
    entries: list[dict[str, Any]] = []
    for condition in frozen.config.conditions:
        for outer_fold in range(5):
            for seed in frozen.config.seeds:
                key = (condition, outer_fold, seed)
                if condition in NEW_FIT_CONDITIONS:
                    entries.append(new_entries[key])
                else:
                    anchor = anchor_entries[key]
                    entries.append(
                        {
                            "run_id": str(anchor["run_id"]),
                            "condition": condition,
                            "partition_repeat": 0,
                            "outer_fold": outer_fold,
                            "seed": seed,
                            "entry_role": "immutable_primary_anchor",
                            "source_run_family": "primary_matrix",
                            "selection_seal": dict(anchor["selection_seal"]),
                            "selection_run_record": dict(anchor["selection_run_record"]),
                            "epoch_trace": dict(anchor["epoch_trace"]),
                            "threshold_trace": dict(anchor["threshold_trace"]),
                        }
                    )
    if len(entries) != 200 or sum(
        row["entry_role"] == "immutable_primary_anchor" for row in entries
    ) != 50:
        raise CapacityTrainingError("hybrid selection index cardinality drifted")
    payload = {
        "schema_version": "paper2-v2-work9-capacity-selection-index-1",
        "protocol_id": frozen.config.protocol_id,
        "run_family": RUN_FAMILY,
        "status": "frozen_complete",
        "expected_coordinate_count": 200,
        "config": artifact_record(
            frozen.project_root / WORK8_CONFIG_RELATIVE, frozen.project_root
        ),
        "work8_receipt": artifact_record(
            frozen.project_root / WORK8_RECEIPT_RELATIVE, frozen.project_root
        ),
        "runner_acceptance": artifact_record(
            frozen.project_root / ACCEPTANCE_RECEIPT_RELATIVE, frozen.project_root
        ),
        "experiment_card": artifact_record(
            frozen.project_root / EXPERIMENT_CARD_RELATIVE, frozen.project_root
        ),
        "anchor_policy": artifact_record(
            frozen.project_root / ANCHOR_POLICY_RELATIVE, frozen.project_root
        ),
        "capacity_analysis_protocol": artifact_record(
            frozen.project_root / CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
            frozen.project_root,
        ),
        "figure_spec": artifact_record(
            frozen.project_root / FIGURE_SPEC_RELATIVE, frozen.project_root
        ),
        "execution_addendum": artifact_record(
            frozen.project_root / EXECUTION_ADDENDUM_RELATIVE,
            frozen.project_root,
        ),
        "primary_runner_acceptance": artifact_record(
            frozen.project_root / PRIMARY_RUNNER_ACCEPTANCE_RELATIVE,
            frozen.project_root,
        ),
        "primary_selection_index": artifact_record(
            primary_selection_path, frozen.project_root
        ),
        "completed_gate1_hash_manifest": artifact_record(
            frozen.project_root / GATE1_HASH_MANIFEST_RELATIVE,
            frozen.project_root,
        ),
        "new_coordinate_count": 150,
        "anchor_coordinate_count": 50,
        "entries": entries,
    }
    validate_json_schema(
        payload,
        project_root=frozen.project_root,
        schema_name="work9_selection_index.schema.json",
    )
    verify_exact_hybrid_entries(
        payload["entries"],
        protocol_id=frozen.config.protocol_id,
        seeds=frozen.config.seeds,
        context="hybrid selection index before freeze",
    )
    reference = registry.write_matrix_artifact_once("selection_index.json", payload)
    return reference.path


def _load_complete_selection_index(
    frozen: FrozenWork9Inputs,
    registry: CapacityRegistry,
) -> tuple[Path, Mapping[str, Any], str]:
    path = registry.root / "_matrix/selection_index.json"
    if not path.is_file():
        raise CapacityTrainingError("complete selection index is not frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_json_schema(
        payload,
        project_root=frozen.project_root,
        schema_name="work9_selection_index.schema.json",
    )
    if payload.get("status") != "frozen_complete" or len(payload.get("entries", [])) != 200:
        raise CapacityTrainingError("selection index is not the complete hybrid matrix")
    verify_exact_hybrid_entries(
        payload.get("entries"),
        protocol_id=frozen.config.protocol_id,
        seeds=frozen.config.seeds,
        context="loaded hybrid selection index",
    )
    expected_refs = {
        "config": (WORK8_CONFIG_RELATIVE, WORK8_CONFIG_SHA256),
        "work8_receipt": (WORK8_RECEIPT_RELATIVE, WORK8_RECEIPT_SHA256),
        "runner_acceptance": (
            ACCEPTANCE_RECEIPT_RELATIVE,
            frozen.runner_acceptance_sha256,
        ),
        "experiment_card": (EXPERIMENT_CARD_RELATIVE, EXPERIMENT_CARD_SHA256),
        "anchor_policy": (ANCHOR_POLICY_RELATIVE, ANCHOR_POLICY_SHA256),
        "capacity_analysis_protocol": (
            CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
            CAPACITY_ANALYSIS_PROTOCOL_SHA256,
        ),
        "figure_spec": (FIGURE_SPEC_RELATIVE, FIGURE_SPEC_SHA256),
        "execution_addendum": (EXECUTION_ADDENDUM_RELATIVE, EXECUTION_ADDENDUM_SHA256),
        "primary_runner_acceptance": (
            PRIMARY_RUNNER_ACCEPTANCE_RELATIVE,
            PRIMARY_RUNNER_ACCEPTANCE_SHA256,
        ),
        "primary_selection_index": (
            PRIMARY_SELECTION_INDEX_RELATIVE,
            PRIMARY_SELECTION_INDEX_SHA256,
        ),
        "completed_gate1_hash_manifest": (
            GATE1_HASH_MANIFEST_RELATIVE,
            GATE1_HASH_MANIFEST_SHA256,
        ),
    }
    for name, (relative, digest) in expected_refs.items():
        reference = payload.get(name)
        if not isinstance(reference, Mapping) or reference.get("sha256") != digest:
            raise CapacityTrainingError(f"selection index {name} digest differs")
        loaded = _load_ref(reference, frozen.project_root, ledger=False)
        if loaded != (frozen.project_root / relative).resolve():
            raise CapacityTrainingError(f"selection index {name} path differs")
    for row in payload["entries"]:
        for name in ("selection_seal", "selection_run_record", "epoch_trace", "threshold_trace"):
            _load_ref(row[name], frozen.project_root, ledger=False)
    return path, payload, sha256_file(path)


def verify_capacity_selection_index_for_outer_release(
    project_root: str | Path,
) -> dict[str, Any]:
    """Public no-outer-read release guard for the complete hybrid selection index."""

    frozen = verify_frozen_work9_inputs(project_root, require_runner_acceptance=True)
    registry = CapacityRegistry(frozen.project_root)
    path, payload, digest = _load_complete_selection_index(frozen, registry)
    return {
        "status": "pass",
        "path": path,
        "sha256": digest,
        "coordinate_count": len(payload["entries"]),
        "primary_outer_index_opened_or_hashed": False,
    }


@dataclass(frozen=True)
class OuterEvaluationResult:
    workspace: CapacityRunWorkspace
    segment_ledger_path: Path
    call_ledger_path: Path
    metrics_path: Path
    run_record_path: Path


def verify_outer_run_record_before_terminal(
    *,
    frozen: FrozenWork9Inputs,
    coordinates: CapacityCoordinates,
    built: Any,
    outer_data: Work9FeatureDataset,
    outer_manifest_sha256: str,
    seal: CapacitySelectionSeal,
    seal_path: Path,
    outer_refs: Mapping[str, Any],
    segment_count: int,
    call_count: int,
    device: torch.device,
    record: Mapping[str, Any],
) -> None:
    """Re-derive a crash-resumed outer record before terminal publication.

    A schema-valid ``run_record.json`` is not sufficient evidence after a
    crash: every frozen lineage, data/model/training/selection field and every
    artifact reference must still describe the currently released coordinate.
    Only non-scientific runtime telemetry is accepted from the existing
    record, after strict type/range checks.
    """

    compute = record.get("compute")
    if not isinstance(compute, Mapping):
        raise CapacityTrainingError("outer run record compute block is invalid")
    wall_time = compute.get("wall_time_seconds")
    if (
        isinstance(wall_time, bool)
        or not isinstance(wall_time, (int, float))
        or not math.isfinite(float(wall_time))
        or float(wall_time) < 0.0
    ):
        raise CapacityTrainingError("outer run record wall-time telemetry is invalid")
    telemetry: dict[str, int | float] = {"wall_time_seconds": wall_time}
    for name in (
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "host_max_rss_kib",
    ):
        value = compute.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CapacityTrainingError(
                f"outer run record {name} telemetry is invalid"
            )
        telemetry[name] = value

    expected = _outer_run_record(
        frozen=frozen,
        coordinates=coordinates,
        built=built,
        outer_data=outer_data,
        outer_manifest_sha256=outer_manifest_sha256,
        seal=seal,
        seal_path=seal_path,
        outer_refs=outer_refs,
        segment_count=segment_count,
        call_count=call_count,
        device=device,
        wall_time_seconds=wall_time,
    )
    # These values describe runtime cost rather than scientific lineage.  They
    # cannot be reconstructed after a process crash, but they must remain
    # finite/non-negative and all other compute identity is re-derived above.
    expected_compute = expected["compute"]
    for name, value in telemetry.items():
        expected_compute[name] = value
    if dict(record) != expected:
        changed = sorted(
            key
            for key in set(record) | set(expected)
            if record.get(key) != expected.get(key)
        )
        raise CapacityTrainingError(
            "existing outer run record fails pre-terminal semantic "
            f"rederivation: {changed}"
        )


def _verify_completed_outer_run(
    frozen: FrozenWork9Inputs,
    workspace: CapacityRunWorkspace,
) -> OuterEvaluationResult:
    terminal_path = workspace.path / "TERMINAL.json"
    artifact_index_path = workspace.path / "artifact_index.json"
    run_record_path = workspace.path / "run_record.json"
    authorization_path = workspace.path / "outer_authorization.json"
    if not all(
        path.is_file()
        for path in (
            terminal_path,
            artifact_index_path,
            run_record_path,
            authorization_path,
        )
    ):
        raise CapacityTrainingError("outer terminal chain is incomplete")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "outer_complete":
        raise CapacityTrainingError("outer terminal status is not complete")
    if terminal.get("run_id") != workspace.coordinates.run_id:
        raise CapacityTrainingError("outer terminal run_id mismatch")
    if terminal.get("artifact_index_sha256") != sha256_file(artifact_index_path):
        raise CapacityTrainingError("outer terminal does not bind artifact index")
    if terminal.get("outer_authorization_sha256") != sha256_file(authorization_path):
        raise CapacityTrainingError("outer terminal does not bind authorization")
    index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    validate_json_schema(
        index,
        project_root=frozen.project_root,
        schema_name=RUN_ARTIFACT_INDEX_SCHEMA,
    )
    if index.get("run_id") != workspace.coordinates.run_id:
        raise CapacityTrainingError("outer artifact index run_id mismatch")
    for key in (
        "selection_index",
        "selection_seal",
        "global_claim",
        "outer_authorization",
        "checkpoint",
        "epoch_trace",
        "threshold_trace",
        "outer_eval_metrics",
        "run_record",
    ):
        _load_ref(index[key], frozen.project_root, ledger=False)
    if _load_ref(
        index["outer_authorization"], frozen.project_root, ledger=False
    ) != authorization_path:
        raise CapacityTrainingError("artifact index points to another outer authorization")
    claim_path = _load_ref(index["global_claim"], frozen.project_root, ledger=False)
    expected_claim_path = (
        workspace.path.parent
        / "_outer_eval_claims"
        / f"{workspace.coordinates.claim_key}.json"
    ).resolve()
    if claim_path != expected_claim_path:
        raise CapacityTrainingError("artifact index global claim path differs")
    if terminal.get("global_claim_sha256") != sha256_file(claim_path):
        raise CapacityTrainingError("outer terminal does not bind global claim")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    expected_identity = {
        "run_id": workspace.coordinates.run_id,
        **asdict(workspace.coordinates),
    }
    if any(authorization.get(key) != value for key, value in expected_identity.items()):
        raise CapacityTrainingError("outer authorization coordinate lineage differs")
    if any(claim.get(key) != value for key, value in expected_identity.items()):
        raise CapacityTrainingError("global claim coordinate lineage differs")
    if authorization.get("global_claim", {}).get("sha256") != sha256_file(claim_path):
        raise CapacityTrainingError("outer authorization does not bind global claim")
    for name, ledger in (
        ("selection_index", False),
        ("selection_seal", False),
        ("global_claim", False),
    ):
        _load_ref(authorization[name], frozen.project_root, ledger=ledger)
    if authorization.get("selection_index") != index.get("selection_index"):
        raise CapacityTrainingError("authorization and artifact index selection indexes differ")
    if authorization.get("selection_seal") != index.get("selection_seal"):
        raise CapacityTrainingError("authorization and artifact index selection seals differ")
    if authorization.get("global_claim") != index.get("global_claim"):
        raise CapacityTrainingError("authorization and artifact index claims differ")
    if (
        claim.get("selection_index_sha256") != index["selection_index"]["sha256"]
        or claim.get("selection_seal_sha256") != index["selection_seal"]["sha256"]
        or claim.get("checkpoint_sha256") != index["checkpoint"]["sha256"]
        or authorization.get("checkpoint_sha256") != index["checkpoint"]["sha256"]
    ):
        raise CapacityTrainingError("claim/authorization artifact lineage differs")
    segment_path = _load_ref(
        index["outer_eval_segments"], frozen.project_root, ledger=True
    )
    call_path = _load_ref(index["outer_eval_calls"], frozen.project_root, ledger=True)
    if _load_ref(index["run_record"], frozen.project_root, ledger=False) != run_record_path:
        raise CapacityTrainingError("outer artifact index points to another run record")
    record = json.loads(run_record_path.read_text(encoding="utf-8"))
    validate_json_schema(
        record, project_root=frozen.project_root, schema_name=RUN_RECORD_SCHEMA
    )
    if record.get("status") != "outer_complete":
        raise CapacityTrainingError("outer run record is not complete")
    if (
        record.get("run_id") != workspace.coordinates.run_id
        or any(
            record.get(name) != value
            for name, value in asdict(workspace.coordinates).items()
        )
        or record.get("phase") != "outer_eval"
    ):
        raise CapacityTrainingError("outer run record coordinate/phase differs")
    expected_lineage = _expected_selection_lineage(frozen)
    if any(
        record.get("lineage", {}).get(name) != value
        for name, value in expected_lineage.items()
    ):
        raise CapacityTrainingError("outer run record frozen lineage differs")
    seal_path = _load_ref(index["selection_seal"], frozen.project_root, ledger=False)
    selection_record_path = workspace.path / "selection_run_record.json"
    if not selection_record_path.is_file():
        raise CapacityTrainingError("outer terminal lacks original selection run record")
    seal = load_selection_seal(seal_path, frozen.project_root)
    selection_record = json.loads(selection_record_path.read_text(encoding="utf-8"))
    validate_json_schema(
        selection_record,
        project_root=frozen.project_root,
        schema_name=RUN_RECORD_SCHEMA,
    )
    verify_capacity_selection_seal_against_frozen(
        frozen,
        seal,
        run_root=workspace.path,
        record=selection_record,
    )
    data = record.get("data", {})
    if (
        data.get("train_manifest_sha256") != seal.train_manifest_sha256
        or data.get("validation_manifest_sha256") != seal.validation_manifest_sha256
        or not isinstance(data.get("outer_eval_manifest_sha256"), str)
        or data.get("grouping_policy_id") != frozen.config.grouping_policy_id
    ):
        raise CapacityTrainingError("outer run record data lineage differs")
    expected_device = {
        "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "torch_version": EXPECTED_TORCH_VERSION,
        "cuda_version": EXPECTED_CUDA_VERSION,
        "cudnn_version": EXPECTED_CUDNN_VERSION,
    }
    if (
        record.get("compute", {}).get("device_identity") != expected_device
        or record.get("compute", {}).get("primary_environment_lock_sha256")
        != PRIMARY_ENVIRONMENT_LOCK_SHA256
    ):
        raise CapacityTrainingError("outer run record device/environment identity differs")
    return OuterEvaluationResult(
        workspace,
        segment_path,
        call_path,
        _load_ref(index["outer_eval_metrics"], frozen.project_root, ledger=False),
        run_record_path,
    )


def outer_evaluate_reserved_coordinate(
    project_root: str | Path,
    coordinates: CapacityCoordinates,
    *,
    device: str,
    registry: CapacityRegistry | None = None,
) -> OuterEvaluationResult:
    """Evaluate one already-selected coordinate after matrix-wide release.

    This is an internal Phase-B primitive. The public CLI exposes only the
    matrix driver, never a coordinate debug release.
    """

    frozen = verify_frozen_work9_inputs(project_root, require_runner_acceptance=True)
    verify_cuda_device_identity(device)
    registry = registry or CapacityRegistry(frozen.project_root)
    if coordinates.condition not in NEW_FIT_CONDITIONS:
        raise CapacityTrainingError("immutable primary anchors cannot receive new outer access")
    if registry.root != (frozen.project_root / REGISTRY_RELATIVE).resolve():
        raise CapacityTrainingError("production outer evaluation requires canonical capacity registry")
    seed_everything(coordinates.seed, device)
    outer_started = time.monotonic()
    reset_cuda_peak_memory_stats(device)
    index_path, selection_index, index_sha = _load_complete_selection_index(
        frozen, registry
    )
    entry_matches = [
        row
        for row in selection_index["entries"]
        if (
            row["condition"],
            int(row["partition_repeat"]),
            int(row["outer_fold"]),
            int(row["seed"]),
        )
        == (
            coordinates.condition,
            coordinates.partition_repeat,
            coordinates.outer_fold,
            coordinates.seed,
        )
    ]
    if len(entry_matches) != 1:
        raise CapacityTrainingError("coordinate is not uniquely released by selection index")
    entry = entry_matches[0]
    if (
        entry.get("entry_role") != "new_capacity_fit"
        or entry.get("source_run_family") != RUN_FAMILY
    ):
        raise CapacityTrainingError("immutable primary anchors cannot receive new outer access")
    seal_path = _load_ref(entry["selection_seal"], frozen.project_root, ledger=False)
    seal = load_selection_seal(seal_path, frozen.project_root)
    if seal.coordinates != coordinates:
        raise CapacityTrainingError("released selection seal coordinates differ")
    selection_record_path = registry.root / coordinates.run_id / "selection_run_record.json"
    if not selection_record_path.is_file():
        raise CapacityTrainingError("released selection run record is missing")
    selection_record = json.loads(selection_record_path.read_text(encoding="utf-8"))
    validate_json_schema(
        selection_record,
        project_root=frozen.project_root,
        schema_name=RUN_RECORD_SCHEMA,
    )
    verify_capacity_selection_seal_against_frozen(
        frozen,
        seal,
        run_root=registry.root / coordinates.run_id,
        record=selection_record,
    )
    workspace = registry.reserve(coordinates)
    with workspace.lock():
        final_record = workspace.path / "run_record.json"
        if final_record.exists() and (workspace.path / "TERMINAL.json").exists():
            return _verify_completed_outer_run(frozen, workspace)
        # Global claim happens before authorizing or opening the outer view.
        claim_ref = registry.claim_outer(
            workspace,
            selection_seal_sha256=sha256_file(seal_path),
            selection_index_sha256=index_sha,
            checkpoint_sha256=seal.checkpoint.sha256,
        )
        authorization = {
            "schema_version": "paper2-v2-work9-capacity-outer-authorization-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "selection_index": artifact_record(index_path, frozen.project_root),
            "selection_seal": artifact_record(seal_path, frozen.project_root),
            "global_claim": artifact_record(claim_ref.path, frozen.project_root),
            "checkpoint_sha256": seal.checkpoint.sha256,
            "resume_policy": "same_run_same_selection_only",
        }
        authorization_ref = workspace.write_json_or_verify(
            "outer_authorization.json", authorization
        )

        outer_view = frozen.config.authorize_outer_evaluation(
            as_legacy_selection_seal(seal)
        )
        outer_view.outer_eval_manifest.verify()
        built = frozen.config.instantiate_condition(coordinates.condition)
        store = (
            IndexedTextStore(
                frozen.config.text_store.metadata.path.parent, verify_hashes=True
            )
            if "text" in built.input_spec.global_features
            else None
        )
        outer_data = Work9FeatureDataset(
            outer_view.outer_eval_manifest.path,
            input_spec=built.input_spec,
            project_root=frozen.project_root,
            text_store=store,
            pairing_plan=None,
            cache_mfcc=True,
        )
        outer_directory = workspace.path / "outer_eval"
        if outer_directory.exists():
            actual_files = {
                path.relative_to(outer_directory).as_posix()
                for path in outer_directory.rglob("*")
                if path.is_file()
            }
            if actual_files != {"segments.jsonl", "calls.jsonl", "metrics.json"} or any(
                path.is_symlink() for path in outer_directory.rglob("*")
            ):
                raise CapacityTrainingError("partial outer ledger directory is invalid")
            segment_path = outer_directory / "segments.jsonl"
            call_path = outer_directory / "calls.jsonl"
            metrics_path = outer_directory / "metrics.json"
            segment_rows = _load_manifest_rows(segment_path)
            call_rows = _load_manifest_rows(call_path)
            validate_jsonl_schema(
                segment_rows,
                project_root=frozen.project_root,
                schema_name=SEGMENT_PREDICTION_SCHEMA,
            )
            validate_jsonl_schema(
                call_rows,
                project_root=frozen.project_root,
                schema_name=CALL_PREDICTION_SCHEMA,
            )
            predictions = [
                SegmentPrediction(
                    segment_id=str(row["segment_id"]),
                    call_id=str(row["call_id"]),
                    group_id=str(row["group_id"]),
                    source=str(row["source"]),
                    label=int(row["label"]),
                    probability=float(row["probability"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                )
                for row in segment_rows
            ]
            calls = aggregate_mean_probability(predictions)
            expected_segment_rows, expected_call_rows = _prediction_rows(
                predictions,
                calls,
                coordinates=coordinates,
                partition="outer_eval",
                threshold=seal.threshold,
                checkpoint_sha256=seal.checkpoint.sha256,
                manifest_sha256=outer_view.outer_eval_manifest.sha256,
            )
            if segment_rows != expected_segment_rows or call_rows != expected_call_rows:
                raise CapacityTrainingError("resumed outer ledgers fail semantic recomputation")
            metrics = compute_call_metrics(calls, threshold=seal.threshold)
            expected_metrics_payload = {
                "schema_version": "paper2-v2-work9-capacity-outer-metrics-1",
                "run_id": coordinates.run_id,
                **asdict(coordinates),
                "partition": "outer_eval",
                "aggregation_rule": "mean_probability",
                "threshold": seal.threshold,
                "checkpoint_sha256": seal.checkpoint.sha256,
                "outer_eval_manifest_sha256": outer_view.outer_eval_manifest.sha256,
                "metrics": metrics,
                "interpretation": "within_corpus_source_confounded_outer_development_evaluation",
            }
            if json.loads(metrics_path.read_text(encoding="utf-8")) != expected_metrics_payload:
                raise CapacityTrainingError("resumed outer metrics fail recomputation")
            outer_refs = {
                "segments.jsonl": ArtifactRef(segment_path, sha256_file(segment_path)),
                "calls.jsonl": ArtifactRef(call_path, sha256_file(call_path)),
                "metrics.json": ArtifactRef(metrics_path, sha256_file(metrics_path)),
            }
        else:
            model = built.model.to(torch.device(device))
            _load_model_bytes(model, seal.checkpoint.path.read_bytes())
            predictions = _infer_probabilities(
                model,
                outer_data,
                batch_size=frozen.config.training.batch_size,
                device=torch.device(device),
            )
            calls = aggregate_mean_probability(predictions)
            segment_rows, call_rows = _prediction_rows(
                predictions,
                calls,
                coordinates=coordinates,
                partition="outer_eval",
                threshold=seal.threshold,
                checkpoint_sha256=seal.checkpoint.sha256,
                manifest_sha256=outer_view.outer_eval_manifest.sha256,
            )
            validate_jsonl_schema(
                segment_rows,
                project_root=frozen.project_root,
                schema_name=SEGMENT_PREDICTION_SCHEMA,
            )
            validate_jsonl_schema(
                call_rows,
                project_root=frozen.project_root,
                schema_name=CALL_PREDICTION_SCHEMA,
            )
            metrics = compute_call_metrics(calls, threshold=seal.threshold)
            metrics_payload = {
                "schema_version": "paper2-v2-work9-capacity-outer-metrics-1",
                "run_id": coordinates.run_id,
                **asdict(coordinates),
                "partition": "outer_eval",
                "aggregation_rule": "mean_probability",
                "threshold": seal.threshold,
                "checkpoint_sha256": seal.checkpoint.sha256,
                "outer_eval_manifest_sha256": outer_view.outer_eval_manifest.sha256,
                "metrics": metrics,
                "interpretation": "within_corpus_source_confounded_outer_development_evaluation",
            }
            outer_refs = workspace.publish_directory_once(
                "outer_eval",
                {
                    "segments.jsonl": canonical_jsonl_bytes(segment_rows),
                    "calls.jsonl": canonical_jsonl_bytes(call_rows),
                    "metrics.json": canonical_json_bytes(metrics_payload),
                },
            )
        outer_record_path = workspace.path / "run_record.json"
        if outer_record_path.exists():
            outer_run_record = json.loads(outer_record_path.read_text(encoding="utf-8"))
            validate_json_schema(
                outer_run_record,
                project_root=frozen.project_root,
                schema_name=RUN_RECORD_SCHEMA,
            )
            verify_outer_run_record_before_terminal(
                frozen=frozen,
                coordinates=coordinates,
                built=built,
                outer_data=outer_data,
                outer_manifest_sha256=outer_view.outer_eval_manifest.sha256,
                seal=seal,
                seal_path=seal_path,
                outer_refs=outer_refs,
                segment_count=len(segment_rows),
                call_count=len(call_rows),
                device=torch.device(device),
                record=outer_run_record,
            )
            record_ref = ArtifactRef(
                outer_record_path.resolve(), sha256_file(outer_record_path)
            )
        else:
            outer_run_record = _outer_run_record(
                frozen=frozen,
                coordinates=coordinates,
                built=built,
                outer_data=outer_data,
                outer_manifest_sha256=outer_view.outer_eval_manifest.sha256,
                seal=seal,
                seal_path=seal_path,
                outer_refs=outer_refs,
                segment_count=len(segment_rows),
                call_count=len(call_rows),
                device=torch.device(device),
                wall_time_seconds=time.monotonic() - outer_started,
            )
            validate_json_schema(
                outer_run_record,
                project_root=frozen.project_root,
                schema_name=RUN_RECORD_SCHEMA,
            )
            verify_outer_run_record_before_terminal(
                frozen=frozen,
                coordinates=coordinates,
                built=built,
                outer_data=outer_data,
                outer_manifest_sha256=outer_view.outer_eval_manifest.sha256,
                seal=seal,
                seal_path=seal_path,
                outer_refs=outer_refs,
                segment_count=len(segment_rows),
                call_count=len(call_rows),
                device=torch.device(device),
                record=outer_run_record,
            )
            record_ref = workspace.write_json_once(
                "run_record.json", outer_run_record
            )
        artifact_index = {
            "schema_version": "paper2-v2-work9-capacity-run-artifact-index-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "selection_index": artifact_record(index_path, frozen.project_root),
            "selection_seal": artifact_record(seal_path, frozen.project_root),
            "global_claim": artifact_record(claim_ref.path, frozen.project_root),
            "outer_authorization": artifact_record(
                authorization_ref.path, frozen.project_root
            ),
            "checkpoint": artifact_record(seal.checkpoint.path, frozen.project_root),
            "epoch_trace": artifact_record(seal.epoch_trace.path, frozen.project_root),
            "threshold_trace": artifact_record(
                seal.threshold_trace.path, frozen.project_root
            ),
            "outer_eval_segments": artifact_record(
                outer_refs["segments.jsonl"].path,
                frozen.project_root,
                row_count=len(segment_rows),
            ),
            "outer_eval_calls": artifact_record(
                outer_refs["calls.jsonl"].path,
                frozen.project_root,
                row_count=len(call_rows),
            ),
            "outer_eval_metrics": artifact_record(
                outer_refs["metrics.json"].path, frozen.project_root
            ),
            "run_record": artifact_record(record_ref.path, frozen.project_root),
        }
        validate_json_schema(
            artifact_index,
            project_root=frozen.project_root,
            schema_name=RUN_ARTIFACT_INDEX_SCHEMA,
        )
        index_ref = workspace.write_json_or_verify(
            "artifact_index.json", artifact_index
        )
        workspace.write_json_or_verify(
            "TERMINAL.json",
            {
                "schema_version": "paper2-v2-work9-capacity-terminal-1",
                "run_id": coordinates.run_id,
                **asdict(coordinates),
                "status": "outer_complete",
                "artifact_index_sha256": index_ref.sha256,
                "global_claim_sha256": claim_ref.sha256,
                "outer_authorization_sha256": authorization_ref.sha256,
            },
        )
        return OuterEvaluationResult(
            workspace,
            outer_refs["segments.jsonl"].path,
            outer_refs["calls.jsonl"].path,
            outer_refs["metrics.json"].path,
            record_ref.path,
        )


def _outer_run_record(
    *,
    frozen: FrozenWork9Inputs,
    coordinates: CapacityCoordinates,
    built: Any,
    outer_data: Work9FeatureDataset,
    outer_manifest_sha256: str,
    seal: CapacitySelectionSeal,
    seal_path: Path,
    outer_refs: Mapping[str, Any],
    segment_count: int,
    call_count: int,
    device: torch.device,
    wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": "paper2-v2-work9-capacity-run-record-1",
        "run_id": coordinates.run_id,
        **asdict(coordinates),
        "phase": "outer_eval",
        "status": "outer_complete",
        "lineage": {
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": frozen.runner_acceptance_sha256,
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
            "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "primary_runner_acceptance_sha256": PRIMARY_RUNNER_ACCEPTANCE_SHA256,
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
        },
        "data": {
            "train_manifest_sha256": seal.train_manifest_sha256,
            "validation_manifest_sha256": seal.validation_manifest_sha256,
            "outer_eval_manifest_sha256": outer_manifest_sha256,
            "grouping_policy_id": frozen.config.grouping_policy_id,
            "feature_store_ids": {
                "text_feature_set_id": frozen.config.text_store.feature_set_id,
                "metadata_sha256": frozen.config.text_store.metadata.sha256,
                "index_sha256": frozen.config.text_store.index_sha256,
                "vectors_sha256": frozen.config.text_store.vectors_sha256,
            },
            "mfcc_cache_policy": CACHE_POLICY,
            "mfcc_cache_sha256": {
                "train": None,
                "validation": None,
                "outer_eval": outer_data.cache_content_sha256,
            },
        },
        "model": {
            "architecture": built.architecture,
            "total_parameters": built.total_parameters,
            "trainable_parameters": built.trainable_parameters,
        },
        "randomness": {
            "seed": coordinates.seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            **DETERMINISM_POLICY,
        },
        "training": {
            "optimizer": {
                **OPTIMIZER_POLICY,
                "learning_rate": frozen.config.training.learning_rate,
                "weight_decay": frozen.config.training.weight_decay,
            },
            "scheduler": {
                **SCHEDULER_POLICY,
                "factor": frozen.config.training.scheduler_factor,
                "patience": frozen.config.training.scheduler_patience,
                "minimum_learning_rate": frozen.config.training.minimum_learning_rate,
            },
            "loss_objective": "call_equal_fixed_global_scale_v1",
            "fit_phase_complete_before_outer_release": True,
        },
        "selection": {
            "selected_epoch": seal.selected_epoch,
            "validation_call_nll": seal.validation_call_nll,
            "threshold": seal.threshold,
        },
        "artifacts": {
            "selection_seal": artifact_record(seal_path, frozen.project_root),
            "checkpoint": artifact_record(seal.checkpoint.path, frozen.project_root),
            "outer_eval_segments": artifact_record(
                outer_refs["segments.jsonl"].path,
                frozen.project_root,
                row_count=segment_count,
            ),
            "outer_eval_calls": artifact_record(
                outer_refs["calls.jsonl"].path,
                frozen.project_root,
                row_count=call_count,
            ),
            "outer_eval_metrics": artifact_record(
                outer_refs["metrics.json"].path, frozen.project_root
            ),
        },
        "compute": {
            "device": str(device),
            "device_identity": verify_cuda_device_identity(device),
            "primary_environment_lock_sha256": PRIMARY_ENVIRONMENT_LOCK_SHA256,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "wall_time_seconds": wall_time_seconds,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
            ),
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "failure": None,
    }


def freeze_complete_outer_ledger_index(
    project_root: str | Path,
    *,
    registry: CapacityRegistry | None = None,
) -> Path:
    frozen = verify_frozen_work9_inputs(project_root, require_runner_acceptance=True)
    registry = registry or CapacityRegistry(frozen.project_root)
    selection_path, selection_index, _ = _load_complete_selection_index(frozen, registry)
    new_outer_entries: dict[tuple[str, int, int], dict[str, Any]] = {}
    for selected in selection_index["entries"]:
        if selected["entry_role"] != "new_capacity_fit":
            continue
        coordinates = CapacityCoordinates(
            frozen.config.protocol_id,
            RUN_FAMILY,
            str(selected["condition"]),
            int(selected["partition_repeat"]),
            int(selected["outer_fold"]),
            int(selected["seed"]),
        )
        run_root = registry.root / coordinates.run_id
        if not run_root.is_dir():
            raise CapacityTrainingError(
                f"outer matrix incomplete at coordinate {coordinates.run_id}"
            )
        completed = _verify_completed_outer_run(
            frozen, CapacityRunWorkspace(run_root, coordinates)
        )
        record_path = completed.run_record_path
        segment_path = completed.segment_ledger_path
        call_path = completed.call_ledger_path
        metrics_path = completed.metrics_path
        seal_path = _load_ref(
            selected["selection_seal"], frozen.project_root, ledger=False
        )
        seal = load_selection_seal(seal_path, frozen.project_root)
        new_outer_entries[(coordinates.condition, coordinates.outer_fold, coordinates.seed)] = {
                "run_id": coordinates.run_id,
                "condition": coordinates.condition,
                "partition_repeat": coordinates.partition_repeat,
                "outer_fold": coordinates.outer_fold,
                "seed": coordinates.seed,
                "entry_role": "new_capacity_fit",
                "source_run_family": RUN_FAMILY,
                "threshold": seal.threshold,
                "checkpoint_sha256": seal.checkpoint.sha256,
                "outer_eval_segment_ledger": artifact_record(
                    segment_path,
                    frozen.project_root,
                    row_count=sum(1 for line in segment_path.open() if line.strip()),
                ),
                "outer_eval_call_ledger": artifact_record(
                    call_path,
                    frozen.project_root,
                    row_count=sum(1 for line in call_path.open() if line.strip()),
                ),
                "outer_eval_metrics": artifact_record(
                    metrics_path, frozen.project_root
                ),
                "run_record": artifact_record(record_path, frozen.project_root),
            }
    if len(new_outer_entries) != 150:
        raise CapacityTrainingError("capacity outer inventory is not 150 complete entries")

    verify_primary_outer_anchor_lineage(
        frozen.project_root,
        hybrid_selection_index_path=selection_path,
    )
    primary_outer_path = frozen.project_root / PRIMARY_OUTER_INDEX_RELATIVE
    primary_outer = json.loads(primary_outer_path.read_text(encoding="utf-8"))
    anchor_outer_entries = {
        (str(row["condition"]), int(row["outer_fold"]), int(row["seed"])): row
        for row in primary_outer["entries"]
        if str(row["condition"]) in ANCHOR_CONDITIONS
    }
    if len(anchor_outer_entries) != 50:
        raise CapacityTrainingError("primary outer index has incomplete anchor inventory")
    entries: list[dict[str, Any]] = []
    for condition in frozen.config.conditions:
        for outer_fold in range(5):
            for seed in frozen.config.seeds:
                key = (condition, outer_fold, seed)
                if condition in NEW_FIT_CONDITIONS:
                    entries.append(new_outer_entries[key])
                else:
                    anchor = anchor_outer_entries[key]
                    entries.append(
                        {
                            **dict(anchor),
                            "entry_role": "immutable_primary_anchor",
                            "source_run_family": "primary_matrix",
                        }
                    )
    if len(entries) != 200 or sum(
        row["entry_role"] == "immutable_primary_anchor" for row in entries
    ) != 50:
        raise CapacityTrainingError("hybrid outer index cardinality drifted")
    verify_exact_hybrid_entries(
        entries,
        protocol_id=frozen.config.protocol_id,
        seeds=frozen.config.seeds,
        context="hybrid outer index before freeze",
    )
    payload = {
        "schema_version": "paper2-v2-work9-capacity-outer-ledger-index-1",
        "protocol_id": frozen.config.protocol_id,
        "run_family": RUN_FAMILY,
        "status": "frozen_complete",
        "selection_index": artifact_record(selection_path, frozen.project_root),
        "runner_acceptance": artifact_record(
            frozen.project_root / ACCEPTANCE_RECEIPT_RELATIVE,
            frozen.project_root,
        ),
        "capacity_analysis_protocol": artifact_record(
            frozen.project_root / CAPACITY_ANALYSIS_PROTOCOL_RELATIVE,
            frozen.project_root,
        ),
        "figure_spec": artifact_record(
            frozen.project_root / FIGURE_SPEC_RELATIVE, frozen.project_root
        ),
        "execution_addendum": artifact_record(
            frozen.project_root / EXECUTION_ADDENDUM_RELATIVE,
            frozen.project_root,
        ),
        "primary_runner_acceptance": artifact_record(
            frozen.project_root / PRIMARY_RUNNER_ACCEPTANCE_RELATIVE,
            frozen.project_root,
        ),
        "primary_outer_index": artifact_record(
            primary_outer_path, frozen.project_root
        ),
        "completed_gate1_hash_manifest": artifact_record(
            frozen.project_root / GATE1_HASH_MANIFEST_RELATIVE,
            frozen.project_root,
        ),
        "expected_coordinate_count": 200,
        "new_coordinate_count": 150,
        "anchor_coordinate_count": 50,
        "entries": entries,
    }
    validate_json_schema(
        payload,
        project_root=frozen.project_root,
        schema_name="work9_outer_ledger_index.schema.json",
    )
    return registry.write_matrix_artifact_once("outer_ledger_index.json", payload).path


def _complete_call_smoke_rows(
    manifest_path: Path, *, maximum_segments: int = 96
) -> list[dict[str, Any]]:
    rows = _load_manifest_rows(manifest_path)
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_call[str(row["call_id"])].append(row)
    selected_calls: list[str] = []
    total = 0
    for label in (0, 1):
        candidates = sorted(
            call_id
            for call_id, call_rows in by_call.items()
            if int(call_rows[0]["label"]) == label
        )
        if not candidates:
            raise CapacityTrainingError("smoke manifest lacks both labels")
        chosen = min(candidates, key=lambda value: (len(by_call[value]), value))
        selected_calls.append(chosen)
        total += len(by_call[chosen])
    for call_id in sorted(by_call):
        if call_id in selected_calls:
            continue
        size = len(by_call[call_id])
        if total + size > maximum_segments:
            continue
        selected_calls.append(call_id)
        total += size
    selected = [row for row in rows if str(row["call_id"]) in set(selected_calls)]
    if not selected or {int(row["label"]) for row in selected} != {0, 1}:
        raise CapacityTrainingError("smoke call selection failed")
    return selected


def run_cuda_smoke(
    project_root: str | Path,
    *,
    output_directory: str | Path,
    device: str = "cuda:0",
    condition: str,
) -> Path:
    """Run a deterministic, train/validation-only real-feature CUDA smoke.

    This is deliberately not a primary run and never constructs an outer view.
    The caller provides a new output directory so two independent invocations
    can be compared before freezing the runner acceptance receipt.
    """

    if not str(device).startswith("cuda"):
        raise CapacityTrainingError("acceptance smoke must run on CUDA")
    seed = 17
    frozen = verify_frozen_work9_inputs(
        project_root, require_runner_acceptance=False
    )
    if condition not in {"A-mean", "AT-cross-randtrain"}:
        raise CapacityTrainingError("smoke must cover a frozen acceptance branch")
    device_identity = verify_cuda_device_identity(device)
    from .work9_capacity_acceptance import environment_lock, verify_draft_against_current

    acceptance_draft, acceptance_draft_sha = verify_draft_against_current(
        frozen.project_root
    )
    pre_smoke_environment = environment_lock()
    pre_smoke_sys_path = list(sys.path)
    try:
        output = validate_isolated_relative_root(
            frozen.project_root,
            output_directory,
            allowed_parent=SMOKE_EVIDENCE_ROOT_RELATIVE,
            context="acceptance smoke output",
            require_new=True,
        )
    except (ValueError, FileExistsError) as error:
        raise CapacityTrainingError(str(error)) from error
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch_device = torch.device(device)
    reset_cuda_peak_memory_stats(torch_device)
    coordinates = CapacityCoordinates(
        frozen.config.protocol_id, RUN_FAMILY, condition, 0, 0, seed
    )
    selection = frozen.config.selection_view(0, 0)
    # Retain a small cached/uncached equivalence benchmark, but the acceptance
    # training itself below uses the complete real train/validation manifests.
    train_rows = _complete_call_smoke_rows(selection.train_manifest.path)
    cache_manifest = output / "cache_equivalence_manifest.jsonl"
    from .work9_capacity_registry import atomic_write_once

    atomic_write_once(cache_manifest, canonical_jsonl_bytes(train_rows))
    built = frozen.config.instantiate_smoke_condition(coordinates.condition)
    expected_contract = {
        "A-mean": (("mfcc",), ()),
        "AT-cross-randtrain": (("mfcc",), ("text",)),
    }[condition]
    if (
        built.input_spec.sequence_features,
        built.input_spec.global_features,
    ) != expected_contract:
        raise CapacityTrainingError(f"{condition} smoke branch contract drifted")
    smoke_store = (
        IndexedTextStore(frozen.config.text_store.metadata.path.parent, verify_hashes=True)
        if "text" in built.input_spec.global_features
        else None
    )

    cache_started = time.monotonic()
    cached_subset = Work9FeatureDataset(
        cache_manifest,
        input_spec=built.input_spec,
        project_root=frozen.project_root,
        text_store=smoke_store,
        pairing_plan=None,
        cache_mfcc=True,
    )
    cached_construction_seconds = time.monotonic() - cache_started
    uncached = Work9FeatureDataset(
        cache_manifest,
        input_spec=built.input_spec,
        project_root=frozen.project_root,
        text_store=smoke_store,
        pairing_plan=None,
        cache_mfcc=False,
    )
    uncached_started = time.monotonic()
    for index in range(len(uncached)):
        if not torch.equal(
            uncached[index].sequences["mfcc"], cached_subset[index].sequences["mfcc"]
        ):
            raise CapacityTrainingError("MFCC cache changed feature semantics")
    uncached_iteration_seconds = time.monotonic() - uncached_started
    cached_started = time.monotonic()
    for index in range(len(cached_subset)):
        _ = cached_subset[index].sequences["mfcc"]
    cached_iteration_seconds = time.monotonic() - cached_started
    smoke_registry = CapacityRegistry(
        frozen.project_root,
        root=(output / "selection_registry").relative_to(frozen.project_root),
        allow_test_root=True,
    )
    result = fit_select_coordinate(
        frozen.project_root,
        coordinates,
        device=device,
        require_runner_acceptance=False,
        registry=smoke_registry,
        epoch_budget=1,
    )
    result.seal.verify(result.workspace.path)
    seal_raw = json.loads(result.selection_seal_path.read_text(encoding="utf-8"))
    validate_json_schema(
        seal_raw,
        project_root=frozen.project_root,
        schema_name=SELECTION_SEAL_SCHEMA,
    )
    record_raw = json.loads(result.selection_run_record_path.read_text(encoding="utf-8"))
    validate_json_schema(
        record_raw,
        project_root=frozen.project_root,
        schema_name=RUN_RECORD_SCHEMA,
    )
    for required in (
        "selection/epoch_trace.json",
        "selection/threshold_trace.json",
        "selection/validation_segments.jsonl",
        "selection/validation_calls.jsonl",
        "selection_artifact_index.json",
        "SELECTION_COMPLETE.json",
    ):
        if not (result.workspace.path / required).is_file():
            raise CapacityTrainingError(f"acceptance smoke selection chain missing {required}")
    if (result.workspace.path / "outer_eval").exists() or (
        result.workspace.path / "outer_authorization.json"
    ).exists():
        raise CapacityTrainingError("acceptance smoke accessed outer_eval")
    _, post_smoke_draft_sha = verify_draft_against_current(frozen.project_root)
    post_smoke_environment = environment_lock()
    post_smoke_sys_path = list(sys.path)
    if post_smoke_draft_sha != acceptance_draft_sha:
        raise CapacityTrainingError("runner acceptance DRAFT changed during CUDA smoke")
    if post_smoke_environment != pre_smoke_environment:
        raise CapacityTrainingError("fixed-root environment lock changed during CUDA smoke")
    semantic = {
        "schema_version": CUDA_SMOKE_SEMANTIC_SCHEMA_VERSION,
        "config_sha256": WORK8_CONFIG_SHA256,
        "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
        "experiment_card_sha256": frozen.experiment_card_sha256,
        "anchor_policy_sha256": ANCHOR_POLICY_SHA256,
        "capacity_analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
        "runner_acceptance_draft_sha256": acceptance_draft_sha,
        "bundle_digests": acceptance_draft["bundle_digests"],
        "environment_lock_sha256": pre_smoke_environment[
            "environment_lock_sha256"
        ],
        "condition": coordinates.condition,
        "hidden_dim": frozen.config.capacity_sensitivity.matched_hidden_dims[condition],
        "parameter_count": frozen.config.capacity_sensitivity.matched_parameter_counts[condition],
        "seed": seed,
        "partition_repeat": 0,
        "outer_fold": 0,
        "epochs": 1,
        "train_manifest_sha256": selection.train_manifest.sha256,
        "validation_manifest_sha256": selection.val_manifest.sha256,
        "train_segment_count": len(_load_manifest_rows(selection.train_manifest.path)),
        "validation_segment_count": len(
            _load_manifest_rows(selection.val_manifest.path)
        ),
        "mfcc_cache_policy": CACHE_POLICY,
        "train_mfcc_cache_sha256": record_raw["data"]["mfcc_cache_sha256"]["train"],
        "validation_mfcc_cache_sha256": record_raw["data"]["mfcc_cache_sha256"][
            "validation"
        ],
        "model_checkpoint_sha256": result.seal.checkpoint.sha256,
        "validation_segment_ledger_sha256": result.seal.validation_segment_ledger.sha256,
        "validation_call_ledger_sha256": result.seal.validation_call_ledger.sha256,
        "selected_epoch": result.seal.selected_epoch,
        "validation_call_nll": result.seal.validation_call_nll,
        "threshold": result.seal.threshold,
        "selection_chain_verified": True,
        "outer_eval_accessed": False,
        "text_store_opened": smoke_store is not None,
    }
    semantic_bytes = canonical_json_bytes(semantic)
    semantic_ref = atomic_write_once(output / "semantic_output.json", semantic_bytes)
    receipt = {
        "schema_version": CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "condition": condition,
        "device_type": "cuda",
        "device": str(torch_device),
        **device_identity,
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "runner_acceptance_draft_sha256": acceptance_draft_sha,
        "bundle_digests": acceptance_draft["bundle_digests"],
        "environment_lock_sha256": pre_smoke_environment[
            "environment_lock_sha256"
        ],
        "environment_packages_sha256": pre_smoke_environment["packages_sha256"],
        "environment_inventory_method": pre_smoke_environment["inventory_method"],
        "environment_distribution_roots": pre_smoke_environment[
            "distribution_roots"
        ],
        "runtime_sys_path": {
            "before": pre_smoke_sys_path,
            "after": post_smoke_sys_path,
            "before_sha256": sha256_bytes(canonical_json_bytes(pre_smoke_sys_path)),
            "after_sha256": sha256_bytes(canonical_json_bytes(post_smoke_sys_path)),
            "changed": pre_smoke_sys_path != post_smoke_sys_path,
            "excluded_from_environment_inventory": True,
        },
        "semantic_output_sha256": semantic_ref.sha256,
        "semantic_output_bytes": semantic_ref.path.stat().st_size,
        "wall_time_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(torch_device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(torch_device),
        "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cache_benchmark": {
            "cached_construction_seconds": cached_construction_seconds,
            "cached_iteration_seconds": cached_iteration_seconds,
            "uncached_iteration_seconds": uncached_iteration_seconds,
            "semantic_equality": True,
            "cache_bytes": cached_subset.cache_bytes,
        },
        "safety": {
            "outer_eval_accessed": False,
            "primary_selection_registry_read_only_verified": True,
            "primary_outer_registry_accessed": False,
            "primary_registry_modified": False,
            "scientific_selection_claimed": False,
        },
    }
    receipt_ref = atomic_write_once(
        output / "smoke_receipt.json", canonical_json_bytes(receipt)
    )
    return receipt_ref.path


def record_coordinate_failure(
    project_root: str | Path,
    coordinates: CapacityCoordinates,
    *,
    phase: str,
    error: BaseException,
) -> Path:
    """Append a diagnostic without burning a resumable primary coordinate."""

    if coordinates.condition not in NEW_FIT_CONDITIONS:
        raise CapacityTrainingError(
            "failure records are restricted to the 150 new capacity coordinates"
        )
    registry = CapacityRegistry(project_root)
    workspace = registry.reserve(coordinates)
    reference = workspace.record_failure(
        {
            "schema_version": "paper2-v2-work9-capacity-failure-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "phase": phase,
            "error_type": type(error).__name__,
            "message": str(error),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "resumable_only_with_same_run_and_selection": True,
        }
    )
    return reference.path
