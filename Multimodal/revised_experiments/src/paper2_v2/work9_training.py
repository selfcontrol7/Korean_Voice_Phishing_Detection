"""Deterministic two-phase training coordinator for the Work 9 primary matrix.

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

from .conditions import ConditionId
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
from .work9_registry import Work9Registry, Work9RunWorkspace
from .work9_types import (
    ACCEPTANCE_RECEIPT_RELATIVE,
    ACCEPTANCE_SCHEMA_VERSION,
    CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
    CUDA_SMOKE_SEMANTIC_SCHEMA_VERSION,
    EXPERIMENT_CARD_RELATIVE,
    GATE1_ANALYSIS_PROTOCOL_RELATIVE,
    GATE1_ANALYSIS_PROTOCOL_SHA256,
    GATE1_POLICY_RELATIVE,
    GATE1_POLICY_SHA256,
    RUN_FAMILY,
    SUPERSEDED_ACCEPTANCE_GENERATIONS,
    WORK8_CONFIG_RELATIVE,
    WORK8_CONFIG_SHA256,
    WORK8_RECEIPT_RELATIVE,
    WORK8_RECEIPT_SHA256,
    Work9Coordinates,
    Work9SelectionSeal,
    artifact_record,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_selection_seal,
    sha256_bytes,
)


SCHEMA_ROOT_RELATIVE = Path("revised_experiments/registry/schemas/work9")
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


class Work9TrainingError(RuntimeError):
    """Fail-closed coordinator error."""


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
        raise Work9TrainingError(
            f"{schema_name} validation failed: {'; '.join(messages)}"
        )


def validate_jsonl_schema(
    rows: Sequence[Mapping[str, Any]], *, project_root: Path, schema_name: str
) -> None:
    if not rows:
        raise Work9TrainingError(f"{schema_name} ledger is empty")
    validator = _compiled_schema_validator(str(project_root.resolve()), schema_name)
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
        if errors:
            messages = [f"{list(error.path)}: {error.message}" for error in errors[:10]]
            raise Work9TrainingError(
                f"ledger row {index} fails {schema_name}: {'; '.join(messages)}"
            )


@dataclass(frozen=True)
class FrozenWork9Inputs:
    project_root: Path
    config: MatrixConfig
    work8_receipt: Mapping[str, Any]
    runner_acceptance: Mapping[str, Any] | None
    runner_acceptance_sha256: str | None
    experiment_card_sha256: str


def _verify_exact_file(path: Path, expected_sha256: str, context: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{context} missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise Work9TrainingError(
            f"{context} hash mismatch: expected {expected_sha256}, got {actual}"
        )


def _verify_ref_mapping(
    raw: Mapping[str, Any], project_root: Path, *, context: str
) -> Path:
    if not {"path", "sha256"} <= set(raw):
        raise Work9TrainingError(f"{context} is missing path/sha256")
    allowed_shapes = (
        {"path", "sha256"},
        {"path", "sha256", "bytes"},
        {"path", "sha256", "bytes", "row_count"},
    )
    if set(raw) not in allowed_shapes:
        raise Work9TrainingError(f"{context} has unexpected reference keys")
    raw_path = Path(str(raw["path"]))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise Work9TrainingError(f"{context} path must be safe and project-relative")
    path = (project_root / raw_path).resolve()
    if project_root.resolve() not in path.parents:
        raise Work9TrainingError(f"{context} path escaped project root")
    _verify_exact_file(path, str(raw["sha256"]), context)
    if "bytes" in raw and int(raw["bytes"]) != path.stat().st_size:
        raise Work9TrainingError(f"{context} byte count mismatch")
    return path


def _verify_runner_acceptance_environment(acceptance: Mapping[str, Any]) -> str:
    """Recompute and bind the accepted, smoke, and unit-test environment."""

    from .work9_acceptance import environment_lock, verify_environment_lock_mapping

    embedded = acceptance.get("environment_lock")
    try:
        embedded_sha = verify_environment_lock_mapping(embedded)
    except Exception as error:
        raise Work9TrainingError("runner acceptance environment lock is invalid") from error
    current = environment_lock()
    if embedded != current:
        raise Work9TrainingError("current environment differs from runner acceptance")
    smokes = acceptance.get("cuda_smokes")
    if not isinstance(smokes, list) or len(smokes) != 2 or any(
        not isinstance(item, Mapping)
        or item.get("environment_lock_sha256") != embedded_sha
        for item in smokes
    ):
        raise Work9TrainingError("CUDA smokes do not bind accepted environment")
    unit_tests = acceptance.get("unit_tests")
    if not isinstance(unit_tests, Mapping) or unit_tests.get(
        "environment_lock_sha256"
    ) != embedded_sha:
        raise Work9TrainingError("unit tests do not bind accepted environment")
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
    gate1_path = root / GATE1_POLICY_RELATIVE
    gate1_analysis_path = root / GATE1_ANALYSIS_PROTOCOL_RELATIVE
    card_path = root / EXPERIMENT_CARD_RELATIVE
    _verify_exact_file(config_path, WORK8_CONFIG_SHA256, "frozen Work 8 config")
    _verify_exact_file(receipt_path, WORK8_RECEIPT_SHA256, "frozen Work 8 receipt")
    _verify_exact_file(gate1_path, GATE1_POLICY_SHA256, "frozen Gate 1 policy")
    _verify_exact_file(
        gate1_analysis_path,
        GATE1_ANALYSIS_PROTOCOL_SHA256,
        "frozen Gate 1 analysis protocol",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass" or receipt.get("safety", {}).get(
        "training_launched"
    ) is not False:
        raise Work9TrainingError("Work 8 receipt is not the passing no-training receipt")
    for index, source in enumerate(receipt.get("source_contracts", [])):
        path = _verify_ref_mapping(source, root, context=f"Work 8 source {index}")
        if int(source.get("bytes", -1)) != path.stat().st_size:
            raise Work9TrainingError("Work 8 source byte count mismatch")

    # Parsing is intentionally outer-blind. Work 8 already sealed and hashed
    # the full tree; Work 9 Phase A re-verifies only non-outer inputs here.
    config = load_matrix_config(
        config_path,
        project_root=root,
        verify_files=False,
        verify_manifest_isolation=False,
    )
    for artifact in (
        config.gate0_receipt,
        config.exclusion_inventory,
        config.fold_protocol_artifact,
        config.fold_assignments_artifact,
        config.fold_checksums_artifact,
    ):
        artifact.verify()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card.get("status") != "frozen_before_work9_model_outcomes":
        raise Work9TrainingError("Work 9 experiment card is not complete")
    if card.get("protocol_id") != config.protocol_id or card.get("run_family") != RUN_FAMILY:
        raise Work9TrainingError("experiment card coordinates differ from Work 8")
    if card.get("source_label_limitation", "").find("AIHub") < 0:
        raise Work9TrainingError("experiment card source limitation is malformed")
    if card.get("work8_config", {}).get("sha256") != WORK8_CONFIG_SHA256:
        raise Work9TrainingError("experiment card does not bind the Work 8 config")
    if card.get("work8_receipt", {}).get("sha256") != WORK8_RECEIPT_SHA256:
        raise Work9TrainingError("experiment card does not bind the Work 8 receipt")
    if card.get("decision_policy", {}).get("sha256") != GATE1_POLICY_SHA256:
        raise Work9TrainingError("experiment card does not bind Gate 1 policy")
    card_sha = sha256_file(card_path)

    acceptance_path = root / ACCEPTANCE_RECEIPT_RELATIVE
    acceptance: Mapping[str, Any] | None = None
    acceptance_sha: str | None = None
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        acceptance_sha = sha256_file(acceptance_path)
        if acceptance.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION:
            raise Work9TrainingError("runner acceptance schema is unsupported")
        if acceptance.get("status") != "accepted_after_two_cuda_smokes":
            raise Work9TrainingError("runner acceptance is not final")
        supersedes = acceptance.get("supersedes", {})
        if set(supersedes) != {"current_acceptance_generation", "generations"}:
            raise Work9TrainingError("runner acceptance supersession keys changed")
        if supersedes.get("current_acceptance_generation") != "work9_runner_acceptance_v3":
            raise Work9TrainingError("runner acceptance generation is not v3")
        generations = supersedes.get("generations")
        if not isinstance(generations, list) or len(generations) != len(
            SUPERSEDED_ACCEPTANCE_GENERATIONS
        ):
            raise Work9TrainingError("runner acceptance supersession history is incomplete")
        for raw_generation, expected in zip(
            generations, SUPERSEDED_ACCEPTANCE_GENERATIONS, strict=True
        ):
            if not isinstance(raw_generation, Mapping) or set(raw_generation) != {
                "acceptance_generation",
                "draft",
                "failure",
                "reason",
            }:
                raise Work9TrainingError("runner acceptance generation record is malformed")
            generation, draft_relative, draft_sha, failure_relative, failure_sha, reason = expected
            if (
                raw_generation["acceptance_generation"] != generation
                or raw_generation["reason"] != reason
            ):
                raise Work9TrainingError("runner acceptance generation identity changed")
            draft_path = _verify_ref_mapping(
                raw_generation["draft"], root, context=f"superseded {generation} DRAFT"
            )
            failure_path = _verify_ref_mapping(
                raw_generation["failure"],
                root,
                context=f"superseded {generation} failure",
            )
            if (
                draft_path != (root / draft_relative).resolve()
                or raw_generation["draft"].get("sha256") != draft_sha
                or failure_path != (root / failure_relative).resolve()
                or raw_generation["failure"].get("sha256") != failure_sha
            ):
                raise Work9TrainingError(
                    f"runner acceptance points to other {generation} evidence"
                )
            if (draft_path.stat().st_mode | failure_path.stat().st_mode) & 0o222:
                raise Work9TrainingError(
                    f"superseded {generation} evidence is writable"
                )
        expected_lineage = {
            "work8_config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "experiment_card_sha256": card_sha,
            "gate1_policy_sha256": GATE1_POLICY_SHA256,
            "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
        }
        mismatches = {
            key: (wanted, acceptance.get(key))
            for key, wanted in expected_lineage.items()
            if acceptance.get(key) != wanted
        }
        if mismatches:
            raise Work9TrainingError(f"runner acceptance lineage mismatch: {mismatches}")
        accepted_analysis_path = _verify_ref_mapping(
            acceptance.get("gate1_analysis_protocol", {}),
            root,
            context="Gate 1 analysis protocol",
        )
        if accepted_analysis_path != gate1_analysis_path.resolve():
            raise Work9TrainingError("runner acceptance points to another analysis protocol")
        for index, source in enumerate(acceptance.get("source_contracts", [])):
            _verify_ref_mapping(source, root, context=f"Work 9 accepted source {index}")
        from .work9_acceptance import bundle_digests, source_contracts

        if acceptance.get("source_contracts") != source_contracts(root):
            raise Work9TrainingError("runner acceptance source inventory is incomplete")
        if acceptance.get("bundle_digests") != bundle_digests(root):
            raise Work9TrainingError("runner source/schema/test bundle changed after acceptance")
        _verify_runner_acceptance_environment(acceptance)
        smokes = acceptance.get("cuda_smokes", [])
        if len(smokes) != 2 or any(item.get("device_type") != "cuda" for item in smokes):
            raise Work9TrainingError("runner acceptance requires exactly two CUDA smokes")
        semantic_hashes = {item.get("semantic_output_sha256") for item in smokes}
        if len(semantic_hashes) != 1 or None in semantic_hashes:
            raise Work9TrainingError("CUDA smoke semantic outputs did not reproduce")
        if any(
            item.get("cublas_workspace_config") != ":4096:8" for item in smokes
        ):
            raise Work9TrainingError("CUDA smoke receipt lacks deterministic cuBLAS config")
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
            raise Work9TrainingError("runner acceptance has no passing unit-test receipt")
        _verify_ref_mapping(
            unit_tests.get("receipt", {}), root, context="unit-test receipt"
        )
        _verify_ref_mapping(unit_tests.get("log", {}), root, context="unit-test log")
    elif require_runner_acceptance:
        raise Work9TrainingError(
            "production is locked: final runner acceptance requires two real CUDA smokes"
        )

    return FrozenWork9Inputs(root, config, receipt, acceptance, acceptance_sha, card_sha)


def require_process_determinism(seed: int, device: str | torch.device) -> None:
    """Fail before CUDA work if process-level determinism was not pre-established."""

    declared = str(device)
    expected_hash_seed = str(seed)
    if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
        raise Work9TrainingError(
            f"PYTHONHASHSEED must equal coordinate seed {expected_hash_seed} before process start"
        )
    if declared.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise Work9TrainingError(
            "CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch"
        )


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
            raise Work9TrainingError("CUDA requested but unavailable")
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
    return save_safetensors(state, metadata={"format": "paper2-v2-work9-safetensors-1"})


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


def _coordinate_fields(coordinates: Work9Coordinates, *, partition: str) -> dict[str, Any]:
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
                raise Work9TrainingError("inference produced invalid probabilities")
            for segment_id, probability in zip(
                batch.segment_ids, probabilities.tolist(), strict=True
            ):
                if segment_id in probability_by_segment:
                    raise Work9TrainingError("duplicate segment during inference")
                probability_by_segment[segment_id] = float(probability)
    if set(probability_by_segment) != set(dataset.segment_ids):
        raise Work9TrainingError("inference did not cover the complete manifest")
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
    coordinates: Work9Coordinates,
    epoch: int,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "paper2-v2-work9-validation-segment-score-1",
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
    coordinates: Work9Coordinates,
    epoch: int,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "paper2-v2-work9-validation-call-score-1",
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
    coordinates: Work9Coordinates,
    partition: str,
    threshold: float,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = _coordinate_fields(coordinates, partition=partition)
    segments = [
        {
            "schema_version": "paper2-v2-work9-segment-prediction-1",
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
            "schema_version": "paper2-v2-work9-call-prediction-1",
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
    coordinates: Work9Coordinates,
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
        "schema_version": "paper2-v2-work9-threshold-trace-1",
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
    receipt: Mapping[str, Any], coordinates: Work9Coordinates, train_sha256: str
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
        raise Work9TrainingError("Work 8 receipt has no unique pairing plan for coordinate")
    return str(matches[0]["plan_sha256"])


def _build_pairing(
    coordinates: Work9Coordinates,
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
            raise Work9TrainingError("random-train condition did not produce a plan")
        expected = _expected_pairing_sha(receipt, coordinates, train_sha256)
        if plan.sha256 != expected:
            raise Work9TrainingError(
                f"regenerated random-pairing digest differs: {plan.sha256} != {expected}"
            )
    elif plan is not None:
        raise Work9TrainingError("non-random condition unexpectedly received pairing plan")
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
        "schema_version": "paper2-v2-work9-training-state-1",
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
    workspace: Work9RunWorkspace
    seal: Work9SelectionSeal
    selection_seal_path: Path
    selection_run_record_path: Path


def _verify_completed_selection_run(
    frozen: FrozenWork9Inputs,
    workspace: Work9RunWorkspace,
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
        raise Work9TrainingError("completed selection seal coordinate mismatch")
    seal.verify(workspace.path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    validate_json_schema(
        record, project_root=frozen.project_root, schema_name=RUN_RECORD_SCHEMA
    )
    if record.get("status") != "selection_complete":
        raise Work9TrainingError("selection run record is not complete")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("run_id") != workspace.coordinates.run_id
        or marker.get("selection_seal_sha256") != sha256_file(seal_path)
        or marker.get("outer_eval_accessed") is not False
    ):
        raise Work9TrainingError("selection completion marker lineage mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    validate_json_schema(
        index,
        project_root=frozen.project_root,
        schema_name=SELECTION_ARTIFACT_INDEX_SCHEMA,
    )
    if index.get("run_id") != workspace.coordinates.run_id:
        raise Work9TrainingError("selection artifact index run_id mismatch")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Work9TrainingError("selection artifact inventory is malformed")
    if _load_ref(artifacts["selection_seal"], frozen.project_root, ledger=False) != seal_path:
        raise Work9TrainingError("selection artifact index points to another seal")
    if _load_ref(
        artifacts["selection_run_record"], frozen.project_root, ledger=False
    ) != record_path:
        raise Work9TrainingError("selection artifact index points to another run record")
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
    coordinates: Work9Coordinates,
    *,
    device: str,
    require_runner_acceptance: bool = True,
    registry: Work9Registry | None = None,
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
        raise Work9TrainingError("coordinate is outside frozen Work 8 config")
    if epoch_budget is not None and require_runner_acceptance:
        raise Work9TrainingError("production fit cannot override the frozen epoch budget")
    total_epochs = config.training.epoch_budget if epoch_budget is None else int(epoch_budget)
    if total_epochs <= 0:
        raise ValueError("epoch budget must be positive")
    generator = seed_everything(coordinates.seed, device)
    device_value = torch.device(device)
    invocation_started = time.monotonic()
    reset_cuda_peak_memory_stats(device_value)
    registry = registry or Work9Registry(frozen.project_root)
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
            raise Work9TrainingError("selection view exposed outer_eval")
        train_rows = _load_manifest_rows(selection.train_manifest.path)
        plan = _build_pairing(
            coordinates,
            train_rows,
            receipt=frozen.work8_receipt,
            train_sha256=selection.train_manifest.sha256,
        )
        built = config.instantiate_condition(coordinates.condition)
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
                        "schema_version": "paper2-v2-work9-pairing-semantic-1",
                        "label_policy": "unconstrained_global_random",
                        "donor_scope": "same_fold_train_manifest",
                        "plan_sha256": plan.sha256,
                        "ledger_sha256": plan_ref.sha256,
                    },
                )
        run_plan = {
            "schema_version": "paper2-v2-work9-run-plan-1",
            **asdict(coordinates),
            "run_id": coordinates.run_id,
            "phase": "fit_select",
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": frozen.runner_acceptance_sha256,
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "gate1_policy_sha256": GATE1_POLICY_SHA256,
            "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
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
                raise Work9TrainingError("epoch resume history is noncontiguous")
            for record in records:
                trace = record.get("trace_entry", {})
                if int(trace.get("epoch", -1)) != int(record["epoch"]):
                    raise Work9TrainingError("epoch record/trace epoch mismatch")
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
                    raise Work9TrainingError("resumed epoch artifact escaped reserved run")
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
                    raise Work9TrainingError("training repeated a segment in one epoch")
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
                    raise Work9TrainingError("non-finite training loss")
                batch_loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise Work9TrainingError("non-finite gradient")
                optimizer.step()
                weighted_loss_sum += math.fsum(
                    (batch_device.sample_weights * element_losses)
                    .detach()
                    .to(torch.float64)
                    .cpu()
                    .tolist()
                )
            if seen != set(train_data.segment_ids):
                raise Work9TrainingError("training epoch did not visit every segment exactly once")
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
                "schema_version": "paper2-v2-work9-epoch-record-1",
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
            raise Work9TrainingError("fit selection produced no complete epochs")
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
            raise Work9TrainingError("selected checkpoint validation NLL did not reproduce")
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
            "schema_version": "paper2-v2-work9-epoch-trace-1",
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
            "schema_version": "paper2-v2-work9-selection-seal-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": frozen.runner_acceptance_sha256,
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "gate1_policy_sha256": GATE1_POLICY_SHA256,
            "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
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
                raise Work9TrainingError("existing selection run record lineage differs")
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
        inventory = {
            "schema_version": "paper2-v2-work9-selection-artifact-index-1",
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
                "schema_version": "paper2-v2-work9-selection-complete-1",
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
    coordinates: Work9Coordinates,
    workspace: Work9RunWorkspace,
    built: Any,
    train_data: Work9FeatureDataset,
    validation_data: Work9FeatureDataset,
    selection: Any,
    plan: PairingPlan | None,
    plan_ref: Any,
    seal: Work9SelectionSeal,
    total_epochs: int,
    device: torch.device,
    wall_time_seconds: float,
    invocation_wall_time_seconds: float,
    peak_cuda_allocated_bytes: int,
    peak_cuda_reserved_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": "paper2-v2-work9-run-record-1",
        "run_id": coordinates.run_id,
        **asdict(coordinates),
        "phase": "selection",
        "status": "selection_complete",
        "lineage": {
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": frozen.runner_acceptance_sha256,
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "gate1_policy_sha256": GATE1_POLICY_SHA256,
            "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
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


def as_legacy_selection_seal(seal: Work9SelectionSeal) -> SelectionSeal:
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
        raise Work9TrainingError("matrix artifact reference has unexpected keys")
    raw_path = Path(str(raw["path"]))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise Work9TrainingError("matrix artifact path must be project-relative and safe")
    root = project_root.resolve()
    path = (root / raw_path).resolve()
    if root not in path.parents:
        raise Work9TrainingError("matrix artifact escaped project root")
    _verify_exact_file(path, str(raw["sha256"]), "matrix artifact")
    if path.stat().st_size != int(raw["bytes"]):
        raise Work9TrainingError("matrix artifact byte count mismatch")
    if ledger:
        if not isinstance(raw["row_count"], int) or isinstance(raw["row_count"], bool):
            raise Work9TrainingError("ledger artifact row_count is not an integer")
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != raw["row_count"]:
            raise Work9TrainingError("ledger artifact row_count mismatch")
    elif raw["row_count"] is not None:
        raise Work9TrainingError("non-ledger matrix artifact has row_count")
    return path


def freeze_complete_selection_index(
    project_root: str | Path,
    *,
    registry: Work9Registry | None = None,
) -> Path:
    """Freeze Phase-A release only when all 200 selection seals validate."""

    frozen = verify_frozen_work9_inputs(project_root, require_runner_acceptance=True)
    assert frozen.runner_acceptance_sha256 is not None
    registry = registry or Work9Registry(frozen.project_root)
    expected = tuple(
        Work9Coordinates(
            frozen.config.protocol_id, RUN_FAMILY, condition, 0, outer_fold, seed
        )
        for condition in frozen.config.conditions
        for outer_fold in range(5)
        for seed in frozen.config.seeds
    )
    if len(expected) != 200:
        raise Work9TrainingError("frozen primary matrix is not exactly 200 coordinates")
    entries = []
    for coordinates in expected:
        run_root = registry.root / coordinates.run_id
        marker = run_root / "SELECTION_COMPLETE.json"
        seal_path = run_root / "selection/selection_seal.json"
        record_path = run_root / "selection_run_record.json"
        artifact_index_path = run_root / "selection_artifact_index.json"
        for path in (marker, seal_path, record_path, artifact_index_path):
            if not path.is_file():
                raise Work9TrainingError(
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
            raise Work9TrainingError("selection seal coordinate mismatch")
        if seal.runner_acceptance_sha256 != frozen.runner_acceptance_sha256:
            raise Work9TrainingError("selection seal uses another runner acceptance")
        if seal.experiment_card_sha256 != frozen.experiment_card_sha256:
            raise Work9TrainingError("selection seal uses another experiment card")
        seal.verify(run_root)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        validate_json_schema(
            record, project_root=frozen.project_root, schema_name=RUN_RECORD_SCHEMA
        )
        if record.get("status") != "selection_complete":
            raise Work9TrainingError("selection run record is not complete")
        marker_raw = json.loads(marker.read_text(encoding="utf-8"))
        if marker_raw.get("selection_seal_sha256") != sha256_file(seal_path):
            raise Work9TrainingError("selection completion marker does not bind seal")
        entries.append(
            {
                "run_id": coordinates.run_id,
                "condition": coordinates.condition,
                "partition_repeat": coordinates.partition_repeat,
                "outer_fold": coordinates.outer_fold,
                "seed": coordinates.seed,
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
        )
    payload = {
        "schema_version": "paper2-v2-work9-selection-index-1",
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
        "gate1_policy": artifact_record(
            frozen.project_root / GATE1_POLICY_RELATIVE, frozen.project_root
        ),
        "gate1_analysis_protocol": artifact_record(
            frozen.project_root / GATE1_ANALYSIS_PROTOCOL_RELATIVE,
            frozen.project_root,
        ),
        "entries": entries,
    }
    validate_json_schema(
        payload,
        project_root=frozen.project_root,
        schema_name="work9_selection_index.schema.json",
    )
    reference = registry.write_matrix_artifact_once("selection_index.json", payload)
    return reference.path


def _load_complete_selection_index(
    frozen: FrozenWork9Inputs,
    registry: Work9Registry,
) -> tuple[Path, Mapping[str, Any], str]:
    path = registry.root / "_matrix/selection_index.json"
    if not path.is_file():
        raise Work9TrainingError("complete selection index is not frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_json_schema(
        payload,
        project_root=frozen.project_root,
        schema_name="work9_selection_index.schema.json",
    )
    if payload.get("status") != "frozen_complete" or len(payload.get("entries", [])) != 200:
        raise Work9TrainingError("selection index is not the complete primary matrix")
    acceptance_ref = payload["runner_acceptance"]
    if acceptance_ref["sha256"] != frozen.runner_acceptance_sha256:
        raise Work9TrainingError("selection index acceptance differs from current freeze")
    _load_ref(acceptance_ref, frozen.project_root, ledger=False)
    analysis_ref = payload["gate1_analysis_protocol"]
    if analysis_ref["sha256"] != GATE1_ANALYSIS_PROTOCOL_SHA256:
        raise Work9TrainingError("selection index analysis protocol differs")
    _load_ref(analysis_ref, frozen.project_root, ledger=False)
    return path, payload, sha256_file(path)


@dataclass(frozen=True)
class OuterEvaluationResult:
    workspace: Work9RunWorkspace
    segment_ledger_path: Path
    call_ledger_path: Path
    metrics_path: Path
    run_record_path: Path


def _verify_completed_outer_run(
    frozen: FrozenWork9Inputs,
    workspace: Work9RunWorkspace,
) -> OuterEvaluationResult:
    terminal_path = workspace.path / "TERMINAL.json"
    artifact_index_path = workspace.path / "artifact_index.json"
    run_record_path = workspace.path / "run_record.json"
    if not all(path.is_file() for path in (terminal_path, artifact_index_path, run_record_path)):
        raise Work9TrainingError("outer terminal chain is incomplete")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "outer_complete":
        raise Work9TrainingError("outer terminal status is not complete")
    if terminal.get("run_id") != workspace.coordinates.run_id:
        raise Work9TrainingError("outer terminal run_id mismatch")
    if terminal.get("artifact_index_sha256") != sha256_file(artifact_index_path):
        raise Work9TrainingError("outer terminal does not bind artifact index")
    index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    validate_json_schema(
        index,
        project_root=frozen.project_root,
        schema_name=RUN_ARTIFACT_INDEX_SCHEMA,
    )
    if index.get("run_id") != workspace.coordinates.run_id:
        raise Work9TrainingError("outer artifact index run_id mismatch")
    for key in (
        "selection_index",
        "selection_seal",
        "checkpoint",
        "epoch_trace",
        "threshold_trace",
        "outer_eval_metrics",
        "run_record",
    ):
        _load_ref(index[key], frozen.project_root, ledger=False)
    segment_path = _load_ref(
        index["outer_eval_segments"], frozen.project_root, ledger=True
    )
    call_path = _load_ref(index["outer_eval_calls"], frozen.project_root, ledger=True)
    if _load_ref(index["run_record"], frozen.project_root, ledger=False) != run_record_path:
        raise Work9TrainingError("outer artifact index points to another run record")
    record = json.loads(run_record_path.read_text(encoding="utf-8"))
    validate_json_schema(
        record, project_root=frozen.project_root, schema_name=RUN_RECORD_SCHEMA
    )
    if record.get("status") != "outer_complete":
        raise Work9TrainingError("outer run record is not complete")
    return OuterEvaluationResult(
        workspace,
        segment_path,
        call_path,
        _load_ref(index["outer_eval_metrics"], frozen.project_root, ledger=False),
        run_record_path,
    )


def outer_evaluate_reserved_coordinate(
    project_root: str | Path,
    coordinates: Work9Coordinates,
    *,
    device: str,
    registry: Work9Registry | None = None,
) -> OuterEvaluationResult:
    """Evaluate one already-selected coordinate after matrix-wide release.

    This is an internal Phase-B primitive. The public CLI exposes only the
    matrix driver, never a coordinate debug release.
    """

    frozen = verify_frozen_work9_inputs(project_root, require_runner_acceptance=True)
    registry = registry or Work9Registry(frozen.project_root)
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
        raise Work9TrainingError("coordinate is not uniquely released by selection index")
    entry = entry_matches[0]
    seal_path = _load_ref(entry["selection_seal"], frozen.project_root, ledger=False)
    seal = load_selection_seal(seal_path, frozen.project_root)
    if seal.coordinates != coordinates:
        raise Work9TrainingError("released selection seal coordinates differ")
    seal.verify(registry.root / coordinates.run_id)
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
            "schema_version": "paper2-v2-work9-outer-authorization-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "selection_index": artifact_record(index_path, frozen.project_root),
            "selection_seal": artifact_record(seal_path, frozen.project_root),
            "global_claim": artifact_record(claim_ref.path, frozen.project_root),
            "checkpoint_sha256": seal.checkpoint.sha256,
            "resume_policy": "same_run_same_selection_only",
        }
        workspace.write_json_or_verify("outer_authorization.json", authorization)

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
                raise Work9TrainingError("partial outer ledger directory is invalid")
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
                raise Work9TrainingError("resumed outer ledgers fail semantic recomputation")
            metrics = compute_call_metrics(calls, threshold=seal.threshold)
            expected_metrics_payload = {
                "schema_version": "paper2-v2-work9-outer-metrics-1",
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
                raise Work9TrainingError("resumed outer metrics fail recomputation")
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
                "schema_version": "paper2-v2-work9-outer-metrics-1",
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
            if (
                outer_run_record.get("run_id") != coordinates.run_id
                or outer_run_record.get("status") != "outer_complete"
                or outer_run_record.get("lineage", {}).get("config_sha256")
                != WORK8_CONFIG_SHA256
            ):
                raise Work9TrainingError("existing outer run record lineage differs")
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
            record_ref = workspace.write_json_once(
                "run_record.json", outer_run_record
            )
        artifact_index = {
            "schema_version": "paper2-v2-work9-run-artifact-index-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
            "selection_index": artifact_record(index_path, frozen.project_root),
            "selection_seal": artifact_record(seal_path, frozen.project_root),
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
                "schema_version": "paper2-v2-work9-terminal-1",
                "run_id": coordinates.run_id,
                **asdict(coordinates),
                "status": "outer_complete",
                "artifact_index_sha256": index_ref.sha256,
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
    coordinates: Work9Coordinates,
    built: Any,
    outer_data: Work9FeatureDataset,
    outer_manifest_sha256: str,
    seal: Work9SelectionSeal,
    seal_path: Path,
    outer_refs: Mapping[str, Any],
    segment_count: int,
    call_count: int,
    device: torch.device,
    wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": "paper2-v2-work9-run-record-1",
        "run_id": coordinates.run_id,
        **asdict(coordinates),
        "phase": "outer_eval",
        "status": "outer_complete",
        "lineage": {
            "config_sha256": WORK8_CONFIG_SHA256,
            "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
            "runner_acceptance_sha256": frozen.runner_acceptance_sha256,
            "experiment_card_sha256": frozen.experiment_card_sha256,
            "gate1_policy_sha256": GATE1_POLICY_SHA256,
            "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
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
    registry: Work9Registry | None = None,
) -> Path:
    frozen = verify_frozen_work9_inputs(project_root, require_runner_acceptance=True)
    registry = registry or Work9Registry(frozen.project_root)
    selection_path, selection_index, _ = _load_complete_selection_index(frozen, registry)
    entries = []
    for selected in selection_index["entries"]:
        coordinates = Work9Coordinates(
            frozen.config.protocol_id,
            RUN_FAMILY,
            str(selected["condition"]),
            int(selected["partition_repeat"]),
            int(selected["outer_fold"]),
            int(selected["seed"]),
        )
        run_root = registry.root / coordinates.run_id
        if not run_root.is_dir():
            raise Work9TrainingError(
                f"outer matrix incomplete at coordinate {coordinates.run_id}"
            )
        completed = _verify_completed_outer_run(
            frozen, Work9RunWorkspace(run_root, coordinates)
        )
        record_path = completed.run_record_path
        segment_path = completed.segment_ledger_path
        call_path = completed.call_ledger_path
        metrics_path = completed.metrics_path
        seal_path = _load_ref(
            selected["selection_seal"], frozen.project_root, ledger=False
        )
        seal = load_selection_seal(seal_path, frozen.project_root)
        entries.append(
            {
                "run_id": coordinates.run_id,
                "condition": coordinates.condition,
                "partition_repeat": coordinates.partition_repeat,
                "outer_fold": coordinates.outer_fold,
                "seed": coordinates.seed,
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
        )
    payload = {
        "schema_version": "paper2-v2-work9-outer-ledger-index-1",
        "protocol_id": frozen.config.protocol_id,
        "run_family": RUN_FAMILY,
        "status": "frozen_complete",
        "selection_index": artifact_record(selection_path, frozen.project_root),
        "gate1_analysis_protocol": artifact_record(
            frozen.project_root / GATE1_ANALYSIS_PROTOCOL_RELATIVE,
            frozen.project_root,
        ),
        "expected_coordinate_count": 200,
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
            raise Work9TrainingError("smoke manifest lacks both labels")
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
        raise Work9TrainingError("smoke call selection failed")
    return selected


def run_cuda_smoke(
    project_root: str | Path,
    *,
    output_directory: str | Path,
    device: str = "cuda:0",
) -> Path:
    """Run a deterministic, train/validation-only real-feature CUDA smoke.

    This is deliberately not a primary run and never constructs an outer view.
    The caller provides a new output directory so two independent invocations
    can be compared before freezing the runner acceptance receipt.
    """

    if not str(device).startswith("cuda"):
        raise Work9TrainingError("acceptance smoke must run on CUDA")
    seed = 17
    frozen = verify_frozen_work9_inputs(
        project_root, require_runner_acceptance=False
    )
    from .work9_acceptance import environment_lock, verify_draft_against_current

    acceptance_draft, acceptance_draft_sha = verify_draft_against_current(
        frozen.project_root
    )
    pre_smoke_environment = environment_lock()
    pre_smoke_sys_path = list(sys.path)
    output = Path(output_directory).resolve()
    if frozen.project_root not in output.parents:
        raise Work9TrainingError("acceptance smoke output must remain under project root")
    if output.exists():
        raise FileExistsError(f"smoke output must be new: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch_device = torch.device(device)
    reset_cuda_peak_memory_stats(torch_device)
    coordinates = Work9Coordinates(
        frozen.config.protocol_id, RUN_FAMILY, "A-query", 0, 0, seed
    )
    selection = frozen.config.selection_view(0, 0)
    # Retain a small cached/uncached equivalence benchmark, but the acceptance
    # training itself below uses the complete real train/validation manifests.
    train_rows = _complete_call_smoke_rows(selection.train_manifest.path)
    cache_manifest = output / "cache_equivalence_manifest.jsonl"
    from .work9_registry import atomic_write_once

    atomic_write_once(cache_manifest, canonical_jsonl_bytes(train_rows))
    built = frozen.config.instantiate_condition(coordinates.condition)
    if built.input_spec.global_features or built.input_spec.sequence_features != ("mfcc",):
        raise Work9TrainingError("A-query smoke branch contract drifted")

    cache_started = time.monotonic()
    cached_subset = Work9FeatureDataset(
        cache_manifest,
        input_spec=built.input_spec,
        project_root=frozen.project_root,
        text_store=None,
        pairing_plan=None,
        cache_mfcc=True,
    )
    cached_construction_seconds = time.monotonic() - cache_started
    uncached = Work9FeatureDataset(
        cache_manifest,
        input_spec=built.input_spec,
        project_root=frozen.project_root,
        text_store=None,
        pairing_plan=None,
        cache_mfcc=False,
    )
    uncached_started = time.monotonic()
    for index in range(len(uncached)):
        if not torch.equal(
            uncached[index].sequences["mfcc"], cached_subset[index].sequences["mfcc"]
        ):
            raise Work9TrainingError("MFCC cache changed feature semantics")
    uncached_iteration_seconds = time.monotonic() - uncached_started
    cached_started = time.monotonic()
    for index in range(len(cached_subset)):
        _ = cached_subset[index].sequences["mfcc"]
    cached_iteration_seconds = time.monotonic() - cached_started
    smoke_registry = Work9Registry(
        frozen.project_root,
        root=output / "selection_registry",
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
            raise Work9TrainingError(f"acceptance smoke selection chain missing {required}")
    if (result.workspace.path / "outer_eval").exists() or (
        result.workspace.path / "outer_authorization.json"
    ).exists():
        raise Work9TrainingError("acceptance smoke accessed outer_eval")
    _, post_smoke_draft_sha = verify_draft_against_current(frozen.project_root)
    post_smoke_environment = environment_lock()
    post_smoke_sys_path = list(sys.path)
    if post_smoke_draft_sha != acceptance_draft_sha:
        raise Work9TrainingError("runner acceptance DRAFT changed during CUDA smoke")
    if post_smoke_environment != pre_smoke_environment:
        raise Work9TrainingError("fixed-root environment lock changed during CUDA smoke")
    semantic = {
        "schema_version": CUDA_SMOKE_SEMANTIC_SCHEMA_VERSION,
        "config_sha256": WORK8_CONFIG_SHA256,
        "work8_receipt_sha256": WORK8_RECEIPT_SHA256,
        "experiment_card_sha256": frozen.experiment_card_sha256,
        "gate1_policy_sha256": GATE1_POLICY_SHA256,
        "gate1_analysis_protocol_sha256": GATE1_ANALYSIS_PROTOCOL_SHA256,
        "runner_acceptance_draft_sha256": acceptance_draft_sha,
        "bundle_digests": acceptance_draft["bundle_digests"],
        "environment_lock_sha256": pre_smoke_environment[
            "environment_lock_sha256"
        ],
        "condition": coordinates.condition,
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
        "text_store_opened": False,
    }
    semantic_bytes = canonical_json_bytes(semantic)
    semantic_ref = atomic_write_once(output / "semantic_output.json", semantic_bytes)
    receipt = {
        "schema_version": CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "device_type": "cuda",
        "device": str(torch_device),
        "device_name": torch.cuda.get_device_name(torch_device),
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
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
            "primary_registry_accessed": False,
            "scientific_selection_claimed": False,
        },
    }
    receipt_ref = atomic_write_once(
        output / "smoke_receipt.json", canonical_json_bytes(receipt)
    )
    return receipt_ref.path


def record_coordinate_failure(
    project_root: str | Path,
    coordinates: Work9Coordinates,
    *,
    phase: str,
    error: BaseException,
) -> Path:
    """Append a diagnostic without burning a resumable primary coordinate."""

    registry = Work9Registry(project_root)
    workspace = registry.reserve(coordinates)
    reference = workspace.record_failure(
        {
            "schema_version": "paper2-v2-work9-failure-1",
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
