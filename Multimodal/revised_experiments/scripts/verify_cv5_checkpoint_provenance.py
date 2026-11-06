#!/usr/bin/env python3
"""Verify the frozen Track-D 5-fold acoustic checkpoint provenance chain.

The verifier is deliberately read-only unless ``--output-directory`` is
supplied.  For every fold/feature coordinate it:

* loads the checkpoint with ``torch.load(..., weights_only=True)``;
* maps it to the fold validation/test manifests, archived segment
  probabilities, original segment-level test result, and call-level result;
* checks those artifacts against the frozen SHA-256 inventory;
* replays CPU inference and checks row identity, probability agreement, and
  decision-threshold stability;
* recomputes segment-level test metrics; and
* reproduces validation-only call-policy selection and the archived test
  call-level metrics.

This is an integrity/provenance verifier, not a training or model-selection
entry point.  It never modifies the frozen inputs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
FEATURE_TYPES = ("egemaps", "mfcc", "wav2vec2", "all")
EXPECTED_INPUT_DIMS = {"egemaps": 88, "mfcc": 13, "wav2vec2": 768, "all": 869}
SPLITS = ("val", "test")
ALPHAS = (0.3, 0.5, 0.7, 0.9)
TAUS = (0.3, 0.5, 0.7, 0.9)
IDENTITY_FIELDS = ("call_id", "segment_id", "start", "end", "label", "source")
EXPECTED_STATE_KEYS = (
    "net.0.weight",
    "net.0.bias",
    "net.3.weight",
    "net.3.bias",
)


class AudioClassifier(nn.Module):
    """Architecture used by ``modeling/train_audio_baseline.py``."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT,
        help="Repository root containing Multimodal/ (auto-detected by default).",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Frozen artifact inventory (defaults to the v1 freeze inventory).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Write JSON/CSV receipts here. If omitted, no files are written.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--probability-atol",
        type=float,
        default=1e-5,
        help="Maximum archived-versus-CPU probability drift (default: 1e-5).",
    )
    parser.add_argument(
        "--metric-atol",
        type=float,
        default=1e-7,
        help=(
            "Maximum scalar metric drift (default: 1e-7). The tolerance covers "
            "the archived batch-size/CUDA and current CPU replay paths; confusion "
            "matrices and threshold decisions must still match exactly."
        ),
    )
    parser.add_argument("--call-score-atol", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _resolve_inside(root: Path, raw_path: str, description: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{description} escapes Multimodal root: {raw_path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {description}: {resolved}")
    return resolved


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(payload)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _discover_segment_result(multimodal_root: Path, fold: int, feature_type: str) -> Path:
    directory = (
        multimodal_root
        / "modeling/logs/eval_results"
        / f"audio_only_{feature_type}_fold{fold}"
    )
    matches = sorted(directory.glob(f"test_results_{feature_type}_*.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one archived segment result for fold{fold}/{feature_type}; "
            f"found {len(matches)} under {directory}"
        )
    return matches[0]


def _coordinate_paths(
    repository_root: Path, multimodal_root: Path
) -> tuple[list[dict[str, Any]], set[str]]:
    mappings: list[dict[str, Any]] = []
    target_paths: set[str] = set()
    for fold in range(5):
        manifests = {
            split: multimodal_root
            / "data/cv_5fold"
            / f"fold{fold}"
            / f"{split}_segment_manifest.jsonl"
            for split in SPLITS
        }
        for feature_type in FEATURE_TYPES:
            checkpoint = (
                multimodal_root
                / "modeling/models/cv_5fold"
                / f"fold{fold}"
                / f"best_audio_model_{feature_type}.pth"
            )
            predictions = {
                split: multimodal_root
                / "modeling/logs/track_d"
                / f"fold{fold}"
                / "segment_predictions"
                / f"{split}_{feature_type}.jsonl"
                for split in SPLITS
            }
            segment_result = _discover_segment_result(
                multimodal_root, fold, feature_type
            )
            call_result = (
                multimodal_root
                / "modeling/logs/track_d"
                / f"fold{fold}"
                / "call_level"
                / f"test_results_{feature_type}.json"
            )
            mapping = {
                "coordinate": f"fold{fold}/{feature_type}",
                "fold": fold,
                "feature_type": feature_type,
                "checkpoint": checkpoint,
                "manifests": manifests,
                "predictions": predictions,
                "segment_result": segment_result,
                "call_level_result": call_result,
            }
            mappings.append(mapping)
            paths = [
                checkpoint,
                segment_result,
                call_result,
                *manifests.values(),
                *predictions.values(),
            ]
            for path in paths:
                target_paths.add(_repo_relative(path, repository_root))
    return mappings, target_paths


def _read_inventory(path: Path, target_paths: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            artifact_path = row.get("path")
            if artifact_path not in target_paths:
                continue
            if artifact_path in found and found[artifact_path] != row:
                raise RuntimeError(
                    f"conflicting inventory rows for {artifact_path} (line {line_number})"
                )
            found[artifact_path] = row
    return found


def _verify_hashes(
    repository_root: Path,
    target_paths: set[str],
    inventory_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for relative in sorted(target_paths):
        path = repository_root / relative
        inventory_row = inventory_rows.get(relative)
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha = _sha256(path) if exists else None
        expected_sha = inventory_row.get("sha256") if inventory_row else None
        expected_size = inventory_row.get("size_bytes") if inventory_row else None
        status = "pass"
        if not exists:
            status = "missing_file"
        elif inventory_row is None:
            status = "missing_inventory_entry"
        elif actual_size != expected_size:
            status = "size_mismatch"
        elif actual_sha != expected_sha:
            status = "sha256_mismatch"
        if status != "pass":
            failures.append(f"hash:{relative}:{status}")
        records.append(
            {
                "path": relative,
                "role": inventory_row.get("role") if inventory_row else None,
                "status": status,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
            }
        )
    return records, failures


def _safe_load_model(checkpoint: Path, expected_input_dim: int) -> AudioClassifier:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "installed PyTorch lacks weights_only=True; refusing unsafe fallback"
        ) from error
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint is not a state dictionary: {checkpoint}")
    if tuple(state.keys()) != EXPECTED_STATE_KEYS:
        raise RuntimeError(
            f"unexpected state keys in {checkpoint}: {tuple(state.keys())}"
        )
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(f"checkpoint contains non-tensor state: {checkpoint}")
    expected_shapes = {
        "net.0.weight": (256, expected_input_dim),
        "net.0.bias": (256,),
        "net.3.weight": (1, 256),
        "net.3.bias": (1,),
    }
    actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"state shape mismatch in {checkpoint}: {actual_shapes} != {expected_shapes}"
        )
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise RuntimeError(f"checkpoint contains non-finite weights: {checkpoint}")
    model = AudioClassifier(expected_input_dim)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _feature_vectors(
    multimodal_root: Path, entries: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    mfcc_rows: list[np.ndarray] = []
    egemaps_rows: list[np.ndarray] = []
    wav2vec_rows: list[np.ndarray] = []
    for entry in entries:
        mfcc_path = _resolve_inside(
            multimodal_root, str(entry["mfcc_path"]), "MFCC feature"
        )
        egemaps_path = _resolve_inside(
            multimodal_root, str(entry["egemaps_path"]), "eGeMAPS feature"
        )
        wav2vec_path = _resolve_inside(
            multimodal_root, str(entry["wav2vec2_path"]), "Wav2Vec2 feature"
        )
        mfcc_rows.append(
            np.asarray(np.load(mfcc_path, allow_pickle=False), dtype=np.float32).mean(axis=0)
        )
        egemaps_rows.append(
            np.asarray(np.load(egemaps_path, allow_pickle=False), dtype=np.float32).reshape(-1)
        )
        wav2vec_rows.append(
            np.asarray(np.load(wav2vec_path, allow_pickle=False), dtype=np.float32).mean(axis=0)
        )
    matrices = {
        "mfcc": np.stack(mfcc_rows),
        "egemaps": np.stack(egemaps_rows),
        "wav2vec2": np.stack(wav2vec_rows),
    }
    matrices["all"] = np.concatenate(
        [matrices["mfcc"], matrices["egemaps"], matrices["wav2vec2"]], axis=1
    )
    for feature_type, matrix in matrices.items():
        expected = (len(entries), EXPECTED_INPUT_DIMS[feature_type])
        if matrix.shape != expected:
            raise RuntimeError(
                f"{feature_type} feature matrix shape {matrix.shape} != {expected}"
            )
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"non-finite {feature_type} feature values")
    return matrices


def _replay_split(
    multimodal_root: Path,
    entries: list[dict[str, Any]],
    models: dict[str, AudioClassifier],
    batch_size: int,
) -> dict[str, list[float]]:
    probabilities = {feature_type: [] for feature_type in FEATURE_TYPES}
    with torch.inference_mode():
        for start in range(0, len(entries), batch_size):
            batch_entries = entries[start : start + batch_size]
            matrices = _feature_vectors(multimodal_root, batch_entries)
            for feature_type in FEATURE_TYPES:
                tensor = torch.from_numpy(matrices[feature_type])
                batch_probs = torch.sigmoid(models[feature_type](tensor))
                probabilities[feature_type].extend(
                    float(value) for value in batch_probs.tolist()
                )
    return probabilities


def _identity_value(row: dict[str, Any], field: str) -> Any:
    if field == "source":
        return row.get(field, "unknown")
    return row.get(field)


def _compare_rows(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    feature_type: str,
) -> tuple[int, list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    if len(manifest_rows) != len(prediction_rows):
        mismatches.append(
            {
                "kind": "row_count",
                "manifest": len(manifest_rows),
                "predictions": len(prediction_rows),
            }
        )
    for index, (manifest, prediction) in enumerate(
        zip(manifest_rows, prediction_rows)
    ):
        differing = [
            field
            for field in IDENTITY_FIELDS
            if _identity_value(manifest, field) != _identity_value(prediction, field)
        ]
        if prediction.get("feature_type") != feature_type:
            differing.append("feature_type")
        if differing and len(mismatches) < 10:
            mismatches.append(
                {"kind": "identity", "row_index": index, "fields": differing}
            )
    count = abs(len(manifest_rows) - len(prediction_rows))
    count += sum(
        any(
            _identity_value(manifest, field) != _identity_value(prediction, field)
            for field in IDENTITY_FIELDS
        )
        or prediction.get("feature_type") != feature_type
        for manifest, prediction in zip(manifest_rows, prediction_rows)
    )
    return count, mismatches


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("ROC-AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    rank_sum = 0.0
    index = 0
    while index < len(ordered_scores):
        end = index + 1
        while end < len(ordered_scores) and ordered_scores[end] == ordered_scores[index]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * float(ordered_labels[index:end].sum())
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _binary_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, Any]:
    y_true = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(probabilities), dtype=np.float64)
    if len(y_true) != len(scores) or len(y_true) == 0:
        raise RuntimeError("invalid metric input lengths")
    predicted = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (predicted == 1)))
    tn = int(np.sum((y_true == 0) & (predicted == 0)))
    fp = int(np.sum((y_true == 0) & (predicted == 1)))
    fn = int(np.sum((y_true == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": _roc_auc(y_true, scores),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _metric_deltas(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    return {
        key: abs(float(actual[key]) - float(expected[key]))
        for key in ("accuracy", "f1_score", "precision", "recall", "roc_auc")
    }


def _group_probabilities(
    rows: list[dict[str, Any]], probabilities: list[float]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[str(row["call_id"])].append(
            {
                "call_id": str(row["call_id"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": int(row["label"]),
                "source": str(row.get("source", "unknown")),
                "prob": float(probability),
            }
        )
    for segments in grouped.values():
        segments.sort(key=lambda item: item["start"])
    return dict(grouped)


def _aggregate_call(
    call_id: str,
    segments: list[dict[str, Any]],
    method: str,
    tau: float,
    alpha: float | None = None,
) -> dict[str, Any]:
    score = 0.0
    alert_index: int | None = None
    if method == "ema":
        if alpha is None:
            raise RuntimeError("EMA aggregation requires alpha")
        for index, segment in enumerate(segments):
            score = alpha * score + (1.0 - alpha) * float(segment["prob"])
            if alert_index is None and score >= tau:
                alert_index = index
    elif method == "running_max":
        for index, segment in enumerate(segments):
            score = max(score, float(segment["prob"]))
            if alert_index is None and score >= tau:
                alert_index = index
    elif method == "majority":
        score = sum(float(segment["prob"]) for segment in segments) / len(segments)
        if score >= tau:
            alert_index = len(segments) - 1
    else:
        raise RuntimeError(f"unknown aggregation method: {method}")
    return {
        "call_id": call_id,
        "label": int(segments[-1]["label"]),
        "source": str(segments[-1]["source"]),
        "n_segments": len(segments),
        "duration": max(float(segment["end"]) for segment in segments),
        "final_score": float(score),
        "final_decision": int(score >= tau),
        "alerted": alert_index is not None,
        "alert_time": (
            float(segments[alert_index]["end"]) if alert_index is not None else None
        ),
        "alert_segment_index": alert_index,
    }


def _aggregate_all(
    grouped: dict[str, list[dict[str, Any]]],
    method: str,
    tau: float,
    alpha: float | None = None,
) -> list[dict[str, Any]]:
    return [
        _aggregate_call(call_id, segments, method, tau, alpha)
        for call_id, segments in grouped.items()
    ]


def _call_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in results], dtype=np.int64)
    decisions = np.asarray([row["final_decision"] for row in results], dtype=np.int64)
    tp = int(np.sum((labels == 1) & (decisions == 1)))
    tn = int(np.sum((labels == 0) & (decisions == 0)))
    fp = int(np.sum((labels == 0) & (decisions == 1)))
    fn = int(np.sum((labels == 1) & (decisions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / len(labels),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _select_call_configs(
    rows: list[dict[str, Any]], probabilities: list[float]
) -> dict[str, dict[str, float | None]]:
    grouped = _group_probabilities(rows, probabilities)
    ema_candidates: list[dict[str, float]] = []
    for alpha in ALPHAS:
        for tau in TAUS:
            metrics = _call_metrics(_aggregate_all(grouped, "ema", tau, alpha))
            ema_candidates.append({"alpha": alpha, "tau": tau, "f1": metrics["f1"]})
    running_candidates: list[dict[str, float]] = []
    for tau in TAUS:
        metrics = _call_metrics(_aggregate_all(grouped, "running_max", tau))
        running_candidates.append({"tau": tau, "f1": metrics["f1"]})
    best_ema = max(ema_candidates, key=lambda row: row["f1"])
    best_running = max(running_candidates, key=lambda row: row["f1"])
    return {
        "ema": {"alpha": best_ema["alpha"], "tau": best_ema["tau"]},
        "running_max": {"alpha": None, "tau": best_running["tau"]},
    }


def _compare_call_metrics(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, float]:
    return {
        key: abs(float(actual[key]) - float(expected[key]))
        for key in ("f1", "precision", "recall", "accuracy", "fpr", "tp", "tn", "fp", "fn")
    }


def _evaluate_call_results(
    test_rows: list[dict[str, Any]],
    archived_probabilities: list[float],
    replay_probabilities: list[float],
    result: dict[str, Any],
    metric_atol: float,
    call_score_atol: float,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    archived_grouped = _group_probabilities(test_rows, archived_probabilities)
    replay_grouped = _group_probabilities(test_rows, replay_probabilities)
    configs = {
        "ema": {
            "alpha": float(result["ema_best_config"]["alpha"]),
            "tau": float(result["ema_best_config"]["tau"]),
            "reported_key": "ema_metrics",
            "calls_key": "ema_calls",
        },
        "running_max": {
            "alpha": None,
            "tau": float(result["running_max_best_config"]["tau"]),
            "reported_key": "running_max_metrics",
            "calls_key": "running_max_calls",
        },
        "any_segment_baseline": {
            "method": "running_max",
            "alpha": None,
            "tau": 0.5,
            "reported_key": "any_segment_baseline_metrics",
            "calls_key": None,
        },
        "majority": {
            "alpha": None,
            "tau": 0.5,
            "reported_key": "majority_metrics",
            "calls_key": None,
        },
    }
    details: dict[str, Any] = {}
    for name, config in configs.items():
        method = str(config.get("method", name))
        archived_calls = _aggregate_all(
            archived_grouped, method, float(config["tau"]), config["alpha"]
        )
        replay_calls = _aggregate_all(
            replay_grouped, method, float(config["tau"]), config["alpha"]
        )
        archived_metrics = _call_metrics(archived_calls)
        replay_metrics = _call_metrics(replay_calls)
        reported_metrics = result[str(config["reported_key"])]
        archive_deltas = _compare_call_metrics(archived_metrics, reported_metrics)
        replay_deltas = _compare_call_metrics(replay_metrics, reported_metrics)
        replay_by_id = {row["call_id"]: row for row in replay_calls}
        score_differences = [
            abs(row["final_score"] - replay_by_id[row["call_id"]]["final_score"])
            for row in archived_calls
        ]
        decision_crossings = sum(
            row["final_decision"] != replay_by_id[row["call_id"]]["final_decision"]
            for row in archived_calls
        )
        stored_call_mismatches = 0
        calls_key = config["calls_key"]
        if calls_key is not None:
            stored_by_id = {row["call_id"]: row for row in result[str(calls_key)]}
            for row in archived_calls:
                stored = stored_by_id.get(row["call_id"])
                if stored is None:
                    stored_call_mismatches += 1
                    continue
                if (
                    stored.get("final_decision") != row["final_decision"]
                    or abs(float(stored.get("final_score")) - row["final_score"]) > metric_atol
                ):
                    stored_call_mismatches += 1
            stored_call_mismatches += len(set(stored_by_id) - {row["call_id"] for row in archived_calls})
        if max(archive_deltas.values(), default=0.0) > metric_atol:
            failures.append(f"call_metrics:{name}:archived_vs_reported")
        if max(replay_deltas.values(), default=0.0) > metric_atol:
            failures.append(f"call_metrics:{name}:replay_vs_reported")
        if max(score_differences, default=0.0) > call_score_atol:
            failures.append(f"call_score:{name}:above_tolerance")
        if decision_crossings:
            failures.append(f"call_decision:{name}:{decision_crossings}_crossings")
        if stored_call_mismatches:
            failures.append(f"stored_calls:{name}:{stored_call_mismatches}_mismatches")
        details[name] = {
            "config": {"alpha": config["alpha"], "tau": config["tau"]},
            "archived_prediction_metrics": archived_metrics,
            "replay_metrics": replay_metrics,
            "reported_metrics": reported_metrics,
            "max_archived_metric_delta": max(archive_deltas.values(), default=0.0),
            "max_replay_metric_delta": max(replay_deltas.values(), default=0.0),
            "max_call_score_abs_difference": max(score_differences, default=0.0),
            "call_decision_threshold_crossings": decision_crossings,
            "stored_call_mismatches": stored_call_mismatches,
        }
    return details, failures


def _verify_coordinate(
    mapping: dict[str, Any],
    repository_root: Path,
    split_cache: dict[int, dict[str, dict[str, Any]]],
    replay_cache: dict[int, dict[str, Any]],
    probability_atol: float,
    metric_atol: float,
    call_score_atol: float,
    threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    coordinate = str(mapping["coordinate"])
    fold = int(mapping["fold"])
    feature_type = str(mapping["feature_type"])
    failures: list[str] = []
    split_details: dict[str, Any] = {}
    for split in SPLITS:
        manifest_rows = split_cache[fold][split]["manifest_rows"]
        prediction_rows = split_cache[fold][split]["prediction_rows"][feature_type]
        replay_probabilities = replay_cache[fold][split][feature_type]
        archived_probabilities = [float(row["prob"]) for row in prediction_rows]
        identity_count, identity_examples = _compare_rows(
            manifest_rows, prediction_rows, feature_type
        )
        if len(replay_probabilities) != len(archived_probabilities):
            failures.append(f"{coordinate}:{split}:replay_count")
            probability_differences = np.asarray([], dtype=np.float64)
            threshold_crossings = -1
            nonfinite = -1
        else:
            replay_array = np.asarray(replay_probabilities, dtype=np.float64)
            archived_array = np.asarray(archived_probabilities, dtype=np.float64)
            probability_differences = np.abs(replay_array - archived_array)
            threshold_crossings = int(
                np.sum((replay_array >= threshold) != (archived_array >= threshold))
            )
            nonfinite = int(
                np.sum(~np.isfinite(replay_array)) + np.sum(~np.isfinite(archived_array))
            )
        above_tolerance = int(np.sum(probability_differences > probability_atol))
        if identity_count:
            failures.append(f"{coordinate}:{split}:{identity_count}_identity_mismatches")
        if nonfinite:
            failures.append(f"{coordinate}:{split}:{nonfinite}_nonfinite_probabilities")
        if above_tolerance:
            failures.append(f"{coordinate}:{split}:{above_tolerance}_probability_differences")
        if threshold_crossings:
            failures.append(f"{coordinate}:{split}:{threshold_crossings}_threshold_crossings")
        split_details[split] = {
            "manifest": _repo_relative(mapping["manifests"][split], repository_root),
            "archived_predictions": _repo_relative(
                mapping["predictions"][split], repository_root
            ),
            "rows": len(manifest_rows),
            "row_identity_mismatches": identity_count,
            "row_identity_examples": identity_examples,
            "max_probability_abs_difference": float(
                probability_differences.max(initial=0.0)
            ),
            "mean_probability_abs_difference": float(
                probability_differences.mean() if probability_differences.size else 0.0
            ),
            "probabilities_above_tolerance": above_tolerance,
            "segment_decision_threshold": threshold,
            "segment_decision_threshold_crossings": threshold_crossings,
            "nonfinite_probabilities": nonfinite,
        }

    test_rows = split_cache[fold]["test"]["manifest_rows"]
    test_prediction_rows = split_cache[fold]["test"]["prediction_rows"][feature_type]
    archived_test_probabilities = [float(row["prob"]) for row in test_prediction_rows]
    replay_test_probabilities = replay_cache[fold]["test"][feature_type]
    labels = [int(row["label"]) for row in test_rows]
    reported_segment_metrics = _load_json(mapping["segment_result"])
    archived_segment_metrics = _binary_metrics(labels, archived_test_probabilities, threshold)
    replay_segment_metrics = _binary_metrics(labels, replay_test_probabilities, threshold)
    archive_metric_deltas = _metric_deltas(
        archived_segment_metrics, reported_segment_metrics
    )
    replay_metric_deltas = _metric_deltas(replay_segment_metrics, reported_segment_metrics)
    if archived_segment_metrics["confusion_matrix"] != reported_segment_metrics["confusion_matrix"]:
        failures.append(f"{coordinate}:segment_metrics:archived_confusion_matrix")
    if replay_segment_metrics["confusion_matrix"] != reported_segment_metrics["confusion_matrix"]:
        failures.append(f"{coordinate}:segment_metrics:replay_confusion_matrix")
    if max(archive_metric_deltas.values()) > metric_atol:
        failures.append(f"{coordinate}:segment_metrics:archived_delta")
    if max(replay_metric_deltas.values()) > metric_atol:
        failures.append(f"{coordinate}:segment_metrics:replay_delta")

    val_rows = split_cache[fold]["val"]["manifest_rows"]
    archived_val_probabilities = [
        float(row["prob"])
        for row in split_cache[fold]["val"]["prediction_rows"][feature_type]
    ]
    selected_configs = _select_call_configs(val_rows, archived_val_probabilities)
    reported_call_result = _load_json(mapping["call_level_result"])
    reported_configs = {
        "ema": {
            "alpha": float(reported_call_result["ema_best_config"]["alpha"]),
            "tau": float(reported_call_result["ema_best_config"]["tau"]),
        },
        "running_max": {
            "alpha": None,
            "tau": float(reported_call_result["running_max_best_config"]["tau"]),
        },
    }
    if selected_configs != reported_configs:
        failures.append(f"{coordinate}:validation_selected_config_mismatch")
    call_details, call_failures = _evaluate_call_results(
        test_rows,
        archived_test_probabilities,
        replay_test_probabilities,
        reported_call_result,
        metric_atol,
        call_score_atol,
    )
    failures.extend(f"{coordinate}:{failure}" for failure in call_failures)

    checkpoint = mapping["checkpoint"]
    result = {
        "coordinate": coordinate,
        "fold": fold,
        "feature_type": feature_type,
        "status": "pass" if not failures else "fail",
        "checkpoint": _repo_relative(checkpoint, repository_root),
        "checkpoint_load": {
            "weights_only": True,
            "input_dim": EXPECTED_INPUT_DIMS[feature_type],
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in replay_cache[fold]["models"][feature_type].parameters()
                if parameter.requires_grad
            ),
        },
        "mapped_segment_result": _repo_relative(mapping["segment_result"], repository_root),
        "mapped_call_level_result": _repo_relative(
            mapping["call_level_result"], repository_root
        ),
        "splits": split_details,
        "segment_test_metrics": {
            "reported": reported_segment_metrics,
            "from_archived_predictions": archived_segment_metrics,
            "from_replayed_checkpoint": replay_segment_metrics,
            "max_archived_metric_delta": max(archive_metric_deltas.values()),
            "max_replay_metric_delta": max(replay_metric_deltas.values()),
        },
        "call_level": {
            "validation_selected_configs": selected_configs,
            "reported_configs": reported_configs,
            "test_results": call_details,
        },
        "failures": failures,
    }
    return result, failures


def _prepare_fold_cache(
    mappings: list[dict[str, Any]],
    fold: int,
    multimodal_root: Path,
    batch_size: int,
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, Any]],
]:
    fold_rows = [row for row in mappings if int(row["fold"]) == fold]
    fold_mappings = {row["feature_type"]: row for row in fold_rows}
    if set(fold_mappings) != set(FEATURE_TYPES):
        raise RuntimeError(f"incomplete coordinate map for fold{fold}")
    models = {
        feature_type: _safe_load_model(
            fold_mappings[feature_type]["checkpoint"],
            EXPECTED_INPUT_DIMS[feature_type],
        )
        for feature_type in FEATURE_TYPES
    }
    split_cache: dict[int, dict[str, dict[str, Any]]] = {fold: {}}
    replay_cache: dict[int, dict[str, Any]] = {fold: {"models": models}}
    for split in SPLITS:
        manifest_path = fold_mappings[FEATURE_TYPES[0]]["manifests"][split]
        manifest_rows = _load_jsonl(manifest_path)
        prediction_rows = {
            feature_type: _load_jsonl(
                fold_mappings[feature_type]["predictions"][split]
            )
            for feature_type in FEATURE_TYPES
        }
        split_cache[fold][split] = {
            "manifest_rows": manifest_rows,
            "prediction_rows": prediction_rows,
        }
        replay_cache[fold][split] = _replay_split(
            multimodal_root, manifest_rows, models, batch_size
        )
    return split_cache, replay_cache


def _write_outputs(
    output_directory: Path,
    report: dict[str, Any],
    coordinate_results: list[dict[str, Any]],
    hash_records: list[dict[str, Any]],
) -> list[str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "cv5_checkpoint_provenance_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    coordinate_path = output_directory / "cv5_checkpoint_provenance_coordinates.csv"
    coordinate_fields = [
        "coordinate",
        "status",
        "checkpoint",
        "val_rows",
        "test_rows",
        "val_max_probability_abs_difference",
        "test_max_probability_abs_difference",
        "val_threshold_crossings",
        "test_threshold_crossings",
        "max_replay_metric_delta",
        "failures",
    ]
    with coordinate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=coordinate_fields)
        writer.writeheader()
        for result in coordinate_results:
            writer.writerow(
                {
                    "coordinate": result["coordinate"],
                    "status": result["status"],
                    "checkpoint": result["checkpoint"],
                    "val_rows": result["splits"]["val"]["rows"],
                    "test_rows": result["splits"]["test"]["rows"],
                    "val_max_probability_abs_difference": result["splits"]["val"]["max_probability_abs_difference"],
                    "test_max_probability_abs_difference": result["splits"]["test"]["max_probability_abs_difference"],
                    "val_threshold_crossings": result["splits"]["val"]["segment_decision_threshold_crossings"],
                    "test_threshold_crossings": result["splits"]["test"]["segment_decision_threshold_crossings"],
                    "max_replay_metric_delta": result["segment_test_metrics"]["max_replay_metric_delta"],
                    "failures": ";".join(result["failures"]),
                }
            )
    hash_path = output_directory / "cv5_checkpoint_provenance_hashes.csv"
    hash_fields = [
        "path",
        "role",
        "status",
        "expected_sha256",
        "actual_sha256",
        "expected_size_bytes",
        "actual_size_bytes",
    ]
    with hash_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=hash_fields)
        writer.writeheader()
        writer.writerows(hash_records)
    return [str(json_path), str(coordinate_path), str(hash_path)]


def run(arguments: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    repository_root = arguments.repository_root.resolve()
    multimodal_root = repository_root / "Multimodal"
    if not multimodal_root.is_dir():
        raise FileNotFoundError(f"Multimodal directory not found: {multimodal_root}")
    inventory = (
        arguments.inventory.resolve()
        if arguments.inventory is not None
        else multimodal_root
        / "revised_experiments/artifacts/v1_freeze_2026-08-09/artifact_inventory.jsonl"
    )
    if not inventory.is_file():
        raise FileNotFoundError(f"artifact inventory not found: {inventory}")
    if arguments.batch_size <= 0 or arguments.torch_threads <= 0:
        raise ValueError("batch size and torch thread count must be positive")
    if arguments.probability_atol < 0 or arguments.metric_atol < 0 or arguments.call_score_atol < 0:
        raise ValueError("tolerances must be non-negative")
    if not 0.0 <= arguments.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    torch.set_num_threads(arguments.torch_threads)
    torch.use_deterministic_algorithms(True)
    mappings, target_paths = _coordinate_paths(repository_root, multimodal_root)
    inventory_rows = _read_inventory(inventory, target_paths)
    hash_records, failures = _verify_hashes(
        repository_root, target_paths, inventory_rows
    )
    coordinate_results: list[dict[str, Any]] = []
    for fold in range(5):
        split_cache, replay_cache = _prepare_fold_cache(
            mappings, fold, multimodal_root, arguments.batch_size
        )
        for mapping in mappings:
            if int(mapping["fold"]) != fold:
                continue
            result, coordinate_failures = _verify_coordinate(
                mapping,
                repository_root,
                split_cache,
                replay_cache,
                arguments.probability_atol,
                arguments.metric_atol,
                arguments.call_score_atol,
                arguments.threshold,
            )
            coordinate_results.append(result)
            failures.extend(coordinate_failures)
        del split_cache, replay_cache
        gc.collect()

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "scope": "Track-D 5-fold acoustic checkpoint provenance",
        "read_only_inputs": True,
        "repository_root": str(repository_root),
        "inventory": _repo_relative(inventory, repository_root),
        "inventory_sha256": _sha256(inventory),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_threads": arguments.torch_threads,
            "safe_checkpoint_load": "torch.load(weights_only=True)",
        },
        "policy": {
            "probability_atol": arguments.probability_atol,
            "metric_atol": arguments.metric_atol,
            "call_score_atol": arguments.call_score_atol,
            "segment_decision_threshold": arguments.threshold,
        },
        "summary": {
            "coordinates_expected": 20,
            "coordinates_checked": len(coordinate_results),
            "coordinates_passed": sum(
                result["status"] == "pass" for result in coordinate_results
            ),
            "checkpoints_safely_loaded": len(coordinate_results),
            "inventory_artifacts_expected": len(target_paths),
            "inventory_artifacts_passed": sum(
                row["status"] == "pass" for row in hash_records
            ),
            "row_identity_mismatches": sum(
                result["splits"][split]["row_identity_mismatches"]
                for result in coordinate_results
                for split in SPLITS
            ),
            "segment_threshold_crossings": sum(
                result["splits"][split]["segment_decision_threshold_crossings"]
                for result in coordinate_results
                for split in SPLITS
            ),
            "call_threshold_crossings": sum(
                detail["call_decision_threshold_crossings"]
                for result in coordinate_results
                for detail in result["call_level"]["test_results"].values()
            ),
            "max_probability_abs_difference": max(
                result["splits"][split]["max_probability_abs_difference"]
                for result in coordinate_results
                for split in SPLITS
            ),
            "failure_count": len(failures),
        },
        "coordinate_results": coordinate_results,
        "hash_verification": hash_records,
        "failures": failures,
    }
    written: list[str] = []
    if arguments.output_directory is not None:
        written = _write_outputs(
            arguments.output_directory.resolve(), report, coordinate_results, hash_records
        )
    return report, written


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report, written = run(arguments)
    except Exception as error:
        payload = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "files_written": [],
        }
        print(json.dumps(payload, sort_keys=True))
        return 2
    summary = {
        "status": report["status"],
        **report["summary"],
        "files_written": written,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
