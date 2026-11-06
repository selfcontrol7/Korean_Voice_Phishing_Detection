"""Fold-aware learned shortcut diagnostics for the sealed Work 6 protocol.

These models are diagnostics, not Paper 2 candidate systems.  Every transform
is fitted on the outer-training calls, the decision threshold is selected on
validation calls, and outer evaluation is scored once without feeding results
back into folds, model configuration, or Work 8.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .work5_audit import normalize_text_strict, sha256_file


PROBE_SCHEMA = "paper2-v2-learned-shortcut-probes-1"


class ShortcutProbeError(RuntimeError):
    """The frozen-fold or train/validation/evaluation contract was violated."""


def _canonical_json(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ShortcutProbeError(f"{path}:{row_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ShortcutProbeError(f"empty JSONL artifact: {path}")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(_canonical_json(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")


def verify_checksum_manifest(root: Path, checksum_path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ShortcutProbeError(
                f"invalid checksum row {checksum_path}:{line_number}"
            ) from error
        if relative in expected:
            raise ShortcutProbeError(f"duplicate checksum path {relative!r}")
        path = root / relative
        if not path.is_file():
            raise ShortcutProbeError(f"frozen Work 6 artifact missing: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise ShortcutProbeError(
                f"frozen Work 6 hash mismatch for {relative}: {actual} != {digest}"
            )
        expected[relative] = digest
    if not expected:
        raise ShortcutProbeError("Work 6 checksum manifest is empty")
    return expected


@dataclass(frozen=True)
class FoldRows:
    outer_fold: int
    train: tuple[dict[str, Any], ...]
    val: tuple[dict[str, Any], ...]
    outer_eval: tuple[dict[str, Any], ...]


def _validate_partition_rows(rows: Sequence[Mapping[str, Any]], role: str, fold: int) -> None:
    call_ids = [str(row.get("call_id", "")) for row in rows]
    if any(not call_id for call_id in call_ids):
        raise ShortcutProbeError(f"outer fold {fold} {role} has blank call_id")
    if len(set(call_ids)) != len(call_ids):
        raise ShortcutProbeError(f"outer fold {fold} {role} repeats a call")
    if any(str(row.get("partition", "")) != role for row in rows):
        raise ShortcutProbeError(f"outer fold {fold} manifest role disagrees with rows")
    if any(int(row.get("outer_fold", -1)) != fold for row in rows):
        raise ShortcutProbeError(f"outer fold {fold} manifest has wrong outer_fold")
    if any(not str(row.get("group_id", "")) for row in rows):
        raise ShortcutProbeError(f"outer fold {fold} {role} has blank group_id")


def load_frozen_folds(work6_root: Path) -> tuple[dict[str, Any], list[FoldRows], dict[str, Any]]:
    protocol_path = work6_root / "protocol.json"
    checksum_path = work6_root / "SHA256SUMS.txt"
    if not protocol_path.is_file() or not checksum_path.is_file():
        raise ShortcutProbeError("Work 6 tree requires protocol.json and SHA256SUMS.txt")
    checksums = verify_checksum_manifest(work6_root, checksum_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = str(protocol.get("artifact_status", "")).lower()
    if status not in {"frozen", "sealed", "frozen_v1"}:
        raise ShortcutProbeError(
            f"learned probes require a frozen Work 6 tree; artifact_status={status!r}"
        )
    if protocol.get("terminology", {}).get("outer_partition_name") != "outer_eval":
        raise ShortcutProbeError("Work 6 must name the assessment partition outer_eval")
    folds: list[FoldRows] = []
    outer_seen: Counter[str] = Counter()
    for fold_record in sorted(protocol.get("folds", []), key=lambda row: int(row["outer_fold"])):
        fold = int(fold_record["outer_fold"])
        partitions: dict[str, tuple[dict[str, Any], ...]] = {}
        groups: dict[str, set[str]] = {}
        calls: dict[str, set[str]] = {}
        for role in ("train", "val", "outer_eval"):
            relative = str(fold_record["manifests"][role]["call_manifest"])
            if relative not in checksums:
                raise ShortcutProbeError(f"call manifest absent from Work 6 hashes: {relative}")
            rows = tuple(_read_jsonl(work6_root / relative))
            _validate_partition_rows(rows, role, fold)
            partitions[role] = rows
            groups[role] = {str(row["group_id"]) for row in rows}
            calls[role] = {str(row["call_id"]) for row in rows}
        for left, right in (("train", "val"), ("train", "outer_eval"), ("val", "outer_eval")):
            if calls[left] & calls[right]:
                raise ShortcutProbeError(f"call leakage between {left} and {right} in fold {fold}")
            if groups[left] & groups[right]:
                raise ShortcutProbeError(f"group leakage between {left} and {right} in fold {fold}")
        outer_seen.update(calls["outer_eval"])
        folds.append(
            FoldRows(
                outer_fold=fold,
                train=partitions["train"],
                val=partitions["val"],
                outer_eval=partitions["outer_eval"],
            )
        )
    expected_folds = int(protocol.get("summary", {}).get("outer_folds", 5))
    if len(folds) != expected_folds or [fold.outer_fold for fold in folds] != list(range(expected_folds)):
        raise ShortcutProbeError(f"expected canonical outer folds 0..{expected_folds - 1}")
    assignments = _read_jsonl(work6_root / "assignments.jsonl")
    eligible = {str(row["call_id"]) for row in assignments if row.get("role") == "eligible"}
    if set(outer_seen) != eligible or any(count != 1 for count in outer_seen.values()):
        raise ShortcutProbeError("every eligible call must appear in outer_eval exactly once")
    frozen_identity = {
        "work6_protocol_sha256": sha256_file(protocol_path),
        "work6_checksum_manifest_sha256": sha256_file(checksum_path),
        "verified_file_count": len(checksums),
        "eligible_call_count": len(eligible),
    }
    return protocol, folds, frozen_identity


@dataclass(frozen=True)
class CallFeatures:
    call_id: str
    label: int
    text: str
    numeric: Mapping[str, float]


NUMERIC_FEATURES = {
    "call_duration_seconds": lambda row: row["audio"]["duration_seconds"],
    "segment_count": lambda row: row["transcript"]["segment_count"],
    "word_count": lambda row: row["transcript"]["word_count"],
    "mean_segment_duration_seconds": lambda row: row["transcript"]["mean_segment_duration_seconds"],
    "mean_segment_confidence": lambda row: row["transcript"]["mean_segment_confidence"],
    "local_speaker_count": lambda row: row["transcript"]["local_speaker_count"],
    "words_per_second": lambda row: (
        row["transcript"]["word_count"] / row["audio"]["duration_seconds"]
    ),
}


def load_call_features(
    *,
    project_root: Path,
    provenance_path: Path,
    expected_provenance_sha256: str,
    eligible_call_ids: set[str],
) -> dict[str, CallFeatures]:
    actual_hash = sha256_file(provenance_path)
    if actual_hash != expected_provenance_sha256:
        raise ShortcutProbeError(
            "call provenance does not match the artifact bound into Work 6: "
            f"{actual_hash} != {expected_provenance_sha256}"
        )
    rows = {str(row["call_id"]): row for row in _read_jsonl(provenance_path)}
    missing = sorted(eligible_call_ids - set(rows))
    if missing:
        raise ShortcutProbeError(f"eligible calls missing from provenance: {missing[:5]}")
    features: dict[str, CallFeatures] = {}
    for call_id in sorted(eligible_call_ids):
        row = rows[call_id]
        transcript = row.get("transcript")
        if not isinstance(transcript, Mapping):
            raise ShortcutProbeError(f"{call_id} lacks transcript provenance")
        relative_path = Path(str(transcript.get("relative_path", "")))
        transcript_path = relative_path if relative_path.is_absolute() else project_root / relative_path
        if not transcript_path.is_file():
            raise ShortcutProbeError(f"transcript file is missing for {call_id}")
        expected_hash = str(transcript.get("byte_sha256", ""))
        if sha256_file(transcript_path) != expected_hash:
            raise ShortcutProbeError(f"transcript hash mismatch for {call_id}")
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        raw_text = payload.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ShortcutProbeError(f"top-level transcript text is absent for {call_id}")
        numeric: dict[str, float] = {}
        for name, extractor in NUMERIC_FEATURES.items():
            try:
                value = float(extractor(row))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                raise ShortcutProbeError(f"invalid numeric feature {name} for {call_id}") from error
            if not math.isfinite(value):
                raise ShortcutProbeError(f"nonfinite numeric feature {name} for {call_id}")
            numeric[name] = value
        features[call_id] = CallFeatures(
            call_id=call_id,
            label=int(row["label"]),
            text=normalize_text_strict(raw_text),
            numeric=numeric,
        )
    return features


def call_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ShortcutProbeError("metric inputs must be nonempty and aligned")
    if not np.isfinite(y_prob).all() or np.any((y_prob < 0) | (y_prob > 1)):
        raise ShortcutProbeError("probabilities must be finite and within [0,1]")
    if set(y_true) != {0, 1}:
        raise ShortcutProbeError("each diagnostic metric partition must contain both labels")
    y_pred = (y_prob >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "call_count": len(y_true),
        "label_counts": {"0": int((y_true == 0).sum()), "1": int((y_true == 1).sum())},
        "threshold": round(float(threshold), 12),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 12),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 12),
        "positive_f1": round(float(f1_score(y_true, y_pred)), 12),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 12),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 12),
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 12),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 12),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 12),
        "confusion_matrix_label_order_0_1": matrix.tolist(),
    }


def select_validation_threshold(
    labels: Sequence[int], probabilities: Sequence[float], specification: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    start = float(specification["grid_start"])
    stop = float(specification["grid_stop"])
    step = float(specification["grid_step"])
    count = int(round((stop - start) / step)) + 1
    grid = [round(start + index * step, 12) for index in range(count)]
    candidates: list[tuple[float, float, float, dict[str, Any]]] = []
    for threshold in grid:
        metrics = call_metrics(labels, probabilities, threshold)
        candidates.append(
            (-float(metrics["macro_f1"]), abs(threshold - 0.5), threshold, metrics)
        )
    _, _, threshold, metrics = min(candidates, key=lambda row: row[:3])
    return threshold, metrics


def _labels_for(rows: Sequence[Mapping[str, Any]], features: Mapping[str, CallFeatures]) -> np.ndarray:
    labels: list[int] = []
    for row in rows:
        call_id = str(row["call_id"])
        expected = int(row["label"])
        observed = features[call_id].label
        if expected != observed:
            raise ShortcutProbeError(f"fold/provenance label mismatch for {call_id}")
        labels.append(observed)
    return np.asarray(labels, dtype=np.int64)


def _numeric_matrix(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, CallFeatures],
    names: Sequence[str],
) -> np.ndarray:
    unknown = sorted(set(names) - set(NUMERIC_FEATURES))
    if unknown:
        raise ShortcutProbeError(f"unsupported numeric feature(s): {unknown}")
    return np.asarray(
        [[features[str(row["call_id"])].numeric[name] for name in names] for row in rows],
        dtype=np.float64,
    )


def _prediction_rows(
    *,
    probe_id: str,
    fold: int,
    partition: str,
    manifest_rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, probability in zip(manifest_rows, probabilities, strict=True):
        output.append(
            {
                "schema_version": "paper2-v2-shortcut-call-prediction-1",
                "probe_id": probe_id,
                "partition_repeat": int(row["partition_repeat"]),
                "outer_fold": fold,
                "partition": partition,
                "call_id": str(row["call_id"]),
                "group_id": str(row["group_id"]),
                "label": int(row["label"]),
                "probability": round(float(probability), 12),
                "threshold": round(float(threshold), 12),
                "prediction": int(float(probability) >= threshold),
            }
        )
    return output


def _fit_logistic(config: Mapping[str, Any], x_train: Any, y_train: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=float(config["C"]),
        class_weight=config["class_weight"],
        solver=str(config["solver"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tolerance"]),
        random_state=int(config.get("random_seed", 0)),
    )
    model.fit(x_train, y_train)
    if int(np.max(model.n_iter_)) >= int(config["max_iter"]):
        raise ShortcutProbeError("logistic regression reached max_iter without convergence")
    return model


def _aggregate_prediction_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    predictions = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "call_count": len(rows),
        "label_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "threshold_strategy": "fold-specific validation-selected threshold",
        "thresholds_used": sorted({round(float(row["threshold"]), 12) for row in rows}),
        "accuracy": round(float(accuracy_score(labels, predictions)), 12),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 12),
        "positive_f1": round(float(f1_score(labels, predictions)), 12),
        "mcc": round(float(matthews_corrcoef(labels, predictions)), 12),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 12),
        "average_precision": round(float(average_precision_score(labels, probabilities)), 12),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1])), 12),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 12),
        "confusion_matrix_label_order_0_1": matrix.tolist(),
    }


def _fold_metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "accuracy",
        "macro_f1",
        "positive_f1",
        "mcc",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
    )
    result: dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray([float(row["outer_eval_metrics"][metric]) for row in rows])
        result[metric] = {
            "mean_across_outer_folds": round(float(values.mean()), 12),
            "population_standard_deviation": round(float(values.std(ddof=0)), 12),
            "minimum": round(float(values.min()), 12),
            "maximum": round(float(values.max()), 12),
        }
    return result


def run_learned_shortcut_probes(
    *,
    project_root: Path,
    config_path: Path,
    work6_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    work6_root = work6_root.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {destination}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("diagnostic_only"):
        raise ShortcutProbeError("shortcut probe config must explicitly declare diagnostic_only=true")
    protocol, folds, frozen_identity = load_frozen_folds(work6_root)
    if protocol.get("protocol_id") != config["fold_protocol_id"]:
        raise ShortcutProbeError("probe config targets a different Work 6 protocol")
    if int(protocol.get("partition_repeat", -1)) != int(config["partition_repeat"]):
        raise ShortcutProbeError("probe config partition repeat disagrees with Work 6")

    work6_provenance = protocol["input_artifacts"]["call_provenance"]
    provenance_path = Path(str(work6_provenance["path"])).resolve()
    eligible = {
        str(row["call_id"])
        for fold in folds
        for row in fold.outer_eval
    }
    features = load_call_features(
        project_root=project_root,
        provenance_path=provenance_path,
        expected_provenance_sha256=str(work6_provenance["sha256"]),
        eligible_call_ids=eligible,
    )
    logistic_config = dict(config["logistic_regression"])
    logistic_config["random_seed"] = int(config["random_seed"])
    threshold_spec = config["threshold_selection"]
    fold_metrics: list[dict[str, Any]] = []
    outer_predictions: list[dict[str, Any]] = []
    validation_predictions: list[dict[str, Any]] = []

    for fold in folds:
        y_train = _labels_for(fold.train, features)
        y_val = _labels_for(fold.val, features)
        y_outer = _labels_for(fold.outer_eval, features)
        for probe in config["probes"]:
            probe_id = str(probe["probe_id"])
            kind = str(probe["kind"])
            if kind == "text":
                from sklearn.feature_extraction.text import TfidfVectorizer

                vectorizer_spec = probe["vectorizer"]
                vectorizer = TfidfVectorizer(
                    analyzer=str(vectorizer_spec["analyzer"]),
                    ngram_range=tuple(int(value) for value in vectorizer_spec["ngram_range"]),
                    lowercase=bool(vectorizer_spec["lowercase"]),
                    min_df=int(vectorizer_spec["min_df"]),
                    max_df=float(vectorizer_spec["max_df"]),
                    norm=str(vectorizer_spec["norm"]),
                    use_idf=bool(vectorizer_spec["use_idf"]),
                    smooth_idf=bool(vectorizer_spec["smooth_idf"]),
                    sublinear_tf=bool(vectorizer_spec["sublinear_tf"]),
                    dtype=np.float64,
                )
                train_text = [features[str(row["call_id"])].text for row in fold.train]
                val_text = [features[str(row["call_id"])].text for row in fold.val]
                outer_text = [features[str(row["call_id"])].text for row in fold.outer_eval]
                x_train = vectorizer.fit_transform(train_text)
                x_val = vectorizer.transform(val_text)
                model = _fit_logistic(logistic_config, x_train, y_train)
                val_probability = model.predict_proba(x_val)[:, 1]
                model_audit = {
                    "input_kind": "full-call normalized text",
                    "feature_count": int(x_train.shape[1]),
                    "training_nonzero_count": int(x_train.nnz),
                    "vectorizer_fitted_on": "outer_train_only",
                    "text_coefficients_or_ngrams_saved": False,
                    "convergence_iterations": int(np.max(model.n_iter_)),
                }
                threshold, val_metrics = select_validation_threshold(
                    y_val, val_probability, threshold_spec
                )
                # Outer evaluation is transformed and scored only after the
                # train-only model and validation-only threshold are fixed.
                x_outer = vectorizer.transform(outer_text)
                outer_probability = model.predict_proba(x_outer)[:, 1]
            elif kind == "numeric":
                from sklearn.preprocessing import StandardScaler

                names = tuple(str(name) for name in probe["features"])
                x_train_raw = _numeric_matrix(fold.train, features, names)
                x_val_raw = _numeric_matrix(fold.val, features, names)
                scaler = StandardScaler().fit(x_train_raw)
                x_train = scaler.transform(x_train_raw)
                x_val = scaler.transform(x_val_raw)
                model = _fit_logistic(logistic_config, x_train, y_train)
                val_probability = model.predict_proba(x_val)[:, 1]
                model_audit = {
                    "input_kind": "numeric call metadata",
                    "feature_names": list(names),
                    "feature_count": len(names),
                    "scaler_fitted_on": "outer_train_only",
                    "scaler_mean": [round(float(value), 12) for value in scaler.mean_],
                    "scaler_scale": [round(float(value), 12) for value in scaler.scale_],
                    "convergence_iterations": int(np.max(model.n_iter_)),
                }
                threshold, val_metrics = select_validation_threshold(
                    y_val, val_probability, threshold_spec
                )
                x_outer_raw = _numeric_matrix(fold.outer_eval, features, names)
                outer_probability = model.predict_proba(scaler.transform(x_outer_raw))[:, 1]
            else:
                raise ShortcutProbeError(f"unsupported probe kind {kind!r}")

            outer_metrics = call_metrics(y_outer, outer_probability, threshold)
            val_rows = _prediction_rows(
                probe_id=probe_id,
                fold=fold.outer_fold,
                partition="val",
                manifest_rows=fold.val,
                probabilities=val_probability,
                threshold=threshold,
            )
            outer_rows = _prediction_rows(
                probe_id=probe_id,
                fold=fold.outer_fold,
                partition="outer_eval",
                manifest_rows=fold.outer_eval,
                probabilities=outer_probability,
                threshold=threshold,
            )
            validation_predictions.extend(val_rows)
            outer_predictions.extend(outer_rows)
            fold_metrics.append(
                {
                    "schema_version": "paper2-v2-shortcut-fold-metrics-1",
                    "probe_id": probe_id,
                    "partition_repeat": int(config["partition_repeat"]),
                    "outer_fold": fold.outer_fold,
                    "train_call_count": len(fold.train),
                    "train_group_count": len({str(row["group_id"]) for row in fold.train}),
                    "val_call_count": len(fold.val),
                    "val_group_count": len({str(row["group_id"]) for row in fold.val}),
                    "outer_eval_call_count": len(fold.outer_eval),
                    "outer_eval_group_count": len(
                        {str(row["group_id"]) for row in fold.outer_eval}
                    ),
                    "selected_threshold": round(float(threshold), 12),
                    "threshold_selected_on": "val",
                    "validation_metrics": val_metrics,
                    "outer_eval_metrics": outer_metrics,
                    "model_audit": model_audit,
                }
            )

    probe_ids = [str(probe["probe_id"]) for probe in config["probes"]]
    aggregate: dict[str, Any] = {}
    for probe_id in probe_ids:
        predictions = [row for row in outer_predictions if row["probe_id"] == probe_id]
        seen = Counter(str(row["call_id"]) for row in predictions)
        if set(seen) != eligible or set(seen.values()) != {1}:
            raise ShortcutProbeError(f"{probe_id} did not outer-evaluate every eligible call once")
        probe_fold_metrics = [row for row in fold_metrics if row["probe_id"] == probe_id]
        aggregate[probe_id] = {
            "pooled_outer_eval_call_metrics": _aggregate_prediction_metrics(predictions),
            "outer_fold_distribution": _fold_metric_summary(probe_fold_metrics),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    environment = {
        "numpy": np.__version__,
    }
    import sklearn

    environment["scikit_learn"] = sklearn.__version__
    output_protocol = {
        "schema_version": PROBE_SCHEMA,
        "artifact_id": config["artifact_id"],
        "artifact_status": "sealed_diagnostic",
        "diagnostic_only": True,
        "must_not_change_folds_or_work8": True,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "work6_root": str(work6_root),
        "work6_identity": frozen_identity,
        "call_provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        },
        "selection_contract": {
            "vectorizers_scalers_and_models": "outer_train_only",
            "threshold": "outer validation only",
            "outer_eval": "scored once after model and threshold fixed",
            "outer_eval_feedback": "forbidden",
        },
        "environment": environment,
    }
    audit = {
        "status": "pass",
        "fold_count": len(folds),
        "probe_count": len(probe_ids),
        "eligible_call_count": len(eligible),
        "outer_prediction_rows": len(outer_predictions),
        "validation_prediction_rows": len(validation_predictions),
        "outer_eval_once_per_call_per_probe": True,
        "group_disjoint_partitions": True,
        "work6_hashes_verified": True,
        "outer_results_used_for_selection": False,
    }
    _write_json(staging / "protocol.json", output_protocol)
    _write_jsonl(
        staging / "fold_metrics.jsonl",
        sorted(fold_metrics, key=lambda row: (row["probe_id"], row["outer_fold"])),
    )
    _write_jsonl(
        staging / "outer_eval_predictions.jsonl",
        sorted(
            outer_predictions,
            key=lambda row: (row["probe_id"], row["outer_fold"], row["call_id"]),
        ),
    )
    _write_jsonl(
        staging / "validation_predictions.jsonl",
        sorted(
            validation_predictions,
            key=lambda row: (row["probe_id"], row["outer_fold"], row["call_id"]),
        ),
    )
    _write_json(staging / "aggregate_metrics.json", aggregate)
    _write_json(staging / "audit.json", audit)
    files = sorted(path for path in staging.iterdir() if path.is_file())
    (staging / "hashes.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    os.replace(staging, destination)
    return {"audit": audit, "aggregate_metrics": aggregate}
