"""Deterministic Work 8 configuration construction and no-training dry run.

The generated artifacts bind the matched MFCC matrix to the frozen Work 6
fold tree and sealed canonical-mean KR-SBERT store.  This module constructs
models and validates real feature contracts, but it never optimizes a model,
selects a checkpoint, or reads outer-evaluation predictions.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .batch import ExperimentBatch
from .conditions import CORE_CONDITIONS, build_condition
from .config import (
    CAPACITY_MATCHED_PARAMETER_COUNTS,
    CAPACITY_MATCHED_WIDTHS,
    MatrixConfig,
    matrix_config_from_mapping,
    sha256_file,
)
from .data import FeatureDataset, FeatureSample, collate_samples
from .folds import tree_digest
from .pairing import PairingItem, build_random_train_pairing, pairing_plan_for_split
from .text_features import IndexedTextStore


CONFIG_RELATIVE_PATH = Path(
    "revised_experiments/configs/work8_mfcc_matched_within_corpus_v1.json"
)
RECEIPT_RELATIVE_PATH = Path(
    "revised_experiments/artifacts/work8_preregistration_v1/dry_run_receipt.json"
)
FOLD_TREE_RELATIVE_PATH = Path("data/paper2_v2/folds/grouped_nested_v1")
GATE0_RELATIVE_PATH = Path(
    "revised_experiments/artifacts/gate0/gate0_receipt_2026-08-09.json"
)
TEXT_STORE_RELATIVE_PATH = Path(
    "revised_experiments/artifacts/text_features/"
    "text-krsbert-92c6c2c7-mean-mask128-f32-v1"
)

PROTOCOL_ID = "paper2-v2-mfcc-matched-within-corpus-v1"
NEURAL_SEEDS = (17, 29, 43, 71, 101)
PAIRING_ALGORITHM_ID = "group_disjoint_circular_shift_largest_group_v1"

SOURCE_CONTRACT_PATHS = (
    "revised_experiments/src/paper2_v2/batch.py",
    "revised_experiments/src/paper2_v2/data.py",
    "revised_experiments/src/paper2_v2/pooling.py",
    "revised_experiments/src/paper2_v2/models.py",
    "revised_experiments/src/paper2_v2/text_features.py",
    "revised_experiments/src/paper2_v2/conditions.py",
    "revised_experiments/src/paper2_v2/pairing.py",
    "revised_experiments/src/paper2_v2/config.py",
    "revised_experiments/src/paper2_v2/evaluation.py",
    "revised_experiments/src/paper2_v2/registry.py",
    "revised_experiments/src/paper2_v2/work8_preregistration.py",
    "revised_experiments/scripts/build_work8_preregistration.py",
)


class PreregistrationError(RuntimeError):
    pass


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PreregistrationError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise PreregistrationError(f"{path}:{row_number} is not an object")
            rows.append(row)
    if not rows:
        raise PreregistrationError(f"required JSONL is empty: {path}")
    return rows


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise PreregistrationError(f"artifact is outside project root: {path}") from error


def _artifact(root: Path, path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative_path(root, path), "sha256": sha256_file(path)}


def _assert_read_only(paths: Iterable[Path], *, context: str) -> None:
    writable = [str(path) for path in paths if path.stat().st_mode & 0o222]
    if writable:
        raise PreregistrationError(
            f"{context} contains files with write bits set: {writable[:5]}"
        )


def _capacity_mapping() -> dict[str, Any]:
    return {
        "target_condition": "AT-cross",
        "target_parameter_count": 530_689,
        "matching_rule": "nearest_count_width_multiple_of_8_search_8_to_1024",
        "maximum_relative_gap": 0.05,
        "matched_hidden_dims": dict(CAPACITY_MATCHED_WIDTHS),
        "matched_parameter_counts": dict(CAPACITY_MATCHED_PARAMETER_COUNTS),
    }


def build_config_mapping(project_root: str | Path) -> dict[str, Any]:
    """Construct the byte-deterministic final configuration from real artifacts."""

    root = Path(project_root).resolve()
    fold_root = root / FOLD_TREE_RELATIVE_PATH
    protocol_path = fold_root / "protocol.json"
    assignments_path = fold_root / "assignments.jsonl"
    checksums_path = fold_root / "SHA256SUMS.txt"
    gate0_path = root / GATE0_RELATIVE_PATH
    text_store_root = root / TEXT_STORE_RELATIVE_PATH
    text_metadata_path = text_store_root / "feature_store.json"

    protocol = _load_json(protocol_path)
    if protocol.get("artifact_status") != "frozen":
        raise PreregistrationError("Work 6 protocol must be frozen before Work 8")
    if protocol.get("terminology", {}).get("outer_partition_name") != "outer_eval":
        raise PreregistrationError("Work 6 protocol does not use outer_eval terminology")
    store = _load_json(text_metadata_path)
    if store.get("status") != "sealed":
        raise PreregistrationError("canonical text store is not sealed")

    fold_files = [path for path in fold_root.rglob("*") if path.is_file()]
    store_files = [path for path in text_store_root.iterdir() if path.is_file()]
    _assert_read_only(fold_files, context="frozen Work 6 tree")
    _assert_read_only(store_files, context="sealed Work 4 text store")
    _assert_read_only((gate0_path,), context="Gate 0 receipt")

    folds: list[dict[str, Any]] = []
    for fold in sorted(
        protocol.get("folds", []),
        key=lambda item: (int(item["partition_repeat"]), int(item["outer_fold"])),
    ):
        item: dict[str, Any] = {
            "partition_repeat": int(fold["partition_repeat"]),
            "outer_fold": int(fold["outer_fold"]),
        }
        for role in ("train", "val", "outer_eval"):
            relative = Path(fold["manifests"][role]["segment_manifest"])
            item[f"{role}_manifest"] = _artifact(root, fold_root / relative)
        folds.append(item)

    config: dict[str, Any] = {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "status": "locked",
        "conditions": list(CORE_CONDITIONS),
        "seeds": list(NEURAL_SEEDS),
        "gate0_receipt": _artifact(root, gate0_path),
        "exclusion_inventory": _artifact(root, assignments_path),
        "fold_protocol": {
            "origin": "paper2_v2_grouped_nested",
            "protocol_id": str(protocol["protocol_id"]),
            "grouping_policy_id": str(protocol["grouping_policy_id"]),
            "protocol_artifact": _artifact(root, protocol_path),
            "assignments_artifact": _artifact(root, assignments_path),
            "checksums_artifact": _artifact(root, checksums_path),
            "tree_sha256": tree_digest(fold_root),
            "folds": folds,
        },
        "text_store": {
            "feature_set_id": str(store["feature_set_id"]),
            "model_id": str(store["model_id"]),
            "model_revision": str(store["model_revision"]),
            "pooling": str(store["pooling"]),
            "dtype": str(store["dtype"]),
            "vector_dim": int(store["shape"][1]),
            "index_sha256": str(store["index_sha256"]),
            "vectors_sha256": str(store["vectors_sha256"]),
            "metadata": _artifact(root, text_metadata_path),
        },
        "model": {
            "hidden_dim": 256,
            "num_heads": 8,
            "dropout": 0.2,
            "text_dim": 768,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "batch_size": 16,
            "epoch_budget": 50,
            "early_stop_patience": 8,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "minimum_learning_rate": 0.000001,
            "loss": "BCEWithLogitsLoss",
            "sampling": "inverse_segments_per_call_loss_weight",
            "augmentation": "none",
            "scheduler_monitor": "validation_call_nll",
            "early_stopping_monitor": "validation_call_nll",
            "mixed_precision": False,
            "deterministic_algorithms": True,
            "dataloader_workers": 0,
            "shuffle_train": True,
            "drop_last": False,
            "seed_sources": [
                "python",
                "numpy",
                "torch_cpu",
                "torch_cuda",
                "dataloader",
            ],
        },
        "pairing": {
            "condition": "AT-cross-randtrain",
            "algorithm_id": PAIRING_ALGORITHM_ID,
            "donor_scope": "same_fold_train_manifest",
            "label_policy": "unconstrained_global_random",
            "randomized_roles": ["train"],
            "correct_pair_roles": ["val", "outer_eval"],
            "seed_source": "neural_seed",
            "one_to_one": True,
            "forbid_same_segment": True,
            "forbid_same_call": True,
            "forbid_same_group": True,
        },
        "selection": {
            "independent_unit": "call",
            "primary_metric": "macro_f1",
            "aggregation_rule": "mean_probability",
            "checkpoint_monitor": "validation_call_nll",
            "threshold_start": 0.01,
            "threshold_end": 0.99,
            "threshold_step": 0.01,
            "threshold_tie_break": [
                "lower_fpr",
                "higher_sensitivity",
                "closest_to_0.5",
                "higher_threshold",
            ],
            "calibration_rule": "none",
            "outer_evaluation_once": True,
        },
        "statistics": {
            "primary_comparisons": [
                ["AT-cross", "A-query"],
                ["AT-cross", "AT-concat"],
                ["AT-cross", "AT-gate"],
            ],
            "primary_estimand": (
                "mean_across_seeds_of_paired_call_macro_f1_difference"
            ),
            "reported_call_metrics": [
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
            ],
            "segment_metrics_role": "secondary_weak_label",
            "bootstrap_method": (
                "paired_relation_group_cluster_keep_all_calls_segments_seeds"
            ),
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20260809,
            "confidence_level": 0.95,
            "permutation_method": (
                "paired_relation_group_cluster_swap_all_calls_seeds"
            ),
            "permutation_replicates": 10_000,
            "permutation_seed": 20260810,
            "multiplicity_method": "holm",
            "familywise_alpha": 0.05,
            "practical_claim_rule": (
                "no_practical_utility_claim_without_pre_work9_cost_policy"
            ),
            "smallest_practically_meaningful_effect": None,
        },
        "capacity_sensitivity": _capacity_mapping(),
    }
    return config


def _verify_gate0(config: MatrixConfig, root: Path) -> dict[str, Any]:
    payload = _load_json(config.gate0_receipt.path)
    if payload.get("status") != "pass":
        raise PreregistrationError("Gate 0 receipt status is not pass")
    verified_files = []
    for item in payload.get("files", []):
        path = root / str(item["path"])
        expected = str(item["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise PreregistrationError(
                f"Gate 0 evidence drift for {path}: expected {expected}, got {actual}"
            )
        verified_files.append(str(item["path"]))
    if not verified_files:
        raise PreregistrationError("Gate 0 receipt contains no verified files")
    return {
        "receipt_sha256": config.gate0_receipt.sha256,
        "status": "pass",
        "verified_file_count": len(verified_files),
    }


def _manifest_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls: dict[str, int] = {}
    groups: set[str] = set()
    for row in rows:
        call_id = str(row["call_id"])
        group_id = str(row["group_id"])
        label = int(row["label"])
        if label not in {0, 1}:
            raise PreregistrationError(f"invalid label in manifest: {label}")
        if call_id in calls and calls[call_id] != label:
            raise PreregistrationError(f"inconsistent label for call {call_id}")
        calls[call_id] = label
        groups.add(group_id)
    label_counts = Counter(calls.values())
    return {
        "call_count": len(calls),
        "group_count": len(groups),
        "segment_count": len(rows),
        "label_call_counts": {
            "0": int(label_counts[0]),
            "1": int(label_counts[1]),
        },
    }


def _verify_sample_weights(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    segment_counts = Counter(str(row["call_id"]) for row in rows)
    for row in rows:
        call_id = str(row["call_id"])
        expected = 1.0 / segment_counts[call_id]
        actual = float(row["sample_weight"])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
            raise PreregistrationError(
                f"{path}: inverse-call sample weight mismatch for {call_id}"
            )


def _inventory_manifests(
    config: MatrixConfig,
    store: IndexedTextStore,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]], dict[str, Any]]:
    protocol = _load_json(config.fold_protocol_artifact.path)
    expected_summaries = {
        (int(fold["partition_repeat"]), int(fold["outer_fold"]), role):
        fold["summary"][role]
        for fold in protocol["folds"]
        for role in ("train", "val", "outer_eval")
    }
    fold_receipts: list[dict[str, Any]] = []
    train_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
    canonical_segments: dict[str, tuple[Any, ...]] = {}
    canonical_rows: dict[str, dict[str, Any]] = {}
    source_label_calls: dict[str, tuple[str, int]] = {}

    for fold in config.folds:
        role_receipts: dict[str, Any] = {}
        for role, artifact in (
            ("train", fold.train_manifest),
            ("val", fold.val_manifest),
            ("outer_eval", fold.outer_eval_manifest),
        ):
            rows = _load_jsonl(artifact.path)
            _verify_sample_weights(rows, artifact.path)
            summary = _manifest_summary(rows)
            wanted = expected_summaries[(fold.partition_repeat, fold.outer_fold, role)]
            if summary != wanted:
                raise PreregistrationError(
                    f"manifest summary drift for fold {fold.key} {role}: "
                    f"expected {wanted}, got {summary}"
                )
            if role == "train":
                train_rows[fold.key] = rows
            for row in rows:
                segment_id = str(row["segment_id"])
                identity = (
                    str(row["call_id"]),
                    str(row["group_id"]),
                    int(row["label"]),
                    str(row["source"]),
                    float(row["start"]),
                    float(row["end"]),
                    str(row["mfcc_path"]),
                )
                prior = canonical_segments.setdefault(segment_id, identity)
                if prior != identity:
                    raise PreregistrationError(
                        f"segment metadata changes across folds: {segment_id}"
                    )
                canonical_rows.setdefault(segment_id, dict(row))
                call_id, source, label = identity[0], identity[3], identity[2]
                prior_call = source_label_calls.setdefault(call_id, (source, label))
                if prior_call != (source, label):
                    raise PreregistrationError(f"call metadata changes across folds: {call_id}")
            role_receipts[role] = {
                "manifest_sha256": artifact.sha256,
                **summary,
            }
        fold_receipts.append(
            {
                "partition_repeat": fold.partition_repeat,
                "outer_fold": fold.outer_fold,
                "roles": role_receipts,
                "selection_view_exposes_outer_eval": hasattr(
                    config.selection_view(fold.partition_repeat, fold.outer_fold),
                    "outer_eval_manifest",
                ),
            }
        )

    missing_text: list[str] = []
    missing_mfcc: list[str] = []
    for segment_id, row in canonical_rows.items():
        if segment_id not in store:
            missing_text.append(segment_id)
        mfcc_path = Path(str(row["mfcc_path"]))
        resolved_mfcc = mfcc_path if mfcc_path.is_absolute() else root / mfcc_path
        if not resolved_mfcc.is_file():
            missing_mfcc.append(segment_id)
    if missing_text or missing_mfcc:
        raise PreregistrationError(
            "eligible feature coverage failed: "
            f"missing_text={missing_text[:5]}, missing_mfcc={missing_mfcc[:5]}"
        )

    source_label_counts = Counter(source_label_calls.values())
    inventory = {
        "unique_call_count": len(source_label_calls),
        "unique_segment_count": len(canonical_rows),
        "text_store_missing_segments": 0,
        "mfcc_missing_segments": 0,
        "source_label_call_counts": [
            {"source": source, "label": label, "call_count": count}
            for (source, label), count in sorted(source_label_counts.items())
        ],
    }
    return fold_receipts, train_rows, {"inventory": inventory, "rows": canonical_rows}


def _make_actual_batch(
    root: Path,
    row: Mapping[str, Any],
    *,
    requires_mfcc: bool,
    requires_text: bool,
    store: IndexedTextStore,
    donor_segment_id: str | None = None,
) -> ExperimentBatch:
    sequences = {}
    globals_ = {}
    if requires_mfcc:
        mfcc_path = Path(str(row["mfcc_path"]))
        resolved = mfcc_path if mfcc_path.is_absolute() else root / mfcc_path
        sequences["mfcc"] = FeatureDataset._load_array(
            resolved, expected_dim=13, sequence=True
        )
    if requires_text:
        globals_["text"] = store.get(donor_segment_id or str(row["segment_id"]))
    sample = FeatureSample(
        sequences=sequences,
        globals=globals_,
        label=float(row["label"]),
        sample_weight=float(row["sample_weight"]),
        call_id=str(row["call_id"]),
        segment_id=str(row["segment_id"]),
        group_id=str(row["group_id"]),
        source=str(row["source"]),
        start=float(row["start"]),
        end=float(row["end"]),
    )
    built = build_condition(
        "AT-cross" if requires_mfcc and requires_text else (
            "A-mean" if requires_mfcc else "T"
        )
    )
    return collate_samples((sample,), built.input_spec)


def _condition_receipts(
    config: MatrixConfig,
    train_rows: Mapping[tuple[int, int], list[dict[str, Any]]],
    pairing_plans: Mapping[tuple[int, int, int], Any],
    store: IndexedTextStore,
    root: Path,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    state_keys: dict[str, tuple[str, ...]] = {}
    for condition_index, condition in enumerate(CORE_CONDITIONS):
        torch.manual_seed(20260809 + condition_index)
        built = config.instantiate_condition(condition)
        state_keys[condition] = tuple(built.model.state_dict())
        fold_checks = []
        for fold_key in sorted(train_rows):
            rows = train_rows[fold_key]
            row = min(rows, key=lambda item: str(item["segment_id"]))
            donor_segment_id = None
            if condition == "AT-cross-randtrain":
                donor_segment_id = pairing_plans[
                    (fold_key[0], fold_key[1], config.seeds[0])
                ].donor_segment_id(str(row["segment_id"]))
            batch = _make_actual_batch(
                root,
                row,
                requires_mfcc=built.definition.requires_mfcc,
                requires_text=built.definition.requires_text,
                store=store,
                donor_segment_id=donor_segment_id,
            )
            built.model.eval()
            with torch.inference_mode():
                logits = built.model(batch)
            if logits.shape != (1,) or not bool(torch.isfinite(logits).all()):
                raise PreregistrationError(
                    f"dry forward failed for {condition}, fold {fold_key}"
                )
            fold_checks.append(
                {
                    "partition_repeat": fold_key[0],
                    "outer_fold": fold_key[1],
                    "receiver_segment_id": str(row["segment_id"]),
                    "donor_segment_id": donor_segment_id,
                    "logit_shape": [1],
                    "finite": True,
                }
            )
        receipts.append(
            {
                "condition": condition,
                "primary_hidden_dim": config.model.hidden_dim,
                "primary_parameter_count": built.total_parameters,
                "trainable_parameter_count": built.trainable_parameters,
                "sequence_features": list(built.input_spec.sequence_features),
                "global_features": list(built.input_spec.global_features),
                "pairing_mode": built.definition.pairing_mode.value,
                "text_resolution": (
                    "indexed_store_by_segment_id"
                    if built.definition.requires_text
                    else "not_requested"
                ),
                "fold_forward_checks": fold_checks,
            }
        )
    if state_keys["AT-cross"] != state_keys["AT-cross-randtrain"]:
        raise PreregistrationError("paired/random-train cross models changed architecture")
    return receipts


def _pairing_receipts(
    config: MatrixConfig,
    train_rows: Mapping[tuple[int, int], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int], Any]]:
    receipts: list[dict[str, Any]] = []
    plans: dict[tuple[int, int, int], Any] = {}
    for fold in config.folds:
        rows = train_rows[fold.key]
        items = [PairingItem.from_mapping(row) for row in rows]
        label_by_segment = {
            str(row["segment_id"]): int(row["label"])
            for row in rows
        }
        for seed in config.seeds:
            plan = build_random_train_pairing(items, seed=seed)
            repeated = build_random_train_pairing(items, seed=seed)
            if plan.sha256 != repeated.sha256:
                raise PreregistrationError(
                    f"pairing plan is nondeterministic for fold {fold.key}, seed {seed}"
                )
            if pairing_plan_for_split(
                "AT-cross-randtrain", split_role="val", rows=(), seed=seed
            ) is not None:
                raise PreregistrationError("random pairing leaked into validation")
            if pairing_plan_for_split(
                "AT-cross-randtrain", split_role="outer_eval", rows=(), seed=seed
            ) is not None:
                raise PreregistrationError("random pairing leaked into outer_eval")
            key = (fold.partition_repeat, fold.outer_fold, seed)
            plans[key] = plan
            same_label_pairs = sum(
                label_by_segment[receiver] == label_by_segment[donor]
                for receiver, donor in plan.donor_by_receiver.items()
            )
            opposite_label_pairs = len(plan.donor_by_receiver) - same_label_pairs
            receipts.append(
                {
                    "partition_repeat": fold.partition_repeat,
                    "outer_fold": fold.outer_fold,
                    "seed": seed,
                    "condition": "AT-cross-randtrain",
                    "split_role": "train",
                    "train_manifest_sha256": fold.train_manifest.sha256,
                    "receiver_count": len(plan.donor_by_receiver),
                    "donor_count": len(set(plan.donor_by_receiver.values())),
                    "same_segment_pairs": 0,
                    "same_call_pairs": 0,
                    "same_group_pairs": 0,
                    "label_policy": "unconstrained_global_random",
                    "same_label_pairs": same_label_pairs,
                    "opposite_label_pairs": opposite_label_pairs,
                    "same_label_fraction": same_label_pairs / len(plan.donor_by_receiver),
                    "plan_sha256": plan.sha256,
                    "validation_pairing": "correct",
                    "outer_eval_pairing": "correct",
                }
            )
    return receipts, plans


def _capacity_receipts(config: MatrixConfig) -> list[dict[str, Any]]:
    receipts = []
    for condition in CORE_CONDITIONS:
        hidden_dim = config.capacity_sensitivity.matched_hidden_dims[condition]
        built = build_condition(
            condition,
            hidden_dim=hidden_dim,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            text_dim=config.model.text_dim,
        )
        expected = config.capacity_sensitivity.matched_parameter_counts[condition]
        if built.total_parameters != expected:
            raise PreregistrationError(f"capacity count drift for {condition}")
        receipts.append(
            {
                "condition": condition,
                "hidden_dim": hidden_dim,
                "parameter_count": built.total_parameters,
                "relative_gap_from_target": abs(
                    built.total_parameters
                    - config.capacity_sensitivity.target_parameter_count
                )
                / config.capacity_sensitivity.target_parameter_count,
            }
        )
    return receipts


def build_dry_run_receipt(
    project_root: str | Path,
    *,
    config_mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all Work 8 construction checks without training or predictions."""

    root = Path(project_root).resolve()
    raw = dict(config_mapping or build_config_mapping(root))
    config_bytes = canonical_json_bytes(raw)
    config_sha256 = __import__("hashlib").sha256(config_bytes).hexdigest()
    config = matrix_config_from_mapping(raw, project_root=root)
    gate0 = _verify_gate0(config, root)
    store = IndexedTextStore(config.text_store.metadata.path.parent, verify_hashes=True)
    fold_receipts, train_rows, inventory_payload = _inventory_manifests(
        config, store, root
    )
    pairing_receipts, pairing_plans = _pairing_receipts(config, train_rows)
    condition_receipts = _condition_receipts(
        config, train_rows, pairing_plans, store, root
    )
    capacity_receipts = _capacity_receipts(config)

    source_contracts = []
    for relative in SOURCE_CONTRACT_PATHS:
        path = root / relative
        source_contracts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )

    inventory = inventory_payload["inventory"]
    source_label_cells = {
        (item["source"], item["label"])
        for item in inventory["source_label_call_counts"]
        if item["call_count"]
    }
    return {
        "schema_version": "paper2-v2-work8-dry-run-receipt-1",
        "status": "pass",
        "protocol_id": config.protocol_id,
        "meaning": (
            "Operational preregistration and no-training construction checks passed. "
            "This is not evidence of model effectiveness or independent generalization."
        ),
        "claim_scope": "within_corpus_source_confounded_methodological_evidence_only",
        "config": {
            "path": CONFIG_RELATIVE_PATH.as_posix(),
            "sha256": config_sha256,
            "byte_count": len(config_bytes),
            "status": "locked",
        },
        "gate0": gate0,
        "work6": {
            "protocol_sha256": config.fold_protocol_artifact.sha256,
            "assignments_sha256": config.fold_assignments_artifact.sha256,
            "checksums_sha256": config.fold_checksums_artifact.sha256,
            "tree_sha256": config.fold_tree_sha256,
            "artifact_status": "frozen",
            "folds": fold_receipts,
        },
        "text_store": {
            "feature_set_id": store.feature_set_id,
            "metadata_sha256": config.text_store.metadata.sha256,
            "index_sha256": config.text_store.index_sha256,
            "vectors_sha256": config.text_store.vectors_sha256,
            "shape": [len(store), store.dimension],
            "status": "sealed",
        },
        "eligible_inventory": inventory,
        "source_label_counterfactual_cells_observed": len(source_label_cells) == 4,
        "conditions": condition_receipts,
        "capacity_sensitivity": capacity_receipts,
        "random_train_pairing_plans": pairing_receipts,
        "pairing_plan_count": len(pairing_receipts),
        "source_contracts": source_contracts,
        "safety": {
            "training_launched": False,
            "optimizer_steps": 0,
            "checkpoints_created": 0,
            "outer_eval_manifests_hashed_and_structure_validated": True,
            "outer_eval_predictions_accessed": False,
            "outer_eval_authorization_claimed": False,
            "legacy_cv_5fold_accessed": False,
        },
    }


def _write_once_read_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def write_preregistration(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(project_root).resolve()
    config_target = Path(config_path).resolve() if config_path else root / CONFIG_RELATIVE_PATH
    receipt_target = (
        Path(receipt_path).resolve() if receipt_path else root / RECEIPT_RELATIVE_PATH
    )
    if config_target.exists() or receipt_target.exists():
        raise FileExistsError(
            "refusing to overwrite frozen Work 8 preregistration artifacts"
        )
    raw = build_config_mapping(root)
    config_bytes = canonical_json_bytes(raw)
    receipt = build_dry_run_receipt(root, config_mapping=raw)
    receipt_bytes = canonical_json_bytes(receipt)
    _write_once_read_only(config_target, config_bytes)
    _write_once_read_only(receipt_target, receipt_bytes)
    return config_target, receipt_target


def check_preregistration(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> tuple[str, str]:
    root = Path(project_root).resolve()
    config_target = Path(config_path).resolve() if config_path else root / CONFIG_RELATIVE_PATH
    receipt_target = (
        Path(receipt_path).resolve() if receipt_path else root / RECEIPT_RELATIVE_PATH
    )
    expected_config = canonical_json_bytes(build_config_mapping(root))
    actual_config = config_target.read_bytes()
    if actual_config != expected_config:
        raise PreregistrationError("frozen Work 8 config differs from deterministic builder")
    expected_receipt = canonical_json_bytes(
        build_dry_run_receipt(root, config_mapping=json.loads(actual_config))
    )
    actual_receipt = receipt_target.read_bytes()
    if actual_receipt != expected_receipt:
        raise PreregistrationError("frozen Work 8 receipt differs from deterministic dry run")
    _assert_read_only((config_target, receipt_target), context="Work 8 artifacts")
    return sha256_file(config_target), sha256_file(receipt_target)
