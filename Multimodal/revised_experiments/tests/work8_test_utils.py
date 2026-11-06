from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper2_v2.conditions import CORE_CONDITIONS
from paper2_v2.config import sha256_file
from paper2_v2.folds import tree_digest


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def artifact(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def build_valid_raw_config(root: Path) -> dict[str, object]:
    gate0 = root / "revised_experiments/artifacts/gate0/receipt.json"
    write_json(gate0, {"status": "passed"})
    fold_tree = root / "data/paper2_v2/folds/grouped_nested_v1"
    exclusions = fold_tree / "assignments.jsonl"
    eligible = [
        {
            "call_id": f"eligible_call_{index}",
            "historical_split": "train",
            "role": "eligible",
            "group_id": f"eligible_group_{index}",
        }
        for index in range(5)
    ]
    write_jsonl(
        exclusions,
        [
            *eligible,
            {
                "call_id": "historical_burned_call",
                "historical_split": "test",
                "role": "quarantine",
                "exclusion_reason": "burned_historical_split",
            },
        ],
    )
    feature_store = root / "revised_experiments/artifacts/text_features/canonical/feature_store.json"
    feature_store.parent.mkdir(parents=True, exist_ok=True)
    vectors_path = feature_store.parent / "vectors.npy"
    index_path = feature_store.parent / "index.jsonl"
    np.save(vectors_path, np.zeros((15, 768), dtype=np.float32))
    write_jsonl(
        index_path,
        [
            {"segment_id": f"text_segment_{row}", "row_index": row, "text_sha256": "x"}
            for row in range(15)
        ],
    )
    index_sha256 = sha256_file(index_path)
    vectors_sha256 = sha256_file(vectors_path)
    write_json(
        feature_store,
        {
            "status": "sealed",
            "feature_set_id": "text-krsbert-92c6c2c7-mean-mask128-f32-v1",
            "model_id": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
            "model_revision": "92c6c2c7032f680bff0f9f0c63fadd3f97e635b2",
            "pooling": "canonical_attention_masked_mean",
            "dtype": "float32",
            "shape": [15, 768],
            "index_sha256": index_sha256,
            "vectors_sha256": vectors_sha256,
            "storage": {"layout": "matrix_npy"},
        },
    )

    folds = []
    protocol_folds = []
    for outer_fold in range(5):
        fold_root = (
            root
            / "data/paper2_v2/folds/grouped_nested_v1"
            / "repeat_00"
            / f"outer_{outer_fold:02d}"
        )
        refs = {}
        role_calls = {
            "outer_eval": [outer_fold],
            "val": [(outer_fold + 1) % 5],
            "train": [
                index
                for index in range(5)
                if index not in {outer_fold, (outer_fold + 1) % 5}
            ],
        }
        protocol_manifests = {}
        for role in ("train", "val", "outer_eval"):
            manifest = fold_root / f"{role}_segment_manifest.jsonl"
            rows = [
                {
                    "call_id": f"eligible_call_{call_index}",
                    "segment_id": f"eligible_segment_{call_index}",
                    "group_id": f"eligible_group_{call_index}",
                    "label": call_index % 2,
                }
                for call_index in role_calls[role]
            ]
            write_jsonl(manifest, rows)
            refs[role] = artifact(manifest, root)
            protocol_manifests[role] = {
                "segment_manifest": str(manifest.relative_to(fold_tree)),
                "call_manifest": str(
                    (fold_root / f"{role}_call_manifest.jsonl").relative_to(fold_tree)
                ),
            }
            write_jsonl(
                fold_root / f"{role}_call_manifest.jsonl",
                [
                    {
                        "call_id": row["call_id"],
                        "group_id": row["group_id"],
                        "label": row["label"],
                    }
                    for row in rows
                ],
            )
        folds.append(
            {
                "partition_repeat": 0,
                "outer_fold": outer_fold,
                "train_manifest": refs["train"],
                "val_manifest": refs["val"],
                "outer_eval_manifest": refs["outer_eval"],
            }
        )
        protocol_folds.append(
            {
                "partition_repeat": 0,
                "outer_fold": outer_fold,
                "manifests": protocol_manifests,
            }
        )

    protocol = fold_tree / "protocol.json"
    write_json(
        protocol,
        {
            "artifact_status": "frozen",
            "schema_version": "1.0.0",
            "protocol_id": "paper2-v2-grouped-nested-v1",
            "grouping_policy_id": "hard-components-v1",
            "folds": protocol_folds,
            "terminology": {
                "outer_partition_name": "outer_eval",
                "outer_eval_interpretation": (
                    "within_corpus_outer_development_evaluation"
                ),
            },
        },
    )
    checksums = fold_tree / "SHA256SUMS.txt"
    checksums.write_text("synthetic test fixture\n", encoding="utf-8")

    return {
        "schema_version": "1.0.0",
        "protocol_id": "paper2-v2-mfcc-factorial-v1",
        "status": "locked",
        "conditions": list(CORE_CONDITIONS),
        "seeds": [17, 29, 43, 71, 101],
        "gate0_receipt": artifact(gate0, root),
        "exclusion_inventory": artifact(exclusions, root),
        "fold_protocol": {
            "origin": "paper2_v2_grouped_nested",
            "protocol_id": "paper2-v2-grouped-nested-v1",
            "grouping_policy_id": "hard-components-v1",
            "protocol_artifact": artifact(protocol, root),
            "assignments_artifact": artifact(exclusions, root),
            "checksums_artifact": artifact(checksums, root),
            "tree_sha256": tree_digest(fold_tree),
            "folds": folds,
        },
        "text_store": {
            "feature_set_id": "text-krsbert-92c6c2c7-mean-mask128-f32-v1",
            "model_id": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
            "model_revision": "92c6c2c7032f680bff0f9f0c63fadd3f97e635b2",
            "pooling": "canonical_attention_masked_mean",
            "dtype": "float32",
            "vector_dim": 768,
            "index_sha256": index_sha256,
            "vectors_sha256": vectors_sha256,
            "metadata": artifact(feature_store, root),
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
                "python", "numpy", "torch_cpu", "torch_cuda", "dataloader",
            ],
        },
        "pairing": {
            "condition": "AT-cross-randtrain",
            "algorithm_id": "group_disjoint_circular_shift_largest_group_v1",
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
            "primary_estimand": "mean_across_seeds_of_paired_call_macro_f1_difference",
            "reported_call_metrics": [
                "macro_f1", "positive_f1", "sensitivity", "specificity", "precision",
                "negative_predictive_value", "balanced_accuracy", "mcc", "roc_auc",
                "pr_auc", "fpr", "fnr", "brier", "nll", "ece_10",
            ],
            "segment_metrics_role": "secondary_weak_label",
            "bootstrap_method": "paired_relation_group_cluster_keep_all_calls_segments_seeds",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260809,
            "confidence_level": 0.95,
            "permutation_method": "paired_relation_group_cluster_swap_all_calls_seeds",
            "permutation_replicates": 10000,
            "permutation_seed": 20260810,
            "multiplicity_method": "holm",
            "familywise_alpha": 0.05,
            "practical_claim_rule": "no_practical_utility_claim_without_pre_work9_cost_policy",
            "smallest_practically_meaningful_effect": None,
        },
        "capacity_sensitivity": {
            "target_condition": "AT-cross",
            "target_parameter_count": 530689,
            "matching_rule": "nearest_count_width_multiple_of_8_search_8_to_1024",
            "maximum_relative_gap": 0.05,
            "matched_hidden_dims": {
                "T": 440,
                "A-mean": 720,
                "A-query": 320,
                "A-temp": 320,
                "AT-concat": 240,
                "AT-gate": 224,
                "AT-cross": 256,
                "AT-cross-randtrain": 256,
            },
            "matched_parameter_counts": {
                "T": 533721,
                "A-mean": 531361,
                "A-query": 519361,
                "A-temp": 519361,
                "AT-concat": 536161,
                "AT-gate": 529537,
                "AT-cross": 530689,
                "AT-cross-randtrain": 530689,
            },
        },
    }
