"""Strict Work 8 protocol loading and leakage/isolation guards.

No final configuration is shipped by this module.  A run-ready YAML/JSON can
only validate after the Work 4 feature store and Work 6 grouped folds exist and
their real SHA-256 values are supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .conditions import CORE_CONDITIONS, BuiltCondition, ConditionId, build_condition


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
LEGACY_PATH_PARTS = {"cv_5fold"}
GROUPED_FOLD_ORIGIN = "paper2_v2_grouped_nested"


class ProtocolValidationError(ValueError):
    pass


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(value: str, field: str) -> None:
    if not SHA256_PATTERN.fullmatch(value) or len(set(value)) == 1:
        raise ProtocolValidationError(f"{field} must be a real lowercase SHA-256")


def _reject_placeholder(value: str, field: str) -> None:
    normalized = value.strip().lower()
    if not normalized or any(token in normalized for token in ("replace", "todo", "placeholder")):
        raise ProtocolValidationError(f"{field} contains an unresolved placeholder")


def _require_keys(
    mapping: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(mapping))
    extra = sorted(set(mapping) - required - optional)
    if missing or extra:
        raise ProtocolValidationError(
            f"{context} keys invalid; missing={missing}, unexpected={extra}"
        )


def _path_has_legacy_cv(path: Path) -> bool:
    return any(part.lower() in LEGACY_PATH_PARTS for part in path.parts)


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        _check_sha256(self.sha256, f"sha256 for {self.path}")
        if _path_has_legacy_cv(self.path):
            raise ProtocolValidationError(
                f"legacy data/cv_5fold artifact is forbidden in v2: {self.path}"
            )

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, project_root: Path, context: str
    ) -> "ArtifactRef":
        _require_keys(raw, required={"path", "sha256"}, context=context)
        raw_path = Path(str(raw["path"]))
        if _path_has_legacy_cv(raw_path):
            raise ProtocolValidationError(
                f"legacy data/cv_5fold artifact is forbidden in v2: {raw_path}"
            )
        path = raw_path if raw_path.is_absolute() else project_root / raw_path
        return cls(path.resolve(), str(raw["sha256"]))

    def verify(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"required protocol artifact not found: {self.path}")
        actual = sha256_file(self.path)
        if actual != self.sha256:
            raise ProtocolValidationError(
                f"artifact hash mismatch for {self.path}: expected {self.sha256}, got {actual}"
            )


@dataclass(frozen=True)
class TextStoreRef:
    feature_set_id: str
    model_id: str
    model_revision: str
    pooling: str
    dtype: str
    vector_dim: int
    index_sha256: str
    vectors_sha256: str
    metadata: ArtifactRef

    def __post_init__(self) -> None:
        _reject_placeholder(self.feature_set_id, "text_store.feature_set_id")
        _reject_placeholder(self.model_id, "text_store.model_id")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ProtocolValidationError("text_store.model_revision must be a 40-char commit")
        _reject_placeholder(self.pooling, "text_store.pooling")
        if self.dtype != "float32" or self.vector_dim != 768:
            raise ProtocolValidationError("Work 8 text vectors must be sealed float32 [N,768]")
        _check_sha256(self.index_sha256, "text_store.index_sha256")
        _check_sha256(self.vectors_sha256, "text_store.vectors_sha256")

    def verify(self) -> None:
        self.metadata.verify()
        with self.metadata.path.open("r", encoding="utf-8") as handle:
            stored = json.load(handle)
        expected = {
            "status": "sealed",
            "feature_set_id": self.feature_set_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "pooling": self.pooling,
            "dtype": self.dtype,
            "index_sha256": self.index_sha256,
            "vectors_sha256": self.vectors_sha256,
        }
        mismatches = {
            key: {"expected": value, "actual": stored.get(key)}
            for key, value in expected.items()
            if stored.get(key) != value
        }
        shape = stored.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or shape[1] != self.vector_dim:
            mismatches["shape"] = {
                "expected": ["N", self.vector_dim],
                "actual": shape,
            }
        if mismatches:
            raise ProtocolValidationError(
                f"text store metadata does not match locked contract: {mismatches}"
            )
        # The metadata hash alone is insufficient: independently hash and open
        # the sealed index/matrix so a copied sidecar cannot authorize altered
        # vectors.
        from .text_features import IndexedTextStore

        indexed_store = IndexedTextStore(self.metadata.path.parent, verify_hashes=True)
        if (
            indexed_store.feature_set_id != self.feature_set_id
            or indexed_store.dimension != self.vector_dim
        ):
            raise ProtocolValidationError("opened text store differs from locked contract")


@dataclass(frozen=True)
class FoldDefinition:
    partition_repeat: int
    outer_fold: int
    train_manifest: ArtifactRef
    val_manifest: ArtifactRef
    outer_eval_manifest: ArtifactRef

    @property
    def key(self) -> tuple[int, int]:
        return self.partition_repeat, self.outer_fold


@dataclass(frozen=True)
class ModelPolicy:
    hidden_dim: int
    num_heads: int
    dropout: float
    text_dim: int

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.num_heads <= 0 or self.hidden_dim % self.num_heads:
            raise ProtocolValidationError("model num_heads must divide positive hidden_dim")
        if not 0.0 <= self.dropout < 1.0:
            raise ProtocolValidationError("model dropout must satisfy 0 <= dropout < 1")
        if self.text_dim != 768:
            raise ProtocolValidationError("Work 8 requires a declared 768-D sentence vector")


@dataclass(frozen=True)
class TrainingPolicy:
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epoch_budget: int
    early_stop_patience: int
    scheduler: str
    scheduler_factor: float
    scheduler_patience: int
    minimum_learning_rate: float
    loss: str
    sampling: str
    augmentation: str
    scheduler_monitor: str
    early_stopping_monitor: str
    mixed_precision: bool
    deterministic_algorithms: bool
    dataloader_workers: int
    shuffle_train: bool
    drop_last: bool
    seed_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.optimizer != "AdamW" or self.loss != "BCEWithLogitsLoss":
            raise ProtocolValidationError("Work 8 optimizer/loss must be explicit AdamW/BCEWithLogitsLoss")
        if self.scheduler != "ReduceLROnPlateau":
            raise ProtocolValidationError("unsupported Work 8 scheduler")
        if self.sampling != "inverse_segments_per_call_loss_weight":
            raise ProtocolValidationError("Work 8 sampling policy changed")
        if self.augmentation != "none":
            raise ProtocolValidationError("the primary Work 8 matrix has no augmentation")
        if (
            self.scheduler_monitor != "validation_call_nll"
            or self.early_stopping_monitor != "validation_call_nll"
        ):
            raise ProtocolValidationError(
                "scheduler and early stopping must monitor validation_call_nll"
            )
        if self.mixed_precision:
            raise ProtocolValidationError(
                "the initial deterministic Work 8 matrix is locked to float32"
            )
        if not self.deterministic_algorithms:
            raise ProtocolValidationError("deterministic algorithms must be enabled")
        if self.dataloader_workers != 0:
            raise ProtocolValidationError("initial Work 8 dataloaders require zero workers")
        if not self.shuffle_train or self.drop_last:
            raise ProtocolValidationError(
                "training must shuffle and must retain the final partial batch"
            )
        if self.seed_sources != (
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "dataloader",
        ):
            raise ProtocolValidationError("all declared randomness sources must be seeded")
        positive = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epoch_budget": self.epoch_budget,
            "early_stop_patience": self.early_stop_patience,
            "scheduler_patience": self.scheduler_patience,
        }
        if any(float(value) <= 0 for value in positive.values()):
            raise ProtocolValidationError(f"training values must be positive: {positive}")
        if self.weight_decay < 0 or self.minimum_learning_rate < 0:
            raise ProtocolValidationError("weight decay/minimum learning rate cannot be negative")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ProtocolValidationError("scheduler_factor must be between zero and one")


@dataclass(frozen=True)
class PairingPolicy:
    condition: str
    algorithm_id: str
    donor_scope: str
    label_policy: str
    randomized_roles: tuple[str, ...]
    correct_pair_roles: tuple[str, ...]
    seed_source: str
    one_to_one: bool
    forbid_same_segment: bool
    forbid_same_call: bool
    forbid_same_group: bool

    def __post_init__(self) -> None:
        expected = {
            "condition": (self.condition, "AT-cross-randtrain"),
            "algorithm_id": (
                self.algorithm_id,
                "group_disjoint_circular_shift_largest_group_v1",
            ),
            "donor_scope": (self.donor_scope, "same_fold_train_manifest"),
            "label_policy": (self.label_policy, "unconstrained_global_random"),
            "randomized_roles": (self.randomized_roles, ("train",)),
            "correct_pair_roles": (self.correct_pair_roles, ("val", "outer_eval")),
            "seed_source": (self.seed_source, "neural_seed"),
        }
        changed = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if changed:
            raise ProtocolValidationError(f"locked random-pairing policy changed: {changed}")
        if not all(
            (
                self.one_to_one,
                self.forbid_same_segment,
                self.forbid_same_call,
                self.forbid_same_group,
            )
        ):
            raise ProtocolValidationError("random train pairing invariants must all be enabled")


THRESHOLD_TIE_BREAK = (
    "lower_fpr",
    "higher_sensitivity",
    "closest_to_0.5",
    "higher_threshold",
)

PRIMARY_COMPARISONS = (
    ("AT-cross", "A-query"),
    ("AT-cross", "AT-concat"),
    ("AT-cross", "AT-gate"),
)

REPORTED_CALL_METRICS = (
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

CAPACITY_MATCHED_WIDTHS = {
    "T": 440,
    "A-mean": 720,
    "A-query": 320,
    "A-temp": 320,
    "AT-concat": 240,
    "AT-gate": 224,
    "AT-cross": 256,
    "AT-cross-randtrain": 256,
}

CAPACITY_MATCHED_PARAMETER_COUNTS = {
    "T": 533_721,
    "A-mean": 531_361,
    "A-query": 519_361,
    "A-temp": 519_361,
    "AT-concat": 536_161,
    "AT-gate": 529_537,
    "AT-cross": 530_689,
    "AT-cross-randtrain": 530_689,
}


@dataclass(frozen=True)
class SelectionPolicy:
    independent_unit: str
    primary_metric: str
    aggregation_rule: str
    checkpoint_monitor: str
    threshold_start: float
    threshold_end: float
    threshold_step: float
    threshold_tie_break: tuple[str, ...]
    calibration_rule: str
    outer_evaluation_once: bool

    def __post_init__(self) -> None:
        expected = {
            "independent_unit": (self.independent_unit, "call"),
            "primary_metric": (self.primary_metric, "macro_f1"),
            "aggregation_rule": (self.aggregation_rule, "mean_probability"),
            "checkpoint_monitor": (self.checkpoint_monitor, "validation_call_nll"),
            "calibration_rule": (self.calibration_rule, "none"),
        }
        changed = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if changed:
            raise ProtocolValidationError(f"locked selection policies changed: {changed}")
        if not self.outer_evaluation_once:
            raise ProtocolValidationError("outer evaluation must be authorized once only")
        if self.threshold_tie_break != THRESHOLD_TIE_BREAK:
            raise ProtocolValidationError("threshold tie-break order changed")
        if not (
            math.isclose(self.threshold_start, 0.01)
            and math.isclose(self.threshold_end, 0.99)
            and math.isclose(self.threshold_step, 0.01)
        ):
            raise ProtocolValidationError("threshold grid must be 0.01..0.99 in 0.01 steps")


@dataclass(frozen=True)
class StatisticsPolicy:
    primary_comparisons: tuple[tuple[str, str], ...]
    primary_estimand: str
    reported_call_metrics: tuple[str, ...]
    segment_metrics_role: str
    bootstrap_method: str
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    permutation_method: str
    permutation_replicates: int
    permutation_seed: int
    multiplicity_method: str
    familywise_alpha: float
    practical_claim_rule: str
    smallest_practically_meaningful_effect: float | None

    def __post_init__(self) -> None:
        if self.primary_comparisons != PRIMARY_COMPARISONS:
            raise ProtocolValidationError("primary matched comparison family changed")
        expected = {
            "primary_estimand": (
                self.primary_estimand,
                "mean_across_seeds_of_paired_call_macro_f1_difference",
            ),
            "segment_metrics_role": (self.segment_metrics_role, "secondary_weak_label"),
            "bootstrap_method": (
                self.bootstrap_method,
                "paired_relation_group_cluster_keep_all_calls_segments_seeds",
            ),
            "permutation_method": (
                self.permutation_method,
                "paired_relation_group_cluster_swap_all_calls_seeds",
            ),
            "multiplicity_method": (self.multiplicity_method, "holm"),
            "practical_claim_rule": (
                self.practical_claim_rule,
                "no_practical_utility_claim_without_pre_work9_cost_policy",
            ),
        }
        changed = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if changed:
            raise ProtocolValidationError(f"locked statistics policies changed: {changed}")
        if self.reported_call_metrics != REPORTED_CALL_METRICS:
            raise ProtocolValidationError("reported call metric set/order changed")
        if self.bootstrap_replicates < 10_000 or self.permutation_replicates < 10_000:
            raise ProtocolValidationError("primary resampling procedures require >=10,000 replicates")
        if not math.isclose(self.confidence_level, 0.95):
            raise ProtocolValidationError("primary confidence level must be 0.95")
        if not math.isclose(self.familywise_alpha, 0.05):
            raise ProtocolValidationError("primary familywise alpha must be 0.05")
        if self.smallest_practically_meaningful_effect is not None:
            raise ProtocolValidationError(
                "no smallest practical effect may be invented without a frozen cost policy"
            )


@dataclass(frozen=True)
class CapacitySensitivityPolicy:
    target_condition: str
    target_parameter_count: int
    matching_rule: str
    maximum_relative_gap: float
    matched_hidden_dims: Mapping[str, int]
    matched_parameter_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.target_condition != "AT-cross" or self.target_parameter_count != 530_689:
            raise ProtocolValidationError("capacity target must remain AT-cross(H=256)")
        if self.matching_rule != "nearest_count_width_multiple_of_8_search_8_to_1024":
            raise ProtocolValidationError("capacity-only matching rule changed")
        if not math.isclose(self.maximum_relative_gap, 0.05):
            raise ProtocolValidationError("capacity sensitivity maximum gap must be 5%")
        widths = {str(key): int(value) for key, value in self.matched_hidden_dims.items()}
        counts = {
            str(key): int(value) for key, value in self.matched_parameter_counts.items()
        }
        if widths != CAPACITY_MATCHED_WIDTHS or counts != CAPACITY_MATCHED_PARAMETER_COUNTS:
            raise ProtocolValidationError("capacity-matched widths/counts changed")
        for condition, count in counts.items():
            relative_gap = abs(count - self.target_parameter_count) / self.target_parameter_count
            if relative_gap > self.maximum_relative_gap:
                raise ProtocolValidationError(
                    f"capacity match for {condition} exceeds maximum relative gap"
                )
        object.__setattr__(self, "matched_hidden_dims", MappingProxyType(widths))
        object.__setattr__(self, "matched_parameter_counts", MappingProxyType(counts))

    def verify_against_factory(self, model_policy: ModelPolicy) -> None:
        for condition in CORE_CONDITIONS:
            built = build_condition(
                condition,
                hidden_dim=self.matched_hidden_dims[condition],
                num_heads=model_policy.num_heads,
                dropout=model_policy.dropout,
                text_dim=model_policy.text_dim,
                text_path_key="__indexed_text_store__",
            )
            expected = self.matched_parameter_counts[condition]
            if built.total_parameters != expected:
                raise ProtocolValidationError(
                    f"capacity count drift for {condition}: expected {expected}, "
                    f"got {built.total_parameters}"
                )


@dataclass(frozen=True)
class SelectionView:
    protocol_id: str
    partition_repeat: int
    outer_fold: int
    train_manifest: ArtifactRef
    val_manifest: ArtifactRef


@dataclass(frozen=True)
class SelectionSeal:
    protocol_id: str
    partition_repeat: int
    outer_fold: int
    condition: str
    seed: int
    checkpoint: ArtifactRef
    validation_predictions_sha256: str
    threshold: float

    def __post_init__(self) -> None:
        if self.condition not in CORE_CONDITIONS:
            raise ProtocolValidationError("selection seal has unknown condition")
        _check_sha256(self.validation_predictions_sha256, "validation_predictions_sha256")
        if not 0.0 <= self.threshold <= 1.0:
            raise ProtocolValidationError("selection threshold must be within [0,1]")


@dataclass(frozen=True)
class OuterEvaluationView:
    protocol_id: str
    partition_repeat: int
    outer_fold: int
    condition: str
    seed: int
    outer_eval_manifest: ArtifactRef
    checkpoint: ArtifactRef
    threshold: float


@dataclass(frozen=True)
class MatrixConfig:
    schema_version: str
    protocol_id: str
    status: str
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    gate0_receipt: ArtifactRef
    exclusion_inventory: ArtifactRef
    fold_origin: str
    fold_protocol_id: str
    grouping_policy_id: str
    fold_protocol_artifact: ArtifactRef
    fold_assignments_artifact: ArtifactRef
    fold_checksums_artifact: ArtifactRef
    fold_tree_sha256: str
    folds: tuple[FoldDefinition, ...]
    text_store: TextStoreRef
    model: ModelPolicy
    training: TrainingPolicy
    pairing: PairingPolicy
    selection: SelectionPolicy
    statistics: StatisticsPolicy
    capacity_sensitivity: CapacitySensitivityPolicy

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0" or self.status != "locked":
            raise ProtocolValidationError("only locked Work 8 schema 1.0.0 is run-ready")
        _reject_placeholder(self.protocol_id, "protocol_id")
        _reject_placeholder(self.fold_protocol_id, "fold_protocol_id")
        _reject_placeholder(self.grouping_policy_id, "grouping_policy_id")
        if self.conditions != CORE_CONDITIONS:
            raise ProtocolValidationError(
                f"conditions must be exactly the locked matrix order {CORE_CONDITIONS}"
            )
        if len(self.seeds) < 5 or len(set(self.seeds)) != len(self.seeds):
            raise ProtocolValidationError("at least five unique predeclared neural seeds required")
        if self.fold_origin != GROUPED_FOLD_ORIGIN:
            raise ProtocolValidationError("legacy or non-grouped fold origin is forbidden")
        _check_sha256(self.fold_tree_sha256, "fold_protocol.tree_sha256")
        if not self.folds:
            raise ProtocolValidationError("no folds declared")
        keys = [fold.key for fold in self.folds]
        if len(set(keys)) != len(keys):
            raise ProtocolValidationError("duplicate repeat/outer-fold definition")
        repeats = {repeat for repeat, _ in keys}
        outer = {fold for _, fold in keys}
        if repeats != {0} or outer != set(range(5)) or len(keys) != 5:
            raise ProtocolValidationError(
                "initial protocol must be one fixed partition repeat x five outer folds"
            )
        expected_names = {
            "train": "train_segment_manifest.jsonl",
            "val": "val_segment_manifest.jsonl",
            "outer_eval": "outer_eval_segment_manifest.jsonl",
        }
        for fold in self.folds:
            manifests = {
                "train": fold.train_manifest,
                "val": fold.val_manifest,
                "outer_eval": fold.outer_eval_manifest,
            }
            for role, artifact in manifests.items():
                if artifact.path.name != expected_names[role]:
                    raise ProtocolValidationError(
                        f"fold {fold.key} {role} must use {expected_names[role]!r}"
                    )
                if "paper2_v2" not in artifact.path.parts:
                    raise ProtocolValidationError(
                        f"fold {fold.key} {role} is outside data/paper2_v2"
                    )
        self.capacity_sensitivity.verify_against_factory(self.model)

    def condition(self, condition: str) -> ConditionId:
        if condition not in self.conditions:
            raise ProtocolValidationError(f"condition {condition!r} is not preregistered")
        return ConditionId(condition)

    def instantiate_condition(self, condition: str) -> BuiltCondition:
        """Build a model and indexed-store input contract from locked settings."""

        self.condition(condition)
        return build_condition(
            condition,
            hidden_dim=self.model.hidden_dim,
            num_heads=self.model.num_heads,
            dropout=self.model.dropout,
            text_dim=self.model.text_dim,
            text_path_key="__indexed_text_store__",
        )

    def fold(self, partition_repeat: int, outer_fold: int) -> FoldDefinition:
        for fold in self.folds:
            if fold.key == (partition_repeat, outer_fold):
                return fold
        raise ProtocolValidationError(
            f"fold repeat={partition_repeat}, outer={outer_fold} is not preregistered"
        )

    def selection_view(self, partition_repeat: int, outer_fold: int) -> SelectionView:
        fold = self.fold(partition_repeat, outer_fold)
        # The returned object intentionally has no outer-evaluation path.
        return SelectionView(
            protocol_id=self.protocol_id,
            partition_repeat=partition_repeat,
            outer_fold=outer_fold,
            train_manifest=fold.train_manifest,
            val_manifest=fold.val_manifest,
        )

    def authorize_outer_evaluation(self, seal: SelectionSeal) -> OuterEvaluationView:
        if seal.protocol_id != self.protocol_id:
            raise ProtocolValidationError("selection seal belongs to another protocol")
        self.condition(seal.condition)
        if seal.seed not in self.seeds:
            raise ProtocolValidationError("selection seal uses a non-preregistered seed")
        fold = self.fold(seal.partition_repeat, seal.outer_fold)
        seal.checkpoint.verify()
        return OuterEvaluationView(
            protocol_id=self.protocol_id,
            partition_repeat=seal.partition_repeat,
            outer_fold=seal.outer_fold,
            condition=seal.condition,
            seed=seal.seed,
            outer_eval_manifest=fold.outer_eval_manifest,
            checkpoint=seal.checkpoint,
            threshold=seal.threshold,
        )

    def verify_artifacts(self) -> None:
        for artifact in (
            self.gate0_receipt,
            self.exclusion_inventory,
            self.fold_protocol_artifact,
            self.fold_assignments_artifact,
            self.fold_checksums_artifact,
        ):
            artifact.verify()
        self._verify_frozen_fold_protocol()
        self.text_store.verify()
        for fold in self.folds:
            fold.train_manifest.verify()
            fold.val_manifest.verify()
            fold.outer_eval_manifest.verify()

    def _verify_frozen_fold_protocol(self) -> None:
        with self.fold_protocol_artifact.path.open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
        expected = {
            "artifact_status": "frozen",
            "schema_version": "1.0.0",
            "protocol_id": self.fold_protocol_id,
            "grouping_policy_id": self.grouping_policy_id,
        }
        mismatches = {
            key: {"expected": wanted, "actual": protocol.get(key)}
            for key, wanted in expected.items()
            if protocol.get(key) != wanted
        }
        terminology = protocol.get("terminology", {})
        if terminology.get("outer_partition_name") != "outer_eval":
            mismatches["outer_partition_name"] = {
                "expected": "outer_eval",
                "actual": terminology.get("outer_partition_name"),
            }
        if terminology.get("outer_eval_interpretation") != (
            "within_corpus_outer_development_evaluation"
        ):
            mismatches["outer_eval_interpretation"] = {
                "expected": "within_corpus_outer_development_evaluation",
                "actual": terminology.get("outer_eval_interpretation"),
            }
        if mismatches:
            raise ProtocolValidationError(
                f"frozen Work 6 protocol differs from Work 8 contract: {mismatches}"
            )

        protocol_root = self.fold_protocol_artifact.path.parent.resolve()
        declared_paths = {
            (int(item["partition_repeat"]), int(item["outer_fold"]), role):
            (protocol_root / item["manifests"][role]["segment_manifest"]).resolve()
            for item in protocol.get("folds", [])
            for role in ("train", "val", "outer_eval")
        }
        configured_paths: dict[tuple[int, int, str], Path] = {}
        for fold in self.folds:
            configured_paths[(fold.partition_repeat, fold.outer_fold, "train")] = (
                fold.train_manifest.path
            )
            configured_paths[(fold.partition_repeat, fold.outer_fold, "val")] = (
                fold.val_manifest.path
            )
            configured_paths[(fold.partition_repeat, fold.outer_fold, "outer_eval")] = (
                fold.outer_eval_manifest.path
            )
        if declared_paths != configured_paths:
            raise ProtocolValidationError(
                "configured manifests do not exactly match frozen Work 6 protocol paths"
            )

        from .folds import tree_digest, validate_fold_tree

        validate_fold_tree(protocol_root)
        actual_tree_digest = tree_digest(protocol_root)
        if actual_tree_digest != self.fold_tree_sha256:
            raise ProtocolValidationError(
                "frozen Work 6 tree digest mismatch: "
                f"expected {self.fold_tree_sha256}, got {actual_tree_digest}"
            )

    def validate_manifest_isolation(self) -> None:
        """Verify call/group disjointness and Work 6 exclusions for every fold."""

        excluded_calls = _read_excluded_calls(self.exclusion_inventory.path)
        for fold in self.folds:
            role_sets = {
                "train": _manifest_identity_sets(fold.train_manifest.path),
                "val": _manifest_identity_sets(fold.val_manifest.path),
                "outer_eval": _manifest_identity_sets(fold.outer_eval_manifest.path),
            }
            roles = tuple(role_sets)
            for left_index, left in enumerate(roles):
                calls, groups = role_sets[left]
                leaked_exclusions = sorted(calls & excluded_calls)
                if leaked_exclusions:
                    raise ProtocolValidationError(
                        f"fold {fold.key} {left} contains excluded/burned calls: "
                        f"{leaked_exclusions[:5]}"
                    )
                for right in roles[left_index + 1 :]:
                    other_calls, other_groups = role_sets[right]
                    if calls & other_calls:
                        raise ProtocolValidationError(
                            f"fold {fold.key} call leakage between {left} and {right}"
                        )
                    if groups & other_groups:
                        raise ProtocolValidationError(
                            f"fold {fold.key} group leakage between {left} and {right}"
                        )


def _manifest_identity_sets(path: Path) -> tuple[set[str], set[str]]:
    calls: set[str] = set()
    groups: set[str] = set()
    segment_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            call_id = str(row.get("call_id", ""))
            group_id = str(row.get("group_id", ""))
            segment_id = str(row.get("segment_id", ""))
            if not call_id or not group_id or not segment_id:
                raise ProtocolValidationError(
                    f"{path}:{row_number} requires call_id, group_id, segment_id"
                )
            if segment_id in segment_ids:
                raise ProtocolValidationError(f"duplicate segment_id {segment_id!r} in {path}")
            calls.add(call_id)
            groups.add(group_id)
            segment_ids.add(segment_id)
    if not segment_ids:
        raise ProtocolValidationError(f"manifest is empty: {path}")
    return calls, groups


def _read_excluded_calls(path: Path) -> set[str]:
    """Read Work 6 JSONL exclusions/assignments without assuming private metadata."""

    excluded: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reason = row.get("exclusion_reason")
            historical_test = row.get("historical_split") == "test"
            excluded_role = row.get("role") in {"excluded", "quarantine"}
            if reason or historical_test or excluded_role:
                call_id = str(row.get("call_id", ""))
                if not call_id:
                    raise ProtocolValidationError("excluded assignment row has no call_id")
                excluded.add(call_id)
    if not excluded:
        raise ProtocolValidationError(
            "exclusion inventory contains no burned/quarantined calls"
        )
    return excluded


def _artifact(raw: Mapping[str, Any], project_root: Path, context: str) -> ArtifactRef:
    return ArtifactRef.from_mapping(raw, project_root=project_root, context=context)


def matrix_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    project_root: str | Path,
    verify_files: bool = True,
    verify_manifest_isolation: bool = True,
) -> MatrixConfig:
    """Parse a strict mapping; unknown keys and placeholder hashes are errors."""

    root = Path(project_root).resolve()
    _require_keys(
        raw,
        required={
            "schema_version", "protocol_id", "status", "conditions", "seeds",
            "gate0_receipt", "exclusion_inventory", "fold_protocol", "text_store",
            "model", "training", "pairing", "selection", "statistics",
            "capacity_sensitivity",
        },
        context="matrix config",
    )
    fold_raw = raw["fold_protocol"]
    _require_keys(
        fold_raw,
        required={
            "origin", "protocol_id", "grouping_policy_id", "protocol_artifact",
            "assignments_artifact", "checksums_artifact", "tree_sha256", "folds",
        },
        context="fold_protocol",
    )
    folds = []
    for index, item in enumerate(fold_raw["folds"]):
        _require_keys(
            item,
            required={
                "partition_repeat", "outer_fold", "train_manifest", "val_manifest",
                "outer_eval_manifest",
            },
            context=f"fold_protocol.folds[{index}]",
        )
        folds.append(
            FoldDefinition(
                partition_repeat=int(item["partition_repeat"]),
                outer_fold=int(item["outer_fold"]),
                train_manifest=_artifact(item["train_manifest"], root, "train_manifest"),
                val_manifest=_artifact(item["val_manifest"], root, "val_manifest"),
                outer_eval_manifest=_artifact(
                    item["outer_eval_manifest"], root, "outer_eval_manifest"
                ),
            )
        )

    text_raw = raw["text_store"]
    _require_keys(
        text_raw,
        required={
            "feature_set_id", "model_id", "model_revision", "pooling", "dtype",
            "vector_dim", "index_sha256", "vectors_sha256", "metadata",
        },
        context="text_store",
    )
    model_raw = raw["model"]
    _require_keys(
        model_raw,
        required={"hidden_dim", "num_heads", "dropout", "text_dim"},
        context="model",
    )
    training_raw = raw["training"]
    _require_keys(
        training_raw,
        required={
            "optimizer", "learning_rate", "weight_decay", "batch_size", "epoch_budget",
            "early_stop_patience", "scheduler", "scheduler_factor",
            "scheduler_patience", "minimum_learning_rate", "loss", "sampling",
            "augmentation", "scheduler_monitor", "early_stopping_monitor",
            "mixed_precision", "deterministic_algorithms", "dataloader_workers",
            "shuffle_train", "drop_last", "seed_sources",
        },
        context="training",
    )
    pairing_raw = raw["pairing"]
    _require_keys(
        pairing_raw,
        required={
            "condition", "algorithm_id", "donor_scope", "label_policy",
            "randomized_roles",
            "correct_pair_roles", "seed_source", "one_to_one",
            "forbid_same_segment", "forbid_same_call", "forbid_same_group",
        },
        context="pairing",
    )
    selection_raw = raw["selection"]
    _require_keys(
        selection_raw,
        required={
            "independent_unit", "primary_metric", "aggregation_rule",
            "checkpoint_monitor", "threshold_start", "threshold_end",
            "threshold_step", "threshold_tie_break", "calibration_rule",
            "outer_evaluation_once",
        },
        context="selection",
    )
    statistics_raw = raw["statistics"]
    _require_keys(
        statistics_raw,
        required={
            "primary_comparisons", "primary_estimand", "reported_call_metrics",
            "segment_metrics_role", "bootstrap_method", "bootstrap_replicates",
            "bootstrap_seed", "confidence_level", "permutation_method",
            "permutation_replicates", "permutation_seed", "multiplicity_method",
            "familywise_alpha", "practical_claim_rule",
            "smallest_practically_meaningful_effect",
        },
        context="statistics",
    )
    capacity_raw = raw["capacity_sensitivity"]
    _require_keys(
        capacity_raw,
        required={
            "target_condition", "target_parameter_count", "matching_rule",
            "maximum_relative_gap", "matched_hidden_dims", "matched_parameter_counts",
        },
        context="capacity_sensitivity",
    )

    config = MatrixConfig(
        schema_version=str(raw["schema_version"]),
        protocol_id=str(raw["protocol_id"]),
        status=str(raw["status"]),
        conditions=tuple(str(value) for value in raw["conditions"]),
        seeds=tuple(int(value) for value in raw["seeds"]),
        gate0_receipt=_artifact(raw["gate0_receipt"], root, "gate0_receipt"),
        exclusion_inventory=_artifact(
            raw["exclusion_inventory"], root, "exclusion_inventory"
        ),
        fold_origin=str(fold_raw["origin"]),
        fold_protocol_id=str(fold_raw["protocol_id"]),
        grouping_policy_id=str(fold_raw["grouping_policy_id"]),
        fold_protocol_artifact=_artifact(
            fold_raw["protocol_artifact"], root, "fold protocol artifact"
        ),
        fold_assignments_artifact=_artifact(
            fold_raw["assignments_artifact"], root, "fold assignments artifact"
        ),
        fold_checksums_artifact=_artifact(
            fold_raw["checksums_artifact"], root, "fold checksums artifact"
        ),
        fold_tree_sha256=str(fold_raw["tree_sha256"]),
        folds=tuple(folds),
        text_store=TextStoreRef(
            feature_set_id=str(text_raw["feature_set_id"]),
            model_id=str(text_raw["model_id"]),
            model_revision=str(text_raw["model_revision"]),
            pooling=str(text_raw["pooling"]),
            dtype=str(text_raw["dtype"]),
            vector_dim=int(text_raw["vector_dim"]),
            index_sha256=str(text_raw["index_sha256"]),
            vectors_sha256=str(text_raw["vectors_sha256"]),
            metadata=_artifact(text_raw["metadata"], root, "text store metadata"),
        ),
        model=ModelPolicy(
            hidden_dim=int(model_raw["hidden_dim"]),
            num_heads=int(model_raw["num_heads"]),
            dropout=float(model_raw["dropout"]),
            text_dim=int(model_raw["text_dim"]),
        ),
        training=TrainingPolicy(
            optimizer=str(training_raw["optimizer"]),
            learning_rate=float(training_raw["learning_rate"]),
            weight_decay=float(training_raw["weight_decay"]),
            batch_size=int(training_raw["batch_size"]),
            epoch_budget=int(training_raw["epoch_budget"]),
            early_stop_patience=int(training_raw["early_stop_patience"]),
            scheduler=str(training_raw["scheduler"]),
            scheduler_factor=float(training_raw["scheduler_factor"]),
            scheduler_patience=int(training_raw["scheduler_patience"]),
            minimum_learning_rate=float(training_raw["minimum_learning_rate"]),
            loss=str(training_raw["loss"]),
            sampling=str(training_raw["sampling"]),
            augmentation=str(training_raw["augmentation"]),
            scheduler_monitor=str(training_raw["scheduler_monitor"]),
            early_stopping_monitor=str(training_raw["early_stopping_monitor"]),
            mixed_precision=bool(training_raw["mixed_precision"]),
            deterministic_algorithms=bool(training_raw["deterministic_algorithms"]),
            dataloader_workers=int(training_raw["dataloader_workers"]),
            shuffle_train=bool(training_raw["shuffle_train"]),
            drop_last=bool(training_raw["drop_last"]),
            seed_sources=tuple(str(value) for value in training_raw["seed_sources"]),
        ),
        pairing=PairingPolicy(
            condition=str(pairing_raw["condition"]),
            algorithm_id=str(pairing_raw["algorithm_id"]),
            donor_scope=str(pairing_raw["donor_scope"]),
            label_policy=str(pairing_raw["label_policy"]),
            randomized_roles=tuple(
                str(value) for value in pairing_raw["randomized_roles"]
            ),
            correct_pair_roles=tuple(
                str(value) for value in pairing_raw["correct_pair_roles"]
            ),
            seed_source=str(pairing_raw["seed_source"]),
            one_to_one=bool(pairing_raw["one_to_one"]),
            forbid_same_segment=bool(pairing_raw["forbid_same_segment"]),
            forbid_same_call=bool(pairing_raw["forbid_same_call"]),
            forbid_same_group=bool(pairing_raw["forbid_same_group"]),
        ),
        selection=SelectionPolicy(
            independent_unit=str(selection_raw["independent_unit"]),
            primary_metric=str(selection_raw["primary_metric"]),
            aggregation_rule=str(selection_raw["aggregation_rule"]),
            checkpoint_monitor=str(selection_raw["checkpoint_monitor"]),
            threshold_start=float(selection_raw["threshold_start"]),
            threshold_end=float(selection_raw["threshold_end"]),
            threshold_step=float(selection_raw["threshold_step"]),
            threshold_tie_break=tuple(selection_raw["threshold_tie_break"]),
            calibration_rule=str(selection_raw["calibration_rule"]),
            outer_evaluation_once=bool(selection_raw["outer_evaluation_once"]),
        ),
        statistics=StatisticsPolicy(
            primary_comparisons=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in statistics_raw["primary_comparisons"]
            ),
            primary_estimand=str(statistics_raw["primary_estimand"]),
            reported_call_metrics=tuple(statistics_raw["reported_call_metrics"]),
            segment_metrics_role=str(statistics_raw["segment_metrics_role"]),
            bootstrap_method=str(statistics_raw["bootstrap_method"]),
            bootstrap_replicates=int(statistics_raw["bootstrap_replicates"]),
            bootstrap_seed=int(statistics_raw["bootstrap_seed"]),
            confidence_level=float(statistics_raw["confidence_level"]),
            permutation_method=str(statistics_raw["permutation_method"]),
            permutation_replicates=int(statistics_raw["permutation_replicates"]),
            permutation_seed=int(statistics_raw["permutation_seed"]),
            multiplicity_method=str(statistics_raw["multiplicity_method"]),
            familywise_alpha=float(statistics_raw["familywise_alpha"]),
            practical_claim_rule=str(statistics_raw["practical_claim_rule"]),
            smallest_practically_meaningful_effect=(
                None
                if statistics_raw["smallest_practically_meaningful_effect"] is None
                else float(statistics_raw["smallest_practically_meaningful_effect"])
            ),
        ),
        capacity_sensitivity=CapacitySensitivityPolicy(
            target_condition=str(capacity_raw["target_condition"]),
            target_parameter_count=int(capacity_raw["target_parameter_count"]),
            matching_rule=str(capacity_raw["matching_rule"]),
            maximum_relative_gap=float(capacity_raw["maximum_relative_gap"]),
            matched_hidden_dims=dict(capacity_raw["matched_hidden_dims"]),
            matched_parameter_counts=dict(capacity_raw["matched_parameter_counts"]),
        ),
    )
    if verify_files:
        config.verify_artifacts()
    if verify_manifest_isolation:
        config.validate_manifest_isolation()
    return config


def load_matrix_config(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_files: bool = True,
    verify_manifest_isolation: bool = True,
) -> MatrixConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as error:  # pragma: no cover - environment guard
                raise RuntimeError("PyYAML is required to load YAML protocol files") from error
            raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ProtocolValidationError("matrix configuration must be an object")
    return matrix_config_from_mapping(
        raw,
        project_root=project_root,
        verify_files=verify_files,
        verify_manifest_isolation=verify_manifest_isolation,
    )


def frozen_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Small helper for callers that retain a validated raw config snapshot."""

    return MappingProxyType(dict(mapping))
