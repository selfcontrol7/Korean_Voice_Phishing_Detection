"""Immutable identifiers and lineage records for the Work 9 primary matrix."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .conditions import CORE_CONDITIONS
from .config import ArtifactRef, ProtocolValidationError, sha256_file


WORK8_CONFIG_RELATIVE = Path(
    "revised_experiments/configs/work8_mfcc_matched_within_corpus_v1.json"
)
WORK8_RECEIPT_RELATIVE = Path(
    "revised_experiments/artifacts/work8_preregistration_v1/dry_run_receipt.json"
)
WORK8_CONFIG_SHA256 = "ab04b6c00979ae94b1adc4cb952cf1d7a3db7e7f2c98c87cda520b8ad18b2fb4"
WORK8_RECEIPT_SHA256 = "5c81d5367be670f9bb863a31c54c847ff7b7340c8bfd80b63fe16e057d1860ed"
GATE1_POLICY_RELATIVE = Path(
    "revised_experiments/configs/work9_gate1_decision_policy_v1.json"
)
GATE1_POLICY_SHA256 = "a6697bebd0df3d6dca99bcfd82dd07bf874f2cf7923fae5d07d0b1201444880b"
GATE1_ANALYSIS_PROTOCOL_RELATIVE = Path(
    "revised_experiments/configs/work9_gate1_analysis_protocol_v1.json"
)
GATE1_ANALYSIS_PROTOCOL_SHA256 = (
    "793ca1dfcb1cb339eafeb504d229c7a5a3d07431060d38c1d2d75a89ac79b9fa"
)
EXPERIMENT_CARD_RELATIVE = Path(
    "revised_experiments/configs/work9_primary_matrix_experiment_card_v1.json"
)
SUPERSEDED_ACCEPTANCE_V1_DRAFT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_runner_acceptance_v1/DRAFT.json"
)
SUPERSEDED_ACCEPTANCE_V1_DRAFT_SHA256 = (
    "05a41300cece9dad5704a40ddfe2e3cf0ec3e4ad1654119a25ece7408d332f6d"
)
SUPERSEDED_ACCEPTANCE_V1_FAILURE_RELATIVE = Path(
    "revised_experiments/artifacts/work9_runner_acceptance_v1/PREFLIGHT_FAILURE.json"
)
SUPERSEDED_ACCEPTANCE_V1_FAILURE_SHA256 = (
    "13378ad46b0dc0cd90946a9415fe2854cd13f7094b458998e74ff0730266a7db"
)
SUPERSEDED_ACCEPTANCE_V2_DRAFT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_runner_acceptance_v2/DRAFT.json"
)
SUPERSEDED_ACCEPTANCE_V2_DRAFT_SHA256 = (
    "e303b1714b20bfe6fec80ae4dfa2571e5564286a7989e4c0e5da3eb23dc347fa"
)
SUPERSEDED_ACCEPTANCE_V2_FAILURE_RELATIVE = Path(
    "revised_experiments/artifacts/work9_runner_acceptance_v2/POST_SELECTION_FAILURE.json"
)
SUPERSEDED_ACCEPTANCE_V2_FAILURE_SHA256 = (
    "3a1dfbec205b6921499820065e6846bb2df6b289a33baf104e47c58a23980425"
)
SUPERSEDED_ACCEPTANCE_GENERATIONS = (
    (
        "work9_runner_acceptance_v1",
        SUPERSEDED_ACCEPTANCE_V1_DRAFT_RELATIVE,
        SUPERSEDED_ACCEPTANCE_V1_DRAFT_SHA256,
        SUPERSEDED_ACCEPTANCE_V1_FAILURE_RELATIVE,
        SUPERSEDED_ACCEPTANCE_V1_FAILURE_SHA256,
        "CUDA telemetry reset preceded explicit CUDA device selection",
    ),
    (
        "work9_runner_acceptance_v2",
        SUPERSEDED_ACCEPTANCE_V2_DRAFT_RELATIVE,
        SUPERSEDED_ACCEPTANCE_V2_DRAFT_SHA256,
        SUPERSEDED_ACCEPTANCE_V2_FAILURE_RELATIVE,
        SUPERSEDED_ACCEPTANCE_V2_FAILURE_SHA256,
        "environment inventory followed runtime-mutated sys.path",
    ),
)
ACCEPTANCE_RECEIPT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_runner_acceptance_v3/runner_acceptance.json"
)
ACCEPTANCE_DRAFT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_runner_acceptance_v3/DRAFT.json"
)
ACCEPTANCE_SCHEMA_VERSION = "paper2-v2-work9-runner-acceptance-3"
CUDA_SMOKE_RECEIPT_SCHEMA_VERSION = "paper2-v2-work9-cuda-smoke-receipt-3"
CUDA_SMOKE_SEMANTIC_SCHEMA_VERSION = "paper2-v2-work9-cuda-smoke-semantic-3"
UNIT_TEST_RECEIPT_SCHEMA_VERSION = "paper2-v2-work9-unit-test-receipt-3"
REGISTRY_RELATIVE = Path("revised_experiments/artifacts/work9_runs_v1")
RUN_FAMILY = "primary_matrix"
EXPECTED_SELECTION_COUNT = 8 * 5 * 5
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def canonical_json_bytes(payload: Mapping[str, Any] | list[Any]) -> bytes:
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


def canonical_jsonl_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("refusing to serialize an empty Work 9 ledger")
    return (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, order=True)
class Work9Coordinates:
    protocol_id: str
    run_family: str
    condition: str
    partition_repeat: int
    outer_fold: int
    seed: int

    def __post_init__(self) -> None:
        if not SAFE_ID.fullmatch(self.protocol_id):
            raise ValueError("protocol_id must be filesystem safe")
        if self.run_family != RUN_FAMILY:
            raise ValueError("Work 9.1 accepts only run_family=primary_matrix")
        if self.condition not in CORE_CONDITIONS:
            raise ValueError("condition is outside the frozen Work 8 matrix")
        if self.partition_repeat != 0 or self.outer_fold not in range(5):
            raise ValueError("coordinate is outside the frozen one-repeat/five-fold matrix")

    @property
    def run_id(self) -> str:
        return (
            f"{self.run_family}__{self.condition}__r{self.partition_repeat:02d}-"
            f"o{self.outer_fold:02d}-s{self.seed}"
        )

    @property
    def claim_key(self) -> str:
        return sha256_bytes(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )


def expected_coordinates(protocol_id: str, seeds: tuple[int, ...]) -> tuple[Work9Coordinates, ...]:
    return tuple(
        Work9Coordinates(protocol_id, RUN_FAMILY, condition, 0, outer_fold, seed)
        for condition in CORE_CONDITIONS
        for outer_fold in range(5)
        for seed in seeds
    )


@dataclass(frozen=True)
class Work9SelectionSeal:
    schema_version: str
    coordinates: Work9Coordinates
    run_id: str
    config_sha256: str
    work8_receipt_sha256: str
    runner_acceptance_sha256: str
    experiment_card_sha256: str
    gate1_policy_sha256: str
    gate1_analysis_protocol_sha256: str
    train_manifest_sha256: str
    validation_manifest_sha256: str
    feature_store_metadata_sha256: str
    feature_store_index_sha256: str
    feature_store_vectors_sha256: str
    pairing_plan_sha256: str | None
    pairing_plan_artifact: ArtifactRef | None
    selected_epoch: int
    validation_call_nll: float
    threshold: float
    checkpoint: ArtifactRef
    validation_segment_ledger: ArtifactRef
    validation_call_ledger: ArtifactRef
    epoch_trace: ArtifactRef
    threshold_trace: ArtifactRef

    def __post_init__(self) -> None:
        if self.schema_version != "paper2-v2-work9-selection-seal-1":
            raise ValueError("unsupported Work 9 selection seal schema")
        if self.run_id != self.coordinates.run_id:
            raise ValueError("selection seal run_id differs from coordinates")
        for name, value in (
            ("config_sha256", self.config_sha256),
            ("work8_receipt_sha256", self.work8_receipt_sha256),
            ("runner_acceptance_sha256", self.runner_acceptance_sha256),
            ("experiment_card_sha256", self.experiment_card_sha256),
            ("gate1_policy_sha256", self.gate1_policy_sha256),
            ("gate1_analysis_protocol_sha256", self.gate1_analysis_protocol_sha256),
            ("train_manifest_sha256", self.train_manifest_sha256),
            ("validation_manifest_sha256", self.validation_manifest_sha256),
            ("feature_store_metadata_sha256", self.feature_store_metadata_sha256),
            ("feature_store_index_sha256", self.feature_store_index_sha256),
            ("feature_store_vectors_sha256", self.feature_store_vectors_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} is not a SHA-256")
        if (self.pairing_plan_sha256 is None) != (self.pairing_plan_artifact is None):
            raise ValueError("pairing plan semantic hash and artifact must both be set or null")
        if self.pairing_plan_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", self.pairing_plan_sha256):
                raise ValueError("pairing plan hash is invalid")
            if self.coordinates.condition != "AT-cross-randtrain":
                raise ValueError("only random-train condition may bind a pairing plan")
        elif self.coordinates.condition == "AT-cross-randtrain":
            raise ValueError("random-train condition requires a pairing plan")
        if self.selected_epoch <= 0 or self.validation_call_nll < 0:
            raise ValueError("selected epoch/NLL are invalid")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("selection threshold is outside [0,1]")

    def artifact_refs(self) -> tuple[ArtifactRef, ...]:
        refs = (
            self.checkpoint,
            self.validation_segment_ledger,
            self.validation_call_ledger,
            self.epoch_trace,
            self.threshold_trace,
        )
        return refs + (() if self.pairing_plan_artifact is None else (self.pairing_plan_artifact,))

    def verify(self, run_root: Path) -> None:
        resolved_root = run_root.resolve()
        for artifact in self.artifact_refs():
            artifact.verify()
            if resolved_root not in artifact.path.resolve().parents:
                raise ProtocolValidationError(
                    f"selection artifact is outside reserved run: {artifact.path}"
                )

    def to_mapping(self, *, project_root: Path) -> dict[str, Any]:
        def count_rows(path: Path) -> int:
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())

        def ref(value: ArtifactRef, *, ledger: bool = False) -> dict[str, Any]:
            return {
                "path": value.path.resolve().relative_to(project_root.resolve()).as_posix(),
                "sha256": value.sha256,
                "bytes": value.path.stat().st_size,
                "row_count": (
                    count_rows(value.path)
                    if ledger
                    else None
                ),
            }

        return {
            "schema_version": self.schema_version,
            **asdict(self.coordinates),
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "work8_receipt_sha256": self.work8_receipt_sha256,
            "runner_acceptance_sha256": self.runner_acceptance_sha256,
            "experiment_card_sha256": self.experiment_card_sha256,
            "gate1_policy_sha256": self.gate1_policy_sha256,
            "gate1_analysis_protocol_sha256": self.gate1_analysis_protocol_sha256,
            "train_manifest_sha256": self.train_manifest_sha256,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "feature_store_metadata_sha256": self.feature_store_metadata_sha256,
            "feature_store_index_sha256": self.feature_store_index_sha256,
            "feature_store_vectors_sha256": self.feature_store_vectors_sha256,
            "pairing_plan_sha256": self.pairing_plan_sha256,
            "pairing_plan_artifact": (
                None
                if self.pairing_plan_artifact is None
                else ref(self.pairing_plan_artifact, ledger=True)
            ),
            "selected_epoch": self.selected_epoch,
            "validation_call_nll": self.validation_call_nll,
            "threshold": self.threshold,
            "checkpoint": ref(self.checkpoint),
            "validation_segment_ledger": ref(
                self.validation_segment_ledger, ledger=True
            ),
            "validation_call_ledger": ref(self.validation_call_ledger, ledger=True),
            "epoch_trace": ref(self.epoch_trace),
            "threshold_trace": ref(self.threshold_trace),
        }

    @property
    def sha256(self) -> str:
        # Artifact paths are already immutable absolute identities here; the
        # persisted mapping is the authoritative portable representation.
        payload = {
            **asdict(self.coordinates),
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "work8_receipt_sha256": self.work8_receipt_sha256,
            "runner_acceptance_sha256": self.runner_acceptance_sha256,
            "experiment_card_sha256": self.experiment_card_sha256,
            "gate1_policy_sha256": self.gate1_policy_sha256,
            "gate1_analysis_protocol_sha256": self.gate1_analysis_protocol_sha256,
            "train_manifest_sha256": self.train_manifest_sha256,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "feature_store_metadata_sha256": self.feature_store_metadata_sha256,
            "feature_store_index_sha256": self.feature_store_index_sha256,
            "feature_store_vectors_sha256": self.feature_store_vectors_sha256,
            "pairing_plan_sha256": self.pairing_plan_sha256,
            "pairing_plan_artifact_sha256": (
                None
                if self.pairing_plan_artifact is None
                else self.pairing_plan_artifact.sha256
            ),
            "selected_epoch": self.selected_epoch,
            "validation_call_nll": self.validation_call_nll,
            "threshold": self.threshold,
            "artifacts": {
                name: value.sha256
                for name, value in (
                    ("checkpoint", self.checkpoint),
                    ("validation_segment_ledger", self.validation_segment_ledger),
                    ("validation_call_ledger", self.validation_call_ledger),
                    ("epoch_trace", self.epoch_trace),
                    ("threshold_trace", self.threshold_trace),
                )
            },
        }
        return sha256_bytes(canonical_json_bytes(payload))


def artifact_from_mapping(
    raw: Mapping[str, Any],
    project_root: Path,
    *,
    ledger: bool,
) -> ArtifactRef:
    expected_keys = {"path", "sha256", "bytes", "row_count"}
    if set(raw) != expected_keys:
        raise ProtocolValidationError(
            f"immutable artifact keys differ: expected {sorted(expected_keys)}, "
            f"got {sorted(raw)}"
        )
    path = Path(str(raw["path"]))
    if path.is_absolute() or ".." in path.parts:
        raise ProtocolValidationError("immutable artifact path must be project-relative")
    root = project_root.resolve()
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise ProtocolValidationError("immutable artifact escaped project root")
    artifact = ArtifactRef(resolved.resolve(), str(raw["sha256"]))
    artifact.verify()
    if not isinstance(raw["bytes"], int) or isinstance(raw["bytes"], bool):
        raise ProtocolValidationError("immutable artifact bytes must be an integer")
    expected_bytes = int(raw["bytes"])
    if expected_bytes <= 0 or artifact.path.stat().st_size != expected_bytes:
        raise ProtocolValidationError("immutable artifact byte count mismatch")
    row_count = raw["row_count"]
    if ledger:
        if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count <= 0:
            raise ProtocolValidationError("ledger artifact requires positive row_count")
        with artifact.path.open("r", encoding="utf-8") as handle:
            actual_rows = sum(1 for line in handle if line.strip())
        if actual_rows != row_count:
            raise ProtocolValidationError("immutable ledger row_count mismatch")
    elif row_count is not None:
        raise ProtocolValidationError("non-ledger artifact row_count must be null")
    return artifact


def load_selection_seal(path: Path, project_root: Path) -> Work9SelectionSeal:
    raw = json.loads(path.read_text(encoding="utf-8"))
    coordinates = Work9Coordinates(
        protocol_id=str(raw["protocol_id"]),
        run_family=str(raw["run_family"]),
        condition=str(raw["condition"]),
        partition_repeat=int(raw["partition_repeat"]),
        outer_fold=int(raw["outer_fold"]),
        seed=int(raw["seed"]),
    )
    return Work9SelectionSeal(
        schema_version=str(raw["schema_version"]),
        coordinates=coordinates,
        run_id=str(raw["run_id"]),
        config_sha256=str(raw["config_sha256"]),
        work8_receipt_sha256=str(raw["work8_receipt_sha256"]),
        runner_acceptance_sha256=str(raw["runner_acceptance_sha256"]),
        experiment_card_sha256=str(raw["experiment_card_sha256"]),
        gate1_policy_sha256=str(raw["gate1_policy_sha256"]),
        gate1_analysis_protocol_sha256=str(
            raw["gate1_analysis_protocol_sha256"]
        ),
        train_manifest_sha256=str(raw["train_manifest_sha256"]),
        validation_manifest_sha256=str(raw["validation_manifest_sha256"]),
        feature_store_metadata_sha256=str(raw["feature_store_metadata_sha256"]),
        feature_store_index_sha256=str(raw["feature_store_index_sha256"]),
        feature_store_vectors_sha256=str(raw["feature_store_vectors_sha256"]),
        pairing_plan_sha256=(
            None if raw["pairing_plan_sha256"] is None else str(raw["pairing_plan_sha256"])
        ),
        pairing_plan_artifact=(
            None
            if raw["pairing_plan_artifact"] is None
            else artifact_from_mapping(
                raw["pairing_plan_artifact"], project_root, ledger=True
            )
        ),
        selected_epoch=int(raw["selected_epoch"]),
        validation_call_nll=float(raw["validation_call_nll"]),
        threshold=float(raw["threshold"]),
        checkpoint=artifact_from_mapping(raw["checkpoint"], project_root, ledger=False),
        validation_segment_ledger=artifact_from_mapping(
            raw["validation_segment_ledger"], project_root, ledger=True
        ),
        validation_call_ledger=artifact_from_mapping(
            raw["validation_call_ledger"], project_root, ledger=True
        ),
        epoch_trace=artifact_from_mapping(raw["epoch_trace"], project_root, ledger=False),
        threshold_trace=artifact_from_mapping(
            raw["threshold_trace"], project_root, ledger=False
        ),
    )


def artifact_record(
    path: Path,
    project_root: Path,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "row_count": row_count,
    }
