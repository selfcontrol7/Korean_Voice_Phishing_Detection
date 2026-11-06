"""Append-only run directories for Paper 2 v2.

Every run receives a unique directory. Files are written with exclusive-create
semantics, and outer evaluation requires a one-time sealed authorization
receipt. Nothing in this module resolves or writes historical v1 paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .conditions import CORE_CONDITIONS
from .config import ArtifactRef, SelectionSeal, sha256_file


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True)
class RunCoordinates:
    protocol_id: str
    condition: str
    partition_repeat: int
    outer_fold: int
    seed: int

    def __post_init__(self) -> None:
        if self.condition not in CORE_CONDITIONS:
            raise ValueError("run condition is not in the Work 8 matrix")
        if self.partition_repeat < 0 or self.outer_fold < 0:
            raise ValueError("repeat and outer fold must be nonnegative")
        if not SAFE_COMPONENT.fullmatch(self.protocol_id):
            raise ValueError("protocol_id must be a filesystem-safe identifier")


class RunHandle:
    def __init__(self, path: Path, coordinates: RunCoordinates) -> None:
        self.path = path
        self.coordinates = coordinates

    def _target(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("run artifact path must be a safe relative path")
        target = (self.path / relative).resolve()
        if self.path.resolve() not in target.parents:
            raise ValueError("run artifact path escapes the reserved run directory")
        return target

    def write_bytes_once(self, relative_path: str | Path, payload: bytes) -> ArtifactRef:
        target = self._target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return ArtifactRef(target, sha256_file(target))

    def write_json_once(
        self, relative_path: str | Path, payload: Mapping[str, Any] | list[Any]
    ) -> ArtifactRef:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=_json_default,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        return self.write_bytes_once(relative_path, encoded)

    def write_jsonl_once(
        self, relative_path: str | Path, rows: Iterable[Mapping[str, Any]]
    ) -> ArtifactRef:
        lines = [
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=_json_default,
                allow_nan=False,
            )
            for row in rows
        ]
        if not lines:
            raise ValueError("refusing to write an empty JSONL artifact")
        return self.write_bytes_once(relative_path, ("\n".join(lines) + "\n").encode("utf-8"))

    def validate_json_schema(
        self, payload: Mapping[str, Any], *, schema_path: str | Path
    ) -> None:
        """Validate against the Work 1 schema, including date-time formats."""

        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover - environment guard
            raise RuntimeError("jsonschema is required for registry validation") from error
        with Path(schema_path).open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
        if errors:
            descriptions = [error.message for error in errors[:10]]
            raise ValueError(f"run record does not satisfy registry schema: {descriptions}")

    def write_validated_run_record_once(
        self,
        payload: Mapping[str, Any],
        *,
        schema_path: str | Path,
        relative_path: str | Path = "run_record.json",
    ) -> ArtifactRef:
        self.validate_json_schema(payload, schema_path=schema_path)
        return self.write_json_once(relative_path, payload)

    def _validate_outer_evaluation_seal(self, seal: SelectionSeal) -> None:
        if (
            seal.protocol_id != self.coordinates.protocol_id
            or seal.condition != self.coordinates.condition
            or seal.partition_repeat != self.coordinates.partition_repeat
            or seal.outer_fold != self.coordinates.outer_fold
            or seal.seed != self.coordinates.seed
        ):
            raise ValueError("selection seal does not match this run")
        seal.checkpoint.verify()

    def _record_outer_evaluation_authorization(
        self, seal: SelectionSeal, *, coordinate_claim: ArtifactRef
    ) -> ArtifactRef:
        payload = {
            "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": seal.protocol_id,
            "condition": seal.condition,
            "partition_repeat": seal.partition_repeat,
            "outer_fold": seal.outer_fold,
            "seed": seal.seed,
            "checkpoint_sha256": seal.checkpoint.sha256,
            "validation_predictions_sha256": seal.validation_predictions_sha256,
            "threshold": seal.threshold,
            "coordinate_claim": {
                "path": str(coordinate_claim.path),
                "sha256": coordinate_claim.sha256,
            },
        }
        return self.write_json_once("outer_evaluation_authorization.json", payload)

    def mark_completed(self, summary: Mapping[str, Any]) -> ArtifactRef:
        return self.write_json_once(
            "TERMINAL.json",
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": dict(summary),
            },
        )

    def mark_failed(self, failure: Mapping[str, Any]) -> ArtifactRef:
        return self.write_json_once(
            "TERMINAL.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure": dict(failure),
            },
        )


class RunRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if "revised_experiments" not in self.root.parts:
            raise ValueError("v2 run registry must live under revised_experiments/")
        if "modeling" in self.root.parts:
            raise ValueError("v2 run registry cannot use historical modeling/ output paths")
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_run_id(coordinates: RunCoordinates) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = uuid.uuid4().hex[:10]
        return (
            f"{coordinates.protocol_id}__{coordinates.condition}__"
            f"r{coordinates.partition_repeat:02d}-o{coordinates.outer_fold:02d}-"
            f"s{coordinates.seed}__{timestamp}-{suffix}"
        )

    def reserve(
        self, coordinates: RunCoordinates, *, run_id: str | None = None
    ) -> RunHandle:
        run_id = run_id or self.generate_run_id(coordinates)
        if (
            not SAFE_COMPONENT.fullmatch(run_id)
            or run_id in {".", ".."}
            or run_id.startswith("_")
        ):
            raise ValueError("run_id must be a safe single path component")
        path = self.root / run_id
        path.mkdir(exist_ok=False)
        handle = RunHandle(path, coordinates)
        handle.write_json_once(
            "reservation.json",
            {
                "run_id": run_id,
                "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
                **asdict(coordinates),
            },
        )
        return handle

    def claim_outer_evaluation(
        self, handle: RunHandle, seal: SelectionSeal
    ) -> ArtifactRef:
        """Authorize outer evaluation once per protocol/condition/fold/seed.

        The claim lives at registry scope, not merely inside a unique run
        directory. Creating another run ID for identical coordinates therefore
        cannot re-open the same outer-evaluation partition.
        """

        if handle.path.parent.resolve() != self.root:
            raise ValueError("run handle does not belong to this registry")
        reservation_path = handle.path / "reservation.json"
        if not reservation_path.is_file():
            raise ValueError("run handle has no registry reservation")
        with reservation_path.open("r", encoding="utf-8") as reservation_file:
            reservation = json.load(reservation_file)
        reserved_coordinates = {
            key: reservation.get(key)
            for key in (
                "protocol_id",
                "condition",
                "partition_repeat",
                "outer_fold",
                "seed",
            )
        }
        if reserved_coordinates != asdict(handle.coordinates):
            raise ValueError("run handle coordinates differ from its registry reservation")
        handle._validate_outer_evaluation_seal(seal)
        coordinate_payload = {
            "protocol_id": seal.protocol_id,
            "condition": seal.condition,
            "partition_repeat": seal.partition_repeat,
            "outer_fold": seal.outer_fold,
            "seed": seal.seed,
        }
        canonical = json.dumps(
            coordinate_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        coordinate_key = hashlib.sha256(canonical).hexdigest()
        claims_root = self.root / "_outer_eval_claims"
        claims_root.mkdir(exist_ok=True)
        claim_path = claims_root / f"{coordinate_key}.json"
        claim_payload = {
            **coordinate_payload,
            "run_id": handle.path.name,
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": seal.checkpoint.sha256,
            "validation_predictions_sha256": seal.validation_predictions_sha256,
            "threshold": seal.threshold,
        }
        encoded = (
            json.dumps(
                claim_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with claim_path.open("xb") as claim_file:
            claim_file.write(encoded)
            claim_file.flush()
            os.fsync(claim_file.fileno())
        coordinate_claim = ArtifactRef(claim_path, sha256_file(claim_path))
        handle._record_outer_evaluation_authorization(
            seal, coordinate_claim=coordinate_claim
        )
        return coordinate_claim
