"""Atomic, append-only registry for the two-phase Work 9 matrix."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import ArtifactRef, sha256_file
from .work9_types import (
    REGISTRY_RELATIVE,
    Work9Coordinates,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temp(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o444)
    return temporary


def atomic_write_once(path: Path, payload: bytes) -> ArtifactRef:
    temporary = _write_temp(path, payload)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return ArtifactRef(path.resolve(), sha256_file(path))


def atomic_write_or_verify(path: Path, payload: bytes) -> ArtifactRef:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable artifact differs on resume: {path}")
        return ArtifactRef(path.resolve(), sha256_file(path))
    try:
        return atomic_write_once(path, payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise
        return ArtifactRef(path.resolve(), sha256_file(path))


class Work9RunWorkspace:
    def __init__(self, path: Path, coordinates: Work9Coordinates) -> None:
        self.path = path.resolve()
        self.coordinates = coordinates

    def target(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute() or value == Path(".") or ".." in value.parts:
            raise ValueError("run artifact must be a safe relative path")
        target = (self.path / value).resolve()
        if self.path not in target.parents:
            raise ValueError("run artifact escapes reserved run directory")
        return target

    def _ensure_open(self, relative: str | Path) -> None:
        if (self.path / "TERMINAL.json").exists() and Path(relative).name != "TERMINAL.json":
            raise RuntimeError("terminal Work 9 run is immutable")

    def write_bytes_once(self, relative: str | Path, payload: bytes) -> ArtifactRef:
        self._ensure_open(relative)
        return atomic_write_once(self.target(relative), payload)

    def write_bytes_or_verify(self, relative: str | Path, payload: bytes) -> ArtifactRef:
        self._ensure_open(relative)
        return atomic_write_or_verify(self.target(relative), payload)

    def write_json_once(
        self, relative: str | Path, payload: Mapping[str, Any] | list[Any]
    ) -> ArtifactRef:
        return self.write_bytes_once(relative, canonical_json_bytes(payload))

    def write_json_or_verify(
        self, relative: str | Path, payload: Mapping[str, Any] | list[Any]
    ) -> ArtifactRef:
        return self.write_bytes_or_verify(relative, canonical_json_bytes(payload))

    def write_jsonl_once(
        self, relative: str | Path, rows: list[Mapping[str, Any]]
    ) -> ArtifactRef:
        return self.write_bytes_once(relative, canonical_jsonl_bytes(rows))

    def write_jsonl_or_verify(
        self, relative: str | Path, rows: list[Mapping[str, Any]]
    ) -> ArtifactRef:
        return self.write_bytes_or_verify(relative, canonical_jsonl_bytes(rows))

    def publish_directory_once(
        self, relative: str | Path, files: Mapping[str, bytes]
    ) -> dict[str, ArtifactRef]:
        self._ensure_open(relative)
        for name in files:
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts or candidate == Path("."):
                raise ValueError("published directory member path is unsafe")
        final = self.target(relative)
        if final.exists():
            if any(path.is_symlink() for path in final.rglob("*")):
                raise FileExistsError(
                    f"immutable directory contains a forbidden symlink: {final}"
                )
            actual_files = {
                path.relative_to(final).as_posix()
                for path in final.rglob("*")
                if path.is_file()
            }
            if actual_files != set(files):
                raise FileExistsError(
                    f"immutable directory inventory differs on resume: {final}"
                )
            references = {}
            for name, payload in files.items():
                references[name] = atomic_write_or_verify(final / name, payload)
            return references
        temporary = final.parent / f".{final.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            for name, payload in files.items():
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                target.chmod(0o444)
            for directory in sorted(
                (item for item in temporary.rglob("*") if item.is_dir()), reverse=True
            ):
                directory.chmod(0o555)
            temporary.chmod(0o555)
            os.rename(temporary, final)
            _fsync_directory(final.parent)
        except BaseException:
            if temporary.exists():
                temporary.chmod(0o755)
                for directory in temporary.rglob("*"):
                    if directory.is_dir():
                        directory.chmod(0o755)
                    else:
                        directory.chmod(0o644)
                shutil.rmtree(temporary)
            raise
        return {
            name: ArtifactRef((final / name).resolve(), sha256_file(final / name))
            for name in files
        }

    def record_failure(self, payload: Mapping[str, Any]) -> ArtifactRef:
        relative = f"failures/failure-{uuid.uuid4().hex}.json"
        return self.write_json_once(relative, payload)

    @contextmanager
    def lock(self) -> Iterator[None]:
        lock_path = self.path / "run.lock"
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"run coordinate is active: {self.path.name}") from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Work9Registry:
    def __init__(
        self,
        project_root: str | Path,
        *,
        root: str | Path | None = None,
        allow_test_root: bool = False,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        canonical = (self.project_root / REGISTRY_RELATIVE).resolve()
        self.root = Path(root).resolve() if root is not None else canonical
        if not allow_test_root and self.root != canonical:
            raise ValueError("production Work 9 registry root is fixed by protocol")
        if "revised_experiments" not in self.root.parts:
            raise ValueError("Work 9 registry must be under revised_experiments")
        self.root.mkdir(parents=True, exist_ok=True)

    def reserve(self, coordinates: Work9Coordinates) -> Work9RunWorkspace:
        path = self.root / coordinates.run_id
        reservation = {
            "schema_version": "paper2-v2-work9-reservation-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
        }
        if not path.exists():
            path.mkdir(exist_ok=False)
            atomic_write_once(path / "reservation.json", canonical_json_bytes(reservation))
        else:
            stored_path = path / "reservation.json"
            if not stored_path.is_file() or json.loads(stored_path.read_text()) != reservation:
                raise RuntimeError("existing run directory has a different reservation")
        return Work9RunWorkspace(path, coordinates)

    def claim_outer(
        self,
        workspace: Work9RunWorkspace,
        *,
        selection_seal_sha256: str,
        selection_index_sha256: str,
        checkpoint_sha256: str,
    ) -> ArtifactRef:
        claims = self.root / "_outer_eval_claims"
        claims.mkdir(exist_ok=True)
        payload = {
            "schema_version": "paper2-v2-work9-outer-claim-1",
            "run_id": workspace.coordinates.run_id,
            **asdict(workspace.coordinates),
            "selection_seal_sha256": selection_seal_sha256,
            "selection_index_sha256": selection_index_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
        target = claims / f"{workspace.coordinates.claim_key}.json"
        # Same-run, same-seal resume is idempotent. Any changed run lineage or
        # checkpoint is a hard one-shot conflict.
        return atomic_write_or_verify(target, canonical_json_bytes(payload))

    def write_matrix_artifact_once(
        self, relative: str | Path, payload: Mapping[str, Any] | list[Any]
    ) -> ArtifactRef:
        target = self.root / "_matrix" / Path(relative)
        return atomic_write_or_verify(target, canonical_json_bytes(payload))
