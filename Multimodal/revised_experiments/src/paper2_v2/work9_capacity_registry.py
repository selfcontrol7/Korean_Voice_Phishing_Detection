"""Atomic, append-only registry for the two-phase Work 9 matrix."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import ArtifactRef, sha256_file
from .work9_capacity_types import (
    ANCHOR_CONDITIONS,
    PRIMARY_REGISTRY_RELATIVE,
    REGISTRY_RELATIVE,
    SMOKE_EVIDENCE_ROOT_RELATIVE,
    TEST_SANDBOX_ROOT_RELATIVE,
    CapacityCoordinates,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)


def _overlaps(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    return left == right or left in right.parents or right in left.parents


def _reject_symlink_components(path: Path, *, stop: Path) -> None:
    """Reject existing symlinks from ``stop`` through ``path`` (inclusive)."""

    stop = stop.resolve()
    candidate = path
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(stop)
        except ValueError as error:
            raise ValueError("isolated output escaped project root") from error
    else:
        relative = candidate
    cursor = stop
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlinked output component is forbidden: {cursor}")


def validate_isolated_relative_root(
    project_root: str | Path,
    value: str | Path,
    *,
    allowed_parent: str | Path,
    context: str,
    require_new: bool = False,
) -> Path:
    """Resolve a custom evidence/test root under one dedicated artifact tree.

    Public custom output knobs are deliberately lexical project-relative paths:
    absolute paths, ``..`` traversal, symlinks, and overlap with either
    production registry or frozen source/config trees fail closed.
    """

    root = Path(project_root).resolve()
    raw = Path(value)
    if raw.is_absolute() or raw == Path(".") or ".." in raw.parts:
        raise ValueError(f"{context} must be a safe project-relative path")
    _reject_symlink_components(raw, stop=root)
    resolved = (root / raw).resolve()
    allowed = (root / Path(allowed_parent)).resolve()
    if allowed not in resolved.parents:
        raise ValueError(f"{context} must be a child of {allowed_parent}")
    forbidden = (
        (root / PRIMARY_REGISTRY_RELATIVE).resolve(),
        (root / REGISTRY_RELATIVE).resolve(),
        (root / "revised_experiments/src").resolve(),
        (root / "revised_experiments/configs").resolve(),
        (root / "revised_experiments/registry/schemas").resolve(),
    )
    if any(_overlaps(resolved, blocked) for blocked in forbidden):
        raise ValueError(f"{context} overlaps a frozen or production tree")
    if require_new and resolved.exists():
        raise FileExistsError(f"{context} must be new: {resolved}")
    return resolved


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


class CapacityRunWorkspace:
    def __init__(self, path: Path, coordinates: CapacityCoordinates) -> None:
        self.path = path.resolve()
        self.coordinates = coordinates

    def target(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute() or value == Path(".") or ".." in value.parts:
            raise ValueError("run artifact must be a safe relative path")
        lexical_target = self.path / value
        _reject_symlink_components(lexical_target, stop=self.path)
        target = lexical_target.resolve()
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
        lock_path = self.target("run.lock")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("run lock must be a regular non-symlink file")
        with os.fdopen(descriptor, "r+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"run coordinate is active: {self.path.name}") from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CapacityRegistry:
    def __init__(
        self,
        project_root: str | Path,
        *,
        root: str | Path | None = None,
        allow_test_root: bool = False,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        canonical = (self.project_root / REGISTRY_RELATIVE).resolve()
        if root is None:
            self.root = canonical
        elif allow_test_root:
            errors: list[Exception] = []
            for parent in (TEST_SANDBOX_ROOT_RELATIVE, SMOKE_EVIDENCE_ROOT_RELATIVE):
                try:
                    self.root = validate_isolated_relative_root(
                        self.project_root,
                        root,
                        allowed_parent=parent,
                        context="capacity isolated registry",
                    )
                    break
                except (ValueError, FileExistsError) as error:
                    errors.append(error)
            else:
                raise ValueError(
                    "capacity isolated registry is outside dedicated test/smoke roots"
                ) from errors[-1]
        else:
            raw = Path(root)
            if raw.is_absolute() or ".." in raw.parts:
                raise ValueError("production registry override must be project-relative")
            self.root = (self.project_root / raw).resolve()
        if not allow_test_root and self.root != canonical:
            raise ValueError("production Work 9 registry root is fixed by protocol")
        _reject_symlink_components(self.root, stop=self.project_root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("capacity registry root cannot be a symlink")

    def reserve(self, coordinates: CapacityCoordinates) -> CapacityRunWorkspace:
        path = self.root / coordinates.run_id
        canonical = (self.project_root / REGISTRY_RELATIVE).resolve()
        if self.root == canonical and coordinates.condition in ANCHOR_CONDITIONS:
            raise ValueError("immutable primary anchors cannot reserve production capacity runs")
        if path.is_symlink():
            raise RuntimeError("pre-existing symlinked run directory is forbidden")
        reservation = {
            "schema_version": "paper2-v2-work9-capacity-reservation-1",
            "run_id": coordinates.run_id,
            **asdict(coordinates),
        }
        if not path.exists():
            path.mkdir(exist_ok=False)
            atomic_write_once(path / "reservation.json", canonical_json_bytes(reservation))
        else:
            if not path.is_dir() or path.is_symlink():
                raise RuntimeError("existing run reservation is not a real directory")
            stored_path = path / "reservation.json"
            if (
                stored_path.is_symlink()
                or not stored_path.is_file()
                or json.loads(stored_path.read_text()) != reservation
            ):
                raise RuntimeError("existing run directory has a different reservation")
        return CapacityRunWorkspace(path, coordinates)

    def claim_outer(
        self,
        workspace: CapacityRunWorkspace,
        *,
        selection_seal_sha256: str,
        selection_index_sha256: str,
        checkpoint_sha256: str,
    ) -> ArtifactRef:
        claims = self.root / "_outer_eval_claims"
        expected_workspace = self.root / workspace.coordinates.run_id
        _reject_symlink_components(expected_workspace, stop=self.root)
        if workspace.path != expected_workspace.resolve():
            raise ValueError("outer claim workspace is outside this capacity registry")
        _reject_symlink_components(claims, stop=self.root)
        if claims.exists():
            if claims.is_symlink() or not claims.is_dir():
                raise ValueError("outer claim root must be a real directory")
        else:
            claims.mkdir(exist_ok=False)
        _reject_symlink_components(claims, stop=self.root)
        payload = {
            "schema_version": "paper2-v2-work9-capacity-outer-claim-1",
            "run_id": workspace.coordinates.run_id,
            **asdict(workspace.coordinates),
            "selection_seal_sha256": selection_seal_sha256,
            "selection_index_sha256": selection_index_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
        target = claims / f"{workspace.coordinates.claim_key}.json"
        _reject_symlink_components(target, stop=self.root)
        if target.is_symlink() or target.resolve().parent != claims.resolve():
            raise ValueError("outer claim target is symlinked or escaped")
        # Same-run, same-seal resume is idempotent. Any changed run lineage or
        # checkpoint is a hard one-shot conflict.
        return atomic_write_or_verify(target, canonical_json_bytes(payload))

    def write_matrix_artifact_once(
        self, relative: str | Path, payload: Mapping[str, Any] | list[Any]
    ) -> ArtifactRef:
        value = Path(relative)
        if value.is_absolute() or value == Path(".") or ".." in value.parts:
            raise ValueError("matrix artifact must be a safe relative path")
        matrix_root = (self.root / "_matrix").resolve()
        target = (matrix_root / value).resolve()
        if matrix_root not in target.parents:
            raise ValueError("matrix artifact escaped the matrix directory")
        _reject_symlink_components(target, stop=self.project_root)
        return atomic_write_or_verify(target, canonical_json_bytes(payload))
