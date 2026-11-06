#!/usr/bin/env python3
"""Create a content-addressed, non-destructive snapshot inventory of Paper 2 v1.

The script does not copy or modify source artifacts. It writes SHA-256 hashes,
Git working-tree metadata, an environment record, and a conservative historical
checkpoint/result candidate ledger. Unknown lineage is recorded as unknown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class FileRecord:
    path: str
    role: str
    size_bytes: int
    mtime_ns: int
    sha256: str


def run(command: list[str], cwd: Path, check: bool = True) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def role_for(relative_path: str) -> str:
    if relative_path.startswith("Multimodal/features/"):
        return "feature"
    if relative_path.startswith("Multimodal/modeling/models/"):
        return "checkpoint"
    if relative_path.startswith("Multimodal/modeling/logs/"):
        return "result_or_log"
    if relative_path.startswith("Multimodal/data/"):
        return "manifest_or_split"
    if relative_path.startswith("Multimodal/modeling/"):
        return "modeling_code"
    if relative_path.startswith("Multimodal/features_extraction/"):
        return "feature_code"
    if relative_path.startswith("Multimodal/preprocessing/"):
        return "preprocessing_code"
    return "project_record"


def iter_scope_files(repo_root: Path) -> list[Path]:
    recursive_roots = [
        repo_root / "Multimodal/features",
        repo_root / "Multimodal/modeling",
        repo_root / "Multimodal/features_extraction",
        repo_root / "Multimodal/preprocessing",
    ]
    explicit_globs = [
        "Multimodal/data/*.jsonl",
        "Multimodal/data/*.json",
        "Multimodal/data/cv_5fold/**/*.jsonl",
        "Multimodal/data/cv_5fold/**/*.json",
        "Multimodal/paper2/paper_b_experiment_setup*",
        "requirements.txt",
        "CLAUDE.md",
    ]
    paths: set[Path] = set()
    for root in recursive_roots:
        if root.exists():
            paths.update(path for path in root.rglob("*") if path.is_file())
    for pattern in explicit_globs:
        paths.update(path for path in repo_root.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(repo_root).as_posix())


def create_record(repo_root: Path, path: Path) -> FileRecord:
    stat = path.stat()
    relative = path.relative_to(repo_root).as_posix()
    return FileRecord(
        path=relative,
        role=role_for(relative),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha256_file(path),
    )


def write_git_records(repo_root: Path, output_dir: Path) -> dict[str, str]:
    commands = {
        "git_head.txt": ["git", "rev-parse", "HEAD"],
        "git_branch.txt": ["git", "branch", "--show-current"],
        "git_status_porcelain_v1.txt": ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        "git_diff_multimodal.patch": [
            "git", "diff", "--binary", "--", "Multimodal/modeling",
            "Multimodal/features_extraction", "Multimodal/preprocessing",
            "Multimodal/data/*.jsonl", "requirements.txt",
        ],
        "git_diff_cached_multimodal.patch": [
            "git", "diff", "--cached", "--binary", "--",
            "Multimodal/modeling", "Multimodal/features_extraction",
            "Multimodal/preprocessing", "Multimodal/data/*.jsonl",
            "requirements.txt",
        ],
    }
    hashes: dict[str, str] = {}
    for filename, command in commands.items():
        path = output_dir / filename
        path.write_text(run(command, repo_root), encoding="utf-8")
        hashes[filename] = sha256_file(path)
    return hashes


def write_environment_record(repo_root: Path, output_dir: Path) -> dict[str, str]:
    environment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    path = output_dir / "environment.json"
    path.write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    freeze_path = output_dir / "pip_freeze.txt"
    freeze_path.write_text(run([sys.executable, "-m", "pip", "freeze"], repo_root), encoding="utf-8")
    gpu_path = output_dir / "nvidia_smi.txt"
    try:
        gpu_text = run(["nvidia-smi"], repo_root)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        gpu_text = f"unavailable: {exc}\n"
    gpu_path.write_text(gpu_text, encoding="utf-8")
    return {item.name: sha256_file(item) for item in (path, freeze_path, gpu_path)}


def result_candidates(repo_root: Path, checkpoint: Path) -> list[Path]:
    eval_root = repo_root / "Multimodal/modeling/logs/eval_results"
    stem = checkpoint.stem.lower()
    tokens = [
        token for token in (
            "mfcc", "egemaps", "wav2vec", "all", "krsbert", "kobert", "both"
        ) if token in stem
    ]
    fold = checkpoint.parent.name if checkpoint.parent.name.startswith("fold") else None
    candidates = sorted(eval_root.rglob("test_results*.json")) if eval_root.exists() else []
    filtered: list[Path] = []
    for candidate in candidates:
        text = candidate.as_posix().lower()
        if fold and fold not in text:
            continue
        if tokens and not all(token in text for token in tokens):
            continue
        filtered.append(candidate)
    return filtered


def write_lineage_ledger(repo_root: Path, output_dir: Path) -> None:
    checkpoints = sorted((repo_root / "Multimodal/modeling/models").rglob("*"))
    checkpoints = [path for path in checkpoints if path.is_file()]
    ledger_path = output_dir / "historical_checkpoint_result_candidates.csv"
    fields = [
        "checkpoint_path", "checkpoint_sha256", "candidate_result_paths",
        "candidate_count", "mapping_status", "mapping_note",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for checkpoint in checkpoints:
            candidates = result_candidates(repo_root, checkpoint)
            count = len(candidates)
            status = "unique_filename_candidate" if count == 1 else (
                "multiple_filename_candidates" if count > 1 else "unknown"
            )
            writer.writerow({
                "checkpoint_path": checkpoint.relative_to(repo_root).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "candidate_result_paths": ";".join(
                    candidate.relative_to(repo_root).as_posix() for candidate in candidates
                ),
                "candidate_count": count,
                "mapping_status": status,
                "mapping_note": (
                    "Filename-derived candidate only; do not treat as verified lineage."
                    if count else "No defensible result mapping recovered; recorded as unknown."
                ),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def finalize_read_only_snapshot(output_dir: Path) -> str:
    """Hash generated snapshot records and remove write bits after completion."""
    manifest_path = output_dir / "snapshot_files.sha256"
    lines = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_sha256 = sha256_file(manifest_path)
    for path in sorted(output_dir.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    output_dir.chmod(0o555)
    return manifest_sha256


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    paths = iter_scope_files(repo_root)
    errors: list[dict[str, str]] = []
    records: list[FileRecord] = []

    def safe_record(path: Path) -> FileRecord | None:
        try:
            return create_record(repo_root, path)
        except Exception as exc:  # preserve a complete failure ledger
            errors.append({"path": str(path), "error": repr(exc)})
            return None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for record in executor.map(safe_record, paths):
            if record is not None:
                records.append(record)

    records.sort(key=lambda record: record.path)
    inventory_path = output_dir / "artifact_inventory.jsonl"
    root_digest = hashlib.sha256()
    with inventory_path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(asdict(record), sort_keys=True, ensure_ascii=False)
            handle.write(line + "\n")
            root_digest.update((line + "\n").encode("utf-8"))

    git_hashes = write_git_records(repo_root, output_dir)
    environment_hashes = write_environment_record(repo_root, output_dir)
    write_lineage_ledger(repo_root, output_dir)

    role_counts: dict[str, int] = {}
    role_bytes: dict[str, int] = {}
    for record in records:
        role_counts[record.role] = role_counts.get(record.role, 0) + 1
        role_bytes[record.role] = role_bytes.get(record.role, 0) + record.size_bytes

    summary = {
        "snapshot_name": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "file_count": len(records),
        "total_bytes": sum(record.size_bytes for record in records),
        "role_counts": role_counts,
        "role_bytes": role_bytes,
        "artifact_inventory_sha256": sha256_file(inventory_path),
        "artifact_merkle_like_root_sha256": root_digest.hexdigest(),
        "git_record_hashes": git_hashes,
        "environment_record_hashes": environment_hashes,
        "error_count": len(errors),
        "errors": errors,
        "lineage_policy": (
            "Filename candidates are not verified mappings. Unknown is retained "
            "when exact checkpoint-result lineage cannot be established."
        ),
    }
    (output_dir / "snapshot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "READ_ONLY_NOTICE.txt").write_text(
        "Exploratory v1 inventory. Do not edit files in this directory.\n",
        encoding="utf-8",
    )
    snapshot_files_sha256 = finalize_read_only_snapshot(output_dir)
    summary["snapshot_files_manifest_sha256"] = snapshot_files_sha256
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
