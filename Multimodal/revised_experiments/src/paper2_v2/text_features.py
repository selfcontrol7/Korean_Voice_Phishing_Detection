"""Versioned, indexed sentence-vector stores for Paper 2 v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IndexedTextStore:
    """Read a sealed `[N,D]` matrix by stable `segment_id`."""

    def __init__(self, directory: str | Path, *, verify_hashes: bool = True) -> None:
        self.directory = Path(directory).resolve()
        metadata_path = self.directory / "feature_store.json"
        index_path = self.directory / "index.jsonl"
        vectors_path = self.directory / "vectors.npy"
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("status") != "sealed":
            raise ValueError(f"text store is not sealed: {self.directory}")
        if verify_hashes:
            expected_index = self.metadata.get("index_sha256")
            expected_vectors = self.metadata.get("vectors_sha256")
            if expected_index != sha256_file(index_path):
                raise ValueError("text-store index hash mismatch")
            if expected_vectors != sha256_file(vectors_path):
                raise ValueError("text-store vector hash mismatch")
        self.vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        if self.vectors.dtype != np.float32 or self.vectors.ndim != 2:
            raise ValueError("text-store vectors must be float32 [N,D]")
        self.row_by_segment: dict[str, int] = {}
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                segment_id = str(record["segment_id"])
                row_index = int(record["row_index"])
                if segment_id in self.row_by_segment:
                    raise ValueError(f"duplicate segment_id in index: {segment_id}")
                if row_index != len(self.row_by_segment):
                    raise ValueError(
                        f"noncontiguous text-store row at line {line_number}: {row_index}"
                    )
                self.row_by_segment[segment_id] = row_index
        if len(self.row_by_segment) != self.vectors.shape[0]:
            raise ValueError("text-store index/vector row mismatch")
        expected_shape = self.metadata.get("shape")
        if expected_shape != list(self.vectors.shape):
            raise ValueError("text-store sidecar shape mismatch")
        self.feature_set_id = str(self.metadata["feature_set_id"])
        self.dimension = int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.row_by_segment)

    def __contains__(self, segment_id: str) -> bool:
        return segment_id in self.row_by_segment

    def get(self, segment_id: str) -> Tensor:
        try:
            row = self.row_by_segment[segment_id]
        except KeyError as exc:
            raise KeyError(f"segment not present in text store: {segment_id}") from exc
        # A writable copy avoids undefined behavior if a caller later modifies a
        # tensor created from a read-only memory map.
        return torch.from_numpy(np.array(self.vectors[row], dtype=np.float32, copy=True))
