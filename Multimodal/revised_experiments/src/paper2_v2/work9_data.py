"""Manifest-bounded datasets and semantics-preserving MFCC cache for Work 9."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from .data import FeatureDataset, FeatureSample, InputSpec
from .pairing import PairingPlan
from .text_features import IndexedTextStore


CACHE_POLICY = "run_local_read_only_mfcc_tensor_cache_v1"


class Work9FeatureDataset(Dataset[FeatureSample]):
    """Feature loader whose complete universe is one frozen fold manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        input_spec: InputSpec,
        project_root: str | Path,
        text_store: IndexedTextStore | None,
        pairing_plan: PairingPlan | None = None,
        cache_mfcc: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.input_spec = input_spec
        self.text_store = text_store
        self.pairing_plan = pairing_plan
        self.cache_mfcc = bool(cache_mfcc)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.rows: list[dict[str, Any]] = [
                json.loads(line) for line in handle if line.strip()
            ]
        if not self.rows:
            raise ValueError("Work 9 manifest is empty")
        self.segment_ids = tuple(str(row["segment_id"]) for row in self.rows)
        if len(set(self.segment_ids)) != len(self.segment_ids):
            raise ValueError("Work 9 manifest contains duplicate segment_id")
        self.row_by_segment = {
            str(row["segment_id"]): row for row in self.rows
        }
        if pairing_plan is not None:
            if "text" not in input_spec.global_features:
                raise ValueError("pairing plan requires an active text branch")
            if set(pairing_plan.donor_by_receiver) != set(self.segment_ids):
                raise ValueError("pairing plan universe differs from train manifest")
            if not set(pairing_plan.donor_by_receiver.values()) <= set(self.segment_ids):
                raise ValueError("pairing donor lies outside the train manifest")
        if "text" in input_spec.global_features and text_store is None:
            raise ValueError("active text branch requires an indexed text store")
        for row in self.rows:
            segment_id = str(row["segment_id"])
            if (
                "text" in input_spec.global_features
                and text_store is not None
                and segment_id not in text_store
            ):
                raise KeyError(f"manifest segment absent from sealed text store: {segment_id}")
        self.call_count = len({str(row["call_id"]) for row in self.rows})
        self.segment_count = len(self.rows)
        call_segments = Counter(str(row["call_id"]) for row in self.rows)
        for row in self.rows:
            expected = 1.0 / call_segments[str(row["call_id"])]
            if not math.isclose(
                float(row.get("sample_weight", float("nan"))),
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("manifest sample_weight is not inverse call length")
        total_weight = sum(float(row["sample_weight"]) for row in self.rows)
        if not math.isclose(total_weight, self.call_count, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("manifest sample weights do not total one per call")
        self.loss_scale = self.segment_count / self.call_count

        self._mfcc: tuple[torch.Tensor, ...] | None = None
        self.cache_content_sha256: str | None = None
        self.cache_bytes = 0
        if "mfcc" in input_spec.sequence_features and self.cache_mfcc:
            values = []
            digest = hashlib.sha256()
            for row in self.rows:
                path = self._resolve(str(row["mfcc_path"]))
                tensor = FeatureDataset._load_array(
                    path, expected_dim=13, sequence=True
                ).contiguous()
                values.append(tensor)
                encoded_id = str(row["segment_id"]).encode("utf-8")
                digest.update(len(encoded_id).to_bytes(8, "big"))
                digest.update(encoded_id)
                digest.update(tensor.shape[0].to_bytes(8, "big"))
                digest.update(tensor.numpy().tobytes(order="C"))
                self.cache_bytes += tensor.numel() * tensor.element_size()
            self._mfcc = tuple(values)
            self.cache_content_sha256 = digest.hexdigest()

    def _resolve(self, raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else self.project_root / path

    def __len__(self) -> int:
        return len(self.rows)

    def _mfcc_value(self, index: int, row: Mapping[str, Any]) -> torch.Tensor:
        if self._mfcc is not None:
            return self._mfcc[index]
        return FeatureDataset._load_array(
            self._resolve(str(row["mfcc_path"])), expected_dim=13, sequence=True
        )

    def __getitem__(self, index: int) -> FeatureSample:
        row = self.rows[index]
        segment_id = str(row["segment_id"])
        sequences = {}
        globals_ = {}
        if "mfcc" in self.input_spec.sequence_features:
            sequences["mfcc"] = self._mfcc_value(index, row)
        if "text" in self.input_spec.global_features:
            if self.text_store is None:  # constructor invariant
                raise RuntimeError("active text branch has no indexed store")
            donor_id = (
                self.pairing_plan.donor_segment_id(segment_id)
                if self.pairing_plan is not None
                else segment_id
            )
            if donor_id not in self.row_by_segment and self.pairing_plan is not None:
                raise RuntimeError("text donor escaped the train manifest")
            globals_["text"] = self.text_store.get(donor_id)
        return FeatureSample(
            sequences=sequences,
            globals=globals_,
            label=float(row["label"]),
            sample_weight=float(row["sample_weight"]),
            call_id=str(row["call_id"]),
            segment_id=segment_id,
            group_id=str(row["group_id"]),
            source=str(row.get("source", "unknown")),
            start=float(row["start"]),
            end=float(row["end"]),
        )
