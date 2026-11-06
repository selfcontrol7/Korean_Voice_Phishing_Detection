"""Branch-selective feature loading and authoritative collation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .batch import ExperimentBatch, SequenceBatch
from .text_features import IndexedTextStore


SEQUENCE_DEFINITIONS = {
    "mfcc": ("mfcc_path", 13),
    "wav2vec2": ("wav2vec2_path", 768),
}
GLOBAL_DEFINITIONS = {
    "egemaps": ("egemaps_path", 88),
    "text": (None, 768),
}


@dataclass(frozen=True)
class InputSpec:
    """Declarative list of branches that may be touched by a condition."""

    sequence_features: tuple[str, ...] = ()
    global_features: tuple[str, ...] = ()
    text_path_key: str = "krsbert_text_path"
    expected_text_dim: int = 768

    def __post_init__(self) -> None:
        if len(set(self.sequence_features)) != len(self.sequence_features):
            raise ValueError("duplicate sequence feature")
        if len(set(self.global_features)) != len(self.global_features):
            raise ValueError("duplicate global feature")
        unknown_sequences = set(self.sequence_features) - set(SEQUENCE_DEFINITIONS)
        unknown_globals = set(self.global_features) - set(GLOBAL_DEFINITIONS)
        if unknown_sequences or unknown_globals:
            raise ValueError(
                f"unsupported inputs: sequences={sorted(unknown_sequences)}, "
                f"globals={sorted(unknown_globals)}"
            )
        if "text" in self.global_features and not self.text_path_key:
            raise ValueError("text_path_key is required when text is selected")
        if self.expected_text_dim <= 0:
            raise ValueError("expected_text_dim must be positive")


@dataclass(frozen=True)
class FeatureSample:
    sequences: Mapping[str, Tensor]
    globals: Mapping[str, Tensor]
    label: float
    sample_weight: float
    call_id: str
    segment_id: str
    group_id: str
    source: str
    start: float
    end: float


class FeatureValidationError(ValueError):
    pass


class FeatureDataset(Dataset[FeatureSample]):
    """Load only the branches declared by `InputSpec`.

    True empty/nonfinite inputs are rejected. Pooling code remains total for a
    manually constructed all-masked row, but Gate 0 requires zero such rows in
    real manifests.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        input_spec: InputSpec,
        feature_base_dir: str | Path,
        text_store: IndexedTextStore | None = None,
        strict: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.feature_base_dir = Path(feature_base_dir).resolve()
        self.input_spec = input_spec
        self.text_store = text_store
        self.strict = strict
        if (
            "text" in input_spec.global_features
            and text_store is not None
            and text_store.dimension != input_spec.expected_text_dim
        ):
            raise ValueError(
                f"text store dimension {text_store.dimension} does not match "
                f"declared {input_spec.expected_text_dim}"
            )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.entries = [json.loads(line) for line in handle if line.strip()]
        if not self.entries:
            raise ValueError("manifest is empty")
        segment_ids = [str(entry.get("segment_id", "")) for entry in self.entries]
        if any(not segment_id for segment_id in segment_ids):
            raise ValueError("every manifest row requires segment_id")
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("duplicate segment_id in manifest")
        self._validate_manifest_contract()

    def _resolve(self, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else self.feature_base_dir / path

    def _required_path_keys(self) -> list[str]:
        keys = [SEQUENCE_DEFINITIONS[name][0] for name in self.input_spec.sequence_features]
        for name in self.input_spec.global_features:
            if name == "text":
                if self.text_store is None:
                    keys.append(self.input_spec.text_path_key)
            else:
                key = GLOBAL_DEFINITIONS[name][0]
                assert key is not None
                keys.append(key)
        return keys

    def _validate_manifest_contract(self) -> None:
        required_keys = {
            "call_id", "segment_id", "label", "start", "end",
            *self._required_path_keys(),
        }
        for row_index, entry in enumerate(self.entries):
            missing = sorted(key for key in required_keys if key not in entry)
            if missing:
                raise ValueError(f"manifest row {row_index} missing keys: {missing}")
            if self.strict:
                for key in self._required_path_keys():
                    path = self._resolve(str(entry[key]))
                    if not path.is_file():
                        raise FileNotFoundError(f"required feature does not exist: {path}")
            if "text" in self.input_spec.global_features and self.text_store is not None:
                if str(entry["segment_id"]) not in self.text_store:
                    raise KeyError(
                        f"segment absent from text store {self.text_store.feature_set_id}: "
                        f"{entry['segment_id']}"
                    )

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _load_array(path: Path, *, expected_dim: int, sequence: bool) -> Tensor:
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32:
            raise FeatureValidationError(f"{path}: expected float32, got {array.dtype}")
        if sequence:
            if array.ndim != 2 or array.shape[1] != expected_dim:
                raise FeatureValidationError(
                    f"{path}: expected [T,{expected_dim}], got {array.shape}"
                )
            if array.shape[0] == 0:
                raise FeatureValidationError(f"{path}: empty sequence")
        else:
            if array.shape == (1, expected_dim):
                array = array[0]
            if array.shape != (expected_dim,):
                raise FeatureValidationError(
                    f"{path}: expected [{expected_dim}], got {array.shape}"
                )
        if not np.isfinite(array).all():
            raise FeatureValidationError(f"{path}: contains nonfinite values")
        return torch.from_numpy(np.ascontiguousarray(array))

    def __getitem__(self, index: int) -> FeatureSample:
        entry = self.entries[index]
        sequences = {}
        globals_ = {}
        for name in self.input_spec.sequence_features:
            path_key, dim = SEQUENCE_DEFINITIONS[name]
            sequences[name] = self._load_array(
                self._resolve(str(entry[path_key])), expected_dim=dim, sequence=True
            )
        for name in self.input_spec.global_features:
            if name == "text":
                if self.text_store is not None:
                    globals_[name] = self.text_store.get(str(entry["segment_id"]))
                    continue
                path_key = self.input_spec.text_path_key
                dim = self.input_spec.expected_text_dim
            else:
                path_key, dim = GLOBAL_DEFINITIONS[name]
                assert path_key is not None
            globals_[name] = self._load_array(
                self._resolve(str(entry[path_key])), expected_dim=dim, sequence=False
            )
        return FeatureSample(
            sequences=sequences,
            globals=globals_,
            label=float(entry["label"]),
            sample_weight=float(entry.get("sample_weight", 1.0)),
            call_id=str(entry["call_id"]),
            segment_id=str(entry["segment_id"]),
            group_id=str(entry.get("group_id", entry["call_id"])),
            source=str(entry.get("source", "unknown")),
            start=float(entry["start"]),
            end=float(entry["end"]),
        )


def _collate_sequence(values: Sequence[Tensor], expected_dim: int) -> SequenceBatch:
    if not values:
        raise ValueError("cannot collate an empty sample list")
    if any(value.ndim != 2 or value.shape[1] != expected_dim for value in values):
        raise ValueError(f"every sequence must be [T,{expected_dim}]")
    lengths = torch.tensor([value.shape[0] for value in values], dtype=torch.int64)
    padded_steps = max(1, int(lengths.max().item()))
    padded = torch.zeros((len(values), padded_steps, expected_dim), dtype=torch.float32)
    for row, value in enumerate(values):
        if value.shape[0]:
            padded[row, : value.shape[0]] = value.to(torch.float32)
    positions = torch.arange(padded_steps).unsqueeze(0)
    valid_mask = positions < lengths.unsqueeze(1)
    return SequenceBatch(
        values=padded,
        lengths=lengths,
        valid_mask=valid_mask,
        key_padding_mask=~valid_mask,
    )


def collate_samples(samples: Sequence[FeatureSample], input_spec: InputSpec) -> ExperimentBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    sequences = {
        name: _collate_sequence(
            [sample.sequences[name] for sample in samples],
            SEQUENCE_DEFINITIONS[name][1],
        )
        for name in input_spec.sequence_features
    }
    globals_ = {}
    for name in input_spec.global_features:
        expected_dim = input_spec.expected_text_dim if name == "text" else GLOBAL_DEFINITIONS[name][1]
        tensors = [sample.globals[name] for sample in samples]
        if any(tensor.shape != (expected_dim,) for tensor in tensors):
            raise ValueError(f"global feature {name!r} must be [{expected_dim}]")
        globals_[name] = torch.stack(tensors).to(torch.float32)
    return ExperimentBatch(
        sequences=sequences,
        globals=globals_,
        labels=torch.tensor([sample.label for sample in samples], dtype=torch.float32),
        sample_weights=torch.tensor(
            [sample.sample_weight for sample in samples], dtype=torch.float32
        ),
        call_ids=tuple(sample.call_id for sample in samples),
        segment_ids=tuple(sample.segment_id for sample in samples),
        group_ids=tuple(sample.group_id for sample in samples),
        sources=tuple(sample.source for sample in samples),
        starts=torch.tensor([sample.start for sample in samples], dtype=torch.float32),
        ends=torch.tensor([sample.end for sample in samples], dtype=torch.float32),
    )


def make_dataloader(
    dataset: FeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
) -> DataLoader[ExperimentBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=partial(collate_samples, input_spec=dataset.input_spec),
    )
