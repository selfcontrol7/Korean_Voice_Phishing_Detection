"""Authoritative, validated batch types for Paper 2 v2."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class SequenceBatch:
    """One padded sequence branch with explicit, unambiguous masking.

    `valid_mask=True` means real input. `key_padding_mask=True` means ignore the
    position in PyTorch attention. They must be exact inverses.
    """

    values: Tensor
    lengths: Tensor
    valid_mask: Tensor
    key_padding_mask: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"values must be [B,T,D], got {tuple(self.values.shape)}")
        batch_size, time_steps, _ = self.values.shape
        if not self.values.is_floating_point():
            raise TypeError("sequence values must be floating point")
        if self.lengths.shape != (batch_size,) or self.lengths.dtype != torch.int64:
            raise TypeError("lengths must be int64 [B]")
        expected_mask_shape = (batch_size, time_steps)
        for name, mask in (
            ("valid_mask", self.valid_mask),
            ("key_padding_mask", self.key_padding_mask),
        ):
            if mask.shape != expected_mask_shape or mask.dtype != torch.bool:
                raise TypeError(f"{name} must be bool [B,T]")
        if torch.any(self.lengths < 0) or torch.any(self.lengths > time_steps):
            raise ValueError("lengths must satisfy 0 <= length <= padded T")
        if not torch.equal(self.valid_mask, ~self.key_padding_mask):
            raise ValueError("valid_mask must be the inverse of key_padding_mask")
        if not torch.equal(self.valid_mask.sum(dim=1).to(torch.int64), self.lengths):
            raise ValueError("lengths must equal the number of valid positions")

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.values.shape[2])

    @property
    def all_masked(self) -> Tensor:
        return self.lengths == 0

    def to(self, device: torch.device | str) -> "SequenceBatch":
        return SequenceBatch(
            values=self.values.to(device),
            lengths=self.lengths.to(device),
            valid_mask=self.valid_mask.to(device),
            key_padding_mask=self.key_padding_mask.to(device),
        )


@dataclass(frozen=True)
class ExperimentBatch:
    """All inputs and identifiers required for training and item-level output."""

    sequences: Mapping[str, SequenceBatch]
    globals: Mapping[str, Tensor]
    labels: Tensor
    sample_weights: Tensor
    call_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    sources: tuple[str, ...]
    starts: Tensor
    ends: Tensor

    def __post_init__(self) -> None:
        batch_size = int(self.labels.shape[0]) if self.labels.ndim == 1 else -1
        if batch_size < 0 or self.labels.dtype != torch.float32:
            raise TypeError("labels must be float32 [B]")
        if self.sample_weights.shape != (batch_size,) or self.sample_weights.dtype != torch.float32:
            raise TypeError("sample_weights must be float32 [B]")
        if self.starts.shape != (batch_size,) or self.ends.shape != (batch_size,):
            raise ValueError("starts and ends must be [B]")
        metadata = (self.call_ids, self.segment_ids, self.group_ids, self.sources)
        if any(len(values) != batch_size for values in metadata):
            raise ValueError("metadata tuple lengths must equal batch size")
        if any(sequence.batch_size != batch_size for sequence in self.sequences.values()):
            raise ValueError("all sequence branches must share batch size")
        for name, values in self.globals.items():
            if values.ndim != 2 or values.shape[0] != batch_size:
                raise ValueError(f"global branch {name!r} must be [B,D]")
            if not values.is_floating_point():
                raise TypeError(f"global branch {name!r} must be floating point")
        object.__setattr__(self, "sequences", MappingProxyType(dict(self.sequences)))
        object.__setattr__(self, "globals", MappingProxyType(dict(self.globals)))

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    def to(self, device: torch.device | str) -> "ExperimentBatch":
        return ExperimentBatch(
            sequences={name: value.to(device) for name, value in self.sequences.items()},
            globals={name: value.to(device) for name, value in self.globals.items()},
            labels=self.labels.to(device),
            sample_weights=self.sample_weights.to(device),
            call_ids=self.call_ids,
            segment_ids=self.segment_ids,
            group_ids=self.group_ids,
            sources=self.sources,
            starts=self.starts.to(device),
            ends=self.ends.to(device),
        )
