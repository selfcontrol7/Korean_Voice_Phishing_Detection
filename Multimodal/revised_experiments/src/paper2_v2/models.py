"""Truthfully named Paper 2 v2 model components.

Set models contain no positional information and are order-invariant. Temporal
models add an explicit absolute position before otherwise matched attention.
Stored eGeMAPS functionals are always global vectors and never attention keys.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .batch import ExperimentBatch, SequenceBatch
from .pooling import MaskedMeanPool, SetAttentionPool, TemporalAttentionPool


class FrameProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: SequenceBatch) -> SequenceBatch:
        return SequenceBatch(
            values=self.network(sequence.values),
            lengths=sequence.lengths,
            valid_mask=sequence.valid_mask,
            key_padding_mask=sequence.key_padding_mask,
        )


class GlobalProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2:
            raise ValueError("global inputs must be [B,D]")
        return self.network(values)


def binary_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class TextOnlyModel(nn.Module):
    def __init__(self, text_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.text = GlobalProjector(text_dim, hidden_dim, dropout)
        self.head = binary_head(hidden_dim, hidden_dim, dropout)

    def forward(self, batch: ExperimentBatch) -> Tensor:
        return self.head(self.text(batch.globals["text"])).squeeze(-1)


class AcousticEncoder(nn.Module):
    """Matched MFCC encoder used by every audio-bearing core condition."""

    def __init__(
        self,
        *,
        input_dim: int = 13,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        pooling: Literal["mean", "set", "temporal"] = "set",
    ) -> None:
        super().__init__()
        self.projector = FrameProjector(input_dim, hidden_dim, dropout)
        self.pooling_name = pooling
        if pooling == "mean":
            self.pool = MaskedMeanPool()
        elif pooling == "set":
            self.pool = SetAttentionPool(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                query_mode="learned",
                dropout=dropout,
            )
        elif pooling == "temporal":
            self.pool = TemporalAttentionPool(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                query_mode="learned",
                dropout=dropout,
            )
        else:
            raise ValueError(f"unsupported pooling: {pooling}")

    def forward(self, sequence: SequenceBatch) -> Tensor:
        projected = self.projector(sequence)
        return self.pool(projected)


class AudioOnlyModel(nn.Module):
    def __init__(
        self,
        *,
        pooling: Literal["mean", "set", "temporal"],
        input_dim: int = 13,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.audio = AcousticEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            pooling=pooling,
        )
        self.head = binary_head(hidden_dim, hidden_dim, dropout)

    def forward(self, batch: ExperimentBatch) -> Tensor:
        return self.head(self.audio(batch.sequences["mfcc"])).squeeze(-1)


class AudioTextConcatModel(nn.Module):
    """Standard late fusion: learned-query acoustic set pool plus text."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.audio = AcousticEncoder(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, pooling="set"
        )
        self.text = GlobalProjector(768, hidden_dim, dropout)
        self.head = binary_head(2 * hidden_dim, hidden_dim, dropout)

    def forward(self, batch: ExperimentBatch) -> Tensor:
        fused = torch.cat(
            [self.audio(batch.sequences["mfcc"]), self.text(batch.globals["text"])], dim=-1
        )
        return self.head(fused).squeeze(-1)


class AudioTextGateModel(nn.Module):
    """Text-conditioned FiLM gate over a learned-query acoustic set pool."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.audio = AcousticEncoder(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, pooling="set"
        )
        self.text = GlobalProjector(768, hidden_dim, dropout)
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.head = binary_head(hidden_dim, hidden_dim, dropout)

    def forward(self, batch: ExperimentBatch) -> Tensor:
        audio = self.audio(batch.sequences["mfcc"])
        gamma, beta = self.film(self.text(batch.globals["text"])).chunk(2, dim=-1)
        fused = (1.0 + torch.tanh(gamma)) * audio + beta
        return self.head(fused).squeeze(-1)


class AudioTextCrossModel(nn.Module):
    """Sentence-conditioned acoustic pooling, with no direct text-to-head path."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        temporal: bool = False,
    ) -> None:
        super().__init__()
        self.audio_projector = FrameProjector(13, hidden_dim, dropout)
        self.text_projector = GlobalProjector(768, hidden_dim, dropout)
        pool_type = TemporalAttentionPool if temporal else SetAttentionPool
        self.pool = pool_type(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            query_mode="external",
            query_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal = temporal
        self.head = binary_head(hidden_dim, hidden_dim, dropout)

    def encode(self, batch: ExperimentBatch) -> Tensor:
        sequence = self.audio_projector(batch.sequences["mfcc"])
        query = self.text_projector(batch.globals["text"])
        return self.pool(sequence, query)

    def forward(self, batch: ExperimentBatch) -> Tensor:
        return self.head(self.encode(batch)).squeeze(-1)


class GlobalEGeMAPSEncoder(nn.Module):
    """Encoder for the stored 88 global functionals; intentionally no attention."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.projector = GlobalProjector(88, hidden_dim, dropout)

    def forward(self, egemaps: Tensor) -> Tensor:
        if egemaps.ndim != 2 or egemaps.shape[1] != 88:
            raise ValueError("eGeMAPS functionals must be [B,88]")
        return self.projector(egemaps)


class GlobalEGeMAPSFusion(nn.Module):
    """Explicit concat or FiLM fusion for global eGeMAPS functionals."""

    def __init__(
        self,
        *,
        primary_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        mode: Literal["concat", "film"] = "concat",
    ) -> None:
        super().__init__()
        if mode not in {"concat", "film"}:
            raise ValueError("mode must be concat or film")
        self.mode = mode
        self.primary = GlobalProjector(primary_dim, hidden_dim, dropout)
        self.egemaps = GlobalEGeMAPSEncoder(hidden_dim, dropout)
        if mode == "concat":
            self.combiner = binary_head(2 * hidden_dim, hidden_dim, dropout)
        else:
            self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.combiner = binary_head(hidden_dim, hidden_dim, dropout)

    def forward(self, primary: Tensor, egemaps: Tensor) -> Tensor:
        primary_hidden = self.primary(primary)
        egemaps_hidden = self.egemaps(egemaps)
        if self.mode == "concat":
            fused = torch.cat([primary_hidden, egemaps_hidden], dim=-1)
        else:
            gamma, beta = self.film(egemaps_hidden).chunk(2, dim=-1)
            fused = (1.0 + torch.tanh(gamma)) * primary_hidden + beta
        return self.combiner(fused).squeeze(-1)
