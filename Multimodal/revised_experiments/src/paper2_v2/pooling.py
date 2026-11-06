"""Mask-correct pooling primitives shared by all v2 models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .batch import SequenceBatch


class MaskedMeanPool(nn.Module):
    def forward(self, sequence: SequenceBatch) -> Tensor:
        mask = sequence.valid_mask.unsqueeze(-1).to(sequence.values.dtype)
        summed = (sequence.values * mask).sum(dim=1)
        denominator = sequence.lengths.clamp_min(1).unsqueeze(-1).to(sequence.values.dtype)
        pooled = summed / denominator
        return torch.where(sequence.all_masked.unsqueeze(-1), torch.zeros_like(pooled), pooled)


def safe_attention(
    attention: nn.MultiheadAttention,
    *,
    query: Tensor,
    sequence: SequenceBatch,
    need_weights: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Run MHA without NaNs for an explicit all-masked fallback row.

    Strict datasets reject empty sequences. This total fallback exists so an
    invalid production item cannot propagate NaNs silently: a zero sentinel is
    temporarily unmasked and the resulting representation is overwritten with
    exact zeros.
    """

    values = sequence.values
    padding_mask = sequence.key_padding_mask
    all_masked = sequence.all_masked
    if torch.any(all_masked):
        values = values.clone()
        padding_mask = padding_mask.clone()
        values[all_masked, 0] = 0
        padding_mask[all_masked, 0] = False
    output, weights = attention(
        query=query,
        key=values,
        value=values,
        key_padding_mask=padding_mask,
        need_weights=need_weights,
    )
    if torch.any(all_masked):
        output = output.masked_fill(all_masked[:, None, None], 0.0)
        if weights is not None:
            weights = weights.masked_fill(all_masked[:, None, None], 0.0)
    return output, weights


class SetAttentionPool(nn.Module):
    """Order-invariant attention pooling with an external or learned query."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        query_mode: str,
        query_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if query_mode not in {"learned", "external"}:
            raise ValueError("query_mode must be 'learned' or 'external'")
        self.embed_dim = embed_dim
        self.query_mode = query_mode
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if query_mode == "learned":
            self.learned_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.learned_query, std=0.02)
            self.query_projection = None
        else:
            if query_dim is None or query_dim <= 0:
                raise ValueError("positive query_dim required for external query")
            self.register_parameter("learned_query", None)
            self.query_projection = (
                nn.Identity() if query_dim == embed_dim else nn.Linear(query_dim, embed_dim)
            )

    def forward(self, sequence: SequenceBatch, query: Tensor | None = None) -> Tensor:
        if sequence.feature_dim != self.embed_dim:
            raise ValueError(
                f"sequence dim {sequence.feature_dim} != attention dim {self.embed_dim}"
            )
        if self.query_mode == "learned":
            if query is not None:
                raise ValueError("learned-query pool does not accept an external query")
            projected_query = self.learned_query.expand(sequence.batch_size, -1, -1)
        else:
            if query is None or query.ndim != 2 or query.shape[0] != sequence.batch_size:
                raise ValueError("external query must be [B,D]")
            assert self.query_projection is not None
            projected_query = self.query_projection(query).unsqueeze(1)
        output, _ = safe_attention(
            self.attention, query=projected_query, sequence=sequence, need_weights=False
        )
        return output.squeeze(1)


class SinusoidalPositionEncoding(nn.Module):
    """Fixed absolute positions with no trainable capacity.

    Position zero is the first real frame of each sample, independent of the
    longest sequence elsewhere in the batch. Padding positions receive zero.
    """

    def __init__(self, embed_dim: int, max_length: int = 20_000) -> None:
        super().__init__()
        if embed_dim <= 0 or max_length <= 0:
            raise ValueError("embed_dim and max_length must be positive")
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10_000.0)) / embed_dim)
        )
        encoding = torch.zeros(max_length, embed_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if embed_dim > 1:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[: encoding[:, 1::2].shape[1]]
            )
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, sequence: SequenceBatch) -> SequenceBatch:
        time_steps = sequence.values.shape[1]
        if time_steps > self.encoding.shape[0]:
            raise ValueError(
                f"sequence length {time_steps} exceeds positional limit "
                f"{self.encoding.shape[0]}"
            )
        positions = self.encoding[:time_steps].to(
            device=sequence.values.device, dtype=sequence.values.dtype
        )
        positioned = sequence.values + positions.unsqueeze(0) * sequence.valid_mask.unsqueeze(-1)
        return SequenceBatch(
            values=positioned,
            lengths=sequence.lengths,
            valid_mask=sequence.valid_mask,
            key_padding_mask=sequence.key_padding_mask,
        )


class TemporalAttentionPool(nn.Module):
    """Order-sensitive attention pooling using fixed absolute positions."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        query_mode: str,
        query_dim: int | None = None,
        dropout: float = 0.0,
        max_length: int = 20_000,
    ) -> None:
        super().__init__()
        self.positions = SinusoidalPositionEncoding(embed_dim, max_length=max_length)
        self.pool = SetAttentionPool(
            embed_dim=embed_dim,
            num_heads=num_heads,
            query_mode=query_mode,
            query_dim=query_dim,
            dropout=dropout,
        )

    @property
    def attention(self) -> nn.MultiheadAttention:
        return self.pool.attention

    def forward(self, sequence: SequenceBatch, query: Tensor | None = None) -> Tensor:
        return self.pool(self.positions(sequence), query)
