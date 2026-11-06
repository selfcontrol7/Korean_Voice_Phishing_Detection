from __future__ import annotations

import unittest

import torch
from torch import nn

from paper2_v2.batch import ExperimentBatch, SequenceBatch
from paper2_v2.models import (
    AudioTextCrossModel,
    GlobalEGeMAPSEncoder,
    GlobalEGeMAPSFusion,
)
from paper2_v2.pooling import SetAttentionPool, TemporalAttentionPool


def make_sequence(values: torch.Tensor) -> SequenceBatch:
    if values.ndim == 2:
        values = values.unsqueeze(0)
    batch, steps, _ = values.shape
    lengths = torch.full((batch,), steps, dtype=torch.int64)
    valid = torch.ones(batch, steps, dtype=torch.bool)
    return SequenceBatch(values, lengths, valid, ~valid)


def make_batch(audio: torch.Tensor, text: torch.Tensor) -> ExperimentBatch:
    sequence = make_sequence(audio)
    batch = sequence.batch_size
    return ExperimentBatch(
        sequences={"mfcc": sequence},
        globals={"text": text},
        labels=torch.zeros(batch),
        sample_weights=torch.ones(batch),
        call_ids=tuple(f"c{i}" for i in range(batch)),
        segment_ids=tuple(f"s{i}" for i in range(batch)),
        group_ids=tuple(f"g{i}" for i in range(batch)),
        sources=tuple("x" for _ in range(batch)),
        starts=torch.zeros(batch),
        ends=torch.ones(batch),
    )


class ArchitectureSemanticsTests(unittest.TestCase):
    def test_set_pool_is_permutation_invariant(self) -> None:
        torch.manual_seed(101)
        pool = SetAttentionPool(
            embed_dim=8, num_heads=2, query_mode="learned", dropout=0.0
        ).eval()
        values = torch.randn(1, 7, 8)
        permutation = torch.tensor([5, 0, 6, 2, 1, 4, 3])
        original = pool(make_sequence(values))
        shuffled = pool(make_sequence(values[:, permutation]))
        torch.testing.assert_close(original, shuffled, atol=1e-6, rtol=1e-6)

    def test_temporal_pool_is_order_sensitive(self) -> None:
        torch.manual_seed(103)
        pool = TemporalAttentionPool(
            embed_dim=8, num_heads=2, query_mode="learned", dropout=0.0
        ).eval()
        values = torch.randn(1, 7, 8)
        permutation = torch.tensor([5, 0, 6, 2, 1, 4, 3])
        difference = (pool(make_sequence(values)) - pool(make_sequence(values[:, permutation]))).abs()
        self.assertGreater(float(difference.max()), 1e-6)

    def test_cross_pool_responds_to_text_query(self) -> None:
        torch.manual_seed(107)
        model = AudioTextCrossModel(hidden_dim=16, num_heads=4, dropout=0.0).eval()
        audio = torch.randn(1, 5, 13)
        first = model.encode(make_batch(audio, torch.zeros(1, 768)))
        second = model.encode(make_batch(audio, torch.ones(1, 768)))
        self.assertGreater(float((first - second).abs().max()), 1e-6)

    def test_set_audio_is_unchanged_by_external_text_not_in_model(self) -> None:
        torch.manual_seed(109)
        pool = SetAttentionPool(
            embed_dim=8, num_heads=2, query_mode="learned", dropout=0.0
        ).eval()
        sequence = make_sequence(torch.randn(1, 5, 8))
        first = pool(sequence)
        _unused_text = torch.randn(1, 768)
        second = pool(sequence)
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)

    def test_global_egemaps_models_contain_no_attention(self) -> None:
        for model in (
            GlobalEGeMAPSEncoder(hidden_dim=16),
            GlobalEGeMAPSFusion(primary_dim=16, hidden_dim=16, mode="concat"),
            GlobalEGeMAPSFusion(primary_dim=16, hidden_dim=16, mode="film"),
        ):
            self.assertFalse(any(isinstance(module, nn.MultiheadAttention) for module in model.modules()))

    def test_global_egemaps_shapes_and_responsiveness(self) -> None:
        torch.manual_seed(113)
        model = GlobalEGeMAPSFusion(
            primary_dim=16, hidden_dim=16, dropout=0.0, mode="concat"
        ).eval()
        primary = torch.randn(3, 16)
        first = model(primary, torch.zeros(3, 88))
        second = model(primary, torch.ones(3, 88))
        self.assertEqual(first.shape, (3,))
        self.assertGreater(float((first - second).abs().max()), 1e-6)
        with self.assertRaises(ValueError):
            model(primary, torch.zeros(3, 1, 88))

    def test_one_key_attention_weight_is_mathematically_constant(self) -> None:
        scores = torch.tensor([[-1000.0], [0.0], [1000.0]])
        expected = torch.ones_like(scores)
        torch.testing.assert_close(torch.softmax(scores, dim=-1), expected)


if __name__ == "__main__":
    unittest.main()
