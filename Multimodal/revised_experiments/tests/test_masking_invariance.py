from __future__ import annotations

import unittest

import torch
from torch import nn

from paper2_v2.batch import SequenceBatch
from paper2_v2.pooling import MaskedMeanPool, SetAttentionPool, safe_attention


def make_sequence(rows: list[torch.Tensor], padded_steps: int | None = None) -> SequenceBatch:
    dim = rows[0].shape[1]
    lengths = torch.tensor([row.shape[0] for row in rows], dtype=torch.int64)
    time_steps = padded_steps or max(1, int(lengths.max()))
    values = torch.zeros(len(rows), time_steps, dim)
    for index, row in enumerate(rows):
        values[index, : row.shape[0]] = row
    valid = torch.arange(time_steps).unsqueeze(0) < lengths.unsqueeze(1)
    return SequenceBatch(values, lengths, valid, ~valid)


class MaskingInvarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.pool = SetAttentionPool(
            embed_dim=4, num_heads=2, query_mode="external", query_dim=4
        ).eval()
        self.query = torch.tensor([[0.2, -0.4, 0.6, 0.8]])
        self.sample = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6], [0.3, 0.1, 0.5, 0.7]]
        )

    def test_batch_invariance_with_short_and_long_companions(self) -> None:
        alone = self.pool(make_sequence([self.sample]), self.query)[0]
        short = torch.tensor([[0.5, 0.4, 0.3, 0.2]])
        long = torch.randn(11, 4)
        mixed_short = self.pool(
            make_sequence([self.sample, short]), torch.cat([self.query, self.query])
        )[0]
        mixed_long = self.pool(
            make_sequence([self.sample, long]), torch.cat([self.query, self.query])
        )[0]
        torch.testing.assert_close(alone, mixed_short, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alone, mixed_long, atol=1e-5, rtol=1e-5)

    def test_arbitrary_padded_values_do_not_change_output(self) -> None:
        baseline = make_sequence([self.sample], padded_steps=8)
        altered_values = baseline.values.clone()
        altered_values[baseline.key_padding_mask] = 1_000_000.0
        altered = SequenceBatch(
            altered_values,
            baseline.lengths,
            baseline.valid_mask,
            baseline.key_padding_mask,
        )
        torch.testing.assert_close(
            self.pool(baseline, self.query),
            self.pool(altered, self.query),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_hard_predictions_are_batch_size_invariant(self) -> None:
        rows = [torch.randn((index % 6) + 1, 4) for index in range(64)]
        queries = torch.randn(64, 4)
        head = nn.Linear(4, 1).eval()
        expected = []
        for row, query in zip(rows, queries):
            expected.append((head(self.pool(make_sequence([row]), query[None])).item() >= 0))
        for batch_size in (1, 16, 64):
            actual = []
            for start in range(0, len(rows), batch_size):
                sequence = make_sequence(rows[start : start + batch_size])
                logits = head(self.pool(sequence, queries[start : start + batch_size])).squeeze(1)
                actual.extend((logits >= 0).tolist())
            self.assertEqual(actual, expected)

    def test_mask_polarity_hand_computed_attention(self) -> None:
        attention = nn.MultiheadAttention(1, 1, bias=True, batch_first=True).eval()
        with torch.no_grad():
            attention.in_proj_weight.fill_(1.0)
            attention.in_proj_bias.zero_()
            attention.out_proj.weight.fill_(1.0)
            attention.out_proj.bias.zero_()
        sequence = SequenceBatch(
            values=torch.tensor([[[2.0], [100.0]]]),
            lengths=torch.tensor([1]),
            valid_mask=torch.tensor([[True, False]]),
            key_padding_mask=torch.tensor([[False, True]]),
        )
        output, _ = safe_attention(
            attention, query=torch.zeros(1, 1, 1), sequence=sequence
        )
        torch.testing.assert_close(output, torch.tensor([[[2.0]]]))

    def test_all_masked_fallback_is_finite_exact_zero(self) -> None:
        sequence = SequenceBatch(
            values=torch.full((2, 1, 4), 99.0),
            lengths=torch.tensor([0, 0]),
            valid_mask=torch.zeros(2, 1, dtype=torch.bool),
            key_padding_mask=torch.ones(2, 1, dtype=torch.bool),
        )
        output = self.pool(sequence, torch.randn(2, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        mean_output = MaskedMeanPool()(sequence)
        self.assertTrue(torch.equal(mean_output, torch.zeros_like(mean_output)))


if __name__ == "__main__":
    unittest.main()
