from __future__ import annotations

import unittest

import torch

from paper2_v2.batch import ExperimentBatch, SequenceBatch
from paper2_v2.conditions import (
    CORE_CONDITIONS,
    ConditionId,
    PairingMode,
    build_condition,
)


def make_batch(*, audio: bool, text: bool, batch_size: int = 3) -> ExperimentBatch:
    sequences = {}
    if audio:
        values = torch.randn(batch_size, 5, 13)
        lengths = torch.tensor([5, 3, 4], dtype=torch.int64)[:batch_size]
        valid = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
        sequences["mfcc"] = SequenceBatch(values, lengths, valid, ~valid)
    globals_ = {"text": torch.randn(batch_size, 768)} if text else {}
    return ExperimentBatch(
        sequences=sequences,
        globals=globals_,
        labels=torch.tensor([0.0, 1.0, 0.0])[:batch_size],
        sample_weights=torch.ones(batch_size),
        call_ids=tuple(f"call_{index}" for index in range(batch_size)),
        segment_ids=tuple(f"segment_{index}" for index in range(batch_size)),
        group_ids=tuple(f"group_{index}" for index in range(batch_size)),
        sources=tuple("source" for _ in range(batch_size)),
        starts=torch.zeros(batch_size),
        ends=torch.ones(batch_size),
    )


class Work8ConditionFactoryTests(unittest.TestCase):
    def test_exact_locked_condition_order_and_forward_contracts(self) -> None:
        self.assertEqual(
            CORE_CONDITIONS,
            (
                "T", "A-mean", "A-query", "A-temp", "AT-concat", "AT-gate",
                "AT-cross", "AT-cross-randtrain",
            ),
        )
        torch.manual_seed(211)
        for condition_id in CORE_CONDITIONS:
            with self.subTest(condition=condition_id):
                built = build_condition(
                    condition_id, hidden_dim=16, num_heads=4, dropout=0.0
                )
                definition = built.definition
                batch = make_batch(
                    audio=definition.requires_mfcc,
                    text=definition.requires_text,
                )
                logits = built.model.eval()(batch)
                self.assertEqual(logits.shape, (3,))
                self.assertGreater(built.total_parameters, 0)
                self.assertEqual(built.total_parameters, built.trainable_parameters)
                self.assertEqual(
                    built.input_spec.sequence_features,
                    ("mfcc",) if definition.requires_mfcc else (),
                )
                self.assertEqual(
                    built.input_spec.global_features,
                    ("text",) if definition.requires_text else (),
                )

    def test_random_train_control_is_architecture_identical(self) -> None:
        torch.manual_seed(223)
        paired = build_condition("AT-cross", hidden_dim=16, num_heads=4)
        torch.manual_seed(227)
        randomized = build_condition("AT-cross-randtrain", hidden_dim=16, num_heads=4)
        self.assertEqual(
            list(paired.model.state_dict()), list(randomized.model.state_dict())
        )
        self.assertEqual(paired.total_parameters, randomized.total_parameters)
        self.assertEqual(paired.architecture["mechanism"], randomized.architecture["mechanism"])
        self.assertEqual(paired.definition.pairing_mode, PairingMode.CORRECT)
        self.assertEqual(
            randomized.definition.pairing_mode, PairingMode.RANDOM_TRAIN_ONLY
        )

    def test_only_text_condition_may_have_zero_audio_inputs(self) -> None:
        text = build_condition(ConditionId.TEXT)
        self.assertEqual(text.input_spec.sequence_features, ())
        for condition_id in CORE_CONDITIONS[1:]:
            self.assertEqual(
                build_condition(condition_id).input_spec.sequence_features,
                ("mfcc",),
            )

    def test_unknown_condition_and_implicit_text_concat_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_condition("AT-not-preregistered")
        with self.assertRaises(ValueError):
            build_condition("AT-cross", text_dim=1536)
        with self.assertRaises(ValueError):
            build_condition("AT-cross", hidden_dim=15, num_heads=4)


if __name__ == "__main__":
    unittest.main()

