from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from paper2_v2.config import ArtifactRef, SelectionPolicy
from paper2_v2.evaluation import (
    EpochValidation,
    SegmentPrediction,
    aggregate_mean_probability,
    call_level_nll,
    compute_call_metrics,
    select_checkpoint,
    select_validation_threshold,
    threshold_grid,
)


def selection_policy() -> SelectionPolicy:
    return SelectionPolicy(
        independent_unit="call",
        primary_metric="macro_f1",
        aggregation_rule="mean_probability",
        checkpoint_monitor="validation_call_nll",
        threshold_start=0.01,
        threshold_end=0.99,
        threshold_step=0.01,
        threshold_tie_break=(
            "lower_fpr", "higher_sensitivity", "closest_to_0.5", "higher_threshold"
        ),
        calibration_rule="none",
        outer_evaluation_once=True,
    )


def segment(
    call: str, index: int, label: int, probability: float, *, group: str | None = None
) -> SegmentPrediction:
    return SegmentPrediction(
        segment_id=f"{call}_{index}",
        call_id=call,
        group_id=group or f"group_{call}",
        label=label,
        probability=probability,
        start=float(index),
        end=float(index + 1),
        source="source",
    )


class Work8EvaluationTests(unittest.TestCase):
    def test_mean_probability_aggregation_and_call_metrics(self) -> None:
        calls = aggregate_mean_probability(
            [
                segment("c0", 0, 0, 0.1), segment("c0", 1, 0, 0.3),
                segment("c1", 0, 0, 0.6), segment("c1", 1, 0, 0.8),
                segment("c2", 0, 1, 0.4), segment("c2", 1, 1, 0.6),
                segment("c3", 0, 1, 0.8), segment("c3", 1, 1, 1.0),
            ]
        )
        self.assertEqual([call.call_id for call in calls], ["c0", "c1", "c2", "c3"])
        self.assertEqual([call.probability for call in calls], [0.2, 0.7, 0.5, 0.9])
        metrics = compute_call_metrics(calls, threshold=0.5)
        self.assertEqual(
            [metrics[key] for key in ("tp", "tn", "fp", "fn")], [2, 1, 1, 0]
        )
        self.assertAlmostEqual(float(metrics["positive_f1"]), 0.8)
        self.assertAlmostEqual(float(metrics["negative_f1"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(metrics["macro_f1"]), 11.0 / 15.0)
        self.assertGreater(call_level_nll(calls), 0.0)

    def test_aggregation_rejects_duplicate_or_inconsistent_call_rows(self) -> None:
        duplicated = segment("call", 0, 0, 0.1)
        with self.assertRaisesRegex(ValueError, "duplicate segment"):
            aggregate_mean_probability([duplicated, duplicated])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            aggregate_mean_probability(
                [segment("call", 0, 0, 0.1), segment("call", 1, 1, 0.9)]
            )

    def test_threshold_grid_and_deterministic_closest_half_tie_break(self) -> None:
        policy = selection_policy()
        grid = threshold_grid(policy)
        self.assertEqual(len(grid), 99)
        self.assertEqual((grid[0], grid[-1]), (0.01, 0.99))
        calls = aggregate_mean_probability(
            [segment("negative", 0, 0, 0.2), segment("positive", 0, 1, 0.8)]
        )
        selected = select_validation_threshold(calls, policy)
        self.assertEqual(selected.threshold, 0.5)
        self.assertEqual(selected.metrics["macro_f1"], 1.0)
        self.assertEqual(selected.evaluated_thresholds, 99)

    def test_checkpoint_selection_is_validation_nll_then_earlier_epoch(self) -> None:
        prediction_hash = hashlib.sha256(b"validation predictions").hexdigest()
        checkpoint_hash = hashlib.sha256(b"checkpoint").hexdigest()
        artifact = ArtifactRef(Path("/tmp/not-opened-checkpoint.pt"), checkpoint_hash)
        epochs = [
            EpochValidation(1, 0.4, artifact, prediction_hash),
            EpochValidation(2, 0.2, artifact, prediction_hash),
            EpochValidation(3, 0.2, artifact, prediction_hash),
        ]
        selected = select_checkpoint(epochs)
        self.assertEqual(selected.epoch, 2)
        with self.assertRaisesRegex(ValueError, "duplicate validation epoch"):
            select_checkpoint([epochs[0], epochs[0]])


if __name__ == "__main__":
    unittest.main()

