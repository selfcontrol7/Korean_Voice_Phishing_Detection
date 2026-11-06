from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from paper2_v2.work9_figures import (
    CONDITIONS,
    DIAGNOSTIC_LABEL,
    FIGURE_SPEC_RELATIVE,
    FORMATS,
    SELECTION_INDEX_RELATIVE,
    FigurePipelineError,
    _expected_figure_ids,
    _resolve_artifact_reference,
    _save_figure,
    _validate_coordinate_grid,
    load_phase_a_bundle,
    pointwise_observed_summary,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


class Work9FigureTests(unittest.TestCase):
    def test_pointwise_summary_never_carries_stopped_runs_forward(self) -> None:
        summary = pointwise_observed_summary(
            [
                [(1, 0.8), (2, 0.4), (3, 0.2)],
                [(1, 1.0)],
                [(1, 0.9), (2, 0.5)],
            ]
        )
        self.assertEqual(summary["epoch"].tolist(), [1, 2, 3])
        self.assertEqual(summary["n"].tolist(), [3, 2, 1])
        self.assertAlmostEqual(float(summary["median"][1]), 0.45)
        self.assertAlmostEqual(float(summary["median"][2]), 0.2)

    def test_phase_a_requires_frozen_selection_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FigurePipelineError, "frozen 200-coordinate"):
                load_phase_a_bundle(temporary)

    def test_outer_named_reference_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = {
                "path": "artifacts/outer_eval_predictions.jsonl",
                "sha256": "0" * 64,
                "bytes": 1,
                "row_count": 1,
            }
            with self.assertRaisesRegex(FigurePipelineError, "outer artifact"):
                _resolve_artifact_reference(raw, root, context="malicious test ref")

    def test_coordinate_grid_rejects_duplicate_and_incomplete_rows(self) -> None:
        rows = [
            {"condition": condition, "outer_fold": fold, "seed": seed}
            for condition in CONDITIONS
            for fold in range(5)
            for seed in (17, 29, 43, 71, 101)
        ]
        _validate_coordinate_grid(rows)
        corrupted = list(rows)
        corrupted[-1] = dict(corrupted[0])
        with self.assertRaisesRegex(FigurePipelineError, "duplicate"):
            _validate_coordinate_grid(corrupted)

    def test_pre_outer_spec_binds_current_selection_index_and_rules(self) -> None:
        spec = json.loads((ROOT / FIGURE_SPEC_RELATIVE).read_text(encoding="utf-8"))
        index_path = ROOT / SELECTION_INDEX_RELATIVE
        self.assertEqual(spec["selection_index"]["sha256"], sha256_file(index_path))
        self.assertFalse(spec["aggregation_rules"]["cherry_picking_permitted"])
        self.assertFalse(spec["aggregation_rules"]["last_observation_carried_forward"])
        self.assertTrue(
            spec["aggregation_rules"]["all_25_coordinates_per_condition_required"]
        )
        self.assertEqual(
            spec["diagnostic_boundary"]["phase_a_label"], DIAGNOSTIC_LABEL
        )
        self.assertEqual(
            [row["figure_id"] for row in spec["figures"]],
            list(_expected_figure_ids("all")),
        )

    def test_actual_phase_a_bundle_is_exact_and_outer_safe(self) -> None:
        bundle = load_phase_a_bundle(ROOT)
        self.assertEqual(len(bundle.runs), 200)
        self.assertEqual(
            {condition: sum(run.condition == condition for run in bundle.runs) for condition in CONDITIONS},
            {condition: 25 for condition in CONDITIONS},
        )
        self.assertTrue(all(run.completed_epochs >= run.selected_epoch for run in bundle.runs))
        referenced = [
            reference.serialized_path
            for run in bundle.runs
            for reference in (
                run.epoch_trace_ref,
                run.threshold_trace_ref,
                run.run_record_ref,
            )
        ]
        self.assertFalse(any("outer_eval" in path for path in referenced))

    def test_export_supports_all_declared_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fig, ax = plt.subplots(figsize=(2, 1.5))
            ax.plot([0, 1], [0, 1])
            outputs = _save_figure(fig, Path(temporary) / "smoke", FORMATS)
            self.assertEqual({path.suffix for path in outputs}, {".svg", ".pdf", ".png"})
            self.assertTrue(all(path.stat().st_size > 100 for path in outputs))


if __name__ == "__main__":
    unittest.main()
