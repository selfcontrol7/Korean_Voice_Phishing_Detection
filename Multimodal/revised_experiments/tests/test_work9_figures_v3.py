from __future__ import annotations

import json
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from paper2_v2 import work9_figures as v2
from paper2_v2.work9_figures_v3 import (
    CONDITIONS,
    DIAGNOSTIC_LABEL,
    FIGURE_SPEC_RELATIVE,
    render_gpu_memory_distribution,
    render_parameter_counts,
    render_protocol_pipeline,
    render_runtime_distribution,
    render_threshold_response,
    load_phase_a_bundle,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


class Work9FigureV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_phase_a_bundle(ROOT)

    def test_v2_source_and_script_remain_at_manifest_bound_hashes(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "revised_experiments/artifacts/work9_phase_a_figures_v2/figure_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for reference in manifest["sources"]:
            self.assertEqual(sha256_file(ROOT / reference["path"]), reference["sha256"])
        self.assertEqual(
            sha256_file(
                ROOT
                / "revised_experiments/artifacts/work9_phase_a_figures_v2/figure_manifest.json"
            ),
            "e812861dd2827d3c659f32b9c6c7c5b760b8657561e689f73067edfd3f19d454",
        )

    def test_v3_spec_fixes_axis_and_efficiency_boundaries(self) -> None:
        spec = json.loads((ROOT / FIGURE_SPEC_RELATIVE).read_text(encoding="utf-8"))
        self.assertEqual(spec["bundle_version"], "work9_phase_a_figures_v3")
        self.assertIn("panel_specific", spec["aggregation_rules"]["threshold_response_axis_policy"])
        self.assertFalse(
            spec["aggregation_rules"][
                "cross_panel_threshold_response_vertical_displacement_comparison_permitted"
            ]
        )
        self.assertFalse(spec["efficiency_scope"]["end_to_end_deployment_claim_enabled"])
        self.assertIn(
            "frozen_precomputed_KR_SBERT_encoder",
            spec["efficiency_scope"]["parameter_counts_exclude"],
        )

    def test_threshold_response_has_no_clipped_median_or_iqr(self) -> None:
        figure = render_threshold_response(self.bundle)
        try:
            axes = figure.axes
            self.assertEqual(len(axes), 8)
            self.assertFalse(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
            lower_limits = []
            for axis in axes:
                line = next(line for line in axis.lines if len(line.get_xdata()) == 99)
                lower, upper = axis.get_ylim()
                lower_limits.append(lower)
                self.assertGreaterEqual(float(min(line.get_ydata())), lower - 1e-12)
                self.assertLessEqual(float(max(line.get_ydata())), upper + 1e-12)
                self.assertTrue(
                    any("panel y-axis:" in text.get_text() for text in axis.texts)
                )
                collections = axis.collections
                band = next(collection for collection in collections if hasattr(collection, "get_paths"))
                vertices = band.get_paths()[0].vertices
                self.assertGreaterEqual(float(vertices[:, 1].min()), lower - 1e-12)
                self.assertLessEqual(float(vertices[:, 1].max()), upper + 1e-12)
            self.assertLess(lower_limits[0], 0.45)
            self.assertGreater(max(lower_limits) - min(lower_limits), 0.4)
            caption = " ".join(text.get_text() for text in figure.texts)
            self.assertIn("lower limits vary", caption)
            self.assertIn(DIAGNOSTIC_LABEL, caption)
        finally:
            plt.close(figure)

    def test_protocol_uses_single_validation_count(self) -> None:
        figure = render_protocol_pipeline(self.bundle.config, self.bundle.fold_protocol)
        try:
            text = " ".join(item.get_text() for axis in figure.axes for item in axis.texts)
            self.assertIn("validation 190 calls", text)
            self.assertNotIn("validation 190–190 calls", text)
        finally:
            plt.close(figure)

    def test_parameter_figure_is_downstream_trainable_scope(self) -> None:
        figure = render_parameter_counts(self.bundle)
        try:
            axis = figure.axes[0]
            self.assertIn("downstream", axis.get_title().lower())
            self.assertIn("Trainable downstream", axis.get_ylabel())
            caption = " ".join(text.get_text() for text in figure.texts)
            self.assertIn("frozen precomputed KR-SBERT", caption)
            self.assertIn("ASR", caption)
            self.assertIn("mobile", caption)
        finally:
            plt.close(figure)

    def test_runtime_figure_states_recorded_epoch_scope(self) -> None:
        figure = render_runtime_distribution(self.bundle)
        try:
            axis = figure.axes[0]
            self.assertIn("training and validation epoch", axis.get_title())
            self.assertIn("recorded epoch wall time", axis.get_ylabel())
            caption = " ".join(text.get_text() for text in figure.texts)
            for phrase in ("precomputed features", "ASR/KR-SBERT", "phone/mobile latency"):
                self.assertIn(phrase, caption)
        finally:
            plt.close(figure)

    def test_gpu_figure_states_downstream_precomputed_scope(self) -> None:
        figure = render_gpu_memory_distribution(self.bundle)
        try:
            self.assertTrue(
                all("downstream CUDA" in axis.get_ylabel() for axis in figure.axes)
            )
            caption = " ".join(text.get_text() for text in figure.texts)
            for phrase in ("precomputed features", "ASR", "KR-SBERT", "phone/mobile"):
                self.assertIn(phrase, caption)
        finally:
            plt.close(figure)

    def test_exact_200_run_input_grid_remains_unchanged(self) -> None:
        self.assertEqual(len(self.bundle.runs), 200)
        self.assertEqual(
            {
                condition: sum(run.condition == condition for run in self.bundle.runs)
                for condition in CONDITIONS
            },
            {condition: 25 for condition in CONDITIONS},
        )
        self.assertFalse(
            any(
                "outer_eval" in reference.serialized_path
                for run in self.bundle.runs
                for reference in (
                    run.epoch_trace_ref,
                    run.threshold_trace_ref,
                    run.run_record_ref,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
