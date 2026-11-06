from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper2_v2.work9_capacity_analysis import FIGURE_IDS, load_frozen_capacity_policies
from paper2_v2.work9_capacity_figures import (
    CapacityFigureError,
    build_capacity_figures,
)
from paper2_v2.work9_capacity_figures_v2 import (
    AFFECTED_FIGURE_IDS,
    EXPECTED_ZOOM_LIMITS,
    FIGURE_OUTPUT_RELATIVE,
    PRESENTATION_CORRECTION_SHA256,
    PRODUCTION_ANALYSIS_REPORT_SHA256,
    build_capacity_figures_v2,
    load_presentation_correction,
    render_capacity_figures_from_report_v2,
    render_capacity_figures_v2,
    verify_presentation_correction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_analysis_v1/report.json"
)
V1_MANIFEST_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v1/figure_manifest.json"
)
V1_HASHES_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v1/hashes.sha256"
)


class CapacityFigureV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((PROJECT_ROOT / REPORT_RELATIVE).read_text())
        cls.policies = load_frozen_capacity_policies(PROJECT_ROOT)
        cls.correction = load_presentation_correction(PROJECT_ROOT)
        cls.v1_hashes = {
            "manifest": hashlib.sha256(
                (PROJECT_ROOT / V1_MANIFEST_RELATIVE).read_bytes()
            ).hexdigest(),
            "hashes": hashlib.sha256(
                (PROJECT_ROOT / V1_HASHES_RELATIVE).read_bytes()
            ).hexdigest(),
        }

    def test_frozen_presentation_correction_preserves_science_and_v1(self) -> None:
        path = (
            PROJECT_ROOT
            / "revised_experiments/configs/"
            "work9_capacity_figure_presentation_correction_v2.json"
        )
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), PRESENTATION_CORRECTION_SHA256)
        verify_presentation_correction(self.correction, project_root=PROJECT_ROOT)
        self.assertEqual(tuple(self.correction["figure_ids"]), FIGURE_IDS)
        self.assertEqual(tuple(self.correction["affected_figure_ids"]), AFFECTED_FIGURE_IDS)
        self.assertEqual(
            self.correction["analysis_report"]["sha256"],
            PRODUCTION_ANALYSIS_REPORT_SHA256,
        )
        self.assertTrue(
            all(value is False for value in self.correction["preservation"].values())
        )
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / V1_MANIFEST_RELATIVE).read_bytes()).hexdigest(),
            self.v1_hashes["manifest"],
        )
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / V1_HASHES_RELATIVE).read_bytes()).hexdigest(),
            self.v1_hashes["hashes"],
        )

    def test_correction_mutations_fail_closed(self) -> None:
        changed = copy.deepcopy(self.correction)
        changed["preservation"]["analysis_report_changes_allowed"] = True
        with self.assertRaisesRegex(CapacityFigureError, "scientific preservation"):
            verify_presentation_correction(changed)

        changed = copy.deepcopy(self.correction)
        changed["figure_ids"][-1] = "post_hoc_best_condition"
        with self.assertRaisesRegex(CapacityFigureError, "identity/inventory"):
            verify_presentation_correction(changed)

        changed = copy.deepcopy(self.correction)
        changed["presentation_contract"]["capacity_metric_profile"][
            "bounded_metric_zoom_ylim"
        ]["macro_f1"] = [0.999, 1.0]
        with self.assertRaisesRegex(CapacityFigureError, "zoom changed"):
            verify_presentation_correction(changed)

    def test_exact_seven_ids_and_only_two_presentation_revisions(self) -> None:
        figures = build_capacity_figures_v2(
            self.report, self.policies.figure_spec, self.correction
        )
        try:
            self.assertEqual(tuple(figures), FIGURE_IDS)
            self.assertEqual(set(AFFECTED_FIGURE_IDS), {
                "capacity_condition_macro_f1",
                "capacity_metric_profile",
            })
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_zoom_ranges_full_scale_context_and_disclosures_are_exact(self) -> None:
        figures = build_capacity_figures_v2(
            self.report, self.policies.figure_spec, self.correction
        )
        try:
            macro = figures["capacity_condition_macro_f1"]
            self.assertEqual(len(macro.axes), 2)
            self.assertEqual(tuple(macro.axes[0].get_ylim()), EXPECTED_ZOOM_LIMITS["macro_f1"])
            self.assertEqual(tuple(macro.axes[1].get_ylim()), (0.0, 1.001))
            macro_text = " ".join(text.get_text() for text in macro.texts)
            self.assertIn("not confidence intervals", macro_text)
            self.assertIn("full-scale context", macro_text)

            profile = figures["capacity_metric_profile"]
            self.assertEqual(len(profile.axes), 8)
            parent_axes = profile.axes
            for index in range(5):
                self.assertEqual(tuple(parent_axes[index].get_ylim()), (0.0, 1.005))
            inset_limits = [
                tuple(axis.get_ylim()) for axis in profile._capacity_zoom_axes
            ]
            self.assertCountEqual(inset_limits, list(EXPECTED_ZOOM_LIMITS.values()))
            profile_text = " ".join(text.get_text() for text in profile.texts)
            self.assertIn("not confidence intervals", profile_text)
            self.assertIn("full 0–1 domain", profile_text)
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_v1_and_v2_parent_axes_use_identical_scientific_plot_arrays(self) -> None:
        v1 = build_capacity_figures(self.report, self.policies.figure_spec)
        v2 = build_capacity_figures_v2(
            self.report, self.policies.figure_spec, self.correction
        )

        def assert_axis_arrays_equal(left, right) -> None:
            self.assertEqual(len(left.lines), len(right.lines))
            for left_line, right_line in zip(left.lines, right.lines, strict=True):
                np.testing.assert_array_equal(left_line.get_xdata(), right_line.get_xdata())
                np.testing.assert_array_equal(left_line.get_ydata(), right_line.get_ydata())
            self.assertEqual(len(left.collections), len(right.collections))
            for left_collection, right_collection in zip(
                left.collections, right.collections, strict=True
            ):
                if hasattr(left_collection, "get_segments"):
                    for left_segment, right_segment in zip(
                        left_collection.get_segments(),
                        right_collection.get_segments(),
                        strict=True,
                    ):
                        np.testing.assert_array_equal(left_segment, right_segment)
                else:
                    np.testing.assert_array_equal(
                        left_collection.get_offsets(), right_collection.get_offsets()
                    )

        try:
            assert_axis_arrays_equal(
                v1["capacity_condition_macro_f1"].axes[0],
                v2["capacity_condition_macro_f1"].axes[0],
            )
            for left_axis, right_axis in zip(
                v1["capacity_metric_profile"].axes,
                v2["capacity_metric_profile"].axes,
                strict=True,
            ):
                assert_axis_arrays_equal(left_axis, right_axis)
        finally:
            for figures in (v1, v2):
                for figure in figures.values():
                    plt.close(figure)

    def test_synthetic_publication_is_atomic_and_manifest_is_science_locked(self) -> None:
        sandbox_parent = (
            PROJECT_ROOT
            / "revised_experiments/artifacts/work9_capacity_test_sandboxes_v1"
        )
        sandbox_parent.mkdir(parents=True, exist_ok=True)
        output = Path(
            tempfile.mkdtemp(prefix="figure-v2-test-", dir=sandbox_parent)
        )
        shutil.rmtree(output)
        relative = output.relative_to(PROJECT_ROOT)
        report_reference = {
            "path": REPORT_RELATIVE.as_posix(),
            "sha256": PRODUCTION_ANALYSIS_REPORT_SHA256,
            "bytes": (PROJECT_ROOT / REPORT_RELATIVE).stat().st_size,
            "row_count": None,
        }
        synthetic = copy.deepcopy(self.report)
        synthetic["status"] = "synthetic_contract_test"
        try:
            manifest_path = render_capacity_figures_from_report_v2(
                PROJECT_ROOT,
                report=synthetic,
                report_reference=report_reference,
                figure_spec=self.policies.figure_spec,
                correction=self.correction,
                output_directory=relative,
                synthetic_test_mode=True,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["figure_ids"], list(FIGURE_IDS))
            self.assertEqual(manifest["figure_count"], 7)
            self.assertEqual(
                manifest["scientific_content_unchanged"]["analysis_report_sha256"],
                PRODUCTION_ANALYSIS_REPORT_SHA256,
            )
            for field in (
                "renderer_source",
                "renderer_cli",
                "renderer_schema",
                "renderer_tests",
            ):
                reference = manifest[field]
                bound = PROJECT_ROOT / reference["path"]
                self.assertEqual(bound.stat().st_size, reference["bytes"])
                self.assertEqual(
                    hashlib.sha256(bound.read_bytes()).hexdigest(),
                    reference["sha256"],
                )
            self.assertEqual(
                [row["presentation_revision"] for row in manifest["figures"]],
                [
                    "ceiling_zoom_disclosed_v2" if figure_id in AFFECTED_FIGURE_IDS else "unchanged_from_v1"
                    for figure_id in FIGURE_IDS
                ],
            )
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 7)
            self.assertEqual(len(list((output / "figures").glob("*.pdf"))), 7)
            self.assertEqual(len(list((output / "figures").glob("*.svg"))), 7)
            for path in output.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            self.assertEqual(output.stat().st_mode & 0o777, 0o555)
        finally:
            if output.exists():
                for path in sorted(output.rglob("*"), reverse=True):
                    if path.is_file():
                        path.chmod(0o644)
                    elif path.is_dir():
                        path.chmod(0o755)
                output.chmod(0o755)
                shutil.rmtree(output)

    def test_production_output_and_report_sha_are_fixed(self) -> None:
        with self.assertRaisesRegex(CapacityFigureError, "report SHA"):
            render_capacity_figures_v2(
                PROJECT_ROOT,
                report_path=REPORT_RELATIVE,
                expected_report_sha256="0" * 64,
                output_directory=FIGURE_OUTPUT_RELATIVE,
            )

        wrong_output = Path(
            "revised_experiments/artifacts/work9_capacity_test_sandboxes_v1/"
            "wrong-production-v2"
        )
        with self.assertRaisesRegex(CapacityFigureError, "fixed"):
            render_capacity_figures_from_report_v2(
                PROJECT_ROOT,
                report=self.report,
                report_reference={
                    "path": REPORT_RELATIVE.as_posix(),
                    "sha256": PRODUCTION_ANALYSIS_REPORT_SHA256,
                    "bytes": (PROJECT_ROOT / REPORT_RELATIVE).stat().st_size,
                    "row_count": None,
                },
                figure_spec=self.policies.figure_spec,
                correction=self.correction,
                output_directory=wrong_output,
                synthetic_test_mode=False,
            )


if __name__ == "__main__":
    unittest.main()
