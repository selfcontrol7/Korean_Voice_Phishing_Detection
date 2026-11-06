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
from paper2_v2.work9_capacity_figures import CapacityFigureError
from paper2_v2.work9_capacity_figures_v2 import (
    PRODUCTION_ANALYSIS_REPORT_SHA256,
    build_capacity_figures_v2,
    load_presentation_correction,
)
from paper2_v2.work9_capacity_figures_v3 import (
    FIGURE_OUTPUT_RELATIVE,
    NARROW_ZOOM_FORMAT,
    NARROW_ZOOM_SPAN_THRESHOLD,
    OTHER_ZOOM_FORMAT,
    PRESENTATION_SUPERSESSION_SHA256,
    V2_FIGURE_MANIFEST_SHA256,
    V2_HASH_MANIFEST_SHA256,
    build_capacity_figures_v3,
    load_presentation_supersession,
    render_capacity_figures_from_report_v3,
    render_capacity_figures_v3,
    verify_presentation_supersession,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_analysis_v1/report.json"
)
V2_MANIFEST_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v2/figure_manifest.json"
)
V2_HASHES_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v2/hashes.sha256"
)


def _assert_axis_arrays_equal(
    case: unittest.TestCase, left: plt.Axes, right: plt.Axes
) -> None:
    case.assertEqual(len(left.lines), len(right.lines))
    for left_line, right_line in zip(left.lines, right.lines, strict=True):
        np.testing.assert_array_equal(left_line.get_xdata(), right_line.get_xdata())
        np.testing.assert_array_equal(left_line.get_ydata(), right_line.get_ydata())
    case.assertEqual(len(left.collections), len(right.collections))
    for left_collection, right_collection in zip(
        left.collections, right.collections, strict=True
    ):
        if hasattr(left_collection, "get_segments"):
            left_segments = left_collection.get_segments()
            right_segments = right_collection.get_segments()
            case.assertEqual(len(left_segments), len(right_segments))
            for left_segment, right_segment in zip(
                left_segments, right_segments, strict=True
            ):
                np.testing.assert_array_equal(left_segment, right_segment)
        else:
            np.testing.assert_array_equal(
                left_collection.get_offsets(), right_collection.get_offsets()
            )


class CapacityFigureV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((PROJECT_ROOT / REPORT_RELATIVE).read_text())
        cls.policies = load_frozen_capacity_policies(PROJECT_ROOT)
        cls.correction_v2 = load_presentation_correction(PROJECT_ROOT)
        cls.supersession_v3 = load_presentation_supersession(PROJECT_ROOT)

    def test_frozen_supersession_binds_immutable_v2_and_citation_status(self) -> None:
        path = (
            PROJECT_ROOT
            / "revised_experiments/configs/"
            "work9_capacity_figure_presentation_supersession_v3.json"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            PRESENTATION_SUPERSESSION_SHA256,
        )
        self.assertEqual(path.stat().st_mode & 0o777, 0o444)
        verify_presentation_supersession(
            self.supersession_v3, project_root=PROJECT_ROOT
        )
        self.assertFalse(self.supersession_v3["v2_citable"])
        self.assertTrue(
            self.supersession_v3["v3_citable_after_integrity_and_visual_qa"]
        )
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / V2_MANIFEST_RELATIVE).read_bytes()).hexdigest(),
            V2_FIGURE_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256((PROJECT_ROOT / V2_HASHES_RELATIVE).read_bytes()).hexdigest(),
            V2_HASH_MANIFEST_SHA256,
        )

    def test_supersession_mutations_fail_closed(self) -> None:
        changed = copy.deepcopy(self.supersession_v3)
        changed["v2_citable"] = True
        with self.assertRaisesRegex(CapacityFigureError, "identity"):
            verify_presentation_supersession(changed)

        changed = copy.deepcopy(self.supersession_v3)
        changed["correction_scope"]["formatter_rule"][
            "zoom_span_threshold_inclusive"
        ] = 0.01
        with self.assertRaisesRegex(CapacityFigureError, "tick-label"):
            verify_presentation_supersession(changed)

        changed = copy.deepcopy(self.supersession_v3)
        changed["preservation"]["plotting_array_changes_allowed"] = True
        with self.assertRaisesRegex(CapacityFigureError, "scientific preservation"):
            verify_presentation_supersession(changed)

    def test_exact_seven_ids_and_all_v2_v3_plotting_arrays_are_identical(self) -> None:
        v2 = build_capacity_figures_v2(
            self.report, self.policies.figure_spec, self.correction_v2
        )
        v3 = build_capacity_figures_v3(
            self.report,
            self.policies.figure_spec,
            self.correction_v2,
            self.supersession_v3,
        )
        try:
            self.assertEqual(tuple(v2), FIGURE_IDS)
            self.assertEqual(tuple(v3), FIGURE_IDS)
            for figure_id in FIGURE_IDS:
                self.assertEqual(len(v2[figure_id].axes), len(v3[figure_id].axes))
                for left, right in zip(
                    v2[figure_id].axes, v3[figure_id].axes, strict=True
                ):
                    _assert_axis_arrays_equal(self, left, right)
                    self.assertEqual(tuple(left.get_xlim()), tuple(right.get_xlim()))
                    self.assertEqual(tuple(left.get_ylim()), tuple(right.get_ylim()))
            for left, right in zip(
                v2["capacity_metric_profile"]._capacity_zoom_axes,
                v3["capacity_metric_profile"]._capacity_zoom_axes,
                strict=True,
            ):
                _assert_axis_arrays_equal(self, left, right)
                self.assertEqual(tuple(left.get_xlim()), tuple(right.get_xlim()))
                self.assertEqual(tuple(left.get_ylim()), tuple(right.get_ylim()))
        finally:
            for figures in (v2, v3):
                for figure in figures.values():
                    plt.close(figure)

    def test_formatter_rule_is_minimal_and_narrow_tick_labels_are_unique(self) -> None:
        figures = build_capacity_figures_v3(
            self.report,
            self.policies.figure_spec,
            self.correction_v2,
            self.supersession_v3,
        )
        try:
            profile = figures["capacity_metric_profile"]
            self.assertEqual(
                profile._capacity_zoom_formatter_formats,
                (
                    OTHER_ZOOM_FORMAT,
                    OTHER_ZOOM_FORMAT,
                    OTHER_ZOOM_FORMAT,
                    NARROW_ZOOM_FORMAT,
                    NARROW_ZOOM_FORMAT,
                ),
            )
            profile.canvas.draw()
            for axis, format_string in zip(
                profile._capacity_zoom_axes,
                profile._capacity_zoom_formatter_formats,
                strict=True,
            ):
                lower, upper = axis.get_ylim()
                expected = (
                    NARROW_ZOOM_FORMAT
                    if upper - lower <= NARROW_ZOOM_SPAN_THRESHOLD + 1e-12
                    else OTHER_ZOOM_FORMAT
                )
                self.assertEqual(format_string, expected)
                ticks = [
                    tick
                    for tick in axis.get_yticks()
                    if lower - 1e-12 <= tick <= upper + 1e-12
                ]
                labels = [axis.yaxis.get_major_formatter()(tick) for tick in ticks]
                self.assertEqual(len(labels), len(set(labels)))
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_synthetic_publication_is_atomic_hash_bound_and_exact_7_by_3(self) -> None:
        sandbox_parent = (
            PROJECT_ROOT
            / "revised_experiments/artifacts/work9_capacity_test_sandboxes_v1"
        )
        sandbox_parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="figure-v3-test-", dir=sandbox_parent))
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
            manifest_path = render_capacity_figures_from_report_v3(
                PROJECT_ROOT,
                report=synthetic,
                report_reference=report_reference,
                figure_spec=self.policies.figure_spec,
                correction_v2=self.correction_v2,
                supersession_v3=self.supersession_v3,
                output_directory=relative,
                synthetic_test_mode=True,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["figure_ids"], list(FIGURE_IDS))
            self.assertEqual(manifest["figure_count"], 7)
            self.assertEqual(
                manifest["citation_status"],
                {"v2_bundle_citable": False, "v3_bundle_citable": True},
            )
            self.assertEqual(
                manifest["scientific_content_unchanged"]["analysis_report_sha256"],
                PRODUCTION_ANALYSIS_REPORT_SHA256,
            )
            for field in (
                "v2_presentation_correction",
                "v3_presentation_supersession",
                "renderer_source",
                "renderer_cli",
                "renderer_schema",
                "renderer_tests",
                "supersedes_v2_figure_manifest",
                "supersedes_v2_hash_manifest",
            ):
                reference = manifest[field]
                bound = PROJECT_ROOT / reference["path"]
                self.assertEqual(bound.stat().st_size, reference["bytes"])
                self.assertEqual(
                    hashlib.sha256(bound.read_bytes()).hexdigest(),
                    reference["sha256"],
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
            render_capacity_figures_v3(
                PROJECT_ROOT,
                report_path=REPORT_RELATIVE,
                expected_report_sha256="0" * 64,
                output_directory=FIGURE_OUTPUT_RELATIVE,
            )

        with self.assertRaisesRegex(CapacityFigureError, "fixed"):
            render_capacity_figures_from_report_v3(
                PROJECT_ROOT,
                report=self.report,
                report_reference={
                    "path": REPORT_RELATIVE.as_posix(),
                    "sha256": PRODUCTION_ANALYSIS_REPORT_SHA256,
                    "bytes": (PROJECT_ROOT / REPORT_RELATIVE).stat().st_size,
                    "row_count": None,
                },
                figure_spec=self.policies.figure_spec,
                correction_v2=self.correction_v2,
                supersession_v3=self.supersession_v3,
                output_directory=Path(
                    "revised_experiments/artifacts/work9_capacity_test_sandboxes_v1/"
                    "wrong-production-v3"
                ),
                synthetic_test_mode=False,
            )


if __name__ == "__main__":
    unittest.main()
