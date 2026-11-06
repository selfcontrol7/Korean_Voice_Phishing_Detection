from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from paper2_v2 import work9_gate1 as scientific_core
from paper2_v2.config import REPORTED_CALL_METRICS
from paper2_v2.evaluation import SegmentPrediction
from paper2_v2.work9_capacity_analysis import (
    COMPARISONS,
    CONDITIONS,
    FIGURE_IDS,
    GATE1_HASH_MANIFEST_SHA256,
    COMPLETED_GATE1_REPORT_SHA256,
    NormalizedCoordinate,
    ValidatedHybridInput,
    CapacityAnalysisError,
    _verify_complete_call_aggregation,
    _validate_analysis_report_schema,
    _validate_jsonl_schema,
    analyze_validated_hybrid,
    load_validated_hybrid_input,
    load_frozen_capacity_policies,
    verify_exact_hybrid_entries,
    verify_normalized_hybrid_contract,
    verify_report_claim_boundary,
)
from paper2_v2.work9_capacity_figures import (
    CapacityFigureError,
    build_capacity_figures,
    load_capacity_analysis_report,
    render_capacity_figures_from_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _probabilities(condition: str, fold: int, seed: int) -> tuple[float, ...]:
    perfect = (0.1, 0.2, 0.8, 0.9)
    if condition in {"AT-cross", "AT-cross-randtrain", "A-mean"}:
        return perfect
    if condition == "A-query":
        return (0.1, 0.6 if (fold + seed) % 2 else 0.2, 0.8, 0.9)
    if condition == "AT-concat":
        return (0.1, 0.2, 0.4 if fold % 2 else 0.8, 0.9)
    if condition == "AT-gate":
        return (0.6 if seed in {17, 43} else 0.1, 0.2, 0.8, 0.9)
    if condition == "A-temp":
        return (0.1, 0.2, 0.8, 0.4 if fold == 4 else 0.9)
    return (0.1, 0.6 if fold == 0 else 0.2, 0.8, 0.9)


def _synthetic_coordinates() -> tuple[NormalizedCoordinate, ...]:
    labels = (0, 0, 1, 1)
    parameter_counts = {
        "T": 533721,
        "A-mean": 531361,
        "A-query": 519361,
        "A-temp": 519361,
        "AT-concat": 536161,
        "AT-gate": 529537,
        "AT-cross": 530689,
        "AT-cross-randtrain": 530689,
    }
    rows: list[NormalizedCoordinate] = []
    for condition_index, condition in enumerate(CONDITIONS):
        anchor = condition in {"AT-cross", "AT-cross-randtrain"}
        family = "primary_matrix" if anchor else "capacity_matched_sensitivity"
        role = "immutable_primary_anchor" if anchor else "new_capacity_fit"
        call_schema = (
            "paper2-v2-work9-call-prediction-1"
            if anchor
            else "paper2-v2-work9-capacity-call-prediction-1"
        )
        record_schema = (
            "paper2-v2-work9-run-record-1"
            if anchor
            else "paper2-v2-work9-capacity-run-record-1"
        )
        for fold in range(5):
            for seed in (17, 29, 43, 71, 101):
                probabilities = _probabilities(condition, fold, seed)
                calls = tuple(
                    scientific_core.Gate1Call(
                        condition=condition,
                        partition_repeat=0,
                        outer_fold=fold,
                        seed=seed,
                        call_id=f"call-f{fold}-{index}",
                        group_id=f"relation-f{fold}-{index}",
                        label=label,
                        probability=probability,
                        hard_prediction=int(probability >= 0.5),
                        threshold=0.5,
                        n_segments=2,
                        source="FSS" if label else "AIHub",
                        start=float(index),
                        end=float(index + 1),
                    )
                    for index, (label, probability) in enumerate(
                        zip(labels, probabilities, strict=True)
                    )
                )
                rows.append(
                    NormalizedCoordinate(
                        run_id=f"{family}__{condition}__r00-o{fold:02d}-s{seed}",
                        condition=condition,
                        partition_repeat=0,
                        outer_fold=fold,
                        seed=seed,
                        entry_role=role,
                        source_run_family=family,
                        native_segment_schema_version=(
                            "paper2-v2-work9-segment-prediction-1"
                            if anchor
                            else "paper2-v2-work9-capacity-segment-prediction-1"
                        ),
                        native_call_schema_version=call_schema,
                        native_run_record_schema_version=record_schema,
                        calls=calls,
                        trainable_parameters=parameter_counts[condition],
                        wall_time_seconds=10.0 + condition_index + fold / 10.0,
                        peak_cuda_allocated_bytes=100_000_000
                        + condition_index * 1_000_000,
                        peak_cuda_reserved_bytes=120_000_000
                        + condition_index * 1_000_000,
                    )
                )
    return tuple(rows)


def _baseline() -> dict[str, object]:
    return {
        condition: {
            "metrics_across_seeds": {
                metric: {
                    "defined_count": 5,
                    "mean": 0.5,
                    "sample_standard_deviation": 0.0,
                    "minimum": 0.5,
                    "maximum": 0.5,
                }
                for metric in REPORTED_CALL_METRICS
            }
        }
        for condition in CONDITIONS
    }


def _validated() -> ValidatedHybridInput:
    dummy = {
        "path": "revised_experiments/artifacts/synthetic-index.json",
        "sha256": "a" * 64,
        "bytes": 1,
        "row_count": None,
    }
    return ValidatedHybridInput(
        coordinates=_synthetic_coordinates(),
        selection_index={"entries": []},
        outer_index={"entries": []},
        selection_index_reference=dummy,
        outer_index_reference={**dummy, "sha256": "b" * 64},
        normalization_audit={},
    )


def _baseline_lineage() -> dict[str, object]:
    return {
        "hash_manifest_sha256": GATE1_HASH_MANIFEST_SHA256,
        "report_sha256": COMPLETED_GATE1_REPORT_SHA256,
        "gate2_decision": "do_not_proceed",
        "may_be_reopened": False,
        "role": "sealed_descriptive_primary_baseline_only",
    }


class CapacityAnalysisFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policies = load_frozen_capacity_policies(PROJECT_ROOT)
        cls.synthetic_report = analyze_validated_hybrid(
            _validated(),
            policies=cls.policies,
            primary_condition_summaries=_baseline(),
            primary_baseline_lineage=_baseline_lineage(),
            synthetic_test_mode=True,
            bootstrap_replicates=64,
            permutation_replicates=64,
        )

    def test_exact_hybrid_inventory_requires_unique_150_capacity_and_50_anchor(
        self,
    ) -> None:
        entries = [
            {
                "condition": coordinate.condition,
                "run_id": coordinate.run_id,
                "partition_repeat": coordinate.partition_repeat,
                "outer_fold": coordinate.outer_fold,
                "seed": coordinate.seed,
                "entry_role": coordinate.entry_role,
                "source_run_family": coordinate.source_run_family,
            }
            for coordinate in _synthetic_coordinates()
        ]
        indexed = verify_exact_hybrid_entries(entries, context="synthetic hybrid")
        self.assertEqual(len(indexed), 200)
        self.assertEqual(
            sum(
                row["source_run_family"] == "capacity_matched_sensitivity"
                for row in entries
            ),
            150,
        )
        self.assertEqual(
            sum(row["source_run_family"] == "primary_matrix" for row in entries), 50
        )

        wrong_role = copy.deepcopy(entries)
        wrong_role[0]["entry_role"] = "immutable_primary_anchor"
        wrong_role[0]["source_run_family"] = "primary_matrix"
        with self.assertRaisesRegex(CapacityAnalysisError, "role/family mismatch"):
            verify_exact_hybrid_entries(wrong_role, context="synthetic hybrid")

        duplicate = copy.deepcopy(entries)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(CapacityAnalysisError, "duplicate coordinate"):
            verify_exact_hybrid_entries(duplicate, context="synthetic hybrid")

        wrong_run = copy.deepcopy(entries)
        wrong_run[0]["run_id"] = "capacity_matched_sensitivity__T__r00-o00-s999"
        with self.assertRaisesRegex(CapacityAnalysisError, "run_id mismatch"):
            verify_exact_hybrid_entries(wrong_run, context="synthetic hybrid")

    def test_normalization_rejects_native_family_or_paired_call_drift(self) -> None:
        rows = list(_synthetic_coordinates())
        audit = verify_normalized_hybrid_contract(rows)
        self.assertEqual(audit["new_capacity_coordinate_count"], 150)
        self.assertEqual(audit["immutable_primary_anchor_coordinate_count"], 50)

        rows[0] = replace(
            rows[0], native_call_schema_version="paper2-v2-work9-call-prediction-1"
        )
        with self.assertRaisesRegex(
            CapacityAnalysisError, "native schema family mismatch"
        ):
            verify_normalized_hybrid_contract(rows)

    def test_each_native_family_schema_is_validated_before_normalization(self) -> None:
        def call_row(*, family: str, schema_version: str) -> dict[str, object]:
            return {
                "schema_version": schema_version,
                "run_id": f"{family}__AT-cross__r00-o00-s17",
                "protocol_id": "paper2-v2-mfcc-matched-within-corpus-v1",
                "run_family": family,
                "partition": "outer_eval",
                "partition_repeat": 0,
                "outer_fold": 0,
                "seed": 17,
                "condition": "AT-cross",
                "call_id": "call-1",
                "group_id": "group-1",
                "source": "AIHub",
                "label": 0,
                "probability": 0.1,
                "hard_prediction": 0,
                "n_segments": 2,
                "start": 0.0,
                "end": 1.0,
                "duration_seconds": 1.0,
                "aggregation_rule": "mean_probability",
                "threshold": 0.5,
                "checkpoint_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            }

        capacity_schema = (
            PROJECT_ROOT
            / "revised_experiments/registry/schemas/work9_capacity/work9_call_prediction.schema.json"
        )
        primary_schema = (
            PROJECT_ROOT
            / "revised_experiments/registry/schemas/work9/work9_call_prediction.schema.json"
        )
        capacity = call_row(
            family="capacity_matched_sensitivity",
            schema_version="paper2-v2-work9-capacity-call-prediction-1",
        )
        primary = call_row(
            family="primary_matrix",
            schema_version="paper2-v2-work9-call-prediction-1",
        )
        _validate_jsonl_schema(
            [capacity], schema_path=capacity_schema, context="synthetic capacity row"
        )
        _validate_jsonl_schema(
            [primary], schema_path=primary_schema, context="synthetic primary row"
        )
        with self.assertRaisesRegex(CapacityAnalysisError, "schema validation failed"):
            _validate_jsonl_schema(
                [capacity], schema_path=primary_schema, context="wrong native family"
            )

    def test_complete_call_probability_is_recomputed_from_all_segments(self) -> None:
        segments = [
            SegmentPrediction(
                segment_id="s1",
                call_id="c1",
                group_id="g1",
                label=1,
                probability=0.2,
                start=0.0,
                end=1.0,
                source="FSS",
            ),
            SegmentPrediction(
                segment_id="s2",
                call_id="c1",
                group_id="g1",
                label=1,
                probability=0.8,
                start=1.0,
                end=2.0,
                source="FSS",
            ),
        ]
        call = {
            "call_id": "c1",
            "group_id": "g1",
            "label": 1,
            "source": "FSS",
            "n_segments": 2,
            "start": 0.0,
            "end": 2.0,
            "duration_seconds": 2.0,
            "probability": 0.5,
        }
        recomputed = _verify_complete_call_aggregation(
            segments, [call], context="synthetic aggregation"
        )
        self.assertEqual(recomputed["c1"].probability, 0.5)
        changed = dict(call, probability=0.51)
        with self.assertRaisesRegex(
            CapacityAnalysisError, "mean probability does not recompute"
        ):
            _verify_complete_call_aggregation(
                segments, [changed], context="synthetic aggregation"
            )

        rows = list(_synthetic_coordinates())
        mutated_call = replace(rows[0].calls[0], call_id="post-hoc-extra-call")
        rows[0] = replace(rows[0], calls=(mutated_call, *rows[0].calls[1:]))
        with self.assertRaisesRegex(
            CapacityAnalysisError, "paired call inventory differs"
        ):
            verify_normalized_hybrid_contract(rows)

    def test_analysis_executes_only_frozen_three_and_schema_validates(self) -> None:
        report = self.synthetic_report
        pairs = tuple(
            (row["primary_condition"], row["comparator_condition"])
            for row in report["secondary_comparisons"]
        )
        self.assertEqual(pairs, COMPARISONS)
        self.assertEqual(report["normalization_audit"]["normalized_coordinate_count"], 200)
        self.assertEqual(report["completed_gate1"]["gate2_decision"], "do_not_proceed")
        self.assertIs(report["completed_gate1"]["may_be_reopened"], False)
        _validate_analysis_report_schema(report, PROJECT_ROOT)

    def test_production_analysis_forbids_replicate_override(self) -> None:
        with self.assertRaisesRegex(
            CapacityAnalysisError, "restricted to synthetic tests"
        ):
            analyze_validated_hybrid(
                _validated(),
                policies=self.policies,
                primary_condition_summaries=_baseline(),
                primary_baseline_lineage=_baseline_lineage(),
                bootstrap_replicates=8,
            )

    def test_gate_reopening_and_fourth_comparison_hard_fail(self) -> None:
        reopened = copy.deepcopy(self.synthetic_report)
        reopened["completed_gate1"]["may_be_reopened"] = True
        with self.assertRaisesRegex(CapacityAnalysisError, "reopen or relabel"):
            verify_report_claim_boundary(reopened)

        expanded = copy.deepcopy(self.synthetic_report)
        expanded["secondary_comparisons"].append(
            copy.deepcopy(expanded["secondary_comparisons"][0])
        )
        with self.assertRaisesRegex(CapacityAnalysisError, "exactly three"):
            verify_report_claim_boundary(expanded)

        relabelled = copy.deepcopy(self.synthetic_report)
        relabelled["new_gate1_decision"] = "pass"
        with self.assertRaisesRegex(CapacityAnalysisError, "forbidden gate/claim"):
            verify_report_claim_boundary(relabelled)

    def test_renderer_builds_only_frozen_seven_and_rejects_expansion(self) -> None:
        figures = build_capacity_figures(
            self.synthetic_report, self.policies.figure_spec
        )
        try:
            self.assertEqual(tuple(figures), FIGURE_IDS)
            self.assertEqual(len(figures), 7)
        finally:
            for figure in figures.values():
                plt.close(figure)

        expanded_spec = copy.deepcopy(self.policies.figure_spec)
        expanded_spec["figures"].append(
            {
                "figure_id": "new_gate1_pass_fail_plot",
                "inferential": True,
                "purpose": "forbidden",
            }
        )
        with self.assertRaisesRegex(
            CapacityFigureError, "non-frozen figure specification"
        ):
            build_capacity_figures(self.synthetic_report, expanded_spec)

    def test_renderer_saves_exactly_seven_ids_in_three_formats(self) -> None:
        artifact_root = (
            PROJECT_ROOT
            / "revised_experiments/artifacts/work9_capacity_test_sandboxes_v1"
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".capacity-figure-synthetic-", dir=artifact_root)
        )
        try:
            report_path = temporary / "synthetic_report.json"
            payload = (
                json.dumps(
                    self.synthetic_report,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            report_path.write_bytes(payload)
            report_reference = {
                "path": report_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "row_count": None,
            }
            destination = temporary / "figures_v1"
            destination_relative = destination.relative_to(PROJECT_ROOT)
            manifest_path = render_capacity_figures_from_report(
                PROJECT_ROOT,
                report=self.synthetic_report,
                report_reference=report_reference,
                figure_spec=self.policies.figure_spec,
                output_directory=destination_relative,
                synthetic_test_mode=True,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["figure_count"], 7)
            self.assertEqual(tuple(manifest["figure_ids"]), FIGURE_IDS)
            self.assertEqual(len(manifest["figures"]), 7)
            figure_files = sorted((destination / "figures").iterdir())
            self.assertEqual(len(figure_files), 21)
            self.assertEqual({path.suffix for path in figure_files}, {".svg", ".pdf", ".png"})
            prefix = destination.relative_to(PROJECT_ROOT).as_posix() + "/figures/"
            for row in manifest["figures"]:
                self.assertEqual(set(row["files"]), {"svg", "pdf", "png_300dpi"})
                for reference in row["files"].values():
                    self.assertTrue(reference["path"].startswith(prefix))
            self.assertFalse(destination.stat().st_mode & 0o222)
            self.assertTrue(
                all(not path.stat().st_mode & 0o222 for path in destination.rglob("*"))
            )
        finally:
            if temporary.exists():
                temporary.chmod(0o755)
                for path in temporary.rglob("*"):
                    path.chmod(0o755 if path.is_dir() else 0o644)
            shutil.rmtree(temporary, ignore_errors=True)

    def test_renderer_output_is_explicitly_synthetic_and_path_isolated(self) -> None:
        reference = {
            "path": "revised_experiments/artifacts/work9_capacity_test_sandboxes_v1/report.json",
            "sha256": "a" * 64,
            "bytes": 1,
            "row_count": None,
        }
        with self.assertRaisesRegex(CapacityFigureError, "mode differs"):
            render_capacity_figures_from_report(
                PROJECT_ROOT,
                report=self.synthetic_report,
                report_reference=reference,
                figure_spec=self.policies.figure_spec,
                output_directory="revised_experiments/artifacts/work9_capacity_figures_v1",
            )
        with self.assertRaisesRegex(CapacityFigureError, "safe project-relative"):
            render_capacity_figures_from_report(
                PROJECT_ROOT,
                report=self.synthetic_report,
                report_reference=reference,
                figure_spec=self.policies.figure_spec,
                output_directory=PROJECT_ROOT / "absolute-forbidden",
                synthetic_test_mode=True,
            )
        with self.assertRaisesRegex(CapacityFigureError, "safe project-relative"):
            render_capacity_figures_from_report(
                PROJECT_ROOT,
                report=self.synthetic_report,
                report_reference=reference,
                figure_spec=self.policies.figure_spec,
                output_directory="revised_experiments/artifacts/work9_capacity_test_sandboxes_v1/../escape",
                synthetic_test_mode=True,
            )

    def test_analysis_and_renderer_top_level_inputs_reject_symlink_aliases(self) -> None:
        artifact_root = (
            PROJECT_ROOT
            / "revised_experiments/artifacts/work9_capacity_test_sandboxes_v1"
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-input-symlink-", dir=artifact_root
        ) as temporary:
            root = Path(temporary)
            real = root / "real.json"
            real.write_text("{}\n", encoding="utf-8")
            alias = root / "alias.json"
            alias.symlink_to(real)
            digest = hashlib.sha256(real.read_bytes()).hexdigest()
            with self.assertRaisesRegex(CapacityAnalysisError, "symlink"):
                load_validated_hybrid_input(
                    PROJECT_ROOT,
                    selection_index_path=alias,
                    expected_selection_index_sha256=digest,
                    outer_index_path=real,
                    expected_outer_index_sha256=digest,
                )
            with self.assertRaisesRegex(CapacityFigureError, "symlink"):
                load_capacity_analysis_report(
                    PROJECT_ROOT,
                    report_path=alias,
                    expected_report_sha256=digest,
                )


if __name__ == "__main__":
    unittest.main()
