from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

from paper2_v2.work9_capacity_postcompletion_closure import (
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    EXPECTED_ACCEPTED_CONTRACT_COUNT,
    EXPECTED_CURRENT_CONTRACT_COUNT,
    PostcompletionClosureError,
    load_closure_config,
    validate_postcompletion_closure,
    verify_accepted_contract_boundary,
    verify_artifact_reference,
    verify_bound_completion_artifacts,
    verify_v3_scientific_and_visual_correction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CapacityPostcompletionClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_closure_config(PROJECT_ROOT)

    def test_frozen_closure_config_is_read_only_and_hash_exact(self) -> None:
        path = PROJECT_ROOT / CONFIG_RELATIVE
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), CONFIG_SHA256)
        self.assertEqual(path.stat().st_mode & 0o777, 0o444)
        self.assertEqual(
            self.config["status"], "frozen_post_completion_closure_contract"
        )

    def test_all_accepted_contracts_are_exact_and_delta_is_only_v2_v3_schemas(self) -> None:
        result = verify_accepted_contract_boundary(PROJECT_ROOT, self.config)
        self.assertEqual(
            result["accepted_contract_count"], EXPECTED_ACCEPTED_CONTRACT_COUNT
        )
        self.assertEqual(
            result["current_contract_count"], EXPECTED_CURRENT_CONTRACT_COUNT
        )
        self.assertEqual(result["changed_accepted_contract_count"], 0)
        self.assertEqual(result["removed_accepted_contract_count"], 0)
        self.assertEqual(
            [row["path"] for row in result["added_contracts"]],
            [
                "revised_experiments/registry/schemas/work9_capacity/"
                "work9_capacity_figure_manifest_v2.schema.json",
                "revised_experiments/registry/schemas/work9_capacity/"
                "work9_capacity_figure_manifest_v3.schema.json",
            ],
        )
        self.assertEqual(
            result["historical_boundary_signal"],
            "paper2_v2.work9_capacity_acceptance.Work9AcceptanceError: "
            "dry receipt source inventory differs from current code",
        )

    def test_all_completion_artifacts_and_three_figure_generations_are_bound(self) -> None:
        result = verify_bound_completion_artifacts(PROJECT_ROOT, self.config)
        self.assertEqual(result["artifact_binding_count"], 14)
        self.assertEqual(result["selection_index_entries"], 200)
        self.assertEqual(result["outer_index_entries"], 200)
        self.assertEqual(
            result["figure_bundle_file_counts_excluding_hash_manifest"],
            {"v1": 22, "v2": 23, "v3": 23},
        )
        self.assertFalse(result["v2_citable"])
        self.assertTrue(result["v3_citable"])

    def test_v3_scientific_arrays_and_limits_equal_v2_and_ticks_are_unique(self) -> None:
        result = verify_v3_scientific_and_visual_correction(
            PROJECT_ROOT, self.config
        )
        self.assertEqual(result["figure_ids_compared"], 7)
        self.assertGreater(result["plotting_array_and_limit_checks"], 400)
        self.assertEqual(result["narrow_formatter"], "%.4f")
        self.assertEqual(result["other_formatter"], "%.3f")
        self.assertTrue(result["unique_tick_labels"])
        self.assertEqual(result["png_300dpi_files_verified"], 7)
        self.assertTrue(result["svg_xml_and_tick_labels_verified"])

    def test_complete_closure_is_pass_but_historical_full_discover_is_not(self) -> None:
        result = validate_postcompletion_closure(PROJECT_ROOT)
        self.assertEqual(result["status"], "pass")
        historical = result["historical_full_discover"]
        self.assertEqual(
            historical["interpretation"],
            "expected_historical_boundary_signal_not_a_pass_and_not_scientific_drift",
        )
        self.assertTrue(historical["must_not_be_suppressed_or_relabelled_as_pass"])
        self.assertEqual(historical["full_discover_reported_errors"], 1)
        self.assertEqual(historical["full_discover_failures"], 0)
        self.assertTrue(
            all(
                result["safety"][key] is False
                for key in (
                    "cuda_smoke_launched",
                    "model_outcome_values_recomputed",
                    "new_outer_evaluation_launched",
                    "new_training_launched",
                    "outer_outcome_ledgers_opened",
                    "primary_or_capacity_registry_modified",
                    "selection_or_reselection_launched",
                )
            )
        )

    def test_mutated_lineage_and_unexpected_contract_delta_fail_closed(self) -> None:
        report_ref = copy.deepcopy(self.config["artifact_bindings"]["analysis_report"])
        report_ref["sha256"] = "0" * 64
        with self.assertRaisesRegex(PostcompletionClosureError, "hash changed"):
            verify_artifact_reference(
                PROJECT_ROOT, report_ref, context="mutated analysis report"
            )

        changed = copy.deepcopy(self.config)
        changed["accepted_contract_boundary"]["expected_additions"] = changed[
            "accepted_contract_boundary"
        ]["expected_additions"][:1]
        with self.assertRaisesRegex(PostcompletionClosureError, "contract delta"):
            verify_accepted_contract_boundary(PROJECT_ROOT, changed)


if __name__ == "__main__":
    unittest.main()
