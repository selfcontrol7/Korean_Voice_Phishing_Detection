from __future__ import annotations

import json
import unittest
from pathlib import Path

from paper2_v2.config import sha256_file
from paper2_v2.work9_capacity_acceptance import source_contracts
from paper2_v2.work9_capacity_training import validate_json_schema
from paper2_v2.work9_capacity_types import (
    ANCHOR_POLICY_SHA256,
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    EXPERIMENT_CARD_SHA256,
    FIGURE_SPEC_SHA256,
    EXECUTION_ADDENDUM_SHA256,
    PRIMARY_ENVIRONMENT_LOCK_SHA256,
    PRIMARY_RUNNER_ACCEPTANCE_SHA256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "revised_experiments/configs"


class CapacityLineageTests(unittest.TestCase):
    def test_frozen_pre_outcome_policies_are_exact_and_read_only(self) -> None:
        expected = {
            "work9_capacity_anchor_policy_v1.json": ANCHOR_POLICY_SHA256,
            "work9_capacity_analysis_protocol_v1.json": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "work9_capacity_figure_spec_v1.json": FIGURE_SPEC_SHA256,
            "work9_capacity_matched_experiment_card_v1.json": EXPERIMENT_CARD_SHA256,
            "work9_capacity_execution_lineage_addendum_v1.json": (
                "5e51337d98d2f6c6f5fbc60922da4ebe4b99c7a7cf07c3097216613c611bbebe"
            ),
            "work9_capacity_execution_lineage_addendum_v2.json": (
                "6036f5aba1985664d06bed250c532ce0c201d8e2b0b8b1de7943ac2a2184a642"
            ),
            "work9_capacity_execution_lineage_addendum_v3.json": EXECUTION_ADDENDUM_SHA256,
        }
        for name, digest in expected.items():
            path = CONFIG_ROOT / name
            self.assertEqual(sha256_file(path), digest)
            self.assertFalse(path.stat().st_mode & 0o222)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["status"], "frozen_before_capacity_model_outcomes")

    def test_execution_addendum_v3_binds_exact_execution_packages(self) -> None:
        addendum = json.loads(
            (
                CONFIG_ROOT
                / "work9_capacity_execution_lineage_addendum_v3.json"
            ).read_text(encoding="utf-8")
        )
        package = addendum["analysis_execution_package"]
        self.assertIs(
            package["implemented_and_synthetic_tested_before_capacity_outcomes"],
            True,
        )
        self.assertEqual(len(package["only_frozen_three_comparisons"]), 3)
        self.assertEqual(len(package["only_frozen_figure_ids"]), 7)
        for reference in package["package_files"]:
            path = PROJECT_ROOT / reference["path"]
            self.assertEqual(path.stat().st_size, reference["bytes"])
            self.assertEqual(sha256_file(path), reference["sha256"])
        for reference in addendum["post_v2_audit_hardening"]["package_files"]:
            path = PROJECT_ROOT / reference["path"]
            self.assertEqual(path.stat().st_size, reference["bytes"])
            self.assertEqual(sha256_file(path), reference["sha256"])

    def test_card_preserves_completed_gate_and_secondary_scope(self) -> None:
        card = json.loads(
            (CONFIG_ROOT / "work9_capacity_matched_experiment_card_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(card["completed_gate1"]["gate2_decision"], "do_not_proceed")
        self.assertIn("a_new_or_reopened_gate1", card["claims_not_enabled"])
        self.assertIn("gate2_continuation", card["claims_not_enabled"])
        self.assertEqual(card["hybrid_matrix"]["new_fit_coordinates"], 150)
        self.assertEqual(card["hybrid_matrix"]["anchor_coordinates"], 50)

    def test_accepted_primary_source_hashes_remain_unchanged(self) -> None:
        expected = {
            "src/paper2_v2/work9_types.py": "6b6da5ebec8c830a86a9da514c864c0ee8706da68589c3b41d04e1b4019ba002",
            "src/paper2_v2/work9_training.py": "0ae727dcc97c6c9c1f1bef4c938c518b4c2e9205226f4859fc0aa802795ea89a",
            "scripts/run_work9.py": "161b0c0efbf8e35262793c975621e453d309b0bf3d6b89692fb429255063c056",
        }
        revised = PROJECT_ROOT / "revised_experiments"
        for relative, digest in expected.items():
            self.assertEqual(sha256_file(revised / relative), digest)

    def test_capacity_acceptance_inventory_excludes_primary_runner_contracts(self) -> None:
        paths = {row["path"] for row in source_contracts(PROJECT_ROOT)}
        self.assertNotIn("revised_experiments/src/paper2_v2/work9_types.py", paths)
        self.assertNotIn("revised_experiments/src/paper2_v2/work9_training.py", paths)
        self.assertNotIn("revised_experiments/src/paper2_v2/work9_registry.py", paths)
        self.assertNotIn("revised_experiments/scripts/run_work9.py", paths)
        self.assertIn("revised_experiments/src/paper2_v2/work9_capacity_training.py", paths)

    def test_hybrid_selection_schema_requires_150_plus_50_roles(self) -> None:
        artifact = {
            "path": "revised_experiments/configs/work9_capacity_anchor_policy_v1.json",
            "sha256": "0" * 64,
            "bytes": 1,
            "row_count": None,
        }
        entries = []
        conditions = [
            "T", "A-mean", "A-query", "A-temp", "AT-concat", "AT-gate",
            "AT-cross", "AT-cross-randtrain",
        ]
        for condition in conditions:
            for fold in range(5):
                for seed in (17, 29, 43, 71, 101):
                    anchor = condition.startswith("AT-cross")
                    entries.append(
                        {
                            "run_id": f"run-{condition}-{fold}-{seed}",
                            "condition": condition,
                            "partition_repeat": 0,
                            "outer_fold": fold,
                            "seed": seed,
                            "entry_role": (
                                "immutable_primary_anchor" if anchor else "new_capacity_fit"
                            ),
                            "source_run_family": (
                                "primary_matrix" if anchor else "capacity_matched_sensitivity"
                            ),
                            "selection_seal": artifact,
                            "selection_run_record": artifact,
                            "epoch_trace": artifact,
                            "threshold_trace": artifact,
                        }
                    )
        payload = {
            "schema_version": "paper2-v2-work9-capacity-selection-index-1",
            "protocol_id": "paper2-v2-mfcc-matched-within-corpus-v1",
            "run_family": "capacity_matched_sensitivity",
            "status": "frozen_complete",
            "expected_coordinate_count": 200,
            "config": artifact,
            "work8_receipt": artifact,
            "runner_acceptance": artifact,
            "experiment_card": artifact,
            "anchor_policy": artifact,
            "capacity_analysis_protocol": artifact,
            "figure_spec": artifact,
            "execution_addendum": artifact,
            "primary_runner_acceptance": artifact,
            "primary_selection_index": artifact,
            "completed_gate1_hash_manifest": artifact,
            "new_coordinate_count": 150,
            "anchor_coordinate_count": 50,
            "entries": entries,
        }
        validate_json_schema(
            payload,
            project_root=PROJECT_ROOT,
            schema_name="work9_selection_index.schema.json",
        )
        self.assertEqual(sum(row["entry_role"] == "new_capacity_fit" for row in entries), 150)
        self.assertEqual(
            sum(row["entry_role"] == "immutable_primary_anchor" for row in entries), 50
        )


if __name__ == "__main__":
    unittest.main()
