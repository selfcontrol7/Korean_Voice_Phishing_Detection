from __future__ import annotations

import json
import unittest
from pathlib import Path

from paper2_v2.config import load_matrix_config, sha256_file
from paper2_v2.work8_preregistration import (
    CONFIG_RELATIVE_PATH,
    RECEIPT_RELATIVE_PATH,
    build_config_mapping,
    canonical_json_bytes,
)


class Work8OperationalPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.root / CONFIG_RELATIVE_PATH
        cls.receipt_path = cls.root / RECEIPT_RELATIVE_PATH

    def test_frozen_real_config_is_exact_deterministic_builder_output(self) -> None:
        expected = canonical_json_bytes(build_config_mapping(self.root))
        self.assertEqual(self.config_path.read_bytes(), expected)
        self.assertEqual(self.config_path.stat().st_mode & 0o222, 0)
        config = load_matrix_config(self.config_path, project_root=self.root)
        self.assertEqual(len(config.folds), 5)
        self.assertEqual(len(config.conditions), 8)
        self.assertEqual(config.fold_tree_sha256, (
            "1439174afa323843346ef1e137457f7a0ef4c0c154c2a223547909a48dee1e1e"
        ))

    def test_frozen_receipt_records_no_training_and_pairing_isolation(self) -> None:
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(receipt["config"]["sha256"], sha256_file(self.config_path))
        self.assertEqual(len(receipt["conditions"]), 8)
        self.assertEqual(receipt["pairing_plan_count"], 25)
        self.assertEqual(
            {item["split_role"] for item in receipt["random_train_pairing_plans"]},
            {"train"},
        )
        self.assertTrue(
            all(
                item["validation_pairing"] == "correct"
                and item["outer_eval_pairing"] == "correct"
                and item["label_policy"] == "unconstrained_global_random"
                for item in receipt["random_train_pairing_plans"]
            )
        )
        self.assertFalse(receipt["safety"]["training_launched"])
        self.assertFalse(receipt["safety"]["outer_eval_predictions_accessed"])
        self.assertFalse(receipt["safety"]["outer_eval_authorization_claimed"])
        self.assertEqual(receipt["eligible_inventory"]["unique_segment_count"], 34_765)
        self.assertEqual(receipt["eligible_inventory"]["text_store_missing_segments"], 0)
        self.assertFalse(receipt["source_label_counterfactual_cells_observed"])
        self.assertEqual(self.receipt_path.stat().st_mode & 0o222, 0)


if __name__ == "__main__":
    unittest.main()
