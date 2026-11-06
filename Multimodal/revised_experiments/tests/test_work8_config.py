from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paper2_v2.config import (
    ArtifactRef,
    ProtocolValidationError,
    SelectionSeal,
    matrix_config_from_mapping,
    sha256_file,
)

from work8_test_utils import artifact, build_valid_raw_config, write_jsonl


class Work8ConfigTests(unittest.TestCase):
    def test_valid_locked_config_and_outer_eval_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            config = matrix_config_from_mapping(raw, project_root=root)
            self.assertEqual(len(config.conditions), 8)
            self.assertEqual(config.seeds, (17, 29, 43, 71, 101))
            view = config.selection_view(0, 0)
            self.assertTrue(view.train_manifest.path.name.startswith("train_"))
            self.assertTrue(view.val_manifest.path.name.startswith("val_"))
            self.assertFalse(hasattr(view, "outer_eval_manifest"))

            checkpoint = root / "revised_experiments/artifacts/checkpoints/model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            seal = SelectionSeal(
                protocol_id=config.protocol_id,
                partition_repeat=0,
                outer_fold=0,
                condition="AT-cross",
                seed=17,
                checkpoint=ArtifactRef(checkpoint, sha256_file(checkpoint)),
                validation_predictions_sha256=hashlib.sha256(b"validation").hexdigest(),
                threshold=0.47,
            )
            outer = config.authorize_outer_evaluation(seal)
            self.assertTrue(outer.outer_eval_manifest.path.name.startswith("outer_eval_"))

    def test_matrix_must_contain_exactly_all_eight_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            raw["conditions"] = raw["conditions"][:-1]
            with self.assertRaisesRegex(ProtocolValidationError, "conditions must be exactly"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=False, verify_manifest_isolation=False
                )

    def test_legacy_cv_5fold_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            raw["fold_protocol"]["folds"][0]["train_manifest"]["path"] = (
                "data/cv_5fold/fold0/train_segment_manifest.jsonl"
            )
            with self.assertRaisesRegex(ProtocolValidationError, "cv_5fold"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=False, verify_manifest_isolation=False
                )

    def test_old_test_partition_key_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            fold = raw["fold_protocol"]["folds"][0]
            fold["test_manifest"] = fold.pop("outer_eval_manifest")
            with self.assertRaisesRegex(ProtocolValidationError, "keys invalid"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=False, verify_manifest_isolation=False
                )

    def test_placeholder_hash_and_unsealed_text_store_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            raw["gate0_receipt"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ProtocolValidationError, "real lowercase SHA-256"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=False, verify_manifest_isolation=False
                )
            raw = build_valid_raw_config(root)
            metadata_path = root / raw["text_store"]["metadata"]["path"]
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored["status"] = "draft"
            metadata_path.write_text(json.dumps(stored) + "\n", encoding="utf-8")
            raw["text_store"]["metadata"] = artifact(metadata_path, root)
            with self.assertRaisesRegex(ProtocolValidationError, "locked contract"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=True, verify_manifest_isolation=False
                )

    def test_group_leakage_and_burned_calls_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            config = matrix_config_from_mapping(
                raw, project_root=root, verify_files=True, verify_manifest_isolation=False
            )
            val_path = config.fold(0, 0).val_manifest.path
            write_jsonl(
                val_path,
                [
                    {
                        "call_id": "eligible_call_2",
                        "segment_id": "leaked_segment",
                        "group_id": "eligible_group_2",
                        "label": 0,
                    }
                ],
            )
            with self.assertRaisesRegex(ProtocolValidationError, "call leakage"):
                config.validate_manifest_isolation()

            write_jsonl(
                val_path,
                [
                    {
                        "call_id": "historical_burned_call",
                        "segment_id": "burned_segment",
                        "group_id": "new_group",
                        "label": 0,
                    }
                ],
            )
            with self.assertRaisesRegex(ProtocolValidationError, "excluded/burned"):
                config.validate_manifest_isolation()

    def test_artifact_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            gate_path = root / raw["gate0_receipt"]["path"]
            gate_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolValidationError, "hash mismatch"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=True, verify_manifest_isolation=False
                )

    def test_primary_comparison_and_statistics_family_are_locked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = build_valid_raw_config(root)
            raw["statistics"]["primary_comparisons"].append(
                ["AT-cross", "AT-cross-randtrain"]
            )
            with self.assertRaisesRegex(ProtocolValidationError, "comparison family"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=False, verify_manifest_isolation=False
                )
            raw = build_valid_raw_config(root)
            raw["statistics"]["multiplicity_method"] = "none"
            with self.assertRaisesRegex(ProtocolValidationError, "statistics policies"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=False, verify_manifest_isolation=False
                )
            raw = build_valid_raw_config(root)
            raw["statistics"]["smallest_practically_meaningful_effect"] = 0.01
            with self.assertRaisesRegex(ProtocolValidationError, "invented"):
                matrix_config_from_mapping(
                    raw, project_root=root, verify_files=False, verify_manifest_isolation=False
                )


if __name__ == "__main__":
    unittest.main()
