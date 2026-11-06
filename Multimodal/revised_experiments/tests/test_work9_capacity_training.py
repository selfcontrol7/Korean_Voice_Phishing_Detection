from __future__ import annotations

from copy import deepcopy
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from paper2_v2.config import ArtifactRef, ProtocolValidationError, sha256_file
from paper2_v2.work9_capacity_acceptance import (
    Work9AcceptanceError,
    bundle_digests,
    source_contracts,
    verify_dry_receipt_payload_against_current,
)
from paper2_v2.work9_capacity_training import (
    CapacityTrainingError,
    _load_ref,
    _verify_ref_mapping,
    validate_capacity_no_training,
    verify_outer_run_record_before_terminal,
    verify_capacity_selection_seal_against_frozen,
    verify_cuda_device_identity,
    verify_exact_hybrid_entries,
    record_coordinate_failure,
    remaining_epoch_numbers,
    verify_frozen_work9_inputs,
    verify_primary_selection_anchor_lineage,
)
from paper2_v2.work9_capacity_types import (
    ANCHOR_CONDITIONS,
    ANCHOR_POLICY_SHA256,
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    EXECUTION_ADDENDUM_SHA256,
    EXPECTED_HYBRID_COUNT,
    EXPECTED_NEW_SELECTION_COUNT,
    NEW_FIT_CONDITIONS,
    PRIMARY_ENVIRONMENT_LOCK_SHA256,
    PRIMARY_RUNNER_ACCEPTANCE_SHA256,
    RUN_FAMILY,
    TEST_SANDBOX_ROOT_RELATIVE,
    WORK8_CONFIG_SHA256,
    WORK8_RECEIPT_SHA256,
    CapacityCoordinates,
    CapacitySelectionSeal,
    artifact_from_mapping,
    expected_coordinates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "paper2-v2-mfcc-matched-within-corpus-v1"
SEEDS = (17, 29, 43, 71, 101)


class CapacityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen = verify_frozen_work9_inputs(
            PROJECT_ROOT, require_runner_acceptance=False
        )

    def test_coordinate_family_and_new_matrix_cardinality(self) -> None:
        coordinates = expected_coordinates(PROTOCOL_ID, SEEDS)
        self.assertEqual(len(coordinates), EXPECTED_NEW_SELECTION_COUNT)
        self.assertEqual(len(set(coordinates)), 150)
        self.assertEqual({row.condition for row in coordinates}, set(NEW_FIT_CONDITIONS))
        self.assertEqual(EXPECTED_HYBRID_COUNT, 200)

    def test_wrong_family_and_unknown_condition_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            CapacityCoordinates(PROTOCOL_ID, "primary_matrix", "A-mean", 0, 0, 17)
        with self.assertRaises(ValueError):
            CapacityCoordinates(PROTOCOL_ID, RUN_FAMILY, "unknown", 0, 0, 17)

    def test_all_frozen_capacity_widths_and_counts_match_factory(self) -> None:
        expected_widths = {
            "T": 440,
            "A-mean": 720,
            "A-query": 320,
            "A-temp": 320,
            "AT-concat": 240,
            "AT-gate": 224,
            "AT-cross": 256,
            "AT-cross-randtrain": 256,
        }
        expected_counts = {
            "T": 533721,
            "A-mean": 531361,
            "A-query": 519361,
            "A-temp": 519361,
            "AT-concat": 536161,
            "AT-gate": 529537,
            "AT-cross": 530689,
            "AT-cross-randtrain": 530689,
        }
        self.assertEqual(
            dict(self.frozen.config.capacity_sensitivity.matched_hidden_dims),
            expected_widths,
        )
        self.assertEqual(
            dict(self.frozen.config.capacity_sensitivity.matched_parameter_counts),
            expected_counts,
        )
        for condition in expected_widths:
            built = self.frozen.config.instantiate_smoke_condition(condition)
            self.assertEqual(built.total_parameters, expected_counts[condition])
            self.assertEqual(built.trainable_parameters, expected_counts[condition])

    def test_anchor_conditions_cannot_use_production_constructor(self) -> None:
        for condition in ANCHOR_CONDITIONS:
            with self.assertRaises(CapacityTrainingError):
                self.frozen.config.instantiate_condition(condition)

    def test_primary_anchor_lineage_is_exact_and_complete(self) -> None:
        audit = verify_primary_selection_anchor_lineage(PROJECT_ROOT)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["anchor_coordinate_count"], 50)
        self.assertFalse(audit["primary_outer_index_opened"])
        self.assertFalse(audit["primary_outer_artifacts_opened"])

    def test_phase_a_preflight_denies_every_primary_outer_path_read(self) -> None:
        original_open = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            text = path.resolve().as_posix()
            primary = "/revised_experiments/artifacts/work9_runs_v1/"
            forbidden = (
                text.endswith("/work9_runs_v1/_matrix/outer_ledger_index.json")
                or (primary in text and "/outer_eval/" in text)
                or (primary in text and text.endswith("/run_record.json"))
            )
            if forbidden:
                raise AssertionError(f"Phase A attempted forbidden primary outer read: {text}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=guarded_open):
            receipt = validate_capacity_no_training(PROJECT_ROOT)
        self.assertFalse(receipt["safety"]["primary_outer_index_opened_or_hashed"])
        self.assertFalse(receipt["safety"]["primary_outer_artifacts_opened_or_hashed"])

    def test_no_training_validation_declares_exact_safety_boundary(self) -> None:
        receipt = validate_capacity_no_training(PROJECT_ROOT)
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(
            receipt["hybrid_coordinate_contract"],
            {
                "logical_coordinates": 200,
                "new_fit_coordinates": 150,
                "immutable_primary_anchors": 50,
            },
        )
        self.assertEqual(receipt["safety"]["optimizer_steps"], 0)
        self.assertFalse(receipt["safety"]["training_launched"])
        self.assertFalse(receipt["safety"]["new_capacity_outer_outcomes_accessed"])

    def test_exact_hybrid_runtime_verifier_rejects_duplicate_and_role_swap(self) -> None:
        entries = []
        for condition in NEW_FIT_CONDITIONS + ANCHOR_CONDITIONS:
            family = RUN_FAMILY if condition in NEW_FIT_CONDITIONS else "primary_matrix"
            role = (
                "new_capacity_fit"
                if condition in NEW_FIT_CONDITIONS
                else "immutable_primary_anchor"
            )
            for fold in range(5):
                for seed in SEEDS:
                    entries.append(
                        {
                            "run_id": f"{family}__{condition}__r00-o{fold:02d}-s{seed}",
                            "condition": condition,
                            "partition_repeat": 0,
                            "outer_fold": fold,
                            "seed": seed,
                            "entry_role": role,
                            "source_run_family": family,
                        }
                    )
        self.assertEqual(
            verify_exact_hybrid_entries(
                entries, protocol_id=PROTOCOL_ID, context="synthetic exact hybrid"
            ),
            {"logical": 200, "new": 150, "anchors": 50},
        )
        duplicate = [dict(row) for row in entries]
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaises(CapacityTrainingError):
            verify_exact_hybrid_entries(
                duplicate, protocol_id=PROTOCOL_ID, context="duplicate hybrid"
            )
        swapped = [dict(row) for row in entries]
        swapped[0]["entry_role"] = "immutable_primary_anchor"
        swapped[0]["source_run_family"] = "primary_matrix"
        with self.assertRaises(CapacityTrainingError):
            verify_exact_hybrid_entries(
                swapped, protocol_id=PROTOCOL_ID, context="role-swapped hybrid"
            )

    def test_every_selection_seal_lineage_field_is_rederived(self) -> None:
        test_parent = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        test_parent.mkdir(parents=True, exist_ok=True)
        coordinates = CapacityCoordinates(PROTOCOL_ID, RUN_FAMILY, "A-mean", 0, 0, 17)
        with TemporaryDirectory(prefix="strict-seal-", dir=test_parent) as temporary:
            run_root = Path(temporary) / coordinates.run_id
            selection = run_root / "selection"
            selection.mkdir(parents=True)
            payloads = {
                "checkpoint.safetensors": b"checkpoint",
                "validation_segments.jsonl": b"{}\n",
                "validation_calls.jsonl": b"{}\n",
                "epoch_trace.json": b"{}\n",
                "threshold_trace.json": b"{}\n",
            }
            refs = {}
            for name, payload in payloads.items():
                path = selection / name
                path.write_bytes(payload)
                refs[name] = ArtifactRef(path.resolve(), sha256_file(path))
            view = self.frozen.config.selection_view(0, 0)
            seal = CapacitySelectionSeal(
                schema_version="paper2-v2-work9-capacity-selection-seal-1",
                coordinates=coordinates,
                run_id=coordinates.run_id,
                config_sha256=WORK8_CONFIG_SHA256,
                work8_receipt_sha256=WORK8_RECEIPT_SHA256,
                runner_acceptance_sha256="0" * 64,
                experiment_card_sha256=self.frozen.experiment_card_sha256,
                anchor_policy_sha256=ANCHOR_POLICY_SHA256,
                capacity_analysis_protocol_sha256=CAPACITY_ANALYSIS_PROTOCOL_SHA256,
                execution_addendum_sha256=EXECUTION_ADDENDUM_SHA256,
                primary_runner_acceptance_sha256=PRIMARY_RUNNER_ACCEPTANCE_SHA256,
                primary_environment_lock_sha256=PRIMARY_ENVIRONMENT_LOCK_SHA256,
                train_manifest_sha256=view.train_manifest.sha256,
                validation_manifest_sha256=view.val_manifest.sha256,
                feature_store_metadata_sha256=self.frozen.config.text_store.metadata.sha256,
                feature_store_index_sha256=self.frozen.config.text_store.index_sha256,
                feature_store_vectors_sha256=self.frozen.config.text_store.vectors_sha256,
                pairing_plan_sha256=None,
                pairing_plan_artifact=None,
                selected_epoch=1,
                validation_call_nll=0.5,
                threshold=0.5,
                checkpoint=refs["checkpoint.safetensors"],
                validation_segment_ledger=refs["validation_segments.jsonl"],
                validation_call_ledger=refs["validation_calls.jsonl"],
                epoch_trace=refs["epoch_trace.json"],
                threshold_trace=refs["threshold_trace.json"],
            )
            verify_capacity_selection_seal_against_frozen(
                self.frozen, seal, run_root=run_root
            )
            for field in (
                "config_sha256",
                "work8_receipt_sha256",
                "runner_acceptance_sha256",
                "experiment_card_sha256",
                "anchor_policy_sha256",
                "capacity_analysis_protocol_sha256",
                "execution_addendum_sha256",
                "primary_runner_acceptance_sha256",
                "primary_environment_lock_sha256",
                "train_manifest_sha256",
                "validation_manifest_sha256",
                "feature_store_metadata_sha256",
                "feature_store_index_sha256",
                "feature_store_vectors_sha256",
            ):
                mutated = replace(seal, **{field: "f" * 64})
                with self.subTest(field=field), self.assertRaises(CapacityTrainingError):
                    verify_capacity_selection_seal_against_frozen(
                        self.frozen, mutated, run_root=run_root
                    )

    def test_device_identity_drift_fails_before_training(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device"),
            patch("torch.cuda.get_device_name", return_value="Wrong GPU"),
            patch("torch.cuda.get_device_capability", return_value=(8, 9)),
        ):
            with self.assertRaises(CapacityTrainingError):
                verify_cuda_device_identity("cuda:0")

    def test_environment_drift_fails_frozen_input_verification(self) -> None:
        with patch(
            "paper2_v2.work9_capacity_acceptance.environment_lock",
            return_value={"environment_lock_sha256": "f" * 64},
        ):
            with self.assertRaises(CapacityTrainingError):
                verify_frozen_work9_inputs(
                    PROJECT_ROOT, require_runner_acceptance=False
                )

    def test_anchor_failure_cannot_pollute_capacity_production_registry(self) -> None:
        coordinates = CapacityCoordinates(
            PROTOCOL_ID, RUN_FAMILY, "AT-cross", 0, 0, 17
        )
        run_path = (
            PROJECT_ROOT
            / "revised_experiments/artifacts/work9_capacity_matched_runs_v1"
            / coordinates.run_id
        )
        self.assertFalse(run_path.exists())
        with self.assertRaises(CapacityTrainingError):
            record_coordinate_failure(
                PROJECT_ROOT,
                coordinates,
                phase="fit",
                error=RuntimeError("synthetic hidden-child failure"),
            )
        self.assertFalse(run_path.exists())

    def test_dry_receipt_cannot_authorize_changed_code_bundle(self) -> None:
        payload = {
            **validate_capacity_no_training(PROJECT_ROOT),
            "source_contracts": source_contracts(PROJECT_ROOT),
            "bundle_digests": bundle_digests(PROJECT_ROOT),
        }
        verify_dry_receipt_payload_against_current(PROJECT_ROOT, payload)
        changed = dict(payload)
        changed["bundle_digests"] = dict(payload["bundle_digests"])
        changed["bundle_digests"]["runner_source_bundle_sha256"] = "f" * 64
        with self.assertRaises(Work9AcceptanceError):
            verify_dry_receipt_payload_against_current(PROJECT_ROOT, changed)

    def test_serialized_artifact_references_reject_symlink_components(self) -> None:
        test_parent = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        test_parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="artifact-ref-symlink-", dir=test_parent) as temporary:
            root = Path(temporary)
            real = root / "real.jsonl"
            real.write_text("{}\n", encoding="utf-8")
            alias = root / "alias.jsonl"
            alias.symlink_to(real)
            reference = {
                "path": alias.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(real),
                "bytes": real.stat().st_size,
                "row_count": 1,
            }
            with self.assertRaises(CapacityTrainingError):
                _verify_ref_mapping(
                    reference, PROJECT_ROOT, context="synthetic symlink reference"
                )
            with self.assertRaises(CapacityTrainingError):
                _load_ref(reference, PROJECT_ROOT, ledger=True)
            with self.assertRaises(ProtocolValidationError):
                artifact_from_mapping(reference, PROJECT_ROOT, ledger=True)

    def test_crash_resume_epoch_boundary_is_contiguous_and_early_stop_safe(self) -> None:
        self.assertEqual(
            remaining_epoch_numbers(
                start_epoch=3, epoch_budget=5, stale_epochs=1, patience=4
            ),
            (3, 4, 5),
        )
        self.assertEqual(
            remaining_epoch_numbers(
                start_epoch=3, epoch_budget=5, stale_epochs=4, patience=4
            ),
            (),
        )

    def test_preterminal_crash_resume_rederives_full_outer_record(self) -> None:
        expected = {
            "schema_version": "paper2-v2-work9-capacity-run-record-1",
            "run_id": "synthetic",
            "lineage": {"config_sha256": WORK8_CONFIG_SHA256},
            "data": {"outer_eval_manifest_sha256": "1" * 64},
            "artifacts": {"outer_eval_metrics": {"sha256": "2" * 64}},
            "compute": {
                "device_identity": {"device_name": "frozen-device"},
                "wall_time_seconds": 1.25,
                "peak_cuda_allocated_bytes": 1024,
                "peak_cuda_reserved_bytes": 2048,
                "host_max_rss_kib": 4096,
            },
        }
        arguments = {
            "frozen": self.frozen,
            "coordinates": self.coordinate_for_preterminal_test(),
            "built": object(),
            "outer_data": object(),
            "outer_manifest_sha256": "1" * 64,
            "seal": object(),
            "seal_path": Path("selection_seal.json"),
            "outer_refs": {},
            "segment_count": 1,
            "call_count": 1,
            "device": "cuda:0",
        }
        with patch(
            "paper2_v2.work9_capacity_training._outer_run_record",
            return_value=deepcopy(expected),
        ):
            verify_outer_run_record_before_terminal(
                **arguments, record=deepcopy(expected)
            )
            mutated = deepcopy(expected)
            mutated["lineage"]["config_sha256"] = "f" * 64
            with self.assertRaises(CapacityTrainingError):
                verify_outer_run_record_before_terminal(
                    **arguments, record=mutated
                )

    @staticmethod
    def coordinate_for_preterminal_test() -> CapacityCoordinates:
        return CapacityCoordinates(PROTOCOL_ID, RUN_FAMILY, "A-mean", 0, 0, 17)


if __name__ == "__main__":
    unittest.main()
