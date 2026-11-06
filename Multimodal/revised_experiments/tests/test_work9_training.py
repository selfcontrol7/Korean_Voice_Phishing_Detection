from __future__ import annotations

import json
import importlib.metadata
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from paper2_v2.config import load_matrix_config
from paper2_v2.data import FeatureSample, collate_samples
from paper2_v2.evaluation import CallPrediction, SegmentPrediction
from paper2_v2.work9_acceptance import (
    Work9AcceptanceError,
    _base_payload,
    _distribution_package_records,
    environment_lock,
    verify_environment_lock_mapping,
)
from paper2_v2.work9_training import (
    Work9TrainingError,
    _build_pairing,
    _load_ref,
    _prediction_rows,
    _verify_completed_selection_run,
    _verify_ref_mapping,
    _verify_runner_acceptance_environment,
    equal_call_batch_loss,
    remaining_epoch_numbers,
    reset_cuda_peak_memory_stats,
    require_process_determinism,
    threshold_trace_payload,
    validate_json_schema,
)
from paper2_v2.work9_types import ACCEPTANCE_SCHEMA_VERSION, RUN_FAMILY, Work9Coordinates
from paper2_v2.work9_registry import Work9Registry


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "revised_experiments/configs/work8_mfcc_matched_within_corpus_v1.json"


def config():
    return load_matrix_config(
        CONFIG_PATH,
        project_root=ROOT,
        verify_files=False,
        verify_manifest_isolation=False,
    )


class Work9TrainingTests(unittest.TestCase):
    def test_acceptance_v3_binds_both_superseded_failures(self) -> None:
        payload = _base_payload(ROOT)
        self.assertEqual(
            payload["schema_version"], ACCEPTANCE_SCHEMA_VERSION
        )
        self.assertEqual(
            payload["supersedes"]["current_acceptance_generation"],
            "work9_runner_acceptance_v3",
        )
        generations = payload["supersedes"]["generations"]
        self.assertEqual(
            [row["acceptance_generation"] for row in generations],
            ["work9_runner_acceptance_v1", "work9_runner_acceptance_v2"],
        )
        failure = generations[0]["failure"]
        self.assertEqual(
            failure["path"],
            "revised_experiments/artifacts/work9_runner_acceptance_v1/"
            "PREFLIGHT_FAILURE.json",
        )
        self.assertEqual(len(failure["sha256"]), 64)
        self.assertEqual(
            generations[1]["failure"]["path"],
            "revised_experiments/artifacts/work9_runner_acceptance_v2/"
            "POST_SELECTION_FAILURE.json",
        )

    def test_environment_lock_ignores_runtime_sys_path_contamination(self) -> None:
        baseline = environment_lock()
        with tempfile.TemporaryDirectory() as temporary:
            contaminant = Path(temporary) / "fake_contaminant-1.0.dist-info"
            contaminant.mkdir()
            (contaminant / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: fake-contaminant\nVersion: 1.0\n",
                encoding="utf-8",
            )
            with mock.patch.object(sys, "path", [*sys.path, temporary]):
                ambient = {
                    distribution.metadata.get("Name")
                    for distribution in importlib.metadata.distributions()
                    if distribution.metadata.get("Name")
                }
                contaminated = environment_lock()
        self.assertIn("fake-contaminant", ambient)
        self.assertEqual(contaminated, baseline)
        self.assertEqual(
            baseline["inventory_method"],
            "importlib_metadata_explicit_sysconfig_purelib_platlib_record_hashes_v1",
        )
        self.assertEqual(len(baseline["distribution_roots"]), 1)

    def test_environment_distribution_records_fail_closed_and_sort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def distribution(directory: str, name: object, version: object):
                metadata = root / directory
                metadata.mkdir()
                (metadata / "METADATA").write_text(
                    f"Name: {name}\nVersion: {version}\n", encoding="utf-8"
                )
                (metadata / "RECORD").write_text("METADATA,,\n", encoding="utf-8")
                return SimpleNamespace(
                    metadata={} if name is None else {"Name": name},
                    version=version,
                    _path=metadata,
                )

            alpha = distribution("alpha-1.dist-info", "Alpha_Name", "1")
            beta = distribution("beta-1.dist-info", "beta-name", "1")
            first = _distribution_package_records([beta, alpha], [str(root)])
            second = _distribution_package_records([alpha, beta], [str(root)])
            self.assertEqual(first, second)
            self.assertEqual([row["name"] for row in first], ["alpha-name", "beta-name"])
            missing = distribution("missing-1.dist-info", None, "1")
            with self.assertRaisesRegex(Work9AcceptanceError, "valid Name"):
                _distribution_package_records([missing], [str(root)])
            duplicate = distribution("duplicate-1.dist-info", "alpha-name", "2")
            with self.assertRaisesRegex(Work9AcceptanceError, "duplicate"):
                _distribution_package_records([alpha, duplicate], [str(root)])
            invalid_version = distribution("invalid-1.dist-info", "valid", "")
            with self.assertRaisesRegex(Work9AcceptanceError, "version"):
                _distribution_package_records([invalid_version], [str(root)])

    def test_environment_lock_self_hash_and_acceptance_bindings_fail_closed(self) -> None:
        lock = environment_lock()
        self.assertEqual(
            verify_environment_lock_mapping(lock), lock["environment_lock_sha256"]
        )
        acceptance = {
            "environment_lock": lock,
            "cuda_smokes": [
                {"environment_lock_sha256": lock["environment_lock_sha256"]},
                {"environment_lock_sha256": lock["environment_lock_sha256"]},
            ],
            "unit_tests": {
                "environment_lock_sha256": lock["environment_lock_sha256"]
            },
        }
        self.assertEqual(
            _verify_runner_acceptance_environment(acceptance),
            lock["environment_lock_sha256"],
        )
        mutated_lock = json.loads(json.dumps(lock))
        mutated_lock["platform"] = "changed"
        with self.assertRaisesRegex(Work9AcceptanceError, "self-hash"):
            verify_environment_lock_mapping(mutated_lock)
        mismatched_current = json.loads(json.dumps(lock))
        mismatched_current["platform"] = "different-current-platform"
        with mock.patch(
            "paper2_v2.work9_acceptance.environment_lock",
            return_value=mismatched_current,
        ):
            with self.assertRaisesRegex(Work9TrainingError, "current environment"):
                _verify_runner_acceptance_environment(acceptance)
        acceptance["unit_tests"]["environment_lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(Work9TrainingError, "unit tests"):
            _verify_runner_acceptance_environment(acceptance)

    def test_equal_call_fixed_global_scale_has_equal_epoch_coefficients(self) -> None:
        logits = torch.zeros(4, requires_grad=True)
        labels = torch.zeros(4)
        weights = torch.tensor([1.0, 1 / 3, 1 / 3, 1 / 3])
        loss, elements = equal_call_batch_loss(
            logits,
            labels,
            weights,
            segment_count=4,
            call_count=2,
        )
        self.assertAlmostEqual(float(loss), math.log(2.0), places=6)
        self.assertAlmostEqual(float((weights * elements).sum() / 2), math.log(2.0), places=6)
        loss.backward()
        # Each call contributes exactly half of the objective/gradient even
        # though one has one segment and the other has three.
        self.assertAlmostEqual(float(logits.grad[0]), float(logits.grad[1:].sum()), places=7)

    def test_persisted_early_stop_state_never_trains_an_extra_epoch(self) -> None:
        self.assertEqual(
            remaining_epoch_numbers(
                start_epoch=9, epoch_budget=50, stale_epochs=8, patience=8
            ),
            (),
        )
        self.assertEqual(
            remaining_epoch_numbers(
                start_epoch=9, epoch_budget=10, stale_epochs=7, patience=8
            ),
            (9, 10),
        )

    def test_crash_after_selection_directory_is_not_false_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = Work9Registry(
                root,
                root=root / "revised_experiments/runs",
                allow_test_root=True,
            )
            coordinates = Work9Coordinates(
                "paper2-v2-test", RUN_FAMILY, "T", 0, 0, 17
            )
            workspace = registry.reserve(coordinates)
            (workspace.path / "selection").mkdir()
            (workspace.path / "selection/selection_seal.json").write_text(
                "{}\n", encoding="utf-8"
            )
            frozen = SimpleNamespace(project_root=root)
            self.assertIsNone(_verify_completed_selection_run(frozen, workspace))

    def test_all_eight_conditions_complete_forward_backward(self) -> None:
        matrix = config()
        for condition in matrix.conditions:
            with self.subTest(condition=condition):
                torch.manual_seed(1)
                built = matrix.instantiate_condition(condition)
                samples = []
                for index, length in enumerate((3, 5)):
                    samples.append(
                        FeatureSample(
                            sequences={"mfcc": torch.randn(length, 13)},
                            globals={"text": torch.randn(768)},
                            label=float(index),
                            sample_weight=1.0,
                            call_id=f"c{index}",
                            segment_id=f"s{index}",
                            group_id=f"g{index}",
                            source="source",
                            start=0.0,
                            end=1.0,
                        )
                    )
                batch = collate_samples(samples, built.input_spec)
                logits = built.model(batch)
                loss, _ = equal_call_batch_loss(
                    logits,
                    batch.labels,
                    batch.sample_weights,
                    segment_count=2,
                    call_count=2,
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(
                    all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in built.model.parameters()
                    )
                )

    def test_prediction_timing_uses_exact_manifest_precision(self) -> None:
        start = 0.1234567890123
        end = 1.9876543210987
        prediction = SegmentPrediction(
            "s", "c", "g", 1, 0.75, start, end, "source"
        )
        call = CallPrediction("c", "g", 1, 0.75, 1, start, end, "source")
        coordinates = Work9Coordinates("paper2-v2-test", RUN_FAMILY, "T", 0, 0, 17)
        segments, calls = _prediction_rows(
            [prediction],
            [call],
            coordinates=coordinates,
            partition="validation",
            threshold=0.5,
            checkpoint_sha256="a" * 64,
            manifest_sha256="b" * 64,
        )
        self.assertEqual(segments[0]["start"], start)
        self.assertEqual(segments[0]["end"], end)
        self.assertEqual(segments[0]["duration_seconds"], end - start)
        self.assertEqual(calls[0]["duration_seconds"], end - start)
        self.assertIn("hard_prediction", calls[0])
        self.assertNotIn("prediction", calls[0])

    def test_threshold_trace_has_locked_99_candidate_shape(self) -> None:
        matrix = config()
        coordinates = Work9Coordinates(
            matrix.protocol_id, RUN_FAMILY, "AT-cross", 0, 0, 17
        )
        calls = [
            CallPrediction("c0", "g0", 0, 0.2, 1, 0.0, 1.0, "AIHub"),
            CallPrediction("c1", "g1", 1, 0.8, 1, 0.0, 1.0, "FSS"),
        ]
        payload = threshold_trace_payload(
            calls,
            config=matrix,
            coordinates=coordinates,
            validation_call_ledger={
                "path": "x.jsonl",
                "sha256": "a" * 64,
                "bytes": 1,
                "row_count": 2,
            },
        )
        self.assertEqual(len(payload["candidates"]), 99)
        expected_metric_keys = {
            "n_calls",
            "confusion",
            *matrix.statistics.reported_call_metrics,
        }
        self.assertEqual(set(payload["candidates"][0]["metrics"]), expected_metric_keys)
        validate_json_schema(
            payload,
            project_root=ROOT,
            schema_name="work9_threshold_trace.schema.json",
        )

    def test_mismatched_frozen_pairing_digest_is_rejected(self) -> None:
        rows = [
            {"segment_id": "s0", "call_id": "c0", "group_id": "g0"},
            {"segment_id": "s1", "call_id": "c0", "group_id": "g0"},
            {"segment_id": "s2", "call_id": "c1", "group_id": "g1"},
            {"segment_id": "s3", "call_id": "c1", "group_id": "g1"},
        ]
        coordinates = Work9Coordinates(
            "paper2-v2-test", RUN_FAMILY, "AT-cross-randtrain", 0, 0, 17
        )
        receipt = {
            "random_train_pairing_plans": [
                {
                    "condition": coordinates.condition,
                    "partition_repeat": 0,
                    "outer_fold": 0,
                    "seed": 17,
                    "train_manifest_sha256": "b" * 64,
                    "plan_sha256": "a" * 64,
                }
            ]
        }
        with self.assertRaisesRegex(Work9TrainingError, "digest differs"):
            _build_pairing(
                coordinates,
                rows,
                receipt=receipt,
                train_sha256="b" * 64,
            )

    def test_reference_key_and_path_guards_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "value.json"
            file.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Work9TrainingError, "missing path/sha256"):
                _verify_ref_mapping({"path": str(file), "extra": 1}, root, context="x")
            with self.assertRaisesRegex(Work9TrainingError, "unexpected"):
                _verify_ref_mapping(
                    {
                        "path": str(file),
                        "sha256": "a" * 64,
                        "surprise": True,
                    },
                    root,
                    context="x",
                )
            with self.assertRaisesRegex(Work9TrainingError, "relative and safe"):
                _load_ref(
                    {
                        "path": "../escape",
                        "sha256": "a" * 64,
                        "bytes": 1,
                        "row_count": None,
                    },
                    root,
                    ledger=False,
                )

    def test_cuda_preflight_requires_process_environment(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(Work9TrainingError, "PYTHONHASHSEED"):
                require_process_determinism(17, "cuda:0")
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "17"}, clear=True):
            with self.assertRaisesRegex(Work9TrainingError, "CUBLAS"):
                require_process_determinism(17, "cuda:0")
        with mock.patch.dict(
            os.environ,
            {"PYTHONHASHSEED": "17", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            clear=True,
        ):
            require_process_determinism(17, "cuda:0")

    def test_cuda_telemetry_selects_device_before_reset(self) -> None:
        events: list[tuple[str, object]] = []
        with mock.patch.object(
            torch.cuda,
            "set_device",
            side_effect=lambda device: events.append(("set", device)),
        ), mock.patch.object(
            torch.cuda,
            "reset_peak_memory_stats",
            side_effect=lambda device: events.append(("reset", device)),
        ):
            reset_cuda_peak_memory_stats("cuda:0")
            reset_cuda_peak_memory_stats("cpu")
        self.assertEqual(
            events,
            [("set", torch.device("cuda:0")), ("reset", torch.device("cuda:0"))],
        )

    def test_all_work9_schemas_use_exact_four_field_artifact_refs(self) -> None:
        schema_root = ROOT / "revised_experiments/registry/schemas/work9"
        for path in schema_root.glob("*.schema.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            artifact = raw.get("$defs", {}).get("artifact")
            if artifact is None:
                continue
            self.assertEqual(
                set(artifact["required"]), {"path", "sha256", "bytes", "row_count"}
            )
            self.assertFalse(artifact["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
