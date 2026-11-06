from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2_v2.work9_capacity_acceptance import (
    Work9AcceptanceError,
    _load_smoke,
    _load_test_receipt,
)
from paper2_v2.work9_capacity_registry import (
    CapacityRegistry,
    validate_isolated_relative_root,
)
from paper2_v2.work9_capacity_types import (
    CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_CUDA_VERSION,
    EXPECTED_CUDNN_VERSION,
    EXPECTED_DEVICE_NAME,
    EXPECTED_TORCH_VERSION,
    REGISTRY_RELATIVE,
    RUN_FAMILY,
    TEST_SANDBOX_ROOT_RELATIVE,
    UNIT_TEST_RECEIPT_SCHEMA_VERSION,
    CapacityCoordinates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CapacityRegistryTests(unittest.TestCase):
    def coordinate(self) -> CapacityCoordinates:
        return CapacityCoordinates(
            "paper2-v2-mfcc-matched-within-corpus-v1",
            RUN_FAMILY,
            "A-mean",
            0,
            0,
            17,
        )

    def test_canonical_root_is_separate_from_primary(self) -> None:
        capacity = (PROJECT_ROOT / REGISTRY_RELATIVE).resolve()
        primary = (
            PROJECT_ROOT / "revised_experiments/artifacts/work9_runs_v1"
        ).resolve()
        self.assertNotEqual(capacity, primary)
        self.assertEqual(capacity.name, "work9_capacity_matched_runs_v1")

    def test_production_registry_refuses_primary_root(self) -> None:
        primary = PROJECT_ROOT / "revised_experiments/artifacts/work9_runs_v1"
        with self.assertRaises(ValueError):
            CapacityRegistry(PROJECT_ROOT, root=primary)

    def test_test_registry_is_append_only_and_coordinate_bound(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-registry-test-", dir=artifact_root
        ) as temporary:
            registry = CapacityRegistry(
                PROJECT_ROOT,
                root=Path(temporary).relative_to(PROJECT_ROOT),
                allow_test_root=True,
            )
            workspace = registry.reserve(self.coordinate())
            first = workspace.write_json_once("probe.json", {"value": 1})
            self.assertTrue(first.path.is_file())
            with self.assertRaises(FileExistsError):
                workspace.write_json_once("probe.json", {"value": 2})
            reservation = json.loads(
                (workspace.path / "reservation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reservation["run_family"], RUN_FAMILY)
            self.assertEqual(reservation["condition"], "A-mean")

    def test_isolated_roots_reject_absolute_traversal_production_and_frozen(self) -> None:
        for value in (
            PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE / "absolute",
            Path("revised_experiments/artifacts/work9_capacity_test_sandboxes_v1/../escape"),
            Path("revised_experiments/artifacts/work9_runs_v1/child"),
            Path("revised_experiments/artifacts/work9_capacity_matched_runs_v1/child"),
            Path("revised_experiments/src/child"),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_isolated_relative_root(
                    PROJECT_ROOT,
                    value,
                    allowed_parent=TEST_SANDBOX_ROOT_RELATIVE,
                    context="negative test root",
                )

    def test_matrix_path_traversal_and_symlinked_reservation_fail_closed(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-path-test-", dir=artifact_root
        ) as temporary:
            relative = Path(temporary).relative_to(PROJECT_ROOT)
            registry = CapacityRegistry(PROJECT_ROOT, root=relative, allow_test_root=True)
            with self.assertRaises(ValueError):
                registry.write_matrix_artifact_once("../escape.json", {"x": 1})
            target = Path(temporary) / "target"
            target.mkdir()
            run_path = Path(temporary) / self.coordinate().run_id
            run_path.symlink_to(target, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                registry.reserve(self.coordinate())

    def test_outer_claim_same_lineage_resume_and_changed_lineage_conflict(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-claim-test-", dir=artifact_root
        ) as temporary:
            registry = CapacityRegistry(
                PROJECT_ROOT,
                root=Path(temporary).relative_to(PROJECT_ROOT),
                allow_test_root=True,
            )
            workspace = registry.reserve(self.coordinate())
            first = registry.claim_outer(
                workspace,
                selection_seal_sha256="1" * 64,
                selection_index_sha256="2" * 64,
                checkpoint_sha256="3" * 64,
            )
            resumed = registry.claim_outer(
                workspace,
                selection_seal_sha256="1" * 64,
                selection_index_sha256="2" * 64,
                checkpoint_sha256="3" * 64,
            )
            self.assertEqual(first.sha256, resumed.sha256)
            # Simulate a crash after the global claim and before authorization.
            authorization = workspace.write_json_or_verify(
                "outer_authorization.json", {"claim_sha256": first.sha256}
            )
            self.assertTrue(authorization.path.is_file())
            with self.assertRaises(FileExistsError):
                registry.claim_outer(
                    workspace,
                    selection_seal_sha256="1" * 64,
                    selection_index_sha256="2" * 64,
                    checkpoint_sha256="4" * 64,
                )

    def test_workspace_inner_artifact_and_lock_symlinks_fail_closed(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-inner-symlink-test-", dir=artifact_root
        ) as temporary:
            registry = CapacityRegistry(
                PROJECT_ROOT,
                root=Path(temporary).relative_to(PROJECT_ROOT),
                allow_test_root=True,
            )
            workspace = registry.reserve(self.coordinate())
            real_directory = workspace.path / "real-directory"
            real_directory.mkdir()
            (workspace.path / "artifact-alias").symlink_to(
                real_directory, target_is_directory=True
            )
            with self.assertRaises(ValueError):
                workspace.write_json_once("artifact-alias/probe.json", {"x": 1})
            self.assertFalse((real_directory / "probe.json").exists())

            lock_target = workspace.path / "lock-target"
            lock_target.write_text("untouched", encoding="utf-8")
            (workspace.path / "run.lock").symlink_to(lock_target)
            with self.assertRaises(ValueError):
                with workspace.lock():
                    self.fail("symlinked lock must not be acquired")
            self.assertEqual(lock_target.read_text(encoding="utf-8"), "untouched")

    def test_outer_claims_directory_symlink_escape_fails_closed(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(
                prefix="capacity-claim-registry-", dir=artifact_root
            ) as temporary,
            tempfile.TemporaryDirectory(
                prefix="capacity-claim-escape-", dir=artifact_root
            ) as escaped,
        ):
            registry = CapacityRegistry(
                PROJECT_ROOT,
                root=Path(temporary).relative_to(PROJECT_ROOT),
                allow_test_root=True,
            )
            workspace = registry.reserve(self.coordinate())
            claims_link = registry.root / "_outer_eval_claims"
            claims_link.symlink_to(Path(escaped), target_is_directory=True)
            with self.assertRaises(ValueError):
                registry.claim_outer(
                    workspace,
                    selection_seal_sha256="1" * 64,
                    selection_index_sha256="2" * 64,
                    checkpoint_sha256="3" * 64,
                )
            self.assertEqual(list(Path(escaped).iterdir()), [])

    def test_acceptance_loaders_preserve_lexical_paths_and_reject_symlinks(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-acceptance-loader-", dir=artifact_root
        ) as temporary:
            root = Path(temporary)
            allowed_relative = root.relative_to(PROJECT_ROOT)
            actual = root / "actual-smoke"
            actual.mkdir()
            (actual / "smoke_receipt.json").write_text("{}", encoding="utf-8")
            alias = root / "aliased-smoke"
            alias.symlink_to(actual, target_is_directory=True)
            with (
                patch(
                    "paper2_v2.work9_capacity_acceptance.SMOKE_EVIDENCE_ROOT_RELATIVE",
                    allowed_relative,
                ),
                self.assertRaises((ValueError, Work9AcceptanceError)),
            ):
                _load_smoke(
                    (alias / "smoke_receipt.json").relative_to(PROJECT_ROOT),
                    PROJECT_ROOT,
                    draft_sha256="1" * 64,
                    bundles={},
                    environment_lock_sha256="2" * 64,
                )

    def test_unit_test_log_must_be_safe_real_sibling_of_receipt(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-test-log-loader-", dir=artifact_root
        ) as temporary:
            root = Path(temporary)
            allowed_relative = root.relative_to(PROJECT_ROOT)
            outside = root / "outside.log"
            outside.write_text("synthetic log", encoding="utf-8")
            for mode in ("traversal", "symlink"):
                invocation = root / mode
                invocation.mkdir()
                log = invocation / "unittest.log"
                if mode == "symlink":
                    log.symlink_to(outside)
                    log_path = log.relative_to(PROJECT_ROOT).as_posix()
                else:
                    log_path = (
                        f"{invocation.relative_to(PROJECT_ROOT).as_posix()}"
                        "/../outside.log"
                    )
                receipt = invocation / "unit_test_receipt.json"
                receipt.write_text(
                    json.dumps(
                        {
                            "schema_version": UNIT_TEST_RECEIPT_SCHEMA_VERSION,
                            "status": "pass",
                            "tests_passed": 1,
                            "runner_acceptance_draft_sha256": "1" * 64,
                            "bundle_digests": {},
                            "environment_lock_sha256": "2" * 64,
                            "log": {
                                "path": log_path,
                                "sha256": "3" * 64,
                                "bytes": outside.stat().st_size,
                                "row_count": None,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                with (
                    self.subTest(mode=mode),
                    patch(
                        "paper2_v2.work9_capacity_acceptance.UNIT_TEST_EVIDENCE_ROOT_RELATIVE",
                        allowed_relative,
                    ),
                    self.assertRaises((ValueError, Work9AcceptanceError)),
                ):
                    _load_test_receipt(
                        receipt.relative_to(PROJECT_ROOT),
                        PROJECT_ROOT,
                        draft_sha256="1" * 64,
                        bundles={},
                        environment_lock_sha256="2" * 64,
                    )

    def test_smoke_semantic_output_declared_bytes_are_reverified(self) -> None:
        artifact_root = PROJECT_ROOT / TEST_SANDBOX_ROOT_RELATIVE
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capacity-smoke-byte-loader-", dir=artifact_root
        ) as temporary:
            invocation = Path(temporary) / "invocation"
            invocation.mkdir()
            semantic = invocation / "semantic_output.json"
            semantic.write_text("{}\n", encoding="utf-8")
            receipt = invocation / "smoke_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": CUDA_SMOKE_RECEIPT_SCHEMA_VERSION,
                        "status": "pass",
                        "device_type": "cuda",
                        "seed": 17,
                        "pythonhashseed": "17",
                        "cublas_workspace_config": ":4096:8",
                        "condition": "A-mean",
                        "safety": {
                            "outer_eval_accessed": False,
                            "primary_selection_registry_read_only_verified": True,
                            "primary_outer_registry_accessed": False,
                            "primary_registry_modified": False,
                            "scientific_selection_claimed": False,
                        },
                        "semantic_output_sha256": hashlib.sha256(
                            semantic.read_bytes()
                        ).hexdigest(),
                        "semantic_output_bytes": semantic.stat().st_size + 1,
                        "runner_acceptance_draft_sha256": "1" * 64,
                        "bundle_digests": {},
                        "environment_lock_sha256": "2" * 64,
                        "device_name": EXPECTED_DEVICE_NAME,
                        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
                        "torch_version": EXPECTED_TORCH_VERSION,
                        "cuda_version": EXPECTED_CUDA_VERSION,
                        "cudnn_version": EXPECTED_CUDNN_VERSION,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "paper2_v2.work9_capacity_acceptance.SMOKE_EVIDENCE_ROOT_RELATIVE",
                    Path(temporary).relative_to(PROJECT_ROOT),
                ),
                self.assertRaises(Work9AcceptanceError),
            ):
                _load_smoke(
                    receipt.relative_to(PROJECT_ROOT),
                    PROJECT_ROOT,
                    draft_sha256="1" * 64,
                    bundles={},
                    environment_lock_sha256="2" * 64,
                )


if __name__ == "__main__":
    unittest.main()
