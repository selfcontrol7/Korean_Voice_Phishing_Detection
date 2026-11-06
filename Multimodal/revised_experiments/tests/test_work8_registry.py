from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from paper2_v2.config import ArtifactRef, SelectionSeal, sha256_file
from paper2_v2.registry import RunCoordinates, RunHandle, RunRegistry


class Work8RegistryTests(unittest.TestCase):
    def _registry(self, root: Path) -> RunRegistry:
        return RunRegistry(root / "revised_experiments/artifacts/runs")

    def _coordinates(self) -> RunCoordinates:
        return RunCoordinates(
            protocol_id="paper2-v2-mfcc-factorial-v1",
            condition="AT-cross",
            partition_repeat=0,
            outer_fold=2,
            seed=17,
        )

    def test_reservation_and_artifacts_are_exclusive_create(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = self._registry(Path(temporary))
            handle = registry.reserve(self._coordinates(), run_id="fixed-safe-run")
            self.assertTrue((handle.path / "reservation.json").is_file())
            record = handle.write_json_once("metrics.json", {"macro_f1": 0.5})
            self.assertEqual(record.sha256, sha256_file(record.path))
            with self.assertRaises(FileExistsError):
                handle.write_json_once("metrics.json", {"macro_f1": 0.6})
            with self.assertRaises(FileExistsError):
                registry.reserve(self._coordinates(), run_id="fixed-safe-run")
            with self.assertRaises(ValueError):
                handle.write_json_once("../escape.json", {})

    def test_outer_evaluation_authorization_is_single_use_and_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self._registry(root)
            coordinates = self._coordinates()
            handle = registry.reserve(coordinates, run_id="outer-once")
            checkpoint = root / "revised_experiments/artifacts/checkpoints/model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"sealed checkpoint")
            seal = SelectionSeal(
                protocol_id=coordinates.protocol_id,
                partition_repeat=coordinates.partition_repeat,
                outer_fold=coordinates.outer_fold,
                condition=coordinates.condition,
                seed=coordinates.seed,
                checkpoint=ArtifactRef(checkpoint, sha256_file(checkpoint)),
                validation_predictions_sha256=hashlib.sha256(b"val predictions").hexdigest(),
                threshold=0.43,
            )
            registry.claim_outer_evaluation(handle, seal)
            with self.assertRaises(FileExistsError):
                registry.claim_outer_evaluation(handle, seal)
            retry_handle = registry.reserve(coordinates, run_id="outer-retry-new-run-id")
            with self.assertRaises(FileExistsError):
                registry.claim_outer_evaluation(retry_handle, seal)

    def test_outer_authorization_rejects_handle_with_forged_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self._registry(root)
            coordinates = self._coordinates()
            reserved = registry.reserve(coordinates, run_id="reserved-coordinates")
            forged_coordinates = RunCoordinates(
                protocol_id=coordinates.protocol_id,
                condition=coordinates.condition,
                partition_repeat=coordinates.partition_repeat,
                outer_fold=coordinates.outer_fold,
                seed=29,
            )
            forged = RunHandle(reserved.path, forged_coordinates)
            checkpoint = root / "revised_experiments/artifacts/checkpoints/forged.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            seal = SelectionSeal(
                protocol_id=forged_coordinates.protocol_id,
                partition_repeat=forged_coordinates.partition_repeat,
                outer_fold=forged_coordinates.outer_fold,
                condition=forged_coordinates.condition,
                seed=forged_coordinates.seed,
                checkpoint=ArtifactRef(checkpoint, sha256_file(checkpoint)),
                validation_predictions_sha256=hashlib.sha256(b"validation").hexdigest(),
                threshold=0.5,
            )
            with self.assertRaisesRegex(ValueError, "differ from its registry reservation"):
                registry.claim_outer_evaluation(forged, seal)

    def test_terminal_state_is_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            handle = self._registry(Path(temporary)).reserve(
                self._coordinates(), run_id="terminal-once"
            )
            handle.mark_completed({"runs": 1})
            with self.assertRaises(FileExistsError):
                handle.mark_failed({"message": "cannot replace completion"})

    def test_registry_refuses_historical_output_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "revised_experiments"):
                RunRegistry(root / "modeling/logs")
            with self.assertRaisesRegex(ValueError, "historical modeling"):
                RunRegistry(root / "revised_experiments/modeling/runs")


if __name__ == "__main__":
    unittest.main()
