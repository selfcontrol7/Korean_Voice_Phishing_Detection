from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper2_v2.work9_registry import Work9Registry
from paper2_v2.work9_types import RUN_FAMILY, Work9Coordinates


class Work9RegistryTests(unittest.TestCase):
    def _registry(self, root: Path) -> Work9Registry:
        return Work9Registry(
            root,
            root=root / "revised_experiments/test_work9_registry",
            allow_test_root=True,
        )

    def _coordinates(self) -> Work9Coordinates:
        return Work9Coordinates("paper2-v2-test", RUN_FAMILY, "AT-cross", 0, 2, 17)

    def test_immutable_write_and_post_terminal_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = self._registry(Path(temporary))
            workspace = registry.reserve(self._coordinates())
            workspace.write_json_once("artifact.json", {"value": 1})
            with self.assertRaises(FileExistsError):
                workspace.write_json_once("artifact.json", {"value": 2})
            workspace.write_json_once("TERMINAL.json", {"status": "done"})
            with self.assertRaisesRegex(RuntimeError, "terminal"):
                workspace.write_json_once("late.json", {})

    def test_same_lineage_outer_claim_resumes_but_changed_lineage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = self._registry(Path(temporary))
            workspace = registry.reserve(self._coordinates())
            values = {
                "selection_seal_sha256": "a" * 64,
                "selection_index_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
            }
            first = registry.claim_outer(workspace, **values)
            second = registry.claim_outer(workspace, **values)
            self.assertEqual(first.sha256, second.sha256)
            with self.assertRaises(FileExistsError):
                registry.claim_outer(
                    workspace,
                    selection_seal_sha256="d" * 64,
                    selection_index_sha256="b" * 64,
                    checkpoint_sha256="c" * 64,
                )

    def test_crash_after_claim_can_only_resume_same_reserved_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = self._registry(Path(temporary))
            coordinates = self._coordinates()
            workspace = registry.reserve(coordinates)
            claim = registry.claim_outer(
                workspace,
                selection_seal_sha256="a" * 64,
                selection_index_sha256="b" * 64,
                checkpoint_sha256="c" * 64,
            )
            # Simulated crash: no outer artifacts exist, but the same
            # deterministic reservation and lineage reuses the claim.
            resumed = registry.reserve(coordinates)
            retry = registry.claim_outer(
                resumed,
                selection_seal_sha256="a" * 64,
                selection_index_sha256="b" * 64,
                checkpoint_sha256="c" * 64,
            )
            self.assertEqual(claim.path, retry.path)
            reservation = json.loads((resumed.path / "reservation.json").read_text())
            self.assertEqual(reservation["run_id"], coordinates.run_id)

    def test_atomic_directory_resume_rejects_extra_or_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = self._registry(Path(temporary))
            workspace = registry.reserve(self._coordinates())
            workspace.publish_directory_once("epoch", {"a.bin": b"a", "b.bin": b"b"})
            workspace.publish_directory_once("epoch", {"a.bin": b"a", "b.bin": b"b"})
            with self.assertRaises(FileExistsError):
                workspace.publish_directory_once("epoch", {"a.bin": b"changed"})
            epoch = workspace.path / "epoch"
            epoch.chmod(0o755)
            (epoch / "extra.bin").write_bytes(b"extra")
            epoch.chmod(0o555)
            with self.assertRaisesRegex(FileExistsError, "inventory differs"):
                workspace.publish_directory_once(
                    "epoch", {"a.bin": b"a", "b.bin": b"b"}
                )

    def test_production_registry_root_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "fixed"):
                Work9Registry(
                    root,
                    root=root / "revised_experiments/alternate",
                )


if __name__ == "__main__":
    unittest.main()
