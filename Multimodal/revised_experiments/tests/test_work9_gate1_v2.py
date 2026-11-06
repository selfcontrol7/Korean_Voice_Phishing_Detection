from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper2_v2 import work9_gate1 as v1
from paper2_v2 import work9_gate1_v2 as v2


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_subtree_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(path: Path, root: Path, *, bytes_override: int | None = None) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _digest(path),
        "bytes": path.stat().st_size if bytes_override is None else bytes_override,
        "row_count": None,
    }


def _reference(path: Path, root: Path) -> v1.ImmutableRef:
    return v1.ImmutableRef.from_mapping(
        _mapping(path, root),
        project_root=root,
        context=path.name,
        ledger=False,
    )


def _trace_reference(
    root: Path,
    checkpoint_mapping: dict[str, object],
) -> v1.ImmutableRef:
    trace = root / "epoch_trace.json"
    trace.write_text(
        json.dumps(
            {
                "selected_epoch": 2,
                "epochs": [
                    {"epoch": 1, "checkpoint": checkpoint_mapping},
                    {"epoch": 2, "checkpoint": checkpoint_mapping},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return _reference(trace, root)


class Work9Gate1ValidatorV2Tests(unittest.TestCase):
    def _delegate(
        self,
        *,
        root: Path,
        indexed: v1.ImmutableRef,
        selected_mapping: dict[str, object],
    ) -> tuple[object, dict[str, object]]:
        trace_ref = _trace_reference(root, selected_mapping)
        captured: dict[str, object] = {}

        def frozen_delegate(**kwargs: object) -> tuple[object, dict[str, object]]:
            captured.update(kwargs)
            return object(), {"delegated": True}

        with mock.patch.object(
            v2, "_FROZEN_V1_VALIDATE_EPOCH_TRACE", frozen_delegate
        ):
            result = v2._validate_epoch_trace_v2(
                project_root=root,
                trace_ref=trace_ref,
                entry={"selected_epoch": 2},
                refs={"checkpoint": indexed},
                config=mock.sentinel.config,
                expected_segments={},
                validation_manifest_sha256="0" * 64,
            )
        return result, captured

    def test_distinct_paths_with_identical_verified_content_are_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "condition" / "fold_0" / "seed_17"
            indexed_path = run / "selection" / "checkpoint.safetensors"
            epoch_path = run / "epochs" / "epoch_002" / "model.safetensors"
            indexed_path.parent.mkdir(parents=True)
            epoch_path.parent.mkdir(parents=True)
            indexed_path.write_bytes(b"identical-checkpoint-bytes")
            epoch_path.write_bytes(indexed_path.read_bytes())

            result, captured = self._delegate(
                root=root,
                indexed=_reference(indexed_path, root),
                selected_mapping=_mapping(epoch_path, root),
            )

            self.assertTrue(result[1]["delegated"])
            delegated = captured["refs"]
            self.assertIsInstance(delegated, dict)
            self.assertEqual(delegated["checkpoint"].path, epoch_path)
            self.assertEqual(
                v2.checkpoint_content_identity(delegated["checkpoint"]),
                v2.checkpoint_content_identity(_reference(indexed_path, root)),
            )

    def test_changed_content_with_same_byte_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "condition" / "fold_0" / "seed_17"
            indexed_path = run / "selection" / "checkpoint.safetensors"
            epoch_path = run / "epochs" / "epoch_002" / "model.safetensors"
            indexed_path.parent.mkdir(parents=True)
            epoch_path.parent.mkdir(parents=True)
            indexed_path.write_bytes(b"AAAA")
            epoch_path.write_bytes(b"BBBB")

            with self.assertRaisesRegex(v1.Gate1AnalysisError, "content identity mismatch"):
                self._delegate(
                    root=root,
                    indexed=_reference(indexed_path, root),
                    selected_mapping=_mapping(epoch_path, root),
                )

    def test_declared_byte_count_mismatch_is_rejected_before_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "condition" / "fold_0" / "seed_17"
            indexed_path = run / "selection" / "checkpoint.safetensors"
            epoch_path = run / "epochs" / "epoch_002" / "model.safetensors"
            indexed_path.parent.mkdir(parents=True)
            epoch_path.parent.mkdir(parents=True)
            indexed_path.write_bytes(b"same-checkpoint")
            epoch_path.write_bytes(indexed_path.read_bytes())
            tampered = _mapping(
                epoch_path,
                root,
                bytes_override=epoch_path.stat().st_size + 1,
            )

            with self.assertRaisesRegex(v1.Gate1AnalysisError, "byte-count mismatch"):
                self._delegate(
                    root=root,
                    indexed=_reference(indexed_path, root),
                    selected_mapping=tampered,
                )

    def test_noncanonical_or_cross_run_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_run = root / "runs" / "first"
            second_run = root / "runs" / "second"
            indexed_path = first_run / "selection" / "checkpoint.safetensors"
            epoch_path = second_run / "epochs" / "epoch_002" / "model.safetensors"
            indexed_path.parent.mkdir(parents=True)
            epoch_path.parent.mkdir(parents=True)
            indexed_path.write_bytes(b"identical-checkpoint")
            epoch_path.write_bytes(indexed_path.read_bytes())

            with self.assertRaisesRegex(v1.Gate1AnalysisError, "canonical run root"):
                self._delegate(
                    root=root,
                    indexed=_reference(indexed_path, root),
                    selected_mapping=_mapping(epoch_path, root),
                )

    def test_noncanonical_checkpoint_filenames_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "one"
            indexed_path = run / "selection" / "selected.safetensors"
            epoch_path = run / "epochs" / "epoch_002" / "model.safetensors"
            indexed_path.parent.mkdir(parents=True)
            epoch_path.parent.mkdir(parents=True)
            indexed_path.write_bytes(b"identical-checkpoint")
            epoch_path.write_bytes(indexed_path.read_bytes())

            with self.assertRaisesRegex(v1.Gate1AnalysisError, "canonical selection"):
                self._delegate(
                    root=root,
                    indexed=_reference(indexed_path, root),
                    selected_mapping=_mapping(epoch_path, root),
                )

    def test_content_identity_includes_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"checkpoint")
            digest = _digest(path)
            first = v1.ImmutableRef(path, digest, path.stat().st_size, None, "a.pt")
            second = v1.ImmutableRef(path, digest, path.stat().st_size, 1, "b.pt")
            self.assertNotEqual(
                v2.checkpoint_content_identity(first),
                v2.checkpoint_content_identity(second),
            )

    def test_run_wrapper_restores_frozen_v1_validator(self) -> None:
        expected = {"status": "pass"}
        with mock.patch.object(v1, "run_gate1_analysis", return_value=expected):
            result = v2.run_gate1_analysis(
                project_root=Path("."),
                matrix_config_path=Path("matrix.json"),
                outer_ledger_index_path=Path("outer.json"),
                expected_outer_ledger_index_sha256="0" * 64,
                decision_policy_path=Path("policy.json"),
                analysis_protocol_path=Path("protocol.json"),
                output_dir=Path("output"),
            )
        self.assertIs(result, expected)
        self.assertIs(v1._validate_epoch_trace, v2._FROZEN_V1_VALIDATE_EPOCH_TRACE)

    def test_figure_spec_v2_changes_lineage_only(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_dir = project_root / "revised_experiments" / "configs"
        base_path = config_dir / "work9_gate1_figure_spec_v1.json"
        overlay_path = config_dir / "work9_gate1_figure_spec_v2.json"
        base = json.loads(base_path.read_text(encoding="utf-8"))
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))

        self.assertEqual(overlay["scope"], "lineage_only_supersession")
        self.assertEqual(overlay["supersedes"]["sha256"], _digest(base_path))
        inherited = overlay["inherited_scientific_sections"]
        for section in (
            "figures",
            "excluded_figures",
            "aggregation_rules",
            "export",
            "claim_boundary",
            "data_access",
            "runtime_gate1_binding",
        ):
            self.assertEqual(
                inherited[section]["sha256"],
                _canonical_subtree_digest(base[section]),
                section,
            )
        for binding in overlay["gate1_validator_v2_bindings"]:
            path = project_root / binding["path"]
            # The test file cannot bind its own final hash without recursion;
            # that binding is checked by the external frozen receipt instead.
            if path.resolve() == Path(__file__).resolve():
                continue
            self.assertEqual(binding["sha256"], _digest(path), binding["role"])

    def test_final_figure_addendum_requires_v2_lineage_companion(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        path = (
            project_root
            / "revised_experiments/configs/work9_gate1_figure_lineage_addendum_v2.json"
        )
        addendum = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            addendum["status"], "frozen_before_successful_gate1_v2_outcomes"
        )
        companion = addendum["required_final_companion_receipt"]
        required = set(companion["must_bind_by_path_sha256_bytes"])
        self.assertIn("final v1-rendered figure_manifest.json", required)
        self.assertIn("frozen validator-lineage figure spec v2", required)
        self.assertIn("frozen Gate1 validator v2 acceptance_receipt.json", required)
        self.assertTrue(
            companion["must_assert"][
                "native_figure_manifest_verified_by_frozen_v1_verifier"
            ]
        )
        supersession = addendum["validator_lineage_supersession"]
        supersession_path = project_root / supersession["path"]
        self.assertEqual(supersession["sha256"], _digest(supersession_path))


if __name__ == "__main__":
    unittest.main()
