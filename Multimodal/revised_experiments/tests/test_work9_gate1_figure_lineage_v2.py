from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

from paper2_v2 import work9_gate1_figure_lineage_v2 as lineage


REAL_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _reference(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": lineage.sha256_file(path),
        "bytes": path.stat().st_size,
        "row_count": None,
    }


def _copy_static_lineage(root: Path) -> None:
    bindings = (
        lineage.BASE_SPEC,
        lineage.LINEAGE_SPEC,
        lineage.LINEAGE_ADDENDUM,
        lineage.VALIDATOR_ACCEPTANCE,
        lineage.VALIDATOR_ACCEPTANCE_HASHES,
        lineage.RENDERER_SOURCE,
        lineage.RENDERER_CLI,
        lineage.VALIDATOR_SOURCE,
        lineage.VALIDATOR_CLI,
        lineage.SUPERSEDED_V2_MANIFEST,
        lineage.SUPERSEDED_V3_MANIFEST,
        lineage.FINAL_V4_MANIFEST,
        lineage.V3_SPEC,
        lineage.V3_SOURCE,
        lineage.V3_CLI,
        lineage.V3_TEST,
        lineage.V4_SPEC,
        lineage.V4_SOURCE,
        lineage.V4_CLI,
        lineage.V4_TEST,
        lineage.SUPERSESSION_RECEIPT,
        lineage.SUPERSESSION_HASHES,
    )
    paths = {binding[0] for binding in bindings}
    for hash_binding in (
        lineage.VALIDATOR_ACCEPTANCE_HASHES,
        lineage.SUPERSESSION_HASHES,
    ):
        source = REAL_ROOT / hash_binding[0]
        for row in source.read_text(encoding="utf-8").splitlines():
            _, name = row.split("  ", 1)
            paths.add(Path(name))
    for relative in paths:
        source = REAL_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _gate1_fixture(root: Path) -> tuple[Path, Path, dict[str, str]]:
    directory = root / lineage.GATE1_RELATIVE
    directory.mkdir(parents=True)
    rows = []
    entries: dict[str, str] = {}
    for name in sorted(lineage.GATE1_FILES):
        path = directory / name
        path.write_text(f"synthetic-{name}\n", encoding="utf-8")
        path.chmod(0o444)
        digest = lineage.sha256_file(path)
        entries[name] = digest
        rows.append(f"{digest}  {name}\n")
    hashes = directory / "hashes.sha256"
    hashes.write_text("".join(rows), encoding="utf-8")
    hashes.chmod(0o444)
    directory.chmod(0o555)
    return directory, hashes, entries


def _read_manifest(_: str | Path, manifest_path: str | Path) -> dict[str, object]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


@contextmanager
def _mock_all_versioned_verifiers() -> object:
    with ExitStack() as stack:
        for name in ("_run_v1_verifier", "_run_v3_verifier", "_run_v4_verifier"):
            stack.enter_context(
                mock.patch.object(lineage, name, side_effect=_read_manifest)
            )
        yield


def _synthetic_manifest(
    root: Path,
    *,
    path: Path,
    schema: str,
    generation_id: str,
    changed_title: bool = False,
) -> Path:
    base = json.loads((root / lineage.BASE_SPEC[0]).read_text(encoding="utf-8"))
    figures = [
        {
            "figure_id": row["figure_id"],
            "title": row["title"],
            "analysis_role": row["analysis_role"],
            "files": [],
        }
        for row in base["figures"]
    ]
    if changed_title:
        figures[0]["title"] = "posthoc changed title"
    _write_json(
        path,
        {
            "schema_version": schema,
            "status": "sealed_complete",
            "protocol_id": lineage.PROTOCOL_ID,
            "generation_id": generation_id,
            "figure_count": 11,
            "file_count": 33,
            "figures": figures,
        },
    )
    return path


class Work9Gate1FigureLineageV2Tests(unittest.TestCase):
    def test_cli_reports_final_v4_metadata_key(self) -> None:
        cli = (
            REAL_ROOT
            / "revised_experiments/scripts/"
            "seal_work9_gate1_figure_lineage_v2.py"
        ).read_text(encoding="utf-8")
        self.assertIn('receipt["copied_from_verified_final_v4_manifest"]', cli)
        self.assertNotIn('receipt["copied_from_verified_native_manifest"]', cli)

    def _fixture(self, temporary: str) -> tuple[Path, Path, Path, dict[str, str]]:
        root = Path(temporary)
        _copy_static_lineage(root)
        gate1, hashes, entries = _gate1_fixture(root)
        return root, gate1, hashes, entries

    def test_current_static_lineage_binds_only_citable_v4(self) -> None:
        verified = lineage.verify_static_lineage(REAL_ROOT)
        self.assertEqual(verified["lineage"]["scope"], "lineage_only_supersession")
        self.assertEqual(
            verified["supersession"]["publication_authority"][
                "only_citable_bundle_id"
            ],
            "work9_gate1_figures_v4",
        )
        self.assertEqual(
            lineage._canonical_digest(verified["v4_spec"]["figures"]),
            lineage.FIGURE_CHOICES_SHA256,
        )
        self.assertEqual(
            lineage._canonical_digest(verified["v4_spec"]["aggregation_rules"]),
            lineage.AGGREGATION_RULES_SHA256,
        )

    def test_synthetic_gate1_seals_and_reverifies_read_only_companion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, gate1, hashes, _ = self._fixture(temporary)
            final_manifest = root / lineage.FINAL_V4_MANIFEST[0]
            with _mock_all_versioned_verifiers(), mock.patch.object(
                lineage, "_validate_gate1_bindings", return_value=None
            ):
                receipt = lineage.seal_figure_lineage(
                    root,
                    final_figure_manifest=final_manifest,
                    expected_final_manifest_sha256=lineage.FINAL_V4_MANIFEST[1],
                    gate1_directory=gate1,
                    expected_gate1_hashes_sha256=lineage.sha256_file(hashes),
                )
                verified = lineage.verify_figure_lineage(root, receipt)
                with self.assertRaises(FileExistsError):
                    lineage.seal_figure_lineage(
                        root,
                        final_figure_manifest=final_manifest,
                        expected_final_manifest_sha256=lineage.FINAL_V4_MANIFEST[1],
                        gate1_directory=gate1,
                        expected_gate1_hashes_sha256=lineage.sha256_file(hashes),
                    )
            self.assertEqual(verified["schema_version"], lineage.SCHEMA)
            self.assertTrue(verified["bundle_disposition"]["work9_gate1_figures_v4"]["citable"])
            self.assertFalse(verified["bundle_disposition"]["work9_gate1_figures_v3"]["citable"])
            self.assertEqual(receipt.parent.stat().st_mode & 0o777, 0o555)
            self.assertTrue(
                all(
                    path.stat().st_mode & 0o777 == 0o444
                    for path in receipt.parent.iterdir()
                )
            )

    def test_final_v4_verifier_failure_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, gate1, hashes, _ = self._fixture(temporary)
            with mock.patch.object(
                lineage, "_run_v1_verifier", side_effect=_read_manifest
            ), mock.patch.object(
                lineage, "_run_v3_verifier", side_effect=_read_manifest
            ), mock.patch.object(
                lineage,
                "_run_v4_verifier",
                side_effect=RuntimeError("synthetic v4 rejection"),
            ):
                with self.assertRaisesRegex(
                    lineage.FigureLineageError, "final v4 figure verifier"
                ):
                    lineage.seal_figure_lineage(
                        root,
                        final_figure_manifest=root / lineage.FINAL_V4_MANIFEST[0],
                        expected_final_manifest_sha256=lineage.FINAL_V4_MANIFEST[1],
                        gate1_directory=gate1,
                        expected_gate1_hashes_sha256=lineage.sha256_file(hashes),
                    )
            self.assertFalse((root / lineage.OUTPUT_RELATIVE).exists())
            self.assertEqual(
                list((root / lineage.OUTPUT_RELATIVE.parent).glob(".*.staging-*")), []
            )

    def test_changed_v4_figure_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "synthetic/final_v4/figure_manifest.json"
            _copy_static_lineage(root)
            manifest = _synthetic_manifest(
                root,
                path=destination,
                schema="paper2-v2-work9-gate1-figure-manifest-4",
                generation_id="work9_gate1_figures_v4",
                changed_title=True,
            )
            with self.assertRaisesRegex(lineage.FigureLineageError, "figure metadata"):
                lineage._validate_versioned_manifest(
                    root,
                    role="synthetic final v4",
                    manifest_path=manifest,
                    expected_manifest_sha256=lineage.sha256_file(manifest),
                    expected_path=manifest.relative_to(root),
                    expected_schema="paper2-v2-work9-gate1-figure-manifest-4",
                    expected_generation_id="work9_gate1_figures_v4",
                    base_spec=json.loads(
                        (root / lineage.BASE_SPEC[0]).read_text(encoding="utf-8")
                    ),
                    verifier=_read_manifest,
                )

    def test_gate1_binding_requires_exact_v2_paths_and_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate1, hashes, entries = _gate1_fixture(root)
            manifest = {
                "inputs": {
                    "gate1_hash_manifest": _reference(hashes, root),
                    "gate1_artifacts": [
                        _reference(gate1 / name, root) for name in sorted(entries)
                    ],
                }
            }
            lineage._validate_gate1_bindings(
                root,
                manifest=manifest,
                gate1_hashes_path=hashes,
                gate1_entries=entries,
            )
            manifest["inputs"]["gate1_artifacts"][0]["path"] = (
                "revised_experiments/artifacts/work9_gate1_v1/REPORT.md"
            )
            with self.assertRaisesRegex(lineage.FigureLineageError, "not canonical"):
                lineage._validate_gate1_bindings(
                    root,
                    manifest=manifest,
                    gate1_hashes_path=hashes,
                    gate1_entries=entries,
                )


if __name__ == "__main__":
    unittest.main()
