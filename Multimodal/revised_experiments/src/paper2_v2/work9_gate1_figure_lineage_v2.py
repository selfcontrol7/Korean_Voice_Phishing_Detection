"""Seal and verify the final Work 9 Gate-1 figure lineage companion.

This module never renders or edits a plot.  It verifies preserved v2-native
and v3 bundles with their versioned verifiers, verifies the final v4 bundle
with its strict verifier, verifies the sealed Gate1-v2 directory, and writes
a separate immutable receipt.  V2-native and v3 remain non-citable
superseded lineage; v4 is the sole citable figure bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCHEMA = "paper2-v2-work9-gate1-figure-lineage-receipt-2"
PROTOCOL_ID = "paper2-v2-mfcc-matched-within-corpus-v1"
OUTPUT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_gate1_figures_v2_lineage"
)
GATE1_RELATIVE = Path("revised_experiments/artifacts/work9_gate1_v2")

BASE_SPEC = (
    Path("revised_experiments/configs/work9_gate1_figure_spec_v1.json"),
    "80be3a1294d55677f3393dc34a403475ab68a2c60e8640c7073177ffa6aec080",
)
LINEAGE_SPEC = (
    Path("revised_experiments/configs/work9_gate1_figure_spec_v2.json"),
    "d7832ed7a5d9119fc8a568d3b7ab201b95128447f5a1875b6234cfdfc724c8d8",
)
LINEAGE_ADDENDUM = (
    Path(
        "revised_experiments/configs/work9_gate1_figure_lineage_addendum_v2.json"
    ),
    "1d42c53cf024736165487465c5a45b80d863184035aa3af80c34d882fa3e33b0",
)
VALIDATOR_ACCEPTANCE = (
    Path(
        "revised_experiments/artifacts/work9_gate1_validator_v2_acceptance_v1/"
        "acceptance_receipt.json"
    ),
    "2b78a33bd529de60e58431c9906586a9763a1d1fee03cfd8da9b8f4e4b3faf86",
)
VALIDATOR_ACCEPTANCE_HASHES = (
    Path(
        "revised_experiments/artifacts/work9_gate1_validator_v2_acceptance_v1/"
        "hashes.sha256"
    ),
    "1d0c3803345dfede82b61bb48366646b3ea335cc8b7188ce27e1bdcce712de85",
)
RENDERER_SOURCE = (
    Path("revised_experiments/src/paper2_v2/work9_gate1_figures.py"),
    "87b72b37e93940c05fb3c710cf404a4403b27f7adf5d61ea914fc7a91a9203c0",
)
RENDERER_CLI = (
    Path("revised_experiments/scripts/generate_work9_gate1_figures.py"),
    "17167d91210660e6f8baafe3933dba59e9e20505acf256a4a6b1ca6d937034e3",
)
VALIDATOR_SOURCE = (
    Path("revised_experiments/src/paper2_v2/work9_gate1_v2.py"),
    "f23ac7d6c0265c21a1a9a775d841fe9c3f8e96d6d0235df6759d474bf026e177",
)
VALIDATOR_CLI = (
    Path("revised_experiments/scripts/run_work9_gate1_analysis_v2.py"),
    "bdec0fc5efb4387065db409f7e3efaf001e174e789d7aa255dc09d213320459d",
)

SUPERSEDED_V2_MANIFEST = (
    Path(
        "revised_experiments/artifacts/work9_gate1_figures_v2_native/"
        "figure_manifest.json"
    ),
    "fdcfb0c1ae5f36289e10f44e0fe6e5801e3e24a9bb57ae865c8246ab51f06301",
)
SUPERSEDED_V3_MANIFEST = (
    Path("revised_experiments/artifacts/work9_gate1_figures_v3/figure_manifest.json"),
    "36ecccc4656d61294ea1cdebfdc4eafc364a902d0cd97908f414a04e6be38a9a",
)
FINAL_V4_MANIFEST = (
    Path("revised_experiments/artifacts/work9_gate1_figures_v4/figure_manifest.json"),
    "bd1b01bca131037aa7c7557d5b4bd814594ec5aa8f068632f4538b5596cb49a1",
)
V3_SPEC = (
    Path("revised_experiments/configs/work9_gate1_figure_spec_v3.json"),
    "537c7416f7da0ad93449a5b8d5c8c7b652ccc3a1e4c993175858c2c3dbab6c1a",
)
V3_SOURCE = (
    Path("revised_experiments/src/paper2_v2/work9_gate1_figures_v3.py"),
    "f0c777d1dc0f575940d66f1bccdf5e029c23c3c37e6229520383ddf56eb3f303",
)
V3_CLI = (
    Path("revised_experiments/scripts/generate_work9_gate1_figures_v3.py"),
    "1ae709b7a4f8b0995d0732d8c281d4861040e6c5987b105a5a124854cf70893b",
)
V3_TEST = (
    Path("revised_experiments/tests/test_work9_gate1_figures_v3.py"),
    "b522c1b67a60ae11975624bf86e50639420440c17d230f44d1aced86aefb9a56",
)
V4_SPEC = (
    Path("revised_experiments/configs/work9_gate1_figure_spec_v4.json"),
    "7d8fb582820c83fd021bfdb5f88c7d786376039cbc67854d215f6a73d0941b20",
)
V4_SOURCE = (
    Path("revised_experiments/src/paper2_v2/work9_gate1_figures_v4.py"),
    "170360bb9d797a7f6f427dc3d5551f393ad61398af2df82063c13d7a49a2c7f4",
)
V4_CLI = (
    Path("revised_experiments/scripts/generate_work9_gate1_figures_v4.py"),
    "2649d60a6a40de8c33d4d9dfe62a3b8d01933848272c529691a8b639e0a4e5f6",
)
V4_TEST = (
    Path("revised_experiments/tests/test_work9_gate1_figures_v4.py"),
    "7b50789d7a6f380575909e62db6bc483ece705cfcf97f2065c35c9e886bb0acd",
)

SUPERSESSION_RECEIPT = (
    Path(
        "revised_experiments/artifacts/work9_gate1_figure_lineage_v4/"
        "supersession_receipt.json"
    ),
    "c8cfc3bc5042c6a4c6ea9ff3299eaae77b9cfb105a2d4392672344116d7cf655",
)
SUPERSESSION_HASHES = (
    Path(
        "revised_experiments/artifacts/work9_gate1_figure_lineage_v4/"
        "hashes.sha256"
    ),
    "f937363a3fffa0760e0e5327ed5651edbc896d7747dd8a0f2efe24cac1830043",
)

FIGURE_CHOICES_SHA256 = (
    "e8d62835292bd09c1ac63d59cc72744269ae0efb22a2883bb89c6e1e30ee9013"
)
AGGREGATION_RULES_SHA256 = (
    "6340021cd291ad38a7fc79b56d07c3e4d7ef5a62f824d247cb1a18ddb0d43b4e"
)
GATE1_FILES = frozenset(
    {
        "protocol.json",
        "validation_audit.json",
        "condition_metrics.jsonl",
        "primary_comparisons.jsonl",
        "stability.json",
        "report.json",
        "REPORT.md",
        "READ_ONLY_NOTICE.txt",
    }
)


class FigureLineageError(RuntimeError):
    """Raised when any immutable lineage requirement fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _safe_file(path: Path, root: Path, context: str) -> Path:
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise FigureLineageError(f"{context} escapes project root") from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FigureLineageError(f"{context} traverses a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise FigureLineageError(f"{context} escapes project root") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise FigureLineageError(f"{context} is not a regular file")
    return resolved


def _reference(path: Path, root: Path) -> dict[str, Any]:
    safe = _safe_file(path, root, str(path))
    return {
        "path": safe.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(safe),
        "bytes": safe.stat().st_size,
        "row_count": None,
    }


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FigureLineageError(f"cannot read {context}: {path}") from error
    if not isinstance(value, Mapping):
        raise FigureLineageError(f"{context} must be a JSON object")
    return value


def _require_bound_file(root: Path, binding: tuple[Path, str], role: str) -> Path:
    relative, expected = binding
    path = _safe_file(root / relative, root, role)
    actual = sha256_file(path)
    if actual != expected:
        raise FigureLineageError(
            f"{role} hash mismatch: expected {expected}, got {actual}"
        )
    return path


def _parse_hashes(path: Path, *, allow_project_paths: bool) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[0] != parts[0].lower():
            raise FigureLineageError(f"malformed hash row {number} in {path}")
        digest, name = parts
        try:
            int(digest, 16)
        except ValueError as error:
            raise FigureLineageError(f"invalid SHA-256 at row {number}") from error
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or name in entries:
            raise FigureLineageError(f"unsafe/duplicate hash name at row {number}")
        if not allow_project_paths and len(posix.parts) != 1:
            raise FigureLineageError(f"Gate1 hash row {number} is not a basename")
        entries[name] = digest
    return entries


def _verify_acceptance_hashes(root: Path, manifest: Path) -> None:
    rows = _parse_hashes(manifest, allow_project_paths=True)
    for name, expected in rows.items():
        path = _safe_file(root / PurePosixPath(name), root, f"acceptance hash {name}")
        if sha256_file(path) != expected:
            raise FigureLineageError(f"acceptance hash mismatch for {name}")


def verify_static_lineage(project_root: str | Path) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    bindings = {
        "scientific_base_figure_spec": BASE_SPEC,
        "validator_lineage_figure_spec": LINEAGE_SPEC,
        "lineage_addendum": LINEAGE_ADDENDUM,
        "validator_acceptance_receipt": VALIDATOR_ACCEPTANCE,
        "validator_acceptance_hashes": VALIDATOR_ACCEPTANCE_HASHES,
        "renderer_source": RENDERER_SOURCE,
        "renderer_cli": RENDERER_CLI,
        "validator_source": VALIDATOR_SOURCE,
        "validator_cli": VALIDATOR_CLI,
        "superseded_v2_manifest": SUPERSEDED_V2_MANIFEST,
        "superseded_v3_manifest": SUPERSEDED_V3_MANIFEST,
        "final_v4_manifest": FINAL_V4_MANIFEST,
        "v3_figure_spec": V3_SPEC,
        "v3_renderer_source": V3_SOURCE,
        "v3_renderer_cli": V3_CLI,
        "v3_renderer_test": V3_TEST,
        "v4_figure_spec": V4_SPEC,
        "v4_renderer_source": V4_SOURCE,
        "v4_renderer_cli": V4_CLI,
        "v4_renderer_test": V4_TEST,
        "figure_supersession_receipt": SUPERSESSION_RECEIPT,
        "figure_supersession_hashes": SUPERSESSION_HASHES,
    }
    paths = {
        role: _require_bound_file(root, binding, role)
        for role, binding in bindings.items()
    }
    _verify_acceptance_hashes(root, paths["validator_acceptance_hashes"])
    _verify_acceptance_hashes(root, paths["figure_supersession_hashes"])

    base = _read_json(paths["scientific_base_figure_spec"], "base figure spec")
    lineage = _read_json(paths["validator_lineage_figure_spec"], "lineage spec")
    addendum = _read_json(paths["lineage_addendum"], "lineage addendum")
    acceptance = _read_json(
        paths["validator_acceptance_receipt"], "validator acceptance"
    )
    v3_spec = _read_json(paths["v3_figure_spec"], "v3 figure spec")
    v4_spec = _read_json(paths["v4_figure_spec"], "v4 figure spec")
    supersession = _read_json(
        paths["figure_supersession_receipt"], "figure supersession receipt"
    )
    if (
        lineage.get("status") != "frozen_before_successful_gate1_v2_outcomes"
        or lineage.get("scope") != "lineage_only_supersession"
        or addendum.get("status") != "frozen_before_successful_gate1_v2_outcomes"
        or acceptance.get("status")
        != "accepted_frozen_before_successful_gate1_v2_outcomes"
    ):
        raise FigureLineageError("v2 lineage/acceptance status mismatch")
    if (
        v3_spec.get("schema_version") != "paper2-v2-work9-gate1-figure-spec-3"
        or v4_spec.get("schema_version") != "paper2-v2-work9-gate1-figure-spec-4"
        or v3_spec.get("protocol_id") != PROTOCOL_ID
        or v4_spec.get("protocol_id") != PROTOCOL_ID
    ):
        raise FigureLineageError("v3/v4 figure specification identity mismatch")
    if (
        _canonical_digest(v4_spec.get("figures")) != FIGURE_CHOICES_SHA256
        or _canonical_digest(v4_spec.get("aggregation_rules"))
        != AGGREGATION_RULES_SHA256
        or v4_spec.get("figures") != base.get("figures")
        or v4_spec.get("aggregation_rules") != base.get("aggregation_rules")
    ):
        raise FigureLineageError("v4 scientific figure or aggregation subtree changed")
    if addendum.get("scientific_base_figure_spec") != {
        "path": BASE_SPEC[0].as_posix(),
        "sha256": BASE_SPEC[1],
    } or addendum.get("validator_lineage_supersession") != {
        "path": LINEAGE_SPEC[0].as_posix(),
        "sha256": LINEAGE_SPEC[1],
    }:
        raise FigureLineageError("lineage addendum binding mismatch")
    renderer = addendum.get("renderer_preservation", {})
    if (
        renderer.get("source_sha256") != RENDERER_SOURCE[1]
        or renderer.get("cli_sha256") != RENDERER_CLI[1]
        or renderer.get("modification_permitted") is not False
    ):
        raise FigureLineageError("renderer preservation binding mismatch")
    if _canonical_digest(base.get("figures")) != FIGURE_CHOICES_SHA256:
        raise FigureLineageError("base figure choices digest mismatch")
    if _canonical_digest(base.get("aggregation_rules")) != AGGREGATION_RULES_SHA256:
        raise FigureLineageError("base aggregation rules digest mismatch")
    inherited = lineage.get("inherited_scientific_sections", {})
    if (
        inherited.get("figures", {}).get("sha256") != FIGURE_CHOICES_SHA256
        or inherited.get("aggregation_rules", {}).get("sha256")
        != AGGREGATION_RULES_SHA256
    ):
        raise FigureLineageError("lineage spec scientific digest mismatch")
    accepted = {
        row.get("role"): (row.get("path"), row.get("sha256"))
        for row in acceptance.get("accepted_v2_bindings", [])
        if isinstance(row, Mapping)
    }
    for role, binding in (
        ("checkpoint_identity_validator_v2", VALIDATOR_SOURCE),
        ("gate1_cli_v2", VALIDATOR_CLI),
        ("post_outer_figure_spec_v2_lineage_supersession", LINEAGE_SPEC),
        ("required_final_figure_lineage_companion_addendum_v2", LINEAGE_ADDENDUM),
    ):
        if accepted.get(role) != (binding[0].as_posix(), binding[1]):
            raise FigureLineageError(f"validator acceptance does not bind {role}")

    if (
        supersession.get("schema_version")
        != "paper2-v2-work9-gate1-figure-supersession-receipt-4"
        or supersession.get("status") != "sealed_complete"
        or supersession.get("protocol_id") != PROTOCOL_ID
    ):
        raise FigureLineageError("figure supersession receipt identity/status mismatch")
    expected_states = (
        ("work9_gate1_figures_v2_native", SUPERSEDED_V2_MANIFEST, False, "work9_gate1_figures_v3"),
        ("work9_gate1_figures_v3", SUPERSEDED_V3_MANIFEST, False, "work9_gate1_figures_v4"),
        ("work9_gate1_figures_v4", FINAL_V4_MANIFEST, True, None),
    )
    states = supersession.get("ordered_states")
    if not isinstance(states, list) or len(states) != len(expected_states):
        raise FigureLineageError("figure supersession state family changed")
    for row, (bundle_id, binding, citable, superseded_by) in zip(
        states, expected_states, strict=True
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("bundle_id") != bundle_id
            or row.get("citable") is not citable
            or row.get("superseded_by") != superseded_by
            or row.get("preserved_read_only") is not True
            or not _artifact_ref_matches(
                row.get("manifest"), root / binding[0], root
            )
            or not isinstance(row.get("reasons"), list)
            or not row["reasons"]
        ):
            raise FigureLineageError(f"invalid supersession state for {bundle_id}")
    authority = supersession.get("publication_authority", {})
    if (
        authority.get("only_citable_bundle_id") != "work9_gate1_figures_v4"
        or not _artifact_ref_matches(
            authority.get("only_citable_manifest"), root / FINAL_V4_MANIFEST[0], root
        )
        or authority.get("v2_native_citable") is not False
        or authority.get("v3_citable") is not False
        or authority.get("v4_citable") is not True
        or authority.get("within_corpus_only") is not True
    ):
        raise FigureLineageError("figure publication authority changed")
    invariants = supersession.get("scientific_invariants", {})
    if (
        invariants.get("figure_choices_sha256") != FIGURE_CHOICES_SHA256
        or invariants.get("aggregation_rules_sha256") != AGGREGATION_RULES_SHA256
        or invariants.get("v4_figure_choices_exactly_equal_frozen_v1") is not True
        or invariants.get("v4_aggregation_rules_exactly_equal_frozen_v1") is not True
        or invariants.get("scientific_data_changed") is not False
        or invariants.get("reported_statistics_changed") is not False
        or invariants.get("inference_changed") is not False
        or invariants.get("figure_count") != 11
        or invariants.get("file_count") != 33
    ):
        raise FigureLineageError("figure supersession scientific invariants changed")
    qa = supersession.get("visual_qa", {})
    if (
        qa.get("status") != "publication_quality_pass"
        or qa.get("png_count_reviewed") != 11
        or qa.get("metric_profile_ceiling_clipping_resolved") is not True
        or qa.get("primary_effect_ci_p_label_overlap_resolved") is not True
        or qa.get("probability_density_panel_specific_y_scales_disclosed") is not True
        or qa.get("misleading_seed_or_fold_inference_detected") is not False
        or qa.get("ap_roc_calibration_or_confusion_semantic_blocker_detected")
        is not False
    ):
        raise FigureLineageError("figure supersession visual-QA status changed")
    expected_binding_paths = {
        BASE_SPEC[0],
        RENDERER_SOURCE[0],
        VALIDATOR_ACCEPTANCE[0],
        VALIDATOR_ACCEPTANCE_HASHES[0],
        LINEAGE_SPEC[0],
        LINEAGE_ADDENDUM[0],
        V3_SPEC[0],
        V3_SOURCE[0],
        V3_CLI[0],
        V3_TEST[0],
        V4_SPEC[0],
        V4_SOURCE[0],
        V4_CLI[0],
        V4_TEST[0],
    }
    receipt_binding_paths: set[Path] = set()
    for group in supersession.get("bindings", {}).values():
        if not isinstance(group, list):
            raise FigureLineageError("figure supersession binding group is malformed")
        for reference in group:
            if not isinstance(reference, Mapping):
                raise FigureLineageError("figure supersession reference is malformed")
            relative = Path(str(reference.get("path", "")))
            if not _artifact_ref_matches(reference, root / relative, root):
                raise FigureLineageError("figure supersession reference changed")
            receipt_binding_paths.add(relative)
    if receipt_binding_paths != expected_binding_paths:
        raise FigureLineageError("figure supersession static binding family changed")
    return {
        "paths": paths,
        "base": base,
        "lineage": lineage,
        "addendum": addendum,
        "v3_spec": v3_spec,
        "v4_spec": v4_spec,
        "supersession": supersession,
    }


def _verify_gate1_directory(
    root: Path,
    *,
    gate1_directory: Path,
    expected_hashes_sha256: str,
) -> tuple[Path, dict[str, str]]:
    canonical = (root / GATE1_RELATIVE).resolve()
    directory = (
        gate1_directory.resolve()
        if gate1_directory.is_absolute()
        else (root / gate1_directory).resolve()
    )
    if directory != canonical or not directory.is_dir() or directory.is_symlink():
        raise FigureLineageError("Gate1 directory must be canonical work9_gate1_v2")
    hashes = _safe_file(directory / "hashes.sha256", root, "Gate1 hashes")
    if sha256_file(hashes) != expected_hashes_sha256:
        raise FigureLineageError("Gate1 hashes literal SHA-256 mismatch")
    entries = _parse_hashes(hashes, allow_project_paths=False)
    if set(entries) != GATE1_FILES:
        raise FigureLineageError("Gate1 hashes do not cover the exact sealed file family")
    if directory.stat().st_mode & 0o222 or hashes.stat().st_mode & 0o222:
        raise FigureLineageError("Gate1-v2 directory or hash manifest is writable")
    for name, expected in entries.items():
        path = _safe_file(directory / name, root, f"Gate1 artifact {name}")
        if sha256_file(path) != expected:
            raise FigureLineageError(f"Gate1 artifact hash mismatch: {name}")
        if path.stat().st_mode & 0o222:
            raise FigureLineageError(f"Gate1 artifact is writable: {name}")
    return hashes, entries


def _artifact_ref_matches(reference: Any, path: Path, root: Path) -> bool:
    return isinstance(reference, Mapping) and dict(reference) == _reference(path, root)


def _expected_figure_metadata(base_spec: Mapping[str, Any]) -> list[tuple[Any, Any, Any]]:
    return [
        (row["figure_id"], row["title"], row["analysis_role"])
        for row in base_spec["figures"]
    ]


def _validate_versioned_manifest(
    root: Path,
    *,
    role: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_path: Path,
    expected_schema: str,
    expected_generation_id: str,
    base_spec: Mapping[str, Any],
    verifier: Any,
) -> Mapping[str, Any]:
    manifest_path = _safe_file(manifest_path, root, f"{role} figure manifest")
    if manifest_path != (root / expected_path).resolve():
        raise FigureLineageError(f"{role} figure manifest path is not canonical")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise FigureLineageError(f"{role} figure manifest literal SHA-256 mismatch")
    try:
        manifest = verifier(root, manifest_path)
    except Exception as error:
        raise FigureLineageError(f"frozen {role} figure verifier failed") from error
    if not isinstance(manifest, Mapping):
        raise FigureLineageError(f"{role} verifier did not return a manifest")
    expected_figures = _expected_figure_metadata(base_spec)
    actual_figures = [
        (row.get("figure_id"), row.get("title"), row.get("analysis_role"))
        for row in manifest.get("figures", [])
        if isinstance(row, Mapping)
    ]
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("generation_id") != expected_generation_id
        or manifest.get("figure_count") != len(expected_figures)
        or manifest.get("file_count") != 3 * len(expected_figures)
        or actual_figures != expected_figures
    ):
        raise FigureLineageError(f"{role} manifest identity or figure metadata changed")
    return manifest


def _validate_gate1_bindings(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    gate1_hashes_path: Path,
    gate1_entries: Mapping[str, str],
) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or not _artifact_ref_matches(
        inputs.get("gate1_hash_manifest"), gate1_hashes_path, root
    ):
        raise FigureLineageError("final v4 manifest does not bind sealed work9_gate1_v2 hashes")
    artifacts = inputs.get("gate1_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(gate1_entries):
        raise FigureLineageError("final v4 manifest Gate1 artifact family changed")
    bound_names: set[str] = set()
    for reference in artifacts:
        if not isinstance(reference, Mapping):
            raise FigureLineageError("final v4 manifest has malformed Gate1 reference")
        posix = PurePosixPath(str(reference.get("path", "")))
        if posix.parent != PurePosixPath(GATE1_RELATIVE.as_posix()):
            raise FigureLineageError("final v4 Gate1 path is not canonical work9_gate1_v2")
        name = posix.name
        path = root / posix
        if name not in gate1_entries or not _artifact_ref_matches(reference, path, root):
            raise FigureLineageError("final v4 Gate1 artifact binding mismatch")
        bound_names.add(name)
    if bound_names != set(gate1_entries):
        raise FigureLineageError("final v4 Gate1 artifact set is incomplete")


def _run_v1_verifier(
    project_root: str | Path, manifest_path: str | Path
) -> Mapping[str, Any]:
    from .work9_gate1_figures import verify_figure_manifest

    return verify_figure_manifest(project_root, manifest_path)


def _run_v3_verifier(
    project_root: str | Path, manifest_path: str | Path
) -> Mapping[str, Any]:
    from .work9_gate1_figures_v3 import verify_figure_manifest

    return verify_figure_manifest(project_root, manifest_path)


def _run_v4_verifier(
    project_root: str | Path, manifest_path: str | Path
) -> Mapping[str, Any]:
    from .work9_gate1_figures_v4 import verify_figure_manifest

    return verify_figure_manifest(project_root, manifest_path)


def _copied_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": manifest["protocol_id"],
        "figure_count": manifest["figure_count"],
        "file_count": manifest["file_count"],
        "figures": [
            {
                "figure_id": row["figure_id"],
                "title": row["title"],
                "analysis_role": row["analysis_role"],
            }
            for row in manifest["figures"]
        ],
    }


def seal_figure_lineage(
    project_root: str | Path,
    *,
    final_figure_manifest: str | Path,
    expected_final_manifest_sha256: str,
    gate1_directory: str | Path,
    expected_gate1_hashes_sha256: str,
    output_directory: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    output = (root / OUTPUT_RELATIVE).resolve() if output_directory is None else Path(output_directory).resolve()
    if output != (root / OUTPUT_RELATIVE).resolve():
        raise FigureLineageError("lineage output must use the canonical v2 path")
    if output.exists():
        raise FileExistsError(f"lineage output already exists and is immutable: {output}")
    static = verify_static_lineage(root)
    gate1_hashes, gate1_entries = _verify_gate1_directory(
        root,
        gate1_directory=Path(gate1_directory),
        expected_hashes_sha256=expected_gate1_hashes_sha256,
    )
    v2_manifest = _validate_versioned_manifest(
        root,
        role="superseded v2-native",
        manifest_path=root / SUPERSEDED_V2_MANIFEST[0],
        expected_manifest_sha256=SUPERSEDED_V2_MANIFEST[1],
        expected_path=SUPERSEDED_V2_MANIFEST[0],
        expected_schema="paper2-v2-work9-gate1-figure-manifest-1",
        expected_generation_id="work9_gate1_figures_v1",
        base_spec=static["base"],
        verifier=_run_v1_verifier,
    )
    v3_manifest = _validate_versioned_manifest(
        root,
        role="superseded v3",
        manifest_path=root / SUPERSEDED_V3_MANIFEST[0],
        expected_manifest_sha256=SUPERSEDED_V3_MANIFEST[1],
        expected_path=SUPERSEDED_V3_MANIFEST[0],
        expected_schema="paper2-v2-work9-gate1-figure-manifest-3",
        expected_generation_id="work9_gate1_figures_v3",
        base_spec=static["base"],
        verifier=_run_v3_verifier,
    )
    manifest_path = Path(final_figure_manifest)
    v4_manifest = _validate_versioned_manifest(
        root,
        role="final v4",
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_final_manifest_sha256,
        expected_path=FINAL_V4_MANIFEST[0],
        expected_schema="paper2-v2-work9-gate1-figure-manifest-4",
        expected_generation_id="work9_gate1_figures_v4",
        base_spec=static["base"],
        verifier=_run_v4_verifier,
    )
    if expected_final_manifest_sha256 != FINAL_V4_MANIFEST[1]:
        raise FigureLineageError("final v4 literal SHA-256 differs from frozen identity")
    _validate_gate1_bindings(
        root,
        manifest=v4_manifest,
        gate1_hashes_path=gate1_hashes,
        gate1_entries=gate1_entries,
    )
    v3_lineage = v3_manifest.get("lineage", {})
    v4_lineage = v4_manifest.get("lineage", {})
    if (
        not _artifact_ref_matches(
            v3_lineage.get("superseded_visual_bundle_manifest"),
            root / SUPERSEDED_V2_MANIFEST[0],
            root,
        )
        or v3_lineage.get("superseded_bundle_citable") is not False
        or not _artifact_ref_matches(
            v4_lineage.get("superseded_visual_bundle_manifest"),
            root / SUPERSEDED_V3_MANIFEST[0],
            root,
        )
        or v4_lineage.get("superseded_bundle_citable") is not False
        or v4_lineage.get("v4_bundle_citable") is not True
    ):
        raise FigureLineageError("v2/v3/v4 manifest supersession chain changed")
    paths = static["paths"]
    receipt = {
        "schema_version": SCHEMA,
        "status": "sealed_complete",
        "protocol_id": PROTOCOL_ID,
        "purpose": "Immutable companion binding final citable v4 figures to Gate1 validator-v2 lineage while retaining v2-native and v3 as non-citable superseded evidence.",
        "bindings": {
            "superseded_v2_native_manifest": _reference(paths["superseded_v2_manifest"], root),
            "superseded_v3_manifest": _reference(paths["superseded_v3_manifest"], root),
            "final_citable_v4_manifest": _reference(paths["final_v4_manifest"], root),
            "gate1_v2_hashes": _reference(gate1_hashes, root),
            "scientific_base_figure_spec_v1": _reference(paths["scientific_base_figure_spec"], root),
            "validator_lineage_figure_spec_v2": _reference(paths["validator_lineage_figure_spec"], root),
            "validator_v2_acceptance_receipt": _reference(paths["validator_acceptance_receipt"], root),
            "validator_v2_acceptance_hashes": _reference(paths["validator_acceptance_hashes"], root),
            "lineage_addendum_v2": _reference(paths["lineage_addendum"], root),
            "frozen_renderer_source_v1": _reference(paths["renderer_source"], root),
            "frozen_renderer_cli_v1": _reference(paths["renderer_cli"], root),
            "validator_source_v2": _reference(paths["validator_source"], root),
            "validator_cli_v2": _reference(paths["validator_cli"], root),
            "v3_figure_spec": _reference(paths["v3_figure_spec"], root),
            "v3_renderer_source": _reference(paths["v3_renderer_source"], root),
            "v3_renderer_cli": _reference(paths["v3_renderer_cli"], root),
            "v3_renderer_test": _reference(paths["v3_renderer_test"], root),
            "v4_figure_spec": _reference(paths["v4_figure_spec"], root),
            "v4_renderer_source": _reference(paths["v4_renderer_source"], root),
            "v4_renderer_cli": _reference(paths["v4_renderer_cli"], root),
            "v4_renderer_test": _reference(paths["v4_renderer_test"], root),
            "figure_supersession_receipt_v4": _reference(paths["figure_supersession_receipt"], root),
            "figure_supersession_hashes_v4": _reference(paths["figure_supersession_hashes"], root),
        },
        "copied_from_verified_final_v4_manifest": _copied_metadata(v4_manifest),
        "bundle_disposition": {
            "work9_gate1_figures_v2_native": {"verified": True, "citable": False, "superseded_by": "work9_gate1_figures_v3"},
            "work9_gate1_figures_v3": {"verified": True, "citable": False, "superseded_by": "work9_gate1_figures_v4"},
            "work9_gate1_figures_v4": {"verified": True, "citable": True, "superseded_by": None},
        },
        "canonical_json_digests": {
            "figure_choices": FIGURE_CHOICES_SHA256,
            "aggregation_rules": AGGREGATION_RULES_SHA256,
        },
        "assertions": {
            "native_figure_manifest_verified_by_frozen_v1_verifier": True,
            "gate1_artifact_paths_all_within_work9_gate1_v2": True,
            "figure_choices_changed_after_outcomes": False,
            "aggregation_rules_changed_after_outcomes": False,
            "new_posthoc_figure_added": False,
            "plot_files_modified_by_lineage_binding": False,
        },
        "final_verification": {
            "v1_verifier_passed_for_v2_native": True,
            "v3_verifier_passed": True,
            "v4_verifier_passed": True,
            "v4_is_sole_citable_bundle": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    published = False
    try:
        receipt_path = staging / "lineage_receipt.json"
        notice_path = staging / "READ_ONLY_NOTICE.txt"
        receipt_path.write_bytes(_json_bytes(receipt))
        notice_path.write_text(
            "Immutable Work 9 Gate1-v2 figure-lineage companion. This directory "
            "does not contain or modify plot files. Retain it with final v4; "
            "v2-native and v3 are superseded/non-citable. Do not edit in place.\n",
            encoding="utf-8",
        )
        (staging / "hashes.sha256").write_text(
            "".join(
                f"{sha256_file(staging / name)}  {name}\n"
                for name in ("READ_ONLY_NOTICE.txt", "lineage_receipt.json")
            ),
            encoding="utf-8",
        )
        staging.rename(output)
        published = True
        for path in output.iterdir():
            path.chmod(0o444)
        output.chmod(0o555)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if published and output.exists():
            output.chmod(0o755)
            shutil.rmtree(output, ignore_errors=True)
        raise
    receipt_path = output / "lineage_receipt.json"
    try:
        verify_figure_lineage(root, receipt_path)
    except Exception:
        output.chmod(0o755)
        shutil.rmtree(output, ignore_errors=True)
        raise
    return receipt_path


def verify_figure_lineage(
    project_root: str | Path,
    receipt_path: str | Path,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    expected_receipt = (root / OUTPUT_RELATIVE / "lineage_receipt.json").resolve()
    path = _safe_file(Path(receipt_path), root, "lineage receipt")
    if path != expected_receipt:
        raise FigureLineageError("lineage receipt is not at its canonical path")
    static = verify_static_lineage(root)
    receipt = _read_json(path, "lineage receipt")
    if (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("status") != "sealed_complete"
        or receipt.get("protocol_id") != PROTOCOL_ID
    ):
        raise FigureLineageError("lineage receipt identity/status mismatch")
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise FigureLineageError("lineage receipt bindings are malformed")
    expected_static = {
        "scientific_base_figure_spec_v1": "scientific_base_figure_spec",
        "validator_lineage_figure_spec_v2": "validator_lineage_figure_spec",
        "validator_v2_acceptance_receipt": "validator_acceptance_receipt",
        "validator_v2_acceptance_hashes": "validator_acceptance_hashes",
        "lineage_addendum_v2": "lineage_addendum",
        "frozen_renderer_source_v1": "renderer_source",
        "frozen_renderer_cli_v1": "renderer_cli",
        "validator_source_v2": "validator_source",
        "validator_cli_v2": "validator_cli",
        "superseded_v2_native_manifest": "superseded_v2_manifest",
        "superseded_v3_manifest": "superseded_v3_manifest",
        "final_citable_v4_manifest": "final_v4_manifest",
        "v3_figure_spec": "v3_figure_spec",
        "v3_renderer_source": "v3_renderer_source",
        "v3_renderer_cli": "v3_renderer_cli",
        "v3_renderer_test": "v3_renderer_test",
        "v4_figure_spec": "v4_figure_spec",
        "v4_renderer_source": "v4_renderer_source",
        "v4_renderer_cli": "v4_renderer_cli",
        "v4_renderer_test": "v4_renderer_test",
        "figure_supersession_receipt_v4": "figure_supersession_receipt",
        "figure_supersession_hashes_v4": "figure_supersession_hashes",
    }
    for receipt_name, static_name in expected_static.items():
        if not _artifact_ref_matches(
            bindings.get(receipt_name), static["paths"][static_name], root
        ):
            raise FigureLineageError(f"lineage receipt does not bind {receipt_name}")
    gate1_ref = bindings.get("gate1_v2_hashes")
    if not isinstance(gate1_ref, Mapping):
        raise FigureLineageError("lineage receipt lacks Gate1 hashes binding")
    gate1_hashes = _safe_file(root / str(gate1_ref.get("path", "")), root, "receipt Gate1 hashes")
    if not _artifact_ref_matches(gate1_ref, gate1_hashes, root):
        raise FigureLineageError("lineage receipt Gate1 hashes binding mismatch")
    _, gate1_entries = _verify_gate1_directory(
        root,
        gate1_directory=root / GATE1_RELATIVE,
        expected_hashes_sha256=sha256_file(gate1_hashes),
    )
    v2_manifest = _validate_versioned_manifest(
        root,
        role="superseded v2-native",
        manifest_path=root / SUPERSEDED_V2_MANIFEST[0],
        expected_manifest_sha256=SUPERSEDED_V2_MANIFEST[1],
        expected_path=SUPERSEDED_V2_MANIFEST[0],
        expected_schema="paper2-v2-work9-gate1-figure-manifest-1",
        expected_generation_id="work9_gate1_figures_v1",
        base_spec=static["base"],
        verifier=_run_v1_verifier,
    )
    v3_manifest = _validate_versioned_manifest(
        root,
        role="superseded v3",
        manifest_path=root / SUPERSEDED_V3_MANIFEST[0],
        expected_manifest_sha256=SUPERSEDED_V3_MANIFEST[1],
        expected_path=SUPERSEDED_V3_MANIFEST[0],
        expected_schema="paper2-v2-work9-gate1-figure-manifest-3",
        expected_generation_id="work9_gate1_figures_v3",
        base_spec=static["base"],
        verifier=_run_v3_verifier,
    )
    v4_manifest = _validate_versioned_manifest(
        root,
        role="final v4",
        manifest_path=root / FINAL_V4_MANIFEST[0],
        expected_manifest_sha256=FINAL_V4_MANIFEST[1],
        expected_path=FINAL_V4_MANIFEST[0],
        expected_schema="paper2-v2-work9-gate1-figure-manifest-4",
        expected_generation_id="work9_gate1_figures_v4",
        base_spec=static["base"],
        verifier=_run_v4_verifier,
    )
    del v2_manifest, v3_manifest
    _validate_gate1_bindings(
        root,
        manifest=v4_manifest,
        gate1_hashes_path=gate1_hashes,
        gate1_entries=gate1_entries,
    )
    if receipt.get("copied_from_verified_final_v4_manifest") != _copied_metadata(
        v4_manifest
    ):
        raise FigureLineageError("copied final v4 figure metadata mismatch")
    if receipt.get("canonical_json_digests") != {
        "figure_choices": FIGURE_CHOICES_SHA256,
        "aggregation_rules": AGGREGATION_RULES_SHA256,
    }:
        raise FigureLineageError("lineage receipt scientific digests changed")
    required_assertions = static["addendum"]["required_final_companion_receipt"][
        "must_assert"
    ]
    if receipt.get("assertions") != required_assertions:
        raise FigureLineageError("lineage receipt assertions changed")
    if receipt.get("bundle_disposition") != {
        "work9_gate1_figures_v2_native": {
            "verified": True,
            "citable": False,
            "superseded_by": "work9_gate1_figures_v3",
        },
        "work9_gate1_figures_v3": {
            "verified": True,
            "citable": False,
            "superseded_by": "work9_gate1_figures_v4",
        },
        "work9_gate1_figures_v4": {
            "verified": True,
            "citable": True,
            "superseded_by": None,
        },
    }:
        raise FigureLineageError("lineage receipt bundle disposition changed")
    if receipt.get("final_verification") != {
        "v1_verifier_passed_for_v2_native": True,
        "v3_verifier_passed": True,
        "v4_verifier_passed": True,
        "v4_is_sole_citable_bundle": True,
    }:
        raise FigureLineageError("lineage receipt final verification changed")
    hashes_path = _safe_file(path.parent / "hashes.sha256", root, "lineage hashes")
    rows = _parse_hashes(hashes_path, allow_project_paths=False)
    if set(rows) != {"READ_ONLY_NOTICE.txt", "lineage_receipt.json"}:
        raise FigureLineageError("lineage hash manifest file family changed")
    for name, expected in rows.items():
        item = _safe_file(path.parent / name, root, f"lineage artifact {name}")
        if sha256_file(item) != expected or item.stat().st_mode & 0o222:
            raise FigureLineageError(f"lineage artifact integrity/mode mismatch: {name}")
    if hashes_path.stat().st_mode & 0o222 or path.parent.stat().st_mode & 0o222:
        raise FigureLineageError("lineage companion is writable")
    return receipt


__all__ = [
    "FigureLineageError",
    "seal_figure_lineage",
    "sha256_file",
    "verify_figure_lineage",
    "verify_static_lineage",
]
