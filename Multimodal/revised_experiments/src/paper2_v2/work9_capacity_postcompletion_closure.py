"""Read-only post-completion closure for the capacity sensitivity package.

The accepted pre-outcome runner intentionally remains immutable.  Its source
inventory used a dynamic schema-directory glob, so two immutable post-outcome
figure schemas now produce an expected historical-boundary error.  This
versioned closure proves that every one of the 38 accepted contracts remains
byte-exact, that the inventory delta is additions-only and exactly those two
schemas, and that the citable v3 bundle changes presentation tick labels only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .config import sha256_file  # noqa: E402
from .work9_capacity_acceptance import (  # noqa: E402
    Work9AcceptanceError,
    source_contracts,
    verify_dry_receipt_payload_against_current,
)
from .work9_capacity_analysis import (  # noqa: E402
    FIGURE_IDS,
    load_frozen_capacity_policies,
)
from .work9_capacity_figures import FORMATS, CapacityFigureError  # noqa: E402
from .work9_capacity_figures_v2 import (  # noqa: E402
    build_capacity_figures_v2,
    load_presentation_correction,
)
from .work9_capacity_figures_v3 import (  # noqa: E402
    NARROW_ZOOM_FORMAT,
    NARROW_ZOOM_SPAN_THRESHOLD,
    OTHER_ZOOM_FORMAT,
    build_capacity_figures_v3,
    load_presentation_supersession,
)


CONFIG_RELATIVE = Path(
    "revised_experiments/configs/work9_capacity_postcompletion_closure_v1.json"
)
CONFIG_SHA256 = "2bd23eba72539ef9b5b33060f2a9c618c278d073f14f1281c386f73c79131b65"
RECEIPT_ROOT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_postcompletion_closure_v1"
)
RECEIPT_RELATIVE = RECEIPT_ROOT_RELATIVE / "dedicated_test_receipt.json"
SCHEMA_VERSION = "paper2-v2-work9-capacity-postcompletion-closure-receipt-1"
EXPECTED_ACCEPTED_CONTRACT_COUNT = 38
EXPECTED_CURRENT_CONTRACT_COUNT = 40
EXPECTED_V3_SVG_NARROW_LABELS = (
    ("0.9982", "0.9988", "0.9994", "1.0000"),
    ("0.9982", "0.9988", "0.9994", "1.0000"),
)


class PostcompletionClosureError(RuntimeError):
    pass


def _json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostcompletionClosureError(f"cannot read {context}: {path}") from error
    if not isinstance(value, Mapping):
        raise PostcompletionClosureError(f"{context} is not a JSON object")
    return value


def _reject_symlink_components(path: Path, root: Path) -> None:
    lexical = path if path.is_absolute() else root / path
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise PostcompletionClosureError("closure reference escapes project root") from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PostcompletionClosureError(f"closure reference has symlink component: {cursor}")


def verify_artifact_reference(
    project_root: Path, reference: Mapping[str, Any], *, context: str
) -> Path:
    if set(reference) not in ({"path", "sha256", "bytes"}, {"path", "sha256", "bytes", "row_count"}):
        raise PostcompletionClosureError(f"{context} reference fields differ")
    raw = reference.get("path")
    if not isinstance(raw, str):
        raise PostcompletionClosureError(f"{context} path is not a string")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or raw != pure.as_posix():
        raise PostcompletionClosureError(f"{context} path is not safe relative POSIX")
    root = project_root.resolve()
    lexical = root / raw
    _reject_symlink_components(lexical, root)
    resolved = lexical.resolve()
    if root != resolved and root not in resolved.parents:
        raise PostcompletionClosureError(f"{context} resolves outside project")
    if not resolved.is_file() or resolved.is_symlink():
        raise PostcompletionClosureError(f"{context} is unavailable")
    if resolved.stat().st_size != reference.get("bytes"):
        raise PostcompletionClosureError(f"{context} byte count changed")
    if sha256_file(resolved) != reference.get("sha256"):
        raise PostcompletionClosureError(f"{context} hash changed")
    return resolved


def load_closure_config(project_root: str | Path) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    path = root / CONFIG_RELATIVE
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_mode & 0o222
        or sha256_file(path) != CONFIG_SHA256
    ):
        raise PostcompletionClosureError("frozen post-completion closure config changed")
    config = _json(path, "post-completion closure config")
    if (
        config.get("schema_version")
        != "paper2-v2-work9-capacity-postcompletion-closure-1"
        or config.get("status") != "frozen_post_completion_closure_contract"
    ):
        raise PostcompletionClosureError("closure config identity changed")
    return config


def _reference_map(rows: Any, *, context: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise PostcompletionClosureError(f"{context} is not a list")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise PostcompletionClosureError(f"{context} contains invalid reference")
        if row["path"] in result:
            raise PostcompletionClosureError(f"{context} contains duplicate path")
        result[row["path"]] = row
    return result


def verify_accepted_contract_boundary(
    project_root: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    bindings = config["artifact_bindings"]
    dry_path = verify_artifact_reference(
        project_root, bindings["dry_run_receipt"], context="accepted dry receipt"
    )
    draft_path = verify_artifact_reference(
        project_root,
        bindings["runner_acceptance_draft"],
        context="accepted runner DRAFT",
    )
    final_path = verify_artifact_reference(
        project_root,
        bindings["final_runner_acceptance"],
        context="accepted runner acceptance",
    )
    dry = _json(dry_path, "accepted dry receipt")
    draft = _json(draft_path, "accepted runner DRAFT")
    final = _json(final_path, "accepted runner acceptance")
    accepted = _reference_map(dry.get("source_contracts"), context="accepted contracts")
    if len(accepted) != EXPECTED_ACCEPTED_CONTRACT_COUNT:
        raise PostcompletionClosureError("accepted contract count changed")
    if draft.get("source_contracts") != dry.get("source_contracts"):
        raise PostcompletionClosureError("DRAFT accepted contract inventory differs")
    if final.get("source_contracts") != dry.get("source_contracts"):
        raise PostcompletionClosureError("final accepted contract inventory differs")
    if draft.get("bundle_digests") != dry.get("bundle_digests"):
        raise PostcompletionClosureError("DRAFT bundle digests differ")
    if final.get("bundle_digests") != dry.get("bundle_digests"):
        raise PostcompletionClosureError("final bundle digests differ")
    expected_boundary = config["accepted_contract_boundary"]
    expected_bundles = {
        "runner_source_bundle_sha256": expected_boundary[
            "accepted_runner_source_bundle_sha256"
        ],
        "schemas_bundle_sha256": expected_boundary["accepted_schema_bundle_sha256"],
        "tests_bundle_sha256": expected_boundary["accepted_test_bundle_sha256"],
    }
    if dry.get("bundle_digests") != expected_bundles:
        raise PostcompletionClosureError("accepted bundle digests changed")
    for index, reference in enumerate(accepted.values()):
        verify_artifact_reference(
            project_root, reference, context=f"accepted contract {index}"
        )

    current = _reference_map(
        source_contracts(project_root), context="current dynamic contracts"
    )
    removed = sorted(set(accepted) - set(current))
    changed = sorted(
        path for path in set(accepted) & set(current) if accepted[path] != current[path]
    )
    added = [current[path] for path in sorted(set(current) - set(accepted))]
    expected_added = expected_boundary["expected_additions"]
    if removed or changed or added != expected_added:
        raise PostcompletionClosureError(
            f"contract delta differs: removed={removed}, changed={changed}, added={added}"
        )
    if len(current) != EXPECTED_CURRENT_CONTRACT_COUNT:
        raise PostcompletionClosureError("current contract count changed")

    try:
        verify_dry_receipt_payload_against_current(project_root, dry)
    except Work9AcceptanceError as error:
        historical_signal = f"{type(error).__module__}.{type(error).__name__}: {error}"
    else:
        raise PostcompletionClosureError("historical boundary signal unexpectedly disappeared")
    expected_chain = config["historical_boundary_signal"]["expected_exception_chain"]
    if historical_signal != expected_chain[0]:
        raise PostcompletionClosureError("historical boundary signal changed")
    return {
        "accepted_contract_count": len(accepted),
        "current_contract_count": len(current),
        "changed_accepted_contract_count": len(changed),
        "removed_accepted_contract_count": len(removed),
        "added_contracts": added,
        "historical_boundary_signal": historical_signal,
        "accepted_bundle_digests": dict(expected_bundles),
    }


def _verify_hash_manifest(
    bundle: Path, *, allowed_legacy_unlisted: Mapping[str, str] | None = None
) -> int:
    manifest = bundle / "hashes.sha256"
    declared: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in declared:
            raise PostcompletionClosureError(f"invalid hash manifest line: {line}")
        declared[relative] = digest
    actual = {
        path.relative_to(bundle).as_posix(): sha256_file(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "hashes.sha256"
    }
    allowed = dict(allowed_legacy_unlisted or {})
    for relative, digest in allowed.items():
        if actual.get(relative) != digest or relative in declared:
            raise PostcompletionClosureError(
                f"legacy unlisted artifact differs: {bundle / relative}"
            )
        actual.pop(relative)
    if declared != actual:
        raise PostcompletionClosureError(f"hash inventory differs: {bundle}")
    return len(actual)


def verify_bound_completion_artifacts(
    project_root: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    bindings = config["artifact_bindings"]
    resolved = {
        key: verify_artifact_reference(project_root, reference, context=key)
        for key, reference in bindings.items()
    }
    if len(resolved) != 14:
        raise PostcompletionClosureError("closure artifact binding count changed")
    selection = _json(resolved["selection_index"], "hybrid selection index")
    outer = _json(resolved["outer_ledger_index"], "hybrid outer index")
    if (
        selection.get("status") != "frozen_complete"
        or len(selection.get("entries", ())) != 200
        or outer.get("status") != "frozen_complete"
        or len(outer.get("entries", ())) != 200
    ):
        raise PostcompletionClosureError("hybrid index identity/count changed")
    report = _json(resolved["analysis_report"], "capacity analysis report")
    if report.get("status") != "secondary_sensitivity_complete":
        raise PostcompletionClosureError("analysis report status changed")

    manifests: dict[str, Mapping[str, Any]] = {}
    inventory_counts: dict[str, int] = {}
    for version in ("v1", "v2", "v3"):
        manifest = _json(resolved[f"figure_{version}_manifest"], f"figure {version} manifest")
        if (
            tuple(manifest.get("figure_ids", ())) != FIGURE_IDS
            or tuple(manifest.get("formats", ())) != FORMATS
            or manifest.get("figure_count") != 7
        ):
            raise PostcompletionClosureError(f"figure {version} inventory changed")
        manifests[version] = manifest
        inventory_counts[version] = _verify_hash_manifest(
            resolved[f"figure_{version}_manifest"].parent,
            allowed_legacy_unlisted=(
                {
                    "READ_ONLY_NOTICE.txt": (
                        "123fb395a1aaf676d5694af41e5256e9c765f5d19771c030351f4a07b7769e5f"
                    )
                }
                if version == "v1"
                else None
            ),
        )
    v2 = manifests["v2"]
    v3 = manifests["v3"]
    if (
        v3.get("supersedes_v2_figure_manifest")
        != {**bindings["figure_v2_manifest"], "row_count": None}
        or v3.get("supersedes_v2_hash_manifest")
        != {**bindings["figure_v2_hash_manifest"], "row_count": None}
        or v3.get("citation_status")
        != {"v2_bundle_citable": False, "v3_bundle_citable": True}
        or v3.get("analysis_report") != {**bindings["analysis_report"], "row_count": None}
        or v2.get("analysis_report") != {**bindings["analysis_report"], "row_count": None}
    ):
        raise PostcompletionClosureError("v2/v3 manifest lineage or citation changed")
    return {
        "artifact_binding_count": len(resolved),
        "selection_index_entries": len(selection["entries"]),
        "outer_index_entries": len(outer["entries"]),
        "figure_bundle_file_counts_excluding_hash_manifest": inventory_counts,
        "figure_count_each": 7,
        "format_count_each": 3,
        "v3_citable": True,
        "v2_citable": False,
    }


def _assert_axis_arrays_equal(left: plt.Axes, right: plt.Axes) -> int:
    if len(left.lines) != len(right.lines) or len(left.collections) != len(right.collections):
        raise PostcompletionClosureError("v2/v3 artist inventory differs")
    checks = 0
    for left_line, right_line in zip(left.lines, right.lines, strict=True):
        if not np.array_equal(left_line.get_xdata(), right_line.get_xdata()) or not np.array_equal(
            left_line.get_ydata(), right_line.get_ydata()
        ):
            raise PostcompletionClosureError("v2/v3 line plotting array differs")
        checks += 2
    for left_collection, right_collection in zip(
        left.collections, right.collections, strict=True
    ):
        if hasattr(left_collection, "get_segments"):
            left_segments = left_collection.get_segments()
            right_segments = right_collection.get_segments()
            if len(left_segments) != len(right_segments):
                raise PostcompletionClosureError("v2/v3 segment inventory differs")
            for left_segment, right_segment in zip(
                left_segments, right_segments, strict=True
            ):
                if not np.array_equal(left_segment, right_segment):
                    raise PostcompletionClosureError("v2/v3 segment plotting array differs")
                checks += 1
        else:
            if not np.array_equal(
                left_collection.get_offsets(), right_collection.get_offsets()
            ):
                raise PostcompletionClosureError("v2/v3 collection offsets differ")
            checks += 1
    if left.get_xlim() != right.get_xlim() or left.get_ylim() != right.get_ylim():
        raise PostcompletionClosureError("v2/v3 axis limit differs")
    return checks + 2


def verify_v3_scientific_and_visual_correction(
    project_root: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    report_path = verify_artifact_reference(
        project_root,
        config["artifact_bindings"]["analysis_report"],
        context="analysis report for scientific equality",
    )
    report = _json(report_path, "analysis report for scientific equality")
    policies = load_frozen_capacity_policies(project_root)
    correction = load_presentation_correction(project_root)
    supersession = load_presentation_supersession(project_root)
    v2 = build_capacity_figures_v2(report, policies.figure_spec, correction)
    v3 = build_capacity_figures_v3(
        report, policies.figure_spec, correction, supersession
    )
    array_checks = 0
    try:
        if tuple(v2) != FIGURE_IDS or tuple(v3) != FIGURE_IDS:
            raise PostcompletionClosureError("v2/v3 figure IDs differ")
        for figure_id in FIGURE_IDS:
            if len(v2[figure_id].axes) != len(v3[figure_id].axes):
                raise PostcompletionClosureError("v2/v3 parent axis count differs")
            for left, right in zip(
                v2[figure_id].axes, v3[figure_id].axes, strict=True
            ):
                array_checks += _assert_axis_arrays_equal(left, right)
        v2_insets = v2["capacity_metric_profile"]._capacity_zoom_axes
        v3_insets = v3["capacity_metric_profile"]._capacity_zoom_axes
        for left, right in zip(v2_insets, v3_insets, strict=True):
            array_checks += _assert_axis_arrays_equal(left, right)
        formats = v3["capacity_metric_profile"]._capacity_zoom_formatter_formats
        if formats != (
            OTHER_ZOOM_FORMAT,
            OTHER_ZOOM_FORMAT,
            OTHER_ZOOM_FORMAT,
            NARROW_ZOOM_FORMAT,
            NARROW_ZOOM_FORMAT,
        ):
            raise PostcompletionClosureError("v3 formatter inventory changed")
        v3["capacity_metric_profile"].canvas.draw()
        narrow_labels: list[tuple[str, ...]] = []
        for axis in v3_insets:
            lower, upper = axis.get_ylim()
            labels = tuple(
                axis.yaxis.get_major_formatter()(tick)
                for tick in axis.get_yticks()
                if lower - 1e-12 <= tick <= upper + 1e-12
            )
            if len(labels) != len(set(labels)):
                raise PostcompletionClosureError("v3 inset still has duplicate tick labels")
            if upper - lower <= NARROW_ZOOM_SPAN_THRESHOLD + 1e-12:
                narrow_labels.append(labels)
        if tuple(narrow_labels) != EXPECTED_V3_SVG_NARROW_LABELS:
            raise PostcompletionClosureError("v3 narrow inset tick labels changed")
    finally:
        for figures in (v2, v3):
            for figure in figures.values():
                plt.close(figure)

    bundle = project_root / "revised_experiments/artifacts/work9_capacity_figures_v3"
    svg = bundle / "figures/capacity_metric_profile.svg"
    ElementTree.parse(svg)
    svg_text = svg.read_text(encoding="utf-8")
    for labels in EXPECTED_V3_SVG_NARROW_LABELS:
        for label in labels:
            if f"<!-- {label} -->" not in svg_text:
                raise PostcompletionClosureError(f"rendered v3 SVG lacks tick label {label}")
    png_checks = 0
    for png in sorted((bundle / "figures").glob("*.png")):
        with Image.open(png) as image:
            dpi = image.info.get("dpi")
            if (
                image.width <= 0
                or image.height <= 0
                or not dpi
                or any(abs(float(value) - 300.0) > 0.01 for value in dpi[:2])
            ):
                raise PostcompletionClosureError(f"v3 PNG QA failed: {png}")
            png_checks += 1
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise PostcompletionClosureError("v3 bundle contains symlink")
        wanted = 0o444 if path.is_file() else 0o555
        if path.stat().st_mode & 0o777 != wanted:
            raise PostcompletionClosureError(f"v3 mode differs: {path}")
    return {
        "figure_ids_compared": len(FIGURE_IDS),
        "plotting_array_and_limit_checks": array_checks,
        "narrow_formatter": NARROW_ZOOM_FORMAT,
        "other_formatter": OTHER_ZOOM_FORMAT,
        "narrow_tick_labels": [list(value) for value in EXPECTED_V3_SVG_NARROW_LABELS],
        "unique_tick_labels": True,
        "png_300dpi_files_verified": png_checks,
        "svg_xml_and_tick_labels_verified": True,
        "v3_tree_modes_and_symlinks_verified": True,
    }


def validate_postcompletion_closure(project_root: str | Path) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    config = load_closure_config(root)
    safety = config.get("safety")
    expected_false = {
        "cuda_smoke_launched",
        "model_outcome_values_recomputed",
        "new_outer_evaluation_launched",
        "new_training_launched",
        "outer_outcome_ledgers_opened",
        "primary_or_capacity_registry_modified",
        "selection_or_reselection_launched",
    }
    if (
        not isinstance(safety, Mapping)
        or safety.get("source_artifact_role")
        != "post_completion_read_only_lineage_and_presentation_validation"
        or any(safety.get(key) is not False for key in expected_false)
    ):
        raise PostcompletionClosureError("closure safety contract changed")
    return {
        "status": "pass",
        "schema_version": SCHEMA_VERSION,
        "config": {
            "path": CONFIG_RELATIVE.as_posix(),
            "sha256": CONFIG_SHA256,
            "bytes": (root / CONFIG_RELATIVE).stat().st_size,
            "row_count": None,
        },
        "accepted_contract_boundary": verify_accepted_contract_boundary(root, config),
        "completion_artifacts": verify_bound_completion_artifacts(root, config),
        "scientific_and_visual_correction": verify_v3_scientific_and_visual_correction(
            root, config
        ),
        "historical_full_discover": dict(config["historical_boundary_signal"]),
        "safety": dict(safety),
        "citation_boundary": dict(config["citation_boundary"]),
    }


def artifact_record(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "row_count": None,
    }


def verify_dedicated_receipt(
    project_root: str | Path, receipt_path: str | Path = RECEIPT_RELATIVE
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    lexical = Path(receipt_path)
    path = lexical if lexical.is_absolute() else root / lexical
    _reject_symlink_components(path, root)
    receipt = _json(path, "post-completion dedicated test receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "pass"
        or receipt.get("test_status") != "dedicated_postcompletion_suite_pass"
        or receipt.get("tests_run") != 6
        or receipt.get("errors") != 0
        or receipt.get("failures") != 0
    ):
        raise PostcompletionClosureError("dedicated test receipt identity changed")
    for key in ("config", "validator_source", "validator_cli", "validator_tests", "test_log"):
        verify_artifact_reference(root, receipt[key], context=f"receipt {key}")
    if receipt.get("closure") != validate_postcompletion_closure(root):
        raise PostcompletionClosureError("receipt closure payload differs from current validation")
    return receipt


__all__ = [
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "PostcompletionClosureError",
    "RECEIPT_RELATIVE",
    "RECEIPT_ROOT_RELATIVE",
    "SCHEMA_VERSION",
    "load_closure_config",
    "validate_postcompletion_closure",
    "verify_accepted_contract_boundary",
    "verify_artifact_reference",
    "verify_bound_completion_artifacts",
    "verify_dedicated_receipt",
    "verify_v3_scientific_and_visual_correction",
]
