"""Publication-citable v3 renderer for the seven frozen capacity figures.

V3 is a presentation-only sibling of the immutable v2 candidate.  It reuses
the exact v2 figure builders and changes only the numeric tick formatter on
very narrow metric-profile insets: zoom spans no larger than 0.003 use four
decimal places so distinct tick positions cannot receive duplicate labels.
No plotting array, limit, aggregation, inference, figure ID, or claim changes.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

from .config import sha256_file  # noqa: E402
from .work9_capacity_analysis import (  # noqa: E402
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    EVIDENCE_LABEL,
    EXECUTION_ADDENDUM_SHA256,
    FIGURE_IDS,
    FIGURE_SPEC_SHA256,
    PROTOCOL_ID,
    REPORT_SCHEMA_VERSION,
    RUN_FAMILY,
    ImmutableReference,
    _freeze_tree,
    _read_json,
    _restore_tree_for_cleanup,
    _validate_schema,
)
from .work9_capacity_figures import (  # noqa: E402
    FORMATS,
    CapacityFigureError,
    _artifact_record,
    load_capacity_analysis_report,
)
from .work9_capacity_figures_v2 import (  # noqa: E402
    PRESENTATION_CORRECTION_RELATIVE,
    PRESENTATION_CORRECTION_SHA256,
    PRODUCTION_ANALYSIS_REPORT_SHA256,
    build_capacity_figures_v2,
    load_presentation_correction,
)
from .work9_capacity_registry import validate_isolated_relative_root  # noqa: E402
from .work9_capacity_types import TEST_SANDBOX_ROOT_RELATIVE  # noqa: E402


RENDERER_VERSION = "paper2-v2-work9-capacity-figure-renderer-3"
MANIFEST_SCHEMA_VERSION = "paper2-v2-work9-capacity-figure-manifest-3"
FIGURE_MANIFEST_SCHEMA_RELATIVE = Path(
    "revised_experiments/registry/schemas/work9_capacity/"
    "work9_capacity_figure_manifest_v3.schema.json"
)
FIGURE_OUTPUT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v3"
)
PRESENTATION_SUPERSESSION_RELATIVE = Path(
    "revised_experiments/configs/"
    "work9_capacity_figure_presentation_supersession_v3.json"
)
PRESENTATION_SUPERSESSION_SHA256 = (
    "1375fbe9f9f4b2dda3d82934e2454ab42235cfb2beb4bc92d9d89246955c68a6"
)
V2_FIGURE_MANIFEST_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v2/figure_manifest.json"
)
V2_FIGURE_MANIFEST_SHA256 = (
    "5b0ae401cf237b01502c581a605b9640552971cf0fafc991e013c9d873f089c5"
)
V2_HASH_MANIFEST_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v2/hashes.sha256"
)
V2_HASH_MANIFEST_SHA256 = (
    "b683c83712f6eaacb99302870ec44022fcd5b1790bf5567d01b9a06cb1a46436"
)
NARROW_ZOOM_SPAN_THRESHOLD = 0.003
NARROW_ZOOM_FORMAT = "%.4f"
OTHER_ZOOM_FORMAT = "%.3f"


def _exact_artifact_mapping(
    value: Any,
    *,
    path: str,
    sha256: str,
    bytes_: int,
    context: str,
) -> None:
    expected = {"path": path, "sha256": sha256, "bytes": bytes_}
    if value != expected:
        raise CapacityFigureError(f"{context} reference changed")


def verify_presentation_supersession(
    supersession: Mapping[str, Any], *, project_root: Path | None = None
) -> None:
    """Fail closed if the v2-to-v3 presentation-only contract drifts."""

    if (
        supersession.get("schema_version")
        != "paper2-v2-work9-capacity-figure-presentation-supersession-3"
        or supersession.get("status")
        != "post_v2_visual_qa_tick_label_correction_frozen_before_v3_render"
        or supersession.get("citable_bundle") != "work9_capacity_figures_v3"
        or supersession.get("v2_citable") is not False
        or supersession.get("v3_citable_after_integrity_and_visual_qa") is not True
        or supersession.get("v2_preservation_status")
        != "immutable_superseded_presentation_candidate_not_citable"
        or tuple(supersession.get("figure_ids", ())) != FIGURE_IDS
        or tuple(supersession.get("formats", ())) != FORMATS
    ):
        raise CapacityFigureError("v3 presentation supersession identity changed")

    _exact_artifact_mapping(
        supersession.get("analysis_report"),
        path="revised_experiments/artifacts/work9_capacity_analysis_v1/report.json",
        sha256=PRODUCTION_ANALYSIS_REPORT_SHA256,
        bytes_=351600,
        context="v3 supersession analysis report",
    )
    _exact_artifact_mapping(
        supersession.get("v2_presentation_correction"),
        path=PRESENTATION_CORRECTION_RELATIVE.as_posix(),
        sha256=PRESENTATION_CORRECTION_SHA256,
        bytes_=3541,
        context="v3 supersession v2 correction",
    )
    references = (
        (
            "superseded_v2_figure_manifest",
            V2_FIGURE_MANIFEST_RELATIVE.as_posix(),
            V2_FIGURE_MANIFEST_SHA256,
            11993,
        ),
        (
            "superseded_v2_hash_manifest",
            V2_HASH_MANIFEST_RELATIVE.as_posix(),
            V2_HASH_MANIFEST_SHA256,
            2391,
        ),
        (
            "superseded_v2_renderer",
            "revised_experiments/src/paper2_v2/work9_capacity_figures_v2.py",
            "9acf7919d0257d57364283c6f12fa483776c952820fd5be33974fc35b7b8e40c",
            29374,
        ),
        (
            "superseded_v2_renderer_cli",
            "revised_experiments/scripts/generate_work9_capacity_figures_v2.py",
            "429240f61a262ba96a17c38759efe37d9a3847dec8963dc52c39cf3e06dcf257",
            2280,
        ),
        (
            "superseded_v2_renderer_schema",
            "revised_experiments/registry/schemas/work9_capacity/work9_capacity_figure_manifest_v2.schema.json",
            "f65d598c86cb3110b2fa87dddab7f01c8d3d1ae49fc4f21f83a4f3ea8d178fd6",
            8783,
        ),
        (
            "superseded_v2_renderer_tests",
            "revised_experiments/tests/test_work9_capacity_figures_v2.py",
            "fa0ce6d7c82f17e8f76a031bc5fc03d93f30ab958e8e9719dee9b3aa57fef34f",
            12380,
        ),
    )
    for key, path, sha256, bytes_ in references:
        _exact_artifact_mapping(
            supersession.get(key),
            path=path,
            sha256=sha256,
            bytes_=bytes_,
            context=f"v3 supersession {key}",
        )

    correction = supersession.get("correction_scope")
    formatter = correction.get("formatter_rule") if isinstance(correction, Mapping) else None
    if (
        not isinstance(correction, Mapping)
        or correction.get("affected_figure_id") != "capacity_metric_profile"
        or tuple(correction.get("affected_insets", ())) != ("roc_auc", "pr_auc")
        or formatter
        != {
            "format_if_zoom_span_at_most_0_003": NARROW_ZOOM_FORMAT,
            "format_other_zoom_spans": OTHER_ZOOM_FORMAT,
            "zoom_span_threshold_inclusive": NARROW_ZOOM_SPAN_THRESHOLD,
        }
    ):
        raise CapacityFigureError("v3 tick-label correction contract changed")
    preservation = supersession.get("preservation")
    if (
        not isinstance(preservation, Mapping)
        or set(preservation)
        != {
            "analysis_report_changes_allowed",
            "effect_ci_p_or_holm_changes_allowed",
            "figure_id_or_purpose_changes_allowed",
            "frozen_figure_spec_changes_allowed",
            "plotting_array_changes_allowed",
            "v1_bundle_changes_allowed",
            "v2_bundle_changes_allowed",
            "v2_correction_contract_changes_allowed",
            "zoom_limit_changes_allowed",
        }
        or any(value is not False for value in preservation.values())
    ):
        raise CapacityFigureError("v3 scientific preservation contract changed")

    if project_root is not None:
        root = project_root.resolve()
        for key in (
            "analysis_report",
            "v2_presentation_correction",
            "superseded_v2_figure_manifest",
            "superseded_v2_hash_manifest",
            "superseded_v2_renderer",
            "superseded_v2_renderer_cli",
            "superseded_v2_renderer_schema",
            "superseded_v2_renderer_tests",
        ):
            ImmutableReference.load(
                {**supersession[key], "row_count": None},
                project_root=root,
                context=f"v3 supersession {key}",
                ledger=False,
            )


def load_presentation_supersession(project_root: str | Path) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    path = root / PRESENTATION_SUPERSESSION_RELATIVE
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != PRESENTATION_SUPERSESSION_SHA256
    ):
        raise CapacityFigureError("frozen v3 presentation supersession changed")
    supersession = _read_json(path, "capacity figure presentation supersession v3")
    verify_presentation_supersession(supersession, project_root=root)
    return supersession


def build_capacity_figures_v3(
    report: Mapping[str, Any],
    figure_spec: Mapping[str, Any],
    correction_v2: Mapping[str, Any],
    supersession_v3: Mapping[str, Any],
) -> dict[str, plt.Figure]:
    """Build v2-identical figures, then apply the frozen formatter-only fix."""

    verify_presentation_supersession(supersession_v3)
    figures = build_capacity_figures_v2(report, figure_spec, correction_v2)
    profile = figures["capacity_metric_profile"]
    formats: list[str] = []
    for axis in profile._capacity_zoom_axes:
        lower, upper = axis.get_ylim()
        format_string = (
            NARROW_ZOOM_FORMAT
            if (upper - lower) <= NARROW_ZOOM_SPAN_THRESHOLD + 1e-12
            else OTHER_ZOOM_FORMAT
        )
        axis.yaxis.set_major_formatter(FormatStrFormatter(format_string))
        formats.append(format_string)
    setattr(profile, "_capacity_zoom_formatter_formats", tuple(formats))
    return figures


def render_capacity_figures_from_report_v3(
    project_root: str | Path,
    *,
    report: Mapping[str, Any],
    report_reference: Mapping[str, Any],
    figure_spec: Mapping[str, Any],
    correction_v2: Mapping[str, Any],
    supersession_v3: Mapping[str, Any],
    output_directory: str | Path,
    synthetic_test_mode: bool = False,
) -> Path:
    """Atomically publish the citable v3 bundle without mutating v1 or v2."""

    root = Path(project_root).resolve()
    raw_output = Path(output_directory)
    expected_status = (
        "synthetic_contract_test" if synthetic_test_mode else "secondary_sensitivity_complete"
    )
    if report.get("status") != expected_status:
        raise CapacityFigureError("v3 output mode differs from report status")
    if (
        not synthetic_test_mode
        and report_reference.get("sha256") != PRODUCTION_ANALYSIS_REPORT_SHA256
    ):
        raise CapacityFigureError("v3 production render requires the sealed analysis report")
    try:
        if synthetic_test_mode:
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=TEST_SANDBOX_ROOT_RELATIVE,
                context="synthetic capacity v3 figure output",
                require_new=True,
            )
        else:
            if raw_output != FIGURE_OUTPUT_RELATIVE:
                raise CapacityFigureError(
                    f"production v3 figure output is fixed at {FIGURE_OUTPUT_RELATIVE}"
                )
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=Path("revised_experiments/artifacts"),
                context="production capacity v3 figure output",
                require_new=True,
            )
    except (ValueError, FileExistsError) as error:
        raise CapacityFigureError(str(error)) from error

    figures = build_capacity_figures_v3(
        report, figure_spec, correction_v2, supersession_v3
    )
    staging = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    figure_directory = staging / "figures"
    figure_directory.mkdir(parents=True, exist_ok=False)
    try:
        spec_lookup = {row["figure_id"]: row for row in figure_spec["figures"]}
        manifest_rows: list[dict[str, Any]] = []
        for figure_id in FIGURE_IDS:
            figure = figures[figure_id]
            files: dict[str, Any] = {}
            for format_id in FORMATS:
                suffix = "png" if format_id == "png_300dpi" else format_id
                path = figure_directory / f"{figure_id}.{suffix}"
                options: dict[str, Any] = {
                    "format": suffix,
                    "bbox_inches": "tight",
                    "facecolor": "white",
                }
                if suffix == "png":
                    options["dpi"] = 300
                figure.savefig(path, **options)
                files[format_id] = _artifact_record(
                    path,
                    project_root=root,
                    final_path=destination / "figures" / path.name,
                )
            plt.close(figure)
            spec_row = spec_lookup[figure_id]
            if figure_id == "capacity_metric_profile":
                revision = "zoom_tick_labels_unique_v3"
            elif figure_id == "capacity_condition_macro_f1":
                revision = "carried_forward_ceiling_zoom_v2"
            else:
                revision = "carried_forward_unchanged_v2"
            manifest_rows.append(
                {
                    "figure_id": figure_id,
                    "inferential": bool(spec_row["inferential"]),
                    "purpose": str(spec_row["purpose"]),
                    "files": files,
                    "presentation_revision": revision,
                }
            )

        renderer_source = root / (
            "revised_experiments/src/paper2_v2/work9_capacity_figures_v3.py"
        )
        renderer_cli = root / (
            "revised_experiments/scripts/generate_work9_capacity_figures_v3.py"
        )
        renderer_schema = root / FIGURE_MANIFEST_SCHEMA_RELATIVE
        renderer_tests = root / (
            "revised_experiments/tests/test_work9_capacity_figures_v3.py"
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "status": (
                "synthetic_contract_test"
                if synthetic_test_mode
                else "secondary_sensitivity_figures_tick_labels_corrected"
            ),
            "protocol_id": PROTOCOL_ID,
            "run_family": RUN_FAMILY,
            "evidence_label": EVIDENCE_LABEL,
            "analysis_report": dict(report_reference),
            "analysis_report_schema_version": REPORT_SCHEMA_VERSION,
            "analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "figure_spec_sha256": FIGURE_SPEC_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "v2_presentation_correction": _artifact_record(
                root / PRESENTATION_CORRECTION_RELATIVE, project_root=root
            ),
            "v3_presentation_supersession": _artifact_record(
                root / PRESENTATION_SUPERSESSION_RELATIVE, project_root=root
            ),
            "renderer_source": _artifact_record(renderer_source, project_root=root),
            "renderer_cli": _artifact_record(renderer_cli, project_root=root),
            "renderer_schema": _artifact_record(renderer_schema, project_root=root),
            "renderer_tests": _artifact_record(renderer_tests, project_root=root),
            "supersedes_v2_figure_manifest": _artifact_record(
                root / V2_FIGURE_MANIFEST_RELATIVE, project_root=root
            ),
            "supersedes_v2_hash_manifest": _artifact_record(
                root / V2_HASH_MANIFEST_RELATIVE, project_root=root
            ),
            "supersession_scope": (
                "presentation_only_tick_label_precision_v2_remains_immutable_not_citable"
            ),
            "citation_status": {
                "v2_bundle_citable": False,
                "v3_bundle_citable": True,
            },
            "tick_label_correction": {
                "affected_figure_id": "capacity_metric_profile",
                "affected_insets": ["roc_auc", "pr_auc"],
                "narrow_zoom_span_threshold_inclusive": NARROW_ZOOM_SPAN_THRESHOLD,
                "narrow_zoom_format": NARROW_ZOOM_FORMAT,
                "other_zoom_format": OTHER_ZOOM_FORMAT,
            },
            "scientific_content_unchanged": {
                "analysis_report_sha256": str(report_reference["sha256"]),
                "figure_ids_and_purposes": True,
                "plotting_arrays_and_zoom_limits": True,
                "condition_metrics_and_aggregations": True,
                "effects_confidence_intervals_p_values_and_holm": True,
                "gate1_gate2_and_claim_boundary": True,
            },
            "figure_count": 7,
            "figure_ids": list(FIGURE_IDS),
            "formats": list(FORMATS),
            "figures": manifest_rows,
            "completed_gate1": dict(report["completed_gate1"]),
            "claim_boundary": dict(report["claim_boundary"]),
        }
        _validate_schema(
            manifest,
            schema_path=renderer_schema,
            context="capacity v3 figure manifest",
        )
        (staging / "figure_manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "READ_ONLY_NOTICE.txt").write_text(
            "Citable presentation-corrected v3 of the same seven frozen capacity figures. "
            "V2 is preserved but superseded because two narrow insets had ambiguous "
            "three-decimal tick labels. Scientific data and Gate 1/Gate 2 are unchanged.\n",
            encoding="utf-8",
        )
        hashes = [
            f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}"
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "hashes.sha256"
        ]
        (staging / "hashes.sha256").write_text(
            "\n".join(hashes) + "\n", encoding="utf-8"
        )
        _freeze_tree(staging)
        os.rename(staging, destination)
    except BaseException:
        for figure in figures.values():
            plt.close(figure)
        _restore_tree_for_cleanup(staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "figure_manifest.json"


def render_capacity_figures_v3(
    project_root: str | Path,
    *,
    report_path: str | Path,
    expected_report_sha256: str,
    output_directory: str | Path,
) -> Path:
    """Production entry point for the immutable, citable v3 bundle."""

    root = Path(project_root).resolve()
    if expected_report_sha256 != PRODUCTION_ANALYSIS_REPORT_SHA256:
        raise CapacityFigureError("v3 production report SHA differs from correction")
    report, figure_spec = load_capacity_analysis_report(
        root,
        report_path=report_path,
        expected_report_sha256=expected_report_sha256,
    )
    correction_v2 = load_presentation_correction(root)
    supersession_v3 = load_presentation_supersession(root)
    lexical = Path(report_path)
    absolute_report = lexical if lexical.is_absolute() else root / lexical
    absolute_report = absolute_report.resolve()
    report_reference = {
        "path": absolute_report.relative_to(root).as_posix(),
        "sha256": expected_report_sha256,
        "bytes": absolute_report.stat().st_size,
        "row_count": None,
    }
    return render_capacity_figures_from_report_v3(
        root,
        report=report,
        report_reference=report_reference,
        figure_spec=figure_spec,
        correction_v2=correction_v2,
        supersession_v3=supersession_v3,
        output_directory=output_directory,
        synthetic_test_mode=False,
    )


__all__ = [
    "FIGURE_MANIFEST_SCHEMA_RELATIVE",
    "FIGURE_OUTPUT_RELATIVE",
    "MANIFEST_SCHEMA_VERSION",
    "NARROW_ZOOM_FORMAT",
    "NARROW_ZOOM_SPAN_THRESHOLD",
    "OTHER_ZOOM_FORMAT",
    "PRESENTATION_SUPERSESSION_RELATIVE",
    "PRESENTATION_SUPERSESSION_SHA256",
    "RENDERER_VERSION",
    "V2_FIGURE_MANIFEST_SHA256",
    "V2_HASH_MANIFEST_SHA256",
    "build_capacity_figures_v3",
    "load_presentation_supersession",
    "render_capacity_figures_from_report_v3",
    "render_capacity_figures_v3",
    "verify_presentation_supersession",
]
