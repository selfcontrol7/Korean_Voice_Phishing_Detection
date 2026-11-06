"""Presentation-corrected v2 renderer for the seven frozen capacity figures.

This sibling preserves the immutable v1 renderer and figure bundle.  It reads
the same sealed analysis report, keeps the same seven figure identifiers and
scientific values, and changes only the display of near-ceiling metrics.  Each
truncated axis is explicitly labelled, and the standalone macro-F1 figure
also includes an adjacent full-scale 0-to-1 context panel.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

from .config import sha256_file  # noqa: E402
from .work9_capacity_analysis import (  # noqa: E402
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    CONDITIONS,
    EVIDENCE_LABEL,
    EXECUTION_ADDENDUM_SHA256,
    FIGURE_IDS,
    FIGURE_SPEC_SHA256,
    PROTOCOL_ID,
    REPORT_SCHEMA_VERSION,
    RUN_FAMILY,
    SEEDS,
    ImmutableReference,
    _freeze_tree,
    _read_json,
    _restore_tree_for_cleanup,
    _validate_schema,
    verify_report_claim_boundary,
)
from .work9_capacity_figures import (  # noqa: E402
    COLORS,
    FORMATS,
    CapacityFigureError,
    _artifact_record,
    _capacity_compute_profile,
    _capacity_fold_seed_stability,
    _capacity_parameter_match,
    _capacity_primary_effects,
    _condition_metric_lookup,
    _primary_vs_capacity_condition_change,
    load_capacity_analysis_report,
)
from .work9_capacity_registry import validate_isolated_relative_root  # noqa: E402
from .work9_capacity_types import TEST_SANDBOX_ROOT_RELATIVE  # noqa: E402


RENDERER_VERSION = "paper2-v2-work9-capacity-figure-renderer-2"
MANIFEST_SCHEMA_VERSION = "paper2-v2-work9-capacity-figure-manifest-2"
FIGURE_MANIFEST_SCHEMA_RELATIVE = Path(
    "revised_experiments/registry/schemas/work9_capacity/"
    "work9_capacity_figure_manifest_v2.schema.json"
)
FIGURE_OUTPUT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v2"
)
PRESENTATION_CORRECTION_RELATIVE = Path(
    "revised_experiments/configs/"
    "work9_capacity_figure_presentation_correction_v2.json"
)
PRESENTATION_CORRECTION_SHA256 = (
    "2a6a4870b216e15db2c47ce1a76ce33b60569fef3a625541c6fa05766a763129"
)
V1_FIGURE_MANIFEST_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v1/figure_manifest.json"
)
V1_FIGURE_MANIFEST_SHA256 = (
    "bd17b32d0366ad22a38ce9262f4f617edf08860eca29d4abf0a05be0d8d31763"
)
PRODUCTION_ANALYSIS_REPORT_SHA256 = (
    "cae39e462a083b20f6963912c1837cefa138441a0dd91bea5c4c0d56cf3f5a0b"
)
AFFECTED_FIGURE_IDS = (
    "capacity_condition_macro_f1",
    "capacity_metric_profile",
)
UNAFFECTED_FIGURE_IDS = tuple(
    figure_id for figure_id in FIGURE_IDS if figure_id not in AFFECTED_FIGURE_IDS
)
EXPECTED_ZOOM_LIMITS = {
    "macro_f1": (0.99, 1.001),
    "sensitivity": (0.989, 1.001),
    "specificity": (0.989, 1.001),
    "roc_auc": (0.998, 1.0002),
    "pr_auc": (0.998, 1.0002),
}


def _exact_artifact_mapping(
    value: Any,
    *,
    path: str,
    sha256: str,
    bytes_: int,
    context: str,
) -> None:
    expected = {
        "path": path,
        "sha256": sha256,
        "bytes": bytes_,
    }
    if value != expected:
        raise CapacityFigureError(f"{context} reference changed")


def verify_presentation_correction(
    correction: Mapping[str, Any], *, project_root: Path | None = None
) -> None:
    """Fail closed if the post-analysis presentation-only contract drifts."""

    if (
        correction.get("schema_version")
        != "paper2-v2-work9-capacity-figure-presentation-correction-2"
        or correction.get("status")
        != "post_analysis_presentation_correction_frozen_before_v2_render"
        or correction.get("supersession_scope")
        != "presentation_only_v1_remains_immutable_and_scientifically_valid"
        or tuple(correction.get("figure_ids", ())) != FIGURE_IDS
        or tuple(correction.get("affected_figure_ids", ())) != AFFECTED_FIGURE_IDS
        or tuple(correction.get("unaffected_figure_ids", ())) != UNAFFECTED_FIGURE_IDS
    ):
        raise CapacityFigureError("presentation correction identity/inventory changed")
    _exact_artifact_mapping(
        correction.get("analysis_report"),
        path="revised_experiments/artifacts/work9_capacity_analysis_v1/report.json",
        sha256=PRODUCTION_ANALYSIS_REPORT_SHA256,
        bytes_=351600,
        context="presentation correction analysis report",
    )
    _exact_artifact_mapping(
        correction.get("frozen_figure_spec"),
        path="revised_experiments/configs/work9_capacity_figure_spec_v1.json",
        sha256=FIGURE_SPEC_SHA256,
        bytes_=2540,
        context="presentation correction figure spec",
    )
    _exact_artifact_mapping(
        correction.get("superseded_v1_figure_manifest"),
        path=V1_FIGURE_MANIFEST_RELATIVE.as_posix(),
        sha256=V1_FIGURE_MANIFEST_SHA256,
        bytes_=9718,
        context="presentation correction v1 manifest",
    )
    _exact_artifact_mapping(
        correction.get("superseded_v1_hash_manifest"),
        path="revised_experiments/artifacts/work9_capacity_figures_v1/hashes.sha256",
        sha256="59fff3a87f6d98ed0df76bca53d47f6701ca8e1fdb0475951fdfe0850cb4bba6",
        bytes_=2304,
        context="presentation correction v1 hash manifest",
    )
    _exact_artifact_mapping(
        correction.get("superseded_v1_renderer"),
        path="revised_experiments/src/paper2_v2/work9_capacity_figures.py",
        sha256="cc556ea79569e3b52f86c389df67c40f87857d0b0ea2ea71f67b67fce141e08d",
        bytes_=25782,
        context="presentation correction v1 renderer",
    )
    preservation = correction.get("preservation")
    required_false = {
        "aggregation_changes_allowed",
        "analysis_report_changes_allowed",
        "condition_metric_changes_allowed",
        "effect_ci_p_or_holm_changes_allowed",
        "figure_id_or_purpose_changes_allowed",
        "frozen_figure_spec_changes_allowed",
        "v1_bundle_changes_allowed",
        "v1_renderer_changes_allowed",
    }
    if (
        not isinstance(preservation, Mapping)
        or set(preservation) != required_false
        or any(preservation[key] is not False for key in required_false)
    ):
        raise CapacityFigureError("presentation correction scientific preservation changed")
    contract = correction.get("presentation_contract")
    if not isinstance(contract, Mapping) or set(contract) != set(AFFECTED_FIGURE_IDS):
        raise CapacityFigureError("presentation correction affected contract changed")
    condition_contract = contract["capacity_condition_macro_f1"]
    metric_contract = contract["capacity_metric_profile"]
    if (
        not isinstance(condition_contract, Mapping)
        or tuple(condition_contract.get("main_zoom_ylim", ()))
        != EXPECTED_ZOOM_LIMITS["macro_f1"]
        or tuple(condition_contract.get("full_scale_context_ylim", ()))
        != (0.0, 1.001)
        or not isinstance(metric_contract, Mapping)
        or tuple(metric_contract.get("full_scale_reference", ())) != (0.0, 1.0)
    ):
        raise CapacityFigureError("presentation correction macro-F1 axis contract changed")
    zoom = metric_contract.get("bounded_metric_zoom_ylim")
    if not isinstance(zoom, Mapping) or set(zoom) != set(EXPECTED_ZOOM_LIMITS):
        raise CapacityFigureError("presentation correction metric zoom inventory changed")
    for metric, expected in EXPECTED_ZOOM_LIMITS.items():
        if tuple(zoom[metric]) != expected:
            raise CapacityFigureError(f"presentation correction zoom changed: {metric}")
    disclosures = (
        condition_contract.get("required_disclosure"),
        metric_contract.get("required_disclosure"),
    )
    if any(
        not isinstance(value, str)
        or "full-scale" not in value.lower().replace("full scale", "full-scale")
        or "absolute effect magnitude" not in value.lower()
        for value in disclosures
    ):
        raise CapacityFigureError("presentation correction disclosure changed")

    if project_root is not None:
        root = project_root.resolve()
        for key, context in (
            ("analysis_report", "presentation correction analysis report"),
            ("frozen_figure_spec", "presentation correction frozen figure spec"),
            ("superseded_v1_figure_manifest", "presentation correction v1 manifest"),
            ("superseded_v1_hash_manifest", "presentation correction v1 hashes"),
            ("superseded_v1_renderer", "presentation correction v1 renderer"),
        ):
            value = correction[key]
            ImmutableReference.load(
                {**value, "row_count": None},
                project_root=root,
                context=context,
                ledger=False,
            )


def load_presentation_correction(project_root: str | Path) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    path = root / PRESENTATION_CORRECTION_RELATIVE
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != PRESENTATION_CORRECTION_SHA256
    ):
        raise CapacityFigureError("frozen presentation correction changed")
    correction = _read_json(path, "capacity figure presentation correction v2")
    verify_presentation_correction(correction, project_root=root)
    return correction


def _footer_v2(figure: plt.Figure, *, zoom_disclosure: str) -> None:
    figure.text(0.01, 0.052, EVIDENCE_LABEL, fontsize=8.1, color="#333333")
    figure.text(
        0.01,
        0.031,
        "Descriptive capacity sensitivity; within-corpus and source-confounded.",
        fontsize=7.8,
        color="#555555",
    )
    figure.text(0.01, 0.010, zoom_disclosure, fontsize=7.6, color="#555555")


def _plot_seed_summary(
    ax: plt.Axes,
    *,
    lookup: Mapping[tuple[str, int, int | None], Mapping[str, Any]],
    metric: str,
) -> None:
    for index, condition in enumerate(CONDITIONS):
        values = np.asarray(
            [float(lookup[(condition, seed, None)][metric]) for seed in SEEDS]
        )
        ax.scatter(
            np.full(len(values), index) + np.linspace(-0.11, 0.11, len(values)),
            values,
            color=COLORS[condition],
            alpha=0.45,
            s=30,
        )
        ax.errorbar(
            index,
            values.mean(),
            yerr=[
                [max(0.0, float(values.mean() - values.min()))],
                [max(0.0, float(values.max() - values.mean()))],
            ],
            fmt="D",
            color=COLORS[condition],
            ecolor=COLORS[condition],
            capsize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )


def _capacity_condition_macro_f1_v2(
    report: Mapping[str, Any], correction: Mapping[str, Any]
) -> plt.Figure:
    lookup = _condition_metric_lookup(report)
    contract = correction["presentation_contract"]["capacity_condition_macro_f1"]
    zoom = tuple(float(value) for value in contract["main_zoom_ylim"])
    full = tuple(float(value) for value in contract["full_scale_context_ylim"])
    figure = plt.figure(figsize=(12.4, 5.8))
    grid = figure.add_gridspec(1, 2, width_ratios=(5.2, 1.15), wspace=0.17)
    ax = figure.add_subplot(grid[0, 0])
    context = figure.add_subplot(grid[0, 1])
    x = np.arange(len(CONDITIONS))
    _plot_seed_summary(ax, lookup=lookup, metric="macro_f1")
    ax.set_xticks(x, CONDITIONS, rotation=30, ha="right")
    ax.set_ylabel("Call-level macro-F1 (zoomed axis)")
    ax.set_ylim(*zoom)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_title(
        "Capacity-family call macro-F1: per-seed values, mean, and min–max",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.99,
        0.03,
        f"Zoom: {zoom[0]:.3f}–{zoom[1]:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": "#777777", "pad": 2.0},
    )

    means = np.asarray(
        [
            np.mean(
                [float(lookup[(condition, seed, None)]["macro_f1"]) for seed in SEEDS]
            )
            for condition in CONDITIONS
        ]
    )
    context.scatter(
        np.zeros(len(CONDITIONS)),
        means,
        marker="D",
        s=34,
        color=[COLORS[condition] for condition in CONDITIONS],
        edgecolor="black",
        linewidth=0.4,
    )
    context.set_xlim(-0.7, 0.7)
    context.set_ylim(*full)
    context.set_xticks([])
    context.set_yticks([0.0, 0.5, 1.0])
    context.set_title("Full-scale context\n(0–1)", fontsize=9.2, fontweight="bold")
    context.grid(axis="y", alpha=0.2)
    context.text(
        0.5,
        0.08,
        "Same means\nas zoomed panel",
        transform=context.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    _footer_v2(
        figure,
        zoom_disclosure=(
            "Main y-axis is explicitly truncated for legibility; the adjacent 0–1 panel preserves full-scale context. "
            "Whiskers are observed seed min–max, not confidence intervals; visual distance is not absolute effect magnitude."
        ),
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.22)
    return figure


def _capacity_metric_profile_v2(
    report: Mapping[str, Any], correction: Mapping[str, Any]
) -> plt.Figure:
    lookup = _condition_metric_lookup(report)
    zoom_contract = correction["presentation_contract"]["capacity_metric_profile"][
        "bounded_metric_zoom_ylim"
    ]
    panels = (
        ("macro_f1", "Macro-F1", True),
        ("sensitivity", "Sensitivity", True),
        ("specificity", "Specificity", True),
        ("roc_auc", "ROC AUC", True),
        ("pr_auc", "Average precision", True),
        ("brier", "Brier score", False),
        ("nll", "Negative log loss", False),
        ("ece_10", "ECE (10 bins)", False),
    )
    figure, axes = plt.subplots(2, 4, figsize=(15.2, 8.3), sharex=True)
    x = np.arange(len(CONDITIONS))
    zoom_axes: list[plt.Axes] = []
    for ax, (metric, title, zoomed) in zip(axes.flat, panels, strict=True):
        metric_values: dict[str, np.ndarray] = {}
        for index, condition in enumerate(CONDITIONS):
            values = np.asarray(
                [float(lookup[(condition, seed, None)][metric]) for seed in SEEDS]
            )
            metric_values[condition] = values
            ax.plot(
                [index, index],
                [values.min(), values.max()],
                color=COLORS[condition],
                linewidth=2,
            )
            ax.scatter(
                index,
                values.mean(),
                marker="D",
                s=34,
                color=COLORS[condition],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        ax.set_xticks(x, CONDITIONS, rotation=55, ha="right", fontsize=7.2)
        ax.grid(axis="y", alpha=0.2)
        if zoomed:
            lower, upper = (float(value) for value in zoom_contract[metric])
            ax.set_ylim(0.0, 1.005)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_title(f"{title} (higher ↑)", fontsize=10, fontweight="bold")
            inset = ax.inset_axes([0.12, 0.09, 0.82, 0.56])
            zoom_axes.append(inset)
            for index, condition in enumerate(CONDITIONS):
                values = metric_values[condition]
                inset.plot(
                    [index, index],
                    [values.min(), values.max()],
                    color=COLORS[condition],
                    linewidth=1.8,
                )
                inset.scatter(
                    index,
                    values.mean(),
                    marker="D",
                    s=25,
                    color=COLORS[condition],
                    edgecolor="black",
                    linewidth=0.35,
                    zorder=3,
                )
            inset.set_xlim(-0.5, len(CONDITIONS) - 0.5)
            inset.set_ylim(lower, upper)
            inset.set_xticks([])
            inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
            inset.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            inset.tick_params(axis="y", labelsize=6.2)
            inset.grid(axis="y", alpha=0.18)
            inset.set_title(
                f"Magnified same data/order: y={lower:g}–{upper:g}",
                fontsize=6.5,
                pad=2.0,
            )
        else:
            ax.set_ylim(bottom=0.0)
            ax.set_title(title, fontsize=10, fontweight="bold")
    figure.suptitle(
        "Capacity-family call-level metric profile: seed mean and min–max",
        x=0.02,
        y=0.985,
        ha="left",
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.948,
        "High-score parent axes show the full 0–1 domain; fixed insets repeat the same summaries at the printed magnification.",
        fontsize=8.5,
        color="#444444",
    )
    _footer_v2(
        figure,
        zoom_disclosure=(
            "Vertical ranges are observed seed min–max (not confidence intervals). Insets are descriptive magnifications; "
            "visual distances do not represent absolute effect magnitude."
        ),
    )
    setattr(figure, "_capacity_zoom_axes", tuple(zoom_axes))
    figure.tight_layout(rect=(0, 0.075, 1, 0.92))
    return figure


def build_capacity_figures_v2(
    report: Mapping[str, Any],
    figure_spec: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> dict[str, plt.Figure]:
    """Build exactly the frozen seven IDs with two disclosed presentation fixes."""

    verify_report_claim_boundary(report)
    verify_presentation_correction(correction)
    spec_rows = figure_spec.get("figures")
    if (
        figure_spec.get("evidence_label") != EVIDENCE_LABEL
        or not isinstance(spec_rows, list)
        or tuple(row.get("figure_id") for row in spec_rows) != FIGURE_IDS
        or tuple(figure_spec.get("formats", ())) != FORMATS
    ):
        raise CapacityFigureError("v2 renderer received a non-frozen figure specification")
    builders: tuple[
        tuple[str, Callable[[Mapping[str, Any]], plt.Figure]], ...
    ] = (
        ("capacity_parameter_match", _capacity_parameter_match),
        (
            "capacity_condition_macro_f1",
            lambda value: _capacity_condition_macro_f1_v2(value, correction),
        ),
        ("capacity_primary_effects", _capacity_primary_effects),
        ("primary_vs_capacity_condition_change", _primary_vs_capacity_condition_change),
        ("capacity_fold_seed_stability", _capacity_fold_seed_stability),
        (
            "capacity_metric_profile",
            lambda value: _capacity_metric_profile_v2(value, correction),
        ),
        ("capacity_compute_profile", _capacity_compute_profile),
    )
    if tuple(name for name, _builder in builders) != FIGURE_IDS:
        raise CapacityFigureError("v2 internal renderer inventory changed")
    figures = {name: builder(report) for name, builder in builders}
    if tuple(figures) != FIGURE_IDS:
        raise CapacityFigureError("v2 renderer did not produce exactly seven IDs")
    return figures


def render_capacity_figures_from_report_v2(
    project_root: str | Path,
    *,
    report: Mapping[str, Any],
    report_reference: Mapping[str, Any],
    figure_spec: Mapping[str, Any],
    correction: Mapping[str, Any],
    output_directory: str | Path,
    synthetic_test_mode: bool = False,
) -> Path:
    """Atomically publish the presentation-corrected v2 bundle."""

    root = Path(project_root).resolve()
    raw_output = Path(output_directory)
    expected_status = (
        "synthetic_contract_test"
        if synthetic_test_mode
        else "secondary_sensitivity_complete"
    )
    if report.get("status") != expected_status:
        raise CapacityFigureError("v2 output mode differs from report status")
    if not synthetic_test_mode and report_reference.get("sha256") != PRODUCTION_ANALYSIS_REPORT_SHA256:
        raise CapacityFigureError("v2 production render requires the sealed analysis report")
    try:
        if synthetic_test_mode:
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=TEST_SANDBOX_ROOT_RELATIVE,
                context="synthetic capacity v2 figure output",
                require_new=True,
            )
        else:
            if raw_output != FIGURE_OUTPUT_RELATIVE:
                raise CapacityFigureError(
                    f"production v2 figure output is fixed at {FIGURE_OUTPUT_RELATIVE}"
                )
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=Path("revised_experiments/artifacts"),
                context="production capacity v2 figure output",
                require_new=True,
            )
    except (ValueError, FileExistsError) as error:
        raise CapacityFigureError(str(error)) from error

    figures = build_capacity_figures_v2(report, figure_spec, correction)
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
            manifest_rows.append(
                {
                    "figure_id": figure_id,
                    "inferential": bool(spec_row["inferential"]),
                    "purpose": str(spec_row["purpose"]),
                    "files": files,
                    "presentation_revision": (
                        "ceiling_zoom_disclosed_v2"
                        if figure_id in AFFECTED_FIGURE_IDS
                        else "unchanged_from_v1"
                    ),
                }
            )
        correction_path = root / PRESENTATION_CORRECTION_RELATIVE
        v1_manifest_path = root / V1_FIGURE_MANIFEST_RELATIVE
        renderer_source_path = root / (
            "revised_experiments/src/paper2_v2/work9_capacity_figures_v2.py"
        )
        renderer_cli_path = root / (
            "revised_experiments/scripts/generate_work9_capacity_figures_v2.py"
        )
        renderer_schema_path = root / FIGURE_MANIFEST_SCHEMA_RELATIVE
        renderer_tests_path = root / (
            "revised_experiments/tests/test_work9_capacity_figures_v2.py"
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "status": (
                "synthetic_contract_test"
                if synthetic_test_mode
                else "secondary_sensitivity_figures_presentation_corrected"
            ),
            "protocol_id": PROTOCOL_ID,
            "run_family": RUN_FAMILY,
            "evidence_label": EVIDENCE_LABEL,
            "analysis_report": dict(report_reference),
            "analysis_report_schema_version": REPORT_SCHEMA_VERSION,
            "analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "figure_spec_sha256": FIGURE_SPEC_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
            "presentation_correction": _artifact_record(
                correction_path, project_root=root
            ),
            "renderer_source": _artifact_record(
                renderer_source_path, project_root=root
            ),
            "renderer_cli": _artifact_record(
                renderer_cli_path, project_root=root
            ),
            "renderer_schema": _artifact_record(
                renderer_schema_path, project_root=root
            ),
            "renderer_tests": _artifact_record(
                renderer_tests_path, project_root=root
            ),
            "supersedes_v1_figure_manifest": _artifact_record(
                v1_manifest_path, project_root=root
            ),
            "supersession_scope": (
                "presentation_only_v1_remains_immutable_and_scientifically_valid"
            ),
            "scientific_content_unchanged": {
                "analysis_report_sha256": str(report_reference["sha256"]),
                "figure_ids_and_purposes": True,
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
            schema_path=root / FIGURE_MANIFEST_SCHEMA_RELATIVE,
            context="capacity v2 figure manifest",
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
            "Presentation-corrected v2 of the same seven frozen capacity figures. "
            "Scientific data and Gate 1/Gate 2 remain unchanged; immutable v1 is preserved.\n",
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


def render_capacity_figures_v2(
    project_root: str | Path,
    *,
    report_path: str | Path,
    expected_report_sha256: str,
    output_directory: str | Path,
) -> Path:
    """Production entry point; presentation-only v2 over the sealed report."""

    root = Path(project_root).resolve()
    if expected_report_sha256 != PRODUCTION_ANALYSIS_REPORT_SHA256:
        raise CapacityFigureError("v2 production report SHA differs from correction")
    report, figure_spec = load_capacity_analysis_report(
        root,
        report_path=report_path,
        expected_report_sha256=expected_report_sha256,
    )
    correction = load_presentation_correction(root)
    lexical = Path(report_path)
    absolute_report = lexical if lexical.is_absolute() else root / lexical
    absolute_report = absolute_report.resolve()
    report_reference = {
        "path": absolute_report.relative_to(root).as_posix(),
        "sha256": expected_report_sha256,
        "bytes": absolute_report.stat().st_size,
        "row_count": None,
    }
    return render_capacity_figures_from_report_v2(
        root,
        report=report,
        report_reference=report_reference,
        figure_spec=figure_spec,
        correction=correction,
        output_directory=output_directory,
        synthetic_test_mode=False,
    )


__all__ = [
    "AFFECTED_FIGURE_IDS",
    "EXPECTED_ZOOM_LIMITS",
    "FIGURE_MANIFEST_SCHEMA_RELATIVE",
    "FIGURE_OUTPUT_RELATIVE",
    "MANIFEST_SCHEMA_VERSION",
    "PRESENTATION_CORRECTION_RELATIVE",
    "PRESENTATION_CORRECTION_SHA256",
    "RENDERER_VERSION",
    "build_capacity_figures_v2",
    "load_presentation_correction",
    "render_capacity_figures_from_report_v2",
    "render_capacity_figures_v2",
    "verify_presentation_correction",
]
