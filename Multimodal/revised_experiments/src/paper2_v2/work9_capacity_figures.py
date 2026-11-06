"""Seven-figure renderer for the frozen capacity-matched sensitivity.

The renderer consumes only the sealed capacity analysis report.  It never
opens checkpoints or prediction ledgers and exposes no user-supplied titles,
claim labels, comparison lists, or gate decisions.  Consequently the seven
pre-outcome figure identifiers and their secondary-sensitivity interpretation
cannot be expanded or relabelled at render time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import sha256_file  # noqa: E402
from .work9_capacity_registry import (  # noqa: E402
    _reject_symlink_components,
    validate_isolated_relative_root,
)
from .work9_capacity_types import TEST_SANDBOX_ROOT_RELATIVE  # noqa: E402
from .work9_capacity_analysis import (  # noqa: E402
    ANALYSIS_REPORT_SCHEMA_RELATIVE,
    CAPACITY_ANALYSIS_PROTOCOL_SHA256,
    COMPARISONS,
    CONDITIONS,
    EVIDENCE_LABEL,
    EXECUTION_ADDENDUM_SHA256,
    FIGURE_IDS,
    FIGURE_SPEC_SHA256,
    PROTOCOL_ID,
    REPORT_SCHEMA_VERSION,
    RUN_FAMILY,
    SEEDS,
    CapacityAnalysisError,
    ImmutableReference,
    _freeze_tree,
    _read_json,
    _restore_tree_for_cleanup,
    _validate_schema,
    load_frozen_capacity_policies,
    verify_report_claim_boundary,
)


RENDERER_VERSION = "paper2-v2-work9-capacity-figure-renderer-1"
MANIFEST_SCHEMA_VERSION = "paper2-v2-work9-capacity-figure-manifest-1"
FIGURE_MANIFEST_SCHEMA_RELATIVE = Path(
    "revised_experiments/registry/schemas/work9_capacity/"
    "work9_capacity_figure_manifest.schema.json"
)
FORMATS = ("svg", "pdf", "png_300dpi")
FIGURE_OUTPUT_RELATIVE = Path(
    "revised_experiments/artifacts/work9_capacity_figures_v1"
)
COLORS = {
    "T": "#4C78A8",
    "A-mean": "#72B7B2",
    "A-query": "#54A24B",
    "A-temp": "#B279A2",
    "AT-concat": "#F58518",
    "AT-gate": "#E45756",
    "AT-cross": "#7A5195",
    "AT-cross-randtrain": "#9D755D",
}
COMPARISON_COLORS = ("#3B528B", "#21918C", "#5EC962")


class CapacityFigureError(CapacityAnalysisError):
    """Raised when report lineage or the seven-figure inventory drifts."""


def _load_capacity_report_schema(
    report: Mapping[str, Any], project_root: Path
) -> None:
    _validate_schema(
        report,
        schema_path=project_root / ANALYSIS_REPORT_SCHEMA_RELATIVE,
        context="capacity analysis report for figures",
    )
    verify_report_claim_boundary(report)


def load_capacity_analysis_report(
    project_root: str | Path,
    *,
    report_path: str | Path,
    expected_report_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load a report and revalidate its frozen policies and index lineage."""

    root = Path(project_root).resolve()
    policies = load_frozen_capacity_policies(root)
    raw_path = Path(report_path)
    if raw_path == Path(".") or ".." in raw_path.parts:
        raise CapacityFigureError("analysis report path is lexically unsafe")
    lexical_path = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        _reject_symlink_components(lexical_path, stop=root)
    except ValueError as error:
        raise CapacityFigureError(
            "analysis report path contains a symlink component"
        ) from error
    path = lexical_path.resolve()
    if root not in path.parents or path.is_symlink() or not path.is_file():
        raise CapacityFigureError("analysis report must be a regular project file")
    if sha256_file(path) != expected_report_sha256:
        raise CapacityFigureError("analysis report SHA-256 differs from release")
    report = _read_json(path, "capacity analysis report")
    _load_capacity_report_schema(report, root)
    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping):
        raise CapacityFigureError("analysis report lineage is malformed")
    if (
        lineage.get("analysis_protocol_sha256") != CAPACITY_ANALYSIS_PROTOCOL_SHA256
        or lineage.get("figure_spec_sha256") != FIGURE_SPEC_SHA256
        or lineage.get("execution_addendum_sha256") != EXECUTION_ADDENDUM_SHA256
    ):
        raise CapacityFigureError("analysis report policy lineage changed")
    # The renderer does not reopen outcomes, but it does re-hash both release
    # indexes named by the report so figure lineage cannot float.
    ImmutableReference.load(
        lineage.get("selection_index"),
        project_root=root,
        context="figure report selection index",
        ledger=False,
    )
    ImmutableReference.load(
        lineage.get("outer_index"),
        project_root=root,
        context="figure report outer index",
        ledger=False,
    )
    figure_spec = policies.figure_spec
    if tuple(row.get("figure_id") for row in figure_spec["figures"]) != FIGURE_IDS:
        raise CapacityFigureError("figure specification inventory changed")
    return report, figure_spec


def _footer(figure: plt.Figure, *, inferential: bool = False) -> None:
    figure.text(0.01, 0.034, EVIDENCE_LABEL, fontsize=8.1, color="#333333")
    detail = (
        "Secondary clustered inference: only the frozen three comparisons; completed Gate 1/Gate 2 are unchanged."
        if inferential
        else "Descriptive capacity sensitivity; within-corpus and source-confounded."
    )
    figure.text(0.01, 0.012, detail, fontsize=7.8, color="#555555")


def _profile_lookup(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("parameter_profile")
    if not isinstance(rows, list):
        raise CapacityFigureError("parameter profile is malformed")
    lookup = {str(row["condition"]): row for row in rows if isinstance(row, Mapping)}
    if tuple(lookup) != CONDITIONS:
        # Dict order is part of the frozen display order.
        raise CapacityFigureError("parameter profile order/inventory changed")
    return lookup


def _condition_metric_lookup(
    report: Mapping[str, Any],
) -> dict[tuple[str, int, int | None], Mapping[str, Any]]:
    rows = report.get("condition_metrics")
    if not isinstance(rows, list) or len(rows) != 240:
        raise CapacityFigureError("condition metrics must contain 240 rows")
    lookup: dict[tuple[str, int, int | None], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CapacityFigureError("condition metric row is malformed")
        fold = row.get("outer_fold")
        key = (
            str(row.get("condition")),
            int(row.get("seed")),
            None if fold is None else int(fold),
        )
        if key in lookup:
            raise CapacityFigureError(f"duplicate condition metric row: {key}")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise CapacityFigureError(f"condition metric values are malformed: {key}")
        lookup[key] = metrics
    expected = {
        (condition, seed, fold)
        for condition in CONDITIONS
        for seed in SEEDS
        for fold in (None, 0, 1, 2, 3, 4)
    }
    if set(lookup) != expected:
        raise CapacityFigureError("condition metric coordinate set is incomplete")
    return lookup


def _capacity_parameter_match(report: Mapping[str, Any]) -> plt.Figure:
    profile = _profile_lookup(report)
    values = np.asarray(
        [float(profile[condition]["trainable_parameters"]) for condition in CONDITIONS]
    )
    target = float(profile["AT-cross"]["target_parameters"])
    figure, ax = plt.subplots(figsize=(11.2, 5.8))
    x = np.arange(len(CONDITIONS))
    ax.bar(x, values, color=[COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.5)
    ax.axhline(target, color="#222222", linestyle="--", linewidth=1.3, label="AT-cross target")
    for index, condition in enumerate(CONDITIONS):
        gap = 100.0 * float(profile[condition]["relative_absolute_gap"])
        ax.text(index, values[index] + values.max() * 0.008, f"{gap:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, CONDITIONS, rotation=30, ha="right")
    ax.set_ylabel("Trainable parameters")
    ax.set_title("Capacity matching: exact parameter counts and residual absolute gap", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _capacity_condition_macro_f1(report: Mapping[str, Any]) -> plt.Figure:
    lookup = _condition_metric_lookup(report)
    figure, ax = plt.subplots(figsize=(11.2, 5.8))
    x = np.arange(len(CONDITIONS))
    for index, condition in enumerate(CONDITIONS):
        values = np.asarray(
            [float(lookup[(condition, seed, None)]["macro_f1"]) for seed in SEEDS]
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
    ax.set_xticks(x, CONDITIONS, rotation=30, ha="right")
    ax.set_ylabel("Call-level macro-F1")
    ax.set_ylim(0.0, 1.035)
    ax.set_title("Capacity-family call macro-F1: per-seed values, mean, and min–max", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _capacity_primary_effects(report: Mapping[str, Any]) -> plt.Figure:
    rows = report["secondary_comparisons"]
    figure, ax = plt.subplots(figsize=(10.8, 4.8))
    labels = ("vs A-query", "vs AT-concat", "vs AT-gate")
    extrema: list[float] = []
    for index, row in enumerate(rows):
        observed = float(row["observed_effect"])
        lower = float(row["bootstrap"]["ci_lower"])
        upper = float(row["bootstrap"]["ci_upper"])
        extrema.extend((lower, upper))
        ax.errorbar(
            observed,
            index,
            xerr=[[observed - lower], [upper - observed]],
            fmt="D",
            markersize=7,
            capsize=5,
            color=COMPARISON_COLORS[index],
            linewidth=2.2,
        )
    limit = max(max(abs(value) for value in extrema) * 1.45, 0.01)
    ax.set_xlim(-limit, limit)
    for index, row in enumerate(rows):
        adjusted = float(row["multiplicity"]["adjusted_p_value"])
        ax.text(
            0.985,
            index,
            f"Holm p={adjusted:.4g}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(3), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired macro-F1 difference (AT-cross − comparator)")
    ax.set_title("Frozen secondary capacity effects with 95% relation-group clustered CIs", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    _footer(figure, inferential=True)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def _primary_vs_capacity_condition_change(report: Mapping[str, Any]) -> plt.Figure:
    rows = report["descriptive_primary_vs_capacity"]
    if [row["condition"] for row in rows] != list(CONDITIONS):
        raise CapacityFigureError("primary-vs-capacity condition order changed")
    changes = np.asarray(
        [float(row["metrics"]["macro_f1"]["capacity_minus_primary"]) for row in rows]
    )
    figure, ax = plt.subplots(figsize=(11.2, 5.8))
    x = np.arange(len(CONDITIONS))
    ax.bar(x, changes, color=[COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, CONDITIONS, rotation=30, ha="right")
    ax.set_ylabel("Capacity − primary H=256 macro-F1")
    ax.set_title("Descriptive within-condition change from the completed primary matrix", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def _capacity_fold_seed_stability(report: Mapping[str, Any]) -> plt.Figure:
    rows = report["secondary_comparisons"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharex=True, sharey=True)
    values: list[np.ndarray] = []
    for row in rows:
        grid = np.asarray(
            [
                [
                    float(row["fold_stability"]["per_fold_per_seed_differences"][str(fold)][seed_index])
                    for seed_index, _seed in enumerate(SEEDS)
                ]
                for fold in range(5)
            ]
        )
        values.append(grid)
    maximum = max(float(np.max(np.abs(grid))) for grid in values)
    maximum = maximum if maximum > 0 else 1.0
    for ax, grid, pair in zip(axes, values, COMPARISONS, strict=True):
        image = ax.imshow(grid, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
        ax.set_title(f"AT-cross − {pair[1]}", fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(5), SEEDS, rotation=45)
        ax.set_yticks(np.arange(5), range(5))
        ax.set_xlabel("Seed")
    axes[0].set_ylabel("Outer fold")
    figure.colorbar(image, ax=axes, shrink=0.72, label="Macro-F1 difference")
    figure.suptitle("Fold/seed direction and dispersion for the frozen three effects", x=0.02, ha="left", fontweight="bold")
    _footer(figure)
    figure.subplots_adjust(left=0.06, right=0.92, top=0.84, bottom=0.19, wspace=0.12)
    return figure


def _capacity_metric_profile(report: Mapping[str, Any]) -> plt.Figure:
    lookup = _condition_metric_lookup(report)
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
    figure, axes = plt.subplots(2, 4, figsize=(15.2, 7.8), sharex=True)
    x = np.arange(len(CONDITIONS))
    for ax, (metric, title, bounded) in zip(axes.flat, panels, strict=True):
        for index, condition in enumerate(CONDITIONS):
            values = np.asarray(
                [float(lookup[(condition, seed, None)][metric]) for seed in SEEDS]
            )
            ax.plot([index, index], [values.min(), values.max()], color=COLORS[condition], linewidth=2)
            ax.scatter(index, values.mean(), marker="D", s=34, color=COLORS[condition], edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x, CONDITIONS, rotation=55, ha="right", fontsize=7.2)
        ax.grid(axis="y", alpha=0.2)
        if bounded:
            ax.set_ylim(0.0, 1.035)
        else:
            ax.set_ylim(bottom=0.0)
    figure.suptitle("Capacity-family call-level metric profile: seed mean and min–max", x=0.02, ha="left", fontweight="bold")
    _footer(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    return figure


def _capacity_compute_profile(report: Mapping[str, Any]) -> plt.Figure:
    rows = report["compute_profile"]
    if [row["condition"] for row in rows] != list(CONDITIONS):
        raise CapacityFigureError("compute profile order changed")
    parameters = np.asarray([float(row["trainable_parameters"]) for row in rows])
    wall = np.asarray([float(row["wall_time_seconds"]["mean"]) for row in rows])
    memory = np.asarray([float(row["peak_cuda_allocated_bytes"]["mean"]) / (1024**2) for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.8))
    x = np.arange(len(CONDITIONS))
    for ax, values, title, ylabel in (
        (axes[0], parameters, "Trainable parameters", "Parameters"),
        (axes[1], wall, "Recorded downstream wall time", "Seconds / coordinate"),
        (axes[2], memory, "Recorded peak CUDA allocation", "MiB / coordinate"),
    ):
        ax.bar(x, values, color=[COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.4)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, CONDITIONS, rotation=55, ha="right", fontsize=7.3)
        ax.grid(axis="y", alpha=0.2)
    figure.suptitle("Downstream compute profile (ASR, encoder extraction, and mobile operation excluded)", x=0.02, ha="left", fontweight="bold")
    _footer(figure)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure


def build_capacity_figures(
    report: Mapping[str, Any], figure_spec: Mapping[str, Any]
) -> dict[str, plt.Figure]:
    """Build exactly the seven frozen figures from a validated report."""

    verify_report_claim_boundary(report)
    spec_rows = figure_spec.get("figures")
    if (
        figure_spec.get("evidence_label") != EVIDENCE_LABEL
        or not isinstance(spec_rows, list)
        or tuple(row.get("figure_id") for row in spec_rows) != FIGURE_IDS
        or tuple(figure_spec.get("formats", ())) != FORMATS
    ):
        raise CapacityFigureError("renderer received a non-frozen figure specification")
    builders: tuple[tuple[str, Callable[[Mapping[str, Any]], plt.Figure]], ...] = (
        ("capacity_parameter_match", _capacity_parameter_match),
        ("capacity_condition_macro_f1", _capacity_condition_macro_f1),
        ("capacity_primary_effects", _capacity_primary_effects),
        ("primary_vs_capacity_condition_change", _primary_vs_capacity_condition_change),
        ("capacity_fold_seed_stability", _capacity_fold_seed_stability),
        ("capacity_metric_profile", _capacity_metric_profile),
        ("capacity_compute_profile", _capacity_compute_profile),
    )
    if tuple(name for name, _builder in builders) != FIGURE_IDS:
        raise CapacityFigureError("internal renderer inventory changed")
    figures = {name: builder(report) for name, builder in builders}
    if tuple(figures) != FIGURE_IDS:
        raise CapacityFigureError("renderer did not produce exactly seven figure IDs")
    return figures


def _artifact_record(
    path: Path, *, project_root: Path, final_path: Path | None = None
) -> dict[str, Any]:
    serialized = path if final_path is None else final_path
    return {
        "path": serialized.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "row_count": None,
    }


def render_capacity_figures_from_report(
    project_root: str | Path,
    *,
    report: Mapping[str, Any],
    report_reference: Mapping[str, Any],
    figure_spec: Mapping[str, Any],
    output_directory: str | Path,
    synthetic_test_mode: bool = False,
) -> Path:
    """Atomically publish exactly seven IDs in the three frozen formats."""

    root = Path(project_root).resolve()
    raw_output = Path(output_directory)
    expected_status = (
        "synthetic_contract_test" if synthetic_test_mode else "secondary_sensitivity_complete"
    )
    if report.get("status") != expected_status:
        raise CapacityFigureError("figure output mode differs from report status")
    try:
        if synthetic_test_mode:
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=TEST_SANDBOX_ROOT_RELATIVE,
                context="synthetic capacity figure output",
                require_new=True,
            )
        else:
            if raw_output != FIGURE_OUTPUT_RELATIVE:
                raise CapacityFigureError(
                    f"production figure output is fixed at {FIGURE_OUTPUT_RELATIVE}"
                )
            destination = validate_isolated_relative_root(
                root,
                raw_output,
                allowed_parent=Path("revised_experiments/artifacts"),
                context="production capacity figure output",
                require_new=True,
            )
    except (ValueError, FileExistsError) as error:
        raise CapacityFigureError(str(error)) from error
    figures = build_capacity_figures(report, figure_spec)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    figure_directory = staging / "figures"
    figure_directory.mkdir(parents=True, exist_ok=False)
    try:
        manifest_rows: list[dict[str, Any]] = []
        spec_lookup = {row["figure_id"]: row for row in figure_spec["figures"]}
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
                }
            )
        if tuple(row["figure_id"] for row in manifest_rows) != FIGURE_IDS:
            raise CapacityFigureError("saved figure inventory changed")
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "status": (
                "synthetic_contract_test"
                if report.get("status") == "synthetic_contract_test"
                else "secondary_sensitivity_figures_complete"
            ),
            "protocol_id": PROTOCOL_ID,
            "run_family": RUN_FAMILY,
            "evidence_label": EVIDENCE_LABEL,
            "analysis_report": dict(report_reference),
            "analysis_report_schema_version": REPORT_SCHEMA_VERSION,
            "analysis_protocol_sha256": CAPACITY_ANALYSIS_PROTOCOL_SHA256,
            "figure_spec_sha256": FIGURE_SPEC_SHA256,
            "execution_addendum_sha256": EXECUTION_ADDENDUM_SHA256,
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
            context="capacity figure manifest",
        )
        (staging / "figure_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        hashes = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "hashes.sha256":
                hashes.append(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}")
        (staging / "hashes.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
        (staging / "READ_ONLY_NOTICE.txt").write_text(
            "Seven frozen capacity-sensitivity figures only; no Gate 1/Gate 2 reopening.\n",
            encoding="utf-8",
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


def render_capacity_figures(
    project_root: str | Path,
    *,
    report_path: str | Path,
    expected_report_sha256: str,
    output_directory: str | Path,
) -> Path:
    """Production renderer; reads no model/prediction outcome artifact."""

    root = Path(project_root).resolve()
    report, spec = load_capacity_analysis_report(
        root,
        report_path=report_path,
        expected_report_sha256=expected_report_sha256,
    )
    report_path_value = Path(report_path).resolve()
    report_reference = {
        "path": report_path_value.relative_to(root).as_posix(),
        "sha256": expected_report_sha256,
        "bytes": report_path_value.stat().st_size,
        "row_count": None,
    }
    return render_capacity_figures_from_report(
        root,
        report=report,
        report_reference=report_reference,
        figure_spec=spec,
        output_directory=output_directory,
        synthetic_test_mode=False,
    )


__all__ = [
    "CapacityFigureError",
    "FIGURE_IDS",
    "MANIFEST_SCHEMA_VERSION",
    "RENDERER_VERSION",
    "build_capacity_figures",
    "load_capacity_analysis_report",
    "render_capacity_figures",
    "render_capacity_figures_from_report",
]
