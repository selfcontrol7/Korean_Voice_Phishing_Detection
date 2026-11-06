"""Scientifically corrected v3 Work 9 Phase-A figure pipeline.

The v2 implementation remains unchanged so its historical manifest can still
verify its exact source and script hashes.  This module reuses its audited
immutable-input loader and unchanged renderers, while correcting threshold-axis
handling and making the downstream/precomputed-feature efficiency scope
explicit.  It never opens outer-evaluation artifacts, prediction ledgers, or
checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import work9_figures as base


FIGURE_MANIFEST_SCHEMA = "paper2-v2-work9-phase-a-figure-manifest-2"
FIGURE_SPEC_RELATIVE = Path(
    "revised_experiments/configs/work9_phase_a_figure_spec_v2.json"
)
FIGURE_SCRIPT_RELATIVE = Path(
    "revised_experiments/scripts/generate_work9_figures_v3.py"
)
FIGURE_TEST_RELATIVE = Path("revised_experiments/tests/test_work9_figures_v3.py")
SUPERSEDED_GENERATIONS = (
    (
        "work9_phase_a_figures_v1",
        Path("revised_experiments/artifacts/work9_phase_a_figures_v1"),
        "9606519e508e9c20819f852428457dd078c9c24332f59f8387d895f94532cdda",
        "superseded_after_visual_audit",
    ),
    (
        "work9_phase_a_figures_v2",
        Path("revised_experiments/artifacts/work9_phase_a_figures_v2"),
        "e812861dd2827d3c659f32b9c6c7c5b760b8657561e689f73067edfd3f19d454",
        "superseded_after_scientific_qa",
    ),
)

# Re-export the audited contract helpers used by focused tests and the CLI.
CONDITIONS = base.CONDITIONS
SEEDS = base.SEEDS
FORMATS = base.FORMATS
COLORS = base.COLORS
DIAGNOSTIC_LABEL = base.DIAGNOSTIC_LABEL
OUTCOME_FREE_LABEL = base.OUTCOME_FREE_LABEL
SELECTION_INDEX_RELATIVE = base.SELECTION_INDEX_RELATIVE
FigurePipelineError = base.FigurePipelineError
PhaseABundle = base.PhaseABundle
PhaseARun = base.PhaseARun
load_phase_a_bundle = base.load_phase_a_bundle
pointwise_observed_summary = base.pointwise_observed_summary
sha256_file = base.sha256_file
_resolve_artifact_reference = base._resolve_artifact_reference
_validate_coordinate_grid = base._validate_coordinate_grid
_save_figure = base._save_figure


def render_protocol_pipeline(
    config: Mapping[str, Any], protocol: Mapping[str, Any]
) -> plt.Figure:
    """Render the v2 audited schematic with its count label simplified."""

    figure = base.render_protocol_pipeline(config, protocol)
    replacements = 0
    for axis in figure.axes:
        for text in axis.texts:
            value = text.get_text()
            if "validation 190–190 calls" in value:
                text.set_text(value.replace("validation 190–190 calls", "validation 190 calls"))
                replacements += 1
    if replacements != 1:
        plt.close(figure)
        raise FigurePipelineError("protocol schematic validation-count label was not unique")
    return figure


def render_threshold_response(bundle: PhaseABundle) -> plt.Figure:
    """Render all threshold curves without clipping and disclose axis variation."""

    fig, axes = plt.subplots(4, 2, figsize=(11.8, 13.2), sharex=True, sharey=False)
    for ax, condition in zip(axes.ravel(), CONDITIONS):
        runs = base._condition_runs(bundle, condition)
        grid = np.asarray(
            [float(row["threshold"]) for row in runs[0].threshold_candidates]
        )
        macro_f1 = np.asarray(
            [
                [float(row["metrics"]["macro_f1"]) for row in run.threshold_candidates]
                for run in runs
            ],
            dtype=float,
        )
        median = np.median(macro_f1, axis=0)
        q25 = np.quantile(macro_f1, 0.25, axis=0, method="linear")
        q75 = np.quantile(macro_f1, 0.75, axis=0, method="linear")
        finite_minimum = min(float(np.min(q25)), float(np.min(median)))
        margin = max(0.015, 0.03 * (1.0 - finite_minimum))
        y_floor = max(0.0, finite_minimum - margin)
        y_ceiling = 1.01
        if float(np.min(q25)) < y_floor or float(np.max(q75)) > y_ceiling:
            raise FigurePipelineError(f"{condition} threshold response would be clipped")

        ax.fill_between(
            grid,
            q25,
            q75,
            color=COLORS[condition],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(grid, median, color=COLORS[condition], lw=1.7)
        rug_height = max(0.008, 0.026 * (y_ceiling - y_floor))
        ax.vlines(
            [run.selected_threshold for run in runs],
            y_floor,
            y_floor + rug_height,
            color="#222222",
            alpha=0.36,
            lw=0.7,
        )
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.set_ylabel("Validation macro-F1")
        ax.set_xlim(0, 1)
        ax.set_ylim(y_floor, y_ceiling)
        ax.text(
            0.98,
            0.055,
            f"panel y-axis: {y_floor:.2f}–{y_ceiling:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#DDDDDD", "alpha": 0.86},
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("Candidate threshold")
    fig.suptitle(
        "Validation threshold-response diagnostics (99 frozen candidates)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.031,
        "Line = median and band = interquartile range across all 25 coordinates; rugs mark selected thresholds. "
        "Panel y-axis lower limits vary and are printed inside each panel—compare within-panel shape, not cross-panel vertical displacement.",
        ha="center",
        fontsize=7.3,
        color="#444444",
    )
    base._stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.057, 0.98, 0.96))
    return fig


def render_runtime_distribution(bundle: PhaseABundle) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.6, 6.5))
    base._distribution_plot(
        ax,
        bundle,
        lambda run: run.wall_time_seconds / 60.0,
        ylabel="Accumulated recorded epoch wall time (minutes)",
    )
    ax.set_title(
        "Recorded downstream training and validation epoch wall time",
        fontweight="bold",
    )
    ax.text(
        0.995,
        0.98,
        "Horizontal bars are medians; points are all 25 coordinates.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    fig.text(
        0.5,
        0.035,
        "Scope: downstream training, validation, and per-epoch selection bookkeeping using precomputed features. "
        "Excludes ASR/KR-SBERT feature extraction, invocation setup/final publication, and phone/mobile latency.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    base._stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.065, 0.98, 0.98))
    return fig


def render_parameter_counts(bundle: PhaseABundle) -> plt.Figure:
    counts: list[int] = []
    for condition in CONDITIONS:
        runs = base._condition_runs(bundle, condition)
        trainable = {run.trainable_parameters for run in runs}
        totals = {run.total_parameters for run in runs}
        if len(trainable) != 1 or len(totals) != 1:
            raise FigurePipelineError(f"{condition} parameter count varies across coordinates")
        trainable_count = trainable.pop()
        if trainable_count != totals.pop():
            raise FigurePipelineError(
                f"{condition} has non-trainable downstream parameters; figure label must change"
            )
        counts.append(trainable_count)
    fig, ax = plt.subplots(figsize=(12.3, 6.3))
    positions = np.arange(len(CONDITIONS))
    bars = ax.bar(
        positions,
        np.asarray(counts) / 1000.0,
        color=[COLORS[condition] for condition in CONDITIONS],
        alpha=0.88,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.2,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=7.1,
        )
    ax.set_xticks(positions, CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("Trainable downstream parameters (thousands)")
    ax.set_title(
        "Executed downstream trainable-model capacity (frozen KR-SBERT excluded)",
        fontweight="bold",
    )
    ax.set_ylim(0, max(counts) / 1000.0 + 50)
    fig.text(
        0.5,
        0.035,
        "Counts cover the executed pooling/fusion/classifier model only; frozen precomputed KR-SBERT, ASR, and mobile-application components are excluded.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    base._stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.065, 0.98, 0.98))
    return fig


def render_gpu_memory_distribution(bundle: PhaseABundle) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharex=True)
    mib = 1024.0 * 1024.0
    base._distribution_plot(
        axes[0],
        bundle,
        lambda run: run.peak_cuda_allocated_bytes / mib,
        ylabel="Peak downstream CUDA allocated memory (MiB)",
    )
    base._distribution_plot(
        axes[1],
        bundle,
        lambda run: run.peak_cuda_reserved_bytes / mib,
        ylabel="Peak downstream CUDA reserved memory (MiB)",
    )
    axes[0].set_title("Allocated", fontweight="bold")
    axes[1].set_title("Reserved", fontweight="bold")
    fig.suptitle(
        "Downstream Phase A peak GPU memory using precomputed features",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.043,
        "Scope: downstream training/validation with precomputed features; excludes ASR, KR-SBERT feature extraction, and phone/mobile memory. "
        "Bars are medians; points are all 25 coordinates.",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    base._stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.072, 0.98, 0.94))
    return fig


def _load_figure_spec(
    project_root: Path,
) -> tuple[base.ArtifactReference, Mapping[str, Any]]:
    path = project_root / FIGURE_SPEC_RELATIVE
    reference = base._artifact_for_file(path, project_root)
    raw = base._read_json(path, context="corrected pre-outer figure specification")
    if not isinstance(raw, Mapping):
        raise FigurePipelineError("figure specification is not an object")
    if (
        raw.get("schema_version") != "paper2-v2-work9-phase-a-figure-spec-2"
        or raw.get("status") != "specified_before_outer_evaluation"
        or raw.get("bundle_version") != "work9_phase_a_figures_v3"
        or raw.get("protocol_id") != base.PROTOCOL_ID
    ):
        raise FigurePipelineError("corrected figure specification identity/status is invalid")
    boundary = raw.get("diagnostic_boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("phase_a_label") != DIAGNOSTIC_LABEL
        or boundary.get("outcome_free_label") != OUTCOME_FREE_LABEL
        or boundary.get("outer_performance_claim_enabled") is not False
        or boundary.get("direct_outer_artifact_access_permitted") is not False
        or boundary.get("inferential_fold_seed_replicate_claim_enabled") is not False
    ):
        raise FigurePipelineError("figure specification weakens the diagnostic boundary")
    aggregation = raw.get("aggregation_rules")
    if not isinstance(aggregation, Mapping) or (
        aggregation.get("all_25_coordinates_per_condition_required") is not True
        or aggregation.get("cherry_picking_permitted") is not False
        or aggregation.get("last_observation_carried_forward") is not False
        or aggregation.get("confidence_intervals_or_p_values_in_phase_a_figures") is not False
        or aggregation.get("cross_panel_threshold_response_vertical_displacement_comparison_permitted") is not False
    ):
        raise FigurePipelineError("figure specification weakens aggregation/axis rules")
    scope = raw.get("efficiency_scope")
    if not isinstance(scope, Mapping) or (
        scope.get("model_scope")
        != "executed_downstream_trainable_model_with_frozen_precomputed_features"
        or scope.get("end_to_end_deployment_claim_enabled") is not False
    ):
        raise FigurePipelineError("figure specification does not constrain efficiency scope")
    export = raw.get("export")
    if not isinstance(export, Mapping) or tuple(export.get("formats", ())) != FORMATS:
        raise FigurePipelineError("figure specification export formats differ")
    figures = raw.get("figures")
    observed_ids = (
        [row.get("figure_id") for row in figures if isinstance(row, Mapping)]
        if isinstance(figures, list)
        else []
    )
    if observed_ids != list(base._expected_figure_ids("all")):
        raise FigurePipelineError("figure specification IDs/order differ")
    return reference, raw


def _render_inventory(
    mode: str,
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    bundle: PhaseABundle | None,
) -> Iterable[tuple[str, plt.Figure, str]]:
    if mode in {"outcome-free", "all"}:
        yield "protocol_pipeline", render_protocol_pipeline(config, protocol), OUTCOME_FREE_LABEL
        yield "architecture_comparison", base.render_architecture_comparison(config), OUTCOME_FREE_LABEL
    if mode not in {"phase-a", "all"}:
        return
    if bundle is None:
        raise FigurePipelineError("Phase-A rendering requires a verified selection bundle")
    yield "phase_a_learning_curves_aggregate", base.render_aggregate_learning_curves(bundle), DIAGNOSTIC_LABEL
    for condition in CONDITIONS:
        yield (
            f"phase_a_learning_curves_{condition}_all25",
            base.render_condition_small_multiples(bundle, condition),
            DIAGNOSTIC_LABEL,
        )
    yield "phase_a_epoch_and_stopping_distributions", base.render_epoch_distributions(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_selected_epoch_heatmaps", base.render_selected_epoch_heatmaps(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_selected_threshold_heatmaps", base.render_selected_threshold_heatmaps(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_selected_threshold_distribution", base.render_threshold_distribution(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_threshold_response", render_threshold_response(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_runtime_distribution", render_runtime_distribution(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_parameter_counts", render_parameter_counts(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_gpu_memory_distribution", render_gpu_memory_distribution(bundle), DIAGNOSTIC_LABEL


def _source_references(project_root: Path) -> list[base.ArtifactReference]:
    paths = (
        Path(base.__file__).resolve(),
        Path(__file__).resolve(),
        (project_root / FIGURE_SCRIPT_RELATIVE).resolve(),
        (project_root / FIGURE_TEST_RELATIVE).resolve(),
    )
    return [base._artifact_for_file(path, project_root) for path in paths]


def _supersession_mapping(project_root: Path) -> dict[str, Any]:
    generations: list[dict[str, Any]] = []
    for version, directory, expected_manifest_sha, expected_status in SUPERSEDED_GENERATIONS:
        manifest = base._artifact_for_file(project_root / directory / "figure_manifest.json", project_root)
        record = base._artifact_for_file(project_root / directory / "SUPERSEDED.json", project_root)
        if manifest.sha256 != expected_manifest_sha:
            raise FigurePipelineError(f"{version} manifest hash changed")
        raw = base._read_json(record.path, context=f"{version} supersession record")
        if (
            not isinstance(raw, Mapping)
            or raw.get("status") != expected_status
            or raw.get("superseded_manifest_sha256") != manifest.sha256
            or raw.get("outer_outcomes_accessed") is not False
        ):
            raise FigurePipelineError(f"{version} supersession record is invalid")
        generations.append(
            {
                "bundle_version": version,
                "reason_code": raw["reason_code"],
                "manifest": manifest.to_mapping(),
                "supersession_record": record.to_mapping(),
            }
        )
    return {"current_bundle_version": "work9_phase_a_figures_v3", "generations": generations}


def generate_work9_figures(
    project_root: str | Path,
    *,
    output_directory: str | Path,
    mode: str = "all",
) -> Path:
    """Generate the corrected v3 bundle atomically without overwriting."""

    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    if root not in output.parents:
        raise FigurePipelineError("figure output directory must be inside project root")
    if output.exists() or output.is_symlink():
        raise FigurePipelineError("refusing to overwrite an existing figure output path")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise FigurePipelineError("figure output parent must not be a symlink")

    spec_ref, spec = _load_figure_spec(root)
    config_ref, protocol_ref, config, protocol = base.load_outcome_free_protocol(root)
    bundle: PhaseABundle | None = None
    if mode in {"phase-a", "all"}:
        bundle = load_phase_a_bundle(root)
        declared_index = spec.get("selection_index", {})
        if (
            declared_index.get("path") != bundle.selection_index_ref.serialized_path
            or declared_index.get("sha256") != bundle.selection_index_ref.sha256
        ):
            raise FigurePipelineError("selection index differs from corrected figure spec")
        if bundle.config_ref.sha256 != config_ref.sha256 or bundle.fold_protocol_ref.sha256 != protocol_ref.sha256:
            raise FigurePipelineError("protocol inputs differ between loaders")

    source_refs = _source_references(root)
    supersession = _supersession_mapping(root)
    formats = tuple(spec["export"]["formats"])
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        figure_directory = staging / "figures"
        figure_directory.mkdir()
        records: list[dict[str, Any]] = []
        generated_ids: list[str] = []
        base._configure_style()
        for figure_id, figure, boundary_label in _render_inventory(
            mode, config=config, protocol=protocol, bundle=bundle
        ):
            generated_ids.append(figure_id)
            files: dict[str, Any] = {}
            for staged_path in _save_figure(figure, figure_directory / figure_id, formats):
                final_path = output / "figures" / staged_path.name
                files[staged_path.suffix.removeprefix(".")] = base._output_ref_mapping(
                    staged_path, final_path, root
                )
            records.append(
                {
                    "figure_id": figure_id,
                    "diagnostic_boundary_label": boundary_label,
                    "files": files,
                }
            )
        if tuple(generated_ids) != base._expected_figure_ids(mode):
            raise FigurePipelineError("rendered inventory differs from corrected specification")

        phase_inputs: list[dict[str, Any]] = []
        if bundle is not None:
            phase_inputs = [
                {
                    "run_id": run.run_id,
                    "condition": run.condition,
                    "outer_fold": run.outer_fold,
                    "seed": run.seed,
                    "epoch_trace": run.epoch_trace_ref.to_mapping(),
                    "threshold_trace": run.threshold_trace_ref.to_mapping(),
                    "selection_run_record": run.run_record_ref.to_mapping(),
                }
                for run in bundle.runs
            ]
        material = {
            "figure_spec_sha256": spec_ref.sha256,
            "selection_index_sha256": None if bundle is None else bundle.selection_index_ref.sha256,
            "source_sha256": [reference.sha256 for reference in source_refs],
            "superseded_manifest_sha256": [
                row["manifest"]["sha256"] for row in supersession["generations"]
            ],
            "mode": mode,
        }
        manifest = {
            "schema_version": FIGURE_MANIFEST_SCHEMA,
            "protocol_id": base.PROTOCOL_ID,
            "bundle_version": "work9_phase_a_figures_v3",
            "status": "complete",
            "generation_id": hashlib.sha256(base._json_bytes(material)).hexdigest(),
            "mode": mode,
            "supersedes": supersession,
            "diagnostic_boundary": spec["diagnostic_boundary"],
            "efficiency_scope": spec["efficiency_scope"],
            "aggregation_rules": spec["aggregation_rules"],
            "data_access_declaration": {
                "direct_outer_artifacts_opened": False,
                "outer_prediction_ledgers_opened": False,
                "outer_outcomes_used": False,
                "checkpoints_opened": False,
                "validation_prediction_ledgers_opened": False,
                "phase_a_inputs_limited_to_indexed_epoch_threshold_and_run_records": bundle is not None,
                "fold_protocol_metadata_used_for_outcome_free_counts": True,
            },
            "environment": base._package_versions(),
            "sources": [reference.to_mapping() for reference in source_refs],
            "inputs": {
                "figure_spec": spec_ref.to_mapping(),
                "selection_index": None if bundle is None else bundle.selection_index_ref.to_mapping(),
                "work8_config": config_ref.to_mapping(),
                "fold_protocol_metadata": protocol_ref.to_mapping(),
                "phase_a_artifacts": phase_inputs,
            },
            "figure_count": len(records),
            "file_count": sum(len(record["files"]) for record in records),
            "figures": records,
        }
        manifest_path = staging / "figure_manifest.json"
        manifest_path.write_bytes(base._json_bytes(manifest))
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if output.exists() or output.is_symlink():
            raise FigurePipelineError("output appeared during generation; refusing overwrite")
        os.rename(staging, output)
        return output / "figure_manifest.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_figure_manifest(
    project_root: str | Path, manifest_path: str | Path
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise FigurePipelineError("corrected figure manifest is missing or symlinked")
    raw = base._read_json(path, context="corrected figure manifest")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != FIGURE_MANIFEST_SCHEMA:
        raise FigurePipelineError("unsupported corrected figure manifest")
    if (
        raw.get("status") != "complete"
        or raw.get("protocol_id") != base.PROTOCOL_ID
        or raw.get("bundle_version") != "work9_phase_a_figures_v3"
    ):
        raise FigurePipelineError("corrected figure manifest identity/status differs")
    for source in raw.get("sources", []):
        _resolve_artifact_reference(source, root, context="v3 figure source")
    supersedes = raw.get("supersedes", {})
    generations = supersedes.get("generations", []) if isinstance(supersedes, Mapping) else []
    if len(generations) != 2:
        raise FigurePipelineError("v3 manifest does not bind both superseded generations")
    for generation in generations:
        _resolve_artifact_reference(
            generation.get("manifest", {}), root, context="superseded figure manifest"
        )
        _resolve_artifact_reference(
            generation.get("supersession_record", {}),
            root,
            context="figure supersession record",
        )
    inputs = raw.get("inputs", {})
    for name in ("figure_spec", "work8_config", "fold_protocol_metadata"):
        _resolve_artifact_reference(inputs.get(name, {}), root, context=f"v3 input {name}")
    if inputs.get("selection_index") is not None:
        _resolve_artifact_reference(
            inputs["selection_index"], root, context="v3 selection index"
        )
    for run in inputs.get("phase_a_artifacts", []):
        for name in ("epoch_trace", "threshold_trace", "selection_run_record"):
            _resolve_artifact_reference(
                run.get(name, {}), root, context=f"v3 input {run.get('run_id')} {name}"
            )
    files_seen = 0
    for record in raw.get("figures", []):
        for extension, reference in record.get("files", {}).items():
            if extension not in FORMATS:
                raise FigurePipelineError("v3 manifest contains an unexpected format")
            _resolve_artifact_reference(reference, root, context=f"v3 figure {record.get('figure_id')}")
            files_seen += 1
    if files_seen != raw.get("file_count") or len(raw.get("figures", [])) != raw.get("figure_count"):
        raise FigurePipelineError("v3 figure/file count mismatch")
    declaration = raw.get("data_access_declaration", {})
    if any(
        declaration.get(key) is not False
        for key in (
            "direct_outer_artifacts_opened",
            "outer_prediction_ledgers_opened",
            "outer_outcomes_used",
            "checkpoints_opened",
            "validation_prediction_ledgers_opened",
        )
    ):
        raise FigurePipelineError("v3 manifest weakens the no-outer boundary")
    scope = raw.get("efficiency_scope", {})
    if scope.get("end_to_end_deployment_claim_enabled") is not False:
        raise FigurePipelineError("v3 manifest enables an end-to-end efficiency claim")
    return raw

