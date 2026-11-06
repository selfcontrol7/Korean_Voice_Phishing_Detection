"""Presentation-corrected v3 figures for the sealed Work 9 Gate-1 analysis.

This module is a deliberately narrow overlay on the frozen v1 scientific
figure core.  It inherits the pre-outcome figure inventory, data loading,
metric recomputation, aggregation rules, and inferential values unchanged.
Only three presentation details are changed: ceiling headroom in the metric
profile, a reserved annotation region in the primary-effect forest plot, and
an explicit warning that probability-density panels have panel-specific
y-axis limits.

The v1/v2-native artifacts are never modified.  V3 has its own specification,
renderer, CLI, tests, manifest schema, generation identity, and immutable
output directory.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paper2_v2 import work9_gate1_figures as base  # noqa: E402


FIGURE_SPEC_RELATIVE = Path(
    "revised_experiments/configs/work9_gate1_figure_spec_v3.json"
)
FIGURE_SCRIPT_RELATIVE = Path(
    "revised_experiments/scripts/generate_work9_gate1_figures_v3.py"
)
FIGURE_SOURCE_RELATIVE = Path(
    "revised_experiments/src/paper2_v2/work9_gate1_figures_v3.py"
)
FIGURE_TEST_RELATIVE = Path(
    "revised_experiments/tests/test_work9_gate1_figures_v3.py"
)
BASE_SPEC_RELATIVE = base.FIGURE_SPEC_RELATIVE
BASE_SOURCE_RELATIVE = base.FIGURE_SOURCE_RELATIVE
SUPERSEDED_MANIFEST_RELATIVE = Path(
    "revised_experiments/artifacts/work9_gate1_figures_v2_native/figure_manifest.json"
)

SPEC_SCHEMA = "paper2-v2-work9-gate1-figure-spec-3"
MANIFEST_SCHEMA = "paper2-v2-work9-gate1-figure-manifest-3"
SPEC_STATUS = "frozen_post_gate1_visual_correction"
GENERATION_ID = "work9_gate1_figures_v3"

PROTOCOL_ID = base.PROTOCOL_ID
FORMATS = base.FORMATS
CONDITIONS = base.CONDITIONS
SEEDS = base.SEEDS
FOLDS = base.FOLDS
COMPARISONS = base.COMPARISONS
COLORS = base.COLORS
COMPARISON_COLORS = base.COMPARISON_COLORS
CLAIM_LABEL = base.CLAIM_LABEL
DESCRIPTIVE_LABEL = base.DESCRIPTIVE_LABEL

ArtifactReference = base.ArtifactReference
CallPrediction = base.CallPrediction
Gate1FigureData = base.Gate1FigureData
Gate1FigureError = base.Gate1FigureError
compute_call_metrics = base.compute_call_metrics
sha256_file = base.sha256_file


VISUAL_CORRECTION = {
    "scope": "presentation_only",
    "scientific_values_changed": False,
    "figure_inventory_changed": False,
    "aggregation_changed": False,
    "inference_changed": False,
    "changes": [
        "metric_profile_upper_headroom_prevents_ceiling_marker_clipping",
        "primary_effect_annotation_region_prevents_ci_label_overlap",
        "probability_density_panels_disclose_panel_specific_y_axis_limits",
        "manifest_generation_identity_matches_v3_bundle",
    ],
}


def expected_figure_ids() -> tuple[str, ...]:
    """Return the unchanged precommitted eleven-figure inventory."""

    return base.expected_figure_ids()


def _read_json(path: Path, context: str) -> Any:
    return base._read_json(path, context)


def _reference(path: Path, root: Path) -> ArtifactReference:
    return base._file_reference(path, root)


def load_figure_spec(project_root: str | Path) -> Mapping[str, Any]:
    """Validate the v3 presentation-only correction overlay and bindings."""

    root = Path(project_root).resolve()
    spec = _read_json(root / FIGURE_SPEC_RELATIVE, "frozen Gate-1 v3 figure specification")
    if not isinstance(spec, Mapping) or (
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("status") != SPEC_STATUS
        or spec.get("protocol_id") != PROTOCOL_ID
    ):
        raise Gate1FigureError("v3 figure specification identity/status mismatch")
    if spec.get("visual_correction") != VISUAL_CORRECTION:
        raise Gate1FigureError("v3 visual-correction boundary changed")
    for section in ("upstream_bindings", "implementation_bindings"):
        rows = spec.get(section)
        if not isinstance(rows, list) or not rows:
            raise Gate1FigureError(f"v3 figure specification {section} is empty")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {"role", "path", "sha256"}:
                raise Gate1FigureError(f"v3 {section}[{index}] is malformed")
            path = base._safe_project_path(str(row["path"]), root, f"v3 {section}[{index}]")
            expected = base._require_sha256(row["sha256"], f"v3 {section}[{index}].sha256")
            if sha256_file(path) != expected:
                raise Gate1FigureError(f"v3 frozen binding changed: {row['role']}")
    figures = spec.get("figures")
    if not isinstance(figures, list) or [row.get("figure_id") for row in figures] != list(
        expected_figure_ids()
    ):
        raise Gate1FigureError("v3 figure family/order differs from the precommitted inventory")
    if spec.get("data_access", {}).get("checkpoints_permitted") is not False:
        raise Gate1FigureError("v3 figure spec must forbid checkpoint access")
    if spec.get("data_access", {}).get("outer_segment_ledgers_permitted") is not False:
        raise Gate1FigureError("v3 figure spec must forbid segment-ledger access")
    return spec


def load_gate1_figure_data(
    project_root: str | Path,
    *,
    gate1_directory: str | Path,
    expected_gate1_hash_manifest_sha256: str,
) -> Gate1FigureData:
    """Load fully revalidated evidence through the frozen v1 scientific core."""

    root = Path(project_root).resolve()
    load_figure_spec(root)
    return base.load_gate1_figure_data(
        root,
        gate1_directory=gate1_directory,
        expected_gate1_hash_manifest_sha256=expected_gate1_hash_manifest_sha256,
    )


def _footer(fig: plt.Figure, *, inferential: bool = False) -> None:
    fig.text(0.01, 0.036, CLAIM_LABEL, fontsize=8.3, color="#333333")
    second = (
        "Only the displayed 95% CIs are relation-group clustered; Holm p-values control the frozen three-test family"
        if inferential
        else DESCRIPTIVE_LABEL
    )
    fig.text(0.01, 0.012, second, fontsize=8.0, color="#555555")


def _metric_profile(data: Gate1FigureData) -> plt.Figure:
    """Render the unchanged profile values with visible ceiling headroom."""

    lookup = base._condition_lookup(data)
    panels = (
        ("macro_f1", "Macro-F1", "higher ↑"),
        ("sensitivity", "Sensitivity", "higher ↑"),
        ("specificity", "Specificity", "higher ↑"),
        ("roc_auc", "ROC AUC", "higher ↑"),
        ("pr_auc", "Average precision (AP)", "higher ↑"),
        ("brier", "Brier score", "lower ↓"),
        ("nll", "Negative log loss", "lower ↓"),
        ("ece_10", "ECE (10 bins)", "lower ↓"),
    )
    bounded_metrics = {"macro_f1", "sensitivity", "specificity", "roc_auc", "pr_auc"}
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.8), sharex=True)
    x = np.arange(len(CONDITIONS))
    for ax, (metric, title, direction) in zip(axes.flat, panels, strict=True):
        for index, condition in enumerate(CONDITIONS):
            values = np.asarray(
                [
                    lookup[(condition, seed, None)][metric]
                    for seed in SEEDS
                    if lookup[(condition, seed, None)][metric] is not None
                ],
                dtype=float,
            )
            if len(values):
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
                    s=38,
                    color=COLORS[condition],
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=3,
                    clip_on=False,
                )
        ax.set_title(f"{title} ({direction})", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x, CONDITIONS, rotation=55, ha="right", fontsize=7.5)
        if metric in bounded_metrics:
            ax.set_ylim(0.0, 1.035)
        elif metric != "nll":
            ax.set_ylim(bottom=0.0)
    fig.suptitle(
        "Outer-development condition profile: seed mean and min–max",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    _footer(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def _primary_effects(data: Gate1FigureData) -> plt.Figure:
    """Render the exact Gate-1 effects with a nonoverlapping p-value column."""

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    labels = (
        "vs A-query\n(incremental test)",
        "vs AT-concat\n(mechanism 1/2)",
        "vs AT-gate\n(mechanism 2/2)",
    )
    y = np.arange(3)
    extrema: list[float] = []
    for index, row in enumerate(data.comparisons):
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
            ecolor=COMPARISON_COLORS[index],
            linewidth=2.2,
        )
    maximum = max(abs(value) for value in extrema)
    limit = 1.38 * maximum if maximum > 0.0 else 1.0
    ax.set_xlim(-limit, limit)
    for index, row in enumerate(data.comparisons):
        holm_p = float(row["multiplicity"]["adjusted_p_value"])
        ax.text(
            0.985,
            index,
            f"Holm p={holm_p:.4g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.axhline(0.5, color="#777777", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired macro-F1 difference (AT-cross − comparator)")
    ax.set_title(
        "Preregistered Gate-1 effects with 95% relation-group clustered bootstrap CIs",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.22)
    _footer(fig, inferential=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _probability_distributions(data: Gate1FigureData) -> plt.Figure:
    """Render the unchanged per-seed densities with an explicit scale warning."""

    calls = base._calls_lookup(data)
    bins = np.linspace(0.0, 1.0, 21)
    centers = (bins[:-1] + bins[1:]) / 2.0
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.7), sharex=True)
    for ax, condition in zip(axes.flat, CONDITIONS, strict=True):
        for label, color, name in ((0, "#2E86AB", "true 0"), (1, "#C23B5A", "true 1")):
            seed_densities = []
            for seed in SEEDS:
                values = np.asarray(
                    [
                        call.probability
                        for call in calls[(condition, seed)]
                        if call.label == label
                    ],
                    dtype=float,
                )
                density, _ = np.histogram(values, bins=bins, density=True)
                seed_densities.append(density)
                ax.step(
                    centers,
                    density,
                    where="mid",
                    color=color,
                    linewidth=0.8,
                    alpha=0.12,
                )
            densities = np.vstack(seed_densities)
            mean_density = densities.mean(axis=0)
            ax.step(
                centers,
                mean_density,
                where="mid",
                color=color,
                linewidth=1.8,
                label=name,
            )
            ax.fill_between(
                centers,
                densities.min(axis=0),
                densities.max(axis=0),
                step="mid",
                color=color,
                alpha=0.08,
            )
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Predicted positive probability")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Threshold-free call-probability distributions by true label",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    fig.text(
        0.99,
        0.925,
        "Panel-specific y-axis limits; compare distribution shape, not peak height across conditions",
        ha="right",
        fontsize=7.8,
        color="#555555",
    )
    _footer(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    return fig


def _save_figure(
    fig: plt.Figure, base_path: Path, formats: Sequence[str]
) -> tuple[Path, ...]:
    outputs: list[Path] = []
    for extension in formats:
        path = base_path.with_suffix(f".{extension}")
        metadata: dict[str, Any] = {"Creator": "Paper2 Work9 Gate1 figure pipeline v3"}
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        outputs.append(path)
    plt.close(fig)
    return tuple(outputs)


def render_figures(
    data: Gate1FigureData,
    output_directory: Path,
    *,
    formats: Sequence[str] = FORMATS,
) -> dict[str, tuple[Path, ...]]:
    """Render the unchanged inventory with the three v3 presentation fixes."""

    if any(extension not in FORMATS for extension in formats) or not formats:
        raise Gate1FigureError("unsupported/empty export format family")
    base._validate_condition_records(data.condition_records)
    base._validate_comparisons(data.comparisons)
    output_directory.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "paper2-work9-gate1-v3",
        }
    )
    builders = {
        "outer_condition_macro_f1": base._condition_macro_f1,
        "outer_condition_metric_profile": _metric_profile,
        "gate1_primary_effects": _primary_effects,
        "gate1_seed_stability": base._seed_stability,
        "gate1_fold_stability": base._fold_stability,
        "outer_macro_f1_fold_seed": base._macro_f1_heatmaps,
        "outer_confusion_matrices": base._confusion_matrices,
        "outer_roc_curves": lambda bundle: base._curve_summary(bundle, kind="roc"),
        "outer_pr_curves": lambda bundle: base._curve_summary(bundle, kind="pr"),
        "outer_calibration_curves": base._calibration_curves,
        "outer_probability_distributions": _probability_distributions,
    }
    results: dict[str, tuple[Path, ...]] = {}
    for figure_id in expected_figure_ids():
        results[figure_id] = _save_figure(
            builders[figure_id](data), output_directory / figure_id, formats
        )
    return results


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def generate_gate1_figures(
    project_root: str | Path,
    *,
    gate1_directory: str | Path,
    expected_gate1_hash_manifest_sha256: str,
    output_directory: str | Path,
) -> Path:
    """Atomically generate and freeze the v3 bundle without touching v2."""

    root = Path(project_root).resolve()
    spec = load_figure_spec(root)
    data = load_gate1_figure_data(
        root,
        gate1_directory=gate1_directory,
        expected_gate1_hash_manifest_sha256=expected_gate1_hash_manifest_sha256,
    )
    if data.gate1_hash_manifest is None or data.outer_index is None:
        raise Gate1FigureError("real Gate-1 evidence references are required for v3 generation")
    output = Path(output_directory).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise Gate1FigureError("v3 figure output must remain inside project root") from error
    if output.exists():
        raise FileExistsError(f"v3 figure output already exists and is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        rendered = render_figures(data, staging / "figures")
        figure_specs = {str(row["figure_id"]): row for row in spec["figures"]}
        figure_rows = []
        for figure_id in expected_figure_ids():
            files = [_reference(path, staging) for path in rendered[figure_id]]
            figure_rows.append(
                {
                    "figure_id": figure_id,
                    "title": figure_specs[figure_id]["title"],
                    "analysis_role": figure_specs[figure_id]["analysis_role"],
                    "files": [reference.as_mapping() for reference in files],
                }
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "sealed_complete",
            "protocol_id": PROTOCOL_ID,
            "generation_id": GENERATION_ID,
            "figure_spec": _reference(root / FIGURE_SPEC_RELATIVE, root).as_mapping(),
            "implementation": {
                "source": _reference(root / FIGURE_SOURCE_RELATIVE, root).as_mapping(),
                "script": _reference(root / FIGURE_SCRIPT_RELATIVE, root).as_mapping(),
                "test": _reference(root / FIGURE_TEST_RELATIVE, root).as_mapping(),
                "base_scientific_renderer": _reference(
                    root / BASE_SOURCE_RELATIVE, root
                ).as_mapping(),
            },
            "lineage": {
                "base_preoutcome_figure_spec": _reference(
                    root / BASE_SPEC_RELATIVE, root
                ).as_mapping(),
                "superseded_visual_bundle_manifest": _reference(
                    root / SUPERSEDED_MANIFEST_RELATIVE, root
                ).as_mapping(),
                "superseded_bundle_citable": False,
                "v3_bundle_citable": True,
                "scientific_data_or_statistics_changed": False,
            },
            "visual_correction": VISUAL_CORRECTION,
            "inputs": {
                "gate1_hash_manifest": data.gate1_hash_manifest.as_mapping(),
                "gate1_artifacts": [reference.as_mapping() for reference in data.gate1_artifacts],
                "outer_ledger_index": data.outer_index.as_mapping(),
                "outer_call_ledgers": [reference.as_mapping() for reference in data.outer_call_ledgers],
            },
            "data_access_declaration": {
                "gate1_outputs_used": True,
                "outer_complete_call_ledgers_used": True,
                "outer_segment_ledgers_used": False,
                "checkpoints_used": False,
                "phase_a_validation_ledgers_used": False,
                "raw_audio_used": False,
                "transcripts_used": False,
                "generated_only_after_gate1_seal": True,
            },
            "claim_boundary": {
                "within_corpus_only": True,
                "independent_unit": "call",
                "practical_utility_claim_permitted": False,
                "seed_fold_curves_descriptive_only": True,
                "only_inferential_figure": "gate1_primary_effects",
            },
            "environment": base._environment(),
            "figure_count": len(figure_rows),
            "file_count": sum(len(row["files"]) for row in figure_rows),
            "figures": figure_rows,
        }
        (staging / "figure_manifest.json").write_bytes(_json_bytes(manifest))
        staging.rename(output)
        for path in sorted(output.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        output.chmod(0o555)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = output / "figure_manifest.json"
    verify_figure_manifest(root, manifest_path)
    return manifest_path


def _verify_project_reference(
    raw: Any, root: Path, context: str, expected_path: Path | None = None
) -> ArtifactReference:
    reference = base._artifact_reference(raw, root, context)
    if expected_path is not None and reference.serialized_path != expected_path.as_posix():
        raise Gate1FigureError(f"{context} path mismatch")
    return reference


def verify_figure_manifest(
    project_root: str | Path, manifest_path: str | Path
) -> Mapping[str, Any]:
    """Strictly verify v3 identity, lineage, inputs, files, and immutability."""

    root = Path(project_root).resolve()
    spec = load_figure_spec(root)
    path = Path(manifest_path).resolve()
    manifest = _read_json(path, "Gate-1 v3 figure manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "sealed_complete"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("generation_id") != GENERATION_ID
    ):
        raise Gate1FigureError("Gate-1 v3 figure manifest identity/status mismatch")
    if manifest.get("visual_correction") != VISUAL_CORRECTION:
        raise Gate1FigureError("Gate-1 v3 visual-correction declaration changed")
    if manifest.get("figure_count") != len(expected_figure_ids()) or manifest.get(
        "file_count"
    ) != len(expected_figure_ids()) * len(FORMATS):
        raise Gate1FigureError("Gate-1 v3 figure/file count mismatch")
    rows = manifest.get("figures")
    if not isinstance(rows, list) or [row.get("figure_id") for row in rows] != list(
        expected_figure_ids()
    ):
        raise Gate1FigureError("Gate-1 v3 figure family/order changed")

    _verify_project_reference(
        manifest.get("figure_spec"), root, "v3 manifest figure spec", FIGURE_SPEC_RELATIVE
    )
    implementation = manifest.get("implementation")
    expected_implementation = {
        "source": FIGURE_SOURCE_RELATIVE,
        "script": FIGURE_SCRIPT_RELATIVE,
        "test": FIGURE_TEST_RELATIVE,
        "base_scientific_renderer": BASE_SOURCE_RELATIVE,
    }
    if not isinstance(implementation, Mapping) or set(implementation) != set(
        expected_implementation
    ):
        raise Gate1FigureError("Gate-1 v3 implementation bindings are malformed")
    for name, expected_path in expected_implementation.items():
        _verify_project_reference(
            implementation[name], root, f"v3 implementation {name}", expected_path
        )

    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping) or (
        set(lineage)
        != {
            "base_preoutcome_figure_spec",
            "superseded_visual_bundle_manifest",
            "superseded_bundle_citable",
            "v3_bundle_citable",
            "scientific_data_or_statistics_changed",
        }
        or lineage.get("superseded_bundle_citable") is not False
        or lineage.get("v3_bundle_citable") is not True
        or lineage.get("scientific_data_or_statistics_changed") is not False
    ):
        raise Gate1FigureError("Gate-1 v3 lineage declaration changed")
    _verify_project_reference(
        lineage["base_preoutcome_figure_spec"],
        root,
        "v3 lineage base spec",
        BASE_SPEC_RELATIVE,
    )
    _verify_project_reference(
        lineage["superseded_visual_bundle_manifest"],
        root,
        "v3 lineage superseded manifest",
        SUPERSEDED_MANIFEST_RELATIVE,
    )

    required_access = {
        "gate1_outputs_used": True,
        "outer_complete_call_ledgers_used": True,
        "outer_segment_ledgers_used": False,
        "checkpoints_used": False,
        "phase_a_validation_ledgers_used": False,
        "raw_audio_used": False,
        "transcripts_used": False,
        "generated_only_after_gate1_seal": True,
    }
    if manifest.get("data_access_declaration") != required_access:
        raise Gate1FigureError("Gate-1 v3 access declaration changed")
    if manifest.get("claim_boundary") != {
        "within_corpus_only": True,
        "independent_unit": "call",
        "practical_utility_claim_permitted": False,
        "seed_fold_curves_descriptive_only": True,
        "only_inferential_figure": "gate1_primary_effects",
    }:
        raise Gate1FigureError("Gate-1 v3 claim boundary changed")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "gate1_hash_manifest",
        "gate1_artifacts",
        "outer_ledger_index",
        "outer_call_ledgers",
    }:
        raise Gate1FigureError("Gate-1 v3 input bindings are malformed")
    _verify_project_reference(inputs["gate1_hash_manifest"], root, "v3 Gate-1 hashes")
    _verify_project_reference(inputs["outer_ledger_index"], root, "v3 outer index")
    gate1_artifacts = inputs["gate1_artifacts"]
    call_ledgers = inputs["outer_call_ledgers"]
    if not isinstance(gate1_artifacts, list) or len(gate1_artifacts) != 8:
        raise Gate1FigureError("v3 manifest must bind eight canonical Gate-1 artifacts")
    if not isinstance(call_ledgers, list) or len(call_ledgers) != 200:
        raise Gate1FigureError("v3 manifest must bind 200 outer call ledgers")
    for index, reference in enumerate(gate1_artifacts):
        _verify_project_reference(reference, root, f"v3 Gate-1 artifact {index}")
    for index, reference in enumerate(call_ledgers):
        _verify_project_reference(reference, root, f"v3 outer call ledger {index}")

    expected_bundle_files = {"figure_manifest.json"}
    for row in rows:
        spec_row = next(item for item in spec["figures"] if item["figure_id"] == row["figure_id"])
        if row.get("title") != spec_row["title"] or row.get("analysis_role") != spec_row[
            "analysis_role"
        ]:
            raise Gate1FigureError(f"v3 figure metadata changed for {row.get('figure_id')}")
        files = row.get("files")
        if not isinstance(files, list) or len(files) != len(FORMATS):
            raise Gate1FigureError(f"v3 figure file set changed for {row.get('figure_id')}")
        suffixes = set()
        for raw in files:
            if not isinstance(raw, Mapping) or set(raw) != {
                "path",
                "sha256",
                "bytes",
                "row_count",
            }:
                raise Gate1FigureError("v3 figure file reference is malformed")
            relative = Path(str(raw["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise Gate1FigureError("v3 figure file has an unsafe path")
            file_path = (path.parent / relative).resolve()
            try:
                file_path.relative_to(path.parent)
            except ValueError as error:
                raise Gate1FigureError("v3 figure file escapes bundle") from error
            if not file_path.is_file() or file_path.is_symlink():
                raise Gate1FigureError("v3 figure file is missing/symlinked")
            if (
                sha256_file(file_path) != raw.get("sha256")
                or file_path.stat().st_size != raw.get("bytes")
                or raw.get("row_count") is not None
            ):
                raise Gate1FigureError("v3 figure file hash/size/row-count mismatch")
            expected_bundle_files.add(relative.as_posix())
            suffixes.add(file_path.suffix.lstrip("."))
        if suffixes != set(FORMATS):
            raise Gate1FigureError(f"v3 figure formats changed for {row.get('figure_id')}")

    actual_bundle_files = {
        item.relative_to(path.parent).as_posix()
        for item in path.parent.rglob("*")
        if item.is_file()
    }
    if actual_bundle_files != expected_bundle_files:
        raise Gate1FigureError("v3 bundle contains missing or unregistered files")
    for item in (path.parent, *path.parent.rglob("*")):
        if item.is_symlink():
            raise Gate1FigureError("v3 bundle contains a symlink")
        mode = stat.S_IMODE(item.stat().st_mode)
        if mode & 0o222:
            raise Gate1FigureError("v3 bundle is not read-only")
    return manifest


__all__ = [
    "CLAIM_LABEL",
    "COMPARISONS",
    "CONDITIONS",
    "FIGURE_SPEC_RELATIVE",
    "SEEDS",
    "VISUAL_CORRECTION",
    "CallPrediction",
    "Gate1FigureData",
    "Gate1FigureError",
    "_metric_profile",
    "_primary_effects",
    "_probability_distributions",
    "compute_call_metrics",
    "expected_figure_ids",
    "generate_gate1_figures",
    "load_figure_spec",
    "load_gate1_figure_data",
    "render_figures",
    "sha256_file",
    "verify_figure_manifest",
]
