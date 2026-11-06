"""Reproducible, outcome-safe figures for Work 9 Phase A.

This module is intentionally isolated from the frozen Work 9 runner and Gate-1
analysis.  It reads only the frozen protocol metadata and, for Phase-A
diagnostics, the three immutable artifacts explicitly referenced by the frozen
200-coordinate selection index:

* epoch traces;
* threshold traces; and
* selection run records.

It never opens checkpoints, validation prediction ledgers, or outer-evaluation
artifacts.  Phase-A plots are descriptive model-selection diagnostics.  They do
not estimate outer performance and do not treat folds/seeds as independent
replicates for inference.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


FIGURE_MANIFEST_SCHEMA = "paper2-v2-work9-phase-a-figure-manifest-1"
SELECTION_INDEX_SCHEMA = "paper2-v2-work9-selection-index-1"
EPOCH_TRACE_SCHEMA = "paper2-v2-work9-epoch-trace-1"
THRESHOLD_TRACE_SCHEMA = "paper2-v2-work9-threshold-trace-1"
RUN_RECORD_SCHEMA = "paper2-v2-work9-run-record-1"
PROTOCOL_ID = "paper2-v2-mfcc-matched-within-corpus-v1"
RUN_FAMILY = "primary_matrix"
CONDITIONS = (
    "T",
    "A-mean",
    "A-query",
    "A-temp",
    "AT-concat",
    "AT-gate",
    "AT-cross",
    "AT-cross-randtrain",
)
SEEDS = (17, 29, 43, 71, 101)
FOLDS = (0, 1, 2, 3, 4)
FORMATS = ("svg", "pdf", "png")
SELECTION_INDEX_RELATIVE = Path(
    "revised_experiments/artifacts/work9_runs_v1/_matrix/selection_index.json"
)
CONFIG_RELATIVE = Path(
    "revised_experiments/configs/work8_mfcc_matched_within_corpus_v1.json"
)
FOLD_PROTOCOL_RELATIVE = Path("data/paper2_v2/folds/grouped_nested_v1/protocol.json")
FIGURE_SPEC_RELATIVE = Path(
    "revised_experiments/configs/work9_phase_a_figure_spec_v1.json"
)
FIGURE_SCRIPT_RELATIVE = Path("revised_experiments/scripts/generate_work9_figures.py")
SUPERSEDED_BUNDLE_RELATIVE = Path(
    "revised_experiments/artifacts/work9_phase_a_figures_v1"
)
SUPERSEDED_MANIFEST_SHA256 = (
    "9606519e508e9c20819f852428457dd078c9c24332f59f8387d895f94532cdda"
)
DIAGNOSTIC_LABEL = (
    "Phase A development/model-selection diagnostic — not outer-evaluation performance"
)
OUTCOME_FREE_LABEL = "Outcome-free protocol schematic — no model outcomes shown"

COLORS = {
    "T": "#7656A8",
    "A-mean": "#2E86AB",
    "A-query": "#1976A3",
    "A-temp": "#155B82",
    "AT-concat": "#E38B2C",
    "AT-gate": "#D26428",
    "AT-cross": "#B63A5B",
    "AT-cross-randtrain": "#777777",
    "train": "#3478A8",
    "validation": "#C23B5A",
}


class FigurePipelineError(RuntimeError):
    """Raised when an immutable input or figure-safety invariant fails."""


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    serialized_path: str
    sha256: str
    bytes: int
    row_count: int | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "path": self.serialized_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class PhaseARun:
    run_id: str
    condition: str
    outer_fold: int
    seed: int
    selected_epoch: int
    completed_epochs: int
    selected_threshold: float
    validation_call_nll: float
    epochs: tuple[Mapping[str, Any], ...]
    threshold_candidates: tuple[Mapping[str, Any], ...]
    wall_time_seconds: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    total_parameters: int
    trainable_parameters: int
    epoch_trace_ref: ArtifactReference
    threshold_trace_ref: ArtifactReference
    run_record_ref: ArtifactReference


@dataclass(frozen=True)
class PhaseABundle:
    selection_index_ref: ArtifactReference
    config_ref: ArtifactReference
    fold_protocol_ref: ArtifactReference
    config: Mapping[str, Any]
    fold_protocol: Mapping[str, Any]
    runs: tuple[PhaseARun, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise FigurePipelineError(f"artifact is outside project root: {path}") from error


def _artifact_for_file(
    path: Path,
    project_root: Path,
    *,
    serialized_path: str | None = None,
    row_count: int | None = None,
) -> ArtifactReference:
    if not path.is_file() or path.is_symlink():
        raise FigurePipelineError(f"required regular file is missing or symlinked: {path}")
    return ArtifactReference(
        path=path.resolve(),
        serialized_path=serialized_path or _project_relative(path, project_root),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        row_count=row_count,
    )


def _contains_outer_artifact_name(path: Path) -> bool:
    return any("outer_eval" in part.lower() or "outer-ledger" in part.lower() for part in path.parts)


def _resolve_artifact_reference(
    raw: Mapping[str, Any],
    project_root: Path,
    *,
    context: str,
    allow_outer_named_path: bool = False,
) -> ArtifactReference:
    expected = {"path", "sha256", "bytes", "row_count"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise FigurePipelineError(f"{context} has an invalid artifact reference")
    serialized = raw["path"]
    if not isinstance(serialized, str):
        raise FigurePipelineError(f"{context} path is not a string")
    relative = Path(serialized)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise FigurePipelineError(f"{context} path is not safe and project-relative")
    if not allow_outer_named_path and _contains_outer_artifact_name(relative):
        raise FigurePipelineError(f"{context} attempts to access an outer artifact")
    root = project_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or path.is_symlink() or not path.is_file():
        raise FigurePipelineError(f"{context} file is missing, symlinked, or escaped root")
    expected_sha = raw["sha256"]
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise FigurePipelineError(f"{context} SHA-256 is invalid")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise FigurePipelineError(f"{context} SHA-256 mismatch")
    expected_bytes = raw["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise FigurePipelineError(f"{context} byte count is invalid")
    if path.stat().st_size != expected_bytes:
        raise FigurePipelineError(f"{context} byte count mismatch")
    row_count = raw["row_count"]
    if row_count is not None and (isinstance(row_count, bool) or not isinstance(row_count, int)):
        raise FigurePipelineError(f"{context} row count is invalid")
    return ArtifactReference(path, serialized, expected_sha, expected_bytes, row_count)


def _read_json(path: Path, *, context: str, permit_protocol_metadata: bool = False) -> Any:
    if _contains_outer_artifact_name(path) and not permit_protocol_metadata:
        raise FigurePipelineError(f"refusing outer-artifact access for {context}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FigurePipelineError(f"cannot read {context}: {path}") from error


def _finite_number(value: Any, *, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigurePipelineError(f"{context} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise FigurePipelineError(f"{context} is not finite or is below its minimum")
    return result


def _validate_coordinate_grid(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (condition, fold, seed)
        for condition in CONDITIONS
        for fold in FOLDS
        for seed in SEEDS
    }
    observed: set[tuple[str, int, int]] = set()
    for row in rows:
        coordinate = (row.get("condition"), row.get("outer_fold"), row.get("seed"))
        if coordinate in observed:
            raise FigurePipelineError(f"duplicate Phase-A coordinate: {coordinate}")
        observed.add(coordinate)  # type: ignore[arg-type]
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise FigurePipelineError(
            f"Phase-A grid is not the exact frozen 200 coordinates; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )


def _validate_epoch_trace(raw: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if raw.get("schema_version") != EPOCH_TRACE_SCHEMA:
        raise FigurePipelineError("unsupported epoch-trace schema")
    for key in ("run_id", "condition", "outer_fold", "seed"):
        if raw.get(key) != row.get(key):
            raise FigurePipelineError(f"epoch trace differs at {key}")
    if raw.get("run_family") != RUN_FAMILY or raw.get("partition_repeat") != 0:
        raise FigurePipelineError("epoch trace is outside the primary selection matrix")
    epochs = raw.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise FigurePipelineError("epoch trace is empty")
    epoch_numbers = [epoch.get("epoch") for epoch in epochs]
    if epoch_numbers != list(range(1, len(epochs) + 1)):
        raise FigurePipelineError("epoch trace is not contiguous from epoch 1")
    nlls: list[float] = []
    for index, epoch in enumerate(epochs, start=1):
        _finite_number(epoch.get("train_reported_loss"), context=f"epoch {index} train loss", minimum=0)
        nlls.append(
            _finite_number(
                epoch.get("validation_call_nll"),
                context=f"epoch {index} validation NLL",
                minimum=0,
            )
        )
    selected_epoch = raw.get("selected_epoch")
    expected_selected = min(range(1, len(nlls) + 1), key=lambda epoch: (nlls[epoch - 1], epoch))
    if selected_epoch != expected_selected:
        raise FigurePipelineError("selected epoch is not earliest minimum validation-call NLL")
    return tuple(epochs)


def _validate_threshold_trace(
    raw: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    if raw.get("schema_version") != THRESHOLD_TRACE_SCHEMA:
        raise FigurePipelineError("unsupported threshold-trace schema")
    for key in ("run_id", "condition", "outer_fold", "seed"):
        if raw.get(key) != row.get(key):
            raise FigurePipelineError(f"threshold trace differs at {key}")
    if raw.get("partition") != "validation" or raw.get("run_family") != RUN_FAMILY:
        raise FigurePipelineError("threshold trace is not a validation-selection artifact")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 99:
        raise FigurePipelineError("threshold trace must contain exactly 99 candidates")
    expected_grid = [round(index / 100.0, 2) for index in range(1, 100)]
    observed_grid: list[float] = []
    for index, candidate in enumerate(candidates):
        threshold = _finite_number(candidate.get("threshold"), context="threshold candidate")
        observed_grid.append(round(threshold, 2))
        metrics = candidate.get("metrics")
        if not isinstance(metrics, Mapping):
            raise FigurePipelineError("threshold candidate metrics are missing")
        _finite_number(metrics.get("macro_f1"), context="threshold macro-F1", minimum=0)
    if observed_grid != expected_grid:
        raise FigurePipelineError("threshold candidates differ from 0.01..0.99")
    selected = _finite_number(raw.get("selected_threshold"), context="selected threshold")
    if round(selected, 2) not in expected_grid:
        raise FigurePipelineError("selected threshold is outside the frozen grid")
    return tuple(candidates)


def _validate_run_record(
    raw: Mapping[str, Any], row: Mapping[str, Any], epoch_count: int
) -> Mapping[str, Any]:
    if raw.get("schema_version") != RUN_RECORD_SCHEMA:
        raise FigurePipelineError("unsupported selection run-record schema")
    for key in ("run_id", "condition", "outer_fold", "seed"):
        if raw.get(key) != row.get(key):
            raise FigurePipelineError(f"selection run record differs at {key}")
    if raw.get("phase") != "selection" or raw.get("status") != "selection_complete":
        raise FigurePipelineError("run record is not a completed Phase-A selection")
    data = raw.get("data")
    if not isinstance(data, Mapping) or data.get("outer_eval_manifest_sha256") is not None:
        raise FigurePipelineError("Phase-A run record indicates outer-evaluation access")
    training = raw.get("training")
    if not isinstance(training, Mapping) or training.get("epochs_completed") != epoch_count:
        raise FigurePipelineError("run-record epoch count differs from epoch trace")
    if raw.get("failure") is not None:
        raise FigurePipelineError("completed Phase-A run record contains a failure")
    return raw


def load_phase_a_bundle(project_root: str | Path) -> PhaseABundle:
    """Load and verify only index-authorized Phase-A diagnostic inputs."""

    root = Path(project_root).resolve()
    index_path = root / SELECTION_INDEX_RELATIVE
    if not index_path.is_file():
        raise FigurePipelineError(
            "Phase-A outcome figures require the frozen 200-coordinate selection index"
        )
    index_ref = _artifact_for_file(index_path, root)
    index = _read_json(index_path, context="frozen selection index")
    if not isinstance(index, Mapping):
        raise FigurePipelineError("selection index is not an object")
    if (
        index.get("schema_version") != SELECTION_INDEX_SCHEMA
        or index.get("protocol_id") != PROTOCOL_ID
        or index.get("run_family") != RUN_FAMILY
        or index.get("status") != "frozen_complete"
        or index.get("expected_coordinate_count") != 200
    ):
        raise FigurePipelineError("selection index is not the frozen Work 9 Phase-A release")
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != 200:
        raise FigurePipelineError("selection index does not contain exactly 200 entries")
    _validate_coordinate_grid(entries)

    config_ref = _resolve_artifact_reference(
        index.get("config", {}), root, context="selection-index config"
    )
    if config_ref.serialized_path != CONFIG_RELATIVE.as_posix():
        raise FigurePipelineError("selection index references an unexpected config")
    config = _read_json(config_ref.path, context="frozen Work 8 config")
    if (
        not isinstance(config, Mapping)
        or config.get("status") != "locked"
        or config.get("protocol_id") != PROTOCOL_ID
        or tuple(config.get("conditions", ())) != CONDITIONS
        or tuple(config.get("seeds", ())) != SEEDS
    ):
        raise FigurePipelineError("frozen config differs from the Work 9 figure contract")

    protocol_declared = config.get("fold_protocol", {}).get("protocol_artifact", {})
    protocol_path_text = protocol_declared.get("path")
    protocol_sha = protocol_declared.get("sha256")
    if protocol_path_text != FOLD_PROTOCOL_RELATIVE.as_posix() or not isinstance(protocol_sha, str):
        raise FigurePipelineError("frozen config has an unexpected fold protocol")
    protocol_path = (root / FOLD_PROTOCOL_RELATIVE).resolve()
    protocol_ref = _artifact_for_file(
        protocol_path, root, serialized_path=FOLD_PROTOCOL_RELATIVE.as_posix()
    )
    if protocol_ref.sha256 != protocol_sha:
        raise FigurePipelineError("fold-protocol hash differs from frozen config")
    protocol = _read_json(
        protocol_path,
        context="outcome-free fold-protocol metadata",
        permit_protocol_metadata=True,
    )

    runs: list[PhaseARun] = []
    for row in entries:
        run_id = row.get("run_id")
        if not isinstance(run_id, str):
            raise FigurePipelineError("selection-index run_id is invalid")
        expected_prefix = f"revised_experiments/artifacts/work9_runs_v1/{run_id}/"
        refs: dict[str, ArtifactReference] = {}
        expected_suffix = {
            "epoch_trace": "selection/epoch_trace.json",
            "threshold_trace": "selection/threshold_trace.json",
            "selection_run_record": "selection_run_record.json",
        }
        for name, suffix in expected_suffix.items():
            ref = _resolve_artifact_reference(
                row.get(name, {}), root, context=f"{run_id} {name}"
            )
            if ref.serialized_path != expected_prefix + suffix:
                raise FigurePipelineError(f"{run_id} {name} path is not its canonical artifact")
            refs[name] = ref

        epoch_raw = _read_json(refs["epoch_trace"].path, context=f"{run_id} epoch trace")
        threshold_raw = _read_json(
            refs["threshold_trace"].path, context=f"{run_id} threshold trace"
        )
        record_raw = _read_json(
            refs["selection_run_record"].path, context=f"{run_id} selection run record"
        )
        if not all(isinstance(item, Mapping) for item in (epoch_raw, threshold_raw, record_raw)):
            raise FigurePipelineError(f"{run_id} diagnostic input is not a JSON object")
        epochs = _validate_epoch_trace(epoch_raw, row)
        candidates = _validate_threshold_trace(threshold_raw, row)
        record = _validate_run_record(record_raw, row, len(epochs))

        selection = record.get("selection", {})
        compute = record.get("compute", {})
        model = record.get("model", {})
        selected_epoch = int(selection.get("selected_epoch"))
        selected_threshold = _finite_number(
            selection.get("threshold"), context=f"{run_id} selected threshold"
        )
        selected_nll = _finite_number(
            selection.get("validation_call_nll"),
            context=f"{run_id} selected validation NLL",
            minimum=0,
        )
        if (
            selected_epoch != epoch_raw.get("selected_epoch")
            or not math.isclose(selected_threshold, float(threshold_raw.get("selected_threshold")), abs_tol=1e-12)
            or not math.isclose(
                selected_nll,
                float(epochs[selected_epoch - 1]["validation_call_nll"]),
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise FigurePipelineError(f"{run_id} selected values disagree across sealed inputs")
        total_parameters = int(model.get("total_parameters"))
        trainable_parameters = int(model.get("trainable_parameters"))
        if total_parameters <= 0 or not 0 < trainable_parameters <= total_parameters:
            raise FigurePipelineError(f"{run_id} parameter counts are invalid")
        runs.append(
            PhaseARun(
                run_id=run_id,
                condition=str(row["condition"]),
                outer_fold=int(row["outer_fold"]),
                seed=int(row["seed"]),
                selected_epoch=selected_epoch,
                completed_epochs=len(epochs),
                selected_threshold=selected_threshold,
                validation_call_nll=selected_nll,
                epochs=epochs,
                threshold_candidates=candidates,
                wall_time_seconds=_finite_number(
                    compute.get("wall_time_seconds"), context=f"{run_id} wall time", minimum=0
                ),
                peak_cuda_allocated_bytes=int(
                    _finite_number(
                        compute.get("peak_cuda_allocated_bytes"),
                        context=f"{run_id} peak CUDA allocated bytes",
                        minimum=0,
                    )
                ),
                peak_cuda_reserved_bytes=int(
                    _finite_number(
                        compute.get("peak_cuda_reserved_bytes"),
                        context=f"{run_id} peak CUDA reserved bytes",
                        minimum=0,
                    )
                ),
                total_parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                epoch_trace_ref=refs["epoch_trace"],
                threshold_trace_ref=refs["threshold_trace"],
                run_record_ref=refs["selection_run_record"],
            )
        )
    runs.sort(key=lambda run: (CONDITIONS.index(run.condition), run.outer_fold, SEEDS.index(run.seed)))
    return PhaseABundle(index_ref, config_ref, protocol_ref, config, protocol, tuple(runs))


def pointwise_observed_summary(
    curves: Sequence[Sequence[tuple[int, float]]],
) -> dict[str, np.ndarray]:
    """Summarize only actually observed epochs; never carry values forward."""

    values_by_epoch: dict[int, list[float]] = {}
    for curve in curves:
        previous = 0
        for epoch, value in curve:
            if epoch <= previous:
                raise FigurePipelineError("curve epochs must be strictly increasing")
            previous = epoch
            numeric = _finite_number(value, context="learning-curve value", minimum=0)
            values_by_epoch.setdefault(epoch, []).append(numeric)
    if not values_by_epoch:
        raise FigurePipelineError("cannot summarize empty learning curves")
    epochs = np.asarray(sorted(values_by_epoch), dtype=int)
    arrays = [np.asarray(values_by_epoch[int(epoch)], dtype=float) for epoch in epochs]
    return {
        "epoch": epochs,
        "median": np.asarray([np.median(values) for values in arrays]),
        "q25": np.asarray([np.quantile(values, 0.25, method="linear") for values in arrays]),
        "q75": np.asarray([np.quantile(values, 0.75, method="linear") for values in arrays]),
        "n": np.asarray([len(values) for values in arrays], dtype=int),
    }


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.18,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "paper2-work9-phase-a-v1",
        }
    )


def _stamp(fig: plt.Figure, label: str) -> None:
    fig.text(0.5, 0.006, label, ha="center", va="bottom", fontsize=7.3, color="#555555")


def _save_figure(fig: plt.Figure, base_path: Path, formats: Sequence[str]) -> list[Path]:
    outputs: list[Path] = []
    for extension in formats:
        if extension not in FORMATS:
            raise FigurePipelineError(f"unsupported figure format: {extension}")
        target = base_path.with_suffix(f".{extension}")
        metadata: dict[str, Any]
        if extension == "pdf":
            metadata = {
                "Title": base_path.name,
                "Author": "Paper 2 Work 9 figure pipeline",
                "Creator": "paper2_v2.work9_figures",
                "CreationDate": None,
                "ModDate": None,
            }
        elif extension == "svg":
            metadata = {
                "Title": base_path.name,
                "Creator": "paper2_v2.work9_figures",
                "Date": "2026-08-10",
            }
        else:
            metadata = {"Software": "paper2_v2.work9_figures"}
        fig.savefig(target, bbox_inches="tight", pad_inches=0.08, metadata=metadata)
        outputs.append(target)
    plt.close(fig)
    return outputs


def _condition_runs(bundle: PhaseABundle, condition: str) -> tuple[PhaseARun, ...]:
    runs = tuple(run for run in bundle.runs if run.condition == condition)
    if len(runs) != 25:
        raise FigurePipelineError(f"{condition} does not have exactly 25 Phase-A runs")
    return runs


def load_outcome_free_protocol(
    project_root: str | Path,
) -> tuple[ArtifactReference, ArtifactReference, Mapping[str, Any], Mapping[str, Any]]:
    """Load frozen protocol metadata without requiring or reading run outcomes."""

    root = Path(project_root).resolve()
    config_path = root / CONFIG_RELATIVE
    config_ref = _artifact_for_file(config_path, root)
    config = _read_json(config_path, context="frozen Work 8 config")
    if (
        not isinstance(config, Mapping)
        or config.get("status") != "locked"
        or config.get("protocol_id") != PROTOCOL_ID
        or tuple(config.get("conditions", ())) != CONDITIONS
        or tuple(config.get("seeds", ())) != SEEDS
    ):
        raise FigurePipelineError("config differs from the outcome-free figure contract")
    declared = config.get("fold_protocol", {}).get("protocol_artifact", {})
    if declared.get("path") != FOLD_PROTOCOL_RELATIVE.as_posix():
        raise FigurePipelineError("config references an unexpected fold protocol")
    protocol_path = root / FOLD_PROTOCOL_RELATIVE
    protocol_ref = _artifact_for_file(protocol_path, root)
    if protocol_ref.sha256 != declared.get("sha256"):
        raise FigurePipelineError("fold-protocol hash differs from frozen config")
    protocol = _read_json(
        protocol_path,
        context="outcome-free fold-protocol metadata",
        permit_protocol_metadata=True,
    )
    if not isinstance(protocol, Mapping) or protocol.get("artifact_status") != "frozen":
        raise FigurePipelineError("fold protocol is not frozen")
    return config_ref, protocol_ref, config, protocol


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text_value: str,
    *,
    facecolor: str,
    edgecolor: str = "#333333",
    fontsize: float = 8.4,
    weight: str = "normal",
) -> None:
    patch = matplotlib.patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text_value,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
    )


def render_protocol_pipeline(
    config: Mapping[str, Any], protocol: Mapping[str, Any]
) -> plt.Figure:
    """Draw the outcome-free data/fold/experiment evidence flow."""

    summary = protocol.get("summary", {})
    folds = protocol.get("folds", [])
    if not isinstance(summary, Mapping) or not isinstance(folds, list) or len(folds) != 5:
        raise FigurePipelineError("fold protocol lacks its frozen summary")
    all_calls = int(summary["all_provenance_calls"])
    retained_calls = int(summary["eligible_retained_calls"])
    groups = int(summary["eligible_groups"])
    quarantined = int(summary["quarantined_calls"])
    duplicate_exclusions = int(summary["exact_audio_duplicate_exclusions"])
    # This is protocol metadata, not a read of an outer manifest or model outcome.
    retained_segments = sum(int(fold["summary"]["outer_eval"]["segment_count"]) for fold in folds)
    train_range = (
        min(int(fold["summary"]["train"]["call_count"]) for fold in folds),
        max(int(fold["summary"]["train"]["call_count"]) for fold in folds),
    )
    val_range = (
        min(int(fold["summary"]["val"]["call_count"]) for fold in folds),
        max(int(fold["summary"]["val"]["call_count"]) for fold in folds),
    )
    outer_range = (
        min(int(fold["summary"]["outer_eval"]["call_count"]) for fold in folds),
        max(int(fold["summary"]["outer_eval"]["call_count"]) for fold in folds),
    )

    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "Paper 2 revised experiment: relation-aware data and evidence flow",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    ax.text(
        0.5,
        0.935,
        "Within-corpus methodological study • five grouped folds • five neural seeds",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    boxes = [
        (0.02, 0.59, 0.145, 0.22, f"Call catalogue\n{all_calls:,} calls\nmaster provenance", "#E9F1F7"),
        (
            0.195,
            0.59,
            0.165,
            0.22,
            f"Relation + leakage audit\n{quarantined:,} calls quarantined\n{duplicate_exclusions} exact duplicate excluded",
            "#F2EEE7",
        ),
        (
            0.39,
            0.59,
            0.165,
            0.22,
            f"Eligible grouped corpus\n{retained_calls:,} calls • {groups:,} groups\n{retained_segments:,} segments",
            "#E7F2EA",
        ),
        (
            0.585,
            0.59,
            0.175,
            0.22,
            "Five relation-disjoint folds\n"
            f"train {train_range[0]}–{train_range[1]} calls\n"
            f"validation {val_range[0]}–{val_range[1]} calls\n"
            f"locked outer {outer_range[0]}–{outer_range[1]} calls",
            "#FFF0D9",
        ),
        (
            0.79,
            0.59,
            0.19,
            0.22,
            "Phase A: model selection\n8 conditions × 5 folds × 5 seeds\n200 sealed selections",
            "#F5E8ED",
        ),
    ]
    for box in boxes:
        _rounded_box(ax, *box[:5], facecolor=box[5])
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.annotate(
            "",
            xy=(right[0] - 0.006, right[1] + right[3] / 2),
            xytext=(left[0] + left[2] + 0.006, left[1] + left[3] / 2),
            arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#555555"},
        )

    _rounded_box(
        ax,
        0.72,
        0.20,
        0.22,
        0.17,
        "Frozen selection index\nall 200 selections verified\nbefore any outer release",
        facecolor="#E8EDF8",
        weight="bold",
    )
    _rounded_box(
        ax,
        0.39,
        0.20,
        0.22,
        0.17,
        "Phase B: one-shot outer scoring\nwithin-corpus development evidence\n(no external confirmation)",
        facecolor="#F1EBF5",
    )
    _rounded_box(
        ax,
        0.06,
        0.20,
        0.22,
        0.17,
        "Gate 1 paired analysis\ncall/relation-group aware\npre-registered decision policy",
        facecolor="#E7F3EF",
    )
    for start, end in ((0.72, 0.61), (0.39, 0.28)):
        ax.annotate(
            "",
            xy=(end + 0.006, 0.285),
            xytext=(start - 0.006, 0.285),
            arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#555555"},
        )
    ax.annotate(
        "",
        xy=(0.83, 0.375),
        xytext=(0.885, 0.585),
        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#555555"},
    )
    ax.text(0.89, 0.48, "freeze + verify", ha="left", va="center", fontsize=8, color="#555555")
    ax.text(
        0.5,
        0.105,
        "Known scope limitation: retained benign calls are AIHub-derived and retained vishing calls are FSS; "
        "source-independent fraud detection is not claimed.",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#7B263A",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFF7F8", "ec": "#D9A5B1"},
    )
    _stamp(fig, OUTCOME_FREE_LABEL)
    return fig


def render_architecture_comparison(config: Mapping[str, Any]) -> plt.Figure:
    """Draw a compact comparison of the eight frozen conditions."""

    mechanisms = {
        "T": ("Text", "KR-SBERT sentence vector → MLP", "Transcript-only baseline"),
        "A-mean": ("MFCC", "masked arithmetic mean → classifier", "Simple audio baseline"),
        "A-query": ("MFCC", "learned-query set pooling → classifier", "Matched acoustic reference"),
        "A-temp": ("MFCC + position", "absolute-position query pooling", "Tests temporal order"),
        "AT-concat": ("MFCC + text", "audio set vector ⊕ text → classifier", "Standard early/feature fusion"),
        "AT-gate": ("MFCC + text", "text-conditioned FiLM/gating", "Adaptive gated fusion"),
        "AT-cross": ("MFCC + text", "text-conditioned acoustic set pooling", "Proposed cross-attention"),
        "AT-cross-randtrain": (
            "MFCC + wrong train text",
            "same AT-cross architecture; random train pairs",
            "Pairing-specificity control",
        ),
    }
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Eight-condition primary multimodal experiment matrix", fontsize=14, fontweight="bold", y=0.985)
    headers = ((0.025, "Condition"), (0.19, "Input"), (0.40, "Mechanism"), (0.76, "Scientific role"))
    for x, label in headers:
        ax.text(x, 0.925, label, fontsize=9.5, fontweight="bold", ha="left", va="center")
    ax.plot([0.02, 0.98], [0.9, 0.9], color="#555555", lw=1.1)
    row_height = 0.098
    for index, condition in enumerate(CONDITIONS):
        y = 0.845 - index * row_height
        input_name, mechanism, role = mechanisms[condition]
        ax.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (0.02, y - 0.035),
                0.135,
                0.071,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                facecolor=COLORS[condition],
                edgecolor="none",
                alpha=0.94,
            )
        )
        ax.text(0.0875, y, condition, color="white", fontweight="bold", ha="center", va="center", fontsize=8.5)
        ax.text(0.19, y, input_name, ha="left", va="center", fontsize=8.5)
        ax.text(0.40, y, mechanism, ha="left", va="center", fontsize=8.4)
        ax.text(0.76, y, role, ha="left", va="center", fontsize=8.4)
        ax.plot([0.02, 0.98], [y - 0.049, y - 0.049], color="#DDDDDD", lw=0.65)
    ax.text(
        0.5,
        0.045,
        "All conditions share the same grouped folds, five seeds, training budget, call-equal loss weighting, "
        "validation checkpoint rule, and threshold grid. AT-cross-randtrain differs only in its training pairing view.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F7F7F7", "ec": "#CCCCCC"},
    )
    _stamp(fig, OUTCOME_FREE_LABEL)
    return fig


def render_aggregate_learning_curves(bundle: PhaseABundle) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(12.6, 13.2), sharex=True)
    axes_flat = axes.ravel()
    for ax, condition in zip(axes_flat, CONDITIONS):
        runs = _condition_runs(bundle, condition)
        train = pointwise_observed_summary(
            [[(int(row["epoch"]), float(row["train_reported_loss"])) for row in run.epochs] for run in runs]
        )
        validation = pointwise_observed_summary(
            [[(int(row["epoch"]), float(row["validation_call_nll"])) for row in run.epochs] for run in runs]
        )
        for summary, color, label, linestyle in (
            (train, COLORS["train"], "reported training loss", "--"),
            (validation, COLORS["validation"], "validation call NLL", "-"),
        ):
            ax.fill_between(summary["epoch"], summary["q25"], summary["q75"], color=color, alpha=0.15, linewidth=0)
            ax.plot(summary["epoch"], summary["median"], color=color, lw=1.6, linestyle=linestyle, label=label)
        ax.set_yscale("log")
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.set_ylabel("Loss / NLL (log scale)")
        ax.set_xlim(1, 50)
        checkpoints = [1, 10, 20, 30, 40, 50]
        counts = []
        by_epoch = dict(zip(validation["epoch"].tolist(), validation["n"].tolist()))
        for epoch in checkpoints:
            if epoch in by_epoch:
                counts.append(f"{epoch}:{by_epoch[epoch]}")
        ax.text(0.99, 0.97, "observed n  " + "  ".join(counts), transform=ax.transAxes, ha="right", va="top", fontsize=6.6, color="#666666")
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.967), ncol=2, frameon=False)
    fig.suptitle(
        "Phase A learning trajectories across 25 fold–seed coordinates per condition",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.026,
        "Lines are pointwise medians; bands are pointwise interquartile ranges among runs still observed at that epoch. "
        "No last-observation carry-forward; observed n therefore decreases after early stopping.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.052, 0.98, 0.947))
    return fig


def render_condition_small_multiples(bundle: PhaseABundle, condition: str) -> plt.Figure:
    runs = _condition_runs(bundle, condition)
    lookup = {(run.outer_fold, run.seed): run for run in runs}
    all_values = [
        float(row[key])
        for run in runs
        for row in run.epochs
        for key in ("train_reported_loss", "validation_call_nll")
        if float(row[key]) > 0
    ]
    ymin = max(min(all_values) * 0.72, 1e-8)
    ymax = max(all_values) * 1.35
    fig, axes = plt.subplots(5, 5, figsize=(14.5, 12.2), sharex=True, sharey=True)
    for row_index, fold in enumerate(FOLDS):
        for column_index, seed in enumerate(SEEDS):
            ax = axes[row_index, column_index]
            run = lookup[(fold, seed)]
            epoch = [int(value["epoch"]) for value in run.epochs]
            train = [float(value["train_reported_loss"]) for value in run.epochs]
            validation = [float(value["validation_call_nll"]) for value in run.epochs]
            ax.plot(epoch, train, color=COLORS["train"], lw=1.0, linestyle="--")
            ax.plot(epoch, validation, color=COLORS["validation"], lw=1.2)
            ax.axvline(run.selected_epoch, color="#222222", lw=0.8, alpha=0.65)
            ax.scatter(
                [run.selected_epoch],
                [run.validation_call_nll],
                s=14,
                color="#222222",
                zorder=3,
            )
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(1, 50)
            ax.set_title(f"fold {fold} • seed {seed}", fontsize=7.5)
            if column_index == 0:
                ax.set_ylabel("Loss / NLL")
            if row_index == 4:
                ax.set_xlabel("Epoch")
    train_line = matplotlib.lines.Line2D([], [], color=COLORS["train"], linestyle="--", label="training loss")
    val_line = matplotlib.lines.Line2D([], [], color=COLORS["validation"], label="validation call NLL")
    selected_line = matplotlib.lines.Line2D([], [], color="#222222", marker="o", linestyle="-", label="selected epoch")
    fig.legend(handles=[train_line, val_line, selected_line], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.967))
    fig.suptitle(
        f"{condition}: all 25 Phase A learning trajectories",
        fontsize=14,
        fontweight="bold",
        color=COLORS[condition],
        y=0.992,
    )
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.025, 0.98, 0.945))
    return fig


def _coordinate_offsets(runs: Sequence[PhaseARun], scale: float = 0.26) -> np.ndarray:
    """Deterministic display jitter keyed by the frozen fold/seed grid."""

    return np.asarray(
        [
            ((SEEDS.index(run.seed) - 2) / 5.0 + (run.outer_fold - 2) / 35.0) * scale
            for run in runs
        ],
        dtype=float,
    )


def render_epoch_distributions(bundle: PhaseABundle) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    for condition_index, condition in enumerate(CONDITIONS):
        runs = _condition_runs(bundle, condition)
        offsets = _coordinate_offsets(runs)
        selected = np.asarray([run.selected_epoch for run in runs], dtype=float)
        completed = np.asarray([run.completed_epochs for run in runs], dtype=float)
        for x_offset, start, end in zip(offsets, selected, completed):
            ax.plot(
                [condition_index + x_offset - 0.04, condition_index + x_offset + 0.04],
                [start, end],
                color="#BBBBBB",
                lw=0.65,
                zorder=1,
            )
        ax.scatter(
            condition_index + offsets - 0.04,
            selected,
            s=22,
            color=COLORS[condition],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
            label="selected epoch" if condition_index == 0 else None,
        )
        ax.scatter(
            condition_index + offsets + 0.04,
            completed,
            s=25,
            facecolor="white",
            edgecolor=COLORS[condition],
            linewidth=0.9,
            zorder=2,
            label="epochs completed" if condition_index == 0 else None,
        )
        ax.hlines(
            [np.median(selected), np.median(completed)],
            [condition_index - 0.30, condition_index - 0.30],
            [condition_index + 0.30, condition_index + 0.30],
            colors=[COLORS[condition], "#333333"],
            linestyles=["-", "--"],
            linewidths=[1.7, 1.2],
        )
    ax.axhline(50, color="#777777", linestyle=":", lw=1.0, label="50-epoch budget")
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=20, ha="right")
    ax.set_xlim(-0.55, len(CONDITIONS) - 0.45)
    ax.set_ylim(0, 53)
    ax.set_ylabel("Epoch")
    ax.set_title("Selected checkpoints and stopping epochs (25 coordinates per condition)", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.text(
        0.995,
        0.02,
        "Points are descriptive fold–seed coordinates; no confidence intervals or independent-replicate inference.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.98))
    return fig


def _matrix_for(bundle: PhaseABundle, condition: str, field: str) -> np.ndarray:
    matrix = np.empty((5, 5), dtype=float)
    lookup = {(run.outer_fold, run.seed): run for run in _condition_runs(bundle, condition)}
    for row, fold in enumerate(FOLDS):
        for column, seed in enumerate(SEEDS):
            matrix[row, column] = float(getattr(lookup[(fold, seed)], field))
    return matrix


def _render_heatmaps(
    bundle: PhaseABundle,
    *,
    field: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    value_format: str,
) -> plt.Figure:
    matrices = [_matrix_for(bundle, condition, field) for condition in CONDITIONS]
    vmin = min(float(matrix.min()) for matrix in matrices)
    vmax = max(float(matrix.max()) for matrix in matrices)
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 14.1), sharex=True, sharey=True)
    image_handle = None
    for ax, condition, matrix in zip(axes.ravel(), CONDITIONS, matrices):
        image_handle = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        for row in range(5):
            for column in range(5):
                normalized = 0.5 if vmax == vmin else (matrix[row, column] - vmin) / (vmax - vmin)
                red, green, blue, _ = matplotlib.colormaps[cmap](normalized)
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "#222222" if luminance > 0.56 else "white"
                ax.text(column, row, format(matrix[row, column], value_format), ha="center", va="center", fontsize=7.1, color=color)
        ax.set_xticks(range(5), [str(seed) for seed in SEEDS])
        ax.set_yticks(range(5), [str(fold) for fold in FOLDS])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Outer-fold index")
        ax.grid(False)
    assert image_handle is not None
    colorbar = fig.colorbar(image_handle, ax=axes.ravel().tolist(), fraction=0.024, pad=0.025)
    colorbar.set_label(colorbar_label)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.987)
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.045, hspace=0.28, wspace=0.18)
    return fig


def render_selected_epoch_heatmaps(bundle: PhaseABundle) -> plt.Figure:
    return _render_heatmaps(
        bundle,
        field="selected_epoch",
        title="Phase A selected checkpoint epoch by fold and seed",
        colorbar_label="Selected epoch",
        cmap="viridis",
        value_format=".0f",
    )


def render_selected_threshold_heatmaps(bundle: PhaseABundle) -> plt.Figure:
    return _render_heatmaps(
        bundle,
        field="selected_threshold",
        title="Phase A validation-selected threshold by fold and seed",
        colorbar_label="Selected threshold",
        cmap="magma",
        value_format=".2f",
    )


def render_threshold_distribution(bundle: PhaseABundle) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.6, 6.3))
    values = [[run.selected_threshold for run in _condition_runs(bundle, condition)] for condition in CONDITIONS]
    box = ax.boxplot(
        values,
        positions=np.arange(len(CONDITIONS)),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.5},
        whiskerprops={"color": "#777777"},
        capprops={"color": "#777777"},
    )
    for patch, condition in zip(box["boxes"], CONDITIONS):
        patch.set_facecolor(COLORS[condition])
        patch.set_alpha(0.22)
        patch.set_edgecolor(COLORS[condition])
    for index, condition in enumerate(CONDITIONS):
        runs = _condition_runs(bundle, condition)
        ax.scatter(
            index + _coordinate_offsets(runs),
            [run.selected_threshold for run in runs],
            s=19,
            color=COLORS[condition],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3,
        )
    ax.axhline(0.5, color="#666666", linestyle=":", lw=1.0, label="0.50 reference")
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("Validation-selected call threshold")
    ax.set_ylim(0, 1)
    ax.set_title("Distribution of validation-selected thresholds", fontweight="bold")
    ax.legend(frameon=False)
    ax.text(
        0.995,
        0.02,
        "Boxes and points describe the 25 fold–seed coordinates; they are not inferential uncertainty intervals.",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.4,
        color="#555555",
    )
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.98))
    return fig


def render_threshold_response(bundle: PhaseABundle) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(11.8, 13.2), sharex=True, sharey=True)
    for ax, condition in zip(axes.ravel(), CONDITIONS):
        runs = _condition_runs(bundle, condition)
        grid = np.asarray([float(row["threshold"]) for row in runs[0].threshold_candidates])
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
        ax.fill_between(grid, q25, q75, color=COLORS[condition], alpha=0.16, linewidth=0)
        ax.plot(grid, median, color=COLORS[condition], lw=1.7)
        y_floor = max(0.0, float(np.nanmin(q25)) - 0.015)
        ax.vlines(
            [run.selected_threshold for run in runs],
            y_floor,
            y_floor + 0.012,
            color="#222222",
            alpha=0.36,
            lw=0.7,
        )
        ax.set_title(condition, color=COLORS[condition], fontweight="bold")
        ax.set_ylabel("Validation macro-F1")
        ax.set_xlim(0, 1)
        ax.set_ylim(y_floor, 1.01)
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
        0.026,
        "Line = median and band = interquartile range across all 25 coordinates at each threshold; black rugs mark selected thresholds. "
        "Descriptive only—no confidence intervals or fold/seed independence assumption.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.052, 0.98, 0.96))
    return fig


def _distribution_plot(
    ax: plt.Axes,
    bundle: PhaseABundle,
    value_getter,
    *,
    ylabel: str,
) -> None:
    for index, condition in enumerate(CONDITIONS):
        runs = _condition_runs(bundle, condition)
        values = np.asarray([value_getter(run) for run in runs], dtype=float)
        ax.scatter(
            index + _coordinate_offsets(runs),
            values,
            s=20,
            color=COLORS[condition],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.hlines(np.median(values), index - 0.31, index + 0.31, color="#222222", lw=1.7)
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)


def render_runtime_distribution(bundle: PhaseABundle) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.6, 6.3))
    _distribution_plot(
        ax,
        bundle,
        lambda run: run.wall_time_seconds / 60.0,
        ylabel="Training + validation wall time (minutes)",
    )
    ax.set_title("Phase A coordinate runtime", fontweight="bold")
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
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.98))
    return fig


def render_parameter_counts(bundle: PhaseABundle) -> plt.Figure:
    totals: list[int] = []
    trainable: list[int] = []
    for condition in CONDITIONS:
        runs = _condition_runs(bundle, condition)
        total_values = {run.total_parameters for run in runs}
        trainable_values = {run.trainable_parameters for run in runs}
        if len(total_values) != 1 or len(trainable_values) != 1:
            raise FigurePipelineError(f"{condition} parameter count varies across coordinates")
        totals.append(total_values.pop())
        trainable.append(trainable_values.pop())
    fig, ax = plt.subplots(figsize=(12.3, 6.0))
    positions = np.arange(len(CONDITIONS))
    bars = ax.bar(positions, np.asarray(totals) / 1000.0, color=[COLORS[c] for c in CONDITIONS], alpha=0.88)
    for bar, total, train in zip(bars, totals, trainable):
        label = f"{total:,}"
        if train != total:
            label += f"\n({train:,} trainable)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.2, label, ha="center", va="bottom", fontsize=7.1)
    ax.set_xticks(positions, CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("Parameters (thousands)")
    ax.set_title("Model capacity in the executed Phase A matrix", fontweight="bold")
    ax.set_ylim(0, max(totals) / 1000.0 + 50)
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.98))
    return fig


def render_gpu_memory_distribution(bundle: PhaseABundle) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.9), sharex=True)
    mib = 1024.0 * 1024.0
    _distribution_plot(
        axes[0],
        bundle,
        lambda run: run.peak_cuda_allocated_bytes / mib,
        ylabel="Peak CUDA allocated memory (MiB)",
    )
    _distribution_plot(
        axes[1],
        bundle,
        lambda run: run.peak_cuda_reserved_bytes / mib,
        ylabel="Peak CUDA reserved memory (MiB)",
    )
    axes[0].set_title("Allocated", fontweight="bold")
    axes[1].set_title("Reserved", fontweight="bold")
    fig.suptitle("Phase A peak GPU-memory footprint", fontsize=14, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.04,
        "Horizontal bars are medians; points are all 25 fold–seed coordinates per condition.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    _stamp(fig, DIAGNOSTIC_LABEL)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))
    return fig


def _load_figure_spec(
    project_root: Path,
) -> tuple[ArtifactReference, Mapping[str, Any]]:
    path = project_root / FIGURE_SPEC_RELATIVE
    reference = _artifact_for_file(path, project_root)
    raw = _read_json(path, context="pre-outer figure specification")
    if not isinstance(raw, Mapping):
        raise FigurePipelineError("figure specification is not an object")
    if (
        raw.get("schema_version") != "paper2-v2-work9-phase-a-figure-spec-1"
        or raw.get("status") != "specified_before_outer_evaluation"
        or raw.get("protocol_id") != PROTOCOL_ID
    ):
        raise FigurePipelineError("figure specification identity/status is invalid")
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
    ):
        raise FigurePipelineError("figure specification weakens no-cherry-pick/no-LOCF rules")
    export = raw.get("export")
    if not isinstance(export, Mapping) or tuple(export.get("formats", ())) != FORMATS:
        raise FigurePipelineError("figure specification export formats differ")
    figures = raw.get("figures")
    if not isinstance(figures, list):
        raise FigurePipelineError("figure specification has no figure inventory")
    observed_ids = [row.get("figure_id") for row in figures if isinstance(row, Mapping)]
    if observed_ids != list(_expected_figure_ids("all")):
        raise FigurePipelineError("figure specification IDs/order differ from the implementation")
    return reference, raw


def _expected_figure_ids(mode: str) -> tuple[str, ...]:
    outcome_free = ("protocol_pipeline", "architecture_comparison")
    phase_a = (
        "phase_a_learning_curves_aggregate",
        *(f"phase_a_learning_curves_{condition}_all25" for condition in CONDITIONS),
        "phase_a_epoch_and_stopping_distributions",
        "phase_a_selected_epoch_heatmaps",
        "phase_a_selected_threshold_heatmaps",
        "phase_a_selected_threshold_distribution",
        "phase_a_threshold_response",
        "phase_a_runtime_distribution",
        "phase_a_parameter_counts",
        "phase_a_gpu_memory_distribution",
    )
    if mode == "outcome-free":
        return outcome_free
    if mode == "phase-a":
        return phase_a
    if mode == "all":
        return outcome_free + phase_a
    raise FigurePipelineError(f"unsupported generation mode: {mode}")


def _render_inventory(
    mode: str,
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    bundle: PhaseABundle | None,
) -> Iterable[tuple[str, plt.Figure, str]]:
    if mode in {"outcome-free", "all"}:
        yield "protocol_pipeline", render_protocol_pipeline(config, protocol), OUTCOME_FREE_LABEL
        yield "architecture_comparison", render_architecture_comparison(config), OUTCOME_FREE_LABEL
    if mode not in {"phase-a", "all"}:
        return
    if bundle is None:
        raise FigurePipelineError("Phase-A rendering requires a verified selection bundle")
    yield (
        "phase_a_learning_curves_aggregate",
        render_aggregate_learning_curves(bundle),
        DIAGNOSTIC_LABEL,
    )
    for condition in CONDITIONS:
        yield (
            f"phase_a_learning_curves_{condition}_all25",
            render_condition_small_multiples(bundle, condition),
            DIAGNOSTIC_LABEL,
        )
    yield (
        "phase_a_epoch_and_stopping_distributions",
        render_epoch_distributions(bundle),
        DIAGNOSTIC_LABEL,
    )
    yield "phase_a_selected_epoch_heatmaps", render_selected_epoch_heatmaps(bundle), DIAGNOSTIC_LABEL
    yield (
        "phase_a_selected_threshold_heatmaps",
        render_selected_threshold_heatmaps(bundle),
        DIAGNOSTIC_LABEL,
    )
    yield (
        "phase_a_selected_threshold_distribution",
        render_threshold_distribution(bundle),
        DIAGNOSTIC_LABEL,
    )
    yield "phase_a_threshold_response", render_threshold_response(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_runtime_distribution", render_runtime_distribution(bundle), DIAGNOSTIC_LABEL
    yield "phase_a_parameter_counts", render_parameter_counts(bundle), DIAGNOSTIC_LABEL
    yield (
        "phase_a_gpu_memory_distribution",
        render_gpu_memory_distribution(bundle),
        DIAGNOSTIC_LABEL,
    )


def _source_references(project_root: Path) -> list[ArtifactReference]:
    module_path = Path(__file__).resolve()
    script_path = (project_root / FIGURE_SCRIPT_RELATIVE).resolve()
    return [
        _artifact_for_file(module_path, project_root),
        _artifact_for_file(script_path, project_root),
    ]


def _supersession_mapping(project_root: Path) -> dict[str, Any]:
    manifest = _artifact_for_file(
        project_root / SUPERSEDED_BUNDLE_RELATIVE / "figure_manifest.json",
        project_root,
    )
    record = _artifact_for_file(
        project_root / SUPERSEDED_BUNDLE_RELATIVE / "SUPERSEDED.json",
        project_root,
    )
    if manifest.sha256 != SUPERSEDED_MANIFEST_SHA256:
        raise FigurePipelineError("superseded v1 figure manifest hash changed")
    raw = _read_json(record.path, context="v1 figure supersession record")
    if (
        not isinstance(raw, Mapping)
        or raw.get("status") != "superseded_after_visual_audit"
        or raw.get("superseded_manifest_sha256") != manifest.sha256
        or raw.get("outer_outcomes_accessed") is not False
    ):
        raise FigurePipelineError("v1 figure supersession record is invalid")
    return {
        "reason_code": raw["reason_code"],
        "manifest": manifest.to_mapping(),
        "supersession_record": record.to_mapping(),
    }


def _output_ref_mapping(
    staged_path: Path,
    final_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    if not staged_path.is_file() or staged_path.is_symlink():
        raise FigurePipelineError(f"figure output is missing or symlinked: {staged_path}")
    return {
        "path": _project_relative(final_path, project_root),
        "sha256": sha256_file(staged_path),
        "bytes": staged_path.stat().st_size,
        "row_count": None,
    }


def _package_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("matplotlib", "numpy", "pillow", "contourpy", "fonttools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "matplotlib_backend": matplotlib.get_backend(),
        "packages": packages,
    }


def generate_work9_figures(
    project_root: str | Path,
    *,
    output_directory: str | Path,
    mode: str = "all",
) -> Path:
    """Generate an atomic, non-overwriting Work 9 figure bundle.

    ``mode='outcome-free'`` produces only the protocol and architecture
    schematics.  ``mode='phase-a'`` and ``mode='all'`` fail closed unless the
    complete frozen selection index exists and matches the pre-outer figure
    specification.
    """

    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    if root not in output.parents:
        raise FigurePipelineError("figure output directory must be inside the project root")
    if output.exists() or output.is_symlink():
        raise FigurePipelineError("refusing to overwrite an existing figure output path")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise FigurePipelineError("figure output parent must not be a symlink")

    spec_ref, spec = _load_figure_spec(root)
    config_ref, protocol_ref, config, protocol = load_outcome_free_protocol(root)
    bundle: PhaseABundle | None = None
    if mode in {"phase-a", "all"}:
        bundle = load_phase_a_bundle(root)
        declared_index = spec.get("selection_index", {})
        if (
            declared_index.get("path") != bundle.selection_index_ref.serialized_path
            or declared_index.get("sha256") != bundle.selection_index_ref.sha256
        ):
            raise FigurePipelineError(
                "selection index does not match the pre-outer figure specification"
            )
        if bundle.config_ref.sha256 != config_ref.sha256 or bundle.fold_protocol_ref.sha256 != protocol_ref.sha256:
            raise FigurePipelineError("protocol inputs differ between diagram and Phase-A loaders")

    source_refs = _source_references(root)
    supersession = _supersession_mapping(root)
    formats = tuple(spec["export"]["formats"])
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        figures_directory = staging / "figures"
        figures_directory.mkdir()
        figure_records: list[dict[str, Any]] = []
        generated_ids: list[str] = []
        _configure_style()
        for figure_id, figure, boundary_label in _render_inventory(
            mode, config=config, protocol=protocol, bundle=bundle
        ):
            generated_ids.append(figure_id)
            staged_base = figures_directory / figure_id
            saved = _save_figure(figure, staged_base, formats)
            files: dict[str, Any] = {}
            for staged_path in saved:
                final_path = output / "figures" / staged_path.name
                files[staged_path.suffix.removeprefix(".")] = _output_ref_mapping(
                    staged_path, final_path, root
                )
            figure_records.append(
                {
                    "figure_id": figure_id,
                    "diagnostic_boundary_label": boundary_label,
                    "files": files,
                }
            )
        if tuple(generated_ids) != _expected_figure_ids(mode):
            raise FigurePipelineError("rendered figure inventory differs from specification")

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
        generation_material = {
            "figure_spec_sha256": spec_ref.sha256,
            "selection_index_sha256": None if bundle is None else bundle.selection_index_ref.sha256,
            "source_sha256": [reference.sha256 for reference in source_refs],
            "mode": mode,
        }
        generation_id = hashlib.sha256(_json_bytes(generation_material)).hexdigest()
        manifest = {
            "schema_version": FIGURE_MANIFEST_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "generation_id": generation_id,
            "mode": mode,
            "supersedes": supersession,
            "diagnostic_boundary": spec["diagnostic_boundary"],
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
            "environment": _package_versions(),
            "sources": [reference.to_mapping() for reference in source_refs],
            "inputs": {
                "figure_spec": spec_ref.to_mapping(),
                "selection_index": None if bundle is None else bundle.selection_index_ref.to_mapping(),
                "work8_config": config_ref.to_mapping(),
                "fold_protocol_metadata": protocol_ref.to_mapping(),
                "phase_a_artifacts": phase_inputs,
            },
            "figure_count": len(figure_records),
            "file_count": sum(len(record["files"]) for record in figure_records),
            "figures": figure_records,
        }
        manifest_path = staging / "figure_manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if output.exists() or output.is_symlink():
            raise FigurePipelineError("figure output appeared during generation; refusing overwrite")
        os.rename(staging, output)
        return output / "figure_manifest.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_figure_manifest(project_root: str | Path, manifest_path: str | Path) -> Mapping[str, Any]:
    """Verify all immutable inputs and outputs recorded by a generated bundle."""

    root = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise FigurePipelineError("figure manifest is missing or symlinked")
    raw = _read_json(path, context="figure manifest")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != FIGURE_MANIFEST_SCHEMA:
        raise FigurePipelineError("unsupported figure manifest")
    if raw.get("status") != "complete" or raw.get("protocol_id") != PROTOCOL_ID:
        raise FigurePipelineError("figure manifest is incomplete or for another protocol")
    for source in raw.get("sources", []):
        _resolve_artifact_reference(source, root, context="figure source")
    supersedes = raw.get("supersedes")
    if not isinstance(supersedes, Mapping):
        raise FigurePipelineError("figure manifest lacks the v1 supersession binding")
    _resolve_artifact_reference(
        supersedes.get("manifest", {}), root, context="superseded v1 figure manifest"
    )
    _resolve_artifact_reference(
        supersedes.get("supersession_record", {}),
        root,
        context="v1 figure supersession record",
    )
    inputs = raw.get("inputs", {})
    for name in ("figure_spec", "work8_config", "fold_protocol_metadata"):
        _resolve_artifact_reference(inputs.get(name, {}), root, context=f"figure input {name}")
    if inputs.get("selection_index") is not None:
        _resolve_artifact_reference(
            inputs["selection_index"], root, context="figure input selection index"
        )
    for run in inputs.get("phase_a_artifacts", []):
        for name in ("epoch_trace", "threshold_trace", "selection_run_record"):
            _resolve_artifact_reference(
                run.get(name, {}), root, context=f"figure input {run.get('run_id')} {name}"
            )
    files_seen = 0
    for record in raw.get("figures", []):
        for extension, reference in record.get("files", {}).items():
            if extension not in FORMATS:
                raise FigurePipelineError("figure manifest contains an unexpected format")
            _resolve_artifact_reference(reference, root, context=f"figure {record.get('figure_id')}")
            files_seen += 1
    if files_seen != raw.get("file_count"):
        raise FigurePipelineError("figure manifest file count mismatch")
    if len(raw.get("figures", [])) != raw.get("figure_count"):
        raise FigurePipelineError("figure manifest figure count mismatch")
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
        raise FigurePipelineError("figure manifest does not preserve the no-outer boundary")
    return raw
