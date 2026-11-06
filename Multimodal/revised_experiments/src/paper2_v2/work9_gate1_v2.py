"""Preservation-safe Work 9 Gate-1 validator v2 compatibility shim.

The frozen v1 analysis remains the sole implementation of every scientific
validation, metric, estimator, interval, test, multiplicity adjustment, and
decision rule.  This shim changes one representation-level equivalence check:
the selected checkpoint recorded inside the epoch trace may have a different
project-relative path from the selected-checkpoint copy bound by the selection
seal when both immutable references verify the same SHA-256, byte count, and
row count.

Both files are independently opened and checked by v1 ``ImmutableRef`` before
the content-identity tuple is compared.  Any content, size, or row-count
difference remains a hard failure.  The v1 module and evidence are never
modified on disk.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from pathlib import PurePosixPath
from threading import RLock
from typing import Any, Iterator, Mapping

from . import work9_gate1 as _v1
from .config import MatrixConfig


VALIDATOR_VERSION = "paper2-v2-work9-gate1-validator-2"
CHECKPOINT_EQUIVALENCE = "verified_sha256_bytes_row_count"

Gate1AnalysisError = _v1.Gate1AnalysisError
ImmutableRef = _v1.ImmutableRef

_FROZEN_V1_VALIDATE_EPOCH_TRACE = _v1._validate_epoch_trace
_PATCH_LOCK = RLock()


def checkpoint_content_identity(reference: ImmutableRef) -> tuple[str, int, int | None]:
    """Return the complete immutable content identity used by validator v2."""

    return (reference.sha256, reference.bytes, reference.row_count)


def _validate_canonical_checkpoint_paths(
    *,
    epoch_checkpoint: ImmutableRef,
    indexed_checkpoint: ImmutableRef,
    selected_epoch: int,
) -> None:
    """Require the two canonical runner locations within the same run root."""

    epoch_parts = PurePosixPath(epoch_checkpoint.serialized_path).parts
    indexed_parts = PurePosixPath(indexed_checkpoint.serialized_path).parts
    epoch_suffix = ("epochs", f"epoch_{selected_epoch:03d}", "model.safetensors")
    indexed_suffix = ("selection", "checkpoint.safetensors")
    if (
        len(epoch_parts) <= len(epoch_suffix)
        or tuple(epoch_parts[-len(epoch_suffix) :]) != epoch_suffix
    ):
        raise Gate1AnalysisError(
            "selected epoch checkpoint does not use the canonical "
            "epochs/epoch_NNN/model.safetensors path"
        )
    if (
        len(indexed_parts) <= len(indexed_suffix)
        or tuple(indexed_parts[-len(indexed_suffix) :]) != indexed_suffix
    ):
        raise Gate1AnalysisError(
            "indexed checkpoint does not use the canonical "
            "selection/checkpoint.safetensors path"
        )
    epoch_root = epoch_parts[: -len(epoch_suffix)]
    indexed_root = indexed_parts[: -len(indexed_suffix)]
    if epoch_root != indexed_root:
        raise Gate1AnalysisError(
            "selected epoch and indexed checkpoints do not share one canonical run root"
        )


def _declared_selected_checkpoint(
    *,
    project_root: Path,
    trace_ref: ImmutableRef,
) -> ImmutableRef | None:
    """Resolve the trace-declared selection without replacing v1 validation.

    Returning ``None`` for malformed/ambiguous structure delegates the failure
    to the frozen v1 validator.  A uniquely declared checkpoint is fully
    verified from disk before it can participate in the v2 equivalence rule.
    """

    trace = _v1._read_json(trace_ref.path, "epoch trace")
    if not isinstance(trace, Mapping):
        return None
    selected_epoch = trace.get("selected_epoch")
    epochs = trace.get("epochs")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch <= 0
        or not isinstance(epochs, list)
    ):
        return None
    matches = [
        row
        for row in epochs
        if isinstance(row, Mapping) and row.get("epoch") == selected_epoch
    ]
    if len(matches) != 1:
        return None
    checkpoint = matches[0].get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return None
    return ImmutableRef.from_mapping(
        checkpoint,
        project_root=project_root,
        context="epoch trace declared selected checkpoint (validator v2)",
        ledger=False,
    )


def _validate_epoch_trace_v2(
    *,
    project_root: Path,
    trace_ref: ImmutableRef,
    entry: Mapping[str, Any],
    refs: Mapping[str, ImmutableRef],
    config: MatrixConfig,
    expected_segments: Mapping[str, _v1.ManifestSegment],
    validation_manifest_sha256: str,
) -> tuple[_v1.ValidatedLedgerPair, dict[str, Any]]:
    """Run v1 epoch validation after the single documented equivalence shim."""

    selected = _declared_selected_checkpoint(
        project_root=project_root,
        trace_ref=trace_ref,
    )
    indexed = refs.get("checkpoint")
    if selected is None or not isinstance(indexed, ImmutableRef):
        return _FROZEN_V1_VALIDATE_EPOCH_TRACE(
            project_root=project_root,
            trace_ref=trace_ref,
            entry=entry,
            refs=refs,
            config=config,
            expected_segments=expected_segments,
            validation_manifest_sha256=validation_manifest_sha256,
        )

    selected_identity = checkpoint_content_identity(selected)
    indexed_identity = checkpoint_content_identity(indexed)
    _validate_canonical_checkpoint_paths(
        epoch_checkpoint=selected,
        indexed_checkpoint=indexed,
        selected_epoch=int(entry.get("selected_epoch", -1)),
    )
    if selected_identity != indexed_identity:
        raise Gate1AnalysisError(
            "selected epoch checkpoint and indexed checkpoint content identity "
            "mismatch under validator v2 (sha256, bytes, row_count)"
        )

    # If the canonical mappings already agree, invoke v1 unchanged.  Otherwise
    # replace only the comparison-side reference.  V1 still recomputes the
    # winning epoch and validates every epoch ledger/checkpoint independently.
    if selected.as_mapping() == indexed.as_mapping():
        delegated_refs = refs
    else:
        delegated_refs = dict(refs)
        delegated_refs["checkpoint"] = selected

    return _FROZEN_V1_VALIDATE_EPOCH_TRACE(
        project_root=project_root,
        trace_ref=trace_ref,
        entry=entry,
        refs=delegated_refs,
        config=config,
        expected_segments=expected_segments,
        validation_manifest_sha256=validation_manifest_sha256,
    )


@contextmanager
def _installed_epoch_validator() -> Iterator[None]:
    """Install the v2 shim only for one serialized v1 validation call."""

    with _PATCH_LOCK:
        if _v1._validate_epoch_trace is not _FROZEN_V1_VALIDATE_EPOCH_TRACE:
            raise Gate1AnalysisError(
                "cannot install Gate-1 validator v2: v1 epoch validator is already patched"
            )
        _v1._validate_epoch_trace = _validate_epoch_trace_v2
        try:
            yield
        finally:
            _v1._validate_epoch_trace = _FROZEN_V1_VALIDATE_EPOCH_TRACE


def validate_gate1_inputs(
    *,
    project_root: Path,
    matrix_config_path: Path,
    outer_ledger_index_path: Path,
    expected_outer_ledger_index_sha256: str,
    config: MatrixConfig | None = None,
) -> tuple[MatrixConfig, _v1.ValidatedGate1Input]:
    """Validate inputs with frozen v1 logic plus content-identity equivalence."""

    with _installed_epoch_validator():
        return _v1.validate_gate1_inputs(
            project_root=project_root,
            matrix_config_path=matrix_config_path,
            outer_ledger_index_path=outer_ledger_index_path,
            expected_outer_ledger_index_sha256=expected_outer_ledger_index_sha256,
            config=config,
        )


def run_gate1_analysis(
    *,
    project_root: Path,
    matrix_config_path: Path,
    outer_ledger_index_path: Path,
    expected_outer_ledger_index_sha256: str,
    decision_policy_path: Path,
    analysis_protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run frozen v1 Gate-1 analysis through the v2 validator shim."""

    with _installed_epoch_validator():
        return _v1.run_gate1_analysis(
            project_root=project_root,
            matrix_config_path=matrix_config_path,
            outer_ledger_index_path=outer_ledger_index_path,
            expected_outer_ledger_index_sha256=expected_outer_ledger_index_sha256,
            decision_policy_path=decision_policy_path,
            analysis_protocol_path=analysis_protocol_path,
            output_dir=output_dir,
        )


__all__ = [
    "CHECKPOINT_EQUIVALENCE",
    "Gate1AnalysisError",
    "VALIDATOR_VERSION",
    "checkpoint_content_identity",
    "run_gate1_analysis",
    "validate_gate1_inputs",
]
