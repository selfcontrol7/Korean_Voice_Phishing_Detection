#!/usr/bin/env python3
"""Fail-closed CLI for the isolated matched-capacity sensitivity family."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "revised_experiments/src"
CONFIG_PATH = (
    PROJECT_ROOT
    / "revised_experiments/configs/work8_mfcc_matched_within_corpus_v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate frozen inputs without training")
    smoke = subparsers.add_parser("smoke", help="run train/val-only CUDA smoke")
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--output-directory", type=Path, required=True)
    smoke.add_argument(
        "--condition",
        choices=("A-mean", "AT-cross-randtrain"),
        required=True,
    )
    run = subparsers.add_parser("run", help="execute one complete matrix phase")
    run.add_argument(
        "--phase",
        choices=("fit-select", "freeze-selection-index", "outer-eval"),
        required=True,
    )
    run.add_argument("--device", default="cuda:0")

    fit = subparsers.add_parser("_fit-coordinate", help=argparse.SUPPRESS)
    outer = subparsers.add_parser("_outer-coordinate", help=argparse.SUPPRESS)
    for child in (fit, outer):
        child.add_argument("--condition", required=True)
        child.add_argument("--outer-fold", type=int, required=True)
        child.add_argument("--seed", type=int, required=True)
        child.add_argument("--device", required=True)
    return parser


def _raw_matrix() -> tuple[str, list[str], list[int]]:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if raw.get("status") != "locked":
        raise RuntimeError("Work 8 config is not locked")
    return str(raw["protocol_id"]), list(raw["conditions"]), list(raw["seeds"])


def _configure_import_path() -> None:
    sys.path.insert(0, str(SOURCE_ROOT))


def _require_cuda_environment(seed: int, device: str) -> None:
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(
            f"PYTHONHASHSEED={seed} must be set before starting this process"
        )
    if device.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before starting this process"
        )


def _coordinates(arguments: argparse.Namespace):
    from paper2_v2.work9_capacity_types import RUN_FAMILY, CapacityCoordinates

    protocol_id, _, _ = _raw_matrix()
    return CapacityCoordinates(
        protocol_id,
        RUN_FAMILY,
        arguments.condition,
        0,
        arguments.outer_fold,
        arguments.seed,
    )


def _run_child(phase: str, condition: str, fold: int, seed: int, device: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["WORK9_CAPACITY_INTERNAL_MATRIX_PHASE"] = phase
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"_{phase}-coordinate",
        "--condition",
        condition,
        "--outer-fold",
        str(fold),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "validate":
        _configure_import_path()
        from paper2_v2.work9_capacity_training import (
            verify_frozen_work9_inputs,
            verify_primary_selection_anchor_lineage,
        )

        frozen = verify_frozen_work9_inputs(
            PROJECT_ROOT, require_runner_acceptance=False
        )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "protocol_id": frozen.config.protocol_id,
                    "runner_acceptance": (
                        "final"
                        if frozen.runner_acceptance_sha256 is not None
                        else "blocked_pending_two_cuda_smokes"
                    ),
                    "experiment_card_sha256": frozen.experiment_card_sha256,
                    "new_fit_coordinates": 150,
                    "immutable_primary_anchors": 50,
                    "anchor_lineage": verify_primary_selection_anchor_lineage(PROJECT_ROOT),
                    "outer_eval_predictions_accessed": False,
                    "training_launched": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "smoke":
        _require_cuda_environment(17, arguments.device)
        _configure_import_path()
        from paper2_v2.work9_capacity_training import run_cuda_smoke

        receipt = run_cuda_smoke(
            PROJECT_ROOT,
            output_directory=arguments.output_directory,
            device=arguments.device,
            condition=arguments.condition,
        )
        print(receipt)
        return 0
    if arguments.command in {"_fit-coordinate", "_outer-coordinate"}:
        internal_phase = arguments.command.removeprefix("_").removesuffix("-coordinate")
        if os.environ.get("WORK9_CAPACITY_INTERNAL_MATRIX_PHASE") != internal_phase:
            raise RuntimeError("individual capacity coordinate execution is internal only")
        _require_cuda_environment(arguments.seed, arguments.device)
        _configure_import_path()
        coordinates = _coordinates(arguments)
        from paper2_v2.work9_capacity_training import (
            fit_select_coordinate,
            outer_evaluate_reserved_coordinate,
            record_coordinate_failure,
        )

        try:
            if internal_phase == "fit":
                fit_select_coordinate(
                    PROJECT_ROOT, coordinates, device=arguments.device
                )
            else:
                outer_evaluate_reserved_coordinate(
                    PROJECT_ROOT, coordinates, device=arguments.device
                )
        except BaseException as error:
            try:
                record_coordinate_failure(
                    PROJECT_ROOT,
                    coordinates,
                    phase=internal_phase,
                    error=error,
                )
            finally:
                raise
        return 0

    if arguments.command == "run":
        protocol_id, _, seeds = _raw_matrix()
        del protocol_id
        _configure_import_path()
        from paper2_v2.work9_capacity_types import NEW_FIT_CONDITIONS

        conditions = list(NEW_FIT_CONDITIONS)
        if arguments.phase == "fit-select":
            from paper2_v2.work9_capacity_training import (
                verify_cuda_device_identity,
                verify_frozen_work9_inputs,
            )

            verify_frozen_work9_inputs(PROJECT_ROOT, require_runner_acceptance=True)
            verify_cuda_device_identity(arguments.device)
            for condition in conditions:
                for fold in range(5):
                    for seed in seeds:
                        _run_child("fit", condition, fold, seed, arguments.device)
            return 0
        if arguments.phase == "freeze-selection-index":
            from paper2_v2.work9_capacity_training import freeze_complete_selection_index

            print(freeze_complete_selection_index(PROJECT_ROOT))
            return 0
        # Matrix-wide outer release only; no public coordinate selector exists.
        from paper2_v2.work9_capacity_training import (
            freeze_complete_outer_ledger_index,
            verify_capacity_selection_index_for_outer_release,
            verify_cuda_device_identity,
        )

        verify_cuda_device_identity(arguments.device)
        release = verify_capacity_selection_index_for_outer_release(PROJECT_ROOT)
        if release["coordinate_count"] != 200:
            raise RuntimeError("outer release requires exact strict hybrid verification")
        for condition in conditions:
            for fold in range(5):
                for seed in seeds:
                    _run_child("outer", condition, fold, seed, arguments.device)
        print(freeze_complete_outer_ledger_index(PROJECT_ROOT))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(main())
