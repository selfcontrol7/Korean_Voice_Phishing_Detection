#!/usr/bin/env python3
"""Seal or verify the immutable Gate1-v2 figure-lineage companion receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "revised_experiments/src"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument(
        "--final-figure-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "revised_experiments/artifacts/work9_gate1_figures_v4/"
            "figure_manifest.json"
        ),
    )
    seal.add_argument("--final-figure-manifest-sha256", required=True)
    seal.add_argument(
        "--gate1-directory",
        type=Path,
        default=PROJECT_ROOT / "revised_experiments/artifacts/work9_gate1_v2",
    )
    seal.add_argument("--gate1-hashes-sha256", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--receipt",
        type=Path,
        default=(
            PROJECT_ROOT
            / "revised_experiments/artifacts/work9_gate1_figures_v2_lineage/"
            "lineage_receipt.json"
        ),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    sys.path.insert(0, str(SOURCE_ROOT))
    from paper2_v2.work9_gate1_figure_lineage_v2 import (
        FigureLineageError,
        seal_figure_lineage,
        sha256_file,
        verify_figure_lineage,
    )

    try:
        if arguments.command == "seal":
            receipt_path = seal_figure_lineage(
                PROJECT_ROOT,
                final_figure_manifest=arguments.final_figure_manifest,
                expected_final_manifest_sha256=arguments.final_figure_manifest_sha256,
                gate1_directory=arguments.gate1_directory,
                expected_gate1_hashes_sha256=arguments.gate1_hashes_sha256,
            )
        else:
            receipt_path = arguments.receipt.resolve()
        receipt = verify_figure_lineage(PROJECT_ROOT, receipt_path)
    except (FigureLineageError, FileExistsError, OSError, ValueError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "pass",
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "figure_count": receipt["copied_from_verified_final_v4_manifest"][
                    "figure_count"
                ],
                "plot_files_modified": receipt["assertions"][
                    "plot_files_modified_by_lineage_binding"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
