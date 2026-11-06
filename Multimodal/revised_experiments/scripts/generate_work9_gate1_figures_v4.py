#!/usr/bin/env python3
"""Generate or verify the presentation-corrected Work 9 Gate-1 v4 figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "revised_experiments/src"
DEFAULT_GATE1_DIRECTORY = PROJECT_ROOT / "revised_experiments/artifacts/work9_gate1_v2"
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "revised_experiments/artifacts/work9_gate1_figures_v4"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser(
        "generate", help="validate sealed Gate-1 evidence and atomically render v4"
    )
    generate.add_argument(
        "--gate1-directory", type=Path, default=DEFAULT_GATE1_DIRECTORY
    )
    generate.add_argument(
        "--gate1-hashes-sha256",
        required=True,
        help="literal SHA-256 of the sealed Gate-1 hashes.sha256 file",
    )
    generate.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    verify = subparsers.add_parser("verify", help="strictly verify a sealed v4 bundle")
    verify.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "figure_manifest.json",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    sys.path.insert(0, str(SOURCE_ROOT))
    from paper2_v2.work9_gate1_figures_v4 import (
        generate_gate1_figures,
        sha256_file,
        verify_figure_manifest,
    )

    if arguments.command == "generate":
        manifest_path = generate_gate1_figures(
            PROJECT_ROOT,
            gate1_directory=arguments.gate1_directory,
            expected_gate1_hash_manifest_sha256=arguments.gate1_hashes_sha256,
            output_directory=arguments.output_directory,
        )
    else:
        manifest_path = arguments.manifest.resolve()
    manifest = verify_figure_manifest(PROJECT_ROOT, manifest_path)
    print(
        json.dumps(
            {
                "status": "pass",
                "generation_id": manifest["generation_id"],
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "figure_count": manifest["figure_count"],
                "file_count": manifest["file_count"],
                "within_corpus_only": manifest["claim_boundary"]["within_corpus_only"],
                "scientific_values_changed": manifest["visual_correction"][
                    "scientific_values_changed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
