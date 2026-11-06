#!/usr/bin/env python3
"""Generate or verify the isolated Work 9 Phase-A figure bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "revised_experiments/src"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "revised_experiments/artifacts/work9_phase_a_figures_v2"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="render a new atomic figure bundle")
    generate.add_argument(
        "--mode",
        choices=("outcome-free", "phase-a", "all"),
        default="all",
        help="phase-a/all require the exact frozen selection index",
    )
    generate.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify", help="verify hashes and no-outer declaration")
    verify.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT / "figure_manifest.json",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    sys.path.insert(0, str(SOURCE_ROOT))
    from paper2_v2.work9_figures import (
        generate_work9_figures,
        verify_figure_manifest,
    )

    if arguments.command == "generate":
        manifest_path = generate_work9_figures(
            PROJECT_ROOT,
            output_directory=arguments.output_directory,
            mode=arguments.mode,
        )
        payload = verify_figure_manifest(PROJECT_ROOT, manifest_path)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "manifest": str(manifest_path),
                    "generation_id": payload["generation_id"],
                    "figure_count": payload["figure_count"],
                    "file_count": payload["file_count"],
                    "outer_outcomes_used": payload["data_access_declaration"][
                        "outer_outcomes_used"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    payload = verify_figure_manifest(PROJECT_ROOT, arguments.manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest": str(arguments.manifest.resolve()),
                "generation_id": payload["generation_id"],
                "figure_count": payload["figure_count"],
                "file_count": payload["file_count"],
                "outer_outcomes_used": payload["data_access_declaration"][
                    "outer_outcomes_used"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
