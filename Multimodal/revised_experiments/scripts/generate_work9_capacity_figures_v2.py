#!/usr/bin/env python3
"""Render the presentation-corrected v2 of the seven capacity figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "revised_experiments/src"))

from paper2_v2.config import sha256_file  # noqa: E402
from paper2_v2.work9_capacity_figures_v2 import (  # noqa: E402
    FIGURE_OUTPUT_RELATIVE,
    PRODUCTION_ANALYSIS_REPORT_SHA256,
    render_capacity_figures_v2,
)


def _sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-report", type=Path, required=True)
    parser.add_argument("--analysis-report-sha256", type=_sha256, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=FIGURE_OUTPUT_RELATIVE,
    )
    arguments = parser.parse_args()
    if arguments.analysis_report_sha256 != PRODUCTION_ANALYSIS_REPORT_SHA256:
        parser.error("v2 requires the sealed capacity analysis report SHA-256")
    if arguments.output_directory != FIGURE_OUTPUT_RELATIVE:
        parser.error(f"v2 production output is fixed at {FIGURE_OUTPUT_RELATIVE}")
    manifest = render_capacity_figures_v2(
        PROJECT_ROOT,
        report_path=arguments.analysis_report,
        expected_report_sha256=arguments.analysis_report_sha256,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "status": "secondary_sensitivity_figures_presentation_corrected",
                "figure_count": 7,
                "manifest": manifest.relative_to(PROJECT_ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest),
                "scientific_content_unchanged": True,
                "immutable_v1_preserved": True,
                "gate1_gate2_unchanged": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
