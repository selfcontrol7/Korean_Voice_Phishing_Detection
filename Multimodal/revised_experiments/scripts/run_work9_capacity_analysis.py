#!/usr/bin/env python3
"""Run the frozen Work 9 capacity-matched secondary analysis.

This command has no Gate-1/Gate-2 decision option.  The supplied index hashes
must identify one already frozen 200-coordinate hybrid release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "revised_experiments/src"))

from paper2_v2.config import sha256_file  # noqa: E402
from paper2_v2.work9_capacity_analysis import run_capacity_analysis  # noqa: E402


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase 64-character SHA-256")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-index", type=Path, required=True)
    parser.add_argument("--selection-index-sha256", type=_sha256, required=True)
    parser.add_argument("--outer-index", type=Path, required=True)
    parser.add_argument("--outer-index-sha256", type=_sha256, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_capacity_analysis(
        PROJECT_ROOT,
        selection_index_path=arguments.selection_index,
        expected_selection_index_sha256=arguments.selection_index_sha256,
        outer_index_path=arguments.outer_index,
        expected_outer_index_sha256=arguments.outer_index_sha256,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "status": "secondary_sensitivity_complete",
                "report": report.relative_to(PROJECT_ROOT).as_posix(),
                "report_sha256": sha256_file(report),
                "gate1_gate2_unchanged": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
