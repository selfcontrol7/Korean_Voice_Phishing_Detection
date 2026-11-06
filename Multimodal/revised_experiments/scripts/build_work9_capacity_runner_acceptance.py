#!/usr/bin/env python3
"""Build the Work 9 runner freeze; finalization requires two CUDA smokes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "revised_experiments/src"))

from paper2_v2.work9_capacity_acceptance import build_draft, finalize_acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--draft", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    parser.add_argument("--smoke-receipt", type=Path, action="append", default=[])
    parser.add_argument("--unit-test-receipt", type=Path)
    arguments = parser.parse_args()
    if arguments.draft:
        if arguments.smoke_receipt or arguments.unit_test_receipt is not None:
            parser.error("draft mode does not accept smoke/test results")
        print(build_draft(PROJECT_ROOT))
        return 0
    if len(arguments.smoke_receipt) != 2 or arguments.unit_test_receipt is None:
        parser.error(
            "finalize requires two --smoke-receipt values and --unit-test-receipt"
        )
    print(
        finalize_acceptance(
            PROJECT_ROOT,
            smoke_receipts=arguments.smoke_receipt,
            unit_test_receipt=arguments.unit_test_receipt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
