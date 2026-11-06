#!/usr/bin/env python3
"""Validate the outcome-free draft blinded-error protocol scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "revised_experiments/src"))

from paper2_v2.work9_blinded_error_protocol import (  # noqa: E402
    build_draft_readiness_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--require-freeze-ready",
        action="store_true",
        help=(
            "Fail unless every PI decision is resolved. This is expected to fail "
            "for the current draft and never freezes or selects cases."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = build_draft_readiness_receipt(args.project_root)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    if args.require_freeze_ready and receipt["unresolved_pi_decision_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
