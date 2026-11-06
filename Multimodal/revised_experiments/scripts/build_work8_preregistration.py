#!/usr/bin/env python3
"""Build or independently re-check the frozen Work 8 preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2_v2.config import sha256_file
from paper2_v2.work8_preregistration import (
    check_preregistration,
    write_preregistration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write:
        config_path, receipt_path = write_preregistration(args.project_root)
        config_sha256 = sha256_file(config_path)
        receipt_sha256 = sha256_file(receipt_path)
    else:
        config_sha256, receipt_sha256 = check_preregistration(args.project_root)
        root = args.project_root.resolve()
        config_path = root / (
            "revised_experiments/configs/"
            "work8_mfcc_matched_within_corpus_v1.json"
        )
        receipt_path = root / (
            "revised_experiments/artifacts/work8_preregistration_v1/"
            "dry_run_receipt.json"
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": "write" if args.write else "check",
                "training_launched": False,
                "outer_eval_predictions_accessed": False,
                "config": str(config_path),
                "config_sha256": config_sha256,
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

