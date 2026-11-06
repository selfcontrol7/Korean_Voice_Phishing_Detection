#!/usr/bin/env python3
"""Build the immutable no-training capacity preflight receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "revised_experiments/src"))

from paper2_v2.work9_capacity_acceptance import bundle_digests, source_contracts
from paper2_v2.work9_capacity_registry import atomic_write_once
from paper2_v2.work9_capacity_training import validate_capacity_no_training
from paper2_v2.work9_capacity_types import DRY_RECEIPT_RELATIVE, canonical_json_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DRY_RECEIPT_RELATIVE.parent,
    )
    arguments = parser.parse_args()
    if (
        arguments.output_directory.is_absolute()
        or ".." in arguments.output_directory.parts
        or arguments.output_directory != DRY_RECEIPT_RELATIVE.parent
    ):
        raise ValueError("capacity dry receipt output is fixed by preregistration")
    output = (PROJECT_ROOT / DRY_RECEIPT_RELATIVE.parent).resolve()
    cursor = PROJECT_ROOT.resolve()
    for part in DRY_RECEIPT_RELATIVE.parent.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"capacity dry receipt path contains symlink: {cursor}")
    if output.exists():
        raise FileExistsError(f"capacity receipt output must be new: {output}")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        **validate_capacity_no_training(PROJECT_ROOT),
        "source_contracts": source_contracts(PROJECT_ROOT),
        "bundle_digests": bundle_digests(PROJECT_ROOT),
    }
    reference = atomic_write_once(
        output / "dry_run_receipt.json", canonical_json_bytes(payload)
    )
    output.chmod(0o555)
    print(reference.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
