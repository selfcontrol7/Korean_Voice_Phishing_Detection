#!/usr/bin/env python3
"""Run and immutably receipt the exact Work 9 acceptance test suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "revised_experiments/src"))

from paper2_v2.config import sha256_file
from paper2_v2.work9_acceptance import verify_draft_against_current
from paper2_v2.work9_registry import atomic_write_once
from paper2_v2.work9_types import UNIT_TEST_RECEIPT_SCHEMA_VERSION, canonical_json_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"unit-test output must be new: {output}")
    output.mkdir(parents=True, exist_ok=False)
    draft, draft_sha = verify_draft_against_current(PROJECT_ROOT)
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "revised_experiments/tests",
        "-v",
    ]
    environment = dict(os.environ)
    source_path = str(PROJECT_ROOT / "revised_experiments/src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    _, post_test_draft_sha = verify_draft_against_current(PROJECT_ROOT)
    if post_test_draft_sha != draft_sha:
        raise RuntimeError("runner acceptance DRAFT changed during unit tests")
    log_ref = atomic_write_once(output / "unittest.log", completed.stdout)
    text = completed.stdout.decode("utf-8", errors="replace")
    matches = re.findall(r"Ran (\d+) tests?", text)
    test_count = int(matches[-1]) if matches else 0
    status = "pass" if completed.returncode == 0 and test_count > 0 else "fail"
    receipt = {
        "schema_version": UNIT_TEST_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "tests_passed": test_count if status == "pass" else 0,
        "return_code": completed.returncode,
        "command": " ".join(command),
        "runner_acceptance_draft_sha256": draft_sha,
        "bundle_digests": draft["bundle_digests"],
        "environment_lock_sha256": draft["environment_lock"][
            "environment_lock_sha256"
        ],
        "environment_packages_sha256": draft["environment_lock"]["packages_sha256"],
        "environment_inventory_method": draft["environment_lock"][
            "inventory_method"
        ],
        "environment_distribution_roots": draft["environment_lock"][
            "distribution_roots"
        ],
        "log": {
            "path": log_ref.path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(log_ref.path),
            "bytes": log_ref.path.stat().st_size,
            "row_count": None,
        },
    }
    receipt_ref = atomic_write_once(
        output / "unit_test_receipt.json", canonical_json_bytes(receipt)
    )
    print(receipt_ref.path)
    if status != "pass":
        print(text, file=sys.stderr)
    return completed.returncode if completed.returncode else (0 if test_count else 1)


if __name__ == "__main__":
    raise SystemExit(main())
