#!/usr/bin/env python3
"""Run and seal the dedicated Work 9 capacity post-completion closure suite."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "revised_experiments"
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from paper2_v2.config import sha256_file  # noqa: E402
from paper2_v2.work9_capacity_postcompletion_closure import (  # noqa: E402
    CONFIG_RELATIVE,
    RECEIPT_ROOT_RELATIVE,
    SCHEMA_VERSION,
    artifact_record,
    validate_postcompletion_closure,
    verify_dedicated_receipt,
)


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o444 if path.is_file() else 0o555)
    os.chmod(root, 0o555)


def main() -> int:
    destination = PROJECT_ROOT / RECEIPT_ROOT_RELATIVE
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"post-completion closure receipt already exists: {destination}")
    staging = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=False, exist_ok=False)
    mpl_directory = Path(tempfile.mkdtemp(prefix="work9-capacity-closure-mpl-"))
    try:
        command = [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_work9_capacity_postcompletion_closure",
            "-v",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src"
        environment["MPLCONFIGDIR"] = str(mpl_directory)
        completed = subprocess.run(
            command,
            cwd=EXPERIMENT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        log_path = staging / "dedicated_test.log"
        log_path.write_text(combined, encoding="utf-8")
        if completed.returncode != 0 or "Ran 6 tests" not in combined or "\nOK\n" not in combined:
            raise RuntimeError(
                "dedicated post-completion closure suite failed; staging evidence preserved"
            )
        closure = validate_postcompletion_closure(PROJECT_ROOT)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "test_status": "dedicated_postcompletion_suite_pass",
            "tests_run": 6,
            "errors": 0,
            "failures": 0,
            "command": (
                "PYTHONPATH=src python -m unittest "
                "tests.test_work9_capacity_postcompletion_closure -v"
            ),
            "config": artifact_record(PROJECT_ROOT / CONFIG_RELATIVE, PROJECT_ROOT),
            "validator_source": artifact_record(
                PROJECT_ROOT
                / "revised_experiments/src/paper2_v2/"
                "work9_capacity_postcompletion_closure.py",
                PROJECT_ROOT,
            ),
            "validator_cli": artifact_record(Path(__file__), PROJECT_ROOT),
            "validator_tests": artifact_record(
                PROJECT_ROOT
                / "revised_experiments/tests/"
                "test_work9_capacity_postcompletion_closure.py",
                PROJECT_ROOT,
            ),
            "test_log": artifact_record(
                log_path,
                PROJECT_ROOT,
            ),
            "closure": closure,
            "receipt_scope": (
                "post_completion_read_only_closure_not_runner_reacceptance"
            ),
            "legacy_full_discover_status": (
                "expected_historical_boundary_signal_not_pass"
            ),
        }
        # The staging log's final reference must name its immutable destination.
        receipt["test_log"]["path"] = (
            RECEIPT_ROOT_RELATIVE / "dedicated_test.log"
        ).as_posix()
        receipt_path = staging / "dedicated_test_receipt.json"
        receipt_path.write_text(
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "READ_ONLY_NOTICE.txt").write_text(
            "Read-only post-completion closure. This does not reaccept the runner, "
            "launch training/smokes/selection/outer evaluation, or suppress the "
            "documented historical full-discover boundary signal.\n",
            encoding="utf-8",
        )
        hashes = [
            f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}"
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "hashes.sha256"
        ]
        (staging / "hashes.sha256").write_text(
            "\n".join(hashes) + "\n", encoding="utf-8"
        )
        _freeze_tree(staging)
        os.rename(staging, destination)
        verified = verify_dedicated_receipt(PROJECT_ROOT)
        print(
            json.dumps(
                {
                    "status": verified["status"],
                    "tests_run": verified["tests_run"],
                    "receipt": (
                        RECEIPT_ROOT_RELATIVE / "dedicated_test_receipt.json"
                    ).as_posix(),
                    "receipt_sha256": sha256_file(
                        destination / "dedicated_test_receipt.json"
                    ),
                    "historical_full_discover_status": verified[
                        "legacy_full_discover_status"
                    ],
                },
                sort_keys=True,
            )
        )
    except BaseException:
        # Preserve any generated failure evidence in its unique staging directory.
        raise
    finally:
        shutil.rmtree(mpl_directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
