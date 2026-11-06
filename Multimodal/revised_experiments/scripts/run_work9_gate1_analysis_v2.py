#!/usr/bin/env python3
"""Run Work 9 Gate-1 with the preservation-safe checkpoint-identity validator v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=Path(
            "revised_experiments/configs/work8_mfcc_matched_within_corpus_v1.json"
        ),
    )
    parser.add_argument("--outer-ledger-index", type=Path, required=True)
    parser.add_argument(
        "--outer-ledger-index-sha256",
        required=True,
        help="Precommitted lowercase SHA-256 of the frozen 200-entry outer index.",
    )
    parser.add_argument(
        "--decision-policy",
        type=Path,
        default=Path("revised_experiments/configs/work9_gate1_decision_policy_v1.json"),
    )
    parser.add_argument(
        "--analysis-protocol",
        type=Path,
        default=Path(
            "revised_experiments/configs/work9_gate1_analysis_protocol_v1.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "revised_experiments" / "src"))
    from paper2_v2.work9_gate1_v2 import (
        CHECKPOINT_EQUIVALENCE,
        VALIDATOR_VERSION,
        Gate1AnalysisError,
        run_gate1_analysis,
    )

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        result = run_gate1_analysis(
            project_root=root,
            matrix_config_path=rooted(args.matrix_config),
            outer_ledger_index_path=rooted(args.outer_ledger_index),
            expected_outer_ledger_index_sha256=args.outer_ledger_index_sha256,
            decision_policy_path=rooted(args.decision_policy),
            analysis_protocol_path=rooted(args.analysis_protocol),
            output_dir=rooted(args.output_dir),
        )
    except (Gate1AnalysisError, FileExistsError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "validator_version": VALIDATOR_VERSION,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "validator_version": VALIDATOR_VERSION,
                "checkpoint_equivalence": CHECKPOINT_EQUIVALENCE,
                "output_dir": result["output_dir"],
                "hash_manifest_sha256": result["hash_manifest_sha256"],
                "primary_comparisons": result["report"]["primary_comparisons"],
                "gate1_continuation_classification": result["report"][
                    "gate1_continuation_classification"
                ],
                "practical_utility": result["report"]["practical_utility"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
