#!/usr/bin/env python3
"""Analyze a complete immutable Work 9 primary-matrix outer-ledger index."""

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
    from paper2_v2.work9_gate1 import Gate1AnalysisError, run_gate1_analysis

    matrix = args.matrix_config if args.matrix_config.is_absolute() else root / args.matrix_config
    index = (
        args.outer_ledger_index
        if args.outer_ledger_index.is_absolute()
        else root / args.outer_ledger_index
    )
    decision_policy = (
        args.decision_policy
        if args.decision_policy.is_absolute()
        else root / args.decision_policy
    )
    analysis_protocol = (
        args.analysis_protocol
        if args.analysis_protocol.is_absolute()
        else root / args.analysis_protocol
    )
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        result = run_gate1_analysis(
            project_root=root,
            matrix_config_path=matrix,
            outer_ledger_index_path=index,
            expected_outer_ledger_index_sha256=args.outer_ledger_index_sha256,
            decision_policy_path=decision_policy,
            analysis_protocol_path=analysis_protocol,
            output_dir=output,
        )
    except (Gate1AnalysisError, FileExistsError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"status": "fail", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
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
