#!/usr/bin/env python3
"""CLI for Work 6 relation-aware grouped fold construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2_v2.folds import (
    FoldBuildConfig,
    build_grouped_folds,
    freeze_fold_tree,
    validate_fold_tree,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one repeat of five source-aware within-corpus outer folds. "
            "Run only after the Work 5 provenance/relation artifacts are reviewed."
        )
    )
    parser.add_argument("--call-provenance", type=Path, required=True)
    parser.add_argument(
        "--relations",
        type=Path,
        action="append",
        required=True,
        help="Work 5 relation component/edge JSONL; repeat for multiple inventories",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        action="append",
        required=True,
        help="Historical segment JSONL; provide train, val, and burned test manifests",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition-seed", type=int, default=20260809)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="seal read-only after independent reproducibility review",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_grouped_folds(
        call_provenance_path=args.call_provenance,
        relation_component_paths=args.relations,
        segment_manifest_paths=args.segments,
        output_dir=args.output_dir,
        config=FoldBuildConfig(partition_seed=args.partition_seed),
    )
    validate_fold_tree(result.output_dir)
    if args.freeze:
        freeze_fold_tree(result.output_dir)
    print(
        json.dumps(
            {
                "status": "frozen" if args.freeze else "generated_unfrozen",
                "output_dir": str(result.output_dir),
                "retained_call_count": result.retained_call_count,
                "quarantined_call_count": result.quarantined_call_count,
                "exact_duplicate_exclusion_count": result.exact_duplicate_exclusion_count,
                "next_action": "review Work 6 audit, then freeze hashes before Work 8 training",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
