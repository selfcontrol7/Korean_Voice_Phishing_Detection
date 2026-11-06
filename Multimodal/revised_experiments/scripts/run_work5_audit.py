#!/usr/bin/env python3
"""Generate the immutable Paper 2 v2 Work 5 audit artifact set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Multimodal project directory; commands are normally run from Multimodal/",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("revised_experiments/configs/work5_audit_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revised_experiments/artifacts/work5_data_audit_v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    source_dir = root / "revised_experiments" / "src"
    sys.path.insert(0, str(source_dir))
    from paper2_v2.work5_audit import AuditError, run_work5_audit

    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        summary = run_work5_audit(project_root=root, config_path=config, output_dir=output)
    except AuditError as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
