#!/usr/bin/env python3
"""Run sealed Work 5 learned shortcut diagnostics on canonical Work 6 folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("revised_experiments/configs/work5_learned_shortcut_probes_v1.json"),
    )
    parser.add_argument(
        "--work6-root",
        type=Path,
        default=Path("data/paper2_v2/folds/grouped_nested_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revised_experiments/artifacts/work5_learned_shortcut_probes_grouped_v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "revised_experiments" / "src"))
    from paper2_v2.shortcut_probes import ShortcutProbeError, run_learned_shortcut_probes

    config = args.config if args.config.is_absolute() else root / args.config
    work6 = args.work6_root if args.work6_root.is_absolute() else root / args.work6_root
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        result = run_learned_shortcut_probes(
            project_root=root,
            config_path=config,
            work6_root=work6,
            output_dir=output,
        )
    except (ShortcutProbeError, FileExistsError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
