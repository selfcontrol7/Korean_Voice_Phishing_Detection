#!/usr/bin/env python3
"""Validate v1 feature files against the v2 loader contract without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from paper2_v2.data import FeatureDataset, InputSpec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--global-feature", action="append", default=[])
    parser.add_argument("--text-path-key", default="krsbert_text_path")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    spec = InputSpec(
        sequence_features=tuple(args.sequence),
        global_features=tuple(args.global_feature),
        text_path_key=args.text_path_key,
    )
    report: dict[str, object] = {
        "schema_version": "paper2-v2-feature-contract-audit-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "input_spec": {
            "sequence_features": list(spec.sequence_features),
            "global_features": list(spec.global_features),
            "text_path_key": spec.text_path_key,
        },
        "manifests": [],
    }
    total_rows = 0
    total_errors = 0
    for raw_manifest in args.manifest:
        manifest = raw_manifest.resolve()
        entry: dict[str, object] = {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        }
        errors: Counter[str] = Counter()
        examples: list[str] = []
        try:
            dataset = FeatureDataset(
                manifest,
                input_spec=spec,
                feature_base_dir=root,
                strict=True,
            )
            entry["rows"] = len(dataset)
            for index in range(len(dataset)):
                try:
                    dataset[index]
                except Exception as exc:  # audit records every bad row
                    errors[type(exc).__name__] += 1
                    if len(examples) < 20:
                        examples.append(f"row={index}: {exc}")
            total_rows += len(dataset)
        except Exception as exc:
            errors[type(exc).__name__] += 1
            examples.append(f"manifest: {exc}")
            entry["rows"] = 0
        error_count = sum(errors.values())
        total_errors += error_count
        entry["error_count"] = error_count
        entry["error_types"] = dict(sorted(errors.items()))
        entry["error_examples"] = examples
        report["manifests"].append(entry)
    report["total_rows"] = total_rows
    report["total_errors"] = total_errors
    report["status"] = "pass" if total_errors == 0 else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": total_rows, "errors": total_errors}))
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
