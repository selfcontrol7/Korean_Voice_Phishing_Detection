#!/usr/bin/env python3
"""Validate sealed KR-SBERT stores and compare the rebuilt CLS to legacy files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from paper2_v2.text_features import IndexedTextStore, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-store", type=Path, required=True)
    parser.add_argument("--mean-store", type=Path, required=True)
    parser.add_argument("--legacy-cls-root", type=Path, required=True)
    parser.add_argument("--official-pooling-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cls_store = IndexedTextStore(args.cls_store, verify_hashes=True)
    mean_store = IndexedTextStore(args.mean_store, verify_hashes=True)
    if cls_store.row_by_segment != mean_store.row_by_segment:
        raise ValueError("CLS and mean indexes differ")
    cls = np.asarray(cls_store.vectors)
    mean = np.asarray(mean_store.vectors)
    if cls.shape != mean.shape:
        raise ValueError("CLS and mean shapes differ")
    if not np.isfinite(cls).all() or not np.isfinite(mean).all():
        raise ValueError("nonfinite rebuilt vector")

    denominators = np.linalg.norm(cls, axis=1) * np.linalg.norm(mean, axis=1)
    cosines = (cls * mean).sum(axis=1) / denominators
    maximum_absolute_error = 0.0
    absolute_error_sum = 0.0
    compared_values = 0
    minimum_cosine = 1.0
    missing = []
    legacy_root = args.legacy_cls_root.resolve()
    for segment_id, row_index in cls_store.row_by_segment.items():
        path = legacy_root / f"{segment_id}_krsbert.npy"
        if not path.is_file():
            missing.append(segment_id)
            continue
        legacy = np.load(path, allow_pickle=False)
        if legacy.shape != (cls.shape[1],) or legacy.dtype != np.float32:
            raise ValueError(f"bad legacy vector contract: {path} {legacy.shape} {legacy.dtype}")
        rebuilt = cls[row_index]
        difference = np.abs(legacy - rebuilt)
        maximum_absolute_error = max(maximum_absolute_error, float(difference.max()))
        absolute_error_sum += float(difference.sum())
        compared_values += int(difference.size)
        cosine = float(np.dot(legacy, rebuilt) / (np.linalg.norm(legacy) * np.linalg.norm(rebuilt)))
        minimum_cosine = min(minimum_cosine, cosine)

    report = {
        "schema_version": "paper2-v2-krsbert-validation-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not missing else "fail",
        "stores": {
            "cls": {
                "feature_set_id": cls_store.feature_set_id,
                "directory": str(cls_store.directory),
                "metadata_sha256": sha256_file(cls_store.directory / "feature_store.json"),
                "index_sha256": cls_store.metadata["index_sha256"],
                "vectors_sha256": cls_store.metadata["vectors_sha256"],
            },
            "mean": {
                "feature_set_id": mean_store.feature_set_id,
                "directory": str(mean_store.directory),
                "metadata_sha256": sha256_file(mean_store.directory / "feature_store.json"),
                "index_sha256": mean_store.metadata["index_sha256"],
                "vectors_sha256": mean_store.metadata["vectors_sha256"],
            },
        },
        "shape": list(cls.shape),
        "all_finite": True,
        "token_audit": cls_store.metadata["token_audit"],
        "cls_vs_mean_cosine": {
            "minimum": float(cosines.min()),
            "median": float(np.median(cosines)),
            "mean": float(cosines.mean()),
            "maximum": float(cosines.max()),
        },
        "rebuilt_cls_vs_legacy": {
            "legacy_root": str(legacy_root),
            "segments_expected": len(cls_store),
            "segments_missing": len(missing),
            "missing_examples": missing[:20],
            "values_compared": compared_values,
            "maximum_absolute_error": maximum_absolute_error,
            "mean_absolute_error": absolute_error_sum / max(1, compared_values),
            "minimum_cosine": minimum_cosine,
        },
    }
    if args.official_pooling_config is not None:
        pooling_path = args.official_pooling_config.resolve()
        pooling_config = json.loads(pooling_path.read_text(encoding="utf-8"))
        report["official_sentence_transformers_pooling"] = {
            "path": str(pooling_path),
            "sha256": sha256_file(pooling_path),
            "config": pooling_config,
            "canonical_mean_confirmed": bool(
                pooling_config.get("pooling_mode_mean_tokens")
                and not pooling_config.get("pooling_mode_cls_token")
                and not pooling_config.get("pooling_mode_max_tokens")
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "shape": report["shape"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
