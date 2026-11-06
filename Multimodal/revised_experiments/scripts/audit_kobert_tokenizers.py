#!/usr/bin/env python3
"""Reproduce the historical-vs-correct KoBERT tokenization audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece
import transformers
from transformers import BertTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_correct_class(snapshot: Path):
    module_path = snapshot / "tokenization_kobert.py"
    spec = importlib.util.spec_from_file_location("paper2_v2_tokenization_kobert", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KoBertTokenizer


def audit(tokenizer, texts: list[str]) -> dict[str, object]:
    total = 0
    unknown = 0
    all_unknown = 0
    maximum = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        count = len(token_ids)
        unknown_count = sum(token_id == tokenizer.unk_token_id for token_id in token_ids)
        total += count
        unknown += unknown_count
        all_unknown += int(count > 0 and unknown_count == count)
        maximum = max(maximum, count)
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "content_tokens": total,
        "unknown_content_tokens": unknown,
        "unknown_content_rate": unknown / max(1, total),
        "all_content_unknown_segments": all_unknown,
        "maximum_content_tokens": maximum,
    }


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.resolve()
    texts = []
    manifests = []
    for raw_path in args.manifest:
        path = raw_path.resolve()
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    texts.append(str(json.loads(line)["text"]))
                    count += 1
        manifests.append({"path": str(path), "sha256": sha256_file(path), "rows": count})
    correct_class = load_correct_class(snapshot)
    historical = BertTokenizer.from_pretrained(snapshot, local_files_only=True)
    correct = correct_class.from_pretrained(snapshot, local_files_only=True)
    report = {
        "schema_version": "paper2-v2-kobert-tokenizer-audit-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "purpose": "diagnose historical KoBERT store; not a selected v2 core representation",
        "snapshot": {
            "path": str(snapshot),
            "revision": snapshot.name,
            "tokenization_code_sha256": sha256_file(snapshot / "tokenization_kobert.py"),
            "sentencepiece_model_sha256": sha256_file(snapshot / "tokenizer_78b3253a26.model"),
            "vocab_sha256": sha256_file(snapshot / "vocab.txt"),
        },
        "manifests": manifests,
        "segments": len(texts),
        "historical_plain_bert_tokenizer": audit(historical, texts),
        "correct_sentencepiece_tokenizer": audit(correct, texts),
        "software": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "sentencepiece": sentencepiece.__version__,
        },
        "decision": (
            "Exclude historical KoBERT vectors from v2. A corrected KoBERT feature store is "
            "optional Priority 2 work and is not needed for the core MFCC+KR-SBERT matrix."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "segments": len(texts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
