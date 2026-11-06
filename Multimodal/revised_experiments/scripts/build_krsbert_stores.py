#!/usr/bin/env python3
"""Build pinned KR-SBERT CLS and canonical masked-mean stores in one pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from paper2_v2.text_features import sha256_file


MODEL_ID = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
MODEL_REVISION = "92c6c2c7032f680bff0f9f0c63fadd3f97e635b2"
MAX_LENGTH = 128
DIMENSION = 768
CLS_STORE_ID = "text-krsbert-92c6c2c7-cls-mask128-f32-v1"
MEAN_STORE_ID = "text-krsbert-92c6c2c7-mean-mask128-f32-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_inventory(manifests: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_segment: dict[str, dict[str, Any]] = {}
    manifest_records = []
    for manifest in manifests:
        path = manifest.resolve()
        split = path.name.split("_segment_manifest", 1)[0]
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                segment_id = str(row["segment_id"])
                record = {
                    "segment_id": segment_id,
                    "call_id": str(row["call_id"]),
                    "text": str(row["text"]),
                    "historical_split": split,
                }
                if segment_id in by_segment and by_segment[segment_id] != record:
                    raise ValueError(f"conflicting duplicate segment: {segment_id}")
                by_segment[segment_id] = record
                rows += 1
        manifest_records.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": rows, "split": split}
        )
    inventory = [by_segment[key] for key in sorted(by_segment)]
    if not inventory:
        raise ValueError("no segments found")
    return inventory, manifest_records


def hash_snapshot_files(snapshot_directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(snapshot_directory.iterdir()):
        if path.is_file():
            records.append(
                {
                    "relative_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def write_index(path: Path, inventory: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row_index, record in enumerate(inventory):
            text_bytes = record["text"].encode("utf-8")
            output = {
                "row_index": row_index,
                "segment_id": record["segment_id"],
                "call_id": record["call_id"],
                "historical_split": record["historical_split"],
                "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
                "text_utf8_bytes": len(text_bytes),
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dirs = {CLS_STORE_ID: output_root / CLS_STORE_ID, MEAN_STORE_ID: output_root / MEAN_STORE_ID}
    existing = [str(path) for path in final_dirs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing stores: {existing}")

    inventory, manifest_records = load_inventory(args.manifest)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    snapshot_directory = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=args.local_files_only,
        )
    ).resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_directory, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(snapshot_directory, local_files_only=True).to(args.device)
    model.eval()
    if int(model.config.hidden_size) != DIMENSION:
        raise ValueError(f"unexpected hidden size: {model.config.hidden_size}")

    temporary_root = Path(tempfile.mkdtemp(prefix="krsbert-build-", dir=output_root))
    try:
        temporary_dirs = {}
        matrices = {}
        for store_id in final_dirs:
            directory = temporary_root / store_id
            directory.mkdir()
            temporary_dirs[store_id] = directory
            write_index(directory / "index.jsonl", inventory)
            matrices[store_id] = np.lib.format.open_memmap(
                directory / "vectors.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(inventory), DIMENSION),
            )

        unk_count = 0
        content_token_count = 0
        maximum_content_tokens = 0
        truncation_count = 0
        started = datetime.now(timezone.utc)
        with torch.inference_mode():
            for start in range(0, len(inventory), args.batch_size):
                end = min(start + args.batch_size, len(inventory))
                texts = [record["text"] for record in inventory[start:end]]
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                attention_mask = encoded["attention_mask"]
                special_mask = encoded.pop("special_tokens_mask")
                content_mask = attention_mask.bool() & ~special_mask.bool()
                content_lengths = content_mask.sum(dim=1)
                maximum_content_tokens = max(maximum_content_tokens, int(content_lengths.max()))
                content_token_count += int(content_mask.sum())
                unk_count += int(((encoded["input_ids"] == tokenizer.unk_token_id) & content_mask).sum())
                over_limit = attention_mask.sum(dim=1) > MAX_LENGTH
                truncation_count += int(over_limit.sum())
                if torch.any(over_limit):
                    offending = [inventory[start + i]["segment_id"] for i in over_limit.nonzero().flatten()]
                    raise ValueError(f"segments exceed max_length={MAX_LENGTH}: {offending[:10]}")
                model_inputs = {name: tensor.to(args.device) for name, tensor in encoded.items()}
                hidden = model(**model_inputs).last_hidden_state
                mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                cls_vectors = hidden[:, 0]
                mean_vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                matrices[CLS_STORE_ID][start:end] = cls_vectors.detach().cpu().numpy().astype(np.float32)
                matrices[MEAN_STORE_ID][start:end] = mean_vectors.detach().cpu().numpy().astype(np.float32)
                if start == 0 or end == len(inventory) or (start // args.batch_size) % 50 == 0:
                    print(json.dumps({"encoded": end, "total": len(inventory)}), flush=True)

        completed = datetime.now(timezone.utc)
        snapshot_files = hash_snapshot_files(snapshot_directory)
        common = {
            "schema_version": "paper2-v2-text-feature-store-1",
            "status": "sealed",
            "created_at_utc": completed.isoformat(),
            "extraction_started_at_utc": started.isoformat(),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "model_class": type(model).__name__,
            "max_length": MAX_LENGTH,
            "padding": "longest_in_batch",
            "truncation": False,
            "normalization": "none",
            "vector_l2_normalization": False,
            "dtype": "float32",
            "shape": [len(inventory), DIMENSION],
            "source_manifests": manifest_records,
            "inventory_order": "ascending segment_id Unicode code-point order",
            "token_audit": {
                "content_tokens": content_token_count,
                "unknown_content_tokens": unk_count,
                "unknown_content_rate": unk_count / max(1, content_token_count),
                "maximum_content_tokens": maximum_content_tokens,
                "segments_truncated": truncation_count,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
            "resolved_snapshot_files": snapshot_files,
        }
        pooling_specs = {
            CLS_STORE_ID: {
                "pooling": "raw_backbone_first_token_cls",
                "pooling_formula": "last_hidden_state[:,0,:]",
                "special_tokens_in_pool": "CLS only",
            },
            MEAN_STORE_ID: {
                "pooling": "canonical_attention_masked_mean",
                "pooling_formula": "sum(last_hidden_state * attention_mask) / sum(attention_mask)",
                "special_tokens_in_pool": "included when attention_mask=1 (CLS and SEP)",
            },
        }
        for store_id, directory in temporary_dirs.items():
            matrices[store_id].flush()
            del matrices[store_id]
            index_path = directory / "index.jsonl"
            vectors_path = directory / "vectors.npy"
            metadata = {
                **common,
                "feature_set_id": store_id,
                **pooling_specs[store_id],
                "index_sha256": sha256_file(index_path),
                "vectors_sha256": sha256_file(vectors_path),
                "vectors_bytes": vectors_path.stat().st_size,
            }
            (directory / "feature_store.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for store_id, final_dir in final_dirs.items():
            temporary_dirs[store_id].rename(final_dir)
        temporary_root.rmdir()
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(json.dumps({"status": "sealed", "stores": [str(path) for path in final_dirs.values()]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
