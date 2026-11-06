"""Add a `source` field (FSS or AllHub) to every manifest entry.

The corpus mixes vishing calls from FSS (phone-channel) with non-vishing
calls from AllHub (different recording conditions), so the data source is
a potential confounder. Several downstream scripts (Track A's call-level
aggregation, Track C's source-confounding probe) need the source as an
explicit field, but it is currently only encoded implicitly in file
paths and call_id prefixes.

Idempotent: re-running on already-tagged manifests is a no-op.
Atomic: writes to a sibling .tmp file and renames, so an interrupted run
never leaves a half-written manifest.

Run from the Multimodal/ directory:
    python preprocessing/add_source_to_manifests.py
    python preprocessing/add_source_to_manifests.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

MANIFESTS = [
    "data/master_manifest.jsonl",
    "data/train_segment_manifest_merged.jsonl",
    "data/val_segment_manifest_merged.jsonl",
    "data/test_segment_manifest_merged.jsonl",
]


def infer_source(entry: dict) -> str:
    call_id = entry.get("call_id", "")
    if call_id.startswith("vishing_"):
        return "FSS"
    audio_path = entry.get("audio_filepath", "")
    if "audio/vishing/" in audio_path:
        return "FSS"
    if "audio/non_vishing/" in audio_path:
        return "AllHub"
    return "AllHub"


def process_manifest(path: Path, dry_run: bool) -> dict:
    if not path.exists():
        print(f"  ⚠️  {path} not found — skipping")
        return {"skipped": True}

    with path.open(encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    counts = Counter()
    mismatches = []
    rewritten = 0

    for entry in entries:
        source = infer_source(entry)
        label = entry.get("label")
        counts[(source, label)] += 1

        if (label == 1 and source != "FSS") or (label == 0 and source != "AllHub"):
            mismatches.append(
                {"call_id": entry.get("call_id"), "label": label, "inferred_source": source}
            )

        if entry.get("source") != source:
            entry["source"] = source
            rewritten += 1

    if mismatches:
        print(f"  ❌ {len(mismatches)} label/source mismatches in {path}:")
        for m in mismatches[:5]:
            print(f"     {m}")
        if len(mismatches) > 5:
            print(f"     ... and {len(mismatches) - 5} more")
        raise SystemExit(
            "Aborting: corpus assumption (label=1 ⇔ FSS, label=0 ⇔ AllHub) violated."
        )

    print(f"  {path.name}: {len(entries)} entries")
    for (src, lbl), n in sorted(counts.items()):
        print(f"     source={src:6s} label={lbl}: {n}")

    if rewritten == 0:
        print(f"     ✓ already tagged — no write needed")
        return {"path": str(path), "entries": len(entries), "rewritten": 0}

    if dry_run:
        print(f"     [dry-run] would rewrite {rewritten} entries with new/changed source")
        return {"path": str(path), "entries": len(entries), "rewritten": rewritten, "dry_run": True}

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"     ✓ rewrote {rewritten} entries (atomic)")
    return {"path": str(path), "entries": len(entries), "rewritten": rewritten}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write")
    args = parser.parse_args()

    cwd = Path.cwd()
    if cwd.name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {cwd}", file=sys.stderr)

    print(f"Adding source field to {len(MANIFESTS)} manifest(s){' [DRY RUN]' if args.dry_run else ''}")
    print("=" * 80)

    for rel in MANIFESTS:
        process_manifest(Path(rel), dry_run=args.dry_run)
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
