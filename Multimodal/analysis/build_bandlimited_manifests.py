"""Clone the segment manifests, swap feature paths to the band-limited directories.

Track C Phase B preparation. After Track C.2 re-extracts features at
band-limited bandwidth into `features/{type}_telephony_bandlimited/`,
we need new manifest files that point the existing dataloader at those
features. The new manifests preserve every metadata field except for
the three feature paths.

Output:
    data/{train,val,test}_segment_manifest_bandlimited.jsonl

Idempotent: re-running just regenerates the files.

Run from the Multimodal/ directory:
    python analysis/build_bandlimited_manifests.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PATH_REPLACEMENTS = {
    "mfcc_path":     ("features/mfcc/",     "features/mfcc_telephony_bandlimited/"),
    "egemaps_path":  ("features/egemaps/",  "features/egemaps_telephony_bandlimited/"),
    "wav2vec2_path": ("features/wav2vec2/", "features/wav2vec2_telephony_bandlimited/"),
}


def rewrite_one(in_path: Path, out_path: Path) -> tuple[int, int]:
    n_total = 0
    n_missing = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            for key, (old, new) in PATH_REPLACEMENTS.items():
                if key in row and row[key].startswith(old):
                    row[key] = new + row[key][len(old):]
            # Sanity: confirm every band-limited feature file exists
            for key in PATH_REPLACEMENTS:
                if key in row and not Path(row[key]).exists():
                    n_missing += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total += 1
    return n_total, n_missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Skip the existence check for band-limited feature files")
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    overall_missing = 0
    for split in splits:
        in_path = Path(f"data/{split}_segment_manifest_merged.jsonl")
        out_path = Path(f"data/{split}_segment_manifest_bandlimited.jsonl")
        if not in_path.exists():
            print(f"⚠️  missing input manifest: {in_path}", file=sys.stderr)
            continue
        n_total, n_missing = rewrite_one(in_path, out_path)
        overall_missing += n_missing
        status = "✓" if n_missing == 0 else f"❌ {n_missing} feature paths missing on disk"
        print(f"  {split}: {n_total} segments → {out_path}  {status}")

    if overall_missing > 0 and not args.allow_missing:
        print(
            f"\n⚠️  {overall_missing} feature paths reference files that don't exist yet. "
            f"Run features_extraction/build_save_bandlimited_features.py first, "
            f"or pass --allow-missing to proceed anyway.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
