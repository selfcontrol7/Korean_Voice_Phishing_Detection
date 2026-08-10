"""Build the matched-channel test manifest for Track C Phase C.

Vishing (FSS, label=1) entries keep their original full-band feature
paths. Non-vishing (AllHub, label=0) entries are pointed at the
codec-augmented features under `features/{type}_codec_aug_nonvishing/`.

This is exactly what we evaluate the *frozen* (full-band-trained)
classifier against to test whether channel artifacts contributed to its
discrimination.

Output:
    data/test_segment_manifest_codec_aug.jsonl

Run from the Multimodal/ directory:
    python analysis/build_codec_aug_manifest.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NON_VISHING_PATH_REPLACEMENTS = {
    "mfcc_path":     ("features/mfcc/",     "features/mfcc_codec_aug_nonvishing/"),
    "egemaps_path":  ("features/egemaps/",  "features/egemaps_codec_aug_nonvishing/"),
    "wav2vec2_path": ("features/wav2vec2/", "features/wav2vec2_codec_aug_nonvishing/"),
}


def rewrite(in_path: Path, out_path: Path) -> tuple[int, int, int, int]:
    n_total = 0
    n_swapped = 0
    n_kept = 0
    n_missing = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            label = row["label"]
            if label == 0:
                # Swap to codec-augmented features
                for key, (old, new) in NON_VISHING_PATH_REPLACEMENTS.items():
                    if key in row and row[key].startswith(old):
                        row[key] = new + row[key][len(old):]
                n_swapped += 1
                # Existence check
                for key in NON_VISHING_PATH_REPLACEMENTS:
                    if key in row and not Path(row[key]).exists():
                        n_missing += 1
            else:
                # Vishing — keep original paths
                n_kept += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total += 1
    return n_total, n_swapped, n_kept, n_missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="test", help="Which split to rebuild (default: test)")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)

    in_path = Path(f"data/{args.split}_segment_manifest_merged.jsonl")
    out_path = Path(f"data/{args.split}_segment_manifest_codec_aug.jsonl")
    if not in_path.exists():
        print(f"⚠️  missing input manifest: {in_path}", file=sys.stderr)
        return 1

    n_total, n_swap, n_keep, n_miss = rewrite(in_path, out_path)
    print(f"  {args.split}: {n_total} segments  ({n_swap} non-vishing swapped, {n_keep} vishing kept)")
    print(f"  → {out_path}")
    if n_miss > 0:
        print(f"  ⚠️  {n_miss} codec-augmented feature paths reference files that don't exist yet")
        if not args.allow_missing:
            print("  Run features_extraction/build_save_codec_features.py first, or pass --allow-missing")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
