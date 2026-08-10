"""Build 5 stratified group k-folds at the call level for Track D.

Each call is one atomic unit — all segments of a call must stay in the
same fold (no segment-level leakage). We split at the call level using
StratifiedGroupKFold, then filter the existing merged segment manifests
by call_id membership to produce per-fold {train, val, test} manifests.

Inputs:
    data/master_manifest.jsonl                       (1417 calls)
    data/{train,val,test}_segment_manifest_merged.jsonl  (39429 segments pooled)

Outputs:
    data/cv_5fold/
      ├── fold0/{train,val,test}_segment_manifest.jsonl
      ├── fold1/, fold2/, fold3/, fold4/
      └── splits.json                                  (provenance + counts)

The aggregator/orchestrator scripts only need fold0..4/{train,val,test}
manifests. The splits.json file documents fold composition for
reproducibility.

Run from the Multimodal/ directory:
    python analysis/build_kfold_manifests.py --n_splits 5 --random_state 42 --val_frac 0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


DATA_DIR = Path("data")
MASTER_MANIFEST = DATA_DIR / "master_manifest.jsonl"
SEGMENT_MANIFESTS = [
    DATA_DIR / "train_segment_manifest_merged.jsonl",
    DATA_DIR / "val_segment_manifest_merged.jsonl",
    DATA_DIR / "test_segment_manifest_merged.jsonl",
]
OUTPUT_ROOT = DATA_DIR / "cv_5fold"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _label_balance(call_ids: list[str], call_to_label: dict[str, int]) -> dict[str, int]:
    counts = Counter(call_to_label[c] for c in call_ids)
    return {"vishing": counts.get(1, 0), "non_vishing": counts.get(0, 0)}


def _val_holdout_from_train(
    train_call_ids: list[str],
    call_to_label: dict[str, int],
    val_frac: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    """Stratified group holdout for val from a list of train call_ids.

    Each call is its own group, so a stratified group split degenerates
    to a stratified split — but we still want a *grouped* split so val
    and train don't share any call.

    Implementation: StratifiedGroupKFold with n_splits = round(1/val_frac);
    take the first sub-fold's test as val.
    """
    n_inner = max(2, int(round(1.0 / val_frac)))
    labels = np.array([call_to_label[c] for c in train_call_ids])
    groups = np.array(train_call_ids)
    sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    train_inner_idx, val_idx = next(sgkf.split(np.zeros(len(train_call_ids)), labels, groups=groups))
    val_calls = [train_call_ids[i] for i in val_idx]
    train_only_calls = [train_call_ids[i] for i in train_inner_idx]
    return train_only_calls, val_calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--val_frac", type=float, default=0.10,
        help="Fraction of the per-fold train set held out as validation (default 0.10)",
    )
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)
        return 1

    # 1. Load master manifest (call-level)
    if not MASTER_MANIFEST.exists():
        print(f"❌ master manifest not found: {MASTER_MANIFEST}", file=sys.stderr)
        return 1
    calls = _load_jsonl(MASTER_MANIFEST)
    call_to_label: dict[str, int] = {c["call_id"]: int(c["label"]) for c in calls}
    all_call_ids = [c["call_id"] for c in calls]
    print(f"Loaded {len(calls)} calls from {MASTER_MANIFEST}")
    print(f"  vishing: {sum(1 for c in calls if c['label']==1)} / "
          f"non_vishing: {sum(1 for c in calls if c['label']==0)}")

    # 2. Pool the segment manifests into one big list, and build a
    #    call_id → list[segment] index for fast filtering.
    segments: list[dict] = []
    for path in SEGMENT_MANIFESTS:
        if not path.exists():
            print(f"❌ segment manifest not found: {path}", file=sys.stderr)
            return 1
        segments.extend(_load_jsonl(path))
    total_segments = len(segments)
    print(f"Pooled {total_segments} segments from {len(SEGMENT_MANIFESTS)} merged manifests")

    by_call: dict[str, list[dict]] = {}
    for seg in segments:
        by_call.setdefault(seg["call_id"], []).append(seg)

    # Sanity: every master call has segments, no orphan calls
    missing_segments = [c for c in all_call_ids if c not in by_call]
    if missing_segments:
        print(f"⚠️  {len(missing_segments)} calls have no segments (first 5): {missing_segments[:5]}",
              file=sys.stderr)
    orphan_segs = [seg["call_id"] for seg in segments if seg["call_id"] not in call_to_label]
    if orphan_segs:
        print(f"⚠️  {len(set(orphan_segs))} segment-level call_ids not in master "
              f"(first 5): {sorted(set(orphan_segs))[:5]}", file=sys.stderr)

    # 3. Outer 5-fold split at the call level (stratified-grouped)
    labels = np.array([call_to_label[c] for c in all_call_ids])
    groups = np.array(all_call_ids)
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    fold_call_ids: dict[int, dict[str, list[str]]] = {}
    fold_segment_counts: dict[int, dict[str, int]] = {}

    for k, (trainval_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(all_call_ids)), labels, groups=groups)):
        test_calls = [all_call_ids[i] for i in test_idx]
        trainval_calls = [all_call_ids[i] for i in trainval_idx]

        # 4. Inner stratified-grouped holdout for val (~val_frac of trainval)
        train_calls, val_calls = _val_holdout_from_train(
            trainval_calls, call_to_label,
            val_frac=args.val_frac,
            random_state=args.random_state + k,
        )

        # Sanity asserts
        assert set(train_calls).isdisjoint(val_calls), f"fold {k}: train ∩ val nonempty"
        assert set(train_calls).isdisjoint(test_calls), f"fold {k}: train ∩ test nonempty"
        assert set(val_calls).isdisjoint(test_calls), f"fold {k}: val ∩ test nonempty"
        assert len(train_calls) + len(val_calls) + len(test_calls) == len(all_call_ids), \
            f"fold {k}: split counts don't sum to {len(all_call_ids)}"

        # Filter the pooled segment list by call_id membership
        train_segs = [s for c in train_calls for s in by_call.get(c, [])]
        val_segs = [s for c in val_calls for s in by_call.get(c, [])]
        test_segs = [s for c in test_calls for s in by_call.get(c, [])]

        seg_total = len(train_segs) + len(val_segs) + len(test_segs)
        assert seg_total == total_segments, f"fold {k}: segment total mismatch ({seg_total} vs {total_segments})"

        # Write per-fold manifests
        fold_dir = args.output_dir / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_segs in (("train", train_segs), ("val", val_segs), ("test", test_segs)):
            out_path = fold_dir / f"{split_name}_segment_manifest.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for seg in split_segs:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")

        # Diagnostics
        bal_test = _label_balance(test_calls, call_to_label)
        bal_val = _label_balance(val_calls, call_to_label)
        bal_train = _label_balance(train_calls, call_to_label)
        print(
            f"\nfold{k}: "
            f"train={len(train_calls)} calls/{len(train_segs)} segs (V={bal_train['vishing']}, NV={bal_train['non_vishing']})  "
            f"val={len(val_calls)} calls/{len(val_segs)} segs (V={bal_val['vishing']}, NV={bal_val['non_vishing']})  "
            f"test={len(test_calls)} calls/{len(test_segs)} segs (V={bal_test['vishing']}, NV={bal_test['non_vishing']})"
        )

        # Hard-assert label balance in test
        v_frac_test = bal_test["vishing"] / max(1, len(test_calls))
        assert 0.45 <= v_frac_test <= 0.55, f"fold {k}: test vishing fraction {v_frac_test:.3f} out of [0.45, 0.55]"

        fold_call_ids[k] = {"train": train_calls, "val": val_calls, "test": test_calls}
        fold_segment_counts[k] = {
            "train": len(train_segs),
            "val": len(val_segs),
            "test": len(test_segs),
        }

    # 5. Provenance
    provenance = {
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "val_frac": args.val_frac,
        "n_calls_total": len(all_call_ids),
        "n_segments_total": total_segments,
        "fold_segment_counts": fold_segment_counts,
        "fold_call_ids": fold_call_ids,
        "master_manifest": str(MASTER_MANIFEST),
        "segment_manifests_pooled": [str(p) for p in SEGMENT_MANIFESTS],
    }
    splits_path = args.output_dir / "splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Wrote {splits_path}")
    print(f"✅ {args.n_splits} folds under {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
