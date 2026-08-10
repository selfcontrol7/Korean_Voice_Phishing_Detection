"""Track C, Phase D — per-source error analysis.

Slice Track A's call-level test results by (source × predicted_label).
Limited by the fact that the corpus has no FSS-non-vishing or
AllHub-vishing samples, so we cannot compute true per-source FPR
within FSS or per-source recall within AllHub. We *can* report:

  - recall on FSS vishing calls (TPs among the vishing class)
  - true-negative rate on AllHub non-vishing calls (TNs among non-vishing)
  - per-aggregation-method (EMA / running-max / segment-baseline / majority)
    × per-feature-type breakdown

Plus the structural caveat in plain text. The honest version of this
limitation is what reviewers actually want.

Run from the Multimodal/ directory:
    python analysis/source_confounding_phase_d.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

TRACK_A_DIR = Path("modeling/logs/track_a/call_level")
OUT_DIR = Path("modeling/logs/track_c/phase_d")

FEATURE_TYPES = ["egemaps", "mfcc", "wav2vec2", "all"]
METHOD_KEYS = [
    ("ema_calls",        "ema",                    "ema_best_config"),
    ("running_max_calls", "running_max",            "running_max_best_config"),
]


def per_source_breakdown(calls: list[dict], method_label: str) -> list[dict]:
    """For a single (feature_type, method) combination, count by source × decision."""
    counts: dict[tuple[str, int, int], int] = {}
    for c in calls:
        src = c["source"]
        truth = int(c["label"])
        pred = int(c["final_decision"])
        key = (src, truth, pred)
        counts[key] = counts.get(key, 0) + 1

    rows: list[dict] = []
    # FSS slice: every label is 1 (no AllHub-vishing exists)
    fss_total = sum(v for (s, t, _), v in counts.items() if s == "FSS" and t == 1)
    fss_pred1 = sum(v for (s, t, p), v in counts.items() if s == "FSS" and t == 1 and p == 1)
    fss_recall = fss_pred1 / fss_total if fss_total else 0.0
    rows.append({
        "method": method_label,
        "source": "FSS",
        "label_in_source": 1,
        "n": fss_total,
        "n_predicted_vishing": fss_pred1,
        "metric_name": "recall (TPR)",
        "metric_value": round(fss_recall, 4),
        "missed": fss_total - fss_pred1,
    })

    # AllHub slice: every label is 0 (no FSS-non-vishing exists)
    all_total = sum(v for (s, t, _), v in counts.items() if s == "AllHub" and t == 0)
    all_pred0 = sum(v for (s, t, p), v in counts.items() if s == "AllHub" and t == 0 and p == 0)
    all_tnr = all_pred0 / all_total if all_total else 0.0
    rows.append({
        "method": method_label,
        "source": "AllHub",
        "label_in_source": 0,
        "n": all_total,
        "n_predicted_vishing": all_total - all_pred0,
        "metric_name": "specificity (TNR) = 1 − FPR",
        "metric_value": round(all_tnr, 4),
        "missed": all_total - all_pred0,  # i.e., false positives
    })
    return rows


def write_outputs(table: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "per_source_confusion.csv"
    md_path = OUT_DIR / "per_source_confusion.md"

    fieldnames = list(table[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in table:
            w.writerow(r)

    cols = ["feature_type", "method", "source", "label_in_source", "n",
            "n_predicted_vishing", "metric_name", "metric_value", "missed"]
    headers = {
        "feature_type": "Feature",
        "method": "Method",
        "source": "Source",
        "label_in_source": "Label (within source)",
        "n": "n calls",
        "n_predicted_vishing": "predicted vishing",
        "metric_name": "Metric",
        "metric_value": "Value",
        "missed": "misclassified",
    }
    with md_path.open("w") as f:
        f.write("# Per-Source Confusion (Test Set)\n\n")
        f.write(
            "Sliced from Track A's call-level test results. The corpus has no "
            "FSS-non-vishing or AllHub-vishing samples, so per-source FPR within "
            "FSS and per-source recall within AllHub are mathematically undefined "
            "and are not reported here.\n\n"
        )
        f.write("| " + " | ".join(headers[c] for c in cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in table:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
        f.write(
            "\n**Note on what we can and cannot compute:** With the FSS source "
            "containing only vishing calls (706) and the AllHub source containing "
            "only non-vishing calls (711), the source × label table has empty "
            "off-diagonal cells. We can report recall on the FSS slice and "
            "specificity on the AllHub slice — both of which equal the corpus-wide "
            "metrics modulo rounding — but we cannot test whether the classifier "
            "would identify a vishing call from AllHub or a non-vishing call from "
            "FSS, because no such samples exist. This is the central reason "
            "Phase B (band-ablation) and Phase C (codec augmentation) are needed.\n"
        )

    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feature_types", default=",".join(FEATURE_TYPES))
    args = parser.parse_args()

    feature_types = [s.strip() for s in args.feature_types.split(",") if s.strip()]
    table: list[dict] = []
    for ft in feature_types:
        path = TRACK_A_DIR / f"test_results_{ft}.json"
        if not path.exists():
            print(f"⚠️  missing Track A results: {path}", file=sys.stderr)
            continue
        with path.open() as f:
            payload = json.load(f)
        for calls_key, method_label, cfg_key in METHOD_KEYS:
            calls = payload.get(calls_key, [])
            cfg = payload.get(cfg_key, {})
            if not calls:
                continue
            method_str = method_label
            if method_label == "ema":
                method_str = f"ema (α={cfg.get('alpha')}, τ={cfg.get('tau')})"
            else:
                method_str = f"running_max (τ={cfg.get('tau')})"
            rows = per_source_breakdown(calls, method_str)
            for r in rows:
                r["feature_type"] = ft
                table.append(r)

    if not table:
        print("No results to summarize. Did Track A run?", file=sys.stderr)
        return 1

    write_outputs(table)
    print("\n=== Per-Source Confusion (test set) ===")
    for r in table:
        print(
            f"  {r['feature_type']:8s}  {r['method']:30s}  {r['source']:6s}  "
            f"n={r['n']:3d}  {r['metric_name']:24s} {r['metric_value']:.4f}  "
            f"misclassified={r['missed']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
