"""Track D — 5-fold cross-validation orchestrator.

End-to-end pipeline:
  1. build_folds   — call analysis/build_kfold_manifests.py once
  2. train         — for each (fold, feature_type), retrain the audio
                     classifier on fold's train set with early-stop on
                     fold's val set (modeling/train_audio_baseline.py)
  3. infer         — segment-level predictions for fold's val + test
                     (modeling/run_segment_inference.py)
  4. call_level    — sweep + test eval per fold
                     (modeling/evaluate_call_level.py)
  5. aggregate     — combine the 5 per-fold tables into mean ± std and
                     a per-fold appendix table (pure-Python)

Idempotency: each step skips if outputs exist; `--force` re-runs.

Run from the Multimodal/ directory:
    # Smoke test (single fold, single feature type, ~3 min)
    python analysis/cv_5fold_orchestrator.py --mode all --feature_types egemaps --folds 0

    # Full sweep (all 4 feature types, all 5 folds, ~3-4 hours)
    python analysis/cv_5fold_orchestrator.py --mode all

    # Re-run aggregation only
    python analysis/cv_5fold_orchestrator.py --mode aggregate --force
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACK_D_ROOT = Path("modeling/logs/track_d")
DATA_FOLD_ROOT = Path("data/cv_5fold")
MODEL_FOLD_ROOT = Path("modeling/models/cv_5fold")

FEATURE_TYPES_DEFAULT = ["egemaps", "mfcc", "wav2vec2", "all"]
N_SPLITS_DEFAULT = 5
STEPS = ["build_folds", "train", "infer", "call_level", "aggregate", "all"]


# ---------------------------------------------------------------------------
# Subprocess helper (unbuffered child Python; same pattern as Phase B/C)
# ---------------------------------------------------------------------------
def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if cmd and ("python" in Path(cmd[0]).name) and "-u" not in cmd[:3]:
        cmd = [cmd[0], "-u"] + list(cmd[1:])
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print(f"  $ {' '.join(cmd)}")
    print(f"    log → {log_path}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"$ {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
        proc.wait()
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s, exit={proc.returncode}")
    return proc.returncode


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _fold_dir(k: int) -> Path:
    return DATA_FOLD_ROOT / f"fold{k}"


def _fold_manifest(k: int, split: str) -> Path:
    return _fold_dir(k) / f"{split}_segment_manifest.jsonl"


def _fold_log_dir(k: int) -> Path:
    return TRACK_D_ROOT / f"fold{k}" / "logs"


def _fold_pred_dir(k: int) -> Path:
    return TRACK_D_ROOT / f"fold{k}" / "segment_predictions"


def _fold_out_dir(k: int) -> Path:
    return TRACK_D_ROOT / f"fold{k}"


def _fold_checkpoint(k: int, ft: str) -> Path:
    return MODEL_FOLD_ROOT / f"fold{k}" / f"best_audio_model_{ft}.pth"


def _train_artifact_path(ft: str, k: int) -> Path:
    """Where train_audio_baseline.py writes the checkpoint, before relocation."""
    return Path(f"modeling/models/best_audio_model_{ft}_fold{k}.pth")


# ---------------------------------------------------------------------------
# Step 1 — Build folds
# ---------------------------------------------------------------------------
def step_build_folds(force: bool, n_splits: int, random_state: int, val_frac: float) -> int:
    print(f"\n[Step 1] Build {n_splits} fold manifests")
    splits_path = DATA_FOLD_ROOT / "splits.json"
    if splits_path.exists() and not force:
        print(f"  ✓ {splits_path} exists; skipping (use --force to rebuild)")
        return 0
    cmd = [
        sys.executable, "analysis/build_kfold_manifests.py",
        "--n_splits", str(n_splits),
        "--random_state", str(random_state),
        "--val_frac", str(val_frac),
    ]
    return _run(cmd, TRACK_D_ROOT / "logs" / "01_build_folds.log")


# ---------------------------------------------------------------------------
# Step 2 — Train one model per (fold, feature_type)
# ---------------------------------------------------------------------------
def step_train(folds: list[int], feature_types: list[str], epochs: int, batch_size: int, lr: float, force: bool) -> int:
    print(f"\n[Step 2] Train {len(folds)} folds × {len(feature_types)} feature types = {len(folds)*len(feature_types)} retrains")
    failed: list[tuple[int, str]] = []
    for k in folds:
        for ft in feature_types:
            ckpt_dest = _fold_checkpoint(k, ft)
            if ckpt_dest.exists() and not force:
                print(f"  ✓ fold{k}/{ft}: checkpoint exists; skipping")
                continue
            ckpt_src = _train_artifact_path(ft, k)
            cmd = [
                sys.executable, "modeling/train_audio_baseline.py",
                "--feature_type", ft,
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--train_manifest", str(_fold_manifest(k, "train")),
                "--val_manifest", str(_fold_manifest(k, "val")),
                "--test_manifest", str(_fold_manifest(k, "test")),
                "--model_suffix", f"_fold{k}",
            ]
            rc = _run(cmd, _fold_log_dir(k) / f"02_train_{ft}.log")
            if rc != 0 or not ckpt_src.exists():
                print(f"  ✗ fold{k}/{ft}: training failed (exit={rc}, src={ckpt_src.exists()})")
                failed.append((k, ft))
                continue
            ckpt_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ckpt_src), str(ckpt_dest))
            print(f"  → moved {ckpt_src.name} to {ckpt_dest}")
    if failed:
        print(f"\n  ❌ Train failed for: {failed}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Step 3 — Per-fold segment inference (val + test)
# ---------------------------------------------------------------------------
def step_infer(folds: list[int], feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 3] Segment-level inference for {len(folds)} folds × {len(feature_types)} features")
    failed: list[tuple[int, str, str]] = []
    for k in folds:
        ckpt_root = MODEL_FOLD_ROOT / f"fold{k}"
        pred_root = _fold_pred_dir(k)
        pred_root.mkdir(parents=True, exist_ok=True)
        for ft in feature_types:
            ckpt = _fold_checkpoint(k, ft)
            if not ckpt.exists():
                print(f"  ✗ fold{k}/{ft}: missing checkpoint {ckpt}")
                failed.append((k, ft, "missing_checkpoint"))
                continue
            for split in ("val", "test"):
                out_path = pred_root / f"{split}_{ft}.jsonl"
                if out_path.exists() and not force:
                    print(f"  ✓ fold{k}/{ft}/{split}: predictions exist; skipping")
                    continue
                cmd = [
                    sys.executable, "modeling/run_segment_inference.py",
                    "--checkpoint", str(ckpt),
                    "--manifest", str(_fold_manifest(k, split)),
                    "--feature_type", ft,
                    "--output", str(out_path),
                ]
                rc = _run(cmd, _fold_log_dir(k) / f"03_infer_{split}_{ft}.log")
                if rc != 0 or not out_path.exists():
                    failed.append((k, ft, split))
    if failed:
        print(f"\n  ❌ Inference failed for: {failed}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Step 4 — Per-fold call-level evaluation
# ---------------------------------------------------------------------------
def step_call_level(folds: list[int], feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 4] Per-fold call-level evaluation")
    for k in folds:
        out_dir = _fold_out_dir(k)
        summary_md = out_dir / "tables" / "call_level_summary.md"
        if summary_md.exists() and not force:
            print(f"  ✓ fold{k}: call-level summary exists; skipping")
            continue
        cmd = [
            sys.executable, "modeling/evaluate_call_level.py",
            "--feature_types", ",".join(feature_types),
            "--pred_dir", str(_fold_pred_dir(k)),
            "--out_dir", str(out_dir),
        ]
        rc = _run(cmd, _fold_log_dir(k) / "04_call_level.log")
        if rc != 0:
            print(f"  ✗ fold{k}: call-level eval failed (exit={rc})")
            return 1
    return 0


# ---------------------------------------------------------------------------
# Step 5 — Aggregate per-fold results into mean ± std + per-fold appendix
# ---------------------------------------------------------------------------
AGG_NUMERIC_COLS = ["f1", "precision", "recall", "fpr", "mean_alert_latency", "median_alert_latency"]


def _read_per_fold(folds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for k in folds:
        path = _fold_out_dir(k) / "tables" / "call_level_summary.csv"
        if not path.exists():
            print(f"  ⚠️  missing {path}; skipping fold{k} in aggregate")
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                row["fold"] = k
                rows.append(row)
    return rows


def _to_float(s: str | None) -> float:
    if s in (None, "", "None"):
        return float("nan")
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def step_aggregate(folds: list[int], feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 5] Aggregate {len(folds)}-fold results")
    out_dir = TRACK_D_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "cv_summary.csv"
    summary_md = out_dir / "cv_summary.md"
    per_fold_csv = out_dir / "cv_per_fold.csv"
    per_fold_md = out_dir / "cv_per_fold.md"
    if summary_md.exists() and per_fold_md.exists() and not force:
        print(f"  ✓ {summary_md} exists; skipping")
        return 0

    rows = _read_per_fold(folds)
    if not rows:
        print("  ✗ No per-fold tables found; cannot aggregate")
        return 1

    # ---- Per-fold long table ----
    per_fold_md_cols = [
        ("fold", "Fold"),
        ("feature_type", "Feature"),
        ("method", "Method"),
        ("alpha", "α"),
        ("tau", "τ"),
        ("f1", "F1"),
        ("precision", "P"),
        ("recall", "R"),
        ("fpr", "FPR"),
        ("mean_alert_latency", "Mean alert (s)"),
        ("median_alert_latency", "Median alert (s)"),
    ]
    fields = [c[0] for c in per_fold_md_cols]
    with per_fold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    with per_fold_md.open("w") as f:
        f.write("# Track D — Per-fold Call-Level Results (Appendix)\n\n")
        f.write("Per-fold breakdown across 5 stratified-grouped folds for the "
                "audio classifiers and the two paper-faithful aggregation rules "
                "(EMA, running-max). Aggregated mean ± std is in `cv_summary.md`.\n\n")
        f.write("| " + " | ".join(h for _, h in per_fold_md_cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(per_fold_md_cols)) + "|\n")
        for r in rows:
            cells: list[str] = []
            for col, _ in per_fold_md_cols:
                v = r.get(col, "")
                if col in AGG_NUMERIC_COLS:
                    val = _to_float(v)
                    if v in ("", "None", None):
                        cells.append("—")
                    elif col in ("mean_alert_latency", "median_alert_latency"):
                        cells.append(f"{val:.2f}")
                    else:
                        cells.append(f"{val:.4f}")
                else:
                    cells.append("—" if v in ("", "None", None) else str(v))
            f.write("| " + " | ".join(cells) + " |\n")

    # ---- Aggregate (mean ± std) per (feature, method) ----
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["feature_type"], r["method"])
        grouped.setdefault(key, []).append(r)

    agg_rows: list[dict] = []
    for (ft, method), group in grouped.items():
        agg_row = {"feature_type": ft, "method": method, "n_folds": len(group)}
        for col in AGG_NUMERIC_COLS:
            vals = [_to_float(r.get(col, "")) for r in group]
            vals = [v for v in vals if not (v != v)]  # drop NaN
            if not vals:
                agg_row[f"{col}_mean"] = float("nan")
                agg_row[f"{col}_std"] = float("nan")
            else:
                agg_row[f"{col}_mean"] = statistics.mean(vals)
                agg_row[f"{col}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(agg_row)

    # Stable order: by feature_types arg, then a fixed method order
    method_order = ["ema", "running_max", "any_segment_baseline (τ=0.5)", "majority_vote (τ=0.5)"]
    agg_rows.sort(
        key=lambda r: (
            feature_types.index(r["feature_type"]) if r["feature_type"] in feature_types else 999,
            method_order.index(r["method"]) if r["method"] in method_order else 999,
        )
    )

    fieldnames = (
        ["feature_type", "method", "n_folds"] +
        [f"{c}_{stat}" for c in AGG_NUMERIC_COLS for stat in ("mean", "std")]
    )
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in agg_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    summary_md_cols = [
        ("feature_type", "Feature"),
        ("method", "Method"),
        ("f1", "F1"),
        ("precision", "P"),
        ("recall", "R"),
        ("fpr", "FPR"),
        ("mean_alert_latency", "Mean alert (s)"),
    ]
    with summary_md.open("w") as f:
        f.write("# Track D — 5-Fold Cross-Validation Summary\n\n")
        f.write(f"Aggregated mean ± std across {len(folds)} stratified-grouped folds. "
                f"Per-fold breakdown in `cv_per_fold.md`.\n\n")
        f.write("**Protocol.** sklearn `StratifiedGroupKFold(n_splits=5, shuffle=True, "
                "random_state=42)` on the 1417-call corpus, with each fold's call set further "
                "split into ~90% train / ~10% val by `StratifiedGroupKFold(n_splits=10)`. "
                "All segments of a call are kept together to prevent leakage. "
                "Each fold trains the audio classifier from scratch with the same "
                "hyperparameters as the original Track A baseline. Hyperparameters for the "
                "EMA/Running-max aggregation are independently re-tuned on each fold's val set.\n\n")
        f.write("| " + " | ".join(h for _, h in summary_md_cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(summary_md_cols)) + "|\n")
        for r in agg_rows:
            cells: list[str] = []
            for col, _ in summary_md_cols:
                if col in ("feature_type", "method"):
                    cells.append(str(r[col]))
                else:
                    mean = r.get(f"{col}_mean", float("nan"))
                    std = r.get(f"{col}_std", float("nan"))
                    if mean != mean:  # NaN
                        cells.append("—")
                    elif col in ("mean_alert_latency",):
                        cells.append(f"{mean:.2f} ± {std:.2f}")
                    else:
                        cells.append(f"{mean:.4f} ± {std:.4f}")
            f.write("| " + " | ".join(cells) + " |\n")

    print(f"  ✓ wrote {summary_csv}")
    print(f"  ✓ wrote {summary_md}")
    print(f"  ✓ wrote {per_fold_csv}")
    print(f"  ✓ wrote {per_fold_md}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", default="all", choices=STEPS)
    parser.add_argument("--feature_types", default=",".join(FEATURE_TYPES_DEFAULT))
    parser.add_argument("--folds", default=",".join(str(i) for i in range(N_SPLITS_DEFAULT)),
                        help="Comma-separated fold indices to process (default 0,1,2,3,4)")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS_DEFAULT)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)
        return 1

    feature_types = [ft.strip() for ft in args.feature_types.split(",") if ft.strip()]
    folds = [int(s.strip()) for s in args.folds.split(",") if s.strip() != ""]
    for ft in feature_types:
        if ft not in FEATURE_TYPES_DEFAULT:
            print(f"⚠️  Unknown feature type: {ft}", file=sys.stderr)
            return 1

    (TRACK_D_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Track D — 5-fold CV orchestrator")
    print(f"Folds: {folds}")
    print(f"Feature types: {feature_types}")
    print(f"Mode: {args.mode}")
    print(f"Force: {args.force}")
    print("=" * 80)

    steps = (["build_folds", "train", "infer", "call_level", "aggregate"]
             if args.mode == "all" else [args.mode])

    for step in steps:
        if step == "build_folds":
            rc = step_build_folds(args.force, args.n_splits, args.random_state, args.val_frac)
        elif step == "train":
            rc = step_train(folds, feature_types, args.epochs, args.batch_size, args.lr, args.force)
        elif step == "infer":
            rc = step_infer(folds, feature_types, args.force)
        elif step == "call_level":
            rc = step_call_level(folds, feature_types, args.force)
        elif step == "aggregate":
            rc = step_aggregate(folds, feature_types, args.force)
        else:
            print(f"Unknown step: {step}", file=sys.stderr)
            return 1
        if rc != 0:
            print(f"\n❌ Step '{step}' failed (exit {rc})", file=sys.stderr)
            return rc

    print("\n" + "=" * 80)
    print("✅ Track D step(s) complete.")
    print(f"   • Folds:     {DATA_FOLD_ROOT}/")
    print(f"   • Models:    {MODEL_FOLD_ROOT}/")
    print(f"   • Per-fold:  {TRACK_D_ROOT}/fold{{0..{args.n_splits-1}}}/")
    print(f"   • Aggregate: {TRACK_D_ROOT}/tables/cv_{{summary,per_fold}}.{{csv,md}}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
