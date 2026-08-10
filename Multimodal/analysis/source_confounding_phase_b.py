"""Track C Phase B orchestrator — telephone-band ablation retrain + evaluate.

This script drives the end-to-end Phase B pipeline as a sequence of
idempotent steps. Every step is invoked as a subprocess so the pipeline
matches what a reproducer would run from a clean checkout, with logs
captured to `modeling/logs/track_c/phase_b/logs/`.

Steps (run in order; each is independently re-runnable):
  1. build_manifests   — clone train/val/test manifests with band-limited
                         feature paths (analysis/build_bandlimited_manifests.py).
  2. retrain           — for each feature type, retrain the audio classifier
                         on band-limited features by invoking
                         modeling/train_audio_baseline.py with the
                         --train_manifest / --val_manifest / --test_manifest
                         and --model_suffix _bandlimited flags.
  3. infer             — for each feature type, run modeling/run_segment_inference.py
                         on val + test using the retrained checkpoint and
                         band-limited manifests, writing predictions to
                         modeling/logs/track_c/phase_b/segment_predictions/.
  4. call_level        — invoke modeling/evaluate_call_level.py with
                         --pred_dir / --out_dir pointing at the Phase B
                         outputs to produce the val sweep, test results,
                         summary table, and CDF figures.
  5. compare_table     — load Track A's original call-level summary and the
                         Phase B band-limited summary; emit a side-by-side
                         CSV and Markdown table quantifying the F1 drop
                         attributable to wide-band channel cues.

Idempotency: each step checks whether its outputs already exist on disk
and skips by default. Use `--force` to re-run a step that has already
produced outputs. Use `--feature_types egemaps` for a smoke test before
running the full sweep.

Run from the Multimodal/ directory:
    # Smoke test on eGeMAPS only:
    python analysis/source_confounding_phase_b.py --feature_types egemaps

    # Full pipeline (all four feature types):
    python analysis/source_confounding_phase_b.py

    # Re-run a specific step with force:
    python analysis/source_confounding_phase_b.py --mode call_level --force

    # Run only one step:
    python analysis/source_confounding_phase_b.py --mode retrain --feature_types egemaps,mfcc
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACK_C_ROOT = Path("modeling/logs/track_c/phase_b")
LOG_DIR = TRACK_C_ROOT / "logs"
PRED_DIR = TRACK_C_ROOT / "segment_predictions"
TABLE_DIR = TRACK_C_ROOT / "tables"

ORIGINAL_TABLE = Path("modeling/logs/track_a/tables/call_level_summary.csv")

DATA_DIR = Path("data")
MODEL_DIR = Path("modeling/models")

FEATURE_TYPES_DEFAULT = ["egemaps", "mfcc", "wav2vec2", "all"]
STEPS = ["build_manifests", "retrain", "infer", "call_level", "compare_table", "all"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess, tee output to log_path, return exit code.

    Forces unbuffered stdout from child Python processes so the tee log
    flushes line-by-line in real time.
    """
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


def _bandlimited_manifest(split: str) -> Path:
    return DATA_DIR / f"{split}_segment_manifest_bandlimited.jsonl"


def _checkpoint_path(feature_type: str) -> Path:
    # train_audio_baseline.py constructs:
    #   modeling/models/best_audio_model_{feature_type}{model_suffix}.pth
    return MODEL_DIR / f"best_audio_model_{feature_type}_bandlimited.pth"


def _segment_pred_path(split: str, feature_type: str) -> Path:
    return PRED_DIR / f"{split}_{feature_type}.jsonl"


# ---------------------------------------------------------------------------
# Step 1 — Build band-limited manifests
# ---------------------------------------------------------------------------
def step_build_manifests(force: bool) -> int:
    print("\n[Step 1] Build band-limited manifests")
    outputs = [_bandlimited_manifest(s) for s in ("train", "val", "test")]
    if all(p.exists() for p in outputs) and not force:
        print(f"  ✓ all 3 manifests exist; skipping (use --force to rebuild)")
        return 0
    cmd = [sys.executable, "analysis/build_bandlimited_manifests.py"]
    return _run(cmd, LOG_DIR / "01_build_manifests.log")


# ---------------------------------------------------------------------------
# Step 2 — Retrain audio classifiers on band-limited features
# ---------------------------------------------------------------------------
def step_retrain(feature_types: list[str], epochs: int, batch_size: int, lr: float, force: bool) -> int:
    print(f"\n[Step 2] Retrain classifiers on band-limited features ({len(feature_types)} feature types)")
    failed: list[str] = []
    for ft in feature_types:
        ckpt = _checkpoint_path(ft)
        if ckpt.exists() and not force:
            print(f"  ✓ {ft}: checkpoint exists at {ckpt}; skipping")
            continue
        cmd = [
            sys.executable, "modeling/train_audio_baseline.py",
            "--feature_type", ft,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--train_manifest", str(_bandlimited_manifest("train")),
            "--val_manifest", str(_bandlimited_manifest("val")),
            "--test_manifest", str(_bandlimited_manifest("test")),
            "--model_suffix", "_bandlimited",
        ]
        rc = _run(cmd, LOG_DIR / f"02_retrain_{ft}.log")
        if rc != 0 or not ckpt.exists():
            print(f"  ✗ {ft}: retrain failed (exit={rc}, checkpoint exists={ckpt.exists()})")
            failed.append(ft)
    if failed:
        print(f"\n  ❌ Retrain failed for: {failed}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Step 3 — Run segment inference on val + test
# ---------------------------------------------------------------------------
def step_infer(feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 3] Run segment-level inference (val + test) on band-limited data")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []
    for ft in feature_types:
        ckpt = _checkpoint_path(ft)
        if not ckpt.exists():
            print(f"  ✗ {ft}: missing checkpoint {ckpt}; run step 'retrain' first")
            failed.append((ft, "missing_checkpoint"))
            continue
        for split in ("val", "test"):
            out_path = _segment_pred_path(split, ft)
            if out_path.exists() and not force:
                print(f"  ✓ {ft}/{split}: predictions exist at {out_path}; skipping")
                continue
            cmd = [
                sys.executable, "modeling/run_segment_inference.py",
                "--checkpoint", str(ckpt),
                "--manifest", str(_bandlimited_manifest(split)),
                "--feature_type", ft,
                "--output", str(out_path),
            ]
            rc = _run(cmd, LOG_DIR / f"03_infer_{split}_{ft}.log")
            if rc != 0 or not out_path.exists():
                print(f"  ✗ {ft}/{split}: inference failed (exit={rc})")
                failed.append((ft, split))
    if failed:
        print(f"\n  ❌ Inference failed for: {failed}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Step 4 — Call-level evaluation (sweep + tables + CDF)
# ---------------------------------------------------------------------------
def step_call_level(feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 4] Call-level evaluation (sweep on val, eval on test)")
    summary_md = TABLE_DIR / "call_level_summary.md"
    if summary_md.exists() and not force:
        print(f"  ✓ {summary_md} exists; skipping (use --force to rerun)")
        return 0
    cmd = [
        sys.executable, "modeling/evaluate_call_level.py",
        "--feature_types", ",".join(feature_types),
        "--pred_dir", str(PRED_DIR),
        "--out_dir", str(TRACK_C_ROOT),
    ]
    return _run(cmd, LOG_DIR / "04_call_level.log")


# ---------------------------------------------------------------------------
# Step 5 — Side-by-side comparison table
# ---------------------------------------------------------------------------
def _read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _key(row: dict) -> tuple[str, str]:
    return (row["feature_type"], row["method"])


def step_compare_table(feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 5] Build before/after comparison table")
    out_csv = TABLE_DIR / "before_after_call_level.csv"
    out_md = TABLE_DIR / "before_after_call_level.md"
    if out_csv.exists() and out_md.exists() and not force:
        print(f"  ✓ {out_csv} exists; skipping (use --force to rerun)")
        return 0

    original = _read_summary(ORIGINAL_TABLE)
    bandlimited = _read_summary(TABLE_DIR / "call_level_summary.csv")
    if not original:
        print(f"  ✗ Missing original Track A summary at {ORIGINAL_TABLE}")
        return 1
    if not bandlimited:
        print(f"  ✗ Missing band-limited summary at {TABLE_DIR / 'call_level_summary.csv'}")
        return 1

    by_key_orig = {_key(r): r for r in original}
    by_key_bl = {_key(r): r for r in bandlimited}

    selected_methods = ["ema", "running_max"]
    fts = feature_types

    rows: list[dict] = []
    for ft in fts:
        for method in selected_methods:
            k = (ft, method)
            o = by_key_orig.get(k)
            b = by_key_bl.get(k)
            if o is None or b is None:
                continue

            def _f(row: dict, key: str) -> float:
                v = row.get(key, "")
                try:
                    return float(v) if v not in ("", "None", None) else float("nan")
                except (TypeError, ValueError):
                    return float("nan")

            f1_o, f1_b = _f(o, "f1"), _f(b, "f1")
            row = {
                "feature_type": ft,
                "method": method,
                "alpha_orig": o.get("alpha", "—") or "—",
                "tau_orig": o.get("tau", "—") or "—",
                "alpha_bl": b.get("alpha", "—") or "—",
                "tau_bl": b.get("tau", "—") or "—",
                "f1_original": f"{f1_o:.4f}",
                "f1_bandlimited": f"{f1_b:.4f}",
                "delta_f1": f"{f1_b - f1_o:+.4f}",
                "precision_original": f"{_f(o, 'precision'):.4f}",
                "precision_bandlimited": f"{_f(b, 'precision'):.4f}",
                "recall_original": f"{_f(o, 'recall'):.4f}",
                "recall_bandlimited": f"{_f(b, 'recall'):.4f}",
                "fpr_original": f"{_f(o, 'fpr'):.4f}",
                "fpr_bandlimited": f"{_f(b, 'fpr'):.4f}",
                "mean_alert_orig": f"{_f(o, 'mean_alert_latency'):.2f}",
                "mean_alert_bl": f"{_f(b, 'mean_alert_latency'):.2f}",
            }
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with out_md.open("w") as f:
        f.write("# Track C Phase B — Telephone-Band Ablation: Before/After\n\n")
        f.write("Comparison of call-level F1 between the original full-band model "
                "(Track A) and the band-limited (300–3400 Hz telephone band) "
                "retrained model (Track C Phase B).\n\n")
        f.write("**Interpretation:** a large negative ΔF1 indicates the original "
                "score relied on wide-band channel cues that telephone-band "
                "transmission would remove. A small ΔF1 indicates the model "
                "relies on signal that survives band-limiting.\n\n")
        cols = [
            ("feature_type", "Feature"),
            ("method", "Method"),
            ("f1_original", "F1 (orig)"),
            ("f1_bandlimited", "F1 (band-lim)"),
            ("delta_f1", "ΔF1"),
            ("precision_original", "P (orig)"),
            ("precision_bandlimited", "P (band-lim)"),
            ("recall_original", "R (orig)"),
            ("recall_bandlimited", "R (band-lim)"),
            ("fpr_original", "FPR (orig)"),
            ("fpr_bandlimited", "FPR (band-lim)"),
        ]
        f.write("| " + " | ".join(h for _, h in cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k, _ in cols) + " |\n")
        f.write("\n*Hyperparameters were independently re-tuned on the validation "
                "set for each setting (full-band vs. band-limited), so the α and "
                "τ values may differ between rows of the same feature type.*\n")

    print(f"  ✓ wrote {out_csv}")
    print(f"  ✓ wrote {out_md}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", default="all", choices=STEPS,
                        help="Which step to run (default: all)")
    parser.add_argument("--feature_types", default=",".join(FEATURE_TYPES_DEFAULT),
                        help="Comma-separated feature types to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-run steps even if outputs already exist")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Retrain epochs (default: 20, matches original)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)
        return 1

    feature_types = [ft.strip() for ft in args.feature_types.split(",") if ft.strip()]
    for ft in feature_types:
        if ft not in FEATURE_TYPES_DEFAULT:
            print(f"⚠️  Unknown feature type: {ft}", file=sys.stderr)
            return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Track C Phase B — band-limited retrain + evaluate")
    print(f"Feature types: {feature_types}")
    print(f"Mode: {args.mode}")
    print(f"Force: {args.force}")
    print("=" * 80)

    steps = (["build_manifests", "retrain", "infer", "call_level", "compare_table"]
             if args.mode == "all" else [args.mode])

    for step in steps:
        if step == "build_manifests":
            rc = step_build_manifests(args.force)
        elif step == "retrain":
            rc = step_retrain(feature_types, args.epochs, args.batch_size, args.lr, args.force)
        elif step == "infer":
            rc = step_infer(feature_types, args.force)
        elif step == "call_level":
            rc = step_call_level(feature_types, args.force)
        elif step == "compare_table":
            rc = step_compare_table(feature_types, args.force)
        else:
            print(f"Unknown step: {step}", file=sys.stderr)
            return 1
        if rc != 0:
            print(f"\n❌ Step '{step}' failed with exit code {rc}", file=sys.stderr)
            return rc

    print("\n" + "=" * 80)
    print("✅ Phase B complete.")
    print(f"   • Predictions: {PRED_DIR}/{{val,test}}_*.jsonl")
    print(f"   • Call-level table: {TABLE_DIR}/call_level_summary.{{csv,md}}")
    print(f"   • Before/after table: {TABLE_DIR}/before_after_call_level.{{csv,md}}")
    print(f"   • Logs: {LOG_DIR}/")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
