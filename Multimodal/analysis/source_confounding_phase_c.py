"""Track C Phase C orchestrator — telephony codec augmentation + evaluate.

Idea: simulate "what if AllHub non-vishing audio had also been
transmitted through a phone channel?" by GSM round-tripping the
non-vishing waveforms, re-extracting features, and evaluating the
**existing** full-band-trained classifiers on this matched-channel test
manifest (vishing keeps original FSS features; non-vishing swaps to
codec-augmented features). If the F1 drops, the original classifier
relied on channel artifacts; if it stays high, the discrimination is
genuinely about content/prosody.

Steps (run in order; each step is independently re-runnable):
  1. extract_features  — re-extract MFCC + eGeMAPS + Wav2Vec2 features
                         from codec-augmented non-vishing audio.
                         (features_extraction/build_save_codec_features.py)
                         Long-running (~3–5 h with Wav2Vec2 on GPU).
  2. build_manifest    — clone test manifest with non-vishing feature
                         paths swapped to codec-augmented dirs.
                         (analysis/build_codec_aug_manifest.py)
  3. infer             — for each feature type, run segment-level
                         inference on the matched test manifest using
                         the existing full-band-trained classifier.
  4. call_level        — invoke modeling/evaluate_call_level.py with
                         --pred_dir / --out_dir pointing at Phase C
                         outputs. We pass val predictions from Track A
                         (full-band) for the sweep, then test predictions
                         from Phase C for the matched-channel test eval.
  5. compare_table     — load Track A's original call-level summary and
                         the Phase C codec-augmented summary; emit a
                         side-by-side comparison.

Idempotency: each step skips by default if outputs already exist; use
`--force` to re-run.

Run from the Multimodal/ directory:
    # Heavy step: extract codec features (run once, idempotent)
    python analysis/source_confounding_phase_c.py --mode extract_features

    # All other steps:
    python analysis/source_confounding_phase_c.py --mode all

    # Smoke test (skip Wav2Vec2):
    python analysis/source_confounding_phase_c.py --feature_types egemaps,mfcc
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
TRACK_C_ROOT = Path("modeling/logs/track_c/phase_c")
LOG_DIR = TRACK_C_ROOT / "logs"
PRED_DIR = TRACK_C_ROOT / "segment_predictions"
TABLE_DIR = TRACK_C_ROOT / "tables"

ORIGINAL_TABLE = Path("modeling/logs/track_a/tables/call_level_summary.csv")
ORIGINAL_PRED_DIR = Path("modeling/logs/track_a/segment_predictions")  # for val sweep

DATA_DIR = Path("data")
MODEL_DIR = Path("modeling/models")
FEATURES_DIR = Path("features")

# Map our feature_type label to the existing full-band checkpoint filename.
# Track A trained "all" with the file name "Audio-All" (legacy capitalization);
# the others use plain lowercase. Keep this map in sync with modeling/models/.
CHECKPOINT_FOR = {
    "egemaps":  "best_audio_model_egemaps.pth",
    "mfcc":     "best_audio_model_mfcc.pth",
    "wav2vec2": "best_audio_model_wav2vec2.pth",
    "all":      "best_audio_model_Audio-All.pth",
}

FEATURE_TYPES_DEFAULT = ["egemaps", "mfcc", "wav2vec2", "all"]
STEPS = ["extract_features", "build_manifest", "infer", "call_level", "compare_table", "all"]

CODEC_FEATURE_DIRS = {
    "mfcc":     FEATURES_DIR / "mfcc_codec_aug_nonvishing",
    "egemaps":  FEATURES_DIR / "egemaps_codec_aug_nonvishing",
    "wav2vec2": FEATURES_DIR / "wav2vec2_codec_aug_nonvishing",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess, tee output to log_path, return exit code.

    Forces unbuffered stdout from child Python processes so the tee log
    flushes line-by-line in real time (bug surfaced during Phase B where
    the orchestrator's log was empty for ~30 minutes while training was
    actively running but Python was buffering print() to a pipe).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # If launching a Python interpreter, force unbuffered output (-u + env)
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


def _codec_test_manifest() -> Path:
    return DATA_DIR / "test_segment_manifest_codec_aug.jsonl"


def _segment_pred_path(split: str, feature_type: str) -> Path:
    return PRED_DIR / f"{split}_{feature_type}.jsonl"


def _checkpoint_path(feature_type: str) -> Path:
    return MODEL_DIR / CHECKPOINT_FOR[feature_type]


# ---------------------------------------------------------------------------
# Step 1 — Re-extract codec-augmented features (non-vishing only)
# ---------------------------------------------------------------------------
def _count_npy(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("*.npy"))


def step_extract_features(force: bool, splits: str) -> int:
    print(f"\n[Step 1] Re-extract codec-augmented features (non-vishing only)")
    counts = {ft: _count_npy(d) for ft, d in CODEC_FEATURE_DIRS.items()}
    print(f"  current .npy counts: {counts}")
    # Heuristic: this step has been run if any feature dir has >0 files;
    # the feature-extraction script itself is idempotent and will fill gaps.
    if all(c > 0 for c in counts.values()) and not force:
        print(f"  ✓ codec feature dirs already populated; skipping (use --force to re-run)")
        return 0
    cmd = [
        sys.executable, "features_extraction/build_save_codec_features.py",
        "--splits", splits,
    ]
    if force:
        cmd.append("--overwrite")
    return _run(cmd, LOG_DIR / "01_extract_codec_features.log")


# ---------------------------------------------------------------------------
# Step 2 — Build matched-channel test manifest
# ---------------------------------------------------------------------------
def step_build_manifest(force: bool) -> int:
    print(f"\n[Step 2] Build matched-channel test manifest")
    out = _codec_test_manifest()
    if out.exists() and not force:
        print(f"  ✓ {out} exists; skipping")
        return 0
    cmd = [sys.executable, "analysis/build_codec_aug_manifest.py", "--split", "test"]
    return _run(cmd, LOG_DIR / "02_build_manifest.log")


# ---------------------------------------------------------------------------
# Step 3 — Run segment-level inference on the matched test manifest
# ---------------------------------------------------------------------------
def step_infer(feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 3] Segment-level inference on codec-augmented test manifest")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for ft in feature_types:
        ckpt = _checkpoint_path(ft)
        if not ckpt.exists():
            print(f"  ✗ {ft}: missing checkpoint {ckpt}")
            failed.append(ft)
            continue
        out_path = _segment_pred_path("test", ft)
        if out_path.exists() and not force:
            print(f"  ✓ {ft}: {out_path} exists; skipping")
            continue
        cmd = [
            sys.executable, "modeling/run_segment_inference.py",
            "--checkpoint", str(ckpt),
            "--manifest", str(_codec_test_manifest()),
            "--feature_type", ft,
            "--output", str(out_path),
        ]
        rc = _run(cmd, LOG_DIR / f"03_infer_test_{ft}.log")
        if rc != 0 or not out_path.exists():
            print(f"  ✗ {ft}: inference failed (exit={rc})")
            failed.append(ft)
    if failed:
        print(f"\n  ❌ Inference failed for: {failed}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Step 4 — Call-level evaluation (val sweep on Track A predictions, test on Phase C)
# ---------------------------------------------------------------------------
def step_call_level(feature_types: list[str], force: bool) -> int:
    """We need val predictions to sweep hyperparameters but val audio has not
    been codec-augmented (we only built test_segment_manifest_codec_aug.jsonl).
    The val sweep therefore reuses Track A's full-band val predictions, while
    the Phase C test predictions are evaluated under those val-tuned configs.

    To make `evaluate_call_level.py` happy with this hybrid setup, we
    symlink the full-band val predictions into our pred_dir, then run.
    """
    print(f"\n[Step 4] Call-level evaluation (val sweep from Track A, test from Phase C)")
    summary_md = TABLE_DIR / "call_level_summary.md"
    if summary_md.exists() and not force:
        print(f"  ✓ {summary_md} exists; skipping")
        return 0

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for ft in feature_types:
        src = ORIGINAL_PRED_DIR / f"val_{ft}.jsonl"
        dst = PRED_DIR / f"val_{ft}.jsonl"
        if not src.exists():
            print(f"  ✗ Missing original val predictions for sweep: {src}")
            return 1
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        # Use copy (not symlink) so the artifact bundle is self-contained
        shutil.copyfile(src, dst)
        print(f"  copied val predictions for sweep: {src} → {dst}")

    cmd = [
        sys.executable, "modeling/evaluate_call_level.py",
        "--feature_types", ",".join(feature_types),
        "--pred_dir", str(PRED_DIR),
        "--out_dir", str(TRACK_C_ROOT),
    ]
    return _run(cmd, LOG_DIR / "04_call_level.log")


# ---------------------------------------------------------------------------
# Step 5 — Side-by-side comparison
# ---------------------------------------------------------------------------
def _read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _key(row: dict) -> tuple[str, str]:
    return (row["feature_type"], row["method"])


def step_compare_table(feature_types: list[str], force: bool) -> int:
    print(f"\n[Step 5] Build before/after comparison (full-band vs. codec-matched)")
    out_csv = TABLE_DIR / "before_after_call_level.csv"
    out_md = TABLE_DIR / "before_after_call_level.md"
    if out_csv.exists() and out_md.exists() and not force:
        print(f"  ✓ {out_csv} exists; skipping")
        return 0

    original = _read_summary(ORIGINAL_TABLE)
    codec = _read_summary(TABLE_DIR / "call_level_summary.csv")
    if not original:
        print(f"  ✗ Missing original Track A summary at {ORIGINAL_TABLE}")
        return 1
    if not codec:
        print(f"  ✗ Missing Phase C summary at {TABLE_DIR / 'call_level_summary.csv'}")
        return 1

    by_orig = {_key(r): r for r in original}
    by_codec = {_key(r): r for r in codec}

    selected_methods = ["ema", "running_max"]

    rows: list[dict] = []
    for ft in feature_types:
        for method in selected_methods:
            o = by_orig.get((ft, method))
            c = by_codec.get((ft, method))
            if o is None or c is None:
                continue

            def _f(r: dict, k: str) -> float:
                v = r.get(k, "")
                try:
                    return float(v) if v not in ("", "None", None) else float("nan")
                except (TypeError, ValueError):
                    return float("nan")

            f1_o, f1_c = _f(o, "f1"), _f(c, "f1")
            rows.append({
                "feature_type": ft,
                "method": method,
                "f1_original": f"{f1_o:.4f}",
                "f1_codec_matched": f"{f1_c:.4f}",
                "delta_f1": f"{f1_c - f1_o:+.4f}",
                "precision_original": f"{_f(o, 'precision'):.4f}",
                "precision_codec_matched": f"{_f(c, 'precision'):.4f}",
                "recall_original": f"{_f(o, 'recall'):.4f}",
                "recall_codec_matched": f"{_f(c, 'recall'):.4f}",
                "fpr_original": f"{_f(o, 'fpr'):.4f}",
                "fpr_codec_matched": f"{_f(c, 'fpr'):.4f}",
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with out_md.open("w") as f:
        f.write("# Track C Phase C — Codec-Matched Channel: Before/After\n\n")
        f.write("Comparison of call-level F1 between the original full-band test "
                "(Track A) and the matched-channel test (Track C Phase C) where "
                "non-vishing audio is GSM-codec round-tripped to simulate a phone "
                "channel. The same full-band-trained classifier is used in both "
                "settings; only the inputs differ.\n\n")
        f.write("**Interpretation:** a large negative ΔF1 indicates the classifier "
                "had been exploiting channel-artifact differences between FSS and "
                "AllHub. A small ΔF1 indicates the classifier discriminates on "
                "signal that survives the channel match.\n\n")
        cols = [
            ("feature_type", "Feature"),
            ("method", "Method"),
            ("f1_original", "F1 (orig)"),
            ("f1_codec_matched", "F1 (codec)"),
            ("delta_f1", "ΔF1"),
            ("precision_original", "P (orig)"),
            ("precision_codec_matched", "P (codec)"),
            ("recall_original", "R (orig)"),
            ("recall_codec_matched", "R (codec)"),
            ("fpr_original", "FPR (orig)"),
            ("fpr_codec_matched", "FPR (codec)"),
        ]
        f.write("| " + " | ".join(h for _, h in cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k, _ in cols) + " |\n")

    print(f"  ✓ wrote {out_csv}")
    print(f"  ✓ wrote {out_md}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", default="all", choices=STEPS)
    parser.add_argument("--feature_types", default=",".join(FEATURE_TYPES_DEFAULT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--splits", default="test",
                        help="Splits to extract codec features for. "
                             "Phase C only evaluates test, but you can pre-fill val/train too.")
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
    print(f"Track C Phase C — codec-matched channel evaluation")
    print(f"Feature types: {feature_types}")
    print(f"Mode: {args.mode}")
    print(f"Force: {args.force}")
    print("=" * 80)

    steps = (["extract_features", "build_manifest", "infer", "call_level", "compare_table"]
             if args.mode == "all" else [args.mode])

    for step in steps:
        if step == "extract_features":
            rc = step_extract_features(args.force, args.splits)
        elif step == "build_manifest":
            rc = step_build_manifest(args.force)
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
    print("✅ Phase C complete.")
    print(f"   • Test predictions: {PRED_DIR}/test_*.jsonl")
    print(f"   • Call-level table: {TABLE_DIR}/call_level_summary.{{csv,md}}")
    print(f"   • Before/after table: {TABLE_DIR}/before_after_call_level.{{csv,md}}")
    print(f"   • Logs: {LOG_DIR}/")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
