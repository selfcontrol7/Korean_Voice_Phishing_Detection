"""Track B step 2 — Build the tarball the user copies to the S24 Ultra.

Selects 100 random segments (stratified vishing/non-vishing) and 50 random
vishing test calls from the existing test_segment manifest. Clips each
segment's audio range into a small .wav, copies each call's full .wav,
bundles the TorchScript models, writes a manifest.json with per-segment
and per-call metadata (including EMA hyperparameters from Track A's
val-tuned configs), and emits the tarball.

The manifest also carries pre-computed eGeMAPS features per segment as a
fallback for when opensmile is not installable on Termux ARM64. The
benchmark script uses these only if `import opensmile` fails on the phone.

Output:
    analysis/phone_package/   (directory)
    phone_package.tar.gz       (~50-100 MB)

Run from the Multimodal/ directory:
    python analysis/track_b_prepare_phone_package.py --n_segments 100 --n_calls 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


SEG_MANIFEST = Path("data/test_segment_manifest_merged.jsonl")
MASTER_MANIFEST = Path("data/master_manifest.jsonl")
BEST_CONFIGS = Path("modeling/logs/track_a/call_level/best_configs.json")

PACKAGE_DIR = Path("analysis/phone_package")
SEG_AUDIO_DIR = PACKAGE_DIR / "audio" / "seg_audio"
CALL_AUDIO_DIR = PACKAGE_DIR / "audio" / "call_audio"
TARBALL = Path("analysis/phone_package.tar.gz")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_best_configs() -> dict:
    with BEST_CONFIGS.open() as f:
        return json.load(f)


def _audio_path_for(call_id: str) -> Path:
    if call_id.startswith("vishing_"):
        return Path("data/audio/vishing") / f"{call_id}.wav"
    return Path("data/audio/non_vishing") / f"{call_id}.wav"


def _clip_audio(call_id: str, start: float, end: float, sr: int = 16000) -> np.ndarray:
    src = _audio_path_for(call_id)
    y, sr_in = sf.read(str(src), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr_in != sr:
        # Simple linear resample (the audio is already 16 kHz from the pipeline)
        n_out = int(round(len(y) * sr / sr_in))
        y = np.interp(np.linspace(0, len(y) - 1, n_out), np.arange(len(y)), y).astype(np.float32)
    s = max(0, int(start * sr))
    e = min(len(y), int(end * sr))
    return y[s:e].astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n_segments", type=int, default=100)
    parser.add_argument("--n_calls", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_egemaps_features", action="store_true", default=True,
                        help="Embed pre-computed eGeMAPS features per segment as opensmile fallback")
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)
        return 1

    # Clean any prior package state but keep the .ptl models (Step 1 output)
    for d in (SEG_AUDIO_DIR, CALL_AUDIO_DIR):
        if d.exists():
            shutil.rmtree(d)
    SEG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    CALL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Track B step 2 — build phone package")
    print("=" * 70)

    # 1. Load existing artifacts
    segments = _load_jsonl(SEG_MANIFEST)
    print(f"Loaded {len(segments)} segments from {SEG_MANIFEST}")
    calls_master = _load_jsonl(MASTER_MANIFEST)
    print(f"Loaded {len(calls_master)} calls from {MASTER_MANIFEST}")
    best_cfgs = _load_best_configs()
    print(f"Loaded EMA configs: {list(best_cfgs.keys())}")

    # MFCC + eGeMAPS hyperparameters (val-tuned in Track A)
    mfcc_cfg = best_cfgs["mfcc"]["ema"]
    egem_cfg = best_cfgs["egemaps"]["ema"]
    print(f"  mfcc EMA: α={mfcc_cfg['alpha']}, τ={mfcc_cfg['tau']}")
    print(f"  egemaps EMA: α={egem_cfg['alpha']}, τ={egem_cfg['tau']}")

    # 2. Sample 50 vishing + 50 non-vishing segments
    rng = random.Random(args.seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for s in segments:
        by_label[s["label"]].append(s)
    n_per_label = args.n_segments // 2
    chosen_segments: list[dict] = []
    for lbl in (1, 0):
        bucket = list(by_label[lbl])
        rng.shuffle(bucket)
        chosen_segments.extend(bucket[:n_per_label])
    rng.shuffle(chosen_segments)
    print(f"\nSampled {len(chosen_segments)} segments "
          f"({sum(1 for s in chosen_segments if s['label']==1)} vishing, "
          f"{sum(1 for s in chosen_segments if s['label']==0)} non-vishing)")

    # 3. Clip segment audio
    print(f"Writing {len(chosen_segments)} segment clips to {SEG_AUDIO_DIR}/")
    seg_records: list[dict] = []
    for i, s in enumerate(chosen_segments, 1):
        clip = _clip_audio(s["call_id"], float(s["start"]), float(s["end"]))
        out_wav = SEG_AUDIO_DIR / f"{s['segment_id']}.wav"
        sf.write(str(out_wav), clip, 16000)
        rec = {
            "segment_id": s["segment_id"],
            "call_id": s["call_id"],
            "label": int(s["label"]),
            "start": float(s["start"]),
            "end": float(s["end"]),
            "duration_s": float(s["end"]) - float(s["start"]),
            "alpha": float(mfcc_cfg["alpha"]),  # for the EMA timing demo (single-step)
            "tau": float(mfcc_cfg["tau"]),
        }
        if args.include_egemaps_features:
            egem_path = Path(s["egemaps_path"])
            if egem_path.exists():
                feat = np.load(egem_path).flatten().astype(np.float32)
                rec["egemaps_features"] = feat.tolist()
        seg_records.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(chosen_segments)}")

    # 4. Sample 50 vishing test calls (vishing because Track A's alert-latency
    #    story is about TP vishing calls)
    test_call_ids = {s["call_id"] for s in segments if s["label"] == 1}
    vishing_calls = [c for c in calls_master if c["call_id"] in test_call_ids]
    rng.shuffle(vishing_calls)
    chosen_calls = vishing_calls[:args.n_calls]
    print(f"\nSampled {len(chosen_calls)} vishing test calls")

    # 5. Copy each call's wav + collect its segments from the test manifest
    by_call_segments: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        by_call_segments[s["call_id"]].append(s)
    for c in by_call_segments:
        by_call_segments[c].sort(key=lambda s: float(s["start"]))

    call_records: list[dict] = []
    for i, c in enumerate(chosen_calls, 1):
        src = _audio_path_for(c["call_id"])
        dst = CALL_AUDIO_DIR / f"{c['call_id']}.wav"
        shutil.copyfile(src, dst)
        segs = by_call_segments.get(c["call_id"], [])
        # Pack per-segment metadata for the call (incl. precomputed eGeMAPS)
        packed_segs: list[dict] = []
        for s in segs:
            ps = {
                "segment_id": s["segment_id"],
                "start": float(s["start"]),
                "end": float(s["end"]),
            }
            if args.include_egemaps_features:
                ep = Path(s["egemaps_path"])
                if ep.exists():
                    ps["egemaps_features"] = np.load(ep).flatten().astype(np.float32).tolist()
            packed_segs.append(ps)
        call_records.append({
            "call_id": c["call_id"],
            "label": int(c["label"]),
            "source": c.get("source", "FSS"),
            "alpha": float(mfcc_cfg["alpha"]),  # reused for both features; egem alpha is similar
            "tau": float(mfcc_cfg["tau"]),
            "segments": packed_segs,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(chosen_calls)} calls copied")

    # 6. Manifest JSON
    manifest = {
        "n_segments": len(seg_records),
        "n_calls": len(call_records),
        "seed": args.seed,
        "feature_hparams": {
            "mfcc":    {"alpha": mfcc_cfg["alpha"], "tau": mfcc_cfg["tau"]},
            "egemaps": {"alpha": egem_cfg["alpha"], "tau": egem_cfg["tau"]},
        },
        "segments": seg_records,
        "calls": call_records,
        "egemaps_feature_dim": (len(seg_records[0].get("egemaps_features", [])) if seg_records and "egemaps_features" in seg_records[0] else None),
    }
    (PACKAGE_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    print(f"\nWrote manifest.json with {len(seg_records)} segments + {len(call_records)} calls")

    # 7. Copy benchmark script + README
    shutil.copyfile("analysis/track_b_benchmark.py", PACKAGE_DIR / "benchmark.py")
    (PACKAGE_DIR / "README.md").write_text(_README_TEXT.strip() + "\n")
    print(f"Copied benchmark.py + README.md")

    # 8. Tar everything
    print(f"\nBuilding {TARBALL}...")
    with tarfile.open(TARBALL, "w:gz") as tf:
        for child in PACKAGE_DIR.iterdir():
            tf.add(child, arcname=f"phone_package/{child.name}")
    size_mb = TARBALL.stat().st_size / 1e6
    print(f"✅ Wrote {TARBALL}  ({size_mb:.1f} MB)")

    print("\nNext: copy phone_package.tar.gz to the phone, then follow")
    print("       paper1/writting/track_b_phone_procedure.md")
    return 0


_README_TEXT = """
# Track B Phone Package

Contents:
- `audio/seg_audio/*.wav` — 100 short clips for per-segment latency
- `audio/call_audio/*.wav` — 50 full vishing calls for per-call streaming
- `models/best_audio_model_{mfcc,egemaps}.ptl` — TorchScript classifiers
- `manifest.json` — segment + call metadata (incl. EMA α/τ + fallback eGeMAPS features)
- `benchmark.py` — the on-phone benchmark script

Quick run:
    pkg install -y python git
    pip install --user torch numpy soundfile librosa
    # optional: pip install --user opensmile
    cd phone_package
    python benchmark.py --output phone_results.csv --summary phone_summary.json

Send `phone_results.csv` + `phone_summary.json` back to the workstation.
Full instructions: see `track_b_phone_procedure.md` on the workstation.
"""


if __name__ == "__main__":
    sys.exit(main())
