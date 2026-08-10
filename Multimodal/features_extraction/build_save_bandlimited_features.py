"""Re-extract MFCC, eGeMAPS, Wav2Vec2 features from band-limited audio.

Track C Phase B step. For every audio file in `data/audio/{vishing,non_vishing}/`,
load the waveform, apply the 300–3400 Hz telephone bandpass, then extract
features per-segment using the same parameters as the original
extraction. Output features are written to parallel directories so the
existing full-band features remain untouched.

Optimizations vs. `build_save_aligned_audio_features.py`:
  * eGeMAPS uses `smile.process_signal()` in memory (skips per-segment
    tempfile I/O — order of magnitude faster).
  * Wav2Vec2 model is moved to GPU at startup (the original module-level
    instance lives on CPU; we mutate it).
  * Idempotent — segments whose three .npy outputs already exist are
    skipped.

Run from the Multimodal/ directory (with the vishing venv active):
    python features_extraction/build_save_bandlimited_features.py \\
        --splits train,val,test
    # Re-run later: existing outputs are skipped (resume support)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

# Make `features_extraction.*` and `preprocessing.*` importable whether this
# file is invoked as `python features_extraction/build_save_bandlimited_features.py`
# (Python puts features_extraction/ on the path, not Multimodal/) or via
# `python -m features_extraction.build_save_bandlimited_features`.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from features_extraction.audio_features import (  # type: ignore  # noqa: E402
    _w2v_model,
    _w2v_processor,
    smile,
    extract_mfcc,
)
from preprocessing.apply_telephony_bandpass import apply_bandpass  # noqa: E402


# Output directory layout
FEATURES_DIR = Path("features")
OUT_SUBDIRS = {
    "mfcc": "mfcc_telephony_bandlimited",
    "egemaps": "egemaps_telephony_bandlimited",
    "wav2vec2": "wav2vec2_telephony_bandlimited",
}


def audio_path_for(call_id: str) -> Path:
    if call_id.startswith("vishing_"):
        return Path("data/audio/vishing") / f"{call_id}.wav"
    return Path("data/audio/non_vishing") / f"{call_id}.wav"


def fast_extract_egemaps(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """In-memory eGeMAPS via opensmile.process_signal — no tempfile I/O."""
    feats = smile.process_signal(y, sr)
    return feats.values


def fast_extract_wav2vec2(y: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    """Wav2Vec2 hidden states with the model on GPU (vs. the module's CPU default)."""
    inputs = _w2v_processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _w2v_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)


def all_outputs_exist(seg_id: str) -> bool:
    return all(
        (FEATURES_DIR / sub / f"{seg_id}.npy").exists() for sub in OUT_SUBDIRS.values()
    )


def process_call(
    call_id: str,
    segments: list[dict],
    *,
    sr: int,
    device: torch.device,
    overwrite: bool,
) -> dict:
    """Load + bandpass once, extract features per segment.

    Returns a small stats dict (counts, timings) for progress reporting.
    """
    audio_path = audio_path_for(call_id)
    if not audio_path.exists():
        return {"call_id": call_id, "skipped": True, "reason": f"missing {audio_path}"}

    n_done = 0
    n_skip = 0
    t_load = t_filt = t_mfcc = t_eg = t_w2v = 0.0

    if not overwrite and all(all_outputs_exist(seg["segment_id"]) for seg in segments):
        return {"call_id": call_id, "skipped": True, "reason": "already complete"}

    t0 = time.time()
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    t_load = time.time() - t0

    t0 = time.time()
    y_bp = apply_bandpass(y, sr=sr)
    t_filt = time.time() - t0

    for seg in segments:
        seg_id = seg["segment_id"]
        if not overwrite and all_outputs_exist(seg_id):
            n_skip += 1
            continue

        s = max(0, int(float(seg["start"]) * sr))
        e = min(len(y_bp), int(float(seg["end"]) * sr))
        clip = y_bp[s:e]
        if clip.size < 64:
            # Too short — write zeros so manifest stays consistent
            clip = np.zeros(64, dtype=np.float32)

        # MFCC (CPU)
        path_mfcc = FEATURES_DIR / OUT_SUBDIRS["mfcc"] / f"{seg_id}.npy"
        if overwrite or not path_mfcc.exists():
            t1 = time.time()
            np.save(path_mfcc, extract_mfcc(clip, sr).astype(np.float32))
            t_mfcc += time.time() - t1

        # eGeMAPS (CPU, in-memory)
        path_egem = FEATURES_DIR / OUT_SUBDIRS["egemaps"] / f"{seg_id}.npy"
        if overwrite or not path_egem.exists():
            t1 = time.time()
            np.save(path_egem, fast_extract_egemaps(clip, sr).astype(np.float32))
            t_eg += time.time() - t1

        # Wav2Vec2 (GPU)
        path_w2v = FEATURES_DIR / OUT_SUBDIRS["wav2vec2"] / f"{seg_id}.npy"
        if overwrite or not path_w2v.exists():
            t1 = time.time()
            np.save(path_w2v, fast_extract_wav2vec2(clip, sr, device))
            t_w2v += time.time() - t1

        n_done += 1

    return {
        "call_id": call_id,
        "skipped": False,
        "n_done": n_done,
        "n_skip": n_skip,
        "t_load": round(t_load, 3),
        "t_filt": round(t_filt, 3),
        "t_mfcc": round(t_mfcc, 3),
        "t_egemaps": round(t_eg, 3),
        "t_wav2vec2": round(t_w2v, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to process",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if outputs already exist")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--limit-calls", type=int, default=0,
                        help="Process only the first N calls per split (debug)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (auto if unset)")
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move Wav2Vec2 to GPU (the module-level instance is on CPU by default)
    _w2v_model.to(device)
    _w2v_model.eval()
    print(f"Wav2Vec2 device: {next(_w2v_model.parameters()).device}")

    # Create output directories
    for sub in OUT_SUBDIRS.values():
        (FEATURES_DIR / sub).mkdir(parents=True, exist_ok=True)
        print(f"  output dir: {FEATURES_DIR / sub}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    overall_t0 = time.time()
    overall_summary: list[dict] = []

    for split in splits:
        manifest_path = Path(f"data/{split}_segment_manifest_merged.jsonl")
        if not manifest_path.exists():
            print(f"⚠️  manifest not found: {manifest_path}", file=sys.stderr)
            continue

        print(f"\n=== {split} ===")
        with manifest_path.open(encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]
        print(f"  {len(entries)} segments")

        # Group by call_id, preserve segment order within a call
        by_call: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_call[e["call_id"]].append(e)
        for c in by_call:
            by_call[c].sort(key=lambda s: s["start"])
        print(f"  {len(by_call)} unique call_ids")

        items = list(by_call.items())
        if args.limit_calls > 0:
            items = items[: args.limit_calls]
            print(f"  limit-calls={args.limit_calls} → processing first {len(items)} calls")

        t_start = time.time()
        per_call_stats: list[dict] = []
        for i, (call_id, segs) in enumerate(items):
            res = process_call(
                call_id, segs, sr=args.sr, device=device, overwrite=args.overwrite,
            )
            per_call_stats.append(res)

            # Progress update every 10 calls or at end
            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (len(items) - (i + 1)) / rate if rate > 0 else 0.0
                done = sum(s.get("n_done", 0) for s in per_call_stats)
                skipped = sum(s.get("n_skip", 0) for s in per_call_stats)
                print(
                    f"  [{split}] {i + 1}/{len(items)} calls "
                    f"(segments done={done}, skipped={skipped}, "
                    f"{rate:.2f} calls/s, ETA {eta / 60:.1f} min)"
                )

        overall_summary.append({
            "split": split,
            "n_calls": len(items),
            "n_segments_processed": sum(s.get("n_done", 0) for s in per_call_stats),
            "n_segments_skipped_existing": sum(s.get("n_skip", 0) for s in per_call_stats),
            "elapsed_s": round(time.time() - t_start, 1),
        })

    print("\n=== Summary ===")
    for s in overall_summary:
        print(f"  {s}")
    print(f"\nTotal wall-clock: {(time.time() - overall_t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
