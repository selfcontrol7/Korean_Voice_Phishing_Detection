"""Track B step 3 — On-phone deployment benchmark (Termux on Samsung Galaxy S24 Ultra).

Measures three deployment metrics asked for by Reviewer 1 (latency) and
Reviewer 3 (runtime, latency, energy) for the IEEE Access resubmission:

  * Per-segment latency with stage breakdown
        (audio load | feature extract | TorchScript fwd | EMA update)
  * Per-call streaming wall-clock + time-to-first-alert
  * Peak resident memory (VmRSS) + CPU% per feature-type block
  * Sustained-load energy (battery-percentage delta over a fixed
        duration, optionally cross-checked with charge_counter for newer
        Android), plus temperature delta for thermal-stability reporting

Three execution modes (--mode):
    latency   per-segment + per-call timings + memory snapshots
    energy    sustained N-minute loop with battery sampling
    both      latency mode followed by energy mode (default)

Self-contained: torch, numpy, soundfile required; librosa optional
(numpy MFCC fallback), opensmile-python optional (we prefer the on-phone
`SMILExtract` binary built by build_opensmile.sh). When SMILExtract is
not available, eGeMAPS path falls back to precomputed features in the
manifest, with a clear flag in the output JSON.

Run on the phone (after build_opensmile.sh has produced SMILExtract):
    python benchmark.py --mode both \\
        --output phone_results_v2.csv \\
        --summary phone_summary_v2.json \\
        --energy_json phone_energy.json \\
        --energy_duration 600
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    HAS_SOUNDFILE = False

try:
    import librosa  # type: ignore
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    import opensmile  # type: ignore
    HAS_OPENSMILE = True
except Exception:
    HAS_OPENSMILE = False


# ---------------------------------------------------------------------------
# System probes — battery, memory, CPU
# ---------------------------------------------------------------------------
def has_termux_api() -> bool:
    return shutil.which("termux-battery-status") is not None


def read_battery() -> Optional[dict]:
    """Run `termux-battery-status` and return the parsed JSON.

    Newer Android exposes `voltage`, `current`, `charge_counter` in
    addition to the always-present `percentage`, `temperature`,
    `health`, `status`. Returns None if termux-api is unavailable.
    """
    if not has_termux_api():
        return None
    try:
        out = subprocess.check_output(
            ["termux-battery-status"], timeout=10, text=True
        )
        return json.loads(out)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def read_rss_mb() -> float:
    """Return current resident set size in MB (from /proc/self/status)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return float("nan")


def read_vm_hwm_mb() -> float:
    """Peak RSS since process started (VmHWM in /proc/self/status)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return float("nan")


_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def read_cpu_time_s() -> float:
    """Return cumulative user+system CPU seconds for this process."""
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().split()
        utime, stime = int(fields[13]), int(fields[14])
        return (utime + stime) / _CLK_TCK
    except Exception:
        return float("nan")


def memcpu_snapshot(label: str) -> dict:
    return {
        "label": label,
        "wall_s": time.perf_counter(),
        "cpu_s": read_cpu_time_s(),
        "rss_mb": read_rss_mb(),
        "vm_hwm_mb": read_vm_hwm_mb(),
    }


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_wav(path: Path, sr_target: int = 16000) -> np.ndarray:
    if HAS_SOUNDFILE:
        y, sr = sf.read(str(path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            if HAS_LIBROSA:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_target, res_type="kaiser_fast")
            else:
                n_out = int(round(len(y) * sr_target / sr))
                y = np.interp(np.linspace(0, len(y) - 1, n_out), np.arange(len(y)), y).astype(np.float32)
        return y.astype(np.float32, copy=False)
    if HAS_LIBROSA:
        y, _ = librosa.load(str(path), sr=sr_target, mono=True)
        return y.astype(np.float32, copy=False)
    raise RuntimeError("Need soundfile or librosa to load audio")


# ---------------------------------------------------------------------------
# MFCC extractors (librosa + numpy fallback) — unchanged from v1
# ---------------------------------------------------------------------------
def extract_mfcc_librosa(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=400, hop_length=160)
    return mfcc.T.mean(axis=0).astype(np.float32)


def extract_mfcc_numpy(y: np.ndarray, sr: int = 16000,
                       n_mfcc: int = 13, n_fft: int = 400, hop: int = 160,
                       n_mels: int = 40) -> np.ndarray:
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    n_frames = 1 + (len(y) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros(n_mfcc, dtype=np.float32)
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, n_fft),
        strides=(y.strides[0] * hop, y.strides[0]),
    ).copy()
    win = np.hanning(n_fft).astype(np.float32)
    frames *= win
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel_fb = _mel_filterbank_numpy(n_mels=n_mels, n_fft=n_fft, sr=sr)
    mel = power @ mel_fb.T
    log_mel = np.log(np.maximum(mel, 1e-10))
    dct = _dct2_numpy(log_mel, n_mfcc=n_mfcc)
    return dct.mean(axis=0).astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank_numpy(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    nyq = sr / 2.0
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(nyq), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if c > l:
            fb[m - 1, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        if r > c:
            fb[m - 1, c:r] = (r - np.arange(c, r)) / max(1, r - c)
    return fb


def _dct2_numpy(x: np.ndarray, n_mfcc: int) -> np.ndarray:
    n = x.shape[1]
    k = np.arange(n_mfcc).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return x @ basis.T


# ---------------------------------------------------------------------------
# eGeMAPS extractors — SMILExtract (preferred) + opensmile-python + precomputed
# ---------------------------------------------------------------------------
_SMILE_INSTANCE = None
SMILEXTRACT_BIN: Optional[Path] = None
SMILEXTRACT_CONF: Optional[Path] = None
SMILEXTRACT_SOURCE_INC: Optional[Path] = None
SMILEXTRACT_SINK_INC: Optional[Path] = None
EGEMAPS_BACKEND = "unknown"  # set by setup_egemaps_backend()


def _smile():
    global _SMILE_INSTANCE
    if _SMILE_INSTANCE is None and HAS_OPENSMILE:
        _SMILE_INSTANCE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE_INSTANCE


def find_smilextract() -> Optional[Path]:
    """Locate the on-phone SMILExtract binary built by build_opensmile.sh."""
    found = shutil.which("SMILExtract")
    if found:
        return Path(found)
    home_bin = Path.home() / ".local" / "bin" / "SMILExtract"
    if home_bin.exists():
        return home_bin
    local_bin = Path("./bin/SMILExtract")
    if local_bin.exists():
        return local_bin
    return None


def find_egemaps_conf(opensmile_src: Path = Path.home() / "opensmile-src") -> Optional[Path]:
    """Locate the standalone eGeMAPSv02.conf shipped with the openSMILE source build."""
    candidates = [
        opensmile_src / "config" / "egemaps" / "v02" / "eGeMAPSv02.conf",
        Path.home() / ".local" / "share" / "opensmile" / "eGeMAPSv02.conf",
        Path("./config/egemaps/v02/eGeMAPSv02.conf"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_source_sink_includes(opensmile_src: Path = Path.home() / "opensmile-src") -> tuple[Optional[Path], Optional[Path]]:
    """Locate the standard wave-input source.inc and data-output sink.inc.

    OpenSMILE 3.x's eGeMAPSv02.conf uses `\\{\\cm[source...]}` and
    `\\{\\cm[sink...]}` placeholders that have to be filled in via CLI
    flags `-source <file.inc>` and `-sink <file.inc>`. Without them
    SMILExtract silently no-ops.
    """
    source_candidates = [
        opensmile_src / "config" / "shared" / "standard_wave_input.conf.inc",
        Path.home() / ".local" / "share" / "opensmile" / "config" / "shared" / "standard_wave_input.conf.inc",
        Path("./config/shared/standard_wave_input.conf.inc"),
    ]
    sink_candidates = [
        opensmile_src / "config" / "shared" / "standard_data_output.conf.inc",
        Path.home() / ".local" / "share" / "opensmile" / "config" / "shared" / "standard_data_output.conf.inc",
        Path("./config/shared/standard_data_output.conf.inc"),
    ]
    src = next((c for c in source_candidates if c.exists()), None)
    sink = next((c for c in sink_candidates if c.exists()), None)
    return src, sink


def setup_egemaps_backend(precomputed_available: bool) -> str:
    """Decide which eGeMAPS extraction path to use; populate globals.

    Order of preference:
        1. on-phone SMILExtract binary  → backend "smilextract"
        2. opensmile-python (workstation only) → backend "opensmile_py"
        3. precomputed features in manifest  → backend "precomputed"
    Returns the backend name (also stored in EGEMAPS_BACKEND).
    """
    global SMILEXTRACT_BIN, SMILEXTRACT_CONF, SMILEXTRACT_SOURCE_INC, SMILEXTRACT_SINK_INC, EGEMAPS_BACKEND
    bin_ = find_smilextract()
    conf = find_egemaps_conf()
    src_inc, sink_inc = find_source_sink_includes()
    if bin_ is not None and conf is not None and src_inc is not None and sink_inc is not None:
        SMILEXTRACT_BIN = bin_
        SMILEXTRACT_CONF = conf
        SMILEXTRACT_SOURCE_INC = src_inc
        SMILEXTRACT_SINK_INC = sink_inc
        EGEMAPS_BACKEND = "smilextract"
    elif HAS_OPENSMILE:
        EGEMAPS_BACKEND = "opensmile_py"
    elif precomputed_available:
        EGEMAPS_BACKEND = "precomputed"
    else:
        EGEMAPS_BACKEND = "unavailable"
    return EGEMAPS_BACKEND


def extract_egemaps_smilextract(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Write a tempfile WAV, invoke SMILExtract -> CSV, parse 88-d feature.

    OpenSMILE 3.x eGeMAPSv02.conf uses `\\{\\cm[source...]}` and
    `\\{\\cm[sink...]}` placeholders — without `-source` and `-sink`
    flags pointing at the matching include files, SMILExtract silently
    no-ops (exit code 0, no output written).
    """
    assert SMILEXTRACT_BIN is not None and SMILEXTRACT_CONF is not None
    assert SMILEXTRACT_SOURCE_INC is not None and SMILEXTRACT_SINK_INC is not None
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_wav = td / "in.wav"
        out_csv = td / "out.csv"
        sf.write(str(in_wav), y.astype(np.float32, copy=False), sr)
        subprocess.run(
            [str(SMILEXTRACT_BIN),
             "-C", str(SMILEXTRACT_CONF),
             "-source", str(SMILEXTRACT_SOURCE_INC),
             "-sink",   str(SMILEXTRACT_SINK_INC),
             "-I", str(in_wav),
             "-O", str(out_csv),
             "-instname", "seg",
             "-loglevel", "0"],
            check=True, capture_output=True, timeout=60,
        )
        # SMILExtract CSV format: header rows + 1 data row
        # The last column is "class" or similar; first column is instance name.
        # Numeric features are the 88 functionals.
        with out_csv.open() as fh:
            data_row = None
            for line in fh:
                if line.startswith("'") or line.startswith("seg") or ("," in line and any(ch.isdigit() for ch in line)):
                    if "," in line and not line.strip().startswith("@"):
                        data_row = line.strip()
        if data_row is None:
            # Try a more permissive last-line parse
            with out_csv.open() as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            data_row = lines[-1] if lines else ""
        parts = data_row.split(",")
        # Strip leading instance name + any trailing label
        numeric = []
        for p in parts:
            try:
                numeric.append(float(p))
            except ValueError:
                pass
        return np.asarray(numeric, dtype=np.float32)


def extract_egemaps_opensmilepy(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    smile = _smile()
    feats = smile.process_signal(y, sr).values
    return feats.flatten().astype(np.float32)


def extract_egemaps(y: np.ndarray, sr: int = 16000,
                    precomputed: Optional[np.ndarray] = None) -> np.ndarray:
    if EGEMAPS_BACKEND == "smilextract":
        return extract_egemaps_smilextract(y, sr)
    if EGEMAPS_BACKEND == "opensmile_py":
        return extract_egemaps_opensmilepy(y, sr)
    if EGEMAPS_BACKEND == "precomputed":
        if precomputed is None:
            raise RuntimeError("eGeMAPS backend=precomputed but no precomputed vector was passed")
        return precomputed.astype(np.float32, copy=False)
    raise RuntimeError(f"eGeMAPS unavailable (backend={EGEMAPS_BACKEND})")


def smilextract_parity_check(segments: list[dict], audio_root: Path, n: int = 3) -> dict:
    """Verify SMILExtract output matches the manifest's precomputed features."""
    if EGEMAPS_BACKEND != "smilextract":
        return {"checked": False, "reason": f"backend={EGEMAPS_BACKEND}, parity check N/A"}
    tested = 0
    max_abs = 0.0
    samples = []
    for seg in segments:
        if "egemaps_features" not in seg:
            continue
        wav = audio_root / "seg_audio" / f"{seg['segment_id']}.wav"
        if not wav.exists():
            continue
        y = load_wav(wav)
        extracted = extract_egemaps_smilextract(y)
        precomputed = np.array(seg["egemaps_features"], dtype=np.float32)
        if extracted.shape != precomputed.shape:
            return {
                "checked": True, "passed": False,
                "reason": f"shape mismatch: extracted {extracted.shape} vs precomputed {precomputed.shape}",
                "segment_id": seg["segment_id"],
            }
        diff = float(np.max(np.abs(extracted - precomputed)))
        max_abs = max(max_abs, diff)
        samples.append({"segment_id": seg["segment_id"], "max_abs_diff": diff})
        tested += 1
        if tested >= n:
            break
    passed = max_abs < 1e-3
    return {
        "checked": True, "passed": passed,
        "tested": tested, "max_abs_diff": max_abs,
        "tolerance": 1e-3, "samples": samples,
    }


# ---------------------------------------------------------------------------
# Latency benchmark (per-segment + per-call)
# ---------------------------------------------------------------------------
def benchmark_segments(
    feature_type: str,
    model,
    segments: list[dict],
    audio_root: Path,
    n_reps: int,
    writer: csv.DictWriter,
    egemaps_backend: str,
) -> None:
    for seg_idx, seg in enumerate(segments):
        wav_path = audio_root / "seg_audio" / f"{seg['segment_id']}.wav"
        if not wav_path.exists():
            continue
        precomputed = (np.array(seg["egemaps_features"], dtype=np.float32)
                       if "egemaps_features" in seg else None)
        for rep in range(n_reps):
            t0 = time.perf_counter_ns()
            y = load_wav(wav_path)
            t_load = time.perf_counter_ns()

            if feature_type == "mfcc":
                feat = (extract_mfcc_librosa(y) if HAS_LIBROSA else extract_mfcc_numpy(y))
            else:
                feat = extract_egemaps(y, precomputed=precomputed)
            t_feat = time.perf_counter_ns()

            x = torch.from_numpy(feat).unsqueeze(0)
            with torch.no_grad():
                logit = model(x)
                p = torch.sigmoid(logit).item()
            t_fwd = time.perf_counter_ns()

            alpha = float(seg["alpha"])
            S = alpha * 0.0 + (1.0 - alpha) * p
            t_ema = time.perf_counter_ns()

            rss = read_rss_mb()
            row = {
                "kind": "segment",
                "feature_type": feature_type,
                "egemaps_backend": egemaps_backend if feature_type == "egemaps" else "",
                "segment_id": seg["segment_id"],
                "label": seg["label"],
                "rep": rep,
                "load_ms": (t_load - t0) / 1e6,
                "feat_ms": (t_feat - t_load) / 1e6,
                "fwd_ms": (t_fwd - t_feat) / 1e6,
                "ema_ms": (t_ema - t_fwd) / 1e6,
                "end_to_end_ms": (t_ema - t0) / 1e6,
                "prob": p,
                "score_S": S,
                "rss_mb": rss,
            }
            writer.writerow(row)


def benchmark_calls(
    feature_type: str,
    model,
    calls: list[dict],
    audio_root: Path,
    writer: csv.DictWriter,
    egemaps_backend: str,
) -> None:
    for call in calls:
        call_id = call["call_id"]
        wav_path = audio_root / "call_audio" / f"{call_id}.wav"
        if not wav_path.exists():
            continue
        segments = call["segments"]
        alpha = float(call["alpha"])
        tau = float(call["tau"])

        y_full = load_wav(wav_path)
        sr = 16000

        t0_call = time.perf_counter_ns()
        S = 0.0
        first_alert_ns: Optional[int] = None
        peak_seg_ns = 0
        for seg in segments:
            s = max(0, int(seg["start"] * sr))
            e = min(len(y_full), int(seg["end"] * sr))
            clip = y_full[s:e]
            if clip.size < 64:
                continue
            precomp = (np.array(seg["egemaps_features"], dtype=np.float32)
                       if "egemaps_features" in seg else None)
            seg_t0 = time.perf_counter_ns()
            if feature_type == "mfcc":
                feat = (extract_mfcc_librosa(clip) if HAS_LIBROSA else extract_mfcc_numpy(clip))
            else:
                feat = extract_egemaps(clip, precomputed=precomp)
            x = torch.from_numpy(feat).unsqueeze(0)
            with torch.no_grad():
                p = torch.sigmoid(model(x)).item()
            S = alpha * S + (1.0 - alpha) * p
            seg_t1 = time.perf_counter_ns()
            peak_seg_ns = max(peak_seg_ns, seg_t1 - seg_t0)
            if first_alert_ns is None and S >= tau:
                first_alert_ns = seg_t1 - t0_call
        t1_call = time.perf_counter_ns()

        row = {
            "kind": "call",
            "feature_type": feature_type,
            "egemaps_backend": egemaps_backend if feature_type == "egemaps" else "",
            "call_id": call_id,
            "label": call["label"],
            "total_call_ms": (t1_call - t0_call) / 1e6,
            "first_alert_ms": (first_alert_ns / 1e6) if first_alert_ns is not None else "",
            "peak_seg_ms": peak_seg_ns / 1e6,
            "n_segments": len(segments),
            "final_S": S,
            "tau": tau,
            "alpha": alpha,
        }
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Sustained-load (energy) benchmark
# ---------------------------------------------------------------------------
@dataclass
class EnergySample:
    t_s: float                  # elapsed seconds from block start
    n_inferences: int           # cumulative inference count at sample time
    pct: Optional[float] = None
    temp_c: Optional[float] = None
    voltage_mV: Optional[float] = None
    current_uA: Optional[float] = None
    charge_counter_uAh: Optional[float] = None
    rss_mb: float = float("nan")
    median_latency_ms_so_far: float = float("nan")


@dataclass
class EnergyBlock:
    feature_type: str
    duration_s: float
    n_inferences: int
    pre: dict = field(default_factory=dict)
    post: dict = field(default_factory=dict)
    samples: list[EnergySample] = field(default_factory=list)
    notes: str = ""

    def to_json_dict(self) -> dict:
        return {
            "feature_type": self.feature_type,
            "duration_s": self.duration_s,
            "n_inferences": self.n_inferences,
            "pre": self.pre,
            "post": self.post,
            "samples": [asdict(s) for s in self.samples],
            "notes": self.notes,
        }


def benchmark_energy(
    feature_type: str,
    model,
    segments: list[dict],
    audio_root: Path,
    duration_s: float,
    sample_interval_s: float,
    egemaps_backend: str,
) -> EnergyBlock:
    """Run continuous inference for `duration_s`, sampling battery + RSS periodically."""
    block = EnergyBlock(feature_type=feature_type, duration_s=duration_s, n_inferences=0)
    pre = read_battery() or {}
    block.pre = pre
    if pre and pre.get("status") == "CHARGING":
        block.notes = "ERROR: device was charging at block start"
        block.post = read_battery() or {}
        return block

    # Preload all segments' clips (cap to a reasonable working set) so we don't
    # measure disk I/O over and over; this mirrors the "audio chunk already in
    # memory" production case.
    preloaded: list[tuple[str, np.ndarray, Optional[np.ndarray]]] = []
    for seg in segments:
        wav = audio_root / "seg_audio" / f"{seg['segment_id']}.wav"
        if not wav.exists():
            continue
        precomp = (np.array(seg["egemaps_features"], dtype=np.float32)
                   if "egemaps_features" in seg else None)
        preloaded.append((seg["segment_id"], load_wav(wav), precomp))
    if not preloaded:
        block.notes = "no segments preloaded"
        return block

    latencies_ms: list[float] = []
    t_block_start = time.perf_counter()
    next_sample_at = sample_interval_s
    idx = 0
    while True:
        wall = time.perf_counter() - t_block_start
        if wall >= duration_s:
            break
        _, y, precomp = preloaded[idx % len(preloaded)]
        idx += 1
        t0 = time.perf_counter_ns()
        if feature_type == "mfcc":
            feat = (extract_mfcc_librosa(y) if HAS_LIBROSA else extract_mfcc_numpy(y))
        else:
            feat = extract_egemaps(y, precomputed=precomp)
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            torch.sigmoid(model(x)).item()
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)
        block.n_inferences += 1

        # Periodic battery sample
        if wall >= next_sample_at:
            bat = read_battery() or {}
            block.samples.append(EnergySample(
                t_s=wall,
                n_inferences=block.n_inferences,
                pct=bat.get("percentage"),
                temp_c=bat.get("temperature"),
                voltage_mV=bat.get("voltage"),
                current_uA=bat.get("current"),
                charge_counter_uAh=bat.get("charge_counter"),
                rss_mb=read_rss_mb(),
                median_latency_ms_so_far=float(np.median(latencies_ms[-1000:])),
            ))
            next_sample_at += sample_interval_s

    block.post = read_battery() or {}
    return block


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
SEGMENT_FIELDS = [
    "kind", "feature_type", "egemaps_backend",
    "segment_id", "label", "rep",
    "load_ms", "feat_ms", "fwd_ms", "ema_ms", "end_to_end_ms",
    "prob", "score_S", "rss_mb",
]
CALL_FIELDS = [
    "kind", "feature_type", "egemaps_backend",
    "call_id", "label",
    "total_call_ms", "first_alert_ms", "peak_seg_ms",
    "n_segments", "final_S", "tau", "alpha",
]


class TwoSchemaWriter:
    def __init__(self, fp):
        self.fp = fp
        self._all_fields = sorted(set(SEGMENT_FIELDS) | set(CALL_FIELDS))
        self.writer = csv.DictWriter(fp, fieldnames=self._all_fields, extrasaction="ignore")
        self.writer.writeheader()

    def writerow(self, row: dict) -> None:
        self.writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", default="both",
                        choices=["latency", "energy", "both"])
    parser.add_argument("--output", default="phone_results_v2.csv")
    parser.add_argument("--summary", default="phone_summary_v2.json")
    parser.add_argument("--energy_json", default="phone_energy.json")
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--feature_types", default="mfcc,egemaps")
    parser.add_argument("--package_root", default=".")
    parser.add_argument("--energy_duration", type=float, default=600.0,
                        help="Seconds of sustained inference per feature type (default 600 = 10 min)")
    parser.add_argument("--energy_sample_interval", type=float, default=30.0)
    parser.add_argument("--skip_parity", action="store_true",
                        help="Skip SMILExtract parity check (for emergency runs)")
    args = parser.parse_args()

    # The parity-check block below may demote EGEMAPS_BACKEND from "smilextract"
    # to "precomputed". Declare it global at the top so Python treats the
    # later assignment as a module-level rebind (not a local-only var).
    global EGEMAPS_BACKEND

    root = Path(args.package_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ Missing {manifest_path}", file=sys.stderr)
        return 1
    with manifest_path.open() as f:
        manifest = json.load(f)

    feature_types = [s.strip() for s in args.feature_types.split(",")]

    # Decide eGeMAPS backend (this may be demoted to "precomputed" below if
    # the SMILExtract parity check fails — see the parity block further down).
    precomp_available = ("egemaps" in feature_types) and \
        all("egemaps_features" in s for s in manifest["segments"])
    backend = setup_egemaps_backend(precomp_available)
    if "egemaps" in feature_types and backend == "unavailable":
        print("❌ eGeMAPS backend unavailable AND no precomputed features in manifest.",
              file=sys.stderr)
        return 1

    # Banner
    bat0 = read_battery() or {}
    print("=" * 72)
    print("Track B v2 — on-device deployment benchmark")
    print("=" * 72)
    print(f"python      : {sys.version.split()[0]}")
    print(f"platform    : {platform.platform()}")
    print(f"machine     : {platform.machine()}")
    print(f"torch       : {torch.__version__}")
    print(f"soundfile   : {'yes' if HAS_SOUNDFILE else 'no (FATAL: install libsndfile)'}")
    print(f"librosa     : {'yes' if HAS_LIBROSA else 'no (numpy MFCC fallback)'}")
    print(f"opensmile_py: {'yes' if HAS_OPENSMILE else 'no'}")
    print(f"SMILExtract : {SMILEXTRACT_BIN if SMILEXTRACT_BIN else 'not found'}")
    print(f"egemaps_conf: {SMILEXTRACT_CONF if SMILEXTRACT_CONF else 'not found'}")
    print(f"  -source   : {SMILEXTRACT_SOURCE_INC if SMILEXTRACT_SOURCE_INC else 'not found'}")
    print(f"  -sink     : {SMILEXTRACT_SINK_INC if SMILEXTRACT_SINK_INC else 'not found'}")
    print(f"egemaps_backend: {EGEMAPS_BACKEND}")
    print(f"termux-api  : {'yes' if has_termux_api() else 'no (energy mode disabled)'}")
    if bat0:
        print(f"battery_pre : pct={bat0.get('percentage')}  temp={bat0.get('temperature')}  status={bat0.get('status')}")
        print(f"              voltage={bat0.get('voltage')}  current={bat0.get('current')}  charge_counter={bat0.get('charge_counter')}")
    print(f"feature_types: {feature_types}")
    print(f"mode        : {args.mode}")
    print(f"n_reps      : {args.n_reps}")
    print(f"n_segments  : {len(manifest['segments'])}")
    print(f"n_calls     : {len(manifest['calls'])}")
    if args.mode in ("energy", "both"):
        print(f"energy_dur  : {args.energy_duration:.0f} s per feature")
        print(f"energy_iv   : {args.energy_sample_interval:.0f} s sample interval")
    print("=" * 72)

    # SMILExtract parity check (once, upfront). On failure, fall back to
    # precomputed eGeMAPS features instead of bailing — this lets the benchmark
    # still produce energy/memory/MFCC numbers when SMILExtract is misconfigured
    # (the openSMILE 3.x eGeMAPSv02.conf has no CLI sink shipped by default).
    parity = None
    if EGEMAPS_BACKEND == "smilextract" and not args.skip_parity:
        print("\nRunning SMILExtract parity check vs precomputed features...")
        try:
            parity = smilextract_parity_check(manifest["segments"], root / "audio", n=3)
        except subprocess.CalledProcessError as e:
            parity = {"checked": True, "passed": False,
                      "reason": f"SMILExtract exited {e.returncode}",
                      "stderr": (e.stderr or b"").decode(errors="replace")[:500]}
        except Exception as e:
            parity = {"checked": True, "passed": False,
                      "reason": f"{type(e).__name__}: {e}"}
        print(f"  parity: {parity}")
        if not parity.get("passed", False):
            if precomp_available:
                print("⚠️  SMILExtract parity failed; demoting eGeMAPS to precomputed features "
                      "(extraction time will not be measured on-phone — workstation reference "
                      "used in the manuscript).")
                EGEMAPS_BACKEND = "precomputed"
            else:
                print("❌ SMILExtract parity failed AND no precomputed features available.",
                      file=sys.stderr)
                return 1

    # Latency mode
    latency_checkpoints: list[dict] = []
    if args.mode in ("latency", "both"):
        print("\n--- LATENCY MODE ---")
        latency_checkpoints.append(memcpu_snapshot("start"))
        with open(args.output, "w", newline="") as fp:
            writer = TwoSchemaWriter(fp)
            for ft in feature_types:
                model_path = root / "models" / f"best_audio_model_{ft}.ptl"
                if not model_path.exists():
                    print(f"⚠️  Skipping {ft}: missing {model_path}")
                    continue
                print(f"\n[{ft}] loading model: {model_path}")
                t0 = time.time()
                model = torch.jit.load(str(model_path))
                model.eval()
                latency_checkpoints.append(memcpu_snapshot(f"{ft}/loaded"))
                print(f"  loaded in {(time.time()-t0)*1000:.1f} ms")
                print(f"  benchmarking {len(manifest['segments'])} segments × {args.n_reps} reps")
                t0 = time.time()
                benchmark_segments(
                    ft, model, manifest["segments"], root / "audio",
                    args.n_reps, writer, EGEMAPS_BACKEND,
                )
                latency_checkpoints.append(memcpu_snapshot(f"{ft}/segments_done"))
                print(f"  segments done in {time.time()-t0:.1f}s, rss={read_rss_mb():.1f} MB")
                print(f"  benchmarking {len(manifest['calls'])} full calls")
                t0 = time.time()
                benchmark_calls(
                    ft, model, manifest["calls"], root / "audio",
                    writer, EGEMAPS_BACKEND,
                )
                latency_checkpoints.append(memcpu_snapshot(f"{ft}/calls_done"))
                print(f"  calls done in {time.time()-t0:.1f}s, rss={read_rss_mb():.1f} MB")

    # Energy mode
    energy_blocks: list[dict] = []
    if args.mode in ("energy", "both"):
        print("\n--- ENERGY MODE (SUSTAINED LOAD) ---")
        if not has_termux_api():
            print("⚠️  termux-api not installed; energy mode will not capture battery data.")
        for ft in feature_types:
            model_path = root / "models" / f"best_audio_model_{ft}.ptl"
            if not model_path.exists():
                continue
            model = torch.jit.load(str(model_path))
            model.eval()
            print(f"\n[{ft}] sustained-load block: {args.energy_duration:.0f} s")
            block = benchmark_energy(
                ft, model, manifest["segments"], root / "audio",
                duration_s=args.energy_duration,
                sample_interval_s=args.energy_sample_interval,
                egemaps_backend=EGEMAPS_BACKEND,
            )
            blk_dict = block.to_json_dict()
            energy_blocks.append(blk_dict)
            pre_pct = (block.pre or {}).get("percentage")
            post_pct = (block.post or {}).get("percentage")
            pre_temp = (block.pre or {}).get("temperature")
            post_temp = (block.post or {}).get("temperature")
            delta_pct = (pre_pct - post_pct) if (pre_pct is not None and post_pct is not None) else None
            print(f"  done. n_inferences={block.n_inferences}, "
                  f"pct {pre_pct}→{post_pct} (Δ={delta_pct}), "
                  f"temp {pre_temp}→{post_temp}")
            if block.notes:
                print(f"  NOTES: {block.notes}")

        with open(args.energy_json, "w") as f:
            json.dump({"blocks": energy_blocks}, f, indent=2, default=str)
        print(f"\n✅ Wrote {args.energy_json}")

    # Summary
    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "has_soundfile": HAS_SOUNDFILE,
        "has_librosa": HAS_LIBROSA,
        "has_opensmile_py": HAS_OPENSMILE,
        "has_termux_api": has_termux_api(),
        "smilextract_bin": str(SMILEXTRACT_BIN) if SMILEXTRACT_BIN else None,
        "egemaps_conf": str(SMILEXTRACT_CONF) if SMILEXTRACT_CONF else None,
        "egemaps_backend": EGEMAPS_BACKEND,
        "smilextract_parity": parity,
        "feature_types": feature_types,
        "mode": args.mode,
        "n_reps": args.n_reps,
        "n_segments": len(manifest["segments"]),
        "n_calls": len(manifest["calls"]),
        "energy_duration_s": args.energy_duration if args.mode in ("energy", "both") else None,
        "energy_sample_interval_s": args.energy_sample_interval if args.mode in ("energy", "both") else None,
        "latency_checkpoints": latency_checkpoints,
        "battery_at_start": bat0,
        "battery_capacity_wh_assumed": 19.4,  # S24 Ultra: 5000 mAh × 3.88 V
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if args.mode in ("latency", "both"):
        print(f"✅ Wrote {args.output}")
    print(f"✅ Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
