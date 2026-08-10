"""Round-trip a waveform through the GSM (libgsm) telephone codec.

Track C Phase C helper. Simulates "what if this audio had also gone
through a phone channel" by encoding to 8 kHz GSM and decoding back to
16 kHz mono. Used to channel-match AllHub non-vishing recordings to the
FSS phone-channel side, then re-evaluate the existing classifier on the
matched test set.

Public surface:
    apply_codec(waveform, sr=16000) -> np.ndarray   # ffmpeg-libgsm path
    apply_codec_fallback(waveform, sr=16000) -> np.ndarray  # bandpass+resample
    self_test()
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_FFMPEG = shutil.which("ffmpeg")


def _ffmpeg_available() -> bool:
    return _FFMPEG is not None


def _libgsm_available() -> bool:
    if not _ffmpeg_available():
        return False
    try:
        out = subprocess.run(
            [_FFMPEG, "-hide_banner", "-codecs"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except Exception:
        return False
    # libgsm shows up as both an encoder and decoder
    return "libgsm" in out


def apply_codec(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Encode → decode via ffmpeg + libgsm. Falls back if either is missing."""
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    if not _libgsm_available():
        return apply_codec_fallback(waveform, sr=sr)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_wav = td / "in.wav"
        gsm_path = td / "mid.gsm"
        out_wav = td / "out.wav"
        sf.write(str(in_wav), waveform.astype(np.float32, copy=False), sr)

        # Encode: input.wav (any sr) → mid.gsm (forced 8 kHz mono libgsm)
        try:
            subprocess.run(
                [_FFMPEG, "-y", "-loglevel", "error",
                 "-i", str(in_wav),
                 "-ar", "8000", "-ac", "1", "-c:a", "libgsm", str(gsm_path)],
                check=True, timeout=120,
            )
        except subprocess.CalledProcessError:
            return apply_codec_fallback(waveform, sr=sr)

        # Decode: mid.gsm → out.wav at original sr, mono
        try:
            subprocess.run(
                [_FFMPEG, "-y", "-loglevel", "error",
                 "-i", str(gsm_path),
                 "-ar", str(sr), "-ac", "1", str(out_wav)],
                check=True, timeout=120,
            )
        except subprocess.CalledProcessError:
            return apply_codec_fallback(waveform, sr=sr)

        y, _ = sf.read(str(out_wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

    # Pad/trim to original length so segment slicing remains aligned
    n = waveform.size
    if y.size < n:
        y = np.pad(y, (0, n - y.size), mode="constant")
    elif y.size > n:
        y = y[:n]
    return y.astype(np.float32, copy=False)


def apply_codec_fallback(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Bandpass to 300–3400 Hz + downsample to 8 kHz + upsample back.

    Used when ffmpeg/libgsm isn't available. Captures the bandwidth
    limitation but lacks the lossy nonlinear codec compression.
    """
    from preprocessing.apply_telephony_bandpass import apply_bandpass
    import librosa

    bp = apply_bandpass(waveform, sr=sr)
    # Downsample to 8 kHz
    down = librosa.resample(bp, orig_sr=sr, target_sr=8000, res_type="kaiser_fast")
    # Upsample back to original sr
    up = librosa.resample(down, orig_sr=8000, target_sr=sr, res_type="kaiser_fast")

    n = waveform.size
    if up.size < n:
        up = np.pad(up, (0, n - up.size), mode="constant")
    elif up.size > n:
        up = up[:n]
    return up.astype(np.float32, copy=False)


def self_test() -> None:
    sr = 16000
    n = sr * 2
    t = np.arange(n) / sr

    # Clean tone within telephone band
    signal = 0.3 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    out = apply_codec(signal, sr=sr)
    assert out.dtype == np.float32, out.dtype
    # Length is preserved (after pad/trim)
    assert out.shape == signal.shape, f"length changed: {signal.shape} -> {out.shape}"
    # Round-trip RMS within 6 dB of input (codec is lossy)
    rms_in = float(np.sqrt(np.mean(signal ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    db = 20.0 * np.log10(max(rms_out, 1e-9) / max(rms_in, 1e-9))
    assert -10.0 < db < 6.0, f"1 kHz tone RMS shift {db:.2f} dB out of expected ±6 dB band"

    # 6 kHz tone should be heavily attenuated (above the 4 kHz Nyquist of 8 kHz codec)
    high = 0.3 * np.sin(2 * np.pi * 6000.0 * t).astype(np.float32)
    out_high = apply_codec(high, sr=sr)
    rms_high = float(np.sqrt(np.mean(out_high ** 2)))
    db_high = 20.0 * np.log10(max(rms_high, 1e-9) / max(rms_in, 1e-9))
    assert db_high < -20.0, f"6 kHz tone only attenuated by {db_high:.2f} dB"

    # Empty input handled
    empty = apply_codec(np.zeros(0, dtype=np.float32), sr=sr)
    assert empty.shape == (0,)

    backend = "ffmpeg+libgsm" if _libgsm_available() else "fallback (bandpass+resample)"
    print(f"✅ apply_telephony_codec self-test passed [backend: {backend}] "
          f"(1kHz {db:+.1f} dB | 6kHz {db_high:+.1f} dB)")


if __name__ == "__main__":
    self_test()
