"""Apply a telephone-band bandpass filter (300–3400 Hz) to a waveform.

Track C Phase B helper. Forward-applies the telephone-channel bandwidth
constraint to full-band 16 kHz audio so we can re-extract features
under matched bandwidth conditions for both FSS and AllHub sources.

The filter is a 4th-order Butterworth bandpass implemented as
second-order sections, applied with `sosfiltfilt` for zero-phase
distortion (preserves segment timing / hop count).

Public surface:
    apply_bandpass(waveform, sr=16000, low=300, high=3400) -> np.ndarray
    self_test()  # invariants
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _design_bandpass(low: float, high: float, sr: int, order: int = 4):
    nyq = 0.5 * sr
    sos = signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sos


def apply_bandpass(
    waveform: np.ndarray,
    sr: int = 16000,
    low: float = 300.0,
    high: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """Return a band-limited copy of `waveform` (300–3400 Hz by default).

    Returns float32. Length is preserved (filtfilt does no decimation).
    """
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    sos = _design_bandpass(low, high, sr, order=order)
    y = signal.sosfiltfilt(sos, waveform.astype(np.float64, copy=False))
    return y.astype(np.float32, copy=False)


def self_test() -> None:
    sr = 16000
    n = sr * 2  # 2 seconds
    t = np.arange(n) / sr

    # 1 kHz tone — should pass through with similar amplitude
    tone_pass = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    out_pass = apply_bandpass(tone_pass, sr=sr)
    rms_in = float(np.sqrt(np.mean(tone_pass ** 2)))
    rms_out = float(np.sqrt(np.mean(out_pass ** 2)))
    ratio_db = 20.0 * np.log10(max(rms_out, 1e-12) / max(rms_in, 1e-12))
    assert ratio_db > -3.0, f"1 kHz tone attenuated by {ratio_db:.2f} dB (expected ≥ -3 dB)"

    # 6 kHz tone — well above 3.4 kHz, should be heavily attenuated
    tone_block = 0.5 * np.sin(2 * np.pi * 6000.0 * t)
    out_block = apply_bandpass(tone_block, sr=sr)
    rms_block_out = float(np.sqrt(np.mean(out_block ** 2)))
    block_ratio_db = 20.0 * np.log10(max(rms_block_out, 1e-12) / max(rms_in, 1e-12))
    assert block_ratio_db < -30.0, f"6 kHz tone only attenuated by {block_ratio_db:.2f} dB (expected ≤ -30 dB)"

    # 100 Hz tone — below 300 Hz, should be heavily attenuated
    tone_low = 0.5 * np.sin(2 * np.pi * 100.0 * t)
    out_low = apply_bandpass(tone_low, sr=sr)
    rms_low_out = float(np.sqrt(np.mean(out_low ** 2)))
    low_ratio_db = 20.0 * np.log10(max(rms_low_out, 1e-12) / max(rms_in, 1e-12))
    assert low_ratio_db < -20.0, f"100 Hz tone only attenuated by {low_ratio_db:.2f} dB (expected ≤ -20 dB)"

    # Length is preserved
    assert out_pass.shape == tone_pass.shape

    # Empty waveform handled
    out_empty = apply_bandpass(np.zeros(0, dtype=np.float32))
    assert out_empty.shape == (0,)

    print(f"✅ apply_telephony_bandpass self-test passed "
          f"(1kHz {ratio_db:+.1f} dB | 100Hz {low_ratio_db:+.1f} dB | 6kHz {block_ratio_db:+.1f} dB)")


if __name__ == "__main__":
    self_test()
