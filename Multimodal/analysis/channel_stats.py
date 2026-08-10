"""Hand-crafted, content-agnostic channel-level acoustic features.

These features describe the *recording channel* — bandwidth, spectral
shape, energy distribution — without encoding what was actually said.
They exist for Track C Phase A: if a probe MLP can separate FSS from
AllHub using only these 11 features, then a vishing classifier could
in principle exploit channel cues rather than vishing content.

Public surface:
    compute_channel_features(waveform, sr=16000) -> np.ndarray  # shape (11,)
    FEATURE_NAMES                                                 # length 11
    self_test()                                                   # invariants
"""
from __future__ import annotations

import numpy as np

# Stable order of features. Used for figure titles, CSV headers, etc.
FEATURE_NAMES: list[str] = [
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff95_mean",
    "spectral_flatness_mean",
    "zero_crossing_rate_mean",
    "rms_mean",
    "energy_ratio_low_0_1khz",
    "energy_ratio_mid_1_4khz",
    "energy_ratio_high_4_8khz",
]
N_FEATURES = len(FEATURE_NAMES)

# STFT params are matched to librosa defaults so other parts of the codebase
# (audio_features.py uses n_fft=400, hop=160 for MFCC) and these stats are
# computed at comparable time resolutions.
_N_FFT = 1024
_HOP = 256


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


def _band_energy_ratio(power_spec: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    """Fraction of total power in [low, high) Hz, averaged across frames."""
    in_band = (freqs >= low) & (freqs < high)
    band_power = power_spec[in_band, :].sum(axis=0)
    total_power = power_spec.sum(axis=0) + 1e-12
    ratio = band_power / total_power
    return float(np.mean(ratio))


def compute_channel_features(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Return an 11-dim feature vector describing channel-level acoustics.

    Local librosa imports keep import time small for callers that only
    need FEATURE_NAMES. Empty / silent / very short waveforms degrade to
    zeros gracefully (no NaNs, no exceptions).
    """
    import librosa

    if waveform.size < _N_FFT:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # librosa expects float
    y = waveform.astype(np.float32, copy=False)

    # Frame-level features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=_N_FFT, hop_length=_HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=_N_FFT, hop_length=_HOP)[0]
    rms = librosa.feature.rms(y=y, frame_length=_N_FFT, hop_length=_HOP)[0]

    # Power spectrogram for energy-band ratios
    spec = np.abs(librosa.stft(y=y, n_fft=_N_FFT, hop_length=_HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)

    return np.array(
        [
            _safe_mean(centroid),
            _safe_std(centroid),
            _safe_mean(bandwidth),
            _safe_std(bandwidth),
            _safe_mean(rolloff),
            _safe_mean(flatness),
            _safe_mean(zcr),
            _safe_mean(rms),
            _band_energy_ratio(spec, freqs, 0.0, 1000.0),
            _band_energy_ratio(spec, freqs, 1000.0, 4000.0),
            _band_energy_ratio(spec, freqs, 4000.0, 8000.0),
        ],
        dtype=np.float32,
    )


def self_test() -> None:
    """Sanity checks. Run with `python -m analysis.channel_stats` from Multimodal/."""
    sr = 16000
    duration = 2.0
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Pure 1 kHz tone
    tone_low = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    feats_low = compute_channel_features(tone_low, sr)
    assert 800.0 < feats_low[0] < 1200.0, f"1 kHz tone centroid out of range: {feats_low[0]}"

    # Pure 6 kHz tone
    tone_high = 0.5 * np.sin(2 * np.pi * 6000.0 * t)
    feats_high = compute_channel_features(tone_high, sr)
    assert 5500.0 < feats_high[0] < 6500.0, f"6 kHz tone centroid out of range: {feats_high[0]}"

    # White noise should have rolloff_95 near sr/2 * 0.95 ≈ 7.6 kHz
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n).astype(np.float32) * 0.1
    feats_noise = compute_channel_features(noise, sr)
    assert 6500.0 < feats_noise[4] < 8000.0, f"White-noise rolloff out of range: {feats_noise[4]}"

    # Energy-band ratios sum to ≈ 1.0 for any waveform that fits within 0–8 kHz at sr=16k
    total = feats_noise[8] + feats_noise[9] + feats_noise[10]
    assert abs(total - 1.0) < 0.01, f"Energy ratios sum to {total}, expected ≈ 1.0"

    # High-band ratio is much smaller for the 1 kHz tone than for white noise
    assert feats_low[10] < 0.05, f"1 kHz tone has unexpectedly high HF energy: {feats_low[10]}"
    assert feats_noise[10] > 0.3, f"White noise has too little HF energy: {feats_noise[10]}"

    # Empty / very short waveform → zeros, no exceptions
    feats_empty = compute_channel_features(np.zeros(10, dtype=np.float32), sr)
    assert np.all(feats_empty == 0.0)

    print("✅ channel_stats self-test passed")


if __name__ == "__main__":
    self_test()
