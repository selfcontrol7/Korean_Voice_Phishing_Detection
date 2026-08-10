# Track B — On-Device Latency Summary

**Device.** `x86_64` on `Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39`, PyTorch `2.6.0+cu124`. librosa=available, opensmile=available.

Stages: `load` (audio load), `feat` (feature extract), `fwd` (TorchScript forward + sigmoid), `ema` (one EMA update), `end_to_end` (sum, equals wall-clock per segment).

| Feature | Stage | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|
| egemaps | audio load | 0.89 | 1.57 | 0.52 | 2.43 | 7.27 |
| egemaps | feature extract | 99.82 | 127.89 | 89.94 | 167.65 | 207.65 |
| egemaps | TorchScript fwd | 10.88 | 5.66 | 8.78 | 22.48 | 24.45 |
| egemaps | EMA update | 0.01 | 0.06 | 0.00 | 0.02 | 0.05 |
| egemaps | end-to-end | 111.61 | 128.90 | 102.26 | 179.21 | 218.14 |
| mfcc | audio load | 0.73 | 0.88 | 0.52 | 1.76 | 3.66 |
| mfcc | feature extract | 78.59 | 130.55 | 70.48 | 115.75 | 130.47 |
| mfcc | TorchScript fwd | 40.76 | 14.73 | 38.49 | 65.71 | 75.51 |
| mfcc | EMA update | 0.01 | 0.02 | 0.00 | 0.00 | 0.01 |
| mfcc | end-to-end | 120.08 | 135.98 | 109.96 | 167.55 | 193.92 |

## Per-call (50 full vishing calls)

| Feature | Total call (s) mean ± std | Time to first alert (s) mean ± std | Peak per-segment latency (ms) |
|---|---|---|---|
| egemaps | 3.67 ± 6.63 | 0.09 ± 0.04 | 149.13 ± 25.62 |
| mfcc | 4.55 ± 8.00 | 0.10 ± 0.04 | 156.20 ± 25.70 |

*Streaming pipeline cap: the per-segment latency must stay under the segment audio duration (typically 5–10 s) for the detector to keep up with real-time playback. The reported numbers above are well within that envelope.*
