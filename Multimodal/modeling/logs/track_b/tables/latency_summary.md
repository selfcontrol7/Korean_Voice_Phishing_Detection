# Track B v2 — Latency Summary (Per-segment)

**Device.** `aarch64` on `Android-16-aarch64-64bit-ELF`, PyTorch `2.11.0`. eGeMAPS backend = `precomputed`. 

| Feature | Stage | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|---|---|---|
| egemaps | audio load | 0.39 | 0.31 | 0.36 | 1.06 | 1.60 | 1000 |
| egemaps | feature extract | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1000 |
| egemaps | TorchScript fwd | 0.07 | 0.07 | 0.05 | 0.13 | 0.52 | 1000 |
| egemaps | EMA update | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1000 |
| egemaps | end-to-end | 0.47 | 0.34 | 0.41 | 1.21 | 1.81 | 1000 |
| mfcc | audio load | 0.45 | 0.48 | 0.40 | 1.10 | 2.01 | 1000 |
| mfcc | feature extract | 4.19 | 2.29 | 4.15 | 8.03 | 9.60 | 1000 |
| mfcc | TorchScript fwd | 0.38 | 2.16 | 0.22 | 0.83 | 1.51 | 1000 |
| mfcc | EMA update | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1000 |
| mfcc | end-to-end | 5.03 | 3.54 | 4.92 | 9.45 | 11.60 | 1000 |

## Per-call (50 full vishing calls)

| Feature | Total call (s) mean ± std | Time to first alert (s) mean ± std | Peak per-segment (ms) mean ± std |
|---|---|---|---|
| egemaps | 0.00 ± 0.01 | 0.00 ± 0.00 | 0.39 ± 0.57 |
| mfcc | 0.22 ± 0.37 | 0.01 ± 0.00 | 11.50 ± 4.55 |
