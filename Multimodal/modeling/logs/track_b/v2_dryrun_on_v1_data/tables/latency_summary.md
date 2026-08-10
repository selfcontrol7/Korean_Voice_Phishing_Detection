# Track B v2 — Latency Summary (Per-segment)

**Device.** `aarch64` on `Android-16-aarch64-64bit-ELF`, PyTorch `2.11.0`. eGeMAPS backend = `None`. 

| Feature | Stage | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | p99 (ms) | n |
|---|---|---|---|---|---|---|---|
| egemaps | audio load | 0.29 | 0.15 | 0.31 | 0.53 | 0.60 | 1000 |
| egemaps | feature extract | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 1000 |
| egemaps | TorchScript fwd | 0.05 | 0.03 | 0.04 | 0.07 | 0.11 | 1000 |
| egemaps | EMA update | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1000 |
| egemaps | end-to-end | 0.34 | 0.16 | 0.35 | 0.60 | 0.71 | 1000 |
| mfcc | audio load | 0.97 | 1.96 | 0.40 | 6.25 | 9.65 | 1000 |
| mfcc | feature extract | 4.72 | 2.57 | 4.73 | 9.29 | 11.22 | 1000 |
| mfcc | TorchScript fwd | 0.31 | 2.29 | 0.15 | 0.82 | 1.53 | 1000 |
| mfcc | EMA update | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1000 |
| mfcc | end-to-end | 6.01 | 4.45 | 5.68 | 12.93 | 17.26 | 1000 |

## Per-call (50 full vishing calls)

| Feature | Total call (s) mean ± std | Time to first alert (s) mean ± std | Peak per-segment (ms) mean ± std |
|---|---|---|---|
| egemaps | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.42 ± 0.44 |
| mfcc | 0.20 ± 0.36 | 0.01 ± 0.00 | 9.68 ± 2.23 |
