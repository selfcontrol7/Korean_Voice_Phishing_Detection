# Track B v2 — Memory Footprint

Resident set size (VmRSS) sampled at every per-segment timing row. Reports per-feature mean and peak during the latency benchmark.

| Feature | RSS mean (MB) | RSS std (MB) | RSS peak (p99, MB) | n samples |
|---|---|---|---|---|
| egemaps | 235.5 | 0.3 | 236.0 | 1000 |
| mfcc | 236.1 | 0.5 | 236.6 | 1000 |

## Latency-mode checkpoints (cold→warm)

| Checkpoint | RSS (MB) | VmHWM (MB) | CPU (s) |
|---|---|---|---|
| start | 223.9 | 227.3 | 2.12 |
| mfcc/loaded | 227.3 | 227.3 | 2.14 |
| mfcc/segments_done | 236.0 | 246.5 | 8.59 |
| mfcc/calls_done | 235.6 | 519.6 | 23.90 |
| egemaps/loaded | 235.2 | 519.6 | 23.90 |
| egemaps/segments_done | 235.2 | 519.6 | 25.63 |
| egemaps/calls_done | 235.2 | 519.6 | 27.04 |
