# Track B v2 — On-Device Deployment Summary

**Device.** aarch64 / Android-16-aarch64-64bit-ELF / PyTorch 2.11.0 / TorchScript Mobile.

Reviewer R1 asked for empirical mobile latency. Reviewer R3 asked for runtime, latency, and energy. The table below reports all three plus memory in a single paper-ready view.

| Feature | End-to-end latency (ms) | Memory peak (MB) | Energy / inference (mJ) | Avg power (mW) | ΔTemp (°C) |
|---|---|---|---|---|---|
| egemaps | 0.47 ± 0.34 (med 0.41, p95 1.21) | 236.0 | 0.295 | 2328 | 0.40 |
| mfcc | 5.03 ± 3.54 (med 4.92, p95 9.45) | 236.6 | 21.333 | 3492 | 1.50 |

*Latency reported as mean ± std with median and p95 over 1000 timings per feature type. Energy from a 10-min sustained-load block per feature (charger unplugged).*
