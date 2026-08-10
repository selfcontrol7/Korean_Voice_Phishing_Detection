# Track B v2 — On-Device Deployment Summary

**Device.** aarch64 / Android-16-aarch64-64bit-ELF / PyTorch 2.11.0 / TorchScript Mobile.

Reviewer R1 asked for empirical mobile latency. Reviewer R3 asked for runtime, latency, and energy. The table below reports all three plus memory in a single paper-ready view.

| Feature | End-to-end latency (ms) | Memory peak (MB) | Energy / inference (mJ) | Avg power (mW) | ΔTemp (°C) |
|---|---|---|---|---|---|
| egemaps | 0.34 ± 0.16 (med 0.35, p95 0.60) | nan | nan | nan | nan |
| mfcc | 6.01 ± 4.45 (med 5.68, p95 12.93) | nan | nan | nan | nan |

*Latency reported as mean ± std with median and p95 over 1000 timings per feature type. Energy from a 10-min sustained-load block per feature (charger unplugged).*
