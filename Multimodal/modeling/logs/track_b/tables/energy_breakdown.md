# Track B v2 — Sustained-Load Energy Breakdown

Battery-delta methodology. Assumed battery capacity: **19.40 Wh** (S24 Ultra: 5000 mAh × 3.88 V). Charge-counter-based numbers (when available) are more precise; percentage-delta numbers are 1%-quantized.

| Feature | Duration (s) | n_inferences | Δpct | ΔTemp (°C) | ΔE (Wh) | Avg power (mW) | mJ / inference |
|---|---|---|---|---|---|---|---|
| mfcc | 600 | 98213 | 3.00 | 1.50 | 0.5820 | 3492 | 21.333 |
| egemaps | 600 | 4735517 | 2.00 | 0.40 | 0.3880 | 2328 | 0.295 |

### Charge-counter cross-check (higher resolution)

| Feature | Δcharge (μAh) | ΔE (Wh, cc) | Avg power (mW, cc) | mJ / inference (cc) |
|---|---|---|---|---|
| mfcc | 138236 | 0.5279 | 3168 | 19.351 |
| egemaps | 143173 | 0.5536 | 3322 | 0.421 |
