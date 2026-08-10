# Track C Phase B — Telephone-Band Ablation: Before/After

Comparison of call-level F1 between the original full-band model (Track A) and the band-limited (300–3400 Hz telephone band) retrained model (Track C Phase B).

**Interpretation:** a large negative ΔF1 indicates the original score relied on wide-band channel cues that telephone-band transmission would remove. A small ΔF1 indicates the model relies on signal that survives band-limiting.

| Feature | Method | F1 (orig) | F1 (band-lim) | ΔF1 | P (orig) | P (band-lim) | R (orig) | R (band-lim) | FPR (orig) | FPR (band-lim) |
|---|---|---|---|---|---|---|---|---|---|---|
| egemaps | ema | 0.9930 | 0.9929 | -0.0001 | 0.9861 | 1.0000 | 1.0000 | 0.9859 | 0.0141 | 0.0000 |
| egemaps | running_max | 0.9861 | 0.9150 | -0.0711 | 0.9726 | 0.8537 | 1.0000 | 0.9859 | 0.0282 | 0.1690 |
| mfcc | ema | 0.9930 | 0.9412 | -0.0518 | 0.9861 | 0.9846 | 1.0000 | 0.9014 | 0.0141 | 0.0141 |
| mfcc | running_max | 0.9595 | 0.8947 | -0.0647 | 0.9221 | 0.8395 | 1.0000 | 0.9577 | 0.0845 | 0.1831 |
| wav2vec2 | ema | 0.9565 | 0.9859 | +0.0294 | 0.9851 | 0.9859 | 0.9296 | 0.9859 | 0.0141 | 0.0141 |
| wav2vec2 | running_max | 0.9032 | 0.8383 | -0.0649 | 0.8333 | 0.7292 | 0.9859 | 0.9859 | 0.1972 | 0.3662 |
| all | ema | 1.0000 | 0.9929 | -0.0071 | 1.0000 | 1.0000 | 1.0000 | 0.9859 | 0.0000 | 0.0000 |
| all | running_max | 0.9930 | 0.9524 | -0.0406 | 0.9861 | 0.9211 | 1.0000 | 0.9859 | 0.0141 | 0.0845 |

*Hyperparameters were independently re-tuned on the validation set for each setting (full-band vs. band-limited), so the α and τ values may differ between rows of the same feature type.*
