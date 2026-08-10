# Per-Source Confusion (Test Set)

Sliced from Track A's call-level test results. The corpus has no FSS-non-vishing or AllHub-vishing samples, so per-source FPR within FSS and per-source recall within AllHub are mathematically undefined and are not reported here.

| Feature | Method | Source | Label (within source) | n calls | predicted vishing | Metric | Value | misclassified |
|---|---|---|---|---|---|---|---|---|
| egemaps | ema (α=0.3, τ=0.3) | FSS | 1 | 71 | 71 | recall (TPR) | 1.0 | 0 |
| egemaps | ema (α=0.3, τ=0.3) | AllHub | 0 | 71 | 1 | specificity (TNR) = 1 − FPR | 0.9859 | 1 |
| egemaps | running_max (τ=0.9) | FSS | 1 | 71 | 71 | recall (TPR) | 1.0 | 0 |
| egemaps | running_max (τ=0.9) | AllHub | 0 | 71 | 2 | specificity (TNR) = 1 − FPR | 0.9718 | 2 |
| mfcc | ema (α=0.3, τ=0.3) | FSS | 1 | 71 | 71 | recall (TPR) | 1.0 | 0 |
| mfcc | ema (α=0.3, τ=0.3) | AllHub | 0 | 71 | 1 | specificity (TNR) = 1 − FPR | 0.9859 | 1 |
| mfcc | running_max (τ=0.3) | FSS | 1 | 71 | 71 | recall (TPR) | 1.0 | 0 |
| mfcc | running_max (τ=0.3) | AllHub | 0 | 71 | 6 | specificity (TNR) = 1 − FPR | 0.9155 | 6 |
| wav2vec2 | ema (α=0.5, τ=0.7) | FSS | 1 | 71 | 66 | recall (TPR) | 0.9296 | 5 |
| wav2vec2 | ema (α=0.5, τ=0.7) | AllHub | 0 | 71 | 1 | specificity (TNR) = 1 − FPR | 0.9859 | 1 |
| wav2vec2 | running_max (τ=0.9) | FSS | 1 | 71 | 70 | recall (TPR) | 0.9859 | 1 |
| wav2vec2 | running_max (τ=0.9) | AllHub | 0 | 71 | 14 | specificity (TNR) = 1 − FPR | 0.8028 | 14 |
| all | ema (α=0.3, τ=0.3) | FSS | 1 | 71 | 71 | recall (TPR) | 1.0 | 0 |
| all | ema (α=0.3, τ=0.3) | AllHub | 0 | 71 | 0 | specificity (TNR) = 1 − FPR | 1.0 | 0 |
| all | running_max (τ=0.5) | FSS | 1 | 71 | 71 | recall (TPR) | 1.0 | 0 |
| all | running_max (τ=0.5) | AllHub | 0 | 71 | 1 | specificity (TNR) = 1 − FPR | 0.9859 | 1 |

**Note on what we can and cannot compute:** With the FSS source containing only vishing calls (706) and the AllHub source containing only non-vishing calls (711), the source × label table has empty off-diagonal cells. We can report recall on the FSS slice and specificity on the AllHub slice — both of which equal the corpus-wide metrics modulo rounding — but we cannot test whether the classifier would identify a vishing call from AllHub or a non-vishing call from FSS, because no such samples exist. This is the central reason Phase B (band-ablation) and Phase C (codec augmentation) are needed.
