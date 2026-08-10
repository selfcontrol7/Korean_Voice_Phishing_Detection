# Track D — 5-Fold Cross-Validation Summary

Aggregated mean ± std across 5 stratified-grouped folds. Per-fold breakdown in `cv_per_fold.md`.

**Protocol.** sklearn `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the 1417-call corpus, with each fold's call set further split into ~90% train / ~10% val by `StratifiedGroupKFold(n_splits=10)`. All segments of a call are kept together to prevent leakage. Each fold trains the audio classifier from scratch with the same hyperparameters as the original Track A baseline. Hyperparameters for the EMA/Running-max aggregation are independently re-tuned on each fold's val set.

| Feature | Method | F1 | P | R | FPR | Mean alert (s) |
|---|---|---|---|---|---|---|
| egemaps | ema | 0.9820 ± 0.0103 | 0.9928 ± 0.0088 | 0.9717 ± 0.0181 | 0.0070 ± 0.0086 | 8.87 ± 3.75 |
| egemaps | running_max | 0.9645 ± 0.0088 | 0.9485 ± 0.0202 | 0.9816 ± 0.0163 | 0.0535 ± 0.0232 | 7.50 ± 0.48 |
| egemaps | any_segment_baseline (τ=0.5) | 0.9480 ± 0.0101 | 0.9085 ± 0.0219 | 0.9915 ± 0.0117 | 0.0999 ± 0.0271 | 7.87 ± 1.43 |
| egemaps | majority_vote (τ=0.5) | 0.9877 ± 0.0095 | 1.0000 ± 0.0000 | 0.9759 ± 0.0185 | 0.0000 ± 0.0000 | 251.52 ± 21.97 |
| mfcc | ema | 0.9825 ± 0.0042 | 0.9723 ± 0.0083 | 0.9929 ± 0.0000 | 0.0281 ± 0.0086 | 7.85 ± 0.95 |
| mfcc | running_max | 0.9759 ± 0.0137 | 0.9571 ± 0.0272 | 0.9957 ± 0.0063 | 0.0450 ± 0.0306 | 7.76 ± 0.96 |
| mfcc | any_segment_baseline (τ=0.5) | 0.9553 ± 0.0083 | 0.9158 ± 0.0161 | 0.9986 ± 0.0032 | 0.0914 ± 0.0194 | 7.73 ± 0.99 |
| mfcc | majority_vote (τ=0.5) | 0.9929 ± 0.0050 | 0.9972 ± 0.0063 | 0.9887 ± 0.0095 | 0.0028 ± 0.0063 | 249.37 ± 22.96 |
| wav2vec2 | ema | 0.9428 ± 0.0189 | 0.9409 ± 0.0404 | 0.9475 ± 0.0498 | 0.0618 ± 0.0463 | 19.27 ± 1.16 |
| wav2vec2 | running_max | 0.8522 ± 0.0150 | 0.7509 ± 0.0258 | 0.9858 ± 0.0123 | 0.3263 ± 0.0495 | 10.99 ± 0.70 |
| wav2vec2 | any_segment_baseline (τ=0.5) | 0.7126 ± 0.0112 | 0.5536 ± 0.0135 | 1.0000 ± 0.0000 | 0.8016 ± 0.0441 | 7.78 ± 0.29 |
| wav2vec2 | majority_vote (τ=0.5) | 0.9765 ± 0.0131 | 0.9583 ± 0.0239 | 0.9957 ± 0.0063 | 0.0436 ± 0.0255 | 248.72 ± 22.16 |
| all | ema | 0.9908 ± 0.0053 | 0.9874 ± 0.0076 | 0.9943 ± 0.0059 | 0.0127 ± 0.0077 | 7.65 ± 0.97 |
| all | running_max | 0.9711 ± 0.0131 | 0.9494 ± 0.0272 | 0.9943 ± 0.0092 | 0.0535 ± 0.0310 | 7.60 ± 0.97 |
| all | any_segment_baseline (τ=0.5) | 0.9632 ± 0.0122 | 0.9306 ± 0.0234 | 0.9986 ± 0.0032 | 0.0746 ± 0.0267 | 7.71 ± 0.97 |
| all | majority_vote (τ=0.5) | 0.9950 ± 0.0041 | 1.0000 ± 0.0000 | 0.9901 ± 0.0081 | 0.0000 ± 0.0000 | 249.06 ± 21.61 |
