# Track C Phase C — Codec-Matched Channel: Before/After

Comparison of call-level F1 between the original full-band test (Track A) and the matched-channel test (Track C Phase C) where non-vishing audio is GSM-codec round-tripped to simulate a phone channel. The same full-band-trained classifier is used in both settings; only the inputs differ.

**Interpretation:** a large negative ΔF1 indicates the classifier had been exploiting channel-artifact differences between FSS and AllHub. A small ΔF1 indicates the classifier discriminates on signal that survives the channel match.


| Feature  | Method      | F1 (orig) | F1 (codec) | ΔF1     | P (orig) | P (codec) | R (orig) | R (codec) | FPR (orig) | FPR (codec) |
| -------- | ----------- | --------- | ---------- | ------- | -------- | --------- | -------- | --------- | ---------- | ----------- |
| egemaps  | ema         | 0.9930    | 0.6667     | -0.3263 | 0.9861   | 0.5000    | 1.0000   | 1.0000    | 0.0141     | 1.0000      |
| egemaps  | running_max | 0.9861    | 0.6667     | -0.3194 | 0.9726   | 0.5000    | 1.0000   | 1.0000    | 0.0282     | 1.0000      |
| mfcc     | ema         | 0.9930    | 0.6860     | -0.3070 | 0.9861   | 0.5221    | 1.0000   | 1.0000    | 0.0141     | 0.9155      |
| mfcc     | running_max | 0.9595    | 0.6730     | -0.2865 | 0.9221   | 0.5071    | 1.0000   | 1.0000    | 0.0845     | 0.9718      |
| wav2vec2 | ema         | 0.9565    | 0.9565     | +0.0000 | 0.9851   | 0.9851    | 0.9296   | 0.9296    | 0.0141     | 0.0141      |
| wav2vec2 | running_max | 0.9032    | 0.8434     | -0.0599 | 0.8333   | 0.7368    | 0.9859   | 0.9859    | 0.1972     | 0.3521      |
| all      | ema         | 1.0000    | 0.6762     | -0.3238 | 1.0000   | 0.5108    | 1.0000   | 1.0000    | 0.0000     | 0.9577      |
| all      | running_max | 0.9930    | 0.6667     | -0.3263 | 0.9861   | 0.5000    | 1.0000   | 1.0000    | 0.0141     | 1.0000      |


