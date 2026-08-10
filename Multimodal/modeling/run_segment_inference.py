"""Run a trained audio classifier over a manifest and emit per-segment predictions.

Track A.2 of the Paper 1 IEEE Access revision. The downstream call-level
aggregation needs raw segment-level probabilities tied to (call_id,
segment_id, start, label, source) so it can group, sort, and apply EMA /
running-max rules. Existing scripts (train_audio_baseline.py,
evaluate_summary.py) report aggregated metrics but do not persist
per-segment predictions linked to their IDs, so we add this dedicated
inference utility.

Determinism: model is in eval mode, no dropout, and the DataLoader runs
with shuffle=False, so predictions[i] aligns 1-to-1 with the i-th line
of the input manifest. We hard-assert this length match.

Run from the Multimodal/ directory:
    python modeling/run_segment_inference.py \\
        --checkpoint modeling/models/best_audio_model_egemaps.pth \\
        --manifest data/test_segment_manifest_merged.jsonl \\
        --feature_type egemaps \\
        --output modeling/logs/track_a/segment_predictions/test_egemaps.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audio_only_dataloader import AudioOnlyDataset


# Mirrored verbatim from modeling/train_audio_baseline.py (lines 23-34).
# Inlined here to avoid importing train_audio_baseline, which pulls in
# seaborn at module load time. The architecture must match the one used
# during training or state_dict loading will fail loudly.
class AudioClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def run_inference(
    checkpoint_path: Path,
    manifest_path: Path,
    feature_type: str,
    output_path: Path,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    dataset = AudioOnlyDataset(str(manifest_path), feature_type)
    if len(dataset) != len(entries):
        raise RuntimeError(
            f"Dataset length {len(dataset)} != manifest line count {len(entries)} — "
            f"the dataloader is not iterating manifest entries 1-to-1."
        )

    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[0]
    model = AudioClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs: list[float] = []
    t0 = time.time()
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            probs = torch.sigmoid(model(features)).cpu().numpy()
            all_probs.extend(probs.tolist())
    elapsed = time.time() - t0

    if len(all_probs) != len(entries):
        raise RuntimeError(
            f"Prediction count {len(all_probs)} != manifest entry count {len(entries)} — "
            f"alignment broken; refusing to write a misaligned predictions file."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_label1 = 0
    prob_sum = 0.0
    with output_path.open("w", encoding="utf-8") as fout:
        for entry, prob in zip(entries, all_probs):
            row = {
                "call_id": entry["call_id"],
                "segment_id": entry["segment_id"],
                "start": entry["start"],
                "end": entry["end"],
                "label": entry["label"],
                "source": entry.get("source", "unknown"),
                "prob": float(prob),
                "feature_type": feature_type,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_label1 += int(entry["label"] == 1)
            prob_sum += float(prob)

    summary = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "feature_type": feature_type,
        "output": str(output_path),
        "input_dim": int(input_dim),
        "n_segments": len(entries),
        "n_label1": n_label1,
        "n_label0": len(entries) - n_label1,
        "mean_prob": prob_sum / len(entries) if entries else 0.0,
        "device": str(device),
        "inference_seconds": round(elapsed, 2),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["mfcc", "egemaps", "wav2vec2", "all"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if unset)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cwd = Path.cwd()
    if cwd.name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {cwd}", file=sys.stderr)

    run_inference(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        feature_type=args.feature_type,
        output_path=args.output,
        device=device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
