"""Track B step 1 — Export trained MFCC and eGeMAPS classifiers to TorchScript.

Loads `.pth` checkpoints, instantiates `AudioClassifier`, scripts the
model, applies `optimize_for_mobile`, saves a `.ptl` file ready to be
loaded on Android via `torch.jit.load`.

Self-validates each exported model against the original `.pth` on a
random batch (max abs diff < 1e-5).

Output:
    analysis/phone_package/models/best_audio_model_{mfcc,egemaps}.ptl

Run from the Multimodal/ directory:
    python analysis/track_b_export_torchscript.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile


CHECKPOINTS = {
    "mfcc":    ("modeling/models/best_audio_model_mfcc.pth",    13),
    "egemaps": ("modeling/models/best_audio_model_egemaps.pth", 88),
}
OUT_DIR = Path("analysis/phone_package/models")


# Mirrored from modeling/train_audio_baseline.py — kept here so this script
# has no dependency on a module that imports seaborn.
class AudioClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def export_one(name: str, ckpt_path: Path, input_dim: int) -> Path:
    print(f"\n[{name}]")
    print(f"  loading {ckpt_path}  (input_dim={input_dim})")
    model = AudioClassifier(input_dim=input_dim)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Reference output for parity check
    rng = np.random.default_rng(seed=0)
    x_np = rng.standard_normal((8, input_dim)).astype(np.float32)
    x = torch.from_numpy(x_np)
    with torch.no_grad():
        ref_out = model(x).numpy()

    # Script + mobile-optimize
    scripted = torch.jit.script(model)
    mobile = optimize_for_mobile(scripted)

    out_path = OUT_DIR / f"best_audio_model_{name}.ptl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mobile._save_for_lite_interpreter(str(out_path))

    # Validate — load fresh as if on the phone and compare
    loaded = torch.jit.load(str(out_path))
    loaded.eval()
    with torch.no_grad():
        mob_out = loaded(x).numpy()
    diff = float(np.max(np.abs(ref_out - mob_out)))
    if diff > 1e-5:
        print(f"  ✗ output mismatch: max abs diff {diff:.2e}", file=sys.stderr)
        return Path()
    print(f"  ✓ saved {out_path}  ({out_path.stat().st_size/1024:.1f} KiB)")
    print(f"  ✓ parity vs .pth: max abs diff {diff:.2e}")
    return out_path


def main() -> int:
    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)
        return 1

    exported: list[Path] = []
    for name, (ckpt, dim) in CHECKPOINTS.items():
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"❌ missing checkpoint: {ckpt_path}", file=sys.stderr)
            return 1
        out = export_one(name, ckpt_path, dim)
        if not out.name:
            return 1
        exported.append(out)

    print(f"\n✅ Exported {len(exported)} TorchScript models to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
