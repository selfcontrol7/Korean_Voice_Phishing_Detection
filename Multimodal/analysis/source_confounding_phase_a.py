"""Track C, Phase A — channel-statistics probe.

Compute 11 hand-crafted, content-agnostic channel features per segment
(see analysis/channel_stats.py). Train a small MLP probe on val to
predict source/label, evaluate on test. Generate a per-feature
distribution figure (FSS vs AllHub histograms) so we can see *which*
channel cues differ.

If the probe achieves high accuracy, the existing vishing classifier
*could* be using channel cues rather than vishing-specific signal.
This is the qualitative half of Track C; the quantitative half lives
in Phases B (band-ablation) and C (codec augmentation).

Run from the Multimodal/ directory (with the vishing venv active):
    python analysis/source_confounding_phase_a.py
    python analysis/source_confounding_phase_a.py --workers 8 --skip-extract
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_stats import compute_channel_features, FEATURE_NAMES, N_FEATURES  # noqa: E402

OUT_DIR = Path("modeling/logs/track_c/phase_a")
MANIFESTS = {
    "val": Path("data/val_segment_manifest_merged.jsonl"),
    "test": Path("data/test_segment_manifest_merged.jsonl"),
}


# -----------------------------------------------------------------------------
# Audio path resolution + per-segment feature extraction
# -----------------------------------------------------------------------------
def audio_path_for(call_id: str) -> Path:
    if call_id.startswith("vishing_"):
        return Path("data/audio/vishing") / f"{call_id}.wav"
    return Path("data/audio/non_vishing") / f"{call_id}.wav"


def _process_call(args: tuple) -> tuple[str, list[tuple[int, np.ndarray]]]:
    """Worker: load one audio file, compute channel features for all its segments.

    Returns (call_id, [(global_idx, feature_vec), ...]).
    Returning indices preserves manifest order without requiring a global lock.
    """
    call_id, segments_with_idx = args
    path = audio_path_for(call_id)
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception as e:
        print(f"  ⚠️  failed to load {path}: {e}", file=sys.stderr)
        return call_id, [(idx, np.zeros(N_FEATURES, dtype=np.float32))
                         for idx, _, _ in segments_with_idx]

    out: list[tuple[int, np.ndarray]] = []
    for global_idx, start, end in segments_with_idx:
        s = max(0, int(start * sr))
        e = min(len(y), int(end * sr))
        clip = y[s:e]
        feats = compute_channel_features(clip, sr=sr)
        out.append((global_idx, feats))
    return call_id, out


def extract_features_for_split(split: str, manifest_path: Path, workers: int) -> dict:
    """Compute channel features for every segment in a split.

    Streams entries in manifest order, groups by call_id so each audio
    file is loaded exactly once, then re-orders results back to manifest
    order. Saves features.npy, labels.npy, sources.npy.
    """
    print(f"\n[{split}] reading {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    n = len(entries)
    print(f"  {n} segments")

    # Group segments by call_id, preserving manifest order via global_idx
    by_call: dict[str, list[tuple[int, float, float]]] = {}
    for global_idx, e in enumerate(entries):
        by_call.setdefault(e["call_id"], []).append((global_idx, e["start"], e["end"]))
    print(f"  {len(by_call)} unique call_ids")

    features = np.zeros((n, N_FEATURES), dtype=np.float32)
    labels = np.array([e["label"] for e in entries], dtype=np.int64)
    sources = np.array([e.get("source", "unknown") for e in entries], dtype=object)

    work = list(by_call.items())
    t0 = time.time()
    if workers <= 1:
        for i, item in enumerate(work):
            call_id, results = _process_call(item)
            for idx, vec in results:
                features[idx] = vec
            if (i + 1) % 25 == 0 or (i + 1) == len(work):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(work) - i - 1) / rate if rate > 0 else 0
                print(f"  [{split}] {i + 1}/{len(work)} calls "
                      f"({rate:.2f} calls/s, ETA {eta:.0f}s)")
    else:
        with mp.Pool(processes=workers) as pool:
            done = 0
            for call_id, results in pool.imap_unordered(_process_call, work, chunksize=4):
                for idx, vec in results:
                    features[idx] = vec
                done += 1
                if done % 25 == 0 or done == len(work):
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(work) - done) / rate if rate > 0 else 0
                    print(f"  [{split}] {done}/{len(work)} calls "
                          f"({rate:.2f} calls/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"channel_features_{split}.npy", features)
    np.save(OUT_DIR / f"channel_labels_{split}.npy", labels)
    np.save(OUT_DIR / f"channel_sources_{split}.npy", sources)
    print(f"  saved: {OUT_DIR}/channel_{{features,labels,sources}}_{split}.npy")
    return {"split": split, "n": n, "n_calls": len(by_call), "elapsed_s": elapsed}


# -----------------------------------------------------------------------------
# Probe MLP
# -----------------------------------------------------------------------------
class ProbeMLP(nn.Module):
    """Same architecture as AudioClassifier (input → 256 → 1) for fair comparison."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    epochs: int = 30,
    lr: float = 2e-4,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[ProbeMLP, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalize features (z-score) using train stats
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    Xt = ((X_train - mu) / sigma).astype(np.float32)
    Xv = ((X_val - mu) / sigma).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(Xt),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(Xv),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ProbeMLP(input_dim=Xt.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_f1 = -1.0
    best_state = None
    patience = 5
    bad = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        # Val
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                p = torch.sigmoid(model(x)).cpu().numpy()
                all_p.extend(p.tolist())
                all_y.extend(y.numpy().tolist())
        preds = (np.array(all_p) >= 0.5).astype(int)
        f1 = f1_score(all_y, preds, zero_division=0)
        sched.step(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, {
        "best_val_f1": float(best_f1),
        "input_dim": int(Xt.shape[1]),
        "epochs_run": epoch + 1,
        "feature_mu": mu.flatten().tolist(),
        "feature_sigma": sigma.flatten().tolist(),
    }


def evaluate_probe(model, X, y, mu, sigma, device, batch_size=64):
    Xn = ((X - mu) / sigma).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_p, all_y = [], []
    with torch.no_grad():
        for x, yb in loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            all_p.extend(p.tolist())
            all_y.extend(yb.numpy().tolist())
    preds = (np.array(all_p) >= 0.5).astype(int)
    cm = confusion_matrix(all_y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(all_y, preds)),
        "f1": float(f1_score(all_y, preds, zero_division=0)),
        "precision": float(precision_score(all_y, preds, zero_division=0)),
        "recall": float(recall_score(all_y, preds, zero_division=0)),
        "fpr": float(fp) / max(1, (fp + tn)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# -----------------------------------------------------------------------------
# Distribution figure
# -----------------------------------------------------------------------------
def plot_distributions(
    features_val: np.ndarray, sources_val: np.ndarray,
    features_test: np.ndarray, sources_test: np.ndarray,
    fig_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    feats = np.concatenate([features_val, features_test], axis=0)
    srcs = np.concatenate([sources_val, sources_test], axis=0)

    n = N_FEATURES
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows))
    axes = axes.flatten()

    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        x_fss = feats[srcs == "FSS", i]
        x_all = feats[srcs == "AllHub", i]
        # Robust shared bins
        lo = float(np.percentile(feats[:, i], 1))
        hi = float(np.percentile(feats[:, i], 99))
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 50)
        ax.hist(x_all, bins=bins, alpha=0.55, label="AllHub (label=0)", color="C0", density=True)
        ax.hist(x_fss, bins=bins, alpha=0.55, label="FSS (label=1)", color="C3", density=True)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(N_FEATURES, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Channel-feature distributions on val + test segments\n"
        "(content-agnostic; if FSS and AllHub differ widely, the vishing classifier could exploit channel cues)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_path, format="pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=8, help="Multiprocess workers for feature extraction")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Re-use channel_features_*.npy from disk instead of re-computing")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {Path.cwd()}", file=sys.stderr)

    extraction_summary = {}
    if not args.skip_extract:
        for split, path in MANIFESTS.items():
            extraction_summary[split] = extract_features_for_split(split, path, args.workers)
    else:
        print("[skip-extract] reusing existing channel_features_*.npy")

    X_val = np.load(OUT_DIR / "channel_features_val.npy")
    y_val = np.load(OUT_DIR / "channel_labels_val.npy")
    s_val = np.load(OUT_DIR / "channel_sources_val.npy", allow_pickle=True)
    X_test = np.load(OUT_DIR / "channel_features_test.npy")
    y_test = np.load(OUT_DIR / "channel_labels_test.npy")
    s_test = np.load(OUT_DIR / "channel_sources_test.npy", allow_pickle=True)
    print(f"\nLoaded: val {X_val.shape}, test {X_test.shape}")

    # Train probe on val (we don't use train for the probe — we want to keep
    # train un-touched for the band-ablation Phase B retraining; val is large
    # enough at 4061 samples to fit a ~3K-param probe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("\n=== Training probe MLP on val features → predict source/label ===")
    model, train_info = train_probe(
        X_val, y_val, X_val, y_val, device=device, epochs=args.epochs, seed=args.seed
    )
    mu = np.array(train_info["feature_mu"])
    sigma = np.array(train_info["feature_sigma"])
    val_metrics = evaluate_probe(model, X_val, y_val, mu, sigma, device)
    test_metrics = evaluate_probe(model, X_test, y_test, mu, sigma, device)

    # Single-feature univariate F1 (which feature alone is most discriminative?)
    # Using a logistic-style threshold sweep on val for each feature.
    univariate = {}
    for i, name in enumerate(FEATURE_NAMES):
        x_val_i = X_val[:, i]
        # Threshold sweep using percentiles
        best = {"f1": 0.0, "thr": 0.0, "direction": "+"}
        for q in np.linspace(5, 95, 19):
            thr = float(np.percentile(x_val_i, q))
            for direction in ("+", "-"):
                pred = (x_val_i >= thr).astype(int) if direction == "+" else (x_val_i < thr).astype(int)
                f = f1_score(y_val, pred, zero_division=0)
                if f > best["f1"]:
                    best = {"f1": float(f), "thr": thr, "direction": direction}
        # Apply chosen threshold to test
        x_test_i = X_test[:, i]
        if best["direction"] == "+":
            pred_test = (x_test_i >= best["thr"]).astype(int)
        else:
            pred_test = (x_test_i < best["thr"]).astype(int)
        univariate[name] = {
            "val_f1": best["f1"],
            "val_threshold": best["thr"],
            "direction": best["direction"],
            "test_f1": float(f1_score(y_test, pred_test, zero_division=0)),
            "test_acc": float(accuracy_score(y_test, pred_test)),
        }

    # Save full probe results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "extraction_summary": extraction_summary,
        "probe": {
            "input_dim": train_info["input_dim"],
            "best_val_f1_during_training": train_info["best_val_f1"],
            "epochs_run": train_info["epochs_run"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        "univariate_per_feature": univariate,
        "feature_names": FEATURE_NAMES,
    }
    with (OUT_DIR / "probe_results.json").open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_DIR}/probe_results.json")

    # Distribution figure
    plot_distributions(X_val, s_val, X_test, s_test, OUT_DIR / "feature_distributions.pdf")
    print(f"Saved {OUT_DIR}/feature_distributions.pdf")

    print("\n=== Channel-stat probe on test set ===")
    print(f"  Multivariate MLP (11-dim → 256 → 1):")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"     {k:10s} {v:.4f}")
        else:
            print(f"     {k:10s} {v}")
    print(f"  Top 3 single-feature discriminators (test F1):")
    top3 = sorted(univariate.items(), key=lambda kv: kv[1]["test_f1"], reverse=True)[:3]
    for name, info in top3:
        print(f"     {name:30s} test F1={info['test_f1']:.4f}  (val thr={info['val_threshold']:.3f}, dir {info['direction']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
