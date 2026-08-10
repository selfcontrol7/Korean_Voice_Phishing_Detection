"""Sweep, evaluate, and report call-level aggregation results.

Track A.4 of the Paper 1 IEEE Access revision. Orchestrates:
  Phase B — hyperparameter sweep on val for EMA and running-max
            (paper-faithful formulas; see call_level_aggregation.py).
  Phase C — test evaluation with the val-picked best configs.
  Phase D — baseline comparisons (any-segment ≥ 0.5; majority-vote).
  Phase E — write CSV + Markdown summary table; plot alert-latency CDF
            and EMA (α, τ) heatmaps.

Inputs: predictions JSONLs produced by run_segment_inference.py.
Outputs: under modeling/logs/track_a/{call_level, tables, figures}.

Run from the Multimodal/ directory:
    python modeling/evaluate_call_level.py
    python modeling/evaluate_call_level.py --feature_types egemaps,mfcc
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from call_level_aggregation import (
    CallResult,
    aggregate_all,
    load_predictions,
)

# ----- Paths (defaults; overridable via --pred_dir / --out_dir) -----
PRED_DIR = Path("modeling/logs/track_a/segment_predictions")
OUT_DIR = Path("modeling/logs/track_a")
SWEEP_DIR = OUT_DIR / "call_level"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"


def _set_paths(pred_dir: Path, out_dir: Path) -> None:
    """Override the module-level paths so Track C can reuse this script."""
    global PRED_DIR, OUT_DIR, SWEEP_DIR, TABLE_DIR, FIG_DIR
    PRED_DIR = pred_dir
    OUT_DIR = out_dir
    SWEEP_DIR = OUT_DIR / "call_level"
    TABLE_DIR = OUT_DIR / "tables"
    FIG_DIR = OUT_DIR / "figures"

# ----- Hyperparameter grid (matches plan) -----
ALPHAS = [0.3, 0.5, 0.7, 0.9]
TAUS = [0.3, 0.5, 0.7, 0.9]

FEATURE_TYPES = ["egemaps", "mfcc", "wav2vec2", "all"]


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def call_metrics(results: list[CallResult]) -> dict:
    y_true = np.array([r.label for r in results])
    y_pred = np.array([r.final_decision for r in results])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # Degenerate case (single-class)
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "fpr": 0.0}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fp) / max(1, (fp + tn)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def alert_latency_stats(results: list[CallResult]) -> dict:
    """Latency stats over true-positive vishing calls that alerted."""
    tps = [r for r in results if r.label == 1 and r.final_decision == 1 and r.alert_time is not None]
    n_vishing = sum(1 for r in results if r.label == 1)
    if not tps:
        return {
            "n_alerted_tp": 0,
            "n_vishing": int(n_vishing),
            "frac_alerted_before_end": 0.0,
            "mean_alert_latency": float("nan"),
            "median_alert_latency": float("nan"),
            "p95_alert_latency": float("nan"),
        }
    latencies = np.array([r.alert_time for r in tps])
    early = sum(1 for r in tps if r.alert_time < r.duration)
    return {
        "n_alerted_tp": int(len(tps)),
        "n_vishing": int(n_vishing),
        "frac_alerted_before_end": float(early) / len(tps),
        "mean_alert_latency": float(latencies.mean()),
        "median_alert_latency": float(np.median(latencies)),
        "p95_alert_latency": float(np.percentile(latencies, 95)),
    }


# -----------------------------------------------------------------------------
# Phase B + C: sweep on val, evaluate on test
# -----------------------------------------------------------------------------
def sweep_val(by_call_val: dict[str, list[dict]]) -> list[dict]:
    """Run the full hyperparameter grid on val. Returns rows of metrics."""
    rows: list[dict] = []
    for alpha in ALPHAS:
        for tau in TAUS:
            results = aggregate_all(by_call_val, "ema", alpha=alpha, tau=tau)
            m = call_metrics(results)
            lat = alert_latency_stats(results)
            rows.append({"method": "ema", "alpha": alpha, "tau": tau, **m, **lat})
    for tau in TAUS:
        results = aggregate_all(by_call_val, "running_max", tau=tau)
        m = call_metrics(results)
        lat = alert_latency_stats(results)
        rows.append({"method": "running_max", "alpha": None, "tau": tau, **m, **lat})
    return rows


def pick_best(sweep_rows: list[dict], method: str) -> dict:
    candidates = [r for r in sweep_rows if r["method"] == method]
    return max(candidates, key=lambda r: r["f1"])


def evaluate_on_test(
    by_call_test: dict[str, list[dict]], method: str, *, alpha=None, tau: float
) -> tuple[list[CallResult], dict, dict]:
    results = aggregate_all(by_call_test, method, alpha=alpha, tau=tau)
    return results, call_metrics(results), alert_latency_stats(results)


# -----------------------------------------------------------------------------
# Phase E: outputs
# -----------------------------------------------------------------------------
def write_sweep_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_summary(table: list[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow(row)

    # Markdown table for direct paste
    cols = [
        "feature_type", "method", "alpha", "tau",
        "f1", "precision", "recall", "fpr",
        "mean_alert_latency", "median_alert_latency",
        "frac_alerted_before_end", "n_alerted_tp", "n_vishing",
    ]
    headers = {
        "feature_type": "Feature",
        "method": "Method",
        "alpha": "α",
        "tau": "τ",
        "f1": "F1",
        "precision": "P",
        "recall": "R",
        "fpr": "FPR",
        "mean_alert_latency": "Mean alert (s)",
        "median_alert_latency": "Median alert (s)",
        "frac_alerted_before_end": "% TP early",
        "n_alerted_tp": "TP alerted",
        "n_vishing": "Vishing calls",
    }
    with md_path.open("w") as f:
        f.write("# Call-Level Evaluation (Test Set)\n\n")
        f.write("Configurations selected to maximize F1 on the validation set.\n\n")
        f.write("| " + " | ".join(headers[c] for c in cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for row in table:
            cells = []
            for c in cols:
                v = row.get(c, "")
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    if c == "frac_alerted_before_end":
                        cells.append(f"{v*100:.1f}%")
                    elif c in ("mean_alert_latency", "median_alert_latency"):
                        cells.append(f"{v:.2f}" if not np.isnan(v) else "—")
                    else:
                        cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")


_COLOR_MAP = {"egemaps": "C0", "mfcc": "C1", "wav2vec2": "C2", "all": "C3"}
_MARKER_MAP = {"egemaps": "o", "mfcc": "s", "wav2vec2": "^", "all": "D"}


def _curve_for(payload: dict) -> tuple[np.ndarray, np.ndarray] | None:
    results = payload["results"]
    tps = [r for r in results if r.label == 1 and r.final_decision == 1 and r.alert_time is not None]
    if not tps:
        return None
    latencies = sorted(r.alert_time for r in tps)
    xs = np.array(latencies)
    ys = np.linspace(1.0 / len(xs), 1.0, len(xs))
    return xs, ys


def plot_alert_latency_cdf(per_feature_test_results: dict, fig_path: Path) -> None:
    """Variant: two-panel empirical CDF (EMA | Running-Max).

    The single-panel form was unreadable because eGeMAPS, MFCC, and
    All-features alert at nearly identical timestamps, so 6 of 8 curves
    coincided pixel-for-pixel. Two panels + step plots + colored markers
    expose overlap as marker density.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    methods = [("ema", "EMA (best α, τ on val)"), ("running_max", "Running-max (best τ on val)")]

    x_max = 0.0
    for ax, (method, title) in zip(axes, methods):
        for ft in ["egemaps", "mfcc", "wav2vec2", "all"]:
            payload = per_feature_test_results.get(ft, {}).get(method)
            if payload is None:
                continue
            curve = _curve_for(payload)
            if curve is None:
                continue
            xs, ys = curve
            x_max = max(x_max, float(xs.max()))
            color = _COLOR_MAP.get(ft, "k")
            ax.step(xs, ys, where="post", color=color, linewidth=1.6, alpha=0.85, label=ft)
            ax.scatter(xs, ys, color=color, marker=_MARKER_MAP.get(ft, "o"),
                       s=14, alpha=0.55, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("Alert latency (seconds from call start)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9, loc="lower right", title="Feature")

    axes[0].set_ylabel("Fraction of TP vishing calls alerted")
    for ax in axes:
        ax.set_xlim(0, max(50.0, x_max * 1.05))
    fig.suptitle("Alert-Latency CDF on Test (call-level aggregation)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path, format="pdf")
    plt.close(fig)


def plot_alert_latency_cdf_simplified(
    per_feature_test_results: dict, best_configs: dict, fig_path: Path
) -> None:
    """Variant A: single panel, one curve per feature using its winning method.

    For each feature_type, pick the method (EMA vs Running-max) with the
    higher val F1, then plot just those 4 curves. Each legend entry
    includes the chosen method and its hyperparameters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    x_max = 0.0
    for ft in ["egemaps", "mfcc", "wav2vec2", "all"]:
        cfg = best_configs.get(ft, {})
        ema_f1 = cfg.get("ema", {}).get("f1", -1)
        rm_f1 = cfg.get("running_max", {}).get("f1", -1)
        method = "ema" if ema_f1 >= rm_f1 else "running_max"

        payload = per_feature_test_results.get(ft, {}).get(method)
        if payload is None:
            continue
        curve = _curve_for(payload)
        if curve is None:
            continue
        xs, ys = curve
        x_max = max(x_max, float(xs.max()))

        if method == "ema":
            label = (
                f"{ft} (EMA, α={cfg['ema']['alpha']}, τ={cfg['ema']['tau']})"
            )
        else:
            label = f"{ft} (Running-max, τ={cfg['running_max']['tau']})"

        color = _COLOR_MAP.get(ft, "k")
        ax.step(xs, ys, where="post", color=color, linewidth=1.8, alpha=0.85, label=label)
        ax.scatter(xs, ys, color=color, marker=_MARKER_MAP.get(ft, "o"),
                   s=18, alpha=0.55, edgecolors="none")

    ax.set_xlabel("Alert latency (seconds from call start)")
    ax.set_ylabel("Fraction of TP vishing calls alerted")
    ax.set_title("Alert-Latency CDF — best aggregation per feature (val-tuned)")
    ax.set_xlim(0, max(50.0, x_max * 1.05))
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path, format="pdf")
    plt.close(fig)


def plot_alert_latency_cdf_annotated(
    per_feature_test_results: dict, fig_path: Path
) -> None:
    """Variant B: two-panel CDF with an explicit annotation about overlap.

    Same data as `plot_alert_latency_cdf` but adds a small caption note
    inside the EMA panel calling out that eGeMAPS / MFCC / All-features
    curves are operationally coincident on the fast portion.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    methods = [("ema", "EMA (best α, τ on val)"), ("running_max", "Running-max (best τ on val)")]

    x_max = 0.0
    for ax, (method, title) in zip(axes, methods):
        for ft in ["egemaps", "mfcc", "wav2vec2", "all"]:
            payload = per_feature_test_results.get(ft, {}).get(method)
            if payload is None:
                continue
            curve = _curve_for(payload)
            if curve is None:
                continue
            xs, ys = curve
            x_max = max(x_max, float(xs.max()))
            color = _COLOR_MAP.get(ft, "k")
            ax.step(xs, ys, where="post", color=color, linewidth=1.6, alpha=0.85, label=ft)
            ax.scatter(xs, ys, color=color, marker=_MARKER_MAP.get(ft, "o"),
                       s=16, alpha=0.55, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("Alert latency (seconds from call start)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9, loc="lower right", title="Feature")

    # Inline annotation on the EMA panel (top-right; legend lives bottom-right)
    axes[0].text(
        0.97, 0.97,
        "Note: eGeMAPS / MFCC / All-features\ncurves are nearly coincident — they\nalert at the same segment boundaries\non 70/71 vishing calls.",
        transform=axes[0].transAxes,
        fontsize=8, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )

    axes[0].set_ylabel("Fraction of TP vishing calls alerted")
    for ax in axes:
        ax.set_xlim(0, max(50.0, x_max * 1.05))
    fig.suptitle("Alert-Latency CDF on Test (call-level aggregation)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path, format="pdf")
    plt.close(fig)


def plot_alpha_tau_heatmap(sweep_rows: list[dict], feature_type: str, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.full((len(ALPHAS), len(TAUS)), np.nan)
    for r in sweep_rows:
        if r["method"] != "ema":
            continue
        i = ALPHAS.index(r["alpha"])
        j = TAUS.index(r["tau"])
        grid[i, j] = r["f1"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, vmin=max(0.0, np.nanmin(grid) - 0.02), vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(TAUS)))
    ax.set_xticklabels([f"{t}" for t in TAUS])
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_yticklabels([f"{a}" for a in ALPHAS])
    ax.set_xlabel("τ (alert threshold)")
    ax.set_ylabel("α (smoothing factor)")
    ax.set_title(f"EMA F1 surface — {feature_type}")
    for i in range(len(ALPHAS)):
        for j in range(len(TAUS)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v < 0.7 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Val F1")
    fig.tight_layout()
    fig.savefig(fig_path, format="pdf")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Per-feature pipeline
# -----------------------------------------------------------------------------
def process_feature_type(feature_type: str) -> tuple[list[dict], dict]:
    val_path = PRED_DIR / f"val_{feature_type}.jsonl"
    test_path = PRED_DIR / f"test_{feature_type}.jsonl"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing predictions for {feature_type}: {val_path} or {test_path} "
            f"— run modeling/run_segment_inference.py first."
        )

    print(f"\n=== feature_type={feature_type} ===")
    by_call_val = load_predictions(val_path)
    by_call_test = load_predictions(test_path)
    print(f"  val: {len(by_call_val)} calls, "
          f"{sum(len(v) for v in by_call_val.values())} segments")
    print(f"  test: {len(by_call_test)} calls, "
          f"{sum(len(v) for v in by_call_test.values())} segments")

    # Phase B — sweep on val
    sweep_rows = sweep_val(by_call_val)
    write_sweep_csv(sweep_rows, SWEEP_DIR / f"sweep_val_{feature_type}.csv")

    best_ema = pick_best(sweep_rows, "ema")
    best_rm = pick_best(sweep_rows, "running_max")
    print(f"  best EMA on val: α={best_ema['alpha']}, τ={best_ema['tau']}, F1={best_ema['f1']:.4f}")
    print(f"  best running-max on val: τ={best_rm['tau']}, F1={best_rm['f1']:.4f}")

    # Phase C — test eval with picked configs
    ema_results, ema_metrics, ema_lat = evaluate_on_test(
        by_call_test, "ema", alpha=best_ema["alpha"], tau=best_ema["tau"]
    )
    rm_results, rm_metrics, rm_lat = evaluate_on_test(
        by_call_test, "running_max", tau=best_rm["tau"]
    )
    # Phase D — baselines
    any_results, any_metrics, any_lat = evaluate_on_test(
        by_call_test, "running_max", tau=0.5
    )
    maj_results, maj_metrics, maj_lat = evaluate_on_test(
        by_call_test, "majority", tau=0.5
    )

    # Persist per-call decisions for the picked configs (for downstream analysis)
    test_results_path = SWEEP_DIR / f"test_results_{feature_type}.json"
    test_results_path.parent.mkdir(parents=True, exist_ok=True)
    with test_results_path.open("w") as f:
        json.dump({
            "feature_type": feature_type,
            "ema_best_config": {"alpha": best_ema["alpha"], "tau": best_ema["tau"]},
            "running_max_best_config": {"tau": best_rm["tau"]},
            "ema_metrics": ema_metrics, "ema_latency": ema_lat,
            "running_max_metrics": rm_metrics, "running_max_latency": rm_lat,
            "any_segment_baseline_metrics": any_metrics, "any_segment_baseline_latency": any_lat,
            "majority_metrics": maj_metrics, "majority_latency": maj_lat,
            "ema_calls": [r.as_dict() for r in ema_results],
            "running_max_calls": [r.as_dict() for r in rm_results],
        }, f, indent=2)

    summary_rows = [
        {"feature_type": feature_type, "method": "ema",
         "alpha": best_ema["alpha"], "tau": best_ema["tau"],
         **ema_metrics, **ema_lat},
        {"feature_type": feature_type, "method": "running_max",
         "alpha": None, "tau": best_rm["tau"],
         **rm_metrics, **rm_lat},
        {"feature_type": feature_type, "method": "any_segment_baseline (τ=0.5)",
         "alpha": None, "tau": 0.5,
         **any_metrics, **any_lat},
        {"feature_type": feature_type, "method": "majority_vote (τ=0.5)",
         "alpha": None, "tau": 0.5,
         **maj_metrics, **maj_lat},
    ]

    per_feature_payload = {
        "ema": {"results": ema_results},
        "running_max": {"results": rm_results},
        "any_seg_baseline": {"results": any_results},
        "majority": {"results": maj_results},
        "sweep_rows": sweep_rows,
    }

    return summary_rows, per_feature_payload


# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feature_types", default=",".join(FEATURE_TYPES))
    parser.add_argument("--pred_dir", type=Path, default=None,
                        help="Directory containing val_{ft}.jsonl / test_{ft}.jsonl predictions. "
                             "Defaults to modeling/logs/track_a/segment_predictions.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Directory to write call_level/, tables/, figures/ subdirs. "
                             "Defaults to modeling/logs/track_a.")
    args = parser.parse_args()

    cwd = Path.cwd()
    if cwd.name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, current cwd: {cwd}", file=sys.stderr)

    if args.pred_dir is not None or args.out_dir is not None:
        _set_paths(
            args.pred_dir if args.pred_dir is not None else PRED_DIR,
            args.out_dir if args.out_dir is not None else OUT_DIR,
        )

    feature_types = [ft.strip() for ft in args.feature_types.split(",")]

    summary: list[dict] = []
    per_ft: dict[str, dict] = {}
    best_configs: dict[str, dict] = {}
    for ft in feature_types:
        rows, payload = process_feature_type(ft)
        summary.extend(rows)
        per_ft[ft] = payload
        # Pick best configs from sweep
        best_configs[ft] = {
            "ema": pick_best(payload["sweep_rows"], "ema"),
            "running_max": pick_best(payload["sweep_rows"], "running_max"),
        }

    # Write best_configs.json
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with (SWEEP_DIR / "best_configs.json").open("w") as f:
        json.dump(best_configs, f, indent=2, default=str)

    # Write summary table
    write_summary(
        summary,
        csv_path=TABLE_DIR / "call_level_summary.csv",
        md_path=TABLE_DIR / "call_level_summary.md",
    )

    # Plot three alert-latency CDF variants so the reader can pick:
    #   - main: two-panel (EMA | Running-max), step + markers
    #   - variant A: single-panel, one curve per feature, best-method-per-feature
    #   - variant B: two-panel + inline annotation about overlapping curves
    plot_alert_latency_cdf(per_ft, FIG_DIR / "alert_latency_cdf.pdf")
    plot_alert_latency_cdf_simplified(
        per_ft, best_configs, FIG_DIR / "alert_latency_cdf_simplified.pdf"
    )
    plot_alert_latency_cdf_annotated(
        per_ft, FIG_DIR / "alert_latency_cdf_annotated.pdf"
    )

    # Plot α-τ heatmap per feature type
    for ft in feature_types:
        plot_alpha_tau_heatmap(
            per_ft[ft]["sweep_rows"], ft, FIG_DIR / f"alpha_tau_heatmap_{ft}.pdf"
        )

    print(f"\n✅ Outputs written under {OUT_DIR}/")
    print(f"   • Sweep CSVs: {SWEEP_DIR}/sweep_val_*.csv")
    print(f"   • Best configs: {SWEEP_DIR}/best_configs.json")
    print(f"   • Per-feature test results: {SWEEP_DIR}/test_results_*.json")
    print(f"   • Summary table: {TABLE_DIR}/call_level_summary.{{csv,md}}")
    print(f"   • Alert-latency CDF: {FIG_DIR}/alert_latency_cdf.pdf")
    print(f"   • α/τ heatmaps: {FIG_DIR}/alpha_tau_heatmap_{{egemaps,mfcc,wav2vec2,all}}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
