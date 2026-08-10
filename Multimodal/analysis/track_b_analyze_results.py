"""Track B step 4 (v2) — Post-process the CSV + energy JSON from the phone.

Reads phone_results_v2.csv (latency + RSS + CPU columns) and
phone_energy.json (sustained-load battery + temperature samples per
feature type). Emits paper-ready deployment tables and figures.

Run from the Multimodal/ directory:
    python analysis/track_b_analyze_results.py \\
        --input modeling/logs/track_b/phone_raw/phone_results_v2.csv \\
        --summary_json modeling/logs/track_b/phone_raw/phone_summary_v2.json \\
        --energy_json modeling/logs/track_b/phone_raw/phone_energy.json \\
        --output_dir modeling/logs/track_b
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


STAGES = ["load_ms", "feat_ms", "fwd_ms", "ema_ms", "end_to_end_ms"]
STAGE_LABEL = {
    "load_ms": "audio load",
    "feat_ms": "feature extract",
    "fwd_ms": "TorchScript fwd",
    "ema_ms": "EMA update",
    "end_to_end_ms": "end-to-end",
}

BATTERY_CAPACITY_WH_DEFAULT = 19.4  # S24 Ultra: 5000 mAh × 3.88 V


def _to_float(v) -> float:
    try:
        if v in ("", None, "None"):
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return {k: float("nan") for k in ("mean", "std", "median", "p95", "p99", "n")}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "n": len(vals),
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _energy_summary_block(block: dict, battery_wh: float) -> dict:
    pre, post = block.get("pre", {}), block.get("post", {})
    pre_pct = _to_float(pre.get("percentage"))
    post_pct = _to_float(post.get("percentage"))
    pre_temp = _to_float(pre.get("temperature"))
    post_temp = _to_float(post.get("temperature"))
    duration = _to_float(block.get("duration_s"))
    n_inf = _to_float(block.get("n_inferences"))
    delta_pct = pre_pct - post_pct if (pre_pct == pre_pct and post_pct == post_pct) else float("nan")
    delta_energy_wh = (delta_pct / 100.0) * battery_wh if delta_pct == delta_pct else float("nan")
    avg_power_mw = (delta_energy_wh * 1000.0 * 3600.0 / duration) if (duration and duration == duration) else float("nan")
    energy_per_inf_mj = (delta_energy_wh * 3600.0 * 1000.0 / n_inf) if (n_inf and n_inf == n_inf) else float("nan")
    delta_temp = post_temp - pre_temp if (pre_temp == pre_temp and post_temp == post_temp) else float("nan")

    # If charge_counter available (microampere-hours), get a more precise number
    pre_cc = _to_float(pre.get("charge_counter"))
    post_cc = _to_float(post.get("charge_counter"))
    voltage_v = _to_float(pre.get("voltage")) / 1000.0 if pre.get("voltage") else 3.88
    delta_cc_uah = (pre_cc - post_cc) if (pre_cc == pre_cc and post_cc == post_cc) else float("nan")
    delta_energy_cc_wh = (delta_cc_uah / 1e6) * voltage_v if delta_cc_uah == delta_cc_uah else float("nan")
    avg_power_cc_mw = (delta_energy_cc_wh * 1000.0 * 3600.0 / duration) if (duration and delta_energy_cc_wh == delta_energy_cc_wh) else float("nan")
    energy_per_inf_cc_mj = (delta_energy_cc_wh * 3600.0 * 1000.0 / n_inf) if (n_inf and delta_energy_cc_wh == delta_energy_cc_wh) else float("nan")

    return {
        "feature_type": block.get("feature_type"),
        "duration_s": duration,
        "n_inferences": n_inf,
        "pre_pct": pre_pct, "post_pct": post_pct, "delta_pct": delta_pct,
        "pre_temp_c": pre_temp, "post_temp_c": post_temp, "delta_temp_c": delta_temp,
        "delta_energy_wh": delta_energy_wh,
        "avg_power_mw": avg_power_mw,
        "energy_per_inference_mj": energy_per_inf_mj,
        "charge_counter_available": delta_cc_uah == delta_cc_uah,
        "delta_charge_counter_uAh": delta_cc_uah,
        "delta_energy_wh_cc": delta_energy_cc_wh,
        "avg_power_mw_cc": avg_power_cc_mw,
        "energy_per_inference_mj_cc": energy_per_inf_cc_mj,
        "notes": block.get("notes", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--energy_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("modeling/logs/track_b"))
    parser.add_argument("--battery_wh", type=float, default=BATTERY_CAPACITY_WH_DEFAULT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Missing input: {args.input}", file=sys.stderr)
        return 1
    rows = _read_csv(args.input)
    seg_rows = [r for r in rows if r["kind"] == "segment"]
    call_rows = [r for r in rows if r["kind"] == "call"]
    print(f"Loaded {len(seg_rows)} segment rows + {len(call_rows)} call rows")

    summary = None
    if args.summary_json and args.summary_json.exists():
        with args.summary_json.open() as f:
            summary = json.load(f)

    energy = None
    if args.energy_json and args.energy_json.exists():
        with args.energy_json.open() as f:
            energy = json.load(f)
        print(f"Loaded {len(energy.get('blocks', []))} energy blocks")

    tables_dir = args.output_dir / "tables"
    fig_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-segment per-stage stats (and memory snapshot)
    # ------------------------------------------------------------------
    seg_stats: dict[str, dict[str, dict]] = {}
    rss_stats: dict[str, dict] = {}
    fts = sorted({r["feature_type"] for r in seg_rows})
    for ft in fts:
        seg_stats[ft] = {}
        per_ft = [r for r in seg_rows if r["feature_type"] == ft]
        for stage in STAGES:
            seg_stats[ft][stage] = _stats([_to_float(r.get(stage)) for r in per_ft])
        rss_stats[ft] = _stats([_to_float(r.get("rss_mb")) for r in per_ft])

    # ------------------------------------------------------------------
    # Per-call summary
    # ------------------------------------------------------------------
    call_summary: dict[str, dict] = {}
    for ft in sorted({r["feature_type"] for r in call_rows}):
        per = [r for r in call_rows if r["feature_type"] == ft]
        totals_s = [_to_float(r["total_call_ms"]) / 1000.0 for r in per]
        firsts_s = [_to_float(r["first_alert_ms"]) / 1000.0
                    for r in per if r.get("first_alert_ms") not in (None, "", "None")]
        peaks_ms = [_to_float(r["peak_seg_ms"]) for r in per]
        call_summary[ft] = {
            "total_call_s": _stats(totals_s),
            "first_alert_s": _stats(firsts_s),
            "peak_seg_ms": _stats(peaks_ms),
        }

    # ------------------------------------------------------------------
    # Energy summary
    # ------------------------------------------------------------------
    energy_summary: dict[str, dict] = {}
    if energy:
        for block in energy.get("blocks", []):
            es = _energy_summary_block(block, args.battery_wh)
            energy_summary[es["feature_type"]] = es

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    # latency_summary.md (refreshed, now includes RSS and end-to-end with eGeMAPS extraction)
    seg_csv = tables_dir / "latency_summary.csv"
    seg_md = tables_dir / "latency_summary.md"
    with seg_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_type", "stage", "mean_ms", "std_ms", "median_ms", "p95_ms", "p99_ms", "n"])
        for ft in fts:
            for stage in STAGES:
                s = seg_stats[ft][stage]
                w.writerow([ft, stage,
                            f"{s['mean']:.3f}", f"{s['std']:.3f}",
                            f"{s['median']:.3f}", f"{s['p95']:.3f}", f"{s['p99']:.3f}",
                            s["n"]])

    with seg_md.open("w") as f:
        f.write("# Track B v2 — Latency Summary (Per-segment)\n\n")
        if summary:
            f.write(f"**Device.** `{summary.get('machine')}` on `{summary.get('platform')}`, "
                    f"PyTorch `{summary.get('torch')}`. "
                    f"eGeMAPS backend = `{summary.get('egemaps_backend')}`. ")
            if summary.get('smilextract_bin'):
                pf = summary.get('smilextract_parity') or {}
                if pf.get('passed'):
                    f.write(f"SMILExtract parity vs precomputed: max abs diff = {pf.get('max_abs_diff'):.2e} ✓. ")
            f.write("\n\n")
        f.write("| Feature | Stage | Mean (ms) | Std (ms) | Median (ms) | p95 (ms) | p99 (ms) | n |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for ft in fts:
            for stage in STAGES:
                s = seg_stats[ft][stage]
                f.write(f"| {ft} | {STAGE_LABEL[stage]} | "
                        f"{s['mean']:.2f} | {s['std']:.2f} | {s['median']:.2f} | "
                        f"{s['p95']:.2f} | {s['p99']:.2f} | {s['n']} |\n")
        f.write("\n## Per-call (50 full vishing calls)\n\n")
        f.write("| Feature | Total call (s) mean ± std | Time to first alert (s) mean ± std | Peak per-segment (ms) mean ± std |\n")
        f.write("|---|---|---|---|\n")
        for ft, cs in call_summary.items():
            tot, fst, pk = cs["total_call_s"], cs["first_alert_s"], cs["peak_seg_ms"]
            f.write(f"| {ft} | {tot['mean']:.2f} ± {tot['std']:.2f} | "
                    f"{fst['mean']:.2f} ± {fst['std']:.2f} | "
                    f"{pk['mean']:.2f} ± {pk['std']:.2f} |\n")
    print(f"✓ wrote {seg_md}")

    # memory_breakdown.md
    mem_md = tables_dir / "memory_breakdown.md"
    with mem_md.open("w") as f:
        f.write("# Track B v2 — Memory Footprint\n\n")
        f.write("Resident set size (VmRSS) sampled at every per-segment timing row. "
                "Reports per-feature mean and peak during the latency benchmark.\n\n")
        f.write("| Feature | RSS mean (MB) | RSS std (MB) | RSS peak (p99, MB) | n samples |\n")
        f.write("|---|---|---|---|---|\n")
        for ft, s in rss_stats.items():
            f.write(f"| {ft} | {s['mean']:.1f} | {s['std']:.1f} | {s['p99']:.1f} | {s['n']} |\n")
        if summary and summary.get("latency_checkpoints"):
            f.write("\n## Latency-mode checkpoints (cold→warm)\n\n")
            f.write("| Checkpoint | RSS (MB) | VmHWM (MB) | CPU (s) |\n")
            f.write("|---|---|---|---|\n")
            for c in summary["latency_checkpoints"]:
                rss = _to_float(c.get("rss_mb"))
                hwm = _to_float(c.get("vm_hwm_mb"))
                cpu = _to_float(c.get("cpu_s"))
                f.write(f"| {c.get('label')} | {rss:.1f} | {hwm:.1f} | {cpu:.2f} |\n")
    print(f"✓ wrote {mem_md}")

    # energy_breakdown.md
    if energy_summary:
        en_md = tables_dir / "energy_breakdown.md"
        with en_md.open("w") as f:
            f.write("# Track B v2 — Sustained-Load Energy Breakdown\n\n")
            f.write(f"Battery-delta methodology. Assumed battery capacity: "
                    f"**{args.battery_wh:.2f} Wh** (S24 Ultra: 5000 mAh × 3.88 V). "
                    f"Charge-counter-based numbers (when available) are more precise; "
                    f"percentage-delta numbers are 1%-quantized.\n\n")
            f.write("| Feature | Duration (s) | n_inferences | Δpct | ΔTemp (°C) | ΔE (Wh) | Avg power (mW) | mJ / inference |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for ft, es in energy_summary.items():
                f.write(f"| {ft} | {es['duration_s']:.0f} | {int(es['n_inferences'])} | "
                        f"{es['delta_pct']:.2f} | {es['delta_temp_c']:.2f} | "
                        f"{es['delta_energy_wh']:.4f} | {es['avg_power_mw']:.0f} | "
                        f"{es['energy_per_inference_mj']:.3f} |\n")
            cc_available = any(es.get("charge_counter_available") for es in energy_summary.values())
            if cc_available:
                f.write("\n### Charge-counter cross-check (higher resolution)\n\n")
                f.write("| Feature | Δcharge (μAh) | ΔE (Wh, cc) | Avg power (mW, cc) | mJ / inference (cc) |\n")
                f.write("|---|---|---|---|---|\n")
                for ft, es in energy_summary.items():
                    if es.get("charge_counter_available"):
                        f.write(f"| {ft} | {es['delta_charge_counter_uAh']:.0f} | "
                                f"{es['delta_energy_wh_cc']:.4f} | "
                                f"{es['avg_power_mw_cc']:.0f} | "
                                f"{es['energy_per_inference_mj_cc']:.3f} |\n")
        print(f"✓ wrote {en_md}")

    # deployment_summary.md — the paper-ready combined table
    dep_md = tables_dir / "deployment_summary.md"
    with dep_md.open("w") as f:
        f.write("# Track B v2 — On-Device Deployment Summary\n\n")
        if summary:
            f.write(f"**Device.** {summary.get('machine')} / {summary.get('platform')} / "
                    f"PyTorch {summary.get('torch')} / TorchScript Mobile.\n\n")
        f.write("Reviewer R1 asked for empirical mobile latency. Reviewer R3 asked for "
                "runtime, latency, and energy. The table below reports all three plus memory "
                "in a single paper-ready view.\n\n")
        f.write("| Feature | End-to-end latency (ms) | Memory peak (MB) | Energy / inference (mJ) | Avg power (mW) | ΔTemp (°C) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for ft in fts:
            e2e = seg_stats[ft]["end_to_end_ms"]
            mem = rss_stats[ft]
            es = energy_summary.get(ft, {})
            e_per = es.get("energy_per_inference_mj", float("nan"))
            p_avg = es.get("avg_power_mw", float("nan"))
            d_t = es.get("delta_temp_c", float("nan"))
            f.write(f"| {ft} | "
                    f"{e2e['mean']:.2f} ± {e2e['std']:.2f} (med {e2e['median']:.2f}, p95 {e2e['p95']:.2f}) | "
                    f"{mem['p99']:.1f} | "
                    f"{e_per:.3f} | "
                    f"{p_avg:.0f} | "
                    f"{d_t:.2f} |\n")
        f.write("\n*Latency reported as mean ± std with median and p95 over "
                f"{seg_stats[fts[0]]['end_to_end_ms']['n'] if fts else '—'} timings per feature type. "
                "Energy from a 10-min sustained-load block per feature (charger unplugged).*\n")
    print(f"✓ wrote {dep_md}")

    # Per-call CSV
    if call_rows:
        call_csv = tables_dir / "latency_per_call.csv"
        with call_csv.open("w", newline="") as f:
            cols = ["feature_type", "egemaps_backend", "call_id", "label",
                    "total_call_ms", "first_alert_ms", "peak_seg_ms",
                    "n_segments", "final_S", "tau", "alpha"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in call_rows:
                w.writerow({k: r.get(k, "") for k in cols})

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # latency_cdf.pdf
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        colors = {"mfcc": "C0", "egemaps": "C1"}
        x_max = 0.0
        for ft in fts:
            vals = sorted(_to_float(r.get("end_to_end_ms")) for r in seg_rows if r["feature_type"] == ft)
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            ys = np.linspace(1.0 / len(vals), 1.0, len(vals))
            ax.step(vals, ys, where="post", color=colors.get(ft, "k"),
                    label=f"{ft} (median {np.median(vals):.1f} ms, p95 {np.percentile(vals,95):.1f} ms)",
                    linewidth=1.8)
            x_max = max(x_max, vals[-1])
        ax.set_xlabel("End-to-end per-segment latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("Per-segment latency CDF (S24 Ultra, TorchScript)")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, max(50.0, x_max * 1.05))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / "latency_cdf.pdf", format="pdf")
        plt.close(fig)
        print(f"✓ wrote {fig_dir / 'latency_cdf.pdf'}")

        # energy_over_time.pdf
        if energy:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
            for block in energy.get("blocks", []):
                ft = block.get("feature_type")
                ts = [s["t_s"] for s in block.get("samples", [])]
                pct = [s.get("pct") for s in block.get("samples", [])]
                tmp = [s.get("temp_c") for s in block.get("samples", [])]
                color = colors.get(ft, "k")
                axes[0].plot(ts, pct, marker="o", color=color, label=ft)
                axes[1].plot(ts, tmp, marker="o", color=color, label=ft)
            axes[0].set_ylabel("Battery (%)")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=9, loc="upper right")
            axes[0].set_title("Battery and temperature during sustained-load block")
            axes[1].set_ylabel("Battery temperature (°C)")
            axes[1].set_xlabel("Time within block (s)")
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / "energy_over_time.pdf", format="pdf")
            plt.close(fig)
            print(f"✓ wrote {fig_dir / 'energy_over_time.pdf'}")

            # memory_over_time.pdf
            fig, ax = plt.subplots(figsize=(7.5, 3.5))
            for block in energy.get("blocks", []):
                ft = block.get("feature_type")
                ts = [s["t_s"] for s in block.get("samples", [])]
                rss = [s.get("rss_mb") for s in block.get("samples", [])]
                color = colors.get(ft, "k")
                ax.plot(ts, rss, marker="o", color=color, label=ft)
            ax.set_xlabel("Time within block (s)")
            ax.set_ylabel("Resident memory (MB)")
            ax.set_title("RSS during sustained-load block")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="upper right")
            fig.tight_layout()
            fig.savefig(fig_dir / "memory_over_time.pdf", format="pdf")
            plt.close(fig)
            print(f"✓ wrote {fig_dir / 'memory_over_time.pdf'}")
    except Exception as e:
        print(f"  (figure step partially skipped: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
