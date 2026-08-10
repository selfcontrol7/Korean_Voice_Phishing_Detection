# Track B v2 — Phone Procedure (energy + eGeMAPS-on-phone + memory)

This document picks up **after** the v1 procedure (Track B baseline) which is already complete. We're now adding three things the reviewers explicitly asked for:

1. **Energy** (Reviewer 3): sustained-load battery delta + average power + mJ/inference.
2. **eGeMAPS feature extraction on phone** (Reviewer 1): build openSMILE on Termux so we can measure the full pipeline, not just classifier+EMA.
3. **Memory + CPU footprint**: VmRSS sampled during the latency benchmark.

You'll re-run the benchmark with a patched script. The existing `phone_package/` on your phone (with audio + models + manifest) is **reused as-is** — we just overlay two new files.

**Total wall-clock:** ~30–60 min one-time openSMILE build + ~35 min benchmark.

---

## Step 0 — Workstation prep (already done)

I built `analysis/phone_package_delta.tar.gz` (~30 KB). It contains the patched `benchmark.py`, `build_opensmile.sh`, and this `README_v2.md`.

---

## Step 1 — Termux dependencies (one-time)

In Termux:

```bash
# Battery API + audio backend + build chain
pkg install -y termux-api libsndfile cmake make clang git
```

Then **install the Termux:API Android app from F-Droid** (this is a separate Android app from F-Droid that complements the `termux-api` package; without it, `termux-battery-status` returns no data):

- https://f-droid.org/en/packages/com.termux.api/
- Open the app once after installing, grant the battery permission when prompted.

Verify both pieces are working:

```bash
termux-battery-status
```

You should see JSON like:
```json
{"health": "GOOD", "percentage": 85, "temperature": 30.5, "status": "DISCHARGING", ...}
```

If `termux-battery-status` says command not found or returns nothing, **fix that before continuing** (the energy block silently produces empty results otherwise).

---

## Step 2 — Transfer the delta package

`analysis/phone_package_delta.tar.gz` is small (~30 KB) — any method works.

In Termux, get it into `~/downloads/` (or wherever your `phone_package/` lives) and overlay:

```bash
cd ~/downloads        # or wherever phone_package/ is
tar xzf phone_package_delta.tar.gz   # extracts phone_package_delta/

# Overlay into phone_package/
cp phone_package_delta/benchmark.py     phone_package/benchmark.py
cp phone_package_delta/build_opensmile.sh ~/build_opensmile.sh
chmod +x ~/build_opensmile.sh
```

You can now delete `phone_package_delta/` and the tarball if you want.

---

## Step 3 — Build openSMILE on the phone (~20–40 min, one-time)

This produces a native ARM64 `SMILExtract` binary linked against Termux's bionic libc — the prebuilt openSMILE Linux binaries depend on glibc and won't run on Termux.

```bash
bash ~/build_opensmile.sh
```

The script clones audeering/opensmile, configures with cmake, and runs `make -j$(nproc) SMILExtract`. The build is the slow step (~20–40 min on S24 Ultra). When it's done you'll see:

```
✅ SMILExtract installed to /data/data/com.termux/files/home/.local/bin/SMILExtract
```

Add `~/.local/bin/` to your PATH if not already (Termux usually puts it there automatically; check):

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

SMILExtract -h | head -3        # should print openSMILE 3.x banner
```

---

## Step 4 — Unplug the charger ⚠️

The energy block compares pre/post battery percentage. If the phone is plugged in, the percentage doesn't go down (it might even go up), and the benchmark will report nonsense or refuse to start.

**Action:** unplug, then idle the phone for ~5 minutes so battery temperature settles to a stable baseline.

---

## Step 5 — Run the v2 benchmark (~35 min)

```bash
cd ~/downloads/phone_package
termux-wake-lock                 # keep the CPU from sleeping when screen turns off

python benchmark.py --mode both \
    --output phone_results_v2.csv \
    --summary phone_summary_v2.json \
    --energy_json phone_energy.json \
    --energy_duration 600

termux-wake-unlock
```

What this does:
1. **Latency mode (~15 min):** 100 segments × 10 reps × 2 features = 2000 timings, each with RSS sampled. Per-call streaming on 50 vishing calls.
2. **Energy mode (~20 min):** Two sustained-load blocks — 10 min of continuous MFCC, then 10 min of continuous eGeMAPS — with battery + temperature sampled every 30 s.

At startup, the script runs a parity check that compares SMILExtract's eGeMAPS output to the precomputed features in the manifest. Expected max abs diff: <1e-3. If it fails, the script bails out with a clear message — paste the output here and we triage.

When it finishes you'll see:
```
✅ Wrote phone_results_v2.csv
✅ Wrote phone_summary_v2.json
✅ Wrote phone_energy.json
```

---

## Step 6 — Send results back

Three small files (~500 KB total):

```bash
cp phone_results_v2.csv phone_summary_v2.json phone_energy.json ~/storage/downloads/
```

Then transfer them off the phone (cloud / USB / scp), landing at:

```
~/projects/Korean_Voice_Phishing_Detection/Multimodal/modeling/logs/track_b/phone_raw/
```

(Replaces the v1 files there.)

Ping me when they're there and I'll run the analyzer to produce the paper-ready deployment table: **latency, memory, energy, average power, ΔTemp** in one row per feature type.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `build_opensmile.sh` fails at cmake | Make sure `cmake make clang git` are all installed: `pkg install -y cmake make clang git`. Re-run the script (it skips a fresh clone). |
| Build runs out of memory mid-make | Lower parallelism: edit `build_opensmile.sh` to use `make -j2` instead of `-j$(nproc)`. |
| `SMILExtract -h` says "command not found" | Either add `~/.local/bin` to PATH (see step 3) or call with the full path: `~/.local/bin/SMILExtract -h`. |
| `termux-battery-status` returns nothing | Open the Termux:API Android app (separate from Termux itself), grant battery permission. Then re-test. |
| Parity check fails (max abs diff > 1e-3) | Paste the parity JSON the script prints — most likely cause is the openSMILE source built a different version of the eGeMAPS config than what produced the precomputed features. We pin to v3.0 in the build script. |
| Benchmark aborts saying "device was CHARGING at block start" | You forgot to unplug — unplug, idle 5 min, re-run. |
| Phone gets warm during energy block | Expected. Track the temperature delta — if >5°C, we'll note thermal throttling in the paper. |
| You want a faster smoke run | `python benchmark.py --mode latency --n_reps 2` skips the energy block; or `--energy_duration 60` for a 1-min sustained-load smoke. |

---

## What the analyzer produces (after you send the files back)

- `modeling/logs/track_b/tables/deployment_summary.md` — **the paper-ready table** (5 columns per feature type)
- `modeling/logs/track_b/tables/latency_summary.md` — refreshed latency-only table with eGeMAPS now end-to-end
- `modeling/logs/track_b/tables/energy_breakdown.md` — energy detail per feature
- `modeling/logs/track_b/tables/memory_breakdown.md` — memory snapshots
- `modeling/logs/track_b/figures/latency_cdf.pdf` — per-segment latency CDF
- `modeling/logs/track_b/figures/energy_over_time.pdf` — battery + temperature timeline
- `modeling/logs/track_b/figures/memory_over_time.pdf` — RSS timeline

These plug directly into the paper's §VI Deployment Analysis.
