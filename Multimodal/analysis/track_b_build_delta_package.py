"""Track B v2 — bundle a small follow-up tarball for the phone.

Contents (overlays onto the existing phone_package/ directory):
    benchmark.py            (the patched v2 script)
    build_opensmile.sh      (the on-phone openSMILE build script)
    README_v2.md            (quick reference)

The original phone_package/ from track_b_prepare_phone_package.py is
unchanged on disk; the user untars this delta package on top.

Output: analysis/phone_package_delta.tar.gz (~30 KB)

Run from the Multimodal/ directory:
    python analysis/track_b_build_delta_package.py
"""
from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path


ANALYSIS = Path("analysis")
DELTA_DIR = ANALYSIS / "phone_package_delta"
TARBALL = ANALYSIS / "phone_package_delta.tar.gz"

SOURCES = {
    ANALYSIS / "track_b_benchmark.py":  DELTA_DIR / "benchmark.py",
    ANALYSIS / "build_opensmile.sh":    DELTA_DIR / "build_opensmile.sh",
}


README = """\
# Track B v2 — Phone Package Delta

Drop these files **into your existing `phone_package/` directory** on the phone
(`~/downloads/phone_package/`) and they will overlay the v1 benchmark.

Contents:
- `benchmark.py`         (replaces v1; same CLI but adds --mode/--energy_*/--skip_parity flags)
- `build_opensmile.sh`   (one-time build of the eGeMAPS extraction CLI on phone)

## Quick steps in Termux

```bash
# 0. One-time deps for the build + battery API
pkg install -y cmake make clang git termux-api libsndfile

# 1. Build openSMILE on the phone (~20–40 min)
bash build_opensmile.sh
export PATH="$HOME/.local/bin:$PATH"
SMILExtract -h | head -3   # verify

# 2. Verify Termux:API battery is working (need the Android app from F-Droid)
termux-battery-status

# ⚠️ 3. UNPLUG THE CHARGER (required for the energy block)

# 4. Run the full v2 benchmark (latency + sustained-load energy)
python benchmark.py --mode both \\
    --output phone_results_v2.csv \\
    --summary phone_summary_v2.json \\
    --energy_json phone_energy.json \\
    --energy_duration 600
```

Total wall-clock: ~30–60 min build (once) + ~35 min benchmark.

Send back:
- `phone_results_v2.csv`
- `phone_summary_v2.json`
- `phone_energy.json`

The workstation analyzer turns these into the paper-ready
Latency / Memory / Energy / Power / ΔT deployment table.
"""


def main() -> int:
    if Path.cwd().name != "Multimodal":
        print(f"⚠️  Expected to be run from Multimodal/, cwd={Path.cwd()}", file=sys.stderr)
        return 1
    DELTA_DIR.mkdir(parents=True, exist_ok=True)

    for src, dst in SOURCES.items():
        if not src.exists():
            print(f"❌ missing source: {src}", file=sys.stderr)
            return 1
        shutil.copyfile(src, dst)
        # Make .sh executable inside the tarball
        if dst.suffix == ".sh":
            dst.chmod(0o755)
        print(f"  copied {src} -> {dst}")

    (DELTA_DIR / "README_v2.md").write_text(README)
    print(f"  wrote {DELTA_DIR / 'README_v2.md'}")

    # tar
    print(f"\nBuilding {TARBALL}...")
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tf:
        for child in DELTA_DIR.iterdir():
            tf.add(child, arcname=f"phone_package_delta/{child.name}")
    size_kb = TARBALL.stat().st_size / 1024.0
    print(f"✅ {TARBALL}  ({size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
