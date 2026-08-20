"""Reconcile the processed vishing corpus against the raw FSS download backup.

Matches each processed file in data/audio/vishing/ (uniform 16 kHz mono PCM,
renamed vishing_N.wav) to its raw source file in the FSS backup folder
(original download formats: mp3/wav/mp4, original Korean titles) by audio
duration, and writes a mapping CSV.

Matching stages (in order, each consuming its matches):
  1. exact          1:1 duration match within +/-2 s
  2. offset          1:1 after correcting a constant ~+2.45 s bias
                     (Windows Shell metadata underestimates VBR MP3 durations)
  3. split           one raw recording was split into N processed calls
                     (multi-call downloads; the documented 654 -> 706 split)
  4. concat          N raw clips were concatenated into one processed call

Inputs (prepare before running):
  - C:\\Users\\Public\\fss_durations.tsv    name<TAB>seconds<TAB>bytes for every
    file in the backup folder (produced by a PowerShell Shell.Application
    script; files whose duration the Shell API cannot read have an empty
    seconds column)
  - scratchpad nodur_probe.tsv              ffprobe durations for those files,
    in the same order they appear in fss_durations.tsv
  - data/audio/vishing/*.wav                the processed corpus

Output:
  data/fss_source_file_mapping.csv (UTF-8 with BOM, Excel-friendly)

Usage (from Multimodal/):
    python analysis/match_fss_source_audio.py \
        --d-durations /mnt/c/Users/Public/fss_durations.tsv \
        --nodur-probe <scratchpad>/nodur_probe.tsv
"""

import argparse
import bisect
import csv
import glob
import itertools
import os

import soundfile as sf

MP3_METADATA_BIAS_S = 2.45  # Shell API underestimate for VBR MP3, measured


def load_d_durations(tsv_path, probe_path):
    dd = {}
    nodur_order = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if not p[0].strip():
                continue
            if len(p) > 1 and p[1]:
                dd[p[0]] = float(p[1])
            else:
                dd[p[0]] = None
                nodur_order.append(p[0])
    probes = []
    with open(probe_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                probes.append(float(p[1]))
    if len(probes) != len(nodur_order):
        raise SystemExit(
            f"nodur probe count {len(probes)} != metadata-less file count {len(nodur_order)}"
        )
    for name, sec in zip(nodur_order, probes):
        dd[name] = round(sec, 2)
    return dd


def load_local_durations(audio_dir):
    loc = {}
    for f in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        info = sf.info(f)
        loc[os.path.basename(f)] = round(info.frames / info.samplerate, 2)
    return loc


def match_1to1(dpool, lpool, tolfn, d_shift=0.0):
    d_items = sorted(dpool.items(), key=lambda x: x[1])
    dvals = [v + d_shift for _, v in d_items]
    used = [False] * len(d_items)
    pairs, um_l = [], {}
    for name, sec in sorted(lpool.items(), key=lambda x: x[1]):
        tol = tolfn(sec)
        i = bisect.bisect_left(dvals, sec - tol)
        best, bd = None, None
        while i < len(dvals) and dvals[i] <= sec + tol:
            if not used[i] and (bd is None or abs(dvals[i] - sec) < bd):
                best, bd = i, abs(dvals[i] - sec)
            i += 1
        if best is not None:
            used[best] = True
            pairs.append((name, d_items[best][0]))
        else:
            um_l[name] = sec
    um_d = {d_items[i][0]: d_items[i][1] for i in range(len(d_items)) if not used[i]}
    return pairs, um_l, um_d


def combos_summing_to(target, pool, ks, tol):
    for k in ks:
        for c in itertools.combinations(pool, k):
            if abs(sum(s for _, s in c) - target) <= tol:
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-durations", required=True)
    ap.add_argument("--nodur-probe", required=True)
    ap.add_argument("--audio-dir", default="data/audio/vishing")
    ap.add_argument("--out", default="data/fss_source_file_mapping.csv")
    args = ap.parse_args()

    dd = load_d_durations(args.d_durations, args.nodur_probe)
    loc = load_local_durations(args.audio_dir)
    rows = []  # (local_file, d_source_file, match_type, local_s, d_s, note)

    # stage 1: exact
    pairs, um_l, um_d = match_1to1(dd, loc, lambda s: 2.0)
    for ln, dn in pairs:
        rows.append((ln, dn, "exact", loc[ln], dd[dn], ""))

    # stage 2: offset-corrected (mp3 metadata bias)
    pairs2, um_l2, um_d2 = match_1to1(
        um_d, um_l, lambda s: max(1.5, s * 0.004), d_shift=MP3_METADATA_BIAS_S
    )
    for ln, dn in pairs2:
        rows.append((ln, dn, "offset_corrected", loc[ln], dd[dn],
                     f"metadata bias ~{MP3_METADATA_BIAS_S}s"))

    # stage 3: split (1 raw -> N local)
    used_l = set()
    split_d = set()
    for dn, dsec in sorted(um_d2.items(), key=lambda x: -x[1]):
        avail = [(n, s) for n, s in um_l2.items() if n not in used_l]
        c = combos_summing_to(dsec, avail, (2, 3, 4), max(4.0, dsec * 0.02))
        if c:
            split_d.add(dn)
            for part_i, (ln, lsec) in enumerate(sorted(c, key=lambda x: -x[1]), 1):
                rows.append((ln, dn, "split_part", lsec, dsec,
                             f"part {part_i}/{len(c)} of raw recording"))
                used_l.add(ln)

    # stage 4: concat (N raw -> 1 local)
    rem_l = {n: s for n, s in um_l2.items() if n not in used_l}
    rem_d = {n: s for n, s in um_d2.items() if n not in split_d}
    used_d = set()
    concat_l = set()
    for ln, lsec in sorted(rem_l.items(), key=lambda x: -x[1]):
        avail = [(n, s) for n, s in rem_d.items() if n not in used_d]
        c = combos_summing_to(lsec, avail, range(2, 9), max(5.0, lsec * 0.015))
        if c:
            concat_l.add(ln)
            for part_i, (dn, dsec) in enumerate(sorted(c, key=lambda x: -x[1]), 1):
                rows.append((ln, dn, "concat_part", lsec, dsec,
                             f"raw clip {part_i}/{len(c)} concatenated"))
                used_d.add(dn)

    # residuals
    for ln, lsec in rem_l.items():
        if ln not in concat_l:
            rows.append((ln, "", "unmatched_local", lsec, "",
                         "no source found in backup folder"))
    for dn, dsec in rem_d.items():
        if dn not in used_d:
            rows.append(("", dn, "unmatched_backup", "", dsec,
                         "backup clip not used in corpus"))

    def sort_key(r):
        if r[0].startswith("vishing_"):
            return (0, int(r[0].split("_")[1].split(".")[0]))
        return (1, 0)

    rows.sort(key=sort_key)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["manifest_file", "fss_source_file", "match_type",
                    "local_duration_s", "source_duration_s", "note"])
        w.writerows(rows)

    from collections import Counter
    c = Counter(r[2] for r in rows)
    print(f"wrote {args.out}: {len(rows)} rows")
    for k, v in c.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
