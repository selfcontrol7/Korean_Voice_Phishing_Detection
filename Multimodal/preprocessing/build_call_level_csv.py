"""Build a one-row-per-call CSV of the KorCCVi v2-multimodal-audited corpus.

The corpus has 1,417 calls: 706 FSS vishing and 711 AI Hub non-vishing.
Each row joins:
  - data/master_manifest.jsonl: call id, label, source, transcript path
  - data/{train,val,test}_manifest.jsonl: the historical call-level split
  - data/transcripts/<class>/<call_id>.json: the full CLOVA transcript, segment count, speakers, confidence
  - data/paper2_v2/folds/grouped_nested_v1/assignments.jsonl: membership in the
    1,188-call KorCCVi v2-multimodal-filtered population and the relation group
  - audio duration: revised_experiments/artifacts/work5_data_audit_v1/call_provenance.jsonl
    (a local, unpublished audit artifact); without it, the WAV headers under
    data/audio/ are read, and duration_s is left empty if neither is available
  - data/fss_source_file_mapping.csv: the original FSS download file(s) for each vishing call

The output is UTF-8 with a BOM, so Excel shows Hangul correctly.

Run from Multimodal/:
    python preprocessing/build_call_level_csv.py --output data/KorCCVi_v2_multimodal_calls.csv
"""

import argparse
import csv
import json
import wave
from collections import defaultdict
from pathlib import Path

SOURCE_NAMES = {"FSS": "FSS", "AllHub": "AI Hub"}  # manifests spell AI Hub as "AllHub"
STATUS_NAMES = {
    None: "eligible",
    "burned_historical_test": "historical_test",
    "relation_component_touches_burned_historical_test": "related_to_historical_test",
    "exact_audio_duplicate_noncanonical": "duplicate_audio",
}
COLUMNS = [
    "call_id", "label", "class", "source", "split", "duration_s", "n_segments", "n_speakers",
    "asr_confidence", "in_filtered_1188", "population_status", "relation_group_id",
    "audio_file", "fss_source_file", "fss_match_type", "transcript",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="the Multimodal/ directory")
    parser.add_argument("--output", default="data/KorCCVi_v2_multimodal_calls.csv")
    args = parser.parse_args()
    root = Path(args.root)

    calls = read_jsonl(root / "data/master_manifest.jsonl")
    split = {}
    for name in ("train", "val", "test"):
        for row in read_jsonl(root / f"data/{name}_manifest.jsonl"):
            split[row["call_id"]] = name
    assignment = {row["call_id"]: row for row in read_jsonl(
        root / "data/paper2_v2/folds/grouped_nested_v1/assignments.jsonl")}
    provenance = root / "revised_experiments/artifacts/work5_data_audit_v1/call_provenance.jsonl"
    if provenance.exists():
        duration = {row["call_id"]: row["audio"]["duration_seconds"] for row in read_jsonl(provenance)}
    else:  # fall back to the WAV headers when the audio is available locally
        duration = {}
        for call in calls:
            audio = root / call["audio_filepath"]
            if audio.exists():
                with wave.open(str(audio)) as handle:
                    duration[call["call_id"]] = handle.getnframes() / handle.getframerate()

    # One corpus file can come from several FSS downloads (concatenated clips),
    # and one download can feed several corpus files (split recordings).
    fss_files, fss_types = defaultdict(list), defaultdict(list)
    with open(root / "data/fss_source_file_mapping.csv", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["manifest_file"]:
                continue  # backup clips that are not part of the corpus
            call_id = row["manifest_file"].removesuffix(".wav")
            if row["fss_source_file"]:
                fss_files[call_id].append(row["fss_source_file"])
            if row["match_type"] not in fss_types[call_id]:
                fss_types[call_id].append(row["match_type"])

    rows = []
    for call in calls:
        call_id = call["call_id"]
        transcript = json.loads((root / call["transcript_filepath"]).read_text(encoding="utf-8"))
        segments = transcript["segments"]
        speakers = {segment["speaker"]["label"] for segment in segments if segment.get("speaker")}
        fold = assignment[call_id]
        rows.append({
            "call_id": call_id,
            "label": call["label"],
            "class": "vishing" if call["label"] == 1 else "non_vishing",
            "source": SOURCE_NAMES[call["source"]],
            "split": split[call_id],
            "duration_s": f"{duration[call_id]:.2f}" if call_id in duration else "",
            "n_segments": len(segments),
            "n_speakers": len(speakers),
            "asr_confidence": f"{transcript['confidence']:.4f}",
            "in_filtered_1188": int(fold["role"] == "eligible"),
            "population_status": STATUS_NAMES[fold.get("exclusion_reason")],
            # Same id for calls linked by a shared recording, duplicate or template.
            # It equals the Paper 2 group_id (grp_<hash>) for the 1,188 eligible calls.
            "relation_group_id": fold["relation_component_id"].removeprefix("component_"),
            "audio_file": Path(call["audio_filepath"]).name,
            "fss_source_file": " + ".join(fss_files.get(call_id, [])),
            "fss_match_type": "+".join(fss_types.get(call_id, [])),
            "transcript": " ".join(transcript["text"].split()),
        })

    rows.sort(key=lambda r: (-r["label"], int(r["call_id"][8:]) if r["label"] == 1 else 0, r["call_id"]))
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} calls to {output}")
    print(f"  vishing {sum(r['label'] for r in rows)}, non-vishing {sum(1 - r['label'] for r in rows)}")
    print(f"  segments {sum(r['n_segments'] for r in rows)}")
    print(f"  filtered population {sum(r['in_filtered_1188'] for r in rows)} calls in "
          f"{len({r['relation_group_id'] for r in rows if r['in_filtered_1188']})} relation groups")


if __name__ == "__main__":
    main()
