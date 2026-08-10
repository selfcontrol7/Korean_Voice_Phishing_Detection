"""Add speaker diarization fields to the segment manifests.

Joins each manifest row to its CLOVA transcript segment via
{call_id}.json -> segments[int(segment_id.rsplit('_', 1)[1])] and appends
two fields taken from the transcript's speaker object:

    speaker       CLOVA diarization label ("1", "2", ...) — call-local
    speaker_name  CLOVA display name ("A", "B", ...)      — call-local

Existing fields are untouched, so consumers of the previous schema are
unaffected. The join is validated by asserting the manifest `text` field
is byte-identical to the transcript segment text for every row.

Speaker labels are CALL-LOCAL: speaker "1" in one call has no relation to
speaker "1" in another call. No global speaker identity exists.

Usage (from Multimodal/):
    python preprocessing/add_speaker_to_manifests.py
"""

import json
import os
import sys

SPLITS = ["train", "val", "test"]
MANIFEST = "data/{split}_segment_manifest_merged.jsonl"
TRANSCRIPT_DIRS = ["data/transcripts/vishing", "data/transcripts/non_vishing"]


def transcript_path(call_id: str) -> str:
    for d in TRANSCRIPT_DIRS:
        p = os.path.join(d, f"{call_id}.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no transcript for call_id={call_id}")


def main() -> None:
    seg_cache = {}

    def segments(call_id: str):
        if call_id not in seg_cache:
            with open(transcript_path(call_id), encoding="utf-8") as f:
                seg_cache[call_id] = json.load(f)["segments"]
        return seg_cache[call_id]

    total = 0
    for split in SPLITS:
        path = MANIFEST.format(split=split)
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = int(row["segment_id"].rsplit("_", 1)[1])
                seg = segments(row["call_id"])[idx]
                if seg["text"] != row["text"]:
                    sys.exit(
                        f"TEXT MISMATCH at {row['segment_id']} in {path} — aborting, "
                        "manifest and transcripts are out of sync"
                    )
                spk = seg.get("speaker") or {}
                row["speaker"] = spk.get("label")
                row["speaker_name"] = spk.get("name")
                rows.append(row)

        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_labeled = sum(1 for r in rows if r["speaker"] is not None)
        print(f"{path}: {len(rows)} rows, {n_labeled} with speaker label")
        total += len(rows)

    print(f"done: {total} rows across {len(SPLITS)} splits")


if __name__ == "__main__":
    main()
