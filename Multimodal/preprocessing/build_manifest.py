import json
from pathlib import Path
from tqdm import tqdm

def build_manifest():
    AUDIO_ROOT = Path("data/audio/")
    TRANSCRIPT_ROOT = Path("data/transcripts/")
    OUTPUT = Path("data/master_manifest.jsonl")

    entries = []

    for label_dir, label in [("vishing", 1), ("non_vishing", 0)]:
        audio_dir = AUDIO_ROOT / label_dir
        transcript_dir = TRANSCRIPT_ROOT / label_dir

        for audio_file in tqdm(list(audio_dir.glob("*.wav"))):
            call_id = audio_file.stem
            transcript_file = transcript_dir / f"{call_id}.json"

            if not transcript_file.exists():
                print(f"[Warning] Missing transcript for {call_id}")
                continue

            entries.append({
                "call_id": call_id,
                "audio_filepath": str(audio_file),
                "transcript_filepath": str(transcript_file),
                "label": label
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved manifest with {len(entries)} entries to {OUTPUT}")

if __name__ == "__main__":
    build_manifest()
