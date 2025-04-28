# merge_text_manifests.py

import json
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data")
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    print(f"\n🔹 Merging manifests for {split} split...")

    # Load base manifest (contains audio features)
    base_manifest_path = DATA_DIR / f"{split}_segment_manifest.jsonl"
    with open(base_manifest_path, encoding='utf-8') as f:
        base_entries = {entry["segment_id"]: entry for entry in map(json.loads, f)}

    # Load KoBERT manifest
    kobert_manifest_path = DATA_DIR / f"{split}_segment_manifest_text_kobert.jsonl"
    with open(kobert_manifest_path, encoding='utf-8') as f:
        kobert_entries = {entry["segment_id"]: entry["kobert_text_path"] for entry in map(json.loads, f)}

    # Load KRSBERT manifest
    krsbert_manifest_path = DATA_DIR / f"{split}_segment_manifest_text_krsbert.jsonl"
    with open(krsbert_manifest_path, encoding='utf-8') as f:
        krsbert_entries = {entry["segment_id"]: entry["krsbert_text_path"] for entry in map(json.loads, f)}

    # Merge
    merged_entries = []
    for segment_id, entry in tqdm(base_entries.items()):
        entry["kobert_text_path"] = kobert_entries.get(segment_id)
        entry["krsbert_text_path"] = krsbert_entries.get(segment_id)
        merged_entries.append(entry)

    # Save unified manifest
    output_path = DATA_DIR / f"{split}_segment_manifest_merged.jsonl"
    with open(output_path, "w", encoding='utf-8') as fout:
        for item in merged_entries:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved merged manifest: {output_path}")
