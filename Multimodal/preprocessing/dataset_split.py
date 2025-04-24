import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define paths relative to project root
DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "master_manifest.jsonl"

# Check if the manifest file exists
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Manifest file {MANIFEST_PATH} not found.")

# 1. Load manifest entries
with MANIFEST_PATH.open("r", encoding="utf-8") as f:
    entries = [json.loads(l) for l in f]

# 2. Split dataset
trainval, test = train_test_split(
    entries, test_size=0.10, stratify=[e["label"] for e in entries], random_state=42
)
train, val = train_test_split(
    trainval, test_size=0.11, stratify=[e["label"] for e in trainval], random_state=42
)

# 3. Save splits
def write_split(split_entries, filename):
    split_path = DATA_DIR / filename
    with split_path.open("w", encoding="utf-8") as f:
        for e in split_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

write_split(train, "train_manifest.jsonl")
write_split(val,   "val_manifest.jsonl")
write_split(test,  "test_manifest.jsonl")

# 4. Print summary
print(f"Train: {len(train)} samples")
print(f"Validation: {len(val)} samples")
print(f"Test: {len(test)} samples")
print("*" * 50)

# 5 & 6. Label stats
for split_name, split_data in zip(["Train", "Validation", "Test"], [train, val, test]):
    labels = set(e["label"] for e in split_data)
    print(f"{split_name} labels count: {len(labels)}")
    print(f"{split_name} unique labels: {labels}")
    print("*" * 50)
