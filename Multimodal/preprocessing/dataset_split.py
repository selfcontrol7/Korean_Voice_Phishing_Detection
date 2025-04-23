import json
import os

from sklearn.model_selection import train_test_split

manifest_path = "../data/master_manifest.jsonl"
# Check if the manifest file exists
if not os.path.exists(manifest_path):
    raise FileNotFoundError(f"Manifest file {manifest_path} not found.")

# 1. Load manifest entries
with open(manifest_path, "r", encoding="utf-8") as f:
    entries = [json.loads(l) for l in f]

# 2. Split off test (e.g. 10%) then train/val (e.g. 10% of the remainder for val):
trainval, test = train_test_split(entries, test_size=0.10, stratify=[e["label"] for e in entries], random_state=42)
train, val   = train_test_split(trainval, test_size=0.11, stratify=[e["label"] for e in trainval], random_state=42)
# (0.11≈0.10/0.90 to make val ≈10% of total)

# 3. Save each split
def write_split(split_entries, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in split_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

write_split(train, "../data/train_manifest.jsonl")
write_split(val,   "../data/val_manifest.jsonl")
write_split(test,  "../data/test_manifest.jsonl")

# 4. Print the number of entries in each split
print(f"Train: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")
print("*"*50)

# 5. Print the number of unique labels in each split
train_labels = set(e["label"] for e in train)
val_labels   = set(e["label"] for e in val)
test_labels  = set(e["label"] for e in test)
print(f"Train labels: {len(train_labels)}")
print(f"Validation labels: {len(val_labels)}")
print(f"Test labels: {len(test_labels)}")
print("*"*50)

# 6. Print the unique labels in each split
print(f"Train labels: {train_labels}")
print(f"Validation labels: {val_labels}")
print(f"Test labels: {test_labels}")
print("*"*50)