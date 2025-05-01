# This script will create a dataloader for the text-only dataset using PyTorch.

import json
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for text-only classification
class TextOnlyDataset(Dataset):
    def __init__(self, manifest_path, text_feature):
        with open(manifest_path, encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]
        self.text_feature = text_feature

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx] # Load the entry from the manifest

        # Load the text features based on the specified feature type
        # and apply mean pooling or flattening as needed
        # to get fixed-sized vectors
        if self.text_feature == "both":
            kobert_feat = np.load(entry["kobert_text_path"])
            krsbert_feat = np.load(entry["krsbert_text_path"])
            text_feat = np.concatenate([kobert_feat, krsbert_feat])
        else:
            text_feat = np.load(entry[f"{self.text_feature}_text_path"])

        label = entry["label"]
        return torch.tensor(text_feat, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
