# This script will create a dataloader for the audio-only dataset using PyTorch.

import json
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for audio-only baseline
class AudioOnlyDataset(Dataset):
    def __init__(self, manifest_path, feature_type):
        with open(manifest_path, encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]
        self.feature_type = feature_type

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load the audio features based on the specified feature type
        # and apply mean pooling or flattening as needed
        # to get fixed-sized vectors
        if self.feature_type == "mfcc":
            audio = np.load(entry["mfcc_path"]) # Load MFCC features
            audio_feat = audio.mean(axis=0) # Mean pooling for MFCC features
        elif self.feature_type == "egemaps":
            audio = np.load(entry["egemaps_path"]) # Load eGeMAPS features
            audio_feat = audio.flatten() # Flattening the eGeMAPS features
        elif self.feature_type == "wav2vec2":
            audio = np.load(entry["wav2vec2_path"]) # Load wav2vec2 features
            audio_feat = audio.mean(axis=0) # Mean pooling for wav2vec2 features
        elif self.feature_type == "all":
            mfcc = np.load(entry["mfcc_path"]).mean(axis=0) # Mean pooling for MFCC features
            egemaps = np.load(entry["egemaps_path"]).flatten() # Flattening the eGeMAPS features
            wav2vec2 = np.load(entry["wav2vec2_path"]).mean(axis=0) # Mean pooling for wav2vec2 features
            audio_feat = np.concatenate([mfcc, egemaps, wav2vec2]) # Concatenate the audio features
        else:
            raise ValueError("Unsupported feature type")

        label = entry["label"] # Load the label

        # Convert and return the features and label to PyTorch tensors
        return torch.tensor(audio_feat, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
