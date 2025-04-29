# This script will create a dataloader for the multimodal dataset using PyTorch.

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Define the MultimodalPhishingDataset class
class MultimodalPhishingDataset(Dataset):
    def __init__(self, manifest_path, text_feature="kobert", use_both_text=False):
        with open(manifest_path, encoding='utf-8') as f: # Load the manifest file
            self.entries = [json.loads(line) for line in f] # Read the file line by line and parse each line as a JSON object
        self.text_feature = text_feature # Specify the text feature to be used
        self.use_both_text = use_both_text # Specify whether to use both text features or not

    def __len__(self):
        """
        Returns the number of entries in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Returns the entry at the specified index.
        """
        entry = self.entries[idx] # Get the entry at the specified index

        # Load the audio features
        mfcc = np.load(entry["mfcc_path"])
        egemaps = np.load(entry["egemaps_path"])
        wav2vec2 = np.load(entry["wav2vec2_path"])

        # simple pooling (mean) to get fixed-sized vectors
        mfcc_feat = mfcc.mean(axis=0) # Mean pooling for MFCC features
        egemaps_feat = egemaps.flatten() # Flattening the egemaps features
        wav2vec2_feat = wav2vec2.mean(axis=0) # Mean pooling for wav2vec features

        # concatenate the audio features
        audio_features = np.concatenate([mfcc_feat, egemaps_feat, wav2vec2_feat]) # Concatenate the audio features

        # Load the text features
        if self.use_both_text:
            kobert_feat = np.load(entry["kobert_text_path"]) # Load the kobert text features
            krsbert_feat = np.load(entry["krsbert_text_path"]) # Load the krsbert text features
            text_features = np.concatenate([kobert_feat, krsbert_feat]) # Concatenate the text features
        else:
            text_path = entry[f"{self.text_feature}_text_path"]
            text_features = np.load(text_path)

        # Load the label
        label = entry["label"]

        # Convert and return the features and label to PyTorch tensors
        return {
            "audio_features": torch.tensor(audio_features, dtype=torch.float32),
            "text_features": torch.tensor(text_features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }

# Create a function to create the dataloader (Example of how to use the dataset)
def get_dataloader(manifest_path, batch_size=32, shuffle=True, text_feature="kobert", use_both_text=False):
    """
    Create a dataloader for the multimodal dataset.
    """
    dataset = MultimodalPhishingDataset(manifest_path, text_feature, use_both_text) # Create the dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # Create the dataloader

if __name__ == "__main__":
    train_loader = get_dataloader(
        manifest_path="data/train_segment_manifest_merged.jsonl",
        batch_size=16,
        text_feature="krsbert",
        use_both_text=False
    ) # Create the dataloader for the training set

    # Print the shape of the first batch
    for batch in train_loader:
        print(batch["audio_features"].shape) # Print the shape of the audio features
        print(batch["text_features"].shape) # Print the shape of the text features
        print(batch["label"].shape) # Print the shape of the labels
        break # Break after the first batch
