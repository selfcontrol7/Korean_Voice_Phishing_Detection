import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# ----------------------------------------
# Model Definition (unchanged)
# ----------------------------------------
class MultiBranchCrossModalModel(nn.Module):
    """
    Multi-Branch Cross-Modal Attention with Guided Audio Fusion

    Expects inputs:
      - mfcc_feats: FloatTensor (batch, seq_len_a, mfcc_dim)
      - egemaps_feats: FloatTensor (batch, seq_len_a, egemaps_dim)
      - wav2vec_feats: FloatTensor (batch, seq_len_a, wav2vec_dim)
      - text_feats: FloatTensor (batch, seq_len_t, text_dim)
    Outputs:
      - logits: FloatTensor (batch, num_labels)
    """
    def __init__(
        self,
        mfcc_dim: int = 13,
        egemaps_dim: int = 88,
        wav2vec_dim: int = 768,
        text_dim: int = 768,
        proj_dim: int = 256,
        num_heads: int = 4,
        fusion_dim: int = 256,
        num_labels: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        # Projection layers for each feature stream
        self.mfcc_proj = nn.Sequential(nn.Linear(mfcc_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.egemaps_proj = nn.Sequential(nn.Linear(egemaps_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.wav2vec_proj = nn.Sequential(nn.Linear(wav2vec_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())

        # Cross-attention modules (batch_first=True requires PyTorch >=1.11)
        self.attn_mfcc = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        self.attn_egemaps = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        self.attn_wav2vec = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)

        # Fusion and classifier
        self.fusion = nn.Sequential(nn.Linear(proj_dim * 3, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, mfcc_feats, egemaps_feats, wav2vec_feats, text_feats):
        # Project each stream
        mfcc_p = self.mfcc_proj(mfcc_feats) # (B, T_m, H)
        # print(f"shape mfcc: {mfcc_p.shape}")
        egemaps_p = self.egemaps_proj(egemaps_feats) # (B, T_e, H)
        # print(f"shape egemaps: {egemaps_p.shape}")
        wav2vec_p = self.wav2vec_proj(wav2vec_feats) # (B, T_w, H)
        # print(f"shape wav2vec: {wav2vec_p.shape}")
        text_p = self.text_proj(text_feats) # (B, H)
        # print(f"shape text: {text_p.shape}")

        # text_p = text_p.unsqueeze(1).repeat(1, mfcc_p.size(1), 1) # (B, T_m, H) for cross-attention
        # turn text into a length-1 sequence
        text_q = text_p.unsqueeze(1)  # (B, 1, H)
        # print(f"shape text_q: {text_q.shape}")

        # Cross-attention for each audio modality: queries=text, each audio as key/value
        attn_m, _ = self.attn_mfcc(query=text_q, key=mfcc_p, value=mfcc_p) # (B, 1, H)
        attn_e, _ = self.attn_egemaps(query=text_q, key=egemaps_p, value=egemaps_p)
        attn_w, _ = self.attn_wav2vec(query=text_q, key=wav2vec_p, value=wav2vec_p)

        # # Pool over text sequence/tokens (mean pooling)
        pool_m = attn_m.mean(dim=1)
        pool_e = attn_e.mean(dim=1)
        pool_w = attn_w.mean(dim=1)
        #
        # # Concatenate pooled features and apply fusion
        # fused = self.fusion(torch.cat([pool_m, pool_e, pool_w], dim=-1))

        # squeeze out the seq dimension
        attn_m = attn_m.squeeze(1)  # (B, H)
        attn_e = attn_e.squeeze(1)
        attn_w = attn_w.squeeze(1)

        # now fuse & classify exactly as before
        concat = torch.cat([attn_m, attn_e, attn_w], dim=-1)  # (B, 3H)
        fused = self.fusion(concat)  # (B, H)
        logits = self.classifier(fused)  # (B, num_labels)

        # Apply the final classification layer
        return logits

# ----------------------------------------
# Dataset & DataLoader for precomputed .npy features using manifest
# ----------------------------------------
class MultiModalDataset(Dataset):
    """
    Loads precomputed audio and text embeddings from a JSONL manifest.

    Each manifest line must include:
      - mfcc_path: path to MFCC numpy array (time, mfcc_dim)
      - egemaps_path: path to eGeMAPS numpy array (time, egemaps_dim)
      - wav2vec2_path: path to Wav2Vec2 numpy array (time, wav2vec_dim)
      - text_path: numpy array (T_t, text_dim)
      - label: int or float
    """
    def __init__(self, manifest_path: str, text_feature: str = "krsbert", use_both_text=False):
        with open(manifest_path, 'r', encoding='utf-8') as f: # Load the manifest file
            self.entries = [json.loads(line) for line in f] # Read the file line by line and parse each line as a JSON object

        self.text_feature = text_feature  # Specify the text feature to be used
        self.use_both_text = use_both_text  # Specify whether to use both text features or not

    def __len__(self):
        """
        Returns the number of entries in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]

        # Load the audio features
        mfcc = torch.from_numpy(np.load(e['mfcc_path'])).float() # Load MFCC features and convert to float32 PyTorch tensor
        egemaps = torch.from_numpy(np.load(e['egemaps_path'])).float() # Load eGeMAPS features and convert to float32 PyTorch tensor
        wav2vec = torch.from_numpy(np.load(e['wav2vec2_path'])).float() # Load Wav2Vec2 features and convert to float32 PyTorch tensor

        # load the text features
        if self.use_both_text:
            kobert_feat = torch.from_numpy(np.load(e['kobert_text_path'])).float() # Load the kobert text features and convert to float32 PyTorch tensor
            krsbert_feat = torch.from_numpy(np.load(e['krsbert_text_path'])).float() # Load the krsbert text features and convert to float32 PyTorch tensor
            text_feats = torch.cat([kobert_feat, krsbert_feat], dim=-1)  # Concatenate as torch tensors
        else:
            text_path = e[f"{self.text_feature}_text_path"]
            text_feats = torch.from_numpy(np.load(text_path)).float() # Load the text features and convert to float32 PyTorch tensor

        # Load the label
        label = torch.tensor(e['label'], dtype=torch.float)

        return {
            'mfcc_feats': mfcc,
            'egemaps_feats': egemaps,
            'wav2vec_feats': wav2vec,
            'text_feats': text_feats,
            'label': label
        }

def collate_fn(batch):
    mfcc = pad_sequence([b['mfcc_feats'] for b in batch], batch_first=True)
    egemaps = pad_sequence([b['egemaps_feats'] for b in batch], batch_first=True)
    wav2vec = pad_sequence([b['wav2vec_feats'] for b in batch], batch_first=True)
    text_feats = pad_sequence([b['text_feats'] for b in batch], batch_first=True)
    labels = torch.stack([b['label'] for b in batch])

    return {
        'mfcc_feats': mfcc,
        'egemaps_feats': egemaps,
        'wav2vec_feats': wav2vec,
        'text_feats': text_feats,
        'label': labels
    }

# DataLoader helper

def get_dataloader(manifest_path: str, batch_size: int = 16,
                   shuffle: bool = True, text_feature="krsbert",
                   use_both_text=False, num_workers: int = 4) -> DataLoader:
    ds = MultiModalDataset(manifest_path, text_feature, use_both_text)  # Create the dataset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)

# Example usage
if __name__ == "__main__":
    # Example of how to use the dataset and dataloader
    train_loader = get_dataloader(
        manifest_path="data/train_segment_manifest_merged.jsonl",
        batch_size=16,
        text_feature="krsbert",
        use_both_text=False
    ) # Create the dataloader for the training set

    # Print the shape of the first batch
    for batch in train_loader:
        print(batch['mfcc_feats'].shape)  # Print the shape of the MFCC features
        print(batch['egemaps_feats'].shape)  # Print the shape of the eGeMAPS features
        print(batch['wav2vec_feats'].shape)  # Print the shape of the Wav2Vec2 features
        print(batch['text_feats'].shape)  # Print the shape of the text features
        print(batch['label'].shape)  # Print the shape of the labels
        break  # Just show one batch for demonstration
