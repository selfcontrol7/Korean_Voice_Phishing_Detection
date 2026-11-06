import torch
import torch.nn as nn


class MultiBranchCrossModalModelConfigurable(nn.Module):
    """
    The new “selective‐branch” model that takes booleans
    use_mfcc/use_egemaps/use_wav2vec in its constructor.
    """
    def __init__(self,
                 mfcc_dim=13, egemaps_dim=88, wav2vec_dim=768,
                 text_dim=768, proj_dim=256, num_heads=8,
                 fusion_dim=256, dropout=0.2,
                 use_mfcc=True, use_egemaps=True, use_wav2vec=True):
        super().__init__()
        self.use_mfcc    = use_mfcc
        self.use_egemaps = use_egemaps
        self.use_wav2vec = use_wav2vec

        # instantiate only what’s needed
        if use_mfcc:
            self.mfcc_proj = nn.Sequential(
                nn.Linear(mfcc_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.attn_mfcc = nn.MultiheadAttention(
                embed_dim=proj_dim, num_heads=num_heads, batch_first=True
            )
        if use_egemaps:
            self.egemaps_proj = nn.Sequential(
                nn.Linear(egemaps_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.attn_egemaps = nn.MultiheadAttention(
                embed_dim=proj_dim, num_heads=num_heads, batch_first=True
            )
        if use_wav2vec:
            self.wav2vec_proj = nn.Sequential(
                nn.Linear(wav2vec_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.attn_wav2vec = nn.MultiheadAttention(
                embed_dim=proj_dim, num_heads=num_heads, batch_first=True
            )

        # text branch always on
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # fusion dims adjust to # active branches
        n_branches = sum((use_mfcc, use_egemaps, use_wav2vec))
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * n_branches, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim, 1)

    def forward(self, mfcc_feats, egemaps_feats, wav2vec_feats, text_feats):
        # project text → (B, H) then unsqueeze → (B,1,H)
        text_p = self.text_proj(text_feats)
        q = text_p.unsqueeze(1)

        outs = []
        if self.use_mfcc:
            m = self.mfcc_proj(mfcc_feats)
            a, _ = self.attn_mfcc(q, m, m)
            outs.append(a.squeeze(1))
        if self.use_egemaps:
            e = self.egemaps_proj(egemaps_feats)
            b, _ = self.attn_egemaps(q, e, e)
            outs.append(b.squeeze(1))
        if self.use_wav2vec:
            w = self.wav2vec_proj(wav2vec_feats)
            c, _ = self.attn_wav2vec(q, w, w)
            outs.append(c.squeeze(1))

        x = torch.cat(outs, dim=-1)
        x = self.fusion(x)
        return self.classifier(x)