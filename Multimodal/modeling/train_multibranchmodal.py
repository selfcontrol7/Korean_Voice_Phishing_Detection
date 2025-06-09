import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from multibranchmodal_model import (
    MultiBranchCrossModalModel,
    get_dataloader
)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train"):
        mfcc       = batch['mfcc_feats'].to(device)
        egemaps    = batch['egemaps_feats'].to(device)
        wav2vec    = batch['wav2vec_feats'].to(device)
        text_feats = batch['text_feats'].to(device)
        labels     = batch['label'].to(device).float()

        optimizer.zero_grad()
        logits = model(mfcc, egemaps, wav2vec, text_feats).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mfcc.size(0) # Accumulate loss over the batch

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []

    for batch in tqdm(loader, desc="Validate (Evaluate)"):
        mfcc       = batch['mfcc_feats'].to(device)
        egemaps    = batch['egemaps_feats'].to(device)
        wav2vec    = batch['wav2vec_feats'].to(device)
        text_feats = batch['text_feats'].to(device)
        labels     = batch['label'].to(device).float()

        logits = model(mfcc, egemaps, wav2vec, text_feats).squeeze(-1)
        loss = criterion(logits, labels)
        running_loss += loss.item() * mfcc.size(0)

        prob = torch.sigmoid(logits).cpu().numpy()
        preds.append((prob >= 0.5).astype(int))
        trues.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    f1 = f1_score(trues, preds)
    p  = precision_score(trues, preds)
    r  = recall_score(trues, preds)

    return avg_loss, f1, p, r


def train_and_evaluate(args):
    # device & model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("-" * 150)

    model  = MultiBranchCrossModalModel().to(device)
    print("Model summary")
    print(f"Model: {model.__class__.__name__}")
    print(model)
    print("Model parameters (total):", sum(p.numel() for p in model.parameters()))
    print("Model parameters (trainable):", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model parameters (non-trainable):", sum(p.numel() for p in model.parameters() if not p.requires_grad))
    print("-" * 150)

    # data
    train_loader = get_dataloader(
        manifest_path="data/train_segment_manifest_merged.jsonl",
        batch_size=args.batch_size,
        shuffle=True,
        text_feature=args.text_feature,
        use_both_text=args.use_both_text
    )
    val_loader = get_dataloader(
        manifest_path="data/val_segment_manifest_merged.jsonl",
        batch_size=args.batch_size,
        shuffle=False,
        text_feature=args.text_feature,
        use_both_text=args.use_both_text
    )
    test_loader = get_dataloader(
        manifest_path="data/test_segment_manifest_merged.jsonl",
        batch_size=args.batch_size,
        shuffle=False,
        text_feature=args.text_feature,
        use_both_text=args.use_both_text
    )

    # optimize + loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # logging
    writer = SummaryWriter(log_dir="modeling/logs/multibranchmodal_experiment")
    best_f1 = 0.0
    no_improve_epochs = 0
    patience = args.early_stop_patience
    # os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_p, val_r = validate(model, val_loader, criterion, device)

        # TensorBoard logs
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss",   val_loss,   epoch)
        writer.add_scalar("val/f1",     val_f1,     epoch)
        writer.add_scalar("val/precision", val_p,    epoch)
        writer.add_scalar("val/recall",    val_r,    epoch)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} ┃ F1: {val_f1:.4f}, P: {val_p:.4f}, R: {val_r:.4f}")

        # early stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve_epochs = 0
            # ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
            # torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), "modeling/models/best_multibranchmodal_model.pt")
            print(f"✅→ New best model saved at epoch {epoch} with F1 score: {best_f1:.4f}")
        else:
            no_improve_epochs += 1
            print(f"❌→ No improvement in F1 score for {no_improve_epochs}/{patience} epochs")
            if no_improve_epochs >= patience:
                print(f"⏹️ Early stopping triggered (no improvement in {patience} epochs).")
                break

    writer.close()
    print(f"\nTraining completed. Best Val F1: {best_f1:.4f}")
    print("-" * 150)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Multi-Branch Cross-Modal Model")
    # p.add_argument("--train_manifest",        type=str, required=True)
    # p.add_argument("--val_manifest",          type=str, required=True)
    # p.add_argument("--ckpt_dir",              type=str, default="checkpoints")
    # p.add_argument("--log_dir",               type=str, default="runs")
    p.add_argument("--epochs",                type=int,   default=15)
    p.add_argument("--batch_size",            type=int,   default=16)
    p.add_argument("--lr",                    type=float, default=2e-5)
    p.add_argument("--text_feature",          type=str,   default="krsbert")
    p.add_argument("--use_both_text",         action="store_true")
    p.add_argument("--early_stop_patience",   type=int,   default=5,
                   help="stop training if no val F1 improvement for this many epochs")
    p.add_argument("--device",                type=str,   default=None)
    args = p.parse_args()

    train_and_evaluate(args)
