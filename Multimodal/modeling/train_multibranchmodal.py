import json
import os
import datetime
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
from tqdm import tqdm

from multibranchmodal_model import (
    MultiBranchCrossModalModel,
    FlexibleMultiBranchCrossModalModel,
    get_dataloader
)

import logging
logger = logging.getLogger(__name__)

# Configure logging to print INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def make_modality_tag(model_variant, use_mfcc, use_egemaps, use_wav2vec):
    tags = []
    if model_variant == "original":
        tags.append("mfcc")
        tags.append("egemaps")
        tags.append("wav2vec")
    elif model_variant == "flexible":
        if not use_mfcc and not use_egemaps and not use_wav2vec:
            tags.append("none")
        else:
            if use_mfcc:    tags.append("mfcc")
            if use_egemaps: tags.append("egemaps")
            if use_wav2vec: tags.append("wav2vec")
    # if you ever want to see “none”, uncomment next line
    # if not tags: tags = ["none"]
    return "_".join(tags)

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
    preds, trues, probs = [], [], []

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
        pred = (prob >= 0.5).astype(int)

        probs.append(prob)
        preds.append(pred)
        trues.append(labels.cpu().numpy())

    # flatten lists
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues) # Concatenate probabilities

    # compute metrics
    avg_loss = running_loss / len(loader.dataset)
    f1       = f1_score(trues, preds)
    p        = precision_score(trues, preds)
    r        = recall_score(trues, preds)
    acc      = accuracy_score(trues, preds)
    conf_mat = confusion_matrix(trues, preds)
    roc_auc  = roc_auc_score(trues, probs)

    return avg_loss, f1, p, r, acc, conf_mat, roc_auc

# Function to log training performance
def log_epoch_metrics(epoch: int,
                      num_epochs: int,
                      train_loss: float,
                      val_loss: float,
                      **metrics: float):
    """
    Logs epoch metrics.

    :param epoch: current epoch index (1-based)
    :param num_epochs: total number of epochs
    :param train_loss: training loss for this epoch
    :param val_loss: validation loss for this epoch
    :param metrics: other metrics as name=value pairs (e.g. accuracy=0.85, f1=0.76)
    """
    metrics_str_print = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(
        f"Epoch {epoch}/{num_epochs}: "
        f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}"
        + (f" | {metrics_str_print}" if metrics_str_print else "")
    )

    metrics_str = " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    message = (
        f"Epoch {epoch}/{num_epochs} — "
        f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}"
        + (f" | {metrics_str}" if metrics_str else "")
    )
    logger.info(message)


def save_test_results(test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm, output_dir,
                      modality_tag):
    # Save the test results to a JSON file
    test_results = {
        "accuracy": test_accuracy,
        "f1_score": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "roc_auc": test_roc_auc,
        "confusion_matrix": test_cm.tolist()
    }

    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{output_dir}/test_results_{modality_tag}_{timestamp}.json"

    # Ensure that the target directory exists
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)

    with open(results_filename, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {results_filename}")

    # Plot and save confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(f"Test Confusion Matrix - {modality_tag}")
    plt.savefig(f"{output_dir}/confusion_matrix_{modality_tag}_{timestamp}.png")
    plt.close(fig)


def train_and_evaluate(args):
    # print the arguments
    print("-" * 150)
    print("Training Multi-Branch Cross-Modal Model with the following parameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 150)

    # device & model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("-" * 150)

    if args.model_variant == "original":
        model = MultiBranchCrossModalModel().to(device)
    elif args.model_variant == "flexible":
        # Each argument is set to the opposite of the corresponding args.no_* flag,
        # so if args.no_mfcc is True, then use_mfcc will be False, disabling MFCC features.
        # This allows for flexible model configuration based on command-line arguments.
        model = FlexibleMultiBranchCrossModalModel(
            use_mfcc=not args.no_mfcc,
            use_egemaps=not args.no_egemaps,
            use_wav2vec=not args.no_wav2vec,
        ).to(device)

    print(f"Model summary for {args.model_variant} variant:")
    print(f"Model: {model.__class__.__name__}")
    modality_tag = make_modality_tag(args.model_variant,not args.no_mfcc,not args.no_egemaps,not args.no_wav2vec)
    print(f"Modality tag: {modality_tag}")
    print(model)
    print("-" * 150)
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
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    # Reduce LR by half if validation loss hasn’t improved for 2 epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # we want to *minimize* the monitored metric (val loss)
        factor=0.5,  # new_lr = old_lr * factor
        patience=5,  # wait this many epochs with no improvement
        verbose=True  # print a message when LR is reduced
    )

    # logging
    writer = SummaryWriter(log_dir="modeling/logs/multibranchmodal_experiment")
    best_f1 = 0.0
    no_improve_epochs = 0
    patience = args.early_stop_patience
    # os.makedirs(args.ckpt_dir, exist_ok=True)

    # Create results directory
    results_dir = f"{args.results_dir}/multimodal_{args.text_feature}"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_p, val_r, val_acc, val_cm, val_roc_auc = validate(model, val_loader, criterion, device)
        # scheduler.step(train_loss)
        # Tell the scheduler your validation loss
        scheduler.step(val_loss)

        # log_epoch_metrics(epoch, args.epochs, train_loss, val_acc, val_f1, val_precision, val_recall, val_roc_auc)
        log_epoch_metrics(
            epoch, args.epochs,
            train_loss, val_loss,
            accuracy=val_acc,
            f1=val_f1,
            precision=val_p,
            recall=val_r,
            roc_auc=val_roc_auc
        )

        # TensorBoard logs
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/Precision", val_p, epoch)
        writer.add_scalar("Val/Recall", val_r, epoch)
        writer.add_scalar("Val/ROC AUC", val_roc_auc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        fig, ax = plt.subplots()
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"Validation Confusion Matrix - Epoch {epoch}")
        writer.add_figure("Confusion Matrix/val", fig, epoch)
        plt.close(fig)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} ┃ F1: {val_f1:.4f}, P: {val_p:.4f}, R: {val_r:.4f}")

        # early stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve_epochs = 0 # reset the early stop counter if we have a new best F1 score

            if not os.path.exists("modeling/models"):
                os.makedirs("modeling/models")
            # Save the model state dict to a file with epoch number
            # Save the best model with epoch number in the filename
            # if args.model_variant == "original":
            #     ckpt_path = f"{args.ckpt_dir}/best_multibranchmodal_mfcc_egemaps_wav2vec_{args.epochs}.pt"
            #     torch.save(model.state_dict(), ckpt_path)
            if args.model_variant == "flexible" or args.model_variant == "original":
                # Save the model state dict to a file with epoch number and modality tags
                ckpt_path = f"{args.ckpt_dir}/best_multibranchmodal_{modality_tag}_{args.epochs}.pt"
                torch.save(model.state_dict(), ckpt_path)
            else:
                raise ValueError(f"Unknown model variant: {args.model_variant}. "
                                 f"Expected 'original' or 'flexible', got '{args.model_variant}'.")
            # torch.save(model.state_dict(), f"modeling/models/best_multibranchmodal_model_{args.epochs}.pt")
            print(f"✅→ New best model saved as {ckpt_path} at epoch {epoch} with F1 score: {best_f1:.4f}")

        else:
            no_improve_epochs += 1
            print(f"❌→ No improvement in F1 score for {no_improve_epochs}/{patience} epochs")

        # check early‐stop after updating the counter
        if no_improve_epochs >= patience:
            print(f"⏹️ Early stopping triggered (no improvement in {patience} epochs).")
            break

    print(f"\nTraining completed. Best Val F1: {best_f1:.4f}")
    print("-" * 150)

    # Evaluate on the test set
    # model.load_state_dict(torch.load(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_f1, test_precision, test_recall, test_accuracy, test_cm, test_roc_auc = validate(
        model, test_loader, criterion, device)
    # test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm = evaluate(model, test_loader, device)

    # log_epoch_metrics(
    #     epoch, args.epochs,
    #     train_loss, val_loss,
    #     accuracy=val_acc,
    #     f1=val_f1,
    #     precision=val_p,
    #     recall=val_r,
    #     roc_auc=val_roc_auc
    # )

    print("\n🎯 Test Set Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)
    print("-" * 150)

    # Save test results
    save_test_results(
        test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm,
        results_dir, modality_tag
    )

    fig, ax = plt.subplots()
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Test Confusion Matrix")
    writer.add_figure('ConfusionMatrix/test', fig)
    plt.show()
    plt.close(fig)

    writer.close()
    print("TensorBoard logs saved to modeling/logs/multimodal_experiment")
    print("Model saved to modeling/models/best_multimodal_model.pth")
    print("Test set evaluation completed.")
    print("-" * 150)

    # test_results = {
    #     "accuracy": test_accuracy,
    #     "f1_score": test_f1,
    #     "test_precision": test_precision,
    #     "test_recall": test_recall,
    #     "roc_auc": test_roc_auc,
    #     "confusion_matrix": test_cm.tolist()
    # }
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_filename = f"modeling/logs/eval_results/multibranchmodal/test_results_{timestamp}.json"
    # os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    # with open(results_filename, "w") as f:
    #     json.dump(test_results, f, indent=4)
    # print(f"Test results saved to {results_filename}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Multi-Branch Cross-Modal Model")
    p.add_argument(
        "--model_variant",
        choices=["original", "flexible"],
        default="flexible",
        help="Which model class to use"
    )
    p.add_argument("--no_mfcc", action="store_true",
                   help="🚫 disable MFCC branch")
    p.add_argument("--no_egemaps", action="store_true",
                   help="🚫 disable eGeMAPS branch")
    p.add_argument("--no_wav2vec", action="store_true",
                   help="🚫 disable Wav2Vec2 branch")
    p.add_argument(
        "--epochs",
                   type=int,
                   default=20)
    p.add_argument(
        "--batch_size",
        type=int,
        default=16)
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4)
    p.add_argument(
        "--text_feature",
        type=str,
        default="krsbert")
    p.add_argument(
        "--use_both_text",
        action="store_true")
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="stop training if no val F1 improvement for this many epochs")
    p.add_argument(
        "--device",
        type=str,
        default=None)
    p.add_argument(
        "--results_dir",
        type=str,
        default="modeling/logs/eval_results/multibranchmodal",
        help="Directory to save evaluation results")
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default="modeling/models",
        help="Directory to save model checkpoints")
    args = p.parse_args()

    train_and_evaluate(args)
