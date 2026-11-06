# multimodal_training.py

import argparse
import json
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from multimodal_dataloader import get_dataloader
from multimodal_fusion_model import MultimodalFusionModel

# Define the training function
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        audio_features = batch["audio_features"].to(device)
        text_features = batch["text_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(audio_features, text_features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval() # Set the model to evaluation mode
    running_loss = 0.0
    all_preds = [] # Initialize list for all logits
    all_labels = [] # Initialize list for all labels

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch in dataloader: # Iterate over the batches in the dataloader
            audio_features = batch["audio_features"].to(device) # Move audio features to the device
            text_features = batch["text_features"].to(device) # Move text features to the device
            labels = batch["label"].to(device) # Move labels to the device

            logits = model(audio_features, text_features) # Forward pass
            preds = torch.sigmoid(logits) # Apply sigmoid to get probabilities

            # Store the labels and predictions
            all_preds.extend(preds.cpu().numpy()) # Extend the list with predictions
            all_labels.extend(labels.cpu().numpy()) # Extend the list with labels

    all_preds = np.array(all_preds) # Convert the list of predictions to a numpy array
    all_labels = np.array(all_labels) # Convert the list of labels to a numpy array

    preds_binary = (all_preds > 0.5).astype(int) # Convert probabilities to binary predictions

    # Compute evaluation metrics
    # compute metrics
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, preds_binary) # Compute accuracy
    f1 = f1_score(all_labels, preds_binary) # Compute F1 score
    precision = precision_score(all_labels, preds_binary) # Compute test_precision
    recall = recall_score(all_labels, preds_binary) # Compute test_recall
    try:
        roc_auc = roc_auc_score(all_labels, all_preds) # Compute ROC AUC score usning the probabilities not the binary predictions
    except ValueError:
        roc_auc = 0.0 # Handle the case where ROC AUC cannot be computed

    cm = confusion_matrix(all_labels, preds_binary) # Compute confusion matrix

    return avg_loss, accuracy, f1, precision, recall, roc_auc, cm # Return the evaluation metrics and confusion matrix

# Function to log training performance
def log_epoch_metrics(epoch, num_epochs, train_loss, val_accuracy, val_f1, val_precision, val_recall, val_roc_auc):
    print(
        f"Epoch {epoch}/{num_epochs}: "
        f"Train Loss = {train_loss:.4f} | "
        f"Val Acc = {val_accuracy:.4f} | "
        f"Val F1 = {val_f1:.4f} | "
        f"Val Precision = {val_precision:.4f} | "
        f"Val Recall = {val_recall:.4f} | "
        f"Val ROC AUC = {val_roc_auc:.4f}"
    )

# Main training loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--text_feature", type=str, default="kobert", choices=["kobert", "krsbert"])
    parser.add_argument("--use_both_text", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("-" * 150)

    train_loader = get_dataloader("data/train_segment_manifest_merged.jsonl", batch_size=args.batch_size, shuffle=True, text_feature=args.text_feature, use_both_text=args.use_both_text)
    val_loader = get_dataloader("data/val_segment_manifest_merged.jsonl", batch_size=args.batch_size, shuffle=False, text_feature=args.text_feature, use_both_text=args.use_both_text)
    test_loader = get_dataloader("data/test_segment_manifest_merged.jsonl", batch_size=args.batch_size, shuffle=False, text_feature=args.text_feature, use_both_text=args.use_both_text)

    text_dim = 768 * 2 if args.use_both_text else 768
    model = MultimodalFusionModel(audio_dim=869, text_dim=text_dim, use_both_text=args.use_both_text)
    model.to(device)

    print("Model summary:")
    print(model)
    print("Model parameters (total):", sum(p.numel() for p in model.parameters()))
    print("Model parameters (trainable):", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model parameters (non-trainable):", sum(p.numel() for p in model.parameters() if not p.requires_grad))
    print("-" * 150)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    writer = SummaryWriter(log_dir="modeling/logs/multimodal_experiment")

    best_f1 = 0.0
    early_stop_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy, val_f1, val_precision, val_recall, val_roc_auc, val_cm = evaluate(model, val_loader, device)
        # scheduler.step(train_loss)
        scheduler.step(val_loss)

        log_epoch_metrics(epoch, args.epochs, train_loss, val_accuracy, val_f1, val_precision, val_recall, val_roc_auc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("ROC AUC/val", val_roc_auc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        fig, ax = plt.subplots()
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"Validation Confusion Matrix - Epoch {epoch}")
        writer.add_figure("Confusion Matrix/val", fig, epoch)
        plt.close(fig)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "modeling/models/best_multimodal_model.pth")
            print(f"✅ Best model saved at epoch {epoch} with F1 score: {best_f1:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            print(f"❌ No improvement in F1 score. Early stopping count: {early_stop_count}/{args.patience}")
        if early_stop_count >= args.patience:
            print("⏹️ Early stopping triggered. Training stopped.")
            break

    print("\nTraining completed.")
    print("-" * 150)

    model.load_state_dict(torch.load("modeling/models/best_multimodal_model.pth"))
    test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm = evaluate(model, test_loader, device)

    print("\n🎯 Test Set Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)
    print("-" * 150)

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

    test_results = {
        "accuracy": test_accuracy,
        "f1_score": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "roc_auc": test_roc_auc,
        "confusion_matrix": test_cm.tolist()
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"modeling/logs/eval_results/test_results_{timestamp}.json"
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    with open(results_filename, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {results_filename}")

if __name__ == "__main__":
    main()
