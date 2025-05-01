# train_text_baseline.py
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
from pathlib import Path
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_score, \
    recall_score, roc_curve, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from text_only_dataloader import TextOnlyDataset

# Text classifier model
class TextClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# Training and evaluation functions

def train_model(model, train_loader, val_loader, device, epochs, lr, patience, save_path):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1 = 0
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc, f1, precision, recall, roc, cm = evaluate_model(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss = {total_loss:.4f} | "
            f"Val Acc = {acc:.4f} | "
            f"Val Prec = {precision:.4f} | "
            f"Val Rec = {recall:.4f} | "
            f"Val F1 = {f1:.4f} | "
            f"ROC-AUC = {roc:.4f}"
        )

        scheduler.step(f1) # Reduce learning rate based on validation F1 score

        # Save the best model based on validation F1 score
        if f1 > best_f1:
            best_f1 = f1
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch} with F1 score: {best_f1:.4f}")
        else:
            early_stop_counter += 1
            print(f"❌ No improvement in F1 score. Early stopping count: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"⏹️ Early stopping triggered. Training stopped at epoch {epoch+1}")
            break
    print(f"\nTraining completed. Best F1 score: {best_f1:.4f}")

def evaluate_model(model, dataloader, device, output_dir=None, feature_type=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = torch.sigmoid(model(features)).cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(labels.numpy())

    preds_binary = (np.array(all_preds) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary)
    recall = recall_score(all_labels, preds_binary)
    try:
        roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc = 0.0

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, preds_binary)

    if output_dir:
        # prepare a test results to a JSON file
        test_results = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc,
            "confusion_matrix": cm.tolist()  # Convert numpy array to list for JSON serialization
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"classification_report_{feature_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(classification_report(all_labels, preds_binary, digits=4))
        with open(Path(output_dir) / f"test_results_{feature_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(test_results, f, indent=4)

        # Plot and save ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic_{feature_type}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"roc_curve_{feature_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

        # Plot and save confusion matrix
        cm = confusion_matrix(all_labels, preds_binary)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Test Confusion Matrix_{feature_type}')
        plt.savefig(Path(output_dir) / f"confusion_matrix_{feature_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    return acc, f1, precision, recall, roc, cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_feature", type=str, choices=["kobert", "krsbert", "both"], default="kobert")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4) # Learning rate 1e-3, 2e-4, 1e-4
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    print(f"📦 Training text-only model using: {args.text_feature}, epochs: {args.epochs}, batch size: {args.batch_size}, learning rate: {args.lr}, patience: {args.patience}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(TextOnlyDataset("data/train_segment_manifest_merged.jsonl", args.text_feature), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TextOnlyDataset("data/val_segment_manifest_merged.jsonl", args.text_feature), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TextOnlyDataset("data/test_segment_manifest_merged.jsonl", args.text_feature), batch_size=args.batch_size, shuffle=False)

    dummy_input, _ = next(iter(train_loader))
    input_dim = dummy_input.shape[1]

    model = TextClassifier(input_dim=input_dim).to(device)

    # Print the model architecture
    print("Model architecture:")  # Print the model architecture
    print(model)  # Print the model architecture
    print("-" * 150)  # Print a separator line

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")  # Print the number of trainable parameters
    # print the number of non-trainable parameters
    num_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"Number of non-trainable parameters: {num_non_trainable_params}")  # Print the number of non-trainable parameters
    print("-" * 150)  # Print a separator line

    print("Training the model...")  # Print a message indicating that the model is being trained
    save_model_path = f"modeling/models/best_text_model_{args.text_feature}.pth"
    train_model(model, train_loader, val_loader, device, args.epochs, args.lr, args.patience, save_model_path)

    print("\n📊 Final Test Evaluation:")
    # Load the best model
    model.load_state_dict(torch.load(save_model_path, map_location=device)) # Load the best model
    print(f"Model loaded from {save_model_path}")
    model.to(device) # Move the model to the device

    results_path = f"modeling/logs/eval_results/text_only_{args.text_feature}"
    test_acc, test_f1, test_precision, test_recall, test_roc, test_cm = evaluate_model(model, test_loader, device, output_dir=results_path,
                                                        feature_type=args.text_feature)
    print(f"Test Accuracy: {test_acc:.4f}")  # Print the test accuracy
    print(f"Test F1 Score: {test_f1:.4f}")  # Print the test F1 score
    print(f"Test Precision: {test_precision:.4f}")  # Print the test test_precision
    print(f"Test Recall: {test_recall:.4f}")  # Print the test test_recall
    print(f"Test ROC AUC: {test_roc:.4f}")  # Print the test ROC AUC score
    print("Test Confusion Matrix:")  # Print the confusion matrix for the test set
    print(test_cm)  # Print the confusion matrix
    print("-" * 150)
