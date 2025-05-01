# train_audio_baseline.py
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_score, \
    recall_score, confusion_matrix, roc_curve
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

# import the custom dataset
from audio_only_dataloader import AudioOnlyDataset


# Simple classifier for audio features
class AudioClassifier(nn.Module):
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

# Training and evaluation
def train_model(model, train_loader, val_loader, device, epochs, lr, save_path):
    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr) # Initialize the optimizer with Adam optimizer and the specified learning rate

    # Initialize scheduler if needed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1 = 0
    early_stop_count = 0 # Initialize the early stopping counter
    patience = 5 # Number of epochs to wait for improvement before stopping

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device) # Move data to device

            optimizer.zero_grad() # Reset gradients
            outputs = model(features) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            total_loss += loss.item() # Accumulate loss

        # evaluate the model on the validation set
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

        # Update the learning rate based on validation F1 score
        scheduler.step(f1) # Reduce learning rate if no improvement in F1 score

        # Save the best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1 # Update the best F1 score
            torch.save(model.state_dict(), save_path) # Save the model state
            print(f"✅ Best model saved at epoch {epoch} with F1 score: {best_f1:.4f}")
        else:
            early_stop_count += 1 # Increment early stopping counter
            print(f"❌ No improvement in F1 score. Early stopping count: {early_stop_count}/{patience}")

        if early_stop_count >= patience: # If no improvement for 5 epochs, stop training
            print(f"⏹️ Early stopping triggered. Training stopped at epoch {epoch+1}")
            break
    print(f"\nTraining completed. Best F1 score: {best_f1:.4f}")

def evaluate_model(model, dataloader, device, output_dir=None, feature_type=None):
    model.eval() # Set the model to evaluation mode
    all_preds, all_labels = [], [] # Initialize lists to store predictions and labels

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for features, labels in dataloader: # Iterate over the dataloader
            features = features.to(device)
            outputs = torch.sigmoid(model(features)).cpu().numpy()
            all_preds.extend(outputs) # Store predictions
            all_labels.extend(labels.numpy()) # Store labels

    preds_binary = (np.array(all_preds) >= 0.5).astype(int) # Convert predictions to binary values

    # Compute evaluation metrics
    acc = accuracy_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary)
    recall = recall_score(all_labels, preds_binary)
    try:
        roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc = 0.0

    # confusion matrix
    cm = confusion_matrix(all_labels, preds_binary)

    # Print evaluation metrics and plots in case of test set
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
        plt.title('Test Confusion Matrix')
        plt.savefig(Path(output_dir) / f"confusion_matrix_{feature_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    # return the evaluation metrics
    return acc, f1, precision, recall, roc, cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, choices=["mfcc", "egemaps", "wav2vec2", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4) # Learning rate 1e-3, 2e-4, 1e-4
    args = parser.parse_args() # Parse the command line arguments

    print(f"📦 Training with feature type: {args.feature_type}, epochs: {args.epochs}, batch size: {args.batch_size}, learning rate: {args.lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    train_loader = DataLoader(AudioOnlyDataset("data/train_segment_manifest_merged.jsonl", args.feature_type), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AudioOnlyDataset("data/val_segment_manifest_merged.jsonl", args.feature_type), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(AudioOnlyDataset("data/test_segment_manifest_merged.jsonl", args.feature_type), batch_size=args.batch_size, shuffle=False)

    # Get input dimension from the first batch of the training data
    dummy_input, _ = next(iter(train_loader))
    # Assuming the first batch is representative of the input size
    input_dim = dummy_input.shape[1]
    print("-" * 150)  # Print a separator line

    # Initialize the model
    model = AudioClassifier(input_dim=input_dim).to(device) # Create an instance of the model

    # Print the model architecture
    print("Model architecture:")  # Print the model architecture
    print(model)  # Print the model architecture
    print("-" * 150)  # Print a separator line

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")  # Print the number of trainable parameters
    # print the number of non-trainable parameters
    num_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Number of non-trainable parameters: {num_non_trainable_params}")  # Print the number of non-trainable parameters
    print("-" * 150)  # Print a separator line

    print("Training the model...")  # Print a message indicating that the model is being trained
    save_model_path = f"modeling/models/best_audio_model_{args.feature_type}.pth"
    train_model(model, train_loader, val_loader, device, args.epochs, args.lr, save_model_path)
    print("=" * 150)  # Print a separator line

    print("\n📊 Final Test Evaluation:")
    # Load the best model
    model.load_state_dict(torch.load(save_model_path, map_location=device)) # Load the best model
    print(f"Model loaded from {save_model_path}")
    model.to(device) # Move the model to the device

    results_path = f"modeling/logs/eval_results/audio_only_{args.feature_type}"
    # evaluate the model on the test set
    test_acc, test_f1, test_precision, test_recall, test_roc, test_cm = evaluate_model(model, test_loader, device, output_dir=results_path, feature_type=args.feature_type)
    print(f"Test Accuracy: {test_acc:.4f}")  # Print the test accuracy
    print(f"Test F1 Score: {test_f1:.4f}")  # Print the test F1 score
    print(f"Test Precision: {test_precision:.4f}")  # Print the test test_precision
    print(f"Test Recall: {test_recall:.4f}")  # Print the test test_recall
    print(f"Test ROC AUC: {test_roc:.4f}")  # Print the test ROC AUC score
    print("Test Confusion Matrix:")  # Print the confusion matrix for the test set
    print(test_cm)  # Print the confusion matrix
    print("-" * 150)