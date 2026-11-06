# train_multimodal.py

import json
import datetime
import os
import argparse

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
from pathlib import Path

# Define the training function
def train(model, dataloader, optimizer, loss_fn, device):
    model.train() # Set the model to training mode
    total_loss = 0.0 # Initialize total loss

    for batch in dataloader: # Iterate over the batches in the dataloader
        audio_features = batch["audio_features"].to(device) # Move audio features to the device
        text_features = batch["text_features"].to(device) # Move text features to the device
        labels = batch["label"].to(device) # Move labels to the device

        optimizer.zero_grad() # Zero the gradients before the backward pass
        logits = model(audio_features, text_features) # Forward pass
        loss = loss_fn(logits, labels) # Compute the loss
        loss.backward() # Backward pass
        optimizer.step() # Update the model parameters

        total_loss += loss.item() # Accumulate the total loss

    return total_loss / len(dataloader) # Return the average loss over the dataloader

# Define the evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            audio_features = batch["audio_features"].to(device)
            text_features = batch["text_features"].to(device)
            labels = batch["label"].to(device)

            logits = model(audio_features, text_features)
            preds = torch.sigmoid(logits)

            # Store the labels and predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    preds_binary = (all_preds > 0.5).astype(int)

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary)
    recall = recall_score(all_labels, preds_binary)
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc_auc = 0.0

    cm = confusion_matrix(all_labels, preds_binary)

    return accuracy, f1, precision, recall, roc_auc, cm

# Function to print the metrics for each epoch
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

def save_test_results(test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm, output_dir, text_feature):
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
    results_filename = f"{output_dir}/test_results_{text_feature}_{timestamp}.json"
    
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
    ax.set_title(f"Test Confusion Matrix - {text_feature}")
    plt.savefig(f"{output_dir}/confusion_matrix_{text_feature}_{timestamp}.png")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multimodal fusion model for phishing detection")
    
    # Data parameters
    parser.add_argument("--train_manifest", type=str, default="data/train_segment_manifest_merged.jsonl", 
                        help="Path to the training manifest file")
    parser.add_argument("--val_manifest", type=str, default="data/val_segment_manifest_merged.jsonl", 
                        help="Path to the validation manifest file")
    parser.add_argument("--test_manifest", type=str, default="data/test_segment_manifest_merged.jsonl", 
                        help="Path to the test manifest file")
    
    # Model parameters
    parser.add_argument("--text_feature", type=str, choices=["kobert", "krsbert", "both"], default="kobert",
                        help="Text feature type to use (kobert, krsbert, or both)")
    parser.add_argument("--audio_dim", type=int, default=869, 
                        help="Dimension of the audio features")
    parser.add_argument("--text_dim", type=int, default=768, 
                        help="Dimension of the text features")
    parser.add_argument("--hidden_dim", type=int, default=512, 
                        help="Dimension of the hidden layers")
    parser.add_argument("--dropout", type=float, default=0.3, 
                        help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=2e-4, 
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Patience for early stopping")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, 
                        help="Factor by which the learning rate will be reduced")
    parser.add_argument("--scheduler_patience", type=int, default=5, 
                        help="Number of epochs with no improvement after which learning rate will be reduced")
    
    # Output parameters
    parser.add_argument("--model_dir", type=str, default="modeling/models", 
                        help="Directory to save the trained model")
    parser.add_argument("--log_dir", type=str, default="modeling/logs/multimodal_experiment", 
                        help="Directory for TensorBoard logs")
    parser.add_argument("--results_dir", type=str, default="modeling/logs/eval_results", 
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Print training configuration
    print(f"📦 Training multimodal model with the following configuration:")
    print(f"  - Text feature: {args.text_feature}")
    print(f"  - Audio dimension: {args.audio_dim}")
    print(f"  - Text dimension: {args.text_dim}")
    print(f"  - Hidden dimension: {args.hidden_dim}")
    print(f"  - Dropout rate: {args.dropout}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Early stopping patience: {args.patience}")
    print(f"  - Scheduler factor: {args.scheduler_factor}")
    print(f"  - Scheduler patience: {args.scheduler_patience}")
    print("-" * 100)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("-" * 100)
    
    # Determine if we're using both text features
    use_both_text = args.text_feature == "both"
    
    # Create the dataloaders
    train_loader = get_dataloader(
        args.train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        text_feature=args.text_feature if not use_both_text else "kobert",
        use_both_text=use_both_text
    )
    
    val_loader = get_dataloader(
        args.val_manifest,
        batch_size=args.batch_size,
        shuffle=False,
        text_feature=args.text_feature if not use_both_text else "kobert",
        use_both_text=use_both_text
    )
    
    test_loader = get_dataloader(
        args.test_manifest,
        batch_size=args.batch_size,
        shuffle=False,
        text_feature=args.text_feature if not use_both_text else "kobert",
        use_both_text=use_both_text
    )
    
    # Initialize the model
    model = MultimodalFusionModel(
        audio_dim=args.audio_dim,
        text_dim=args.text_dim,
        use_both_text=use_both_text,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model.to(device)
    
    # Print model information
    print("Model architecture:")
    print(model)
    print("-" * 100)
    
    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    num_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Number of non-trainable parameters: {num_non_trainable_params}")
    print("-" * 100)
    
    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize the loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create model save path
    model_save_path = f"{args.model_dir}/best_multimodal_model_{args.text_feature}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Create results directory
    results_dir = f"{args.results_dir}/multimodal_{args.text_feature}"
    os.makedirs(results_dir, exist_ok=True)
    
    best_f1 = 0.0
    early_stop_count = 0
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_accuracy, val_f1, val_precision, val_recall, val_roc_auc, val_cm = evaluate(model, val_loader, device)
        
        # Step the learning rate scheduler based on the training loss
        scheduler.step(train_loss)
        
        # Log the metrics for the current epoch
        log_epoch_metrics(epoch, args.epochs, train_loss, val_accuracy, val_f1, val_precision, val_recall, val_roc_auc)
        
        # Log the metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("ROC AUC/val", val_roc_auc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Log the confusion matrix to TensorBoard
        fig, ax = plt.subplots()
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"Validation Confusion Matrix - Epoch {epoch}")
        writer.add_figure("Confusion Matrix/val", fig, epoch)
        plt.close(fig)
        
        # Save the model if the F1 score has improved
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Best model saved at epoch {epoch} with F1 score: {best_f1:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            print(f"❌ No improvement in F1 score. Early stopping count: {early_stop_count}/{args.patience}")
            
        if early_stop_count >= args.patience:
            print("⏹️ Early stopping triggered. Training stopped.")
            break
            
    print("\nTraining completed.")
    print("-" * 100)
    
    # Load the best model and evaluate it on the test set
    model.load_state_dict(torch.load(model_save_path))
    test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm = evaluate(model, test_loader, device)
    
    print(f"\n🎯 Test Set Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)
    print("-" * 100)
    
    # Save test results
    save_test_results(
        test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm,
        results_dir, args.text_feature
    )
    
    # Close the TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to {args.log_dir}")
    print(f"Model saved to {model_save_path}")
    print(f"Test set evaluation completed.")
    print("-" * 100)