# multimodal_training.py

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # Importing the PyTorch library for optimization
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix # Importing metrics for evaluation
import matplotlib.pyplot as plt # Importing matplotlib for plotting
import seaborn as sns # Importing seaborn for enhanced visualization
from torch.utils.tensorboard import SummaryWriter # Importing TensorBoard for logging
from multimodal_dataloader import get_dataloader # Importing the dataloader
from multimodal_fusion_model import MultimodalFusionModel # Importing the fusion model

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
    model.eval() # Set the model to evaluation mode
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
    accuracy = accuracy_score(all_labels, preds_binary) # Compute accuracy
    f1 = f1_score(all_labels, preds_binary) # Compute F1 score
    precision = precision_score(all_labels, preds_binary) # Compute precision
    recall = recall_score(all_labels, preds_binary) # Compute recall
    try:
        roc_auc = roc_auc_score(all_labels, all_preds) # Compute ROC AUC score usning the probabilities not the binary predictions
    except ValueError:
        roc_auc = 0.0 # Handle the case where ROC AUC cannot be computed

    cm = confusion_matrix(all_labels, preds_binary) # Compute confusion matrix

    return accuracy, f1, precision, recall, roc_auc, cm # Return the evaluation metrics and confusion matrix

# Function to print the metrics for each epoch
def log_epoch_metrics(epoch, num_epochs, train_loss, val_accuracy, val_f1, val_precision, val_recall, val_roc_auc):
    """
    Logs the performance metrics for a specific epoch during the training process. This function outputs details
    about the training loss and validation metrics including accuracy, F1 score, precision, recall, and ROC AUC,
    allowing for tracking and monitoring of model performance over epochs.

    Args:
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs for the training process.
        train_loss (float): The loss value for the training set.
        val_accuracy (float): The accuracy value for the validation set.
        val_f1 (float): The F1 score for the validation set.
        val_precision (float): The precision value for the validation set.
        val_recall (float): The recall value for the validation set.
        val_roc_auc (float): The ROC AUC score for the validation set.

    Returns:
        None
    """
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
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, otherwise use CPU
    print(f"Using device: {device}") # Print the device being used

    # Create the training dataloader
    train_loader = get_dataloader(
        "data/train_segment_manifest_merged.jsonl",
        batch_size=16,
        shuffle=True,
        text_feature="krsbert",
        use_both_text=False
    )

    # Create the validation dataloader
    val_loader = get_dataloader(
        "data/val_segment_manifest_merged.jsonl",
        batch_size=16,
        shuffle=False,
        text_feature="krsbert",
        use_both_text=False
    )

    # Create the test dataloader
    test_loader = get_dataloader(
        "data/test_segment_manifest_merged.jsonl",
        batch_size=16,
        shuffle=False,
        text_feature="krsbert",
        use_both_text=False
    )

    # Initialize the model
    model = MultimodalFusionModel(
        audio_dim=869, # Dimension of the audio features
        text_dim=768, # Dimension of the text features
        use_both_text=False # Use only one text feature
    )
    model.to(device) # Move the model to the device

    print("Model summary:") # Print a summary of the model
    print("Model architecture:", model) # Print the model architecture
    print("Model input shape:", (16, 869 + 768)) # Print the input shape of the model
    print("Model output shape:", (16, 1)) # Print the output shape of the model
    print("Model parameters (audio):", model.audio_projection) # Print the audio projection layer
    print("Model parameters (text):", model.text_projection) # Print the text projection layer
    print("Model parameters (fusion):", model.fusion_layer) # Print the fusion layer
    print("Model parameters (classifier):", model.classifier) # Print the classifier layer
    print("Model parameters (dropout):", model.droupout) # Print the dropout layer
    print("Model parameters (relu):", model.relu) # Print the ReLU activation function
    print("Model parameters (total):", sum(p.numel() for p in model.parameters())) # Print the total number of parameters in the model
    print("-" * 150)  # Print a separator line

    print("Model parameters:", sum(p.numel() for p in model.parameters())) # Print the number of parameters in the model
    print("Model parameters (trainable):", sum(p.numel() for p in model.parameters() if p.requires_grad)) # Print the number of trainable parameters in the model
    print("Model parameters (non-trainable):", sum(p.numel() for p in model.parameters() if not p.requires_grad)) # Print the number of non-trainable parameters in the model
    print("-" * 150) # Print a separator line

    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-4) # AdamW optimizer with learning rate of 1e-4
    # Initialize the loss function
    loss_fn = nn.BCEWithLogitsLoss() # Binary Cross-Entropy with logits loss function
    # Initialize the learning rate scheduler to reduce the learning rate when a metric has stopped improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, # Learning rate scheduler
        mode='min', # Mode to monitor
        factor=0.5, # Factor by which the learning rate will be reduced
        patience=5, # Number of epochs with no improvement after which learning rate will be reduced
        verbose=True # Verbose output
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="modeling/logs/multimodal_experiment") # Create a TensorBoard writer to log training and validation metrics

    num_epochs = 20 # Number of epochs to train the model
    best_f1 = 0.0 # Initialize the best F1 score
    early_stop_count = 0 # Initialize the early stopping counter
    patience = 5 # Number of epochs to wait for improvement before stopping

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_accuracy, val_f1, val_precision, val_recall, val_roc_auc, val_cm = evaluate(model, val_loader, device) # Evaluate the model on the validation set

        # Step the learning rate scheduler based on the training loss
        scheduler.step(train_loss)

        # Log the metrics for the current epoch
        log_epoch_metrics(epoch, num_epochs, train_loss, val_accuracy, val_f1, val_precision, val_recall, val_roc_auc)

        # Log the metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("ROC AUC/val", val_roc_auc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # Log the confusion matrix to TensorBoard
        fig, ax = plt.subplots() # Create a figure and axis for the confusion matrix
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax) # Plot the confusion matrix
        ax.set_xlabel("Predicted labels") # Set the x-axis label
        ax.set_ylabel("True labels") # Set the y-axis label
        ax.set_title(f"Validation Confusion Matrix - Epoch {epoch}") # Set the title of the plot
        writer.add_figure("Confusion Matrix/val", fig, epoch) # Log the confusion matrix to TensorBoard
        plt.close(fig) # Close the figure to avoid display

        # Save the model if the F1 score has improved
        if val_f1 > best_f1:
            best_f1 = val_f1 # Update the best F1 score
            torch.save(model.state_dict(), "modeling/models/best_multimodal_model.pth") # Save the model state
            print(f"✅ Best model saved at epoch {epoch} with F1 score: {best_f1:.4f}")
            early_stop_count = 0 # Reset the early stopping counter
        else:
            early_stop_count += 1 # Increment the early stopping counter
            print(f"❌ No improvement in F1 score. Early stopping count: {early_stop_count}/{patience}")
        if early_stop_count >= patience:
            print("⏹️ Early stopping triggered. Training stopped.")
            break # Stop training if early stopping is triggered

    # Close the TensorBoard writer
    writer.close() # Close the TensorBoard writer

    # Load the best model and evaluate it on the test set
    model.load_state_dict(torch.load("modeling/models/best_multimodal_model.pth")) # Load the best model state
    test_accuracy, test_f1, test_precision, test_recall, test_roc_auc, test_cm = evaluate(model, test_loader, device) # Evaluate the model on the test set

    print(f"\n🎯 Test Set Performance:")  # Print the evaluation metrics for the test set
    print(f"Test Accuracy: {test_accuracy:.4f}") # Print the test accuracy
    print(f"Test F1 Score: {test_f1:.4f}") # Print the test F1 score
    print(f"Test Precision: {test_precision:.4f}") # Print the test precision
    print(f"Test Recall: {test_recall:.4f}") # Print the test recall
    print(f"Test ROC AUC: {test_roc_auc:.4f}") # Print the test ROC AUC score
    print("Test Confusion Matrix:") # Print the confusion matrix for the test set
    print(test_cm) # Print the confusion matrix
    print("-" * 150)

    # Plot the confusion matrix
    fig, ax = plt.subplots() # Create a figure and axis for the confusion matrix
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax) # Plot the confusion matrix
    ax.set_xlabel("Predicted labels") # Set the x-axis label
    ax.set_ylabel("True labels") # Set the y-axis label
    ax.set_title("Test Confusion Matrix") # Set the title of the plot
    plt.show() # Show the plot
    plt.close(fig) # Close the figure to avoid display
    print("-" * 150)

    # Save the test results to a JSON file
    test_results = {
        "accuracy": test_accuracy,
        "f1_score": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "roc_auc": test_roc_auc,
        "confusion_matrix": test_cm.tolist() # Convert the confusion matrix to a list for JSON serialization
    }
    with open("modeling/models/test_results.json", "w") as f: # Open the file for writing
        json.dump(test_results, f, indent=4) # Write the test results to the file
    print("Test results saved to modeling/models/test_results.json") # Print a message indicating the file location

if __name__ == "__main__":
    main() # Run the main function





