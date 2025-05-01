# evaluate_summary.py
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from multimodal_dataloader import get_dataloader
from multimodal_fusion_model import MultimodalFusionModel

# Load best model and evaluate

def evaluate_on_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    test_loader = get_dataloader("data/test_segment_manifest_merged.jsonl", batch_size=32, shuffle=False, text_feature="kobert", use_both_text=False)

    # Load the best model
    model = MultimodalFusionModel(audio_dim=869, text_dim=768, use_both_text=False)
    model.load_state_dict(torch.load("modeling/models/best_multimodal_model.pth", map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            audio_features = batch["audio_features"].to(device)
            text_features = batch["text_features"].to(device)
            labels = batch["label"].to(device)

            logits = model(audio_features, text_features)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_binary = (all_preds >= 0.5).astype(int)

    # Prepare output directory
    eval_dir = Path("modeling/logs/eval_results")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save a classification report
    report = classification_report(all_labels, preds_binary, digits=4)
    print("\nClassification Report:")
    print(report)
    with open(eval_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(report)

    # ROC-AUC score
    roc_auc = roc_auc_score(all_labels, all_preds)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(eval_dir / f"roc_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

    # Plot and save confusion matrix
    cm = confusion_matrix(all_labels, preds_binary)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Test Confusion Matrix')
    plt.savefig(eval_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

if __name__ == "__main__":
    evaluate_on_test()
