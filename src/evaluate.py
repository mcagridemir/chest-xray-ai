"""
Evaluation script — run on the held-out test set after training.

Produces:
  outputs/confusion_matrix.png
  outputs/roc_curve.png
  outputs/classification_report.txt

Usage:
    python -m src.evaluate
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
)

import config
from src.dataset import get_loaders
from src.model import load_model


@torch.no_grad()
def get_predictions(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs[:, 1])   # probability of PNEUMONIA
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_roc_curve(y_true, y_probs, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#378ADD", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#B4B2A9", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — Test Set", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved → {save_path}")
    return roc_auc


def evaluate():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, _ = get_loaders()
    model = load_model(config.MODEL_PATH, device)

    print("\nRunning evaluation on test set …")
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, digits=4
    )
    print("\n" + report)
    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Macro F1
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro F1: {f1:.4f}")

    # Plots
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"),
    )
    roc_auc = plot_roc_curve(
        y_true, y_probs,
        os.path.join(config.OUTPUT_DIR, "roc_curve.png"),
    )
    print(f"AUC: {roc_auc:.4f}")
    print("\nEvaluation complete. All outputs saved to outputs/")


if __name__ == "__main__":
    evaluate()
