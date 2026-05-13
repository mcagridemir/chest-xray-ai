"""
Training script.

Usage:
    python -m src.train

Two-phase training:
  Phase 1 — backbone frozen, head trained for WARMUP_EPOCHS.
  Phase 2 — top blocks unfrozen, full fine-tune until early stopping.

Checkpoints and training curves are saved to outputs/.
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.dataset import get_loaders
from src.model import XRayClassifier

WARMUP_EPOCHS = 5   # Phase-1 length before unfreezing


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def plot_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#378ADD")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#D85A30")
    axes[0].set_title("Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#378ADD")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#D85A30")
    axes[1].set_title("Accuracy", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {save_path}")


def train():
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, class_weights = get_loaders()
    model = XRayClassifier(pretrained=True).to(device)
    model.count_parameters()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    # ── Phase 1: warm-up (head only) ───────────────────────────────────────────
    model.freeze_backbone()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE * 5,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=WARMUP_EPOCHS)

    print("\n─── Phase 1: head warm-up ───")
    for epoch in range(1, WARMUP_EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"  Epoch {epoch:02d}/{WARMUP_EPOCHS}  "
              f"loss {tl:.4f}/{vl:.4f}  acc {ta:.4f}/{va:.4f}  "
              f"({time.time()-t0:.1f}s)")

    # ── Phase 2: fine-tune top blocks ──────────────────────────────────────────
    model.unfreeze_top_blocks(num_blocks=2)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    remaining = config.NUM_EPOCHS - WARMUP_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

    print("\n─── Phase 2: fine-tune ───")
    for epoch in range(1, remaining + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"  Epoch {WARMUP_EPOCHS+epoch:02d}/{config.NUM_EPOCHS}  "
              f"loss {tl:.4f}/{vl:.4f}  acc {ta:.4f}/{va:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if va > best_val_acc:
            best_val_acc = va
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": WARMUP_EPOCHS + epoch,
            }, config.MODEL_PATH)
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {WARMUP_EPOCHS + epoch}.")
                break

    plot_curves(history, os.path.join(config.OUTPUT_DIR, "training_curves.png"))
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
