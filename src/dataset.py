"""
Dataset utilities — loading, augmentation, and class-weight computation.

Kaggle dataset: paultimothymooney/chest-xray-pneumonia
  chest_xray/
  ├── train/  NORMAL/  PNEUMONIA/
  ├── val/    NORMAL/  PNEUMONIA/
  └── test/   NORMAL/  PNEUMONIA/
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import config


def get_transforms(split: str) -> transforms.Compose:
    """
    Training uses aggressive augmentation to handle class imbalance and
    improve generalisation on out-of-distribution X-rays.
    Validation / test use only resize + normalize (no random ops).
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])


def get_loaders():
    """Return train, val, test DataLoaders and class weights for loss scaling."""
    train_ds = datasets.ImageFolder(config.TRAIN_DIR, transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(config.VAL_DIR,   transform=get_transforms("val"))
    test_ds  = datasets.ImageFolder(config.TEST_DIR,  transform=get_transforms("test"))

    # Class weights to handle imbalanced train set (more pneumonia than normal)
    targets   = np.array(train_ds.targets)
    counts    = np.bincount(targets)
    weights   = 1.0 / counts
    weights  /= weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Classes: {train_ds.classes} | Weights: {class_weights.numpy().round(3)}")
    return train_loader, val_loader, test_loader, class_weights
