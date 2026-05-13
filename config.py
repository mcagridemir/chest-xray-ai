"""
Central configuration — change hyperparameters here, not inside the training code.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = os.getenv("DATA_DIR", "data/data/chest_xray")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
TEST_DIR   = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE   = 224          # EfficientNet-B0 native resolution
MEAN       = [0.485, 0.456, 0.406]   # ImageNet statistics
STD        = [0.229, 0.224, 0.225]

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 4         # early-stopping patience
NUM_WORKERS   = 4

# ── Model ──────────────────────────────────────────────────────────────────────
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
