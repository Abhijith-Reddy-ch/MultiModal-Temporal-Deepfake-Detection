import os
import torch
from pathlib import Path

# Project Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

# Create necessary directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
MODEL_NAME = "resnet18" # options: "resnet18", "resnet50"
NUM_WORKERS = min(4, os.cpu_count() or 1)

# Paths for Splits
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Model specific
IMAGE_SIZE = 224
DROPOUT_RATE = 0.4
FREEZE_EARLY_LAYERS = True

# Advanced Features
USE_FACE_DETECTION = True # Use MTCNN to crop faces
CONFIDENCE_THRESHOLD = 0.5 # Threshold for fake prediction
MIXED_PRECISION = True # Use AMP for training

# UI and Backend Config
API_PORT = 8000
API_HOST = "127.0.0.1"
