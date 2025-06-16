"""
Default configuration parameters for the project
"""

from pathlib import Path
import torch

# Default directory for caching datasets
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets"

# Audio processing defaults
DEFAULT_TRUNCATION = 12  # seconds (maximum length of audio samples)

# Training defaults
DEFAULT_EPOCHS = 20  # Number of training epochs
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Model defaults
MODEL_NAME = "facebook/hubert-large-ls960-ft"  # Pretrained Hubert model