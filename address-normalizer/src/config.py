import os
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional

# Constants for paths and language support
MODELS_DIR = Path("models")
TEST_DATA_DIR = Path("test_data")

# Language-specific configurations
SUPPORTED_LANGUAGES = {
    "DE": {
        "name": "German",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "test_data": ["addresses.csv", "test_cases.csv"]
    },
    "EN": {
        "name": "English",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "test_data": ["addresses.csv", "test_cases.csv"]
    }
}
DEFAULT_LANGUAGE = "DE"

@dataclass
class LanguageConfig:
    code: str = DEFAULT_LANGUAGE
    test_data_path: Path = TEST_DATA_DIR / DEFAULT_LANGUAGE
    model_path: Path = MODELS_DIR / DEFAULT_LANGUAGE
    embedding_model: str = SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["embedding_model"]

@dataclass
class DataConfig:
    chunk_size: int = 10000
    max_pairs_per_group: int = 10
    num_workers: int = min(8, os.cpu_count() or 4)  # Optimize for available CPUs
    batch_size: int = 128  # Increased for better GPU utilization
    pin_memory: bool = True
    prefetch_factor: int = 2  # Prefetch batches for better throughput

@dataclass
class ModelConfig:
    embedding_dim: int = 384  # Matches sentence transformer output
    hidden_dim: int = 512     # Increased for better feature extraction
    dropout: float = 0.2      # Adjusted for better generalization
    learning_rate: float = 0.001
    weight_decay: float = 0.01

@dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 5
    batch_size: int = 64      # Per-language batch size
    num_workers: int = 4      # Per-language worker count
