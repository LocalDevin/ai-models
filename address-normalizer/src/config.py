import os
from dataclasses import dataclass
from typing import Optional

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
    embedding_dim: int = 384
    hidden_dim: int = 256
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    num_epochs: int = 50
    early_stopping_patience: int = 5
