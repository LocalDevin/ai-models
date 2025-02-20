import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc
from pathlib import Path

from ..config import ModelConfig, DataConfig, TrainingConfig
from ..data.dataset import AddressDataset
from ..data.loader import AddressLoader
from .network import SiameseNetwork

# Constants for model persistence and language support
MODELS_DIR = Path("models")
DEFAULT_LANGUAGE = "DE"
SUPPORTED_LANGUAGES = ["DE"]  # Expandable for future languages
EMBEDDING_MODELS = {
    "DE": "sentence-transformers/all-MiniLM-L6-v2",  # Current model
    # Future language-specific models can be added here
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Weight multipliers for hierarchical matching
ZIP_WEIGHT = 2.0    # Highest priority
CITY_WEIGHT = 1.5   # Medium priority
STREET_WEIGHT = 1.2 # Lower priority

# Constants
EMBEDDING_DIM = 384  # MiniLM produces 384-dim vectors
BATCH_SIZE = 32     # Batch processing for better performance
NUM_WORKERS = 4     # Increase if you have more CPU cores for DataLoader
PIN_MEMORY = True if device.type == "cuda" else False

class AddressMatcher:
    def __init__(self, device: Optional[str] = None):
        """Initialize matcher with device-aware components."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.network = SiameseNetwork(self.model_config).to(self.device)
        self.transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.transformer.to(self.device)
        
        # Initialize training components
        self.scaler = torch.amp.GradScaler()  # GradScaler automatically handles device type
        self.criterion = torch.nn.BCELoss()
        self.reference_data = []
        self.embeddings_cache = {}
        
    def initialize_database(self, csv_path: str, sample_size: Optional[int] = None) -> None:
        """Initialize database from CSV with memory-efficient loading."""
        print("Loading reference data...")
        
        # Use AddressLoader for efficient loading
        loader = AddressLoader(csv_path, self.data_config)
        self.reference_data = []
        
        # Load addresses and cache embeddings
        for addr in tqdm(loader, desc="Processing addresses"):
            if sample_size and len(self.reference_data) >= sample_size:
                break
                
            self.reference_data.append(addr)
            if addr['full_address'] not in self.embeddings_cache:
                with torch.autocast(device_type=self.device):
                    self.embeddings_cache[addr['full_address']] = self.transformer.encode(addr['full_address'])
        
        gc.collect()  # Free memory
        print(f"Loaded {len(self.reference_data)} addresses")
    
    def train(self, data_path: str, config: TrainingConfig) -> Dict[str, float]:
        """Train the network with mixed precision and GPU optimization."""
        print("\nStarting training...")
        train_data = [
            # Basic street variations
            ("12345 Berlin Hauptstraße", "12345 Berlin Hauptstrasse", 1),
            ("12345 Berlin Hauptstraße", "12345 Berlin Hauptstr.", 1),
            ("12345 Berlin Hauptstr", "12345 Berlin Hauptstrasse", 1),
            
            # City name variations
            ("60313 Frankfurt am Main Zeil", "60313 Frankfurt a.M. Zeil", 1),
            ("60313 Frankfurt/Main Zeil", "60313 Frankfurt Zeil", 1),
            ("60313 Frankfurt a.M. Zeil", "60313 Frankfurt/M Zeil", 1),
            ("60313 Frankfurt/Main Zeil", "60313 Ffm Zeil", 1),
            
            # Special German characters
            ("70173 Stuttgart Königstraße", "70173 Stuttgart Koenigstrasse", 1),
            ("80331 München Marienplatz", "80331 Muenchen Marienplatz", 1),
            ("79539 Lörrach Hauptstraße", "79539 Loerrach Hauptstr.", 1),
            ("12345 Köln Höhenberger Str.", "12345 Koeln Hoehenberger Str.", 1),
            
            # Common abbreviations and variations
            ("10178 Berlin Alexanderplatz", "10178 Berlin Alex.-Pl.", 1),
            ("10178 Berlin Alexanderpl.", "10178 Berlin Alexander-Platz", 1),
            ("10117 Berlin Unter den Linden", "10117 Berlin U.d. Linden", 1),
            ("12345 Hamburg Sankt Georg", "12345 Hamburg St. Georg", 1),
            ("12345 Berlin Karl-Marx-Str.", "12345 Berlin Karl Marx Strasse", 1),
            ("12345 Berlin An der Spree", "12345 Berlin a.d. Spree", 1),
            ("12345 Berlin Vor dem Tor", "12345 Berlin v.d. Tor", 1),
            
            # Directional variations
            ("12345 Berlin Nord", "12345 Berlin-N", 1),
            ("12345 Berlin Süd", "12345 Berlin-S", 1),
            ("12345 Berlin West", "12345 Berlin-W", 1),
            ("12345 Berlin Ost", "12345 Berlin-O", 1),
            
            # Negative examples (different addresses)
            ("10115 Berlin Invalidenstraße", "10117 Berlin Friedrichstraße", 0),
            ("60313 Frankfurt Zeil", "80331 München Kaufingerstraße", 0),
            ("70173 Stuttgart Königstraße", "70174 Stuttgart Calwerstraße", 0),
            ("12345 Berlin-Nord", "12345 Berlin-Süd", 0),
            ("12345 Hamburg St. Pauli", "12345 Hamburg St. Georg", 0)
        ]
        
        # Load data if path provided
        if data_path:
            loader = AddressLoader(data_path, self.data_config)
            self.reference_data = list(loader)
        
        if not self.reference_data:
            raise ValueError("No reference data available for training")
        
        # Create dataset with training pairs
        dataset = AddressDataset(
            addresses=self.reference_data,
            model=self.transformer,
            num_workers=self.data_config.num_workers
        )
        
        if len(dataset) == 0:
            raise ValueError("Dataset generated 0 training pairs")
            
        train_loader = DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory and self.device == 'cuda'
        )
        
        # Initialize optimizer
        optimizer = optim.AdamW(self.network.parameters(), lr=config.learning_rate)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        metrics = {'train_loss': [], 'val_accuracy': []}
        
        self.network.train()
        for epoch in range(config.num_epochs):
            epoch_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
                for batch_idx, (emb1, emb2, labels) in enumerate(pbar):
                    # Move data to device
                    emb1, emb2 = emb1.to(self.device), emb2.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass with mixed precision
                    optimizer.zero_grad()
                    with torch.autocast(device_type=self.device):
                        outputs = self.network(emb1, emb2)
                        loss = self.criterion(outputs, labels.view(-1, 1))
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    avg_loss = epoch_loss / (batch_idx + 1)
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    
                    # Log memory usage for GPU training
                    if batch_idx % 100 == 0 and self.device == 'cuda':
                        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / len(train_loader)
            metrics['train_loss'].append(avg_epoch_loss)
            
            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            print(f"\nEpoch {epoch+1} - Loss: {avg_epoch_loss:.4f}")
        
        return metrics
    
    def find_matches(self, postal_code: str, city: str, street: str, k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        """Find matches using hierarchical weighting system with GPU optimization."""
        self.network.eval()
        query = f"{postal_code} {city} {street}"
        
        # Use cached embedding or compute new one
        if query not in self.embeddings_cache:
            with torch.autocast(device_type=self.device):
                self.embeddings_cache[query] = self.transformer.encode(query)
        query_embedding = torch.tensor(self.embeddings_cache[query], dtype=torch.float32).to(self.device)
        
        matches = []
        # Process in batches for memory efficiency
        batch_size = 1000
        for i in tqdm(range(0, len(self.reference_data), batch_size), desc="Finding matches"):
            batch = self.reference_data[i:i + batch_size]
            
            # Get cached embeddings for batch
            # Process batch embeddings efficiently
            missing_addrs = [addr for addr in batch if addr['full_address'] not in self.embeddings_cache]
            if missing_addrs:
                with torch.autocast(device_type=self.device):
                    missing_texts = [addr['full_address'] for addr in missing_addrs]
                    missing_embeddings = self.transformer.encode(missing_texts)
                    for addr, emb in zip(missing_addrs, missing_embeddings):
                        self.embeddings_cache[addr['full_address']] = emb
            
            # Convert to tensor efficiently using numpy
            batch_embeddings = np.array([self.embeddings_cache[addr['full_address']] for addr in batch])
            batch_embeddings = torch.from_numpy(batch_embeddings).float().to(self.device)
            
            # Calculate similarity scores in batch
            with torch.no_grad():
                with torch.autocast(device_type=self.device):
                    base_scores = self.network(
                        query_embedding.unsqueeze(0).expand(len(batch), -1),
                        batch_embeddings
                    ).squeeze()
            
            # Convert scores to float32 before processing
            base_scores = base_scores.to(torch.float32).cpu().numpy()
            
            # Process each address in the batch
            for score, addr in zip(base_scores, batch):
                # Apply weights based on exact and partial matches
                multiplier = 1.0
                
                # Apply weights multiplicatively
                if addr['nPLZ'] == postal_code:
                    multiplier *= ZIP_WEIGHT
                
                if addr['cOrtsname'].lower() == city.lower():
                    multiplier *= CITY_WEIGHT
                elif self._partial_match(addr['cOrtsname'].lower(), city.lower()):
                    multiplier *= CITY_WEIGHT * 0.8  # Partial match
                
                if addr['cStrassenname'].lower() == street.lower():
                    multiplier *= STREET_WEIGHT
                elif self._partial_match(addr['cStrassenname'].lower(), street.lower()):
                    multiplier *= STREET_WEIGHT * 0.8  # Partial match
                
                # Apply multiplier and normalize
                final_score = min(1.0, float(score * multiplier))
                matches.append((addr, final_score))
            

        
        # Sort matches efficiently and return top k
        matches.sort(key=lambda x: (-x[1], x[0]['full_address']))  # Sort by score desc, then address asc
        return matches[:k]
    
    def _partial_match(self, str1: str, str2: str) -> bool:
        """Check if strings partially match."""
        return str1 in str2 or str2 in str1
        
    def save_model(self, model_name: str = "latest") -> None:
        """Save trained model and embeddings cache."""
        model_path = self.model_dir / f"{model_name}.pt"
        cache_path = self.model_dir / f"{model_name}_cache.pkl"
        reference_path = self.model_dir / f"{model_name}_reference.pkl"
        
        torch.save({
            'network_state': self.network.state_dict(),
            'language': self.language,
            'model_name': model_name
        }, model_path)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
            
        with open(reference_path, 'wb') as f:
            pickle.dump(self.reference_data, f)
            
    def load_model(self, model_name: str = "latest") -> None:
        """Load trained model and embeddings cache."""
        model_path = self.model_dir / f"{model_name}.pt"
        cache_path = self.model_dir / f"{model_name}_cache.pkl"
        reference_path = self.model_dir / f"{model_name}_reference.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found at {model_path}")
        
        checkpoint = torch.load(model_path)
        if checkpoint['language'] != self.language:
            raise ValueError(f"Model language {checkpoint['language']} doesn't match current language {self.language}")
        
        self.network.load_state_dict(checkpoint['network_state'])
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
                
        if reference_path.exists():
            with open(reference_path, 'rb') as f:
                self.reference_data = pickle.load(f)
        else:
            raise FileNotFoundError(f"No reference data found at {reference_path}")
