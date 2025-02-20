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
SUPPORTED_LANGUAGES = {
    "DE": {
        "name": "German",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "test_data": ["addresses.csv", "test_cases.csv"],
        "weights": {
            "zip": 2.0,    # Highest priority
            "city": 1.5,   # Medium priority
            "street": 1.2  # Lower priority
        }
    },
    "EN": {
        "name": "English",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "test_data": ["addresses.csv", "test_cases.csv"],
        "weights": {
            "zip": 2.0,
            "city": 1.5,
            "street": 1.2
        }
    }
}
DEFAULT_LANGUAGE = "DE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
EMBEDDING_DIM = 384  # MiniLM produces 384-dim vectors
BATCH_SIZE = 32     # Batch processing for better performance
NUM_WORKERS = 4     # Increase if you have more CPU cores for DataLoader
PIN_MEMORY = True if device.type == "cuda" else False

class AddressMatcher:
    def __init__(self, language: str = DEFAULT_LANGUAGE, device: Optional[str] = None):
        """Initialize matcher with language and device settings."""
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
        
        self.language = language
        # Validate device availability
        requested_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if requested_device == 'cuda' and torch.cuda.is_available() else 'cpu'
        if requested_device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU instead.")
        print(f"Using device: {self.device}")
        
        # Initialize components with language-specific settings
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.network = SiameseNetwork(self.model_config).to(self.device)
        self.transformer = SentenceTransformer(SUPPORTED_LANGUAGES[language]['embedding_model'])
        self.transformer.to(self.device)
        
        # Set up model directory
        self.model_dir = MODELS_DIR / language
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        self.scaler = torch.amp.GradScaler()  # GradScaler automatically handles device type
        self.criterion = torch.nn.BCELoss()
        self.reference_data = []
        self.embeddings_cache = {}
        
    def initialize_database(self, csv_path: str, sample_size: Optional[int] = 10000) -> None:
        """Initialize database from CSV with memory-efficient loading.
        
        Args:
            csv_path: Path to the CSV file
            sample_size: Maximum number of addresses to load (default: 10000)
        """
        print("Loading reference data...")
        
        # Validate and adjust path for language-specific data
        data_path = Path(csv_path)
        if not data_path.is_absolute():
            lang_dir = Path("test_data") / self.language
            data_path = lang_dir / data_path.name
            if not data_path.exists():
                raise FileNotFoundError(f"No language-specific data found at {data_path}")
        
        # Use AddressLoader for efficient loading
        loader = AddressLoader(str(data_path), self.data_config, sample_size=sample_size)
        self.reference_data = []
        print(f"Loading up to {sample_size} addresses...")
        
        # Load addresses and cache embeddings
        for addr in loader:
            self.reference_data.append(addr)
            if addr['full_address'] not in self.embeddings_cache:
                with torch.autocast(device_type=self.device):
                    self.embeddings_cache[addr['full_address']] = self.transformer.encode(addr['full_address'])
        
        gc.collect()  # Free memory
        print(f"Loaded {len(self.reference_data)} addresses for language {self.language}")
    
    def train(self, data_path: str, config: TrainingConfig, sample_size: Optional[int] = 10000) -> Dict[str, float]:
        """Train the network with mixed precision and GPU optimization.
        
        Args:
            data_path: Path to training data
            config: Training configuration
            sample_size: Maximum number of addresses to load (default: 10000)
        """
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
            self.initialize_database(data_path, sample_size=sample_size)
        
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
            
        # Configure batch size based on device
        effective_batch_size = 256 if torch.cuda.is_available() else 64
        
        train_loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory and self.device == 'cuda',
            prefetch_factor=self.data_config.prefetch_factor,
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        # Initialize optimizer with larger learning rate for GPU
        lr = config.learning_rate * 2.0 if torch.cuda.is_available() else config.learning_rate
        optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=0.01)
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler(enabled=torch.cuda.is_available())
        
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
        # Use larger batch size for GPU
        batch_size = 512 if torch.cuda.is_available() else 64
        
        # First filter by postal code and city for better matches
        filtered_matches = [
            addr for addr in self.reference_data 
            if (addr['nPLZ'] == postal_code or 
                self._partial_match(addr['cOrtsname'].lower(), city.lower()))
        ]
        
        # If no matches found, fall back to all addresses
        if not filtered_matches:
            filtered_matches = self.reference_data
        
        for i in tqdm(range(0, len(filtered_matches), batch_size), desc="Finding matches"):
            batch = filtered_matches[i:i + batch_size]
            
            # Process missing embeddings in batch
            missing_addrs = [addr for addr in batch if addr['full_address'] not in self.embeddings_cache]
            if missing_addrs:
                with torch.autocast(device_type=self.device):
                    missing_texts = [addr['full_address'] for addr in missing_addrs]
                    missing_embeddings = self.transformer.encode(missing_texts)
                    for addr, emb in zip(missing_addrs, missing_embeddings):
                        self.embeddings_cache[addr['full_address']] = emb
            
            # Convert to tensor efficiently using stack
            batch_embeddings = torch.stack([
                torch.tensor(self.embeddings_cache[addr['full_address']], device=self.device)
                for addr in batch
            ])
            
            # Skip empty batches
            if not batch:
                continue
                
            # Calculate similarity scores in batch
            with torch.no_grad():
                with torch.autocast(device_type=self.device):
                    # Expand query embedding to match batch size
                    query_batch = query_embedding.unsqueeze(0).expand(len(batch), -1)
                    
                    # Get similarity scores
                    similarity, _ = self.network.predict_similarity(query_batch, batch_embeddings)
                    base_scores = similarity.squeeze(-1)  # Keep batch dimension
            
            # Convert scores to float32 and move to CPU
            base_scores = base_scores.to(torch.float32).cpu().numpy()
            if len(base_scores.shape) == 0:  # Handle single score
                base_scores = base_scores.reshape(1)
            
            # Process each address in the batch
            for score, addr in zip(base_scores, batch):
                # Calculate component-wise similarity scores
                postal_match = 1.0 if addr['nPLZ'] == postal_code else 0.0
                
                city_match = 1.0 if addr['cOrtsname'].lower() == city.lower() else (
                    0.8 if self._partial_match(addr['cOrtsname'].lower(), city.lower()) else 0.0
                )
                
                street_match = 1.0 if addr['cStrassenname'].lower() == street.lower() else (
                    0.8 if self._partial_match(addr['cStrassenname'].lower(), street.lower()) else 0.0
                )
                
                # Get language-specific weights
                weights = SUPPORTED_LANGUAGES[self.language]['weights']
                
                # Weighted combination of neural and rule-based scores
                neural_score = float(score)
                component_score = (
                    postal_match * weights['zip'] +
                    city_match * weights['city'] +
                    street_match * weights['street']
                ) / (weights['zip'] + weights['city'] + weights['street'])
                
                # Final score combines both approaches with proper normalization
                neural_score = float(score)  # Base neural score (0-1)
                component_score = min(1.0, component_score)  # Normalize component score
                
                # Weighted combination with stronger bias towards exact matches
                final_score = (0.4 * neural_score + 0.6 * component_score)
                
                # Apply stronger normalization for better score distribution
                if component_score > 0.9:  # Exact or near-exact match
                    normalized_score = 0.8 + (final_score * 0.2)  # Maps to 0.8-1.0
                else:
                    normalized_score = 0.3 + (final_score * 0.5)  # Maps to 0.3-0.8
                matches.append((addr, normalized_score))
            

        
        # Sort matches efficiently and return top k
        matches.sort(key=lambda x: (-x[1], x[0]['full_address']))  # Sort by score desc, then address asc
        return matches[:k]
    
    def _partial_match(self, str1: str, str2: str) -> bool:
        """Check if strings partially match, handling common variations."""
        # Normalize strings
        str1 = str1.lower().replace('ß', 'ss').replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        str2 = str2.lower().replace('ß', 'ss').replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        
        # Handle common abbreviations
        str1 = str1.replace('str.', 'strasse').replace('str ', 'strasse ')
        str2 = str2.replace('str.', 'strasse').replace('str ', 'strasse ')
        
        # Check for exact or partial matches
        return str1 == str2 or str1 in str2 or str2 in str1
        
    def save_model(self, model_name: str = "latest") -> None:
        """Save trained model and embeddings cache with language support."""
        model_path = self.model_dir / f"{model_name}.pt"
        cache_path = self.model_dir / f"{model_name}_cache.pkl"
        reference_path = self.model_dir / f"{model_name}_reference.pkl"
        
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state with language information
        torch.save({
            'network_state': self.network.state_dict(),
            'language': self.language,
            'model_name': model_name,
            'embedding_model': SUPPORTED_LANGUAGES[self.language]['embedding_model'],
            'config': {
                'model': self.model_config.__dict__,
                'data': self.data_config.__dict__
            }
        }, model_path)
        
        # Save cache and reference data
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
            
        with open(reference_path, 'wb') as f:
            pickle.dump(self.reference_data, f)
            
    def load_model(self, model_name: str = "latest") -> None:
        """Load trained model with language validation."""
        model_path = self.model_dir / f"{model_name}.pt"
        cache_path = self.model_dir / f"{model_name}_cache.pkl"
        reference_path = self.model_dir / f"{model_name}_reference.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found at {model_path}")
        
        # Load and validate model
        checkpoint = torch.load(model_path)
        if checkpoint['language'] != self.language:
            raise ValueError(f"Model language {checkpoint['language']} doesn't match current language {self.language}")
        if checkpoint['embedding_model'] != SUPPORTED_LANGUAGES[self.language]['embedding_model']:
            raise ValueError("Embedding model mismatch")
            
        # Update configurations if available
        if 'config' in checkpoint:
            self.model_config = ModelConfig(**checkpoint['config']['model'])
            self.data_config = DataConfig(**checkpoint['config']['data'])
        
        # Load model state and data
        self.network = SiameseNetwork(self.model_config).to(self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
                
        if reference_path.exists():
            with open(reference_path, 'rb') as f:
                self.reference_data = pickle.load(f)
        else:
            raise FileNotFoundError(f"No reference data found at {reference_path}")
