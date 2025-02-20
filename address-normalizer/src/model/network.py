import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from typing import Tuple
from ..config import ModelConfig

class SiameseNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Input features: embedding_dim (L1 distance) + 1 (cosine similarity)
        input_dim = config.embedding_dim + 1
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),  # More stable than BatchNorm for varying batch sizes
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Similarity scoring layers
        self.scoring_layers = nn.Sequential(
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights for better training
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights for better convergence."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision and optimized similarity computation."""
        device = next(self.parameters()).device
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        
        # Move inputs to correct device if needed
        if emb1.device != device:
            emb1 = emb1.to(device, non_blocking=True)
        if emb2.device != device:
            emb2 = emb2.to(device, non_blocking=True)
            
        with torch.amp.autocast(device_type=device_type):
            # Compute multiple similarity metrics
            cos_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8).unsqueeze(1)
            l1_dist = torch.abs(emb1 - emb2)
            
            # Combine features
            combined = torch.cat([cos_sim, l1_dist], dim=1)
            
            # Extract features and compute similarity score
            features = self.feature_layers(combined)
            similarity = self.scoring_layers(features)
            
            return similarity
    
    def predict_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict similarity efficiently for inference."""
        with torch.no_grad(), torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            # Move inputs to device efficiently
            if emb1.device != self.device:
                emb1 = emb1.to(self.device, non_blocking=True)
            if emb2.device != self.device:
                emb2 = emb2.to(self.device, non_blocking=True)
            
            # Compute similarity score
            similarity = self.forward(emb1, emb2)
            
            # Get binary predictions using threshold
            prediction = (similarity > 0.5).float()
            
            return similarity, prediction
            
    @property
    def device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device
