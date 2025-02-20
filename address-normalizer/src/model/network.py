import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from typing import Tuple
from ..config import ModelConfig

class SiameseNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2),
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
        """Forward pass with mixed precision support."""
        device = next(self.parameters()).device
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        
        # Ensure inputs are on the correct device
        if emb1.device != device:
            emb1 = emb1.to(device)
        if emb2.device != device:
            emb2 = emb2.to(device)
            
        with torch.amp.autocast(device_type=device_type):
            # Compute absolute difference for similarity
            diff = torch.abs(emb1 - emb2)
            return self.fc(diff)
    
    def predict_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict similarity with gradient disabled for inference."""
        with torch.no_grad():
            device = next(self.parameters()).device
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            
            with torch.amp.autocast(device_type=device_type):
                similarity = self.forward(emb1, emb2)
                # Return both raw similarity and thresholded prediction
                prediction = (similarity > 0.5).float()
                return similarity, prediction
