import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.config import ModelConfig
from src.model.network import SiameseNetwork

def test_network_performance():
    """Test network performance and GPU support."""
    print("\nTesting network performance...")
    
    # Initialize network
    config = ModelConfig()
    network = SiameseNetwork(config)
    
    # Test GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    network = network.to(device)
    
    # Generate sample data
    batch_size = 32
    emb1 = torch.randn(batch_size, config.embedding_dim).to(device)
    emb2 = torch.randn(batch_size, config.embedding_dim).to(device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    output = network(emb1, emb2)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test prediction
    print("\nTesting prediction...")
    similarity, prediction = network.predict_similarity(emb1, emb2)
    print(f"Similarity shape: {similarity.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique predictions: {torch.unique(prediction).tolist()}")

if __name__ == '__main__':
    test_network_performance()
