import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import DataConfig
from src.data.loader import AddressLoader
from src.data.dataset import AddressDataset
from sentence_transformers import SentenceTransformer

def test_dataset_performance():
    """Test dataset creation and pair generation performance."""
    print("\nTesting dataset performance...")
    
    # Load a sample of addresses
    config = DataConfig(chunk_size=1000)
    loader = AddressLoader('test_data/addresses.csv', config)
    addresses = []
    for i, addr in enumerate(loader):
        if i >= 1000:  # Test with 1000 addresses
            break
        addresses.append(addr)
    
    print(f"\nLoaded {len(addresses)} addresses")
    
    # Initialize model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Create dataset and measure performance
    dataset = AddressDataset(addresses, model, num_workers=4)
    
    # Test pair access
    print("\nTesting pair access...")
    for i in range(min(5, len(dataset))):
        emb1, emb2, label = dataset[i]
        print(f"Pair {i}: shape={emb1.shape}, label={label.item():.1f}")

if __name__ == '__main__':
    test_dataset_performance()
