import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.matcher import AddressMatcher
from src.config import TrainingConfig
import torch

def test_matcher_performance():
    """Test matcher initialization and training."""
    print("\nTesting matcher performance...")
    
    # Create test data
    test_data = [
        {'nPLZ': '12345', 'cOrtsname': 'Berlin', 'cStrassenname': 'Hauptstrasse', 'full_address': '12345 Berlin Hauptstrasse'},
        {'nPLZ': '12345', 'cOrtsname': 'Berlin', 'cStrassenname': 'Hauptstr.', 'full_address': '12345 Berlin Hauptstr.'},
        {'nPLZ': '12345', 'cOrtsname': 'Berlin', 'cStrassenname': 'Nebenstrasse', 'full_address': '12345 Berlin Nebenstrasse'},
        {'nPLZ': '54321', 'cOrtsname': 'Hamburg', 'cStrassenname': 'Hauptstrasse', 'full_address': '54321 Hamburg Hauptstrasse'}
    ]
    
    # Initialize matcher
    matcher = AddressMatcher()
    matcher.reference_data = test_data
    
    # Test training with small dataset
    config = TrainingConfig(num_epochs=2)  # Small number of epochs for testing
    metrics = matcher.train(None, config)  # Pass None as we already set reference_data
    
    print("\nTraining metrics:")
    print(f"Final loss: {metrics['train_loss'][-1]:.4f}")
    
    # Test match finding
    matches = matcher.find_matches('12345', 'Berlin', 'Hauptstrasse', k=3)
    
    print("\nTest matches:")
    for addr, score in matches:
        print(f"Match: {addr['full_address']}, Score: {score:.4f}")

if __name__ == '__main__':
    test_matcher_performance()
