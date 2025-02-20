import torch
from pathlib import Path
import argparse
from src.model.matcher import AddressMatcher
from src.config import TrainingConfig, DataConfig

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate address matcher')
    parser.add_argument('--reference', required=True, type=Path, help='Path to reference addresses CSV')
    parser.add_argument('--test', type=Path, default=None, help='Path to test addresses CSV')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Initialize matcher with optimal device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = AddressMatcher(device)
    
    # Configure training parameters
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=0.001,
        early_stopping_patience=5
    )
    
    data_config = DataConfig(
        batch_size=args.batch_size,
        num_workers=args.workers,
        chunk_size=10000,
        pin_memory=device == 'cuda'
    )
    
    # Train the model
    print(f"\nTraining on device: {device}")
    metrics = matcher.train(args.reference, train_config)
    
    # Test if test file provided
    if args.test:
        print("\nTesting model...")
        test_data = [
            ("12345 Berlin Hauptstraße", "12345 Berlin Hauptstrasse"),
            ("60313 Frankfurt am Main", "60313 Frankfurt a.M."),
            ("80331 München Marienplatz", "80331 Muenchen Marienplatz")
        ]
        
        for addr1, addr2 in test_data:
            postal_code = addr1.split()[0]
            city = addr1.split()[1]
            street = ' '.join(addr1.split()[2:])
            matches = matcher.find_matches(postal_code, city, street, k=3)
            print(f"\nQuery: {addr1}")
            for addr, score in matches:
                print(f"Match: {addr['full_address']}, Score: {score:.4f}")

if __name__ == '__main__':
    main()
