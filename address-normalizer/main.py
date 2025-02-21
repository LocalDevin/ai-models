import torch
from pathlib import Path
import argparse
from src.model.matcher import AddressMatcher
from src.config import TrainingConfig, DataConfig

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate address matcher')
    parser.add_argument('--reference', type=Path, help='Path to reference addresses CSV for training')
    parser.add_argument('--test', type=Path, default=None, help='Path to test addresses CSV')
    parser.add_argument('--save-model', type=str, help='Save model with given name')
    parser.add_argument('--load-model', type=str, help='Load model with given name')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing model if it exists')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--sample-size', type=int, default=10000, help='Maximum number of addresses to load')
    args = parser.parse_args()

    # Initialize matcher with optimal device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = AddressMatcher(language="DE", device=device)
    
    # Configure training parameters
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=0.001,
        early_stopping_patience=5,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    data_config = DataConfig(
        batch_size=args.batch_size,
        num_workers=args.workers,
        chunk_size=10000,
        pin_memory=device == 'cuda'
    )
    
    # Load model if specified
    if args.load_model:
        print(f"\nLoading model: {args.load_model}")
        matcher.load_model(args.load_model)
    
    # Train if reference data provided
    if args.reference:
        print(f"\nTraining on device: {device}")
        metrics = matcher.train(args.reference, train_config, sample_size=args.sample_size)
        
        # Save model if specified
        if args.save_model:
            print(f"\nSaving model: {args.save_model}")
            matcher.save_model(args.save_model, overwrite=args.overwrite)
    
    # Test if test file provided
    if args.test:
        print("\nTesting model...")
        import pandas as pd
        test_df = pd.read_csv(args.test, delimiter=';', dtype={'nPLZ': str})
        
        for _, row in test_df.iterrows():
            postal_code = row['nPLZ']
            city = row['cOrtsname']
            street = row['cStrassenname']
            addr1 = f"{postal_code} {city} {street}"
            
            matches = matcher.find_matches(postal_code, city, street, k=3)
            print(f"\nQuery: {addr1}")
            for addr, score in matches:
                print(f"Match: {addr['full_address']}, Score: {score:.4f}")

if __name__ == '__main__':
    main()
