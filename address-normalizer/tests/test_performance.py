import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import psutil
import torch
from src.model.matcher import AddressMatcher
from src.config import TrainingConfig, DataConfig

class PerformanceMetrics:
    @staticmethod
    def measure_training(matcher, data_path: str, config: TrainingConfig) -> dict:
        """Measure training performance metrics."""
        start = time.time()
        gpu_mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        cpu_mem_start = psutil.Process().memory_info().rss
        
        # Train model and get metrics
        metrics = matcher.train(data_path, config)
        
        return {
            'training_time': time.time() - start,
            'gpu_memory_used': (torch.cuda.memory_allocated() - gpu_mem_start) / 1024**2,
            'cpu_memory_used': (psutil.Process().memory_info().rss - cpu_mem_start) / 1024**2,
            'final_loss': metrics['train_loss'][-1]
        }
    
    @staticmethod
    def measure_inference(matcher, queries: list, k: int = 5) -> dict:
        """Measure inference performance metrics."""
        start = time.time()
        gpu_mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        cpu_mem_start = psutil.Process().memory_info().rss
        
        # Run inference
        results = []
        for postal_code, city, street in queries:
            matches = matcher.find_matches(postal_code, city, street, k=k)
            results.append(matches)
        
        return {
            'inference_time': time.time() - start,
            'gpu_memory_used': (torch.cuda.memory_allocated() - gpu_mem_start) / 1024**2,
            'cpu_memory_used': (psutil.Process().memory_info().rss - cpu_mem_start) / 1024**2,
            'matches_per_query': sum(len(m) for m in results) / len(queries)
        }

def test_performance():
    """Test performance with both small and large datasets."""
    print("\nTesting performance...")
    
    # Test data
    test_queries = [
        ('12345', 'Berlin', 'Hauptstrasse'),
        ('60313', 'Frankfurt', 'Zeil'),
        ('80331', 'München', 'Marienplatz')
    ]
    
    # Test with small dataset
    print("\nTesting with small dataset...")
    matcher_small = AddressMatcher()
    train_config = TrainingConfig(num_epochs=2)
    metrics_small = PerformanceMetrics.measure_training(
        matcher_small, 'test_data/sample.csv', train_config
    )
    
    inference_small = PerformanceMetrics.measure_inference(
        matcher_small, test_queries
    )
    
    print("\nSmall Dataset Metrics:")
    print(f"Training time: {metrics_small['training_time']:.2f}s")
    print(f"Training GPU memory: {metrics_small['gpu_memory_used']:.1f}MB")
    print(f"Training CPU memory: {metrics_small['cpu_memory_used']:.1f}MB")
    print(f"Final loss: {metrics_small['final_loss']:.4f}")
    print(f"Inference time: {inference_small['inference_time']:.2f}s")
    print(f"Inference GPU memory: {inference_small['gpu_memory_used']:.1f}MB")
    print(f"Inference CPU memory: {inference_small['cpu_memory_used']:.1f}MB")
    print(f"Average matches per query: {inference_small['matches_per_query']:.1f}")
    
    # Test with medium dataset
    print("\nTesting with medium dataset...")
    matcher_medium = AddressMatcher()
    metrics_medium = PerformanceMetrics.measure_training(
        matcher_medium, 'test_data/medium.csv', train_config
    )
    
    inference_medium = PerformanceMetrics.measure_inference(
        matcher_medium, test_queries
    )
    
    print("\nMedium Dataset Metrics:")
    print(f"Training time: {metrics_medium['training_time']:.2f}s")
    print(f"Training GPU memory: {metrics_medium['gpu_memory_used']:.1f}MB")
    print(f"Training CPU memory: {metrics_medium['cpu_memory_used']:.1f}MB")
    print(f"Final loss: {metrics_medium['final_loss']:.4f}")
    print(f"Inference time: {inference_medium['inference_time']:.2f}s")
    print(f"Inference GPU memory: {inference_medium['gpu_memory_used']:.1f}MB")
    print(f"Inference CPU memory: {inference_medium['cpu_memory_used']:.1f}MB")
    print(f"Average matches per query: {inference_medium['matches_per_query']:.1f}")

if __name__ == '__main__':
    test_performance()
