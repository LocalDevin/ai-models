import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import DataConfig
from src.data.loader import AddressLoader

def test_loader_performance():
    # Test with first 10000 records
    config = DataConfig(chunk_size=1000)
    data_path = Path(__file__).parent.parent / 'test_data' / 'addresses.csv'
    print(f"Loading data from: {data_path}")
    print(f"File size: {data_path.stat().st_size / 1024 / 1024:.1f}MB")
    print("Testing with first 10000 records...")
    loader = AddressLoader(str(data_path), config)

    # Test chunked loading
    start = time.time()
    count = 0
    for i, _ in enumerate(loader):
        if i >= 10000:
            break
        count += 1
    chunk_time = time.time() - start
    print(f'Processed {count} records in {chunk_time:.2f}s using chunks')

    # Test full loading
    start = time.time()
    df = loader.load_full()
    full_time = time.time() - start
    print(f'Loaded {len(df)} records in {full_time:.2f}s using full load')

if __name__ == '__main__':
    test_loader_performance()
