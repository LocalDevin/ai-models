import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Optional
import mmap
import psutil
from tqdm import tqdm
from ..config import DataConfig

class AddressLoader:
    def __init__(self, file_path: str, config: DataConfig, sample_size: Optional[int] = None):
        self.file_path = Path(file_path).resolve()
        self.config = config
        self.sample_size = sample_size
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self._map_file()
    
    def _map_file(self):
        """Memory-map the CSV file for efficient reading."""
        try:
            self.file = open(self.file_path, 'rb')
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            if hasattr(self, 'file'):
                self.file.close()
            raise RuntimeError(f"Failed to memory-map file: {e}")
    
    def __iter__(self) -> Generator[Dict[str, str], None, None]:
        """Yield address records efficiently using memory mapping."""
        # Reset memory map position and get total size
        self.mm.seek(0)
        total_size = self.file_path.stat().st_size
        processed = 0
        records_yielded = 0
        
        df_iter = pd.read_csv(
            self.mm, 
            delimiter=';',
            chunksize=min(self.config.chunk_size, self.sample_size if self.sample_size else self.config.chunk_size),
            dtype={'nPLZ': str},
            nrows=self.sample_size
        )
        
        with tqdm(total=self.sample_size if self.sample_size else None, desc="Loading addresses") as pbar:
            for chunk in df_iter:
                for _ in chunk.iterrows():
                    if self.sample_size and records_yielded >= self.sample_size:
                        pbar.close()
                        return
                    records_yielded += 1
                    pbar.update(1)
                
                if processed % (100 * 1024 * 1024) == 0:  # Log every 100MB
                    mem = psutil.Process().memory_info()
                    print(f"\nMemory usage: {mem.rss / 1024 / 1024:.1f}MB")
                
                for _, row in chunk.iterrows():
                    if self.sample_size and records_yielded >= self.sample_size:
                        return
                    yield {
                        'nPLZ': row['nPLZ'],
                        'cOrtsname': row['cOrtsname'],
                        'cStrassenname': row['cStrassenname'],
                        'full_address': f"{row['nPLZ']} {row['cOrtsname']} {row['cStrassenname']}"
                    }
                    records_yielded += 1
    
    def load_full(self) -> pd.DataFrame:
        """Load entire dataset efficiently using memory mapping."""
        try:
            # Reset memory map position
            self.mm.seek(0)
            
            # Use a more memory-efficient approach for full loading
            df = pd.read_csv(
                self.mm,
                delimiter=';',
                dtype={'nPLZ': str},
                low_memory=True   # More memory-efficient at cost of some speed
            )
            
            # Log memory usage
            mem = psutil.Process().memory_info()
            print(f"\nFull load memory usage: {mem.rss / 1024 / 1024:.1f}MB")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load full dataset: {e}")
    
    def __del__(self):
        """Clean up memory-mapped resources."""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close()
