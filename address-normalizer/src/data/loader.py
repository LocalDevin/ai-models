import io
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
            # Convert mmap to BytesIO for pandas compatibility
            self.buffer = io.BytesIO(self.mm)
        except Exception as e:
            if hasattr(self, 'file'):
                self.file.close()
            raise RuntimeError(f"Failed to memory-map file: {e}")
    
    def __iter__(self) -> Generator[Dict[str, str], None, None]:
        """Yield address records efficiently using memory mapping."""
        # Reset buffer position
        self.buffer.seek(0)
        records_yielded = 0
        
        # Use larger chunk size for better performance
        chunk_size = 5000 if self.sample_size is None else min(5000, self.sample_size)
        
        # Read data in chunks using BytesIO buffer
        df_iter = pd.read_csv(
            self.buffer, 
            delimiter=';',
            chunksize=chunk_size,
            dtype={'nPLZ': str},
            nrows=self.sample_size
        )
        
        # Process chunks with progress tracking
        with tqdm(total=self.sample_size, desc="Loading addresses", unit="records") as pbar:
            for chunk in df_iter:
                # Process entire chunk at once
                if self.sample_size and records_yielded + len(chunk) > self.sample_size:
                    # Trim chunk to fit sample size
                    chunk = chunk.iloc[:(self.sample_size - records_yielded)]
                
                # Convert chunk to records
                records = [
                    {
                        'nPLZ': row['nPLZ'],
                        'cOrtsname': row['cOrtsname'],
                        'cStrassenname': row['cStrassenname'],
                        'full_address': f"{row['nPLZ']} {row['cOrtsname']} {row['cStrassenname']}"
                    }
                    for _, row in chunk.iterrows()
                ]
                
                # Update progress and yield records
                records_yielded += len(records)
                pbar.update(len(records))
                
                for record in records:
                    yield record
                
                if self.sample_size and records_yielded >= self.sample_size:
                    return
                    
                    if records_yielded % 1000 == 0:  # Log every 1000 records
                        mem = psutil.Process().memory_info()
                        print(f"\nMemory usage: {mem.rss / 1024 / 1024:.1f}MB")
    
    def load_full(self) -> pd.DataFrame:
        """Load entire dataset efficiently using memory mapping."""
        try:
            # Reset buffer position
            self.buffer.seek(0)
            
            # Use a more memory-efficient approach for full loading
            df = pd.read_csv(
                self.buffer,
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
        if hasattr(self, 'buffer'):
            self.buffer.close()
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'file'):
            self.file.close()
