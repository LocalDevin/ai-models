import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import psutil
from pathlib import Path

class AddressDataset(Dataset):
    def __init__(self, addresses: List[dict], model, num_workers: int = 4):
        """Initialize dataset with addresses and model for embeddings."""
        self.addresses = addresses
        self.model = model
        self.pairs = []
        self.embeddings_cache = {}
        print(f"\nGenerating training pairs using {num_workers} workers...")
        self.pairs = self._generate_pairs(num_workers)
        
    def _process_group(self, group: List[Dict]) -> List[Tuple]:
        """Process a group of addresses to generate training pairs."""
        pairs = []
        # Generate positive pairs within the same postal code prefix
        for i in range(len(group)):
            for j in range(i + 1, min(i + 5, len(group))):
                addr1, addr2 = group[i], group[j]
                # Calculate similarity score based on address components
                score = self._calculate_similarity(addr1, addr2)
                if score > 0.5:  # Only include meaningful positive pairs
                    pairs.append((addr1['full_address'], addr2['full_address'], 1))
        
        # Generate negative pairs from different postal codes
        if len(group) > 1:
            for i in range(min(5, len(group))):
                j = (i + len(group) // 2) % len(group)
                addr1, addr2 = group[i], group[j]
                pairs.append((addr1['full_address'], addr2['full_address'], 0))
        
        return pairs
    
    def _calculate_similarity(self, addr1: Dict, addr2: Dict) -> float:
        """Calculate similarity score between two addresses."""
        score = 0.0
        # ZIP code match (highest weight)
        if addr1['nPLZ'] == addr2['nPLZ']:
            score += 0.4
        
        # City match (medium weight)
        if addr1['cOrtsname'].lower() == addr2['cOrtsname'].lower():
            score += 0.3
        elif self._partial_match(addr1['cOrtsname'].lower(), addr2['cOrtsname'].lower()):
            score += 0.2
        
        # Street match (lower weight)
        if addr1['cStrassenname'].lower() == addr2['cStrassenname'].lower():
            score += 0.3
        elif self._partial_match(addr1['cStrassenname'].lower(), addr2['cStrassenname'].lower()):
            score += 0.2
        
        return score
    
    def _partial_match(self, str1: str, str2: str) -> bool:
        """Check if strings partially match."""
        return str1 in str2 or str2 in str1
    
    def _generate_pairs(self, num_workers: int) -> List[Tuple]:
        """Generate training pairs using parallel processing."""
        # Group addresses by postal code prefix for efficient pairing
        groups = {}
        for addr in self.addresses:
            prefix = addr['nPLZ'][:2]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(addr)
        
        pairs = []
        total_groups = len(groups)
        
        # Process groups in parallel with progress tracking
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for prefix, group in groups.items():
                futures.append(executor.submit(self._process_group, group))
            
            # Monitor progress and memory usage
            with tqdm(total=total_groups, desc="Generating pairs") as pbar:
                for future in futures:
                    group_pairs = future.result()
                    pairs.extend(group_pairs)
                    pbar.update(1)
                    
                    # Log memory usage periodically
                    if len(pairs) % 10000 == 0:
                        mem = psutil.Process().memory_info()
                        print(f"\nPairs generated: {len(pairs)}, Memory usage: {mem.rss / 1024 / 1024:.1f}MB")
        
        print(f"\nGenerated {len(pairs)} training pairs")
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Get a pair of addresses and their similarity label."""
        addr1, addr2, label = self.pairs[idx]
        
        # Cache embeddings for efficiency
        if addr1 not in self.embeddings_cache:
            self.embeddings_cache[addr1] = torch.tensor(
                self.model.encode(addr1), dtype=torch.float32
            )
        if addr2 not in self.embeddings_cache:
            self.embeddings_cache[addr2] = torch.tensor(
                self.model.encode(addr2), dtype=torch.float32
            )
        
        return (
            self.embeddings_cache[addr1],
            self.embeddings_cache[addr2],
            torch.tensor(label, dtype=torch.float32)
        )
