import pandas as pd
import random
import string
from typing import List, Tuple, Dict
import numpy as np

def load_addresses(chunksize=10000):
    return pd.read_csv('/home/ubuntu/attachments/f29ebeca-c58a-4b70-b4a2-2acaadcc30d8/addresses_all.csv', 
                      sep=';', 
                      chunksize=chunksize)

def create_address_variation(plz: str, ort: str, strasse: str) -> Tuple[str, str, str]:
    """Create variations of address components with higher variation rate"""
    street_variations = [
        lambda x: x.replace('strasse', 'str.'),
        lambda x: x.replace('str.', 'strasse'),
        lambda x: x.replace('ß', 'ss'),
        lambda x: x.replace('ss', 'ß'),
        lambda x: x.replace(' ', ''),
        lambda x: x.replace('-', ' '),
        lambda x: x.replace('.', ''),
        lambda x: x.lower(),
        lambda x: x.strip() + ' ',
        lambda x: ' ' + x.strip()
    ]
    
    ort_variations = [
        lambda x: x.replace(' am ', ' a.'),
        lambda x: x.replace(' an der ', ' a.d.'),
        lambda x: x.replace(' a.', ' am '),
        lambda x: x.replace(' a.d.', ' an der ')
    ]
    
    # Apply 2-3 random variations to increase diversity
    varied_strasse = strasse
    for _ in range(random.randint(2, 3)):
        varied_strasse = random.choice(street_variations)(varied_strasse)
    
    # Apply city variations
    varied_ort = ort
    if random.random() < 0.3:  # 30% chance to vary city name
        varied_ort = random.choice(ort_variations)(varied_ort)
    
    return plz, varied_ort, varied_strasse

def generate_matching_pair(row: pd.Series) -> Dict:
    plz, ort, strasse = create_address_variation(row['nPLZ'], row['cOrtsname'], row['cStrassenname'])
    return {
        'nPLZ': row['nPLZ'],
        'cOrtsname': row['cOrtsname'],
        'cStrassenname': row['cStrassenname'],
        'match_nPLZ': plz,
        'match_cOrtsname': ort,
        'match_cStrassenname': strasse,
        'is_match': 1
    }

def generate_non_matching_pair(row1: pd.Series, row2: pd.Series) -> Dict:
    return {
        'nPLZ': row1['nPLZ'],
        'cOrtsname': row1['cOrtsname'],
        'cStrassenname': row1['cStrassenname'],
        'match_nPLZ': row2['nPLZ'],
        'match_cOrtsname': row2['cOrtsname'],
        'match_cStrassenname': row2['cStrassenname'],
        'is_match': 0
    }

def get_pair_key(pair: Dict) -> str:
    """Get a normalized key for a pair that handles variations and reverse pairs"""
    def normalize(plz: str, ort: str, strasse: str) -> str:
        # More aggressive normalization to catch duplicates
        norm_strasse = (strasse.lower()
                       .replace('str.', 'strasse')
                       .replace('straße', 'strasse')
                       .replace('ß', 'ss')
                       .replace('-', ' ')
                       .replace('.', '')
                       .strip())
        norm_ort = (ort.lower()
                   .replace(' am ', ' ')
                   .replace(' an der ', ' ')
                   .replace(' a.', ' ')
                   .replace(' a.d.', ' ')
                   .strip())
        return f"{plz}|{norm_ort}|{norm_strasse}"
    
    key1 = normalize(pair['nPLZ'], pair['cOrtsname'], pair['cStrassenname'])
    key2 = normalize(pair['match_nPLZ'], pair['match_cOrtsname'], pair['match_cStrassenname'])
    return '||'.join(sorted([key1, key2]))

def generate_pairs(n_pairs: int) -> pd.DataFrame:
    pairs = []
    seen_pairs = set()
    
    n_matches = n_pairs // 2
    n_non_matches = n_pairs - n_matches
    
    print("Generating matching pairs...")
    match_count = 0
    for chunk_idx, chunk in enumerate(load_addresses(chunksize=1000)):
        if match_count >= n_matches:
            break
        
        if chunk_idx % 10 == 0:
            print(f"Processing chunk {chunk_idx}, matches found: {match_count}/{n_matches}")
            
        for _, row in chunk.iterrows():
            if match_count >= n_matches:
                break
                
            # Try to generate a pair with 20% probability to avoid processing every row
            if random.random() < 0.2:
                pair = generate_matching_pair(row)
                pair_key = (
                    f"{pair['nPLZ']}{pair['cOrtsname']}{pair['cStrassenname']}", 
                    f"{pair['match_nPLZ']}{pair['match_cOrtsname']}{pair['match_cStrassenname']}"
                )
                if pair_key not in seen_pairs and (pair_key[1], pair_key[0]) not in seen_pairs:
                    pairs.append(pair)
                    seen_pairs.add(pair_key)
                    match_count += 1
    
    # Generate non-matching pairs using chunk-based approach
    non_match_count = 0
    chunk_cache = []
    
    print("\nCaching chunks for non-matching pairs...")
    for chunk_idx, chunk in enumerate(load_addresses(chunksize=100)):
        chunk_cache.append(chunk)
        if chunk_idx % 5 == 0:
            print(f"Cached {len(chunk_cache)} chunks")
        if len(chunk_cache) >= 20:  # Keep 20 chunks in memory for more variety
            break
    print(f"Cached {len(chunk_cache)} chunks total")
    
    print("\nGenerating non-matching pairs...")
    while non_match_count < n_non_matches and len(chunk_cache) >= 2:
        if non_match_count % 100 == 0:
            print(f"Generated {non_match_count}/{n_non_matches} non-matching pairs")
        # Get two random chunks from our cache
        chunk_indices = random.sample(range(len(chunk_cache)), 2)
        chunk1 = chunk_cache[chunk_indices[0]]
        chunk2 = chunk_cache[chunk_indices[1]]
        
        # Try to generate pairs from these chunks
        for _ in range(min(10, n_non_matches - non_match_count)):
            row1 = chunk1.iloc[random.randint(0, len(chunk1)-1)]
            row2 = chunk2.iloc[random.randint(0, len(chunk2)-1)]
            
            pair = generate_non_matching_pair(row1, row2)
            pair_key = get_pair_key(pair)
            
            if pair_key not in seen_pairs:
                pairs.append(pair)
                seen_pairs.add(pair_key)
                non_match_count += 1
                if non_match_count >= n_non_matches:
                    break
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Create DataFrame
    df = pd.DataFrame(pairs)
    return df

if __name__ == '__main__':
    # Generate 5000 pairs
    pairs_df = generate_pairs(5000)
    
    # Save to file
    output_file = 'training_sets/address_training_german'
    pairs_df.to_csv(output_file, index=False)
    
    # Save to file with semicolon separator to match input format
    output_file = 'training_sets/address_training_german'
    pairs_df.to_csv(output_file, index=False, sep=';')
    
    # Print statistics
    print(f"Total pairs generated: {len(pairs_df)}")
    print("\nMatch distribution:")
    print(pairs_df['is_match'].value_counts())
    
    # Verify uniqueness
    duplicates = pairs_df.duplicated(subset=['nPLZ', 'cOrtsname', 'cStrassenname', 
                                           'match_nPLZ', 'match_cOrtsname', 'match_cStrassenname'])
    print("\nNumber of duplicate pairs:", duplicates.sum())
