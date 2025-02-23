import pandas as pd
import random
from typing import Dict, Tuple

# Common German city abbreviations
CITY_ABBREV = {
    'am Main': ['a.M.', 'a. M.', '/M', '/Main'],
    'am': ['a.'],
    'an der': ['a.d.', 'a. d.'],
}

# Common spelling variations
UMLAUT_VARS = {
    'ä': ['ae', 'a'],
    'ö': ['oe', 'o'],
    'ü': ['ue', 'u'],
    'ß': ['ss', 's'],
}

# Common street abbreviations
STREET_ABBREV = {
    'strasse': ['str.', 'Str.', 'strase', 'str'],
    'straße': ['str.', 'Str.', 'strasse', 'strase', 'str'],
    'platz': ['pl.', 'Pl.'],
    'weg': ['w.', 'W.'],
}

# Common typo patterns
TYPO_PATTERNS = {
    'ss': ['s'],
    'tt': ['t'],
    'nn': ['n'],
    'ff': ['f'],
    'mm': ['m'],
}

def load_addresses(chunksize=10000):
    return pd.read_csv('/home/ubuntu/attachments/f29ebeca-c58a-4b70-b4a2-2acaadcc30d8/addresses_all.csv', 
                      sep=';', 
                      chunksize=chunksize)

def create_plz_variation(plz: str, variation_rate: float = 0.15) -> str:
    """Create realistic PLZ variations including typos"""
    if random.random() > variation_rate:
        return str(plz)
    
    plz_str = str(plz)
    variation_type = random.random()
    if variation_type < 0.7:  # Neighboring PLZ
        plz_num = int(plz_str)
        return str(plz_num + random.choice([-1, 1]))
    else:  # Typo
        pos = random.randint(0, 4)
        digit = random.randint(0, 9)
        return plz_str[:pos] + str(digit) + plz_str[pos+1:]

def create_street_variation(street: str) -> str:
    """Create street name variations with spelling mistakes and abbreviations"""
    variations = []
    
    # Standard abbreviations from dictionary (high priority)
    lower_street = street.lower()
    for full, abbrevs in STREET_ABBREV.items():
        if full in lower_street:
            variations.extend([street.replace(full, abbrev) for abbrev in abbrevs])
            variations.extend([street.replace(full.title(), abbrev) for abbrev in abbrevs])
    
    # Umlaut variations with potential spelling mistakes
    for umlaut, replacements in UMLAUT_VARS.items():
        if umlaut in street:
            for repl in replacements:
                var = street.replace(umlaut, repl)
                variations.append(var)
                # Add spelling mistake variations for umlaut replacements
                if 'ss' in var.lower():
                    variations.append(var.replace('ss', 's'))
                if 'str' in var.lower():
                    variations.append(var.replace('str', 'str.'))
    
    # Common typos and spelling mistakes (high priority)
    for pattern, replacements in TYPO_PATTERNS.items():
        if pattern in street.lower():
            for repl in replacements:
                variations.append(street.replace(pattern, repl))
    
    # Add space variations
    if ' ' in street:
        variations.append(street.replace(' ', ''))
    else:
        pos = len(street) // 2
        variations.append(street[:pos] + ' ' + street[pos:])
    
    # Add more spelling mistakes
    if len(street) > 5:
        pos = random.randint(1, len(street)-2)
        variations.append(street[:pos] + street[pos+1:])  # Remove character
        variations.append(street[:pos] + street[pos] + street[pos:])  # Double character
    
    # Almost always return a variation to match training data
    if variations and random.random() < 0.95:  # 95% chance to vary
        return random.choice(variations)
    return street

def create_city_variation(city: str) -> str:
    """Create city name variations including abbreviations and spelling mistakes"""
    variations = []
    
    # Official abbreviations (high priority)
    for full, abbrevs in CITY_ABBREV.items():
        if full in city:
            for abbrev in abbrevs:
                variations.append(city.replace(full, abbrev))
    
    # Special cases for major cities (high priority)
    if 'Frankfurt am Main' in city:
        variations.extend(['Ffm', 'Frankfurt', 'Frankfurt/M', 'Frankfurt a.M.'])
    elif city == 'München':
        variations.extend(['Munich', 'Munchen', 'Muenchen'])
    elif city == 'Köln':
        variations.extend(['Koeln', 'Cologne', 'Koln'])
    
    # Umlaut variations with potential spelling mistakes
    for umlaut, replacements in UMLAUT_VARS.items():
        if umlaut in city:
            for repl in replacements:
                var = city.replace(umlaut, repl)
                variations.append(var)
                # Add regional spelling variants
                if 'Berlin' in city:
                    variations.append(var.replace('e', 'ä'))
    
    # Add spelling mistakes for longer names
    if len(city) > 5:
        pos = random.randint(1, len(city)-2)
        variations.append(city[:pos] + city[pos+1:])  # Remove character
        variations.append(city[:pos] + city[pos] + city[pos:])  # Double character
    
    # Almost always return a variation to match training data
    if variations and random.random() < 0.95:  # 95% chance to vary
        return random.choice(variations)
    return city

def create_address_variation(plz: str, ort: str, strasse: str) -> Tuple[str, str, str]:
    """Create variations of address components with higher variation rate"""
    # Independent variation chances for each component
    varied_plz = plz
    varied_ort = ort
    varied_strasse = strasse
    
    # Force variation rates to match training data patterns
    if random.random() < 0.07:  # Target: 6.1%
        varied_plz = create_plz_variation(plz, variation_rate=1.0)
    if random.random() < 0.99:  # Target: 36.4% - Almost always vary cities
        varied_ort = create_city_variation(ort)
    varied_strasse = create_street_variation(strasse)  # Always vary streets (84.8%)
    
    # If no variations were applied, ensure at least one variation
    if varied_plz == plz and varied_ort == ort and varied_strasse == strasse:
        # Choose component based on training data distribution
        weights = [0.061, 0.364, 0.848]  # Exact weights from training data
        component = random.choices(['plz', 'ort', 'strasse'], weights=weights)[0]
        if component == 'plz':
            varied_plz = create_plz_variation(plz, variation_rate=1.0)
        elif component == 'ort':
            varied_ort = create_city_variation(ort)
        else:
            varied_strasse = create_street_variation(strasse)
    
    # Ensure at least one variation for matching pairs
    if varied_plz == plz and varied_ort == ort and varied_strasse == strasse:
        weights = [0.06, 0.36, 0.85]  # Weights based on training data patterns
        component = random.choices(['plz', 'ort', 'strasse'], weights=weights)[0]
        if component == 'plz':
            varied_plz = create_plz_variation(plz)
        elif component == 'ort':
            varied_ort = create_city_variation(ort)
        else:
            varied_strasse = create_street_variation(strasse)
    
    return varied_plz, varied_ort, varied_strasse

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
    """Get a normalized key for a pair that handles variations and reverse pairs.
    Returns a canonical form of the pair that will match regardless of variation patterns."""
    # Create both forward and reverse pairs for better duplicate detection
    pair1 = (str(pair['nPLZ']).strip(), pair['cOrtsname'].strip(), pair['cStrassenname'].strip())
    pair2 = (str(pair['match_nPLZ']).strip(), pair['match_cOrtsname'].strip(), pair['match_cStrassenname'].strip())
    # Sort to ensure A->B and B->A generate the same key
    first, second = sorted([pair1, pair2])
    def normalize(plz: str, ort: str, strasse: str) -> str:
        # Aggressive normalization to catch all variations
        # More aggressive normalization to catch variations
        # Normalize to base form without any variations
        norm_strasse = ''.join(c.lower() for c in str(strasse) if c.isalnum())
        norm_strasse = (norm_strasse
                       .replace('strasse', '')
                       .replace('str', '')
                       .replace('platz', '')
                       .replace('weg', '')
                       .replace('ss', 's')
                       .replace('ae', 'a')
                       .replace('oe', 'o')
                       .replace('ue', 'u'))
        
        # Normalize city name to base form
        norm_ort = ''.join(c.lower() for c in str(ort) if c.isalnum())
        norm_ort = (norm_ort
                   .replace('frankfurtammain', 'frankfurt')
                   .replace('frankfurtam', 'frankfurt')
                   .replace('ffm', 'frankfurt')
                   .replace('ss', 's')
                   .replace('ae', 'a')
                   .replace('oe', 'o')
                   .replace('ue', 'u'))
        
        return f"{plz}|{norm_ort}|{norm_strasse}"
    
    key1 = normalize(first[0], first[1], first[2])
    key2 = normalize(second[0], second[1], second[2])
    return f"{key1}||{key2}"

def generate_pairs(n_pairs: int) -> pd.DataFrame:
    pairs = []
    seen_normalized = set()  # Track normalized pairs to avoid duplicates
    
    def is_duplicate(pair: Dict) -> bool:
        """Check if a pair or its reverse is already in the dataset"""
        norm_key = get_pair_key(pair)
        # Create reverse pair
        reverse_pair = {
            'nPLZ': pair['match_nPLZ'],
            'cOrtsname': pair['match_cOrtsname'],
            'cStrassenname': pair['match_cStrassenname'],
            'match_nPLZ': pair['nPLZ'],
            'match_cOrtsname': pair['cOrtsname'],
            'match_cStrassenname': pair['cStrassenname'],
            'is_match': pair['is_match']
        }
        reverse_key = get_pair_key(reverse_pair)
        return norm_key in seen_normalized or reverse_key in seen_normalized
    
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
                if not is_duplicate(pair):
                    pairs.append(pair)
                    seen_normalized.add(get_pair_key(pair))
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
            if not is_duplicate(pair):
                pairs.append(pair)
                seen_normalized.add(get_pair_key(pair))
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
