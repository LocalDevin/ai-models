import pandas as pd
import random
from typing import Dict, Tuple

# City abbreviation patterns from reference file
CITY_ABBREV = {
    'frankfurt am main': ['Frankfurt a.M.', 'Frankfurt/M', 'Frankfurt/Main', 'Ffm', 'Frankfurt'],
    'münchen': ['Muenchen'],
    'köln': ['Koeln'],
    'berlin': ['Bärlin', 'Berln'],
    'sankt': ['St.']
}

# Common spelling variations
UMLAUT_VARS = {
    'ä': ['ae', 'a'],
    'ö': ['oe', 'o'],
    'ü': ['ue', 'u'],
    'ß': ['ss', 's'],
}

# Street abbreviation patterns from reference file
STREET_ABBREV = {
    'hauptstraße': ['Hauptstrasse', 'Hauptstr.'],
    'straße': ['strasse', 'str.', 'Str.'],
    'platz': ['Pl.', '-Pl.', 'platz'],
    'alexander': ['Alex.-Pl.', 'Alexanderpl.', 'Alexander-Platz'],
    'unter den': ['U.d.'],
    'sankt': ['St.'],
    'an der': ['a.d.'],
    'vor dem': ['v.d.'],
    'nord': ['Berlin-N'],
    'süd': ['Berlin-S'],
    'west': ['Berlin-W'],
    'ost': ['Berlin-O'],
    'karl marx': ['Karl Marx Strasse'],
    'königstraße': ['Koenigstrasse', 'Königstr.'],
    'höhenberger': ['Hoehenberger'],
    'friedrichstraße': ['Friedrichstr.', 'Friedrichstraße'],
    'invalidenstraße': ['Invalidenstraße'],
    'chausseestraße': ['Chaussee Str.'],
    'calwerstraße': ['Calwerstraße'],
    'goethestraße': ['Goethestr.', 'Goethe Str.']
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
    plz_str = str(plz)
    plz_num = int(plz_str)
    return str(plz_num + random.choice([-1, 1]))

def create_street_variation(street: str) -> str:
    """Create street name variations using reference patterns"""
    lower_street = street.lower()
    
    # Follow exact patterns from reference data (12 variations total)
    
    # Hauptstraße variations (3 pairs)
    if 'hauptstraße' in lower_street:
        return random.choice(['Hauptstrasse', 'Hauptstr.'])
    
    # Alexanderplatz variations (2 pairs)
    if 'alexanderplatz' in lower_street:
        return random.choice(['Alex.-Pl.', 'Alexanderpl.', 'Alexander-Platz'])
    
    # Special forms (3 pairs)
    if 'unter den linden' in lower_street:
        return 'U.d. Linden'
    if 'karl-marx-str' in lower_street:
        return 'Karl Marx Strasse'
    if 'an der spree' in lower_street:
        return 'a.d. Spree'
    if 'vor dem tor' in lower_street:
        return 'v.d. Tor'
    
    # Directional forms (4 pairs)
    if 'nord' in lower_street and len(street.split()) == 1:
        return 'Berlin-N'
    if 'süd' in lower_street and len(street.split()) == 1:
        return 'Berlin-S'
    if 'west' in lower_street and len(street.split()) == 1:
        return 'Berlin-W'
    if 'ost' in lower_street and len(street.split()) == 1:
        return 'Berlin-O'
    
    # Standard variations for other streets
    if 'straße' in lower_street:
        return street.replace('straße', random.choice(['strasse', 'str.']))
    
    return street

def create_city_variation(city: str) -> str:
    """Create city name variations using reference patterns"""
    lower_city = city.lower()
    
    # Follow exact patterns from reference data
    if 'frankfurt am main' in lower_city:
        return random.choice([
            'Frankfurt a.M.',
            'Frankfurt/M',
            'Frankfurt/Main',
            'Ffm',
            'Frankfurt'  # Added from reference
        ])
    if 'münchen' in lower_city:
        return 'Muenchen'
    if 'köln' in lower_city:
        return 'Koeln'
    if 'lörrach' in lower_city:
        return 'Loerrach'
    if 'berlin' in lower_city:
        return random.choice(['Bärlin', 'Berln'])
    if 'stuttgart' in lower_city:
        return 'Stutgart'
    
    return city

def create_address_variation(plz: str, ort: str, strasse: str) -> Tuple[str, str, str]:
    """Create variations of address components using reference patterns"""
    varied_plz = plz
    varied_ort = ort
    varied_strasse = strasse
    
    # From reference data (33 matching pairs):
    # Street variations: 28 pairs (84.8%)
    # City variations: 12 pairs (36.4%)
    # PLZ variations: 2 pairs (6.1%)
    
    lower_street = strasse.lower()
    lower_city = ort.lower()
    
    # First check if we can vary based on exact reference patterns
    r = random.random()
    
    # Try to vary street (84.8%)
    if r < 0.848:
        # Only vary if we have a pattern to match
        if 'hauptstraße' in lower_street:
            varied_strasse = random.choice(['Hauptstrasse', 'Hauptstr.'])
            return varied_plz, varied_ort, varied_strasse
        elif 'alexanderplatz' in lower_street:
            varied_strasse = random.choice(['Alex.-Pl.', 'Alexanderpl.', 'Alexander-Platz'])
            return varied_plz, varied_ort, varied_strasse
        elif 'unter den linden' in lower_street:
            varied_strasse = 'U.d. Linden'
            return varied_plz, varied_ort, varied_strasse
        elif 'sankt georg' in lower_street:
            varied_strasse = 'St. Georg'
            return varied_plz, varied_ort, varied_strasse
        elif 'karl-marx-str' in lower_street:
            varied_strasse = 'Karl Marx Strasse'
            return varied_plz, varied_ort, varied_strasse
        elif 'an der spree' in lower_street:
            varied_strasse = 'a.d. Spree'
            return varied_plz, varied_ort, varied_strasse
        elif 'vor dem tor' in lower_street:
            varied_strasse = 'v.d. Tor'
            return varied_plz, varied_ort, varied_strasse
        elif 'nord' in lower_street and len(strasse.split()) == 1:
            varied_strasse = 'Berlin-N'
            return varied_plz, varied_ort, varied_strasse
        elif 'süd' in lower_street and len(strasse.split()) == 1:
            varied_strasse = 'Berlin-S'
            return varied_plz, varied_ort, varied_strasse
        elif 'west' in lower_street and len(strasse.split()) == 1:
            varied_strasse = 'Berlin-W'
            return varied_plz, varied_ort, varied_strasse
        elif 'ost' in lower_street and len(strasse.split()) == 1:
            varied_strasse = 'Berlin-O'
            return varied_plz, varied_ort, varied_strasse
    
    # Try to vary city (36.4%)
    elif r < 0.939:
        # Only vary if we have a pattern to match
        if 'frankfurt am main' in lower_city:
            varied_ort = random.choice(['Frankfurt a.M.', 'Frankfurt/M', 'Frankfurt/Main', 'Ffm'])
            return varied_plz, varied_ort, varied_strasse
        elif 'münchen' in lower_city:
            varied_ort = 'Muenchen'
            return varied_plz, varied_ort, varied_strasse
        elif 'köln' in lower_city:
            varied_ort = 'Koeln'
            return varied_plz, varied_ort, varied_strasse
        elif 'lörrach' in lower_city:
            varied_ort = 'Loerrach'
            return varied_plz, varied_ort, varied_strasse
        elif 'berlin' in lower_city:
            varied_ort = random.choice(['Bärlin', 'Berln'])
            return varied_plz, varied_ort, varied_strasse
    
    # Try to vary PLZ (9.1%)
    else:
        varied_plz = str(int(plz) + random.choice([-1, 1]))
        return varied_plz, varied_ort, varied_strasse
    
    # If we couldn't vary anything, return None
    return None

def generate_matching_pair(row: pd.Series) -> Dict:
    """Generate a matching pair following exact reference patterns"""
    result = create_address_variation(row['nPLZ'], row['cOrtsname'], row['cStrassenname'])
    if result is None:
        return None
        
    plz, ort, strasse = result
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

def normalize(plz: str, ort: str, strasse: str) -> str:
    """Normalize address components to a canonical form for comparison"""
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

def get_pair_key(pair: Dict) -> str:
    """Get a normalized key for a pair that handles variations and reverse pairs"""
    pair1 = (str(pair['nPLZ']).strip(), pair['cOrtsname'].strip(), pair['cStrassenname'].strip())
    pair2 = (str(pair['match_nPLZ']).strip(), pair['match_cOrtsname'].strip(), pair['match_cStrassenname'].strip())
    first, second = sorted([pair1, pair2])
    
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
            
        # Try to generate pairs from each row
        for _, row in chunk.iterrows():
            if match_count >= n_matches:
                break
                
            pair = generate_matching_pair(row)
            if pair is not None and not is_duplicate(pair):
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
