import pandas as pd
import difflib

def analyze_variations(str1, str2):
    """Analyze the type of variation between two strings"""
    if str1 == str2:
        return "exact_match"
    
    # Common abbreviation patterns
    if "strasse" in str1.lower() and "str." in str2.lower():
        return "abbreviation"
    if "am main" in str1.lower() and "a.m." in str2.lower():
        return "city_abbreviation"
        
    # Calculate string similarity
    similarity = difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    if similarity > 0.8:
        # Check for umlaut variations
        if ('ä' in str1 and 'ae' in str2) or ('ö' in str1 and 'oe' in str2) or ('ü' in str1 and 'ue' in str2):
            return "umlaut_variation"
        # Check for spelling mistakes
        return "spelling_variation"
    
    return "other"

# Read training data
df = pd.read_csv('/home/ubuntu/attachments/160d07c4-c77d-42f8-9231-9ea35d830d84/training_pairs.csv', sep=';')

# Analyze variations in matching pairs
matching_pairs = df[df['is_match'] == 1]

variation_types = {
    'street': [],
    'city': [],
    'plz': []
}

for _, row in matching_pairs.iterrows():
    # Analyze street variations
    if row['cStrassenname'] != row['match_cStrassenname']:
        var_type = analyze_variations(row['cStrassenname'], row['match_cStrassenname'])
        variation_types['street'].append(var_type)
    
    # Analyze city variations
    if row['cOrtsname'] != row['match_cOrtsname']:
        var_type = analyze_variations(row['cOrtsname'], row['match_cOrtsname'])
        variation_types['city'].append(var_type)
    
    # Analyze PLZ variations
    if row['nPLZ'] != row['match_nPLZ']:
        variation_types['plz'].append('plz_variation')

print("\nVariation Analysis:")
for field, variations in variation_types.items():
    if variations:
        print(f"\n{field.title()} Variations:")
        total = len(variations)
        for var_type in set(variations):
            count = variations.count(var_type)
            percentage = (count / total) * 100
            print(f"- {var_type}: {count} ({percentage:.1f}%)")

print("\nSample variations:")
for field in ['street', 'city']:
    print(f"\n{field.title()} variation examples:")
    sample = matching_pairs[matching_pairs[f'c{field.title()}sname'] != matching_pairs[f'match_c{field.title()}sname']].head(5)
    for _, row in sample.iterrows():
        print(f"- {row[f'c{field.title()}sname']} -> {row[f'match_c{field.title()}sname']}")
