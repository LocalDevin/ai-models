import pandas as pd

# Read training data
df = pd.read_csv('/home/ubuntu/attachments/160d07c4-c77d-42f8-9231-9ea35d830d84/training_pairs.csv', sep=';')

print('Training data analysis:')
print(f'Total pairs: {len(df)}')
print(f'\nMatch distribution:')
print(df['is_match'].value_counts())

# Check for duplicates
forward = df[['nPLZ', 'cOrtsname', 'cStrassenname', 'match_nPLZ', 'match_cOrtsname', 'match_cStrassenname']].apply(tuple, axis=1)
reverse = df[['match_nPLZ', 'match_cOrtsname', 'match_cStrassenname', 'nPLZ', 'cOrtsname', 'cStrassenname']].apply(tuple, axis=1)
all_pairs = pd.concat([forward, reverse])
print(f'\nDuplicates in training data: {all_pairs.duplicated().sum()}')

# Analyze variation patterns
matching_pairs = df[df['is_match'] == 1]
variations = {
    'plz': sum(row['nPLZ'] != row['match_nPLZ'] for _, row in matching_pairs.iterrows()),
    'city': sum(row['cOrtsname'] != row['match_cOrtsname'] for _, row in matching_pairs.iterrows()),
    'street': sum(row['cStrassenname'] != row['match_cStrassenname'] for _, row in matching_pairs.iterrows())
}

print('\nVariation Analysis:')
for field, count in variations.items():
    percentage = (count / len(matching_pairs)) * 100
    print(f"{field.title()} variations: {count} ({percentage:.1f}%)")
