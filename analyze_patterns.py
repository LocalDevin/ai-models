import pandas as pd

# Read training data
df = pd.read_csv('/home/ubuntu/attachments/160d07c4-c77d-42f8-9231-9ea35d830d84/training_pairs.csv', sep=';')

print('Total pairs:', len(df))
print('\nSample matching pairs (first 10):')
matches = df[df['is_match'] == 1][['cStrassenname', 'match_cStrassenname']].head(10)
for _, row in matches.iterrows():
    print(f'{row.cStrassenname} -> {row.match_cStrassenname}')

print('\nSample city variations:')
city_vars = df[df['cOrtsname'] != df['match_cOrtsname']][['cOrtsname', 'match_cOrtsname']].head(10)
for _, row in city_vars.iterrows():
    print(f'{row.cOrtsname} -> {row.match_cOrtsname}')
