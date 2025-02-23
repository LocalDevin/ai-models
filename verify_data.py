import pandas as pd

# Read the generated data
df = pd.read_csv("training_sets/address_training_german", sep=";")

# Check total number of pairs
print("Total number of pairs:", len(df))

# Check match distribution
print("\nMatch distribution:")
print(df["is_match"].value_counts())

# Verify only 0 and 1 are used as match scores
print("\nUnique match scores:", sorted(df["is_match"].unique()))

# Analyze variation patterns in matching pairs
matching_pairs = df[df["is_match"] == 1]
variations = {
    'plz': sum(row['nPLZ'] != row['match_nPLZ'] for _, row in matching_pairs.iterrows()),
    'city': sum(row['cOrtsname'] != row['match_cOrtsname'] for _, row in matching_pairs.iterrows()),
    'street': sum(row['cStrassenname'] != row['match_cStrassenname'] for _, row in matching_pairs.iterrows())
}

print("\nVariation Analysis:")
for field, count in variations.items():
    percentage = (count / len(matching_pairs)) * 100
    print(f"{field.title()} variations: {count} ({percentage:.1f}%)")

# Check for duplicates in both directions
forward_pairs = df[["nPLZ", "cOrtsname", "cStrassenname", 
                   "match_nPLZ", "match_cOrtsname", "match_cStrassenname"]].apply(tuple, axis=1)
reverse_pairs = df[["match_nPLZ", "match_cOrtsname", "match_cStrassenname",
                   "nPLZ", "cOrtsname", "cStrassenname"]].apply(tuple, axis=1)
all_pairs = pd.concat([forward_pairs, reverse_pairs])
duplicates = all_pairs.duplicated()
print("\nNumber of duplicates (including reverse pairs):", duplicates.sum())

# Analyze variation types
print("\nSample variations in matching pairs:")
varied_pairs = matching_pairs[
    (matching_pairs["nPLZ"] != matching_pairs["match_nPLZ"]) |
    (matching_pairs["cOrtsname"] != matching_pairs["match_cOrtsname"]) |
    (matching_pairs["cStrassenname"] != matching_pairs["match_cStrassenname"])
].head(5)

for _, row in varied_pairs.iterrows():
    print(f"\nOriginal:  {row['nPLZ']}, {row['cOrtsname']}, {row['cStrassenname']}")
    print(f"Variation: {row['match_nPLZ']}, {row['match_cOrtsname']}, {row['match_cStrassenname']}")
