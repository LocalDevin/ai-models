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

# Check for duplicates in both directions
forward_pairs = df[["nPLZ", "cOrtsname", "cStrassenname", 
                   "match_nPLZ", "match_cOrtsname", "match_cStrassenname"]].apply(tuple, axis=1)
reverse_pairs = df[["match_nPLZ", "match_cOrtsname", "match_cStrassenname",
                   "nPLZ", "cOrtsname", "cStrassenname"]].apply(tuple, axis=1)
all_pairs = pd.concat([forward_pairs, reverse_pairs])
duplicates = all_pairs.duplicated()
print("\nNumber of duplicates (including reverse pairs):", duplicates.sum())

# Verify variations in matching pairs
matching_pairs = df[df["is_match"] == 1]
exact_matches = (matching_pairs["cStrassenname"] == matching_pairs["match_cStrassenname"]).sum()
print("\nMatching pairs analysis:")
print(f"Total matching pairs: {len(matching_pairs)}")
print(f"Exact street name matches: {exact_matches}")
print(f"Varied street names: {len(matching_pairs) - exact_matches}")
