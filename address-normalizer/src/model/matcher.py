import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import gc
import os
import pickle
from pathlib import Path

# Constants for model persistence and language support
MODELS_DIR = Path("models")
DEFAULT_LANGUAGE = "DE"
SUPPORTED_LANGUAGES = ["DE"]  # Expandable for future languages
EMBEDDING_MODELS = {
    "DE": "sentence-transformers/all-MiniLM-L6-v2",  # Current model
    # Future language-specific models can be added here
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Weight multipliers for hierarchical matching
ZIP_WEIGHT = 2.0    # Highest priority
CITY_WEIGHT = 1.5   # Medium priority
STREET_WEIGHT = 1.2 # Lower priority

# Constants
EMBEDDING_DIM = 384  # MiniLM produces 384-dim vectors
BATCH_SIZE = 32     # Batch processing for better performance
NUM_WORKERS = 4     # Increase if you have more CPU cores for DataLoader
PIN_MEMORY = True if device.type == "cuda" else False

class AddressMatcher:
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language} not supported. Choose from {SUPPORTED_LANGUAGES}")
        self.language = language
        self.model_dir = MODELS_DIR / language
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.transformer = SentenceTransformer(EMBEDDING_MODELS[language])
        self.network = SiameseNetwork().to(device)
        self.reference_data = []
        self.embeddings_cache = {}
        
    def initialize_database(self, csv_path: str, sample_size: Optional[int] = None) -> None:
        """Initialize database from CSV with memory-efficient loading."""
        print("Loading reference data...")
        
        # Read CSV in chunks for memory efficiency
        chunks = []
        chunk_size = 1000  # Process in smaller chunks
        
        for chunk in pd.read_csv(csv_path, delimiter=';', 
                               chunksize=chunk_size, dtype={'nPLZ': str}):
            if sample_size and sum(len(c) for c in chunks) >= sample_size:
                break
            chunks.append(chunk)
                
        # Combine chunks
        reference_df = pd.concat(chunks)
        if sample_size and len(reference_df) > sample_size:
            reference_df = reference_df.sample(n=sample_size, random_state=42)
        
        print(f"Processing {len(reference_df)} addresses...")

        for _, row in tqdm(reference_df.iterrows(), total=len(reference_df), desc="Processing addresses"):
            addr = {
                'nPLZ': str(row['nPLZ']),
                'cOrtsname': str(row['cOrtsname']),
                'cStrassenname': str(row['cStrassenname'])
            }
            addr['full_address'] = f"{addr['nPLZ']} {addr['cOrtsname']} {addr['cStrassenname']}"
            self.reference_data.append(addr)
            if addr['full_address'] not in self.embeddings_cache:
                # Cache embeddings directly on CPU to save GPU memory if needed
                self.embeddings_cache[addr['full_address']] = self.transformer.encode(addr['full_address'])
        
        gc.collect()  # Free memory
            
        print(f"Loaded {len(self.reference_data)} addresses")
        self._train_network()
    
    def _train_network(self):
        """Train the Siamese network with batched processing."""
        print("Training network...")
        train_data = [
            # Basic street variations
            ("12345 Berlin Hauptstraße", "12345 Berlin Hauptstrasse", 1),
            ("12345 Berlin Hauptstraße", "12345 Berlin Hauptstr.", 1),
            ("12345 Berlin Hauptstr", "12345 Berlin Hauptstrasse", 1),
            
            # City name variations
            ("60313 Frankfurt am Main Zeil", "60313 Frankfurt a.M. Zeil", 1),
            ("60313 Frankfurt/Main Zeil", "60313 Frankfurt Zeil", 1),
            ("60313 Frankfurt a.M. Zeil", "60313 Frankfurt/M Zeil", 1),
            ("60313 Frankfurt/Main Zeil", "60313 Ffm Zeil", 1),
            
            # Special German characters
            ("70173 Stuttgart Königstraße", "70173 Stuttgart Koenigstrasse", 1),
            ("80331 München Marienplatz", "80331 Muenchen Marienplatz", 1),
            ("79539 Lörrach Hauptstraße", "79539 Loerrach Hauptstr.", 1),
            ("12345 Köln Höhenberger Str.", "12345 Koeln Hoehenberger Str.", 1),
            
            # Common abbreviations and variations
            ("10178 Berlin Alexanderplatz", "10178 Berlin Alex.-Pl.", 1),
            ("10178 Berlin Alexanderpl.", "10178 Berlin Alexander-Platz", 1),
            ("10117 Berlin Unter den Linden", "10117 Berlin U.d. Linden", 1),
            ("12345 Hamburg Sankt Georg", "12345 Hamburg St. Georg", 1),
            ("12345 Berlin Karl-Marx-Str.", "12345 Berlin Karl Marx Strasse", 1),
            ("12345 Berlin An der Spree", "12345 Berlin a.d. Spree", 1),
            ("12345 Berlin Vor dem Tor", "12345 Berlin v.d. Tor", 1),
            
            # Directional variations
            ("12345 Berlin Nord", "12345 Berlin-N", 1),
            ("12345 Berlin Süd", "12345 Berlin-S", 1),
            ("12345 Berlin West", "12345 Berlin-W", 1),
            ("12345 Berlin Ost", "12345 Berlin-O", 1),
            
            # Negative examples (different addresses)
            ("10115 Berlin Invalidenstraße", "10117 Berlin Friedrichstraße", 0),
            ("60313 Frankfurt Zeil", "80331 München Kaufingerstraße", 0),
            ("70173 Stuttgart Königstraße", "70174 Stuttgart Calwerstraße", 0),
            ("12345 Berlin-Nord", "12345 Berlin-Süd", 0),
            ("12345 Hamburg St. Pauli", "12345 Hamburg St. Georg", 0)
        ]
        
        dataset = AddressDataset(self.reference_data, self.transformer, self.embeddings_cache, hardcoded_pairs=train_data)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        self.network.train()
        for epoch in tqdm(range(50), desc="Training epochs"):
            epoch_loss = 0
            for emb1, emb2, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.network(emb1, emb2)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        self.network.eval()
        # Save model after training
        self.save_model()
    
    def find_matches(self, postal_code: str, city: str, street: str, k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        """Find matches using hierarchical weighting system."""
        query = f"{postal_code} {city} {street}"
        
        # Use cached embedding or compute new one
        if query not in self.embeddings_cache:
            self.embeddings_cache[query] = self.transformer.encode(query)
        query_embedding = self.embeddings_cache[query]
        
        matches = []
        for ref_addr in tqdm(self.reference_data, desc="Finding matches"):
            # Get cached embedding
            ref_embedding = self.embeddings_cache[ref_addr['full_address']]
            
            # Calculate base similarity score
            base_score = self.network(
                torch.tensor(query_embedding).unsqueeze(0),
                torch.tensor(ref_embedding).unsqueeze(0)
            ).item()
            
            # Apply hierarchical weighting
            score = base_score
            
            # ZIP code match (highest weight)
            if ref_addr['nPLZ'] == postal_code:
                score *= ZIP_WEIGHT
            
            # City match (medium weight) with partial matching
            if ref_addr['cOrtsname'].lower() == city.lower():
                score *= CITY_WEIGHT
            elif self._partial_match(ref_addr['cOrtsname'].lower(), city.lower()):
                score *= CITY_WEIGHT * 0.8  # Partial match gets 80% of weight
            
            # Street match (lower weight) with partial matching
            if ref_addr['cStrassenname'].lower() == street.lower():
                score *= STREET_WEIGHT
            elif self._partial_match(ref_addr['cStrassenname'].lower(), street.lower()):
                score *= STREET_WEIGHT * 0.8  # Partial match gets 80% of weight
            
            # Normalize score to [0,1] range
            score = min(1.0, score)
            matches.append((ref_addr, score))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)[:k]
    
    def _partial_match(self, str1: str, str2: str) -> bool:
        """Check if strings partially match."""
        return str1 in str2 or str2 in str1
        
    def save_model(self, model_name: str = "latest") -> None:
        """Save trained model and embeddings cache."""
        model_path = self.model_dir / f"{model_name}.pt"
        cache_path = self.model_dir / f"{model_name}_cache.pkl"
        reference_path = self.model_dir / f"{model_name}_reference.pkl"
        
        torch.save({
            'network_state': self.network.state_dict(),
            'language': self.language,
            'model_name': model_name
        }, model_path)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
            
        with open(reference_path, 'wb') as f:
            pickle.dump(self.reference_data, f)
            
    def load_model(self, model_name: str = "latest") -> None:
        """Load trained model and embeddings cache."""
        model_path = self.model_dir / f"{model_name}.pt"
        cache_path = self.model_dir / f"{model_name}_cache.pkl"
        reference_path = self.model_dir / f"{model_name}_reference.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No saved model found at {model_path}")
        
        checkpoint = torch.load(model_path)
        if checkpoint['language'] != self.language:
            raise ValueError(f"Model language {checkpoint['language']} doesn't match current language {self.language}")
        
        self.network.load_state_dict(checkpoint['network_state'])
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
                
        if reference_path.exists():
            with open(reference_path, 'rb') as f:
                self.reference_data = pickle.load(f)
        else:
            raise FileNotFoundError(f"No reference data found at {reference_path}")

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, emb1, emb2):
        diff = torch.abs(emb1 - emb2)
        output = self.fc(diff)
        return output

class AddressDataset(Dataset):
    def __init__(self, reference_data, model, embeddings_cache=None, hardcoded_pairs=None):
        self.reference_data = reference_data
        self.model = model
        self.embeddings_cache = embeddings_cache or {}
        self.pairs = []
        
        # Add hardcoded training pairs if provided
        if hardcoded_pairs:
            self.pairs.extend(hardcoded_pairs)
        
        # Generate training pairs from reference data
        print("Generating training pairs...")
        addresses = [(addr['nPLZ'], addr['cOrtsname'], addr['cStrassenname'], addr['full_address']) 
                    for addr in reference_data]
        
        # Group addresses by ZIP and city for efficient pair generation
        zip_groups = {}
        city_groups = {}
        for addr in addresses:
            zip_code, city = addr[0], addr[1].lower()
            if zip_code not in zip_groups:
                zip_groups[zip_code] = []
            if city not in city_groups:
                city_groups[city] = []
            zip_groups[zip_code].append(addr)
            city_groups[city].append(addr)
        
        # Generate positive pairs (same ZIP or city)
        max_pairs_per_group = 10  # Increased from 5 for better coverage
        
        # First, add pairs with same ZIP code (highest priority)
        for zip_code, group in zip_groups.items():
            if len(group) > 1:
                # Take pairs with similar street names for better learning
                sorted_group = sorted(group, key=lambda x: x[2])  # Sort by street name
                for i in range(min(len(group), max_pairs_per_group)):
                    addr1 = sorted_group[i]
                    addr2 = sorted_group[(i + 1) % len(sorted_group)]
                    self.pairs.append((addr1[3], addr2[3], 1))
        
        # Then add pairs with same city but different ZIP (medium priority)
        for city, group in city_groups.items():
            if len(group) > 1:
                # Group by similar street patterns
                street_patterns = {}
                for addr in group:
                    pattern = addr[2].split()[0]  # First word of street name
                    if pattern not in street_patterns:
                        street_patterns[pattern] = []
                    street_patterns[pattern].append(addr)
                
                # Generate pairs within similar street patterns
                for pattern, pattern_group in street_patterns.items():
                    if len(pattern_group) > 1:
                        for i in range(min(len(pattern_group), max_pairs_per_group)):
                            addr1 = pattern_group[i]
                            addr2 = pattern_group[(i + 1) % len(pattern_group)]
                            if addr1[0] != addr2[0]:  # Only if different ZIP codes
                                self.pairs.append((addr1[3], addr2[3], 1))
        
        # Generate negative pairs (different ZIP and city)
        import random
        random.seed(42)
        num_negative = len(self.pairs)  # Balance positive and negative examples
        all_addresses = list(addresses)
        
        # Create groups by first digit of ZIP for more meaningful negative examples
        zip_first_digit = {}
        for addr in all_addresses:
            first_digit = addr[0][0]
            if first_digit not in zip_first_digit:
                zip_first_digit[first_digit] = []
            zip_first_digit[first_digit].append(addr)
        
        # Generate negative pairs from different regions
        for _ in range(num_negative):
            # Pick two different regions
            regions = random.sample(list(zip_first_digit.keys()), 2)
            addr1 = random.choice(zip_first_digit[regions[0]])
            addr2 = random.choice(zip_first_digit[regions[1]])
            self.pairs.append((addr1[3], addr2[3], 0))
        
        print(f"Generated {len(self.pairs)} training pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        addr1, addr2, label = self.pairs[idx]
        
        # Cache embeddings if not already computed
        if addr1 not in self.embeddings_cache:
            self.embeddings_cache[addr1] = self.model.encode(addr1)
        if addr2 not in self.embeddings_cache:
            self.embeddings_cache[addr2] = self.model.encode(addr2)
            
        emb1 = torch.tensor(self.embeddings_cache[addr1], dtype=torch.float32).to(device)
        emb2 = torch.tensor(self.embeddings_cache[addr2], dtype=torch.float32).to(device)
        return emb1, emb2, torch.tensor(label, dtype=torch.float32).to(device)

def main():
    # Initialize matcher
    matcher = AddressMatcher()
    
    # Load and process Strassen.csv
    strassen_path = 'test_data\\addresses.csv'
    matcher.initialize_database(strassen_path)
    
    # Comprehensive test cases with various matching scenarios
    test_cases = [
        # Exact matches
        ('75446', 'Wiernsheim', 'Lerchenweg'),
        # Misspellings
        ('75446', 'Wiernshem', 'Lerchenweg'),     # City misspelling
        ('75446', 'Wiernsheim', 'Lerchenveg'),    # Street misspelling
        ('75336', 'Wiernsheim', 'Lerchenveg'),    # Postal Code misspelling
        # Abbreviations
        ('10001', 'NYC', 'Fifth Avenue'),          # City abbreviation
        ('80331', 'Muenchen', 'Marienpl.'),       # Street abbreviation
        # Extended names
        ('60313', 'Frankfurt am Main', 'Zeil'),    # Extended city name
        ('70173', 'Stuttgart', 'Koenigstr.'),      # Street abbreviation with dots
        # Case variations
        ('12345', 'BERLIN', 'hauptstrasse'),       # Mixed case
        # Special characters
        ('80331', 'München', 'Marienplatz'),       # Umlauts
        ('70173', 'Stuttgart', 'Königstraße'),     # Multiple special chars
        # Partial matches
        ('80331', 'Munch', 'Marienpl'),           # Partial city and street
        ('60313', 'Ffm', 'Zeil Strasse'),         # Very short city abbreviation
    ]
    
    # Track accuracy metrics
    correct_matches = 0
    total_matches = len(test_cases)
    
    print("\nTesting address matching with enhanced test cases:")
    print("=" * 60)
    
    # Run test cases with detailed analysis
    print("\nRunning test cases with detailed analysis:")
    for postal_code, city, street in test_cases:
        print(f"\nInput: {postal_code} {city} {street}")
        matches = matcher.find_matches(postal_code, city, street, k=3)
        print("Top 3 matches:")
        
        # Analyze match quality
        best_score = matches[0][1] if matches else 0
        match_quality = "Excellent" if best_score > 0.9 else "Good" if best_score > 0.7 else "Poor"
        
        for addr, score in matches:
            print(f"- {addr['nPLZ']} {addr['cOrtsname']} {addr['cStrassenname']}")
            print(f"  Score: {score:.4f}")
            print(f"  Match components:")
            print(f"    ZIP: {'✓' if addr['nPLZ'] == postal_code else '✗'}")
            print(f"    City: {'✓' if addr['cOrtsname'].lower() == city.lower() else '~' if matcher._partial_match(addr['cOrtsname'].lower(), city.lower()) else '✗'}")
            print(f"    Street: {'✓' if addr['cStrassenname'].lower() == street.lower() else '~' if matcher._partial_match(addr['cStrassenname'].lower(), street.lower()) else '✗'}")
        
        print(f"Match quality: {match_quality}")
        if best_score > 0.7:  # Consider it a correct match if score is good
            correct_matches += 1
    
    # Print overall accuracy metrics
    accuracy = correct_matches / total_matches
    print("\nOverall Performance Metrics:")
    print("=" * 60)
    print(f"Total test cases: {total_matches}")
    print(f"Correct matches: {correct_matches}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average score for top matches: {sum(matcher.find_matches(tc[0], tc[1], tc[2], k=1)[0][1] for tc in test_cases) / total_matches:.4f}")

if __name__ == '__main__':
    main()