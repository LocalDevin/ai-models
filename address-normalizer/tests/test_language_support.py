import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.matcher import AddressMatcher, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
import torch

def test_language_support():
    """Test language-specific model persistence and matching."""
    print("\nTesting language support...")
    
    # Test German model
    print("\nTesting German (DE) model:")
    de_matcher = AddressMatcher(language="DE")
    
    # Test with production dataset
    print("\nTesting with production dataset:")
    de_matcher.initialize_database("test_data/DE/addresses.csv")
    de_matcher.save_model("test_de_prod")
    
    # Test with test cases
    print("\nTesting with test cases:")
    de_matcher.initialize_database("test_data/DE/test_cases.csv")
    
    # Test matching with various German address formats
    test_cases = [
        # Standard format
        ("12345", "Berlin", "Hauptstraße"),
        # Umlaut variations
        ("80331", "München", "Marienplatz"),
        ("80331", "Muenchen", "Marienplatz"),
        # Abbreviations
        ("12345", "Berlin", "Hauptstr."),
        ("60313", "Frankfurt a.M.", "Zeil"),
        # Special characters
        ("70173", "Stuttgart", "Königstraße"),
        ("70173", "Stuttgart", "Koenigstrasse")
    ]
    
    print("\nTesting address matching:")
    for postal_code, city, street in test_cases:
        print(f"\nQuery: {postal_code} {city} {street}")
        matches = de_matcher.find_matches(postal_code, city, street, k=3)
        print("Matches:")
        for addr, score in matches:
            print(f"  - {addr['full_address']} (score: {score:.4f})")
    
    # Test model loading
    print("\nTesting model loading:")
    new_matcher = AddressMatcher(language="DE")
    new_matcher.load_model("test_de_prod")
    
    # Test GPU memory usage if available
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

if __name__ == '__main__':
    test_language_support()
