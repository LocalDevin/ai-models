import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.matcher import AddressMatcher, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
import torch

def test_language_support():
    """Test language-specific model persistence and matching."""
    print("\nTesting language support...")
    
    # Clean up any existing test models
    import shutil
    test_model_dir = Path("models")
    if test_model_dir.exists():
        shutil.rmtree(test_model_dir)
    
    # Test German model
    print("\nTesting German (DE) model:")
    de_matcher = AddressMatcher(language="DE")
    
    # Test with test cases
    print("\nTesting with test cases:")
    de_matcher.initialize_database("test_data/DE/test_cases.csv")
    
    # Load test cases from CSV
    import pandas as pd
    test_data = pd.read_csv("test_data/DE/test_cases.csv", delimiter=';', dtype={'nPLZ': str})
    test_cases = [(row['nPLZ'], row['cOrtsname'], row['cStrassenname']) 
                  for _, row in test_data.iterrows()]
    
    print("\nTesting address matching:")
    for postal_code, city, street in test_cases:
        print(f"\nQuery: {postal_code} {city} {street}")
        matches = de_matcher.find_matches(postal_code, city, street, k=3)
        print("Matches:")
        for addr, score in matches:
            print(f"  - {addr['full_address']} (score: {score:.4f})")
    
    # Test model saving with overwrite
    print("\nTesting model saving with overwrite:")
    test_model_name = "test_overwrite"
    de_matcher.save_model(test_model_name)  # First save
    try:
        de_matcher.save_model(test_model_name)  # Should fail
        raise AssertionError("Expected FileExistsError when saving without overwrite")
    except FileExistsError:
        print("Correctly caught FileExistsError when saving without overwrite")
    
    # Test overwrite
    de_matcher.save_model(test_model_name, overwrite=True)  # Should succeed
    print("Successfully overwrote existing model")
    
    # Test model loading
    print("\nTesting model loading:")
    new_matcher = AddressMatcher(language="DE")
    new_matcher.load_model(test_model_name)
    
    # Test matching with loaded model
    test_query = ("12345", "Berlin", "Hauptstraße")
    matches = new_matcher.find_matches(*test_query, k=3)
    assert len(matches) > 0, "Loaded model should return matches"
    
    # Test GPU memory usage if available
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

if __name__ == '__main__':
    test_language_support()
