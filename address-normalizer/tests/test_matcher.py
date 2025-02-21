import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.matcher import AddressMatcher
from src.config import TrainingConfig
import torch

def test_specific_matches():
    """Test specific address matching cases."""
    print("\nTesting specific address matches...")
    
    # Initialize matcher with sample size
    matcher = AddressMatcher()
    matcher.initialize_database('test_data/DE/combined_test.csv')  # Use combined dataset with our test cases
    
    # Test cases from requirements
    test_cases = [
        ('75346', 'Wiersheim', 'Ziegelhuete', '75446', 'Wiernsheim', 'Ziegelhütte'),
        ('56294', 'Gierschnach', 'Burstraße', '56294', 'Gierschnach', 'Burstr.'),
        ('54637', 'Schlaid', 'Hauptstrasse', '54636', 'Schleid', 'Hauptstr.'),
        ('54637', 'Wierschem', 'A.d. Spreeg', '56294', 'Wierschem', 'Auf der Spreeg'),
        ('54636', 'Esslingen', 'Am Kalkofen', '54636', 'Eßlingen', 'Am Kalkofen'),
        ('67894', 'Martinshoehe', 'Etzenbachermuele', '66894', 'Martinshöhe', 'Etzenbachermühle')
    ]
    
    print("\nRunning test cases...")
    for input_postal, input_city, input_street, exp_postal, exp_city, exp_street in test_cases:
        print(f"\nTesting: {input_postal} {input_city} {input_street}")
        print(f"Expected: {exp_postal} {exp_city} {exp_street}")
        
        matches = matcher.find_matches(input_postal, input_city, input_street, k=5)
        assert len(matches) > 0, f"No matches found for {input_postal} {input_city} {input_street}"
        
        top_match, score = matches[0]
        print(f"Top match: {top_match['full_address']}, Score: {score:.4f}")
        
        # Verify match quality
        assert top_match['nPLZ'] == exp_postal, \
            f"Incorrect postal code. Got {top_match['nPLZ']}, expected {exp_postal}"
        
        city_sim = matcher._partial_match(top_match['cOrtsname'], exp_city)
        assert city_sim > 0.9, \
            f"City similarity too low ({city_sim:.2f}). Got {top_match['cOrtsname']}, expected {exp_city}"
        
        street_sim = matcher._partial_match(top_match['cStrassenname'], exp_street)
        assert street_sim > 0.9, \
            f"Street similarity too low ({street_sim:.2f}). Got {top_match['cStrassenname']}, expected {exp_street}"
        
        print("✓ Test passed")

def test_matcher_performance():
    """Test matcher initialization and training."""
    print("\nTesting matcher performance...")
    
    # Initialize matcher
    matcher = AddressMatcher()
    
    # Test training with small dataset
    config = TrainingConfig(num_epochs=2)  # Small number of epochs for testing
    metrics = matcher.train('test_data/DE/sample.csv', config)
    
    print("\nTraining metrics:")
    print(f"Final loss: {metrics['train_loss'][-1]:.4f}")
    
    # Test basic match finding
    matches = matcher.find_matches('12345', 'Berlin', 'Hauptstrasse', k=3)
    assert len(matches) > 0, "No matches found for basic test case"
    
    print("\nBasic match test passed")

if __name__ == '__main__':
    test_specific_matches()
    test_matcher_performance()
