import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.matcher import AddressMatcher

def test_address_matching():
    """Test address matching functionality with various test cases."""
    print('\nTesting address matching...\n')
    
    # Initialize matcher with test data
    matcher = AddressMatcher(language="DE")
    matcher.initialize_database('test_data/DE/test_cases.csv')
    
    # Test cases
    test_pairs = [
        # Similar city names
        ('75446', 'Wiernshem', 'Lerchenweg'),
        ('75446', 'Wiernsheim', 'Lerchenveg'),
        
        # International address
        ('10001', 'NYC', 'Fifth Avenue'),
        
        # German addresses with variations
        ('80331', 'Muenchen', 'Marienplatz'),
        ('80331', 'München', 'Marienplatz')
    ]
    
    for postal_code, city, street in test_pairs:
        print(f'Query: {postal_code} {city} {street}')
        matches = matcher.find_matches(postal_code, city, street, k=3)
        print('Matches:')
        for addr, score in matches:
            print(f'  - {addr["full_address"]} (score: {score:.4f})')
        print()

if __name__ == '__main__':
    test_address_matching()
