import argparse
from matcher import AddressMatcher, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
import time

def main():
    parser = argparse.ArgumentParser(description='Enhanced address matching system with hierarchical weighting')
    parser.add_argument('--reference', required=True, help='Path to reference addresses CSV file (Strassen.csv)')
    parser.add_argument('--test', help='Path to test addresses CSV file', default='test_data/test_cases.csv')
    parser.add_argument('--k', type=int, default=3, help='Number of matches to return for each query')
    parser.add_argument('--sample-size', type=int, default=None, help='Optional: Number of reference addresses to sample for testing')
    parser.add_argument('--language', choices=SUPPORTED_LANGUAGES, default=DEFAULT_LANGUAGE,
                      help=f'Language for address matching. Currently supported: {", ".join(SUPPORTED_LANGUAGES)}')
    parser.add_argument('--load-model', help='Load a specific model by name', default='latest')
    
    args = parser.parse_args()
    
    try:
        # Initialize matcher with language
        matcher = AddressMatcher(language=args.language)
        
        # Try to load existing model
        try:
            matcher.load_model(args.load_model)
            print(f"Loaded existing model: {args.load_model}")
        except FileNotFoundError:
            print("No existing model found, will train new model")
            print(f"Initializing matcher with {args.reference}...")
            matcher.initialize_database(args.reference, sample_size=args.sample_size)
        
        # Process test cases
        import pandas as pd
        test_df = pd.read_csv(args.test, delimiter=';')
        
        print("\nProcessing test cases:")
        correct_matches = 0
        total_matches = len(test_df)
        
        for _, row in test_df.iterrows():
            print(test_df)
            postal_code = str(row['nPLZ'])
            city = row['cOrtsname']
            street = row['cStrassenname']
            
            print(f"\nInput: {postal_code} {city} {street}")
            matches = matcher.find_matches(postal_code, city, street, k=args.k)
            
            # Analyze match quality
            best_score = matches[0][1] if matches else 0
            match_quality = "Excellent" if best_score > 0.9 else "Good" if best_score > 0.7 else "Poor"
            
            print("Top matches:")
            for addr, score in matches:
                print(addr)
                print(f"- {addr['nPLZ']} {addr['cOrtsname']} {addr['cStrassenname']}")
                print(f"  Score: {score:.4f}")
                print(f"  Match components:")
                print(f"    ZIP: {'✓' if addr['nPLZ'] == postal_code else '✗'}")
                print(f"    City: {'✓' if addr['cOrtsname'].lower() == city.lower() else '~' if matcher._partial_match(addr['cOrtsname'].lower(), city.lower()) else '✗'}")
                print(f"    Street: {'✓' if addr['cStrassenname'].lower() == street.lower() else '~' if matcher._partial_match(addr['cStrassenname'].lower(), street.lower()) else '✗'}")
            
            print(f"Match quality: {match_quality}")
            if best_score > 0.7:
                correct_matches += 1
        
        # Print overall metrics
        accuracy = correct_matches / total_matches
        print("\nOverall Performance Metrics:")
        print("=" * 60)
        print(f"Total test cases: {total_matches}")
        print(f"Correct matches: {correct_matches}")
        print(f"Accuracy: {accuracy:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())

    time.sleep(100)

    start = time.time()
    main()
    end = time.time()
    print(f"\nExecution time: {end - start:.2f} seconds")

# python main.py --reference test_data/addresses.csv --test test_data//test_cases.csv --k 1 --sample-size 1000