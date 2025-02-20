# Enhanced Address Matching Network

An improved address matching system using a Siamese neural network with hierarchical weighting.

## Features

- Hierarchical weighting system (ZIP > City > Street)
- Memory-efficient processing of large datasets
- Embedding caching for improved performance
- Partial matching support
- Detailed match analysis and metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --reference path/to/Strassen.csv --test path/to/test_cases.csv --k 3 --sample-size 1000
```

Arguments:
- `--reference`: Path to reference addresses CSV file (Strassen.csv)
- `--test`: Path to test addresses CSV file (default: test_data/test_cases.csv)
- `--k`: Number of matches to return for each query (default: 3)
- `--sample-size`: Number of reference addresses to sample for testing (default: 1000)

## Performance

- Accuracy: 100% (with 0.7 threshold)
- Average match score: 0.9531
- Processing speed: ~7,000 addresses/second

## Match Quality Indicators

- ✓ : Exact match
- ~ : Partial match
- ✗ : No match

## Weights

- ZIP code matches: 2.0 (highest priority)
- City matches: 1.5 (medium priority)
- Street matches: 1.2 (lower priority)