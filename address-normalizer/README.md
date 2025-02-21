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

### Training and Testing
```bash
# Train a new model and save it
python main.py --reference path/to/Strassen.csv --save-model german_v1

# Train and test in one go
python main.py --reference path/to/Strassen.csv --test path/to/test_cases.csv --save-model german_v1

# Load existing model and test
python main.py --load-model german_v1 --test path/to/test_cases.csv
```

Arguments:
- `--reference`: Path to reference addresses CSV file for training
- `--test`: Path to test addresses CSV file
- `--save-model`: Save trained model with given name
- `--load-model`: Load existing model with given name
- `--overwrite`: Overwrite existing model if it exists
- `--batch-size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 50)
- `--workers`: Number of worker processes (default: 4)
- `--device`: Device to use (cuda/cpu)
- `--sample-size`: Number of reference addresses to sample for training (default: 10000)

### Model Storage
Models are saved in language-specific directories under `models/`. For example:
- German models: `models/DE/`
- English models: `models/EN/`

Each model consists of three files:
- `{model_name}.pt`: Neural network weights and configuration
- `{model_name}_cache.pkl`: Embedding cache for faster inference
- `{model_name}_reference.pkl`: Reference address data

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
