# DIRTAR: Discovery of Inference Rules from Text for Action Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modernized implementation of the DIRT algorithm (Lin and Pantel, 2001) with modifications for action recognition from natural language text.

## Overview

DIRTAR implements a modified version of the DIRT (Discovery of Inference Rules from Text) algorithm specifically designed for action recognition. The system processes movie scripts and other narrative text to discover semantic relationships and inference rules for action classification.

### Key Features

- **Modified DIRT Algorithm**: Enhanced with lemmatization, constituency parsing, slot similarity, slot types, hypernyms, and semantic discrimination
- **Action Recognition**: Specialized for recognizing and classifying actions in narrative text
- **Movie Script Processing**: Optimized for processing movie script corpora
- **Semantic Parsing**: Frame-net style rules for discriminating candidate nouns from slots
- **Modern Python**: Updated for Python 3.8+ with modern dependencies

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DIRTAR
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download required NLTK data:**
```bash
python -c "import nltk; nltk.download('wordnet')"
```

5. **Install the package in development mode:**
```bash
pip install -e .
```

## Project Structure

```
DIRTAR/
├── src/dirtar/                 # Main package source code
│   ├── __init__.py            # Package initialization
│   ├── dirtar.py              # Core DIRT algorithm implementation
│   ├── sentence_parser.py     # Sentence and clause parsing
│   ├── semantic_parser.py     # Semantic parsing and frame-net rules
│   ├── sentence_splitter.py   # Text preprocessing utilities
│   ├── moviescript_crawler.py # Movie corpus collection
│   ├── assign_labels_*.py     # Label assignment modules
│   ├── score_labels_*.py      # Evaluation modules
│   └── run_dirtar_tests.py    # Test runner
├── data/                      # Data directories
│   ├── experimental_labels/   # Experimental condition outputs
│   ├── scored_labels/         # Evaluation results
│   └── redo_labels_420/       # Additional label data
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
└── README.md                  # This file
```

## Core Components

### 1. DIRT Algorithm (`dirtar.py`)
The main implementation of the modified DIRT algorithm with experimental conditions:
- Lemma-based processing
- Constituency parse integration
- Slot similarity calculations
- Semantic type discrimination
- Hypernym-based generalization

### 2. Sentence Processing (`sentence_parser.py`)
- Parses movie scripts into sentences and clauses
- Uses constituency parsing for clause extraction
- Outputs structured clause triples

### 3. Semantic Parser (`semantic_parser.py`)
- Hand-written frame-net style rules
- Discriminates candidate nouns from slots
- Supports experimental semantic conditions

### 4. Data Processing Pipeline
- **Movie Corpus Collection**: `moviescript_crawler.py`
- **Sentence Splitting**: `sentence_splitter.py`
- **Label Assignment**: `assign_labels_*.py`
- **Evaluation**: `score_labels_*.py`

## Usage

### Basic Usage

```python
from dirtar import dirtar

# Load and process corpus
database = dirtar.readCorpus('movie_clauses.txt')

# Run DIRT algorithm with experimental conditions
results = dirtar.run_experiments(database)
```

### Running Experiments

```python
# Process movie scripts
python src/dirtar/moviescript_crawler.py

# Parse sentences into clauses
python src/dirtar/sentence_parser.py

# Run DIRT algorithm
python src/dirtar/dirtar.py

# Assign labels for evaluation
python src/dirtar/assign_labels_moviedirt.py

# Score results
python src/dirtar/score_labels_dirtar.py
```

## Data Files

### Input Data
- **IE_sent_key.txt**: Test sentences from DUEL corpus with action class labels
- **movie_combo.txt**: Combined movie script corpus (not included due to size)
- **movie_clauses.txt**: Preprocessed clause triples with parse annotations

### Output Data
- **experimental_labels/**: Text files for each experimental condition
- **scored_labels/**: F-score evaluations per experimental condition
- **dirtar_database_*.pkl**: Serialized DIRT databases

## Experimental Conditions

The system supports multiple experimental conditions:
1. **Baseline DIRT**: Standard algorithm
2. **Lemma Integration**: Using lemmatized forms
3. **Slot Similarity**: Enhanced slot matching
4. **Semantic Types**: Type-based discrimination
5. **Hypernym Generalization**: WordNet-based generalization

## Evaluation

The system evaluates action recognition performance using:
- **F-score calculation** for each action class
- **Overall performance** across all experimental conditions
- **Per-action analysis** for detailed evaluation

Results are saved in the `scored_labels/` directory with detailed breakdowns.

## Dependencies

- **nltk>=3.8**: Natural language processing
- **pycorenlp>=0.3.0**: Stanford CoreNLP integration
- **setuptools>=65.0**: Package management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{winer2017dirtar,
  title={DIRTAR: Discovery of Inference Rules from Text for Action Recognition},
  author={Winer, David},
  year={2017},
  note={Modernized implementation 2024}
}
```

## Original Implementation

Based on the DIRT algorithm:
- Lin, D., & Pantel, P. (2001). DIRT - Discovery of Inference Rules from Text. ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

## Contact

For questions or issues, please contact David Winer 

## Changelog

### Version 2.0.0 (2024)
- Modernized for Python 3.8+
- Reorganized project structure
- Updated dependencies
- Added proper packaging
- Enhanced documentation
- Fixed compatibility issues

### Version 1.0.0 (2017)
- Original implementation
- Core DIRT algorithm
- Movie script processing
- Action recognition evaluation
