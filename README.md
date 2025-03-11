# Translation Evaluation Pipeline

> :warning: This pipeline is currently in development and has not been tested yet.


## Features

- Multithreaded evaluation of translation prompts
- Multithreaded BLEU score calculation for faster processing
- Support for multiple languages and two preamble types (few-shot and zero-shot)
  - Few-shot preambles include examples of translations to guide the model
  - Zero-shot preambles do not include examples 


## Directory Structure

```
.
├── data/
│   ├── language_data.json       # Language-specific examples
│   └── test_dataset.json        # Test dataset for evaluation
├── data_preamble/
│   ├── few-shot/                # Few-shot preamble templates
│   └── zero-shot/               # Zero-shot preamble templates
├── pipeline/
│   ├── preamble_generator.py    # Preamble generation utilities
│   ├── pipeline.py              # Main evaluation pipeline
│   ├── clean_utils.py           # Text cleaning utilities
│   ├── eval_utils.py            # Evaluation metrics
│   └── analyze_results.py       # Results analysis
├── results/                     # Evaluation results
├── run_evaluation.py            # Script to run the evaluation pipeline
├── run_analysis.py              # Script to analyse existing results
└── requirements.txt             # Project dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy `.env.template`, add your Cohere API key, and rename the copied file to `.env`.
## Usage

### Running the Evaluation Pipeline

```bash
python run_evaluation.py --workers 10 --languages Spanish French Chinese --preamble-types Zero-Shot
```

Or from Python code:

```python
from pipeline.pipeline import run_evaluation_pipeline

run_evaluation_pipeline(
    dataset_path="path/to/your/dataset.json",
    output_path="path/to/save/results.json",
    max_workers=10,  # Number of parallel workers
    languages=["Spanish", "French", "Chinese"],  # Specific languages to test
    preamble_types=["Zero-Shot"]  # Specific preamble types to test
)
```

### Analyzing Results

You can analyze existing results using the command-line script:

```bash
python run_analysis.py results/evaluation_results.json
```

Or from Python code:

```python
from pipeline.analyze_results import analyze_results

analyze_results("path/to/results.json")
```

This will generate summary statistics and visualizations in the `results/analysis` directory.

## Test Dataset Format

The test dataset should be a JSON file with the following structure:

```json
[
  {
    "input_text": "buy blue cotton shirt for men",
    "translation_spanish": "comprar camisa de algodón azul para hombres",
    "translation_french": "acheter une chemise en coton bleu pour hommes",
    ...
  },
  ...
]
```

## Preamble Templates

Preamble templates are stored as `.txt` files in the `data_preamble` directory, organized by type (few-shot or zero-shot). Each template can include placeholders like `{language}` and `{example1}` that will be replaced with language-specific data.
A preamble could look like this:

```python
Translate the input text into {language}.
```


## Results Analysis

The analysis script generates:

1. Summary statistics in JSON format
2. Visualizations of BLEU scores by language, preamble type, and preamble name
3. Heatmaps showing performance across languages and preamble types
4. Distribution plots of BLEU scores
5. A CSV file with all evaluation results for further analysis

### Analysis Features

- **Average BLEU Scores by Language and Preamble Type**: Shows how different preamble types perform across languages
- **Average BLEU Scores by Language and Preamble Name**: Shows how specific preambles perform across languages
- **BLEU Score Distribution**: Shows the distribution of BLEU scores for different combinations
- **Top Performing Combinations**: Identifies the best language-preamble combinations

