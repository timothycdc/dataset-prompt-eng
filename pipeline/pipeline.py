import cohere as co
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from preamble_generator import Preamble, LANGUAGES, ALL_PREAMBLES
from eval_utils import calculate_bleu_nltk
from clean_utils import normalize_text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
            
# Function to generate translation using Cohere chat API
def generate_translation(input_text: str, preamble: str) -> str:
    """
    Generate a translation using the Cohere chat API.
    """
    try:
        # Creating prompt by applying the preamble
        response = co.chat(
            preamble=preamble,
            message=input_text
        )
        generated_translation = response.text.strip()

        return generated_translation
    except Exception as e:
        print(f"Error generating translation: {e}")
        return ""


def test_single_translation(
    input_text: str, 
    gold_translation: str, 
    language: str, 
    preamble: Preamble
) -> Dict[str, Any]:
    """
    Test a single translation with a specific preamble and language.
    
    Args:
        input_text: The text to translate
        gold_translation: The expected translation
        language: The target language
        preamble: The preamble to use
        
    Returns:
        A dictionary containing the test results
    """
    try:
        # Format the preamble with the language
        formatted_preamble = preamble.format(language)
        
        # Generate translation
        generated_translation = generate_translation(input_text, formatted_preamble)
        
        # Calculate BLEU scores using multithreading
        bleu_scores = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit BLEU calculation tasks for each n-gram
            futures = {
                executor.submit(calculate_bleu_nltk, gold_translation, generated_translation, language, n_gram): n_gram
                for n_gram in range(1, 5)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                n_gram = futures[future]
                try:
                    score = future.result()
                    bleu_scores[f"{n_gram}-gram"] = score
                except Exception as e:
                    print(f"Error calculating BLEU score for {n_gram}-gram: {e}")
                    bleu_scores[f"{n_gram}-gram"] = 0.0
        
        # Create result dictionary
        result = {
            "input_text": input_text,
            "gold_translation": gold_translation,
            "generated_translation": generated_translation,
            "bleu_scores": bleu_scores,
            "preamble_name": preamble.preamble_name,
            "preamble_type": preamble.preamble_type,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    except Exception as e:
        print(f"Error in test_single_translation: {e}")
        return {
            "error": str(e),
            "input_text": input_text,
            "preamble_name": preamble.preamble_name,
            "preamble_type": preamble.preamble_type,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }


def load_test_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load the test dataset from a JSON file.
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the evaluation results to a JSON file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def evaluate_prompts_multithreaded(
    dataset: List[Dict[str, Any]], 
    preambles: Optional[List[Preamble]] = None,
    languages: Optional[List[str]] = None,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Evaluate prompts using multithreading for parallel processing.
    
    Args:
        dataset: The test dataset
        preambles: List of preambles to test (defaults to ALL_PREAMBLES)
        languages: List of languages to test (defaults to LANGUAGES)
        max_workers: Maximum number of worker threads
        
    Returns:
        A list of evaluation results
    """
    results = []
    
    # Use default values if not provided
    if preambles is None:
        preambles = ALL_PREAMBLES
    
    if languages is None:
        languages = list(LANGUAGES)
    
    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Create tasks for all combinations of dataset entries, preambles, and languages
        for entry in dataset:
            input_text = entry.get('input_text', '')
            
            for language in languages:
                # Get the gold translation for this language
                gold_translation = entry.get(f'translation_{language.lower()}', '')
                
                # Skip if no gold translation is available
                if not gold_translation:
                    continue
                
                for preamble in preambles:
                    futures.append(
                        executor.submit(
                            test_single_translation,
                            input_text,
                            gold_translation,
                            language,
                            preamble
                        )
                    )
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                print(f"Processed {i+1}/{len(futures)} tasks", end='\r')
                
            except Exception as e:
                print(f"\nError processing result: {e}")
    
    print(f"\nCompleted {len(results)} evaluations")
    return results


def calculate_average_bleu_per_preamble_language(results: List[Dict[str, Any]]) -> None:
    """
    Calculate and display save/average BLEU scores for each preamble type and language.
    """
    try:
        if not results:
            print("No results available for BLEU score computation.")
            return

        # Create a DataFrame from the results
        df = pd.DataFrame(results)

        # Check if necessary columns are present
        if 'preamble_type' not in df.columns or 'bleu_scores' not in df.columns or 'language' not in df.columns:
            print("Results data is missing necessary columns for BLEU score computation.")
            return

        # Normalize the bleu_scores column (convert from dict to columns)
        bleu_scores_df = pd.json_normalize(df['bleu_scores'])
        df = pd.concat([df.drop(columns=['bleu_scores']), bleu_scores_df], axis=1)
        
        # Define the numeric columns for BLEU scores
        numeric_cols = ['1-gram', '2-gram', '3-gram', '4-gram']
        
        # Calculate average BLEU scores by language and preamble_type
        print("\nAverage BLEU Scores per Preamble Type and Language:")
        grouped_by_type = df.groupby(['language', 'preamble_type'])[numeric_cols].mean()
        print(grouped_by_type)
        
        # Calculate average BLEU scores by language and preamble_name
        print("\nAverage BLEU Scores per Preamble Name and Language:")
        grouped_by_name = df.groupby(['language', 'preamble_name'])[numeric_cols].mean()
        print(grouped_by_name)
        
        # Plot the results
        try:
            # Set the style
            sns.set(style="whitegrid")
            
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # 1. Plot average BLEU scores by language
            language_scores = df.groupby('language')[numeric_cols].mean()
            language_scores.plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Average BLEU Scores by Language')
            axes[0, 0].set_ylabel('BLEU Score')
            
            # 2. Plot average BLEU scores by preamble type
            type_scores = df.groupby('preamble_type')[numeric_cols].mean()
            type_scores.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Average BLEU Scores by Preamble Type')
            axes[0, 1].set_ylabel('BLEU Score')
            
            # 3. Plot average BLEU scores by preamble name
            name_scores = df.groupby('preamble_name')[numeric_cols].mean()
            name_scores.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Average BLEU Scores by Preamble Name')
            axes[1, 0].set_ylabel('BLEU Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Heatmap of 4-gram BLEU scores by language and preamble type
            heatmap_data = df.pivot_table(
                values='4-gram', 
                index='language', 
                columns='preamble_type', 
                aggfunc='mean'
            )
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", ax=axes[1, 1])
            axes[1, 1].set_title('4-gram BLEU Score by Language and Preamble Type')
            
            plt.tight_layout()
            
            # Save the figure
            os.makedirs('../results/plots', exist_ok=True)
            plt.savefig('../results/plots/bleu_scores_summary.png')
            print("\nPlots saved to ../results/plots/bleu_scores_summary.png")
            
        except Exception as e:
            print(f"Error plotting results: {e}")
        
    except ImportError:
        print("pandas is required for BLEU score analysis. Please install it with 'pip install pandas'.")
    except Exception as e:
        print(f"Error calculating average BLEU scores: {e}")


def run_evaluation_pipeline(
    dataset_path: str = "../data/test_dataset.json",
    output_path: str = "../results/evaluation_results.json",
    max_workers: int = 5,
    languages: Optional[List[str]] = None,
    preamble_types: Optional[List[str]] = None
) -> None:
    """
    Run the complete evaluation pipeline.
    
    Args:
        dataset_path: Path to the test dataset
        output_path: Path to save the results
        max_workers: Maximum number of worker threads
        languages: List of languages to test (defaults to all languages)
        preamble_types: List of preamble types to test (defaults to all types)
    """
    print(f"Starting evaluation pipeline with {max_workers} workers")
    
    # Load the test dataset
    dataset = load_test_dataset(dataset_path)
    if not dataset:
        print("No dataset entries found. Exiting.")
        return
    
    print(f"Loaded {len(dataset)} test entries")
    
    # Filter preambles by type if specified
    preambles = ALL_PREAMBLES
    if preamble_types:
        preambles = [p for p in ALL_PREAMBLES if p.preamble_type in preamble_types]
    
    # Run the evaluation
    results = evaluate_prompts_multithreaded(
        dataset=dataset,
        preambles=preambles,
        languages=languages,
        max_workers=max_workers
    )
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path.endswith('.json'):
        output_path = output_path.replace('.json', f'_{timestamp}.json')
    else:
        output_path = f"{output_path}_{timestamp}.json"
    
    save_results(results, output_path)
    
    # Calculate and display average BLEU scores
    calculate_average_bleu_per_preamble_language(results)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"- Total evaluations: {len(results)}")
    print(f"- Languages tested: {languages or list(LANGUAGES)}")
    print(f"- Preamble types tested: {preamble_types or ['Zero-Shot', 'Few-Shot']}")
    print(f"- Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    run_evaluation_pipeline(
        max_workers=10,
        languages=["Spanish", "French", "Chinese"], 
        preamble_types=["Zero-Shot"] 
    )
            
            