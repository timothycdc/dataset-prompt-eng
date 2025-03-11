import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation results from a JSON file.
    
    Args:
        results_path: Path to the results file
        
    Returns:
        A list of evaluation results
    """
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return []


def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the evaluation results for easier analysis.
    
    Args:
        results: The evaluation results
        
    Returns:
        A pandas DataFrame with the results
    """
    # Extract relevant data from each result
    data = []
    for result in results:
        if "error" in result:
            continue
            
        row = {
            "language": result["language"],
            "preamble_name": result["preamble_name"],
            "preamble_type": result["preamble_type"],
            "input_text": result["input_text"],
            "bleu_1gram": result["bleu_scores"]["1-gram"],
            "bleu_2gram": result["bleu_scores"]["2-gram"],
            "bleu_3gram": result["bleu_scores"]["3-gram"],
            "bleu_4gram": result["bleu_scores"]["4-gram"],
            "avg_bleu": sum(result["bleu_scores"].values()) / len(result["bleu_scores"])
        }
        data.append(row)
    
    return pd.DataFrame(data)


def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics from the results DataFrame.
    
    Args:
        df: The results DataFrame
        
    Returns:
        A dictionary of summary statistics
    """
    summary = {}
    
    # Overall statistics
    summary["overall"] = {
        "avg_bleu_1gram": df["bleu_1gram"].mean(),
        "avg_bleu_2gram": df["bleu_2gram"].mean(),
        "avg_bleu_3gram": df["bleu_3gram"].mean(),
        "avg_bleu_4gram": df["bleu_4gram"].mean(),
        "avg_bleu_overall": df["avg_bleu"].mean(),
        "total_evaluations": len(df)
    }
    
    # Statistics by language
    summary["by_language"] = df.groupby("language")["avg_bleu"].mean().to_dict()
    
    # Statistics by preamble type
    summary["by_preamble_type"] = df.groupby("preamble_type")["avg_bleu"].mean().to_dict()
    
    # Statistics by preamble name
    summary["by_preamble_name"] = df.groupby("preamble_name")["avg_bleu"].mean().to_dict()
    
    # Best performing combinations
    best_combinations = df.groupby(["language", "preamble_name", "preamble_type"])["avg_bleu"].mean().reset_index()
    best_combinations = best_combinations.sort_values("avg_bleu", ascending=False)
    summary["best_combinations"] = best_combinations.head(10).to_dict(orient="records")
    
    return summary


def calculate_average_bleu_per_preamble_language(df: pd.DataFrame) -> None:
    """
    Calculate and display average BLEU scores for each preamble type/name and language.
    
    Args:
        df: The results DataFrame
    """
    # Define the numeric columns for BLEU scores
    numeric_cols = ["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]
    
    # Calculate average BLEU scores by language and preamble_type
    print("\nAverage BLEU Scores per Preamble Type and Language:")
    grouped_by_type = df.groupby(["language", "preamble_type"])[numeric_cols].mean()
    print(grouped_by_type)
    
    # Calculate average BLEU scores by language and preamble_name
    print("\nAverage BLEU Scores per Preamble Name and Language:")
    grouped_by_name = df.groupby(["language", "preamble_name"])[numeric_cols].mean()
    print(grouped_by_name)


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate plots from the results DataFrame and save them to the output directory.
    
    Args:
        df: The results DataFrame
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Plot average BLEU scores by language
    language_scores = df.groupby("language")[["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]].mean()
    language_scores.plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Average BLEU Scores by Language")
    axes[0, 0].set_ylabel("BLEU Score")
    
    # 2. Plot average BLEU scores by preamble type
    type_scores = df.groupby("preamble_type")[["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]].mean()
    type_scores.plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("Average BLEU Scores by Preamble Type")
    axes[0, 1].set_ylabel("BLEU Score")
    
    # 3. Plot average BLEU scores by preamble name
    name_scores = df.groupby("preamble_name")[["bleu_1gram", "bleu_2gram", "bleu_3gram", "bleu_4gram"]].mean()
    name_scores.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Average BLEU Scores by Preamble Name")
    axes[1, 0].set_ylabel("BLEU Score")
    axes[1, 0].tick_params(axis="x", rotation=45)
    
    # 4. Heatmap of 4-gram BLEU scores by language and preamble type
    heatmap_data = df.pivot_table(
        values="bleu_4gram", 
        index="language", 
        columns="preamble_type", 
        aggfunc="mean"
    )
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", ax=axes[1, 1])
    axes[1, 1].set_title("4-gram BLEU Score by Language and Preamble Type")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_scores_summary.png"))
    
    # Additional plots
    
    # 5. Plot average BLEU scores by language and preamble name
    plt.figure(figsize=(16, 10))
    heatmap_data_name = df.pivot_table(
        values="avg_bleu", 
        index="language", 
        columns="preamble_name", 
        aggfunc="mean"
    )
    sns.heatmap(heatmap_data_name, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Average BLEU Score by Language and Preamble Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_by_language_and_name.png"))
    
    # 6. Plot distribution of BLEU scores
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="language", y="avg_bleu", hue="preamble_type", data=df)
    plt.title("Distribution of BLEU Scores by Language and Preamble Type")
    plt.xlabel("Language")
    plt.ylabel("Average BLEU Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_distribution.png"))
    
    plt.close("all")


def analyze_results(results_path: str, output_dir: str = "../results/analysis") -> None:
    """
    Analyze the evaluation results and generate summary statistics and plots.
    
    Args:
        results_path: Path to the results file
        output_dir: Directory to save the analysis results
    """
    # Load the results
    results = load_results(results_path)
    if not results:
        print("No results to analyze. Exiting.")
        return
    
    print(f"Analyzing {len(results)} evaluation results")
    
    # Create a DataFrame from the results
    df = create_summary_dataframe(results)
    
    # Generate summary statistics
    summary = generate_summary_statistics(df)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the summary statistics
    summary_path = os.path.join(output_dir, "summary_statistics.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary statistics saved to {summary_path}")
    
    # Calculate and display average BLEU scores by preamble and language
    calculate_average_bleu_per_preamble_language(df)
    
    # Generate and save plots
    plots_dir = os.path.join(output_dir, "plots")
    plot_results(df, plots_dir)
    
    print(f"Plots saved to {plots_dir}")
    
    # Print some key statistics
    print("\nKey Statistics:")
    print(f"- Overall average BLEU score: {summary['overall']['avg_bleu_overall']:.4f}")
    print(f"- Best performing language: {max(summary['by_language'].items(), key=lambda x: x[1])[0]}")
    print(f"- Best performing preamble type: {max(summary['by_preamble_type'].items(), key=lambda x: x[1])[0]}")
    print(f"- Best performing preamble name: {max(summary['by_preamble_name'].items(), key=lambda x: x[1])[0]}")
    
    # Print top 3 best combinations
    print("\nTop 3 Best Performing Combinations:")
    for i, combo in enumerate(summary["best_combinations"][:3]):
        print(f"{i+1}. Language: {combo['language']}, Preamble: {combo['preamble_name']} ({combo['preamble_type']}), BLEU: {combo['avg_bleu']:.4f}")
    
    # Save the DataFrame for further analysis
    df_path = os.path.join(output_dir, "results_dataframe.csv")
    df.to_csv(df_path, index=False)
    print(f"Results DataFrame saved to {df_path}")


if __name__ == "__main__":
    # Get the results file path from command line arguments or use default
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Find the most recent results file
        results_dir = Path("../results")
        if results_dir.exists():
            results_files = list(results_dir.glob("*.json"))
            if results_files:
                results_path = str(max(results_files, key=os.path.getmtime))
            else:
                print("No results files found. Please specify a results file path.")
                sys.exit(1)
        else:
            print("Results directory not found. Please specify a results file path.")
            sys.exit(1)
    
    print(f"Analyzing results from: {results_path}")
    analyze_results(results_path) 