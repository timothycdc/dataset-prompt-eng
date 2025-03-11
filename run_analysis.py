#!/usr/bin/env python3
"""
Script to analyze translation evaluation results.
"""

import argparse
import os
from pipeline.analyze_results import analyze_results

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze translation evaluation results.")
    
    parser.add_argument(
        "results_file", 
        type=str,
        help="Path to the evaluation results JSON file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/analysis",
        help="Directory to save the analysis results"
    )
    
    return parser.parse_args()

def main():
    """Run the analysis with command-line arguments."""
    args = parse_args()
    
    # Ensure the results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the analysis
    analyze_results(args.results_file, args.output_dir)

if __name__ == "__main__":
    main() 