#!/usr/bin/env python3
"""
Script to run the translation evaluation pipeline.
"""

import argparse
import os
from typing import List, Optional
from pipeline.pipeline import run_evaluation_pipeline

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the translation evaluation pipeline.")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/test_dataset.json",
        help="Path to the test dataset JSON file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/evaluation_results.json",
        help="Path to save the evaluation results"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=5,
        help="Maximum number of worker threads"
    )
    
    parser.add_argument(
        "--languages", 
        type=str, 
        nargs="+",
        help="List of languages to test (e.g., Spanish French Chinese)"
    )
    
    parser.add_argument(
        "--preamble-types", 
        type=str, 
        nargs="+",
        choices=["Zero-Shot", "Few-Shot"],
        help="List of preamble types to test (Zero-Shot and/or Few-Shot)"
    )
    
    return parser.parse_args()

def main():
    """Run the evaluation pipeline with command-line arguments."""
    args = parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run the evaluation pipeline
    run_evaluation_pipeline(
        dataset_path=args.dataset,
        output_path=args.output,
        max_workers=args.workers,
        languages=args.languages,
        preamble_types=args.preamble_types
    )

if __name__ == "__main__":
    main() 