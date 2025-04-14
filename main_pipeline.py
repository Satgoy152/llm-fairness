import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import time

from fair_allocation_pipeline import FairAllocationEvaluator
from analysis_visualizer import AllocationAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fair Allocation Evaluation Pipeline')
    
    parser.add_argument('--mode', type=str, choices=['evaluate', 'analyze', 'all'], default='all',
                       help='Mode to run the pipeline in: evaluate, analyze, or all')
    
    parser.add_argument('--base_dir', type=str, default='outputs',
                       help='Base directory containing output files')
    
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv',
                       help='Name of the CSV file to save evaluation results to')
    
    parser.add_argument('--report_dir', type=str, default=None,
                       help='Directory to save the analysis report and plots')
    
    parser.add_argument('--report_name', type=str, default='evaluation_report.md',
                       help='Name of the report file')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to a configuration file for batch evaluation')
    
    parser.add_argument('--agent_range', type=str, default=None,
                       help='Range of agent numbers to evaluate (e.g., "2-4")')
    
    parser.add_argument('--item_range', type=str, default=None,
                       help='Range of item numbers to evaluate (e.g., "3-7")')
    
    parser.add_argument('--llm_models', type=str, nargs='+', default=['gpt4o'],
                       help='LLM models to evaluate (e.g., gpt4o claude-3)')
    
    parser.add_argument('--prompt_types', type=str, nargs='+', default=['zero_shot0'],
                       help='Prompt types to evaluate (e.g., zero_shot0 zero_shot1)')
    
    return parser.parse_args()


def parse_range(range_str: str) -> List[int]:
    """Parse a range string into a list of integers."""
    if range_str is None:
        return []
    
    parts = range_str.split('-')
    if len(parts) == 1:
        return [int(parts[0])]
    elif len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    else:
        raise ValueError(f"Invalid range string: {range_str}")


def generate_batch_config(args) -> List[Dict[str, str]]:
    """Generate a batch configuration from command line arguments."""
    agent_range = parse_range(args.agent_range)
    item_range = parse_range(args.item_range)
    
    batch_config = []
    
    for agents in agent_range:
        for items in item_range:
            for model in args.llm_models:
                for prompt_type in args.prompt_types:
                    path = os.path.join(args.base_dir, f"agents_{agents}", f"items_{items}", model, prompt_type)
                    if os.path.exists(path):
                        batch_config.append({
                            "path": path,
                            "output_csv": args.output_csv
                        })
    
    return batch_config


def load_config_file(config_path: str) -> List[Dict[str, str]]:
    """Load a batch configuration from a file."""
    import json
    
    with open(config_path, 'r') as file:
        return json.load(file)


def merge_results(batch_config: List[Dict[str, str]], base_dir: str) -> pd.DataFrame:
    """Merge results from multiple directories into a single DataFrame."""
    all_results = []
    
    for config in batch_config:
        path = config["path"]
        output_csv = config.get("output_csv", "evaluation_results.csv")
        
        csv_path = os.path.join(path, output_csv)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add model and prompt_type columns
            parts = path.split(os.sep)
            for i, part in enumerate(parts):
                if part.startswith("agents_"):
                    agents_part = part
                    items_part = parts[i+1] if i+1 < len(parts) else ""
                    model_part = parts[i+2] if i+2 < len(parts) else ""
                    prompt_part = parts[i+3] if i+3 < len(parts) else ""
                    
                    df["agents_config"] = agents_part
                    df["items_config"] = items_part
                    df["model"] = model_part
                    df["prompt_type"] = prompt_part
                    break
            
            all_results.append(df)
    
    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_path = os.path.join(base_dir, "merged_results.csv")
        merged_df.to_csv(merged_path, index=False)
        print(f"Merged results saved to {merged_path}")
        return merged_df
    else:
        print("No results found to merge")
        return None


def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    
    start_time = time.time()
    
    # Determine batch configuration
    if args.config:
        batch_config = load_config_file(args.config)
    else:
        batch_config = generate_batch_config(args)
    
    if not batch_config:
        print("No valid directories found for evaluation. Please check your arguments.")
        return
    
    print(f"Found {len(batch_config)} directories to process")
    
    # Run evaluation
    if args.mode in ['evaluate', 'all']:
        print("Starting evaluation...")
        evaluator = FairAllocationEvaluator(args.base_dir)

        
        evaluator.evaluate_allocations_batch(batch_config)
        print("Evaluation complete")
    
    # Merge results
    merged_results = merge_results(batch_config, args.base_dir)
    
    # Run analysis
    if args.mode in ['analyze', 'all'] and merged_results is not None:
        print("Starting analysis...")
        
        # Define report directory
        report_dir = args.report_dir if args.report_dir else args.base_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # Create analyzer
        analyzer = AllocationAnalyzer(os.path.join(args.base_dir, "merged_results.csv"))
        
        # Generate plots
        analyzer.plot_welfare_ratios(report_dir)
        analyzer.plot_envy_free_percentages(report_dir)
        analyzer.plot_llm_performance(report_dir)
        analyzer.plot_correlation_heatmap(report_dir)
        
        # Generate report
        report_path = os.path.join(report_dir, args.report_name)
        analyzer.generate_detailed_report(report_path)
        
        print(f"Analysis complete. Report saved to {report_path}")
    
    end_time = time.time()
    print(f"Pipeline completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()