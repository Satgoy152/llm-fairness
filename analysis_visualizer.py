import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union


class AllocationAnalyzer:
    def __init__(self, results_file: str):
        """
        Initialize the analyzer with the path to the results CSV file.
        
        Args:
            results_file: Path to the results CSV file
        """
        self.results_file = results_file
        self.results_df = pd.read_csv(results_file)
        
    def generate_summary_stats(self) -> pd.DataFrame:
        """
        Generate summary statistics for the results.
        
        Returns:
            pandas.DataFrame: Summary statistics
        """
        # Group by number of agents and items
        grouped = self.results_df.groupby(['num_agents', 'num_items'])
        
        # Calculate summary statistics
        summary = grouped.agg({
            'num_envy_free': ['mean', 'min', 'max'],
            'percent_envy_free': ['mean', 'min', 'max'],
            'num_pareto_optimal': ['mean', 'min', 'max'],
            'llm_egalitarian_welfare': ['mean', 'min', 'max'],
            'llm_nash_welfare': ['mean', 'min', 'max'],
            'llm_utilitarian_welfare': ['mean', 'min', 'max'],
            'egalitarian_ratio': ['mean', 'min', 'max'],
            'nash_ratio': ['mean', 'min', 'max'],
            'utilitarian_ratio': ['mean', 'min', 'max'],
            'is_llm_ef': ['mean', 'sum'],
            'is_llm_ef1': ['mean', 'sum'],
            'is_llm_pareto': ['mean', 'sum'],
            'total_allocations': ['first']  # Just take the first value as this is constant for each (agents, items) pair
        })
        
        return summary
    
    def plot_welfare_ratios(self, save_dir: Optional[str] = None):
        """
        Plot welfare ratios for different combinations of agents and items.
        
        Args:
            save_dir: Directory to save plots to (optional)
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Group by number of agents and items
        groups = self.results_df.groupby(['num_agents', 'num_items'])
        
        # We'll plot mean welfare ratios for each group
        agents_items = []
        egal_ratios = []
        nash_ratios = []
        util_ratios = []
        
        for name, group in groups:
            agents_items.append(f"{name[0]}a-{name[1]}i")
            egal_ratios.append(group['egalitarian_ratio'].mean())
            nash_ratios.append(group['nash_ratio'].mean())
            util_ratios.append(group['utilitarian_ratio'].mean())
        
        # Set up barplot
        x = np.arange(len(agents_items))
        width = 0.25
        
        # Create bars
        plt.bar(x - width, egal_ratios, width, label='Egalitarian')
        plt.bar(x, nash_ratios, width, label='Nash')
        plt.bar(x + width, util_ratios, width, label='Utilitarian')
        
        # Add labels and title
        plt.xlabel('Number of Agents and Items')
        plt.ylabel('Mean Welfare Ratio')
        plt.title('LLM Welfare Ratios by Problem Size')
        plt.xticks(x, agents_items, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout and save if requested
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'welfare_ratios.png'), dpi=300)
        
        plt.show()
    
    def plot_envy_free_percentages(self, save_dir: Optional[str] = None):
        """
        Plot the percentage of allocations that are envy-free for different problem sizes.
        
        Args:
            save_dir: Directory to save plots to (optional)
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Group by number of agents and items
        groups = self.results_df.groupby(['num_agents', 'num_items'])
        
        # We'll plot mean percentage of envy-free allocations for each group
        agents_items = []
        ef_percentages = []
        
        for name, group in groups:
            agents_items.append(f"{name[0]}a-{name[1]}i")
            ef_percentages.append(group['percent_envy_free'].mean())
        
        # Create bar plot
        plt.bar(agents_items, ef_percentages, color='skyblue')
        
        # Add labels and title
        plt.xlabel('Number of Agents and Items')
        plt.ylabel('Percentage of Envy-Free Allocations')
        plt.title('Percentage of Envy-Free Allocations by Problem Size')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout and save if requested
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'envy_free_percentages.png'), dpi=300)
        
        plt.show()
    
    def plot_llm_performance(self, save_dir: Optional[str] = None):
        """
        Plot various metrics of LLM performance.
        
        Args:
            save_dir: Directory to save plots to (optional)
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Group by number of agents and items
        groups = self.results_df.groupby(['num_agents', 'num_items'])
        
        # We'll plot mean percentages for various metrics
        agents_items = []
        ef_percentages = []
        ef1_percentages = []
        pareto_percentages = []
        
        for name, group in groups:
            agents_items.append(f"{name[0]}a-{name[1]}i")
            ef_percentages.append(group['is_llm_ef'].mean() * 100)
            ef1_percentages.append(group['is_llm_ef1'].mean() * 100)
            pareto_percentages.append(group['is_llm_pareto'].mean() * 100)
        
        # Set up bar plot
        x = np.arange(len(agents_items))
        width = 0.25
        
        # Create bars
        plt.bar(x - width, ef_percentages, width, label='Envy-Free')
        plt.bar(x, ef1_percentages, width, label='EF1')
        plt.bar(x + width, pareto_percentages, width, label='Pareto Optimal')
        
        # Add labels and title
        plt.xlabel('Number of Agents and Items')
        plt.ylabel('Percentage (%)')
        plt.title('LLM Allocation Properties by Problem Size')
        plt.xticks(x, agents_items, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout and save if requested
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'llm_performance.png'), dpi=300)
        
        plt.show()
    
    def plot_correlation_heatmap(self, save_dir: Optional[str] = None):
        """
        Plot a correlation heatmap of the various metrics.
        
        Args:
            save_dir: Directory to save plots to (optional)
        """
        # Select numeric columns for correlation
        numeric_cols = [
            'num_agents', 'num_items', 'total_allocations', 'num_envy_free',
            'percent_envy_free', 'num_pareto_optimal', 'llm_egalitarian_welfare',
            'llm_nash_welfare', 'llm_utilitarian_welfare', 'egalitarian_ratio',
            'nash_ratio', 'utilitarian_ratio', 'is_llm_ef', 'is_llm_ef1', 'is_llm_pareto'
        ]
        
        # Create correlation matrix
        correlation = self.results_df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        
        # Add title
        plt.title('Correlation Heatmap of Fair Allocation Metrics')
        
        # Adjust layout and save if requested
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=300)
        
        plt.show()
    
    def generate_detailed_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a detailed report of the results.
        
        Args:
            save_path: Path to save the report to (optional)
            
        Returns:
            str: The report text
        """
        # Calculate summary statistics
        summary = self.generate_summary_stats()
        
        # Start building the report
        report = "# Fair Allocation Evaluation Report\n\n"
        
        # Add overview
        report += "## Overview\n\n"
        report += f"Total problems evaluated: {len(self.results_df)}\n"
        report += f"Problem sizes (agents-items): {sorted(list(set(zip(self.results_df['num_agents'], self.results_df['num_items']))))}\n\n"
        
        # Add summary of LLM performance
        report += "## LLM Performance Summary\n\n"
        report += f"- Envy-Free Allocations: {self.results_df['is_llm_ef'].mean() * 100:.2f}%\n"
        report += f"- EF1 Allocations: {self.results_df['is_llm_ef1'].mean() * 100:.2f}%\n"
        report += f"- Pareto Optimal Allocations: {self.results_df['is_llm_pareto'].mean() * 100:.2f}%\n\n"
        
        # Add welfare ratios
        report += "## Welfare Ratios\n\n"
        report += f"- Mean Egalitarian Welfare Ratio: {self.results_df['egalitarian_ratio'].mean():.4f}\n"
        report += f"- Mean Nash Welfare Ratio: {self.results_df['nash_ratio'].mean():.4f}\n"
        report += f"- Mean Utilitarian Welfare Ratio: {self.results_df['utilitarian_ratio'].mean():.4f}\n\n"
        
        # Add breakdown by problem size
        report += "## Performance by Problem Size\n\n"
        
        # Find unique problem sizes
        problem_sizes = sorted(list(set(zip(self.results_df['num_agents'], self.results_df['num_items']))))
        
        for agents, items in problem_sizes:
            subset = self.results_df[(self.results_df['num_agents'] == agents) & (self.results_df['num_items'] == items)]
            
            report += f"### {agents} Agents, {items} Items\n\n"
            report += f"- Total allocations possible: {subset['total_allocations'].iloc[0]}\n"
            report += f"- Average envy-free allocations: {subset['num_envy_free'].mean():.2f} ({subset['percent_envy_free'].mean():.2f}%)\n"
            report += f"- Average Pareto optimal allocations: {subset['num_pareto_optimal'].mean():.2f}\n"
            report += f"- LLM produces envy-free allocation: {subset['is_llm_ef'].mean() * 100:.2f}%\n"
            report += f"- LLM produces EF1 allocation: {subset['is_llm_ef1'].mean() * 100:.2f}%\n"
            report += f"- LLM produces Pareto optimal allocation: {subset['is_llm_pareto'].mean() * 100:.2f}%\n"
            report += f"- Egalitarian welfare ratio: {subset['egalitarian_ratio'].mean():.4f}\n"
            report += f"- Nash welfare ratio: {subset['nash_ratio'].mean():.4f}\n"
            report += f"- Utilitarian welfare ratio: {subset['utilitarian_ratio'].mean():.4f}\n\n"
        
        # Add conclusions
        report += "## Conclusions\n\n"
        report += "Based on the analysis, we can draw the following conclusions:\n\n"
        
        # Egalitarian welfare performance
        egal_ratio = self.results_df['egalitarian_ratio'].mean()
        if egal_ratio > 0.9:
            report += "- The LLM performs excellently at optimizing egalitarian welfare.\n"
        elif egal_ratio > 0.7:
            report += "- The LLM performs well at optimizing egalitarian welfare.\n"
        else:
            report += "- The LLM could improve at optimizing egalitarian welfare.\n"
        
        # Nash welfare performance
        nash_ratio = self.results_df['nash_ratio'].mean()
        if nash_ratio > 0.9:
            report += "- The LLM performs excellently at optimizing Nash welfare.\n"
        elif nash_ratio > 0.7:
            report += "- The LLM performs well at optimizing Nash welfare.\n"
        else:
            report += "- The LLM could improve at optimizing Nash welfare.\n"
        
        # Utilitarian welfare performance
        util_ratio = self.results_df['utilitarian_ratio'].mean()
        if util_ratio > 0.9:
            report += "- The LLM performs excellently at optimizing utilitarian welfare.\n"
        elif util_ratio > 0.7:
            report += "- The LLM performs well at optimizing utilitarian welfare.\n"
        else:
            report += "- The LLM could improve at optimizing utilitarian welfare.\n"
        
        # Envy-free performance
        ef_rate = self.results_df['is_llm_ef'].mean()
        if ef_rate > 0.9:
            report += "- The LLM is excellent at finding envy-free allocations.\n"
        elif ef_rate > 0.7:
            report += "- The LLM is good at finding envy-free allocations.\n"
        else:
            report += "- The LLM needs improvement in finding envy-free allocations.\n"
        
        # Save the report if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Saved report to {save_path}")
        
        return report


def main():
    # Create analyzer
    results_file = "outputs/agents_2/items_3/gpt4o/zero_shot0/evaluation_results.csv"
    analyzer = AllocationAnalyzer(results_file)
    
    # Generate plots
    save_dir = os.path.dirname(results_file)
    analyzer.plot_welfare_ratios(save_dir)
    analyzer.plot_envy_free_percentages(save_dir)
    analyzer.plot_llm_performance(save_dir)
    analyzer.plot_correlation_heatmap(save_dir)
    
    # Generate report
    report_path = os.path.join(save_dir, 'evaluation_report.md')
    analyzer.generate_detailed_report(report_path)


if __name__ == "__main__":
    main()