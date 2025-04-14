import os
import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union

# Import utility functions 
from stats_utils import (
    generate_all_allocations, 
    find_envy_free_allocations, 
    find_pareto_frontier,
    calculate_welfare_metrics,
    find_max_welfare_allocations
)

from envy_freeness import calculate_envy, evaluate_envy, evaluate_welfare

class FairAllocationEvaluator:
    def __init__(self, base_path: str):
        """
        Initialize the evaluator with the base path where output files are stored.
        
        Args:
            base_path: Base directory containing output files
        """
        self.base_path = base_path
        
    def extract_valuation_table(self, file_path: str) -> np.ndarray:
        """
        Extract the valuation table from an output file.
        
        Args:
            file_path: Path to the output file
            
        Returns:
            numpy.ndarray: Valuation table of shape (num_agents, num_items)
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                # Look for the valuation table in the file
                # First find where the table starts
                table_start = None
                for i, line in enumerate(lines):
                    if re.match(r'^\s*\d+\s+\d+', line):
                        table_start = i
                        break
                
                if table_start is None:
                    raise ValueError(f"Could not find valuation table in {file_path}")
                
                # Find the dimensions of the table
                # The first line after the valuation table header should contain indices
                header_line = lines[table_start-1] if table_start > 0 else lines[table_start]
                items = len(header_line.strip().split()) - 1 if '   ' in header_line else 0
                
                # Count how many rows (agents) there are
                agents = 0
                for i in range(table_start, len(lines)):
                    line = lines[i].strip()
                    if not line or not line[0].isdigit():
                        break
                    agents += 1
                
                if agents == 0 or items == 0:
                    # Try to infer dimensions from the file
                    for i, line in enumerate(lines):
                        if "agents" in line and "items" in line:
                            match = re.search(r'(\d+) agents.*?(\d+) items', line)
                            if match:
                                agents = int(match.group(1))
                                items = int(match.group(2))
                                break
                
                # Read the valuation table
                valuation_table = np.zeros((agents, items))
                for i in range(agents):
                    if table_start + i < len(lines):
                        line = lines[table_start + i].strip()
                        values = line.split()
                        # The first value is the agent index, so skip it
                        if len(values) > 1:
                            for j in range(min(items, len(values) - 1)):
                                valuation_table[i, j] = int(values[j + 1])
                
                return valuation_table
                
        except Exception as e:
            print(f"Error extracting valuation table from {file_path}: {e}")
            return None
    
    def extract_llm_allocation(self, file_path: str) -> Dict[str, List[int]]:
        """
        Extract the LLM's allocation from an output file.
        
        Args:
            file_path: Path to the output file
            
        Returns:
            dict: The LLM's allocation as a dictionary where keys are agent indices
                 and values are lists of item indices
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                
                # Try to find a JSON object in the content
                json_matches = re.findall(r'({[\s\S]*?})', content)
                
                for json_str in json_matches:
                    try:
                        allocation = json.loads(json_str)
                        # Verify this is an allocation object
                        if all(isinstance(key, str) and isinstance(value, list) for key, value in allocation.items()):
                            # Convert string keys to integers if needed
                            return {int(k): v for k, v in allocation.items()}
                    except json.JSONDecodeError:
                        continue
                
                # If no JSON found, try to extract allocation from text
                lines = content.split('\n')
                allocation_dict = {}
                
                for line in lines:
                    # Look for patterns like "Agent 0: [1, 2, 3]" or "0: [1, 2, 3]"
                    match = re.search(r'(?:Agent\s*)?(\d+)\s*:\s*\[([\d\s,]*)\]', line)
                    if match:
                        agent = int(match.group(1))
                        items_str = match.group(2).strip()
                        items = [int(item) for item in items_str.split(',') if item.strip()]
                        allocation_dict[agent] = items
                
                if allocation_dict:
                    return allocation_dict
                
                raise ValueError(f"Could not find allocation in {file_path}")
                
        except Exception as e:
            print(f"Error extracting LLM allocation from {file_path}: {e}")
            return None
    
    def convert_allocation_to_matrix(self, allocation: Dict[str, List[int]], num_agents: int, num_items: int) -> np.ndarray:
        """
        Convert an allocation dictionary to an allocation matrix.
        
        Args:
            allocation: Allocation dictionary
            num_agents: Number of agents
            num_items: Number of items
            
        Returns:
            numpy.ndarray: Allocation matrix of shape (num_agents, num_items)
        """
        allocation_matrix = np.zeros((num_agents, num_items))
        
        for agent, items in allocation.items():
            agent_idx = int(agent) if isinstance(agent, str) else agent
            for item in items:
                if 0 <= item < num_items:
                    allocation_matrix[agent_idx, item] = 1
        
        return allocation_matrix
    
    def find_allocation_index(self, allocation_matrix: np.ndarray, all_allocations: np.ndarray) -> int:
        """
        Find the index of an allocation matrix in the list of all possible allocations.
        
        Args:
            allocation_matrix: Allocation matrix to find
            all_allocations: All possible allocations
            
        Returns:
            int: Index of the allocation in all_allocations, or -1 if not found
        """
        # Reshape allocation_matrix to match the shape of elements in all_allocations
        agent_allocs = np.sum(allocation_matrix, axis=1)
        num_agents, num_items = allocation_matrix.shape
        
        # Compare with each allocation in all_allocations
        for i, alloc in enumerate(all_allocations):
            if np.array_equal(alloc, allocation_matrix):
                return i
        
        return -1
    
    def calculate_utilities(self, allocations: np.ndarray, valuations: np.ndarray) -> np.ndarray:
        """
        Calculate utilities for all allocations.
        
        Args:
            allocations: All possible allocations of shape (num_allocations, num_agents, num_items)
            valuations: Valuation table of shape (num_agents, num_items)
            
        Returns:
            numpy.ndarray: Utilities of shape (num_allocations, num_agents)
        """
        num_allocations, num_agents, num_items = allocations.shape
        utilities = np.zeros((num_allocations, num_agents))
        
        # Use broadcasting to calculate utilities for all allocations at once
        for agent in range(num_agents):
            # Broadcast valuations to match allocations shape
            agent_valuations = valuations[agent].reshape(1, 1, num_items)
            # Multiply allocations by agent's valuations and sum over items
            utilities[:, agent] = np.sum(allocations * agent_valuations, axis=2)
        
        return utilities
    
    def check_ef1(self, allocation_matrix: np.ndarray, valuations: np.ndarray) -> bool:
        """
        Check if an allocation is envy-free up to one item (EF1).
        
        Args:
            allocation_matrix: Allocation matrix
            valuations: Valuation table
            
        Returns:
            bool: True if the allocation is EF1, False otherwise
        """
        num_agents, num_items = valuations.shape
        
        # For each pair of agents (i, j)
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    # Calculate i's valuation of their own bundle
                    own_items = np.where(allocation_matrix[i] == 1)[0]
                    own_value = sum(valuations[i, item] for item in own_items)
                    
                    # Calculate i's valuation of j's bundle
                    other_items = np.where(allocation_matrix[j] == 1)[0]
                    if len(other_items) == 0:
                        # If j has no items, i can't envy j
                        continue
                    
                    # Calculate value of j's bundle minus the most valuable item
                    other_values = [valuations[i, item] for item in other_items]
                    other_value_minus_one = sum(other_values) - max(other_values) if other_values else 0
                    
                    # Check if i envies j after removing the most valuable item
                    if own_value < other_value_minus_one:
                        return False
        
        return True
    
    def check_pareto_optimal(self, allocation_idx: int, is_pareto_optimal: np.ndarray) -> bool:
        """
        Check if an allocation is Pareto optimal.
        
        Args:
            allocation_idx: Index of the allocation
            is_pareto_optimal: Boolean array indicating which allocations are Pareto optimal
            
        Returns:
            bool: True if the allocation is Pareto optimal, False otherwise
        """
        if 0 <= allocation_idx < len(is_pareto_optimal):
            return is_pareto_optimal[allocation_idx]
        return False
    
    def evaluate_single_instance(self, output_file: str) -> Dict[str, Any]:
        """
        Evaluate a single problem instance.
        
        Args:
            output_file: Path to the output file
            
        Returns:
            dict: Evaluation metrics
        """
        valuation_table = self.extract_valuation_table(output_file)
        llm_allocation = self.extract_llm_allocation(output_file)
        
        if valuation_table is None or llm_allocation is None:
            print(f"Skipping {output_file} due to extraction errors")
            return None
        
        num_agents, num_items = valuation_table.shape
        
        # Generate all possible allocations
        all_allocations = generate_all_allocations(num_agents, num_items)
        
        # Calculate utilities for all allocations
        utilities = self.calculate_utilities(all_allocations, valuation_table)
        
        # Find envy-free allocations
        is_envy_free, envy_free_utilities = find_envy_free_allocations(all_allocations, utilities, valuation_table)
        num_envy_free = np.sum(is_envy_free)
        
        # Find Pareto optimal allocations among envy-free ones
        if num_envy_free > 0:
            is_pareto_optimal, pareto_utilities = find_pareto_frontier(envy_free_utilities)
            num_pareto = np.sum(is_pareto_optimal)
        else:
            is_pareto_optimal = np.zeros(0, dtype=bool)
            pareto_utilities = np.zeros((0, num_agents))
            num_pareto = 0
        
        # Calculate maximum welfare
        max_egal_idx, max_nash_idx, max_util_idx, max_egal, max_nash, max_util = find_max_welfare_allocations(utilities)
        
        # Convert LLM allocation to matrix
        llm_allocation_matrix = self.convert_allocation_to_matrix(llm_allocation, num_agents, num_items)
        
        # Find LLM allocation in all allocations
        llm_allocation_idx = self.find_allocation_index(llm_allocation_matrix, all_allocations)
        
        # Calculate LLM allocation welfare
        if llm_allocation_idx >= 0:
            llm_utility = utilities[llm_allocation_idx]
            llm_egal = np.min(llm_utility)
            llm_nash = np.prod(llm_utility + 1e-10)  # Small epsilon to avoid zero products
            llm_util = np.sum(llm_utility)
        else:
            # Calculate welfare directly if allocation isn't in the precomputed list
            llm_utility = np.zeros(num_agents)
            for agent in range(num_agents):
                agent_items = np.where(llm_allocation_matrix[agent] == 1)[0]
                llm_utility[agent] = sum(valuation_table[agent, item] for item in agent_items)
            
            llm_egal = np.min(llm_utility)
            llm_nash = np.prod(llm_utility + 1e-10)
            llm_util = np.sum(llm_utility)
        
        # Check if LLM allocation is envy-free
        is_llm_ef = (llm_allocation_idx >= 0 and is_envy_free[llm_allocation_idx]) or self.is_allocation_envy_free(llm_allocation_matrix, valuation_table)
        
        # Check if LLM allocation is EF1
        is_llm_ef1 = self.check_ef1(llm_allocation_matrix, valuation_table)
        
        # Check if LLM allocation is Pareto optimal among envy-free allocations
        is_llm_pareto = self.check_pareto_optimal(llm_allocation_idx, is_pareto_optimal) if llm_allocation_idx >= 0 else False
        
        # Return all metrics
        return {
            "file_path": output_file,
            "num_agents": num_agents,
            "num_items": num_items,
            "total_allocations": len(all_allocations),
            "num_envy_free": num_envy_free,
            "percent_envy_free": (num_envy_free / len(all_allocations)) * 100 if len(all_allocations) > 0 else 0,
            "num_pareto_optimal": num_pareto,
            "max_egalitarian_welfare": max_egal,
            "max_nash_welfare": max_nash,
            "max_utilitarian_welfare": max_util,
            "llm_egalitarian_welfare": llm_egal,
            "llm_nash_welfare": llm_nash,
            "llm_utilitarian_welfare": llm_util,
            "egalitarian_ratio": llm_egal / max_egal if max_egal > 0 else 0,
            "nash_ratio": llm_nash / max_nash if max_nash > 0 else 0,
            "utilitarian_ratio": llm_util / max_util if max_util > 0 else 0,
            "is_llm_ef": is_llm_ef,
            "is_llm_ef1": is_llm_ef1,
            "is_llm_pareto": is_llm_pareto
        }
    
    def is_allocation_envy_free(self, allocation_matrix: np.ndarray, valuations: np.ndarray) -> bool:
        """
        Check if an allocation is envy-free.
        
        Args:
            allocation_matrix: Allocation matrix
            valuations: Valuation table
            
        Returns:
            bool: True if the allocation is envy-free, False otherwise
        """
        num_agents, num_items = valuations.shape
        
        # For each pair of agents (i, j)
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    # Calculate i's valuation of their own bundle
                    own_items = np.where(allocation_matrix[i] == 1)[0]
                    own_value = sum(valuations[i, item] for item in own_items)
                    
                    # Calculate i's valuation of j's bundle
                    other_items = np.where(allocation_matrix[j] == 1)[0]
                    other_value = sum(valuations[i, item] for item in other_items)
                    
                    # Check if i envies j
                    if own_value < other_value:
                        return False
        
        return True
    
    def run_evaluation(self, output_dir: str, output_csv: str = "evaluation_results.csv"):
        """
        Run evaluation on all output files in a directory.
        
        Args:
            output_dir: Directory containing output files
            output_csv: File to save results to
        """
        results = []
        output_files = [f for f in os.listdir(output_dir) if f.startswith("output_") and f.endswith(".txt")]
        
        for file in tqdm(output_files, desc="Evaluating allocations"):
            file_path = os.path.join(output_dir, file)
            result = self.evaluate_single_instance(file_path)
            if result:
                results.append(result)
        
        # Convert results to DataFrame and save to CSV
        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(output_dir, output_csv), index=False)
            print(f"Saved evaluation results to {os.path.join(output_dir, output_csv)}")
            return df
        else:
            print("No results to save")
            return None

    def evaluate_allocations_batch(self, batch_config: List[Dict[str, str]]):
        """
        Run evaluation on multiple directories based on a configuration.
        
        Args:
            batch_config: List of dictionaries with 'path' and 'output_csv' keys
        """
        for config in batch_config['batch_config']:
            # load config as json
            

            output_dir = config['path']
            output_csv = config.get('output_csv', "evaluation_results.csv")
            
            print(f"Evaluating allocations in {output_dir}")
            self.run_evaluation(output_dir, output_csv)


def main():
    # Create evaluator
    base_path = "outputs"  # Base directory with outputs
    evaluator = FairAllocationEvaluator(base_path)
    
    # Define directories to evaluate
    batch_config = [
        {"path": "outputs/agents_2/items_3/gpt4o/zero_shot0", "output_csv": "evaluation_results.csv"},
        # Add more configurations as needed
    ]
    
    # Run evaluation
    evaluator.evaluate_allocations_batch(batch_config)
    
    # Alternatively, evaluate a single directory
    # evaluator.run_evaluation("outputs/agents_2/items_3/gpt4o/zero_shot0")


if __name__ == "__main__":
    main()