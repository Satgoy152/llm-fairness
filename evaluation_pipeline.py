import numpy as np
import pandas as pd
import json
import glob
import os
from itertools import product
from tqdm import tqdm

def generate_all_allocations(num_agents, num_items):
    """
    Generate all possible allocations of items to agents efficiently.
    
    Args:
        num_agents (int): Number of agents
        num_items (int): Number of items
        
    Returns:
        numpy.ndarray: 3D array of shape (num_allocations, num_agents, num_items)
                      where a 1 indicates an item is assigned to an agent
    """
    # Calculate total number of allocations
    num_allocations = num_agents ** num_items
    
    # Create empty allocation array
    allocations = np.zeros((num_allocations, num_agents, num_items), dtype=int)
    
    # Generate all possible assignments efficiently
    all_assignments = list(product(range(num_agents), repeat=num_items))
    
    # Fill allocations array
    for alloc_idx, assignment in enumerate(all_assignments):
        for item, agent in enumerate(assignment):
            allocations[alloc_idx, agent, item] = 1
            
    return allocations

def calculate_utilities(allocations, valuations):
    """
    Calculate utilities for all allocations efficiently using broadcasting.
    
    Args:
        allocations (numpy.ndarray): 3D array of allocations (num_allocations, num_agents, num_items)
        valuations (numpy.ndarray): 2D array of valuations (num_agents, num_items)
        
    Returns:
        numpy.ndarray: 2D array of utilities (num_allocations, num_agents)
    """
    # Reshape valuations for broadcasting
    valuations_reshaped = valuations.reshape(1, *valuations.shape)
    
    # Calculate utilities using element-wise multiplication and sum
    utilities = np.sum(allocations * valuations_reshaped, axis=2)
    
    return utilities

def find_envy_free_allocations(allocations, utilities, valuations):
    """
    Find allocations that are envy-free using vectorized operations.
    
    Args:
        allocations (numpy.ndarray): 3D array of allocations
        utilities (numpy.ndarray): 2D array of utilities
        valuations (numpy.ndarray): 2D array of valuations
        
    Returns:
        numpy.ndarray: Boolean array indicating which allocations are envy-free
        numpy.ndarray: Filtered utilities for envy-free allocations
    """
    num_allocations, num_agents, num_items = allocations.shape
    
    # Calculate utilities each agent would get from each other agent's allocation
    cross_utilities = np.zeros((num_allocations, num_agents, num_agents))
    
    for i in range(num_agents):
        # Calculate utility agent i would get from each agent's allocation
        agent_valuations = valuations[i]
        cross_utilities[:, i, :] = np.sum(allocations * agent_valuations.reshape(1, 1, -1), axis=2)
    
    # Check for envy-freeness
    is_envy_free = np.ones(num_allocations, dtype=bool)
    
    # Get utility each agent gets from their own allocation
    own_utilities = np.array([np.diagonal(cross_utilities[a]) for a in range(num_allocations)])
    
    # Check if any agent would prefer another agent's allocation
    for a in range(num_allocations):
        for i in range(num_agents):
            if np.any((cross_utilities[a, i, :] > own_utilities[a, i]) & (np.arange(num_agents) != i)):
                is_envy_free[a] = False
                break
    
    # Filter utilities to only include envy-free allocations
    envy_free_utilities = utilities[is_envy_free]
    
    return is_envy_free, envy_free_utilities

def calculate_welfare_metrics(utilities):
    """
    Calculate various welfare metrics for allocations.
    
    Args:
        utilities (numpy.ndarray): 2D array of utilities (num_allocations, num_agents)
        
    Returns:
        tuple: (egalitarian_welfare, nash_welfare, utilitarian_welfare)
    """
    # Egalitarian welfare: minimum utility across agents
    egalitarian_welfare = np.min(utilities, axis=1)
    
    # Nash welfare: product of utilities (add epsilon to avoid zeros)
    epsilon = 1e-10
    nash_welfare = np.prod(utilities + epsilon, axis=1)
    
    # Utilitarian welfare: sum of utilities
    utilitarian_welfare = np.sum(utilities, axis=1)
    
    return egalitarian_welfare, nash_welfare, utilitarian_welfare

def find_max_welfare_allocations(utilities):
    """
    Find allocations with maximum welfare for each metric.
    
    Args:
        utilities (numpy.ndarray): 2D array of utilities
        
    Returns:
        tuple: Indices and values of max welfare allocations
    """
    egalitarian_welfare, nash_welfare, utilitarian_welfare = calculate_welfare_metrics(utilities)
    
    max_egalitarian_idx = np.argmax(egalitarian_welfare)
    max_nash_idx = np.argmax(nash_welfare)
    max_utilitarian_idx = np.argmax(utilitarian_welfare)
    
    max_egalitarian = egalitarian_welfare[max_egalitarian_idx]
    max_nash = nash_welfare[max_nash_idx]
    max_utilitarian = utilitarian_welfare[max_utilitarian_idx]
    
    return (max_egalitarian_idx, max_nash_idx, max_utilitarian_idx, 
            max_egalitarian, max_nash, max_utilitarian)

def calculate_envy(valuations, allocation, num_agents, num_items):
    """
    Calculate envy matrices for a single allocation.
    
    Args:
        valuations (numpy.ndarray): 2D array of valuations
        allocation (numpy.ndarray): 2D array representing allocation
        num_agents (int): Number of agents
        num_items (int): Number of items
        
    Returns:
        tuple: (clipped_envy_matrix, unclipped_envy_matrix)
    """
    # Calculate utilities each agent gets from their allocation
    own_utilities = np.sum(allocation * valuations, axis=1)
    
    # Calculate utilities each agent would get from other agents' allocations
    cross_utilities = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            cross_utilities[i, j] = np.sum(allocation[j] * valuations[i])
    
    # Calculate envy matrices
    unclipped_envy = cross_utilities - own_utilities.reshape(-1, 1)
    clipped_envy = np.maximum(0, unclipped_envy)
    
    # Set self-envy to 0
    np.fill_diagonal(clipped_envy, 0)
    np.fill_diagonal(unclipped_envy, 0)
    
    return clipped_envy, unclipped_envy

def is_ef1(valuations, allocation, num_agents):
    """
    Check if an allocation satisfies EF1 (Envy-Free up to 1 item).
    
    Args:
        valuations (numpy.ndarray): 2D array of valuations
        allocation (numpy.ndarray): 2D array representing allocation
        num_agents (int): Number of agents
        
    Returns:
        bool: True if allocation is EF1, False otherwise
    """
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
                
            # Calculate envy of agent i towards agent j
            items_of_j = np.where(allocation[j] == 1)[0]
            if len(items_of_j) == 0:
                continue
                
            # Calculate utility agent i gets from their allocation
            own_utility = np.sum(allocation[i] * valuations[i])
            
            # Check if removing any item from j's allocation would eliminate envy
            ef1_satisfied = False
            for item in items_of_j:
                # Create temporary allocation with item removed
                temp_allocation = allocation[j].copy()
                temp_allocation[item] = 0
                
                # Calculate utility agent i would get from this modified allocation
                utility_without_item = np.sum(temp_allocation * valuations[i])
                
                # If removing this item eliminates envy, EF1 is satisfied for this pair
                if own_utility >= utility_without_item:
                    ef1_satisfied = True
                    break
                    
            if not ef1_satisfied:
                return False
                
    return True

def is_pareto_optimal(utilities, allocation_idx):
    """
    Check if an allocation is Pareto optimal.
    
    Args:
        utilities (numpy.ndarray): 2D array of utilities for all allocations
        allocation_idx (int): Index of the allocation to check
        
    Returns:
        bool: True if allocation is Pareto optimal, False otherwise
    """
    allocation_utility = utilities[allocation_idx]
    
    # Check if any other allocation dominates this one
    for i in range(len(utilities)):
        if i == allocation_idx:
            continue
            
        # Check if allocation i dominates allocation_idx
        if np.all(utilities[i] >= allocation_utility) and np.any(utilities[i] > allocation_utility):
            return False
            
    return True

def extract_allocation_from_json(output_file):
    """
    Extract allocation matrix from output file.
    
    Args:
        output_file (str): Path to output file
        
    Returns:
        numpy.ndarray or None: Allocation matrix if successful, None otherwise
    """
    try:
        with open(output_file, 'r') as f:
            # Read file content
            content = f.read()
    
        # Find the last occurrence of "json"
        # print(content)
        last_occurrence = content.rfind("json")

        quotes_occurrence = content.rfind("```")
        
        if last_occurrence == -1:
            # If "json" is not found, return None or a custom message
            return None
        
        # Extract everything from the last occurrence to the end
        result = content[last_occurrence + 4:quotes_occurrence]

        output_json = json.loads(result)
        output_dict = {int(k): v for k, v in output_json.items()}

        
        # Determine number of agents and items
        num_agents = len(output_dict)
        num_items = len(output_dict[0])
        
        # Create allocation matrix
        allocation_matrix = np.zeros((num_agents, num_items), dtype=int)
        
        # Fill allocation matrix
        for agent, items in output_dict.items():
            for item in items:
                allocation_matrix[agent, item] = 1
                
        return allocation_matrix
            
    except Exception as e:
        print(f"Error extracting allocation from {output_file}: {e}")
        return None

def extract_valuation_from_file(output_file):
    """
    Extract valuation table from output file.
    
    Args:
        output_file (str): Path to output file
        
    Returns:
        numpy.ndarray or None: Valuation matrix if successful, None otherwise
    """
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
            # Find valuation table
            try:
                start_idx = lines.index('Valuation Table:\n')
                end_idx = lines.index('Output:\n')
                
                valuation_lines = ''.join(lines[start_idx+1:end_idx])
                
                # Parse valuation table
                valuation_str = valuation_lines.replace('[', '').replace(']', '')
                valuation_array = np.fromstring(valuation_str, sep=' ')
                
                # Determine dimensions
                agent_count = 0
                for i, line in enumerate(lines[start_idx+1:end_idx]):
                    if len(line.strip()) > 0:
                        agent_count += 1
                
                item_count = len(valuation_array) // agent_count
                
                # Reshape array
                valuation_matrix = valuation_array.reshape(agent_count, item_count)
                
                return valuation_matrix
                
            except (ValueError, IndexError):
                # Try alternative approach - find array-like structure
                valuation_lines = []
                recording = False
                
                for line in lines:
                    if 'Valuation Table:' in line:
                        recording = True
                        continue
                    if 'Output:' in line:
                        recording = False
                        break
                    if recording:
                        valuation_lines.append(line)
                
                if valuation_lines:
                    # Join lines and parse
                    valuation_str = ' '.join([line.strip() for line in valuation_lines])
                    # Find and extract numeric values
                    import re
                    numbers = re.findall(r'\d+', valuation_str)
                    valuation_array = np.array([int(num) for num in numbers])
                    
                    # Try to infer dimensions
                    # This is a simplification - you may need more sophisticated logic
                    from math import sqrt
                    approx_dim = int(sqrt(len(valuation_array)))
                    
                    # If it's a perfect square, assume equal agents and items
                    if approx_dim * approx_dim == len(valuation_array):
                        return valuation_array.reshape(approx_dim, approx_dim)
                    else:
                        # Try to find a reasonable factorization
                        for i in range(2, 11):  # Try up to 10 agents
                            if len(valuation_array) % i == 0:
                                return valuation_array.reshape(i, len(valuation_array) // i)
                
                return None
                
    except Exception as e:
        print(f"Error extracting valuation from {output_file}: {e}")
        return None

def evaluate_all_outputs(base_path, output_csv="evaluation_results.csv"):
    """
    Evaluate all outputs in a directory structure and save results to CSV.
    
    Args:
        base_path (str): Base path to outputs directory
        output_csv (str): Path to output CSV file
    """
    # Find all output directories
    agent_dirs = glob.glob(f"{base_path}/agents_*/")
    
    # Initialize results DataFrame
    results = []
    
    for agent_dir in agent_dirs:
        # Extract agent count
        agent_count = int(agent_dir.split('agents_')[1].strip('/\\'))
        
        # Find item directories
        item_dirs = glob.glob(f"{agent_dir}items_*/")
        
        for item_dir in item_dirs:
            # Extract item count
            item_count = int(item_dir.split('items_')[1].strip('/\\'))
            
            # Find model directories
            model_dirs = glob.glob(f"{item_dir}*/")
            
            for model_dir in model_dirs:
                # Extract model name
                model_name = os.path.basename(os.path.normpath(model_dir))
                
                # Find prompt type directories
                prompt_dirs = glob.glob(f"{model_dir}*/")
                
                for prompt_dir in prompt_dirs:
                    # Extract prompt type
                    prompt_type = os.path.basename(os.path.normpath(prompt_dir))
                    
                    # Find output files
                    output_files = glob.glob(f"{prompt_dir}output_*.txt")
                    
                    if not output_files:
                        continue
                        
                    print(f"Processing {agent_count} agents, {item_count} items, {model_name}, {prompt_type}...")
                    
                    # Generate all possible allocations (do this only once per agent-item combo)
                    all_allocations = generate_all_allocations(agent_count, item_count)
                    
                    # Process each output file
                    for output_file in tqdm(output_files, desc="Processing outputs"):
                        try:
                            # Extract test ID
                            test_id = int(os.path.basename(output_file).split('_')[1].split('.')[0])
                            
                            # Extract valuation and allocation
                            valuation_matrix = extract_valuation_from_file(output_file)
                            llm_allocation = extract_allocation_from_json(output_file)
                            
                            if valuation_matrix is None or llm_allocation is None:
                                print(f"Skipping {output_file} - could not extract valuation or allocation")
                                continue
                                
                            # Calculate utilities for all allocations
                            all_utilities = calculate_utilities(all_allocations, valuation_matrix)
                            
                            # Find envy-free allocations
                            is_envy_free, ef_utilities = find_envy_free_allocations(
                                all_allocations, all_utilities, valuation_matrix
                            )
                            
                            # Calculate maximum welfare metrics across all allocations
                            _, _, _, max_egalitarian, max_nash, max_utilitarian = find_max_welfare_allocations(all_utilities)
                            
                            # Find maximum welfare metrics across envy-free allocations
                            if np.any(is_envy_free):
                                _, _, _, max_ef_egalitarian, max_ef_nash, max_ef_utilitarian = find_max_welfare_allocations(ef_utilities)
                            else:
                                max_ef_egalitarian, max_ef_nash, max_ef_utilitarian = 0, 0, 0
                            
                            # Find LLM allocation in all_allocations
                            llm_idx = -1
                            for idx, alloc in enumerate(all_allocations):
                                if np.array_equal(alloc, llm_allocation):
                                    llm_idx = idx
                                    break
                                    
                            if llm_idx == -1:
                                print(f"Warning: LLM allocation not found in all possible allocations for {output_file}")
                                continue
                                
                            # Calculate LLM allocation welfare metrics
                            llm_utility = all_utilities[llm_idx]
                            llm_egalitarian = np.min(llm_utility)
                            llm_nash = np.prod(llm_utility + 1e-10)
                            llm_utilitarian = np.sum(llm_utility)
                            
                            # Check if LLM allocation is envy-free
                            llm_is_ef = is_envy_free[llm_idx]
                            
                            # Check if LLM allocation is EF1
                            llm_is_ef1 = is_ef1(valuation_matrix, llm_allocation, agent_count)
                            
                            # Check if LLM allocation is Pareto optimal
                            llm_is_pareto = is_pareto_optimal(all_utilities, llm_idx)
                            
                            # Calculate envy matrices for LLM allocation
                            clipped_envy, unclipped_envy = calculate_envy(
                                valuation_matrix, llm_allocation, agent_count, item_count
                            )
                            
                            # Calculate envy metrics
                            max_envy = np.max(clipped_envy)
                            sum_envy = np.sum(clipped_envy)
                            
                            # Store results
                            result = {
                                'Agents': agent_count,
                                'Items': item_count,
                                'Model': model_name,
                                'Prompt_Type': prompt_type,
                                'Test_ID': test_id,
                                'Output_File': output_file,
                                
                                # Allocation properties
                                'Is_Envy_Free': llm_is_ef,
                                'Is_EF1': llm_is_ef1,
                                'Is_Pareto_Optimal': llm_is_pareto,
                                'Max_Envy': max_envy,
                                'Sum_Envy': sum_envy,
                                
                                # LLM allocation welfare metrics
                                'LLM_Egalitarian_Welfare': llm_egalitarian,
                                'LLM_Nash_Welfare': llm_nash,
                                'LLM_Utilitarian_Welfare': llm_utilitarian,
                                
                                # Maximum possible welfare metrics
                                'Max_Egalitarian_Welfare': max_egalitarian,
                                'Max_Nash_Welfare': max_nash,
                                'Max_Utilitarian_Welfare': max_utilitarian,
                                
                                # Maximum envy-free welfare metrics
                                'Max_EF_Egalitarian_Welfare': max_ef_egalitarian,
                                'Max_EF_Nash_Welfare': max_ef_nash,
                                'Max_EF_Utilitarian_Welfare': max_ef_utilitarian,
                                
                                # Welfare ratios (LLM / Maximum)
                                'Egalitarian_Ratio': llm_egalitarian / max_egalitarian if max_egalitarian > 0 else 0,
                                'Nash_Ratio': llm_nash / max_nash if max_nash > 0 else 0,
                                'Utilitarian_Ratio': llm_utilitarian / max_utilitarian if max_utilitarian > 0 else 0,
                                
                                # Envy-free welfare ratios (LLM / Maximum EF)
                                'EF_Egalitarian_Ratio': llm_egalitarian / max_ef_egalitarian if max_ef_egalitarian > 0 else 0,
                                'EF_Nash_Ratio': llm_nash / max_ef_nash if max_ef_nash > 0 else 0,
                                'EF_Utilitarian_Ratio': llm_utilitarian / max_ef_utilitarian if max_ef_utilitarian > 0 else 0,
                                
                                # Count of envy-free allocations
                                'Num_Envy_Free_Allocations': np.sum(is_envy_free),
                                'Total_Allocations': len(all_allocations),
                                'EF_Percentage': np.sum(is_envy_free) / len(all_allocations) * 100
                            }
                            
                            results.append(result)
                            
                            # Save results to CSV in prompt_dir
                            df = pd.DataFrame([result])
                            csv_path = f"{prompt_dir}evaluation_result_{test_id}.csv"
                            df.to_csv(csv_path, index=False)
                            
                        except Exception as e:
                            print(f"Error processing {output_file}: {e}")
                            continue
    
    # Combine all results
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate aggregate statistics
        aggregated = results_df.groupby(['Agents', 'Items', 'Model', 'Prompt_Type']).agg({
            'Is_Envy_Free': 'mean',
            'Is_EF1': 'mean',
            'Is_Pareto_Optimal': 'mean',
            'Max_Envy': 'mean',
            'Sum_Envy': 'mean',
            'Egalitarian_Ratio': 'mean',
            'Nash_Ratio': 'mean',
            'Utilitarian_Ratio': 'mean',
            'Num_Envy_Free_Allocations': 'mean',
            'EF_Percentage': 'mean',
        }).reset_index()
        
        # Save results
        results_df.to_csv(f"{base_path}/{output_csv}", index=False)
        aggregated.to_csv(f"{base_path}/aggregated_{output_csv}", index=False)
        
        print(f"Evaluation complete. Results saved to {base_path}/{output_csv}")
        
        return results_df
    else:
        print("No results generated.")
        return None

if __name__ == "__main__":
    # Example usage
    base_path = "outputs"
    evaluate_all_outputs(base_path)