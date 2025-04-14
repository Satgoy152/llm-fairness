import numpy as np
from itertools import product
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_all_allocations(num_agents, num_items):
    """
    Generate all possible allocations of items to agents as a 3D numpy array.
    
    Args:
        num_agents (int): Number of agents
        num_items (int): Number of items
        
    Returns:
        numpy.ndarray: 3D array of shape (num_allocations, num_agents, num_items)
                      where a 1 indicates an item is assigned to an agent
    """
    # Calculate the total number of possible allocations
    num_allocations = num_agents ** num_items
    
    # Create an empty 3D numpy array to store all allocations
    allocations = np.zeros((num_allocations, num_agents, num_items), dtype=int)
    
    # Generate all possible assignments of items to agents
    all_assignments = list(product(range(num_agents), repeat=num_items))
    
    # Fill the allocations array
    for alloc_idx, assignment in enumerate(all_assignments):
        for item, agent in enumerate(assignment):
            allocations[alloc_idx, agent, item] = 1
            
    return allocations

# Check if all allocations are envy-free
def calculate_all_envy(valuation_tables: np.ndarray, allocation_matrices, num_agents: int, num_items: int):
    # we are using clipped valuations to avoid negative values
    # formula: max{0, v_i(A_j)- v_i(A_i)}
    # for every agent i we calculate the envy towards every other agent j based on the valuation of agent i
    # Create an allocation matrix where each row represents the items allocated to each agent

    # Use dot product to compute the allocation table (valuations of each agent for other agents' allocations)
    allocation_matrices_T = np.transpose(allocation_matrices, (0, 2, 1))

    allocation_table = np.matmul(valuation_tables, allocation_matrices_T)
    # Compute envy matrices
    envy_matrix_clipped = np.maximum(0, allocation_table - np.diagonal(allocation_table, axis1=1, axis2=2)[:, :, np.newaxis])
    
    envy_matrix_clipped = envy_matrix_clipped.max(axis=1)

    envy_matrix_clipped = envy_matrix_clipped.max(axis=1)

    list_of_envy = [envy_matrix_clipped <= 0]

    return list_of_envy

def calculate_utilities(allocations, valuation_table):
    # use broadcasting to calculate the utilities
    # allocations shape: (allocations, num_agents, num_items)
    # valuation_table shape: (num_agents, num_items)
    # utilities shape: (allocations, num_agents)
    utilities = np.sum(allocations * valuation_table[np.newaxis, :, :], axis=2)
    return utilities

def get_max_nash_welfare(utlilities):
    """Get the max nash welfare of the allocations
    Args:
        utlilities (np.ndarray): (num_allocations, num_agents) array of utilities for each allocation
    """
    nash_welfare = np.prod(utlilities, axis=1)
    max_nash_welfare = np.max(nash_welfare)
    return max_nash_welfare

def get_max_utilitarian_welfare(utlilities):
    """Get the max utilitarian welfare of the allocations
    Args:
        utlilities (np.ndarray): (num_allocations, num_agents) array of utilities for each allocation
    """
    utilitarian_welfare = np.sum(utlilities, axis=1)
    max_utilitarian_welfare = np.max(utilitarian_welfare)
    return max_utilitarian_welfare

def get_max_egalitarian_welfare(utlilities):
    """Get the max egalitarian welfare of the allocations
    Args:
        utlilities (np.ndarray): (num_allocations, num_agents) array of utilities for each allocation
    """
    egalitarian_welfare = np.min(utlilities, axis=1)
    max_egalitarian_welfare = np.max(egalitarian_welfare)
    return max_egalitarian_welfare

def find_pareto_frontier(utilities):
    """
    Find the Pareto frontier from a set of utility vectors.
    
    Args:
        utilities (numpy.ndarray): 2D array of utilities
        
    Returns:
        numpy.ndarray: Boolean array indicating which allocations are Pareto optimal
        numpy.ndarray: Filtered utilities for Pareto optimal allocations
    """
    num_allocations = utilities.shape[0]
    is_pareto_optimal = np.ones(num_allocations, dtype=bool)
    
    for i in range(num_allocations):
        if not is_pareto_optimal[i]:
            continue
            
        for j in range(num_allocations):
            if i == j or not is_pareto_optimal[j]:
                continue
                
            # Check if j dominates i (all utilities in j >= i and at least one >)
            if np.all(utilities[j] >= utilities[i]) and np.any(utilities[j] > utilities[i]):
                is_pareto_optimal[i] = False
                break
    
    # Filter utilities to only include Pareto optimal allocations
    pareto_optimal_utilities = utilities[is_pareto_optimal]
    
    return is_pareto_optimal, pareto_optimal_utilities

def plot_pareto_frontier(utilities, title="Pareto Frontier"):
    """
    Plot the Pareto frontier for 2 or 3 agents.
    
    Args:
        utilities (numpy.ndarray): 2D array of Pareto optimal utilities
        title (str): Plot title
    """
    num_agents = utilities.shape[1]
    
    if num_agents == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(utilities[:, 0], utilities[:, 1], c='blue', marker='o')
        plt.xlabel('Utility for Agent 0')
        plt.ylabel('Utility for Agent 1')
        plt.title(title)
        plt.grid(True)
        plt.savefig('pareto_frontier_2d.png')
        plt.show()
        
    elif num_agents == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(utilities[:, 0], utilities[:, 1], utilities[:, 2], c='blue', marker='o')
        ax.set_xlabel('Utility for Agent 0')
        ax.set_ylabel('Utility for Agent 1')
        ax.set_zlabel('Utility for Agent 2')
        ax.set_title(title)
        plt.savefig('pareto_frontier_3d.png')
        plt.show()
        
    else:
        print(f"Plotting is only supported for 2 or 3 agents. You have {num_agents} agents.")
        print("First few Pareto optimal utility vectors:")
        for i in range(min(5, len(utilities))):
            print(f"Allocation {i}: {utilities[i]}")

def calculate_welfare_metrics(utilities):
    """
    Calculate various welfare metrics for each allocation.
    
    Args:
        utilities (numpy.ndarray): 2D array of shape (num_allocations, num_agents) with utility values
        
    Returns:
        tuple: (egalitarian_welfare, nash_welfare, utilitarian_welfare)
            - egalitarian_welfare: 1D array of shape (num_allocations,)
            - nash_welfare: 1D array of shape (num_allocations,)
            - utilitarian_welfare: 1D array of shape (num_allocations,)
    """
    # Egalitarian welfare: minimum utility across agents for each allocation
    egalitarian_welfare = np.min(utilities, axis=1)
    
    # Nash welfare: product of utilities across agents for each allocation
    # Adding a small epsilon to avoid zero values which would zero out the product
    epsilon = 1e-10
    nash_welfare = np.prod(utilities + epsilon, axis=1)
    
    # Utilitarian welfare: sum of utilities across agents for each allocation
    utilitarian_welfare = np.sum(utilities, axis=1)
    
    return egalitarian_welfare, nash_welfare, utilitarian_welfare

def find_max_welfare_allocations(utilities):
    """
    Find allocations with maximum welfare for each metric.
    
    Args:
        utilities (numpy.ndarray): 2D array of shape (num_allocations, num_agents) with utility values
        
    Returns:
        tuple: (max_egalitarian_idx, max_nash_idx, max_utilitarian_idx, max_egalitarian, max_nash, max_utilitarian)
    """
    egalitarian_welfare, nash_welfare, utilitarian_welfare = calculate_welfare_metrics(utilities)
    
    max_egalitarian_idx = np.argmax(egalitarian_welfare)
    max_nash_idx = np.argmax(nash_welfare)
    max_utilitarian_idx = np.argmax(utilitarian_welfare)
    
    max_egalitarian = egalitarian_welfare[max_egalitarian_idx]
    max_nash = nash_welfare[max_nash_idx]
    max_utilitarian = utilitarian_welfare[max_utilitarian_idx]
    
    return max_egalitarian_idx, max_nash_idx, max_utilitarian_idx, max_egalitarian, max_nash, max_utilitarian
# Example valuation table from the original problem

# path = "outputs/agents_2/items_7/gpt4o/zero_shot0"
# agents = 2
# items = 7

# for i in range(500):
#     i = 7

#     with open(f"{path}/output_{i+1}.txt", "r") as file:
#         # get allocation matrix

#         # get valuation table find where 'Valuation Table:' and read the next lines
#         try:
#             lines = file.readlines()
#             start = lines.index('Valuation Table:\n')
#             end = lines.index('Output:\n')
#             valuation_table = lines[start+1:end]
#             valuation_table = ''.join(valuation_table)
#             valuation_table = np.fromstring(valuation_table.replace('[', '').replace(']', ''), sep=' ')
#             valuation_table = valuation_table.reshape(agents, items)
#             valuation_table = valuation_table.astype(int)
#         except:
#             print(f"Error reading file {i+1}")
#             continue



#     valuations = valuation_table

#     num_agents, num_items = valuations.shape

#     print(f"Generating all possible allocations for {num_agents} agents and {num_items} items...")
#     print(f"This will create {num_agents ** num_items} different allocations.")

#     # Generate all possible allocations
#     allocations = generate_all_allocations(num_agents, num_items)
#     print(f"Generated {allocations.shape[0]} allocations.")

#     # Calculate utilities for all allocations
#     print("Calculating utilities for all allocations...")
#     utilities = calculate_utilities(allocations, valuations)

#     # Find envy-free allocations
#     print("Finding envy-free allocations...")
#     is_envy_free, envy_free_utilities = find_envy_free_allocations(allocations, utilities)
#     num_envy_free = np.sum(is_envy_free)
#     print(f"Found {num_envy_free} envy-free allocations out of {allocations.shape[0]} total allocations.")

#     if num_envy_free == 0:
#         print("No envy-free allocations found. Consider relaxing constraints or using a different approach.")

#     # Find Pareto optimal allocations among envy-free ones
#     print("Finding Pareto optimal allocations among envy-free allocations...")
#     is_pareto_optimal, pareto_optimal_utilities = find_pareto_frontier(envy_free_utilities)
#     num_pareto = np.sum(is_pareto_optimal)
#     print(f"Found {num_pareto} Pareto optimal allocations out of {num_envy_free} envy-free allocations.")

#     # Get the indices of the Pareto optimal allocations in the original allocation array
#     envy_free_indices = np.where(is_envy_free)[0]
#     pareto_indices = envy_free_indices[is_pareto_optimal]

#     # Print some sample Pareto optimal allocations
#     print("\nSample Pareto optimal allocations:")
#     for i in range(min(5, len(pareto_indices))):
#         idx = pareto_indices[i]
#         print(f"\nAllocation {i+1}:")
#         for agent in range(num_agents):
#             items = np.where(allocations[idx, agent] == 1)[0]
#             print(f"  Agent {agent}: {list(items)}, Utility: {utilities[idx, agent]}")

#     # Plot the Pareto frontier if there are 2 or 3 agents
#     if num_agents <= 3:
#         print("\nPlotting Pareto frontier...")
#         plot_pareto_frontier(pareto_optimal_utilities, title="Pareto Frontier of Envy-Free Allocations")

#     break

#     # Save results if desired
#     # save_to_file = input("\nDo you want to save the Pareto optimal allocations to file? (y/n): ")
#     # if save_to_file.lower() == 'y':
#     #     np.save('pareto_optimal_allocations.npy', allocations[pareto_indices])
#     #     np.save('pareto_optimal_utilities.npy', pareto_optimal_utilities)
#     #     print("Saved Pareto optimal allocations and utilities to file.")
