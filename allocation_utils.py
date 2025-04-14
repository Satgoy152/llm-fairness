import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict, Optional, Any, Union


def generate_all_allocations(num_agents: int, num_items: int) -> np.ndarray:
    """
    Generate all possible allocations of items to agents as a matrix.
    
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


def calculate_utilities(allocations: np.ndarray, valuations: np.ndarray) -> np.ndarray:
    """
    Calculate utilities for all allocations using vectorized operations.
    
    Args:
        allocations (numpy.ndarray): 3D array of allocations (num_allocations, num_agents, num_items)
        valuations (numpy.ndarray): 2D array of valuations (num_agents, num_items)
        
    Returns:
        numpy.ndarray: 2D array of utilities (num_allocations, num_agents)
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


def find_envy_free_allocations(allocations: np.ndarray, utilities: np.ndarray, valuations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find allocations that are envy-free (no agent prefers another agent's allocation).
    Uses vectorized operations for better performance.
    
    Args:
        allocations (numpy.ndarray): 3D array of allocations (num_allocations, num_agents, num_items)
        utilities (numpy.ndarray): 2D array of utilities (num_allocations, num_agents)
        valuations (numpy.ndarray): 2D array of valuations (num_agents, num_items)
        
    Returns:
        numpy.ndarray: Boolean array indicating which allocations are envy-free
        numpy.ndarray: Filtered utilities for envy-free allocations
    """
    num_allocations, num_agents, num_items = allocations.shape
    
    # Calculate what utility each agent would get from each other agent's allocation
    cross_utilities = np.zeros((num_allocations, num_agents, num_agents))
    
    for i in range(num_agents):
        # Use broadcasting to calculate utility agent i would get from each agent's allocation
        agent_valuations = valuations[i].reshape(1, 1, -1)  # shape: (1, 1, num_items)
        # For each allocation, calculate what agent i would get from each agent's bundle
        # Result shape: (num_allocations, num_agents)
        cross_utilities[:, i, :] = np.sum(allocations * agent_valuations, axis=2)
    
    # Determine if allocations are envy-free
    is_envy_free = np.ones(num_allocations, dtype=bool)
    
    # For each allocation and each agent i, compare their utility to what they'd get from other agents j
    for a in range(num_allocations):
        for i in range(num_agents):
            # Check if agent i would prefer any other agent's allocation
            for j in range(num_agents):
                if i != j and cross_utilities[a, i, j] > cross_utilities[a, i, i]:
                    is_envy_free[a] = False
                    break
            if not is_envy_free[a]:
                break
    
    # Filter utilities to only include envy-free allocations
    envy_free_utilities = utilities[is_envy_free]
    
    return is_envy_free, envy_free_utilities


def find_pareto_frontier(utilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the Pareto frontier from a set of utility vectors.
    
    Args:
        utilities (numpy.ndarray): 2D array of utilities
        
    Returns:
        numpy.ndarray: Boolean array indicating which allocations are Pareto optimal
        numpy.ndarray: Filtered utilities for Pareto optimal allocations
    """
    if utilities.shape[0] == 0:
        return np.zeros(0, dtype=bool), utilities
    
    num_allocations = utilities.shape[0]
    is_pareto_optimal = np.ones(num_allocations, dtype=bool)
    
    # More efficient implementation for larger datasets
    for i in range(num_allocations):
        if not is_pareto_optimal[i]:
            continue
        
        # Compare allocation i against all other allocations
        dominated = False
        for j in range(num_allocations):
            if i != j and is_pareto_optimal[j]:
                # Check if j dominates i (all utilities in j >= i and at least one >)
                if np.all(utilities[j] >= utilities[i]) and np.any(utilities[j] > utilities[i]):
                    is_pareto_optimal[i] = False
                    break
    
    # Filter utilities to only include Pareto optimal allocations
    pareto_optimal_utilities = utilities[is_pareto_optimal]
    
    return is_pareto_optimal, pareto_optimal_utilities


def calculate_welfare_metrics(utilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def find_max_welfare_allocations(utilities: np.ndarray) -> Tuple[int, int, int, float, float, float]:
    """
    Find allocations with maximum welfare for each metric.
    
    Args:
        utilities (numpy.ndarray): 2D array of shape (num_allocations, num_agents) with utility values
        
    Returns:
        tuple: (max_egalitarian_idx, max_nash_idx, max_utilitarian_idx, max_egalitarian, max_nash, max_utilitarian)
    """
    if utilities.shape[0] == 0:
        return -1, -1, -1, 0, 0, 0
    
    egalitarian_welfare, nash_welfare, utilitarian_welfare = calculate_welfare_metrics(utilities)
    
    max_egalitarian_idx = np.argmax(egalitarian_welfare)
    max_nash_idx = np.argmax(nash_welfare)
    max_utilitarian_idx = np.argmax(utilitarian_welfare)
    
    max_egalitarian = egalitarian_welfare[max_egalitarian_idx]
    max_nash = nash_welfare[max_nash_idx]
    max_utilitarian = utilitarian_welfare[max_utilitarian_idx]
    
    return max_egalitarian_idx, max_nash_idx, max_utilitarian_idx, max_egalitarian, max_nash, max_utilitarian


def check_ef1(allocation_matrix: np.ndarray, valuations: np.ndarray) -> bool:
    """
    Check if an allocation is envy-free up to one item (EF1).
    
    Args:
        allocation_matrix: Allocation matrix of shape (num_agents, num_items)
        valuations: Valuation table of shape (num_agents, num_items)
        
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


def check_efx(allocation_matrix: np.ndarray, valuations: np.ndarray) -> bool:
    """
    Check if an allocation is envy-free up to any item (EFX).
    
    Args:
        allocation_matrix: Allocation matrix of shape (num_agents, num_items)
        valuations: Valuation table of shape (num_agents, num_items)
        
    Returns:
        bool: True if the allocation is EFX, False otherwise
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
                
                # Check if removing any item from j's bundle eliminates i's envy
                for item in other_items:
                    # Calculate value of j's bundle minus this item
                    other_value_minus_item = sum(valuations[i, it] for it in other_items if it != item)
                    
                    # If i still envies j after removing this item, allocation is not EFX
                    if own_value < other_value_minus_item:
                        return False
    
    return True


def plot_pareto_frontier(utilities: np.ndarray, title: str = "Pareto Frontier", save_path: Optional[str] = None):
    """
    Plot the Pareto frontier for 2 or 3 agents.
    
    Args:
        utilities (numpy.ndarray): 2D array of Pareto optimal utilities
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    if utilities.shape[0] == 0:
        print("No data to plot")
        return
    
    num_agents = utilities.shape[1]
    
    if num_agents == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(utilities[:, 0], utilities[:, 1], c='blue', marker='o')
        plt.xlabel('Utility for Agent 0')
        plt.ylabel('Utility for Agent 1')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    elif num_agents == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(utilities[:, 0], utilities[:, 1], utilities[:, 2], c='blue', marker='o')
        ax.set_xlabel('Utility for Agent 0')
        ax.set_ylabel('Utility for Agent 1')
        ax.set_zlabel('Utility for Agent 2')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    else:
        print(f"Plotting is only supported for 2 or 3 agents. You have {num_agents} agents.")
        print("First few Pareto optimal utility vectors:")
        for i in range(min(5, len(utilities))):
            print(f"Allocation {i}: {utilities[i]}")