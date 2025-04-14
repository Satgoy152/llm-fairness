import numpy as np
import json
import pandas as pd
import os
import itertools
from stats_utils import calculate_welfare_metrics, find_max_welfare_allocations

def calculate_envy(valuation_tables: np.ndarray, allocation_matrices, num_agents: int, num_items: int):
    # we are using clipped valuations to avoid negative values
    # formula: max{0, v_i(A_j)- v_i(A_i)}
    # for every agent i we calculate the envy towards every other agent j based on the valuation of agent i
    # Create an allocation matrix where each row represents the items allocated to each agent

    # Use dot product to compute the allocation table (valuations of each agent for other agents' allocations)
    allocation_matrices_T = np.transpose(allocation_matrices, (0, 2, 1))

    allocation_table = np.matmul(valuation_tables, allocation_matrices_T)
    # Compute envy matrices
    envy_matrix_clipped = np.maximum(0, allocation_table - np.diagonal(allocation_table, axis1=1, axis2=2)[:, :, np.newaxis])
    envy_matrix_unclipped = allocation_table - np.diagonal(allocation_table, axis1=1, axis2=2)[:, :, np.newaxis]

    # get the utility of each agent for their own allocation
    own_utilities = np.diagonal(allocation_table, axis1=1, axis2=2)

    # get nash, utilitarian and egalitarian welfare
    nash_welfare = np.prod(own_utilities, axis=1)
    utilitarian_welfare = np.sum(own_utilities, axis=1) 
    egalitarian_welfare = np.min(own_utilities, axis=1)

    return envy_matrix_clipped, envy_matrix_unclipped, nash_welfare, utilitarian_welfare, egalitarian_welfare

def evaluate_envy(envy_matrix_c, envy_matrix_u, agents, items, type_of_dist, path, num_outputs, number_of_possible_allocations,  envy_free_count , max_nash_welfare, max_utilitarian_welfare, max_egalitarian_welfare, nash_welfare, utilitarian_welfare, egalitarian_welfare):

    # get the max and sum of every column
    sum_envy = envy_matrix_c.sum(axis=1)
    sum_envy_unclipped = envy_matrix_u.sum(axis=1)

    # set the diagonals to infinity for max_envy
    n, m, _ = envy_matrix_u.shape
    i, j = np.arange(m), np.arange(m)
    # Use advanced indexing to set the diagonals
    envy_matrix_u[:, i, j] = -np.inf

    max_envy = envy_matrix_c.max(axis=1)
    max_envy_unclipped = envy_matrix_u.max(axis=1)


    test_id_max = num_outputs
    distribution = type_of_dist
    num_agents = agents
    num_items = items

    envy_freeness = pd.DataFrame()

    envy_freeness['Test_Id'] = range(1, test_id_max + 1)
    envy_freeness['Distribution'] = [distribution] * test_id_max
    envy_freeness['output_file_path'] = [path + '/output_' + str(i+1) + '.txt' for i in range(test_id_max)]
    for i in range(num_agents):
        envy_freeness['Agent_' + str(i+1) + '_Max'] = max_envy[:,i]
        envy_freeness['Agent_' + str(i+1) + '_Max_Unclipped'] = max_envy_unclipped[:,i]
        envy_freeness['Agent_' + str(i+1) + '_Sum'] = sum_envy[:,i]
        envy_freeness['Agent_' + str(i+1) + '_Sum_Unclipped'] = sum_envy_unclipped[:,i]

    envy_freeness['Max_Envies'] = max_envy.max(axis=1)
    envy_freeness['Max_Envies_Unclipped'] = max_envy_unclipped.max(axis=1)
    envy_freeness['Sum_Envies'] = sum_envy.sum(axis=1)
    envy_freeness['Sum_Envies_Unclipped'] = sum_envy_unclipped.sum(axis=1)
    envy_freeness['Envy_Freeness'] = envy_freeness['Max_Envies'] <= 0
    envy_freeness['Number_of_Possible_Allocations'] = number_of_possible_allocations
    envy_freeness['Number_of_Envy_Free_Allocations'] = envy_free_count
    envy_freeness['Max_Nash_Welfare'] = max_nash_welfare
    envy_freeness['Max_Utilitarian_Welfare'] = max_utilitarian_welfare
    envy_freeness['Max_Egalitarian_Welfare'] = max_egalitarian_welfare
    envy_freeness['LLM_Nash_Welfare'] = nash_welfare
    envy_freeness['LLM_Utilitarian_Welfare'] = utilitarian_welfare
    envy_freeness['LLM_Egalitarian_Welfare'] = egalitarian_welfare


    
    # if os.path.exists(path + '/envy_output.csv'):
    #     envy_freeness.to_csv(path + '/envy_output.csv', mode='a', index=False, header=False)
    # # check if csv file exists, if so append to it
    # else:
    #     print("File not found, creating new file...")
        # print(envy_freeness)
    envy_freeness.to_csv(path + '/envy_output.csv', index=False)

def evaluate_welfare(envy_df, utilities_list, llm_allocation_indices, path):
    """
    Calculate welfare metrics for each LLM allocation and maximum possible welfare,
    and save the results to a CSV file.
    
    Args:
        envy_df (pandas.DataFrame): DataFrame with envy metrics (has Test_Id, Distribution columns)
        utilities_list (list): List of 2D numpy arrays of utilities for all allocations
        llm_allocation_indices (list): List of indices of LLM allocations in all_allocations
        path (str): Path to save the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with welfare metrics
    """
    # Initialize welfare DataFrame with the same Test_Id, Distribution, and output_file_path columns
    welfare_df = envy_df[['Test_Id', 'Distribution', 'output_file_path']].copy()
    
    # Initialize lists for welfare metrics
    llm_egalitarian = []
    max_egalitarian = []
    egalitarian_ratio = []
    llm_nash = []
    max_nash = []
    nash_ratio = []
    llm_utilitarian = []
    max_utilitarian = []
    utilitarian_ratio = []
    
    # Calculate welfare metrics for each test case
    for i in range(len(utilities_list)):
        if i < len(llm_allocation_indices):
            utilities = utilities_list[i]
            llm_idx = llm_allocation_indices[i]
            
            # Calculate welfare metrics for all allocations
            egalitarian_welfare, nash_welfare, utilitarian_welfare = calculate_welfare_metrics(utilities)
            
            # Find maximum welfare values
            _, _, _, max_egal, max_nash, max_util = find_max_welfare_allocations(utilities)
            
            # Calculate welfare for the LLM allocation
            llm_egal = egalitarian_welfare[llm_idx]
            llm_nash = nash_welfare[llm_idx]
            llm_util = utilitarian_welfare[llm_idx]
            
            # Calculate ratios
            egal_ratio = llm_egal / max_egal if max_egal > 0 else 0
            nash_ratio = llm_nash / max_nash if max_nash > 0 else 0
            util_ratio = llm_util / max_util if max_util > 0 else 0
        else:
            # Use default values if data is missing
            llm_egal = max_egal = egal_ratio = 0
            llm_nash = max_nash = nash_ratio = 0
            llm_util = max_util = util_ratio = 0
        
        # Append to lists
        llm_egalitarian.append(llm_egal)
        max_egalitarian.append(max_egal)
        egalitarian_ratio.append(egal_ratio)
        llm_nash.append(llm_nash)
        max_nash.append(max_nash)
        nash_ratio.append(nash_ratio)
        llm_utilitarian.append(llm_util)
        max_utilitarian.append(max_util)
        utilitarian_ratio.append(util_ratio)
    
    # Add welfare metrics to the DataFrame
    welfare_df['LLM_Egalitarian_Welfare'] = llm_egalitarian
    welfare_df['Max_Egalitarian_Welfare'] = max_egalitarian
    welfare_df['Egalitarian_Ratio'] = egalitarian_ratio
    welfare_df['LLM_Nash_Welfare'] = llm_nash
    welfare_df['Max_Nash_Welfare'] = max_nash
    welfare_df['Nash_Ratio'] = nash_ratio
    welfare_df['LLM_Utilitarian_Welfare'] = llm_utilitarian
    welfare_df['Max_Utilitarian_Welfare'] = max_utilitarian
    welfare_df['Utilitarian_Ratio'] = utilitarian_ratio
    
    # Save to CSV
    welfare_df.to_csv(f"{path}/welfare_metrics.csv", index=False)
    
    return welfare_df
