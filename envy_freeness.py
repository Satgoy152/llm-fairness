import numpy as np
import json
import pandas as pd

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

    return envy_matrix_clipped, envy_matrix_unclipped

def evaluate_envy(envy_matrix_c, envy_matrix_u, agents, items, type_of_dist, path, num_outputs):

    # get the max and sum of every column
    max_envy = envy_matrix_c.max(axis=1)
    max_envy_unclipped = envy_matrix_u.max(axis=1)

    sum_envy = envy_matrix_c.sum(axis=1)
    sum_envy_unclipped = envy_matrix_u.sum(axis=1)

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
    print(envy_freeness['Max_Envies'] <= 0)
    envy_freeness['Envy_Freeness'] = envy_freeness['Max_Envies'] <= 0

    envy_freeness.to_csv(path + '/envy_output.csv', index=False)