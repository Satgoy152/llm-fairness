import numpy as np
import json

def calculate_envy(valuations: np.ndarray, allocation: json, num_agents: int, num_items: int):
    # we are using clipped valuations to avoid negative values
    # formula: max{0, v_i(A_j)- v_i(A_i)}
    # for every agent i we calculate the envy towards every other agent j based on the valuation of agent i


    # create a table for the valuation of agent i on agent j's allocation
    allocation_table = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            for item in allocation[str(j)]:
                allocation_table[i][j] += valuations[item][i]

    envy_matrix = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                for item in allocation[str(j)]:
                    envy_matrix[i][j] += max(0, valuations[item][i] - valuations[item][j])