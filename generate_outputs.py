from valuation_generation import generate_valuations
from prompt_generation import generate_prompts
from model_response import model_init, query_model, extract_json
from envy_freeness import calculate_envy, evaluate_envy
from stats_utils import generate_all_allocations, calculate_utilities, calculate_all_envy, get_max_nash_welfare, get_max_egalitarian_welfare, get_max_utilitarian_welfare

from itertools import product
import numpy as np
import pandas as pd
import logging
import json

def generate_outputs(agents: int, items: int, distribution: str='uniform', model: str = 'gpt4om', temperature: float = 0.7, prompt_type: str = 'zero-shot0', num_outputs: int = 10):
    path = f"outputs/agents_{agents}/items_{items}/{model}/{prompt_type}"

    # create log file
    try:
        with open(f"{path}/log.txt", "w") as file:
            file.write("")
    except FileNotFoundError:
        print(f"Path {path} not found")
        # create new directory
        import os
        os.makedirs(path)
        with open(f"{path}/log.txt", "w") as file:
            file.write("")
    
    logging.basicConfig(filename=f"{path}/log.txt", level=logging.INFO)

    logging.info(f"Generating outputs for {agents} agents and {items} items with a {distribution} distribution using model {model} with temperature {temperature} and prompt type {prompt_type}...")
    '''
    Generate outputs for a given model and prompt type

    Args:
    agents (int): number of agents
    items (int): number of items
    distribution (str): type of distribution for valuations
    model (str): model to use
    temperature (float): temperature for model
    prompt_type (str): type of prompt
    num_outputs (int): number of outputs to generate

    '''
    # 1 model init
    model = model_init(model, temperature)
    if model is None:
        print(f"Model {model} not found")
        return None
    logging.info(f"Model {model} initialized")

    allocation_matrices = np.zeros((num_outputs, agents, items))
    valuation_tables = np.zeros((num_outputs, agents, items))

    # create batch request file jsonl

    # find the last output number generated with glob
    import glob
    try:
        last_output = max([int(file.split('_')[-1].split('.')[0]) for file in glob.glob(f"{path}/output_*.txt")])
    except ValueError:
        last_output = 0


    for i in range(last_output, num_outputs):
        logging.info(f"Starting output {i+1}...")
        # 1 generate the valuations
        table = pd.DataFrame(generate_valuations(agents, items, scale=100))
        valuation_tables[i] = table.values
        # 2 generate the prompts
        input = generate_prompts(agents, items, table, prompt_type=prompt_type)


        # 3 generate the outputs
        query_model(agents, items, input, model, path = f"{path}/output_{i+1}.txt", type_of_dist=distribution, prompt_type=prompt_type, valuation=table.values)

        # 4 extract the json
        

        logging.info(f"Output {i+1} generated")
    
    logging.info(f"Outputs generated for {agents} agents and {items} items with a {distribution} distribution using model {model} with temperature {temperature} and prompt type {prompt_type}...")

    # # extracting and evaluating the outputs
    # logging.info("Extracting and evaluating outputs...")
    
    # envy_matrix_clipped, envy_matrix_unclipped = calculate_envy(valuation_tables, allocation_matrices, agents, items)
    # logging.info("Envy calculated")
    return valuation_tables, allocation_matrices

    # evaluate_envy(envy_matrix_clipped, envy_matrix_unclipped, agents, items, distribution, f"{path}", num_outputs)
    # logging.info("Envy evaluated and saved")


from tqdm import tqdm
from rich.progress import track


def evaluate_outputs(agents: int, items: int, distribution: str='uniform', model: str = 'gpt4om', temperature: float = 0.7, prompt_type: str = 'zero_shot0', num_outputs: int = 10):
    path = f"outputs/agents_{agents}/items_{items}/{model}/{prompt_type}"

    # scrape the valuation tables and allocation matrices from each output.txt
    valuation_tables = np.zeros((num_outputs, agents, items))
    allocation_matrices = np.zeros((num_outputs, agents, items))
    number_of_possible_allocations = np.zeros(num_outputs)
    envy_free_count = np.zeros(num_outputs)
    max_nash_welfare = np.zeros(num_outputs)
    max_utilitarian_welfare = np.zeros(num_outputs)
    max_egalitarian_welfare = np.zeros(num_outputs)


    print(f"Generating all possible allocations for {agents} agents and {items} items...")

    allocations = generate_all_allocations(agents, items)

    # Generate all possible allocations
    print(f"This will create {agents ** items} different allocations.")

    for i in tqdm(range(num_outputs)):
        try:
            with open(f"{path}/output_{i+1}.txt", "r") as file:
                # get allocation matrix
                try:
                    allocation = extract_json(f"{path}/output_{i+1}.txt")
                except:
                    print(f"Error in json processing {i+1}")
                    continue

                allocation_matrix = np.zeros((agents, items))
                try:
                    for j in range(agents):
                        for item in allocation[j]:
                            allocation_matrix[j][item] = 1
                    allocation_matrices[i] = allocation_matrix
                except:
                    print(f"Error in creating allocation matrix {i+1}")
                    continue

                # get valuation table find where 'Valuation Table:' and read the next lines
                lines = file.readlines()
                start = lines.index('Valuation Table:\n')
                end = lines.index('Output:\n')
                valuation_table = lines[start+1:end]
                valuation_table = ''.join(valuation_table)
                valuation_table = np.fromstring(valuation_table.replace('[', '').replace(']', ''), sep=' ')
                valuation_table = valuation_table.reshape(agents, items)
                valuation_table = valuation_table.astype(int)
                valuation_tables[i] = valuation_table
                # except ValueError:
                #     print(f"Error in reading valuation table {i+1}")
                #     continue

                valuations = valuation_table

                env = calculate_all_envy(valuation_table * len(allocations), allocations, agents, items)
                env = np.array(env)
                env_count = np.sum(env)
                # print(f"Number of allocations: {len(allocations)}")
                # print(f"Number of allocations that are envy free: {env_count}")
                number_of_possible_allocations[i] = len(allocations)
                envy_free_count[i] = env_count

                # get the max welfare (nash, utilitarian, egalitarian) of the allocations
                welfare = calculate_utilities(allocations, valuation_table)


                # get the max welfare (nash, utilitarian, egalitarian) of the allocations



                max_nash_welfare[i] = get_max_nash_welfare(welfare)
                max_utilitarian_welfare[i] = get_max_utilitarian_welfare(welfare)
                max_egalitarian_welfare[i] = get_max_egalitarian_welfare(welfare)
                
                

                
        except FileNotFoundError:
            print(f"Output {i+1} not found")
            break

    # create log file
    # with open(f"{path}/log.txt", "w") as file:
    #     file.write("")
    logging.basicConfig(filename=f"{path}/log.txt", level=logging.INFO)

    logging.info(f"Evaluating outputs for {agents} agents and {items} items with a {distribution} distribution using model {model} with temperature {temperature} and prompt type {prompt_type}...")

    # extracting and evaluating the outputs
    logging.info("Extracting and evaluating outputs...")
    
    envy_matrix_clipped, envy_matrix_unclipped, nash_welfare, utilitarian_welfare, egalitarian_welfare = calculate_envy(valuation_tables, allocation_matrices, agents, items)
    envy_df = pd.read_csv(f"{path}/envy_output.csv")
    # logging.info("Envy calculated")

    evaluate_envy(envy_matrix_clipped, envy_matrix_unclipped, agents, items, distribution, f"{path}", num_outputs, number_of_possible_allocations, envy_free_count , max_nash_welfare, max_utilitarian_welfare, max_egalitarian_welfare, nash_welfare, utilitarian_welfare, egalitarian_welfare)
    logging.info("Envy evaluated and saved")

