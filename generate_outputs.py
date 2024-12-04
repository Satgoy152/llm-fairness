from valuation_generation import generate_valuations
from prompt_generation import generate_prompts
from model_response import model_init, query_model, extract_json
from envy_freeness import calculate_envy, evaluate_envy
import numpy as np
import pandas as pd
import logging

def generate_outputs(agents: int, items: int, distribution: str='uniform', model: str = 'gpt4om', temperature: float = 0.7, prompt_type: str = 'zero-shot', num_outputs: int = 10):
    path = f"outputs/agents_{agents}/items_{items}/{model}/{prompt_type}"

    # create log file
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


    for i in range(num_outputs):
        logging.info(f"Starting output {i+1}...")
        # 1 generate the valuations
        table = pd.DataFrame(generate_valuations(agents, items, scale=100))
        valuation_tables[i] = table.values
        # 2 generate the prompts
        input = generate_prompts(agents, items, table, prompt_type=prompt_type)

        # 3 generate the outputs
        query_model(agents, items, input, model, path = f"{path}/output_{i+1}.txt", type_of_dist=distribution)

        # 4 extract the json
        allocation = extract_json(f"{path}/output_{i+1}.txt")

        # allocation_matrix = np.zeros((agents, items))
        for j in range(agents):
            for item in allocation[j]:
                allocation_matrices[i][j][item] = 1  # Mark allocated items as 1

        logging.info(f"Output {i+1} generated")
    
    logging.info(f"Outputs generated for {agents} agents and {items} items with a {distribution} distribution using model {model} with temperature {temperature} and prompt type {prompt_type}...")

    # extracting and evaluating the outputs
    logging.info("Extracting and evaluating outputs...")
    
    envy_matrix_clipped, envy_matrix_unclipped = calculate_envy(valuation_tables, allocation_matrices, agents, items)
    logging.info("Envy calculated")

    evaluate_envy(envy_matrix_clipped, envy_matrix_unclipped, agents, items, distribution, f"{path}", num_outputs)
    logging.info("Envy evaluated and saved")
    

    


    