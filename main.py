from generate_outputs import generate_outputs, evaluate_outputs
import sys

# grab arguments

#
if len(sys.argv) < 8:
    print("Usage: python main.py <agents> <items> <distribution> <model> <temperature> <prompt_type> <num_outputs>")
    # python main.py 3 5 uniform gpt4o 0.7 zero_shot1 1000
    sys.exit(1)

agents = int(sys.argv[1])
items = int(sys.argv[2])
distribution = sys.argv[3]
model = sys.argv[4]
temperature = float(sys.argv[5])
prompt_type = sys.argv[6]
num_outputs = int(sys.argv[7])

valuation_tables, allocation_matrices = generate_outputs(agents=agents, items=items, distribution=distribution, model=model, temperature=temperature, prompt_type=prompt_type, num_outputs=num_outputs)
# evaluate_outputs(agents=agents, items=items, distribution=distribution, model=model, temperature=temperature, prompt_type=prompt_type, num_outputs=num_outputs, valuation_tables=valuation_tables, allocation_matrices=allocation_matrices)