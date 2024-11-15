def generate_prompts(agents: int, items: int, table):
    
    base_prompt = f"""
The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items.  numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

Fairly allocate all the items to the agents so that each agent gets an integer number of items.

Present your allocations at the end in the following json format:
{{
    "0": [0,1,2],
    "1": [3,4],
    "2": [5,6,7]
}}

Where the keys are the agent numbers and the values are lists of the items allocated to that agent.
"""

    return base_prompt