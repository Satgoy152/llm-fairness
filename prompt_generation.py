def generate_prompts(agents: int, items: int, table, prompt_type: str = 'zero_shot0'):
    
    if prompt_type == 'zero_shot0':
        base_prompt = f"""
    The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.

    Present your allocations at the end in the following json format:
    {{
        "0": [0,1,2,...],
        "1": [3,4,...]
    }}
    Only give allocations for agents andi items in the table above.

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent. 
    The last think you give should be the allocations.
    """
    elif prompt_type == 'zero_shot1':
        base_prompt = f"""
        The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.  Only give allocations for agents and items in the table above.

    Present your allocations at the end in the following json format:
    {{
        "0": [0,1,2,...],
        "1": [3,4,...],
        "2": [5,6,...],
    }}

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent. 
    Even if an agent is assigned no items, include them in the json with an empty list.
    Make sure only {items} items are allocated in total.
    The last think you give should be the allocations.
    """
    elif prompt_type == 'zero_shot2':
        base_prompt = f"""
        The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.  Only give allocations for agents and items in the table above.

    Present your allocations at the end in the following json format:
    {{
        "0": [0,1,2,...],
        "1": [3,4,...],
        "2": [5,6,...],
        "3": [7,8,...],
    }}

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent.
    Even if an agent is assigned no items, include them in the json with an empty list.
    Make sure only {items} items are allocated in total.
    The last think you give should be the allocations.
    """
    elif prompt_type == 'zero_shot3':
        base_prompt = f"""
        The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.  Only give allocations for agents and items in the table above.

    Present your allocations at the end in the following json format:
    {{
        "0": [0,1,2,...],
        "1": [3,4,...],
        "2": [5,6,...],
        "3": [7,8,...],
        "4": [9,10,...]
    }}

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent.
    Even if an agent is assigned no items, include them in the json with an empty list.
    Make sure only and atleast {items} items are allocated in total.
    The last think you give should be the allocations.
    """
    elif prompt_type == 'zero_shot4':
        base_prompt = f"""
        The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.  Only give allocations for agents and items in the table above.

    Present your allocations at the end in the following json format:
    {{
        "0": [0,1,2,...],
        "1": [3,4,...],
        "2": [5,6,...],
        "3": [7,8,...],
        "4": [9,10,...],
        "5": [11,12,...]
    }}

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent.
    Even if an agent is assigned no items, include them in the json with an empty list.
    Make sure only and atleast {items} items are allocated in total.    
    The last thing you give should be the allocations.
    """
    elif prompt_type == 'zero_shot5':
        base_prompt = f"""
        The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.  Only give allocations for agents and items in the table above.

    Present your allocations at the end in the following json format:
    {{
        "0": [0,1,2,...],
        "1": [3,4,...],
        "2": [5,6,...],
        "3": [7,8,...],
        "4": [9,10,...],
        "5": [11,12,...],
        "6": [13,14,...]
    }}

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent.
    Even if an agent is assigned no items, include them in the json with an empty list.
    Make sure only and atleast {items} items are allocated in total.
    The last thing you give should be the allocations.
    """
    elif prompt_type == 'zero_shot6':
        base_prompt = f"""
        The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.  Only give allocations for agents and items in the table above.

    Present your allocations at the end in the following json format:
    {{
        "0": [0,1,2,...],
        "1": [3,4,...],
        "2": [5,6,...],
        "3": [7,8,...],
        "4": [9,10,...],
        "5": [11,12,...],
        "6": [13,14,...],
        "7": [15,16,...]
    }}

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent.
    Even if an agent is assigned no items, include them in the json with an empty list.
    Make sure only and atleast {items} items are allocated in total.

    The last thing you give should be the allocations.
    """
    elif prompt_type == 'persona_based0':
        base_prompt = f"""
        You are a computational economist with deep expertise in fair division and envy-freeness.

        The following table represents the valuations of {agents} agents numbered 0,1,2, and so on for {items} items numbered 0,1,2, and so on. For example, the value agent 1 has of item 2 is {table[2][1]}.

{table}

    Fairly allocate all the items to the agents so that each agent gets an integer number of items.  Only give allocations for agents and items in the table above.

    Present your allocations at the end in the following json format:

    {{
        "0": [0,1,2,...],
        "1": [3,4,...]
    }}

    Where the keys are the agent numbers and the values are lists of the items allocated to that agent.
    Even if an agent is assigned no items, include them in the json with an empty list.
    Make sure only and atleast {items} items are allocated in total.
    
    The last thing you give should be the allocations.
    """

    else:
        print("Prompt type not found")
        return None

    return base_prompt