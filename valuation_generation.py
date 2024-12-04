import numpy as np

def generate_valuations(num_agents: int, num_items: int, scale=1.0, type='uniform') -> np.ndarray:
    """
    Generate valuations for agents for items
    :param num_agents: Number of agents
    :param num_items: Number of items
    :param scale: Scale of the valuation
    :param type: Type of distribution to use
    :return: Valuation matrix
    """
    if type == 'uniform':
        return (np.random.uniform(0, 1, (num_agents, num_items)) * scale).astype(int)
    elif type == 'exponential':
        return (np.random.exponential(scale, (num_agents, num_items))).astype(int)
    elif type == 'lognormal':
        return (np.random.lognormal(0, 1, (num_agents, num_items)) * scale).astype(int)
    else:
        raise ValueError('Invalid type')
    