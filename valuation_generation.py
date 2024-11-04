import numpy as np

def generate_valuations(num_agents, num_items, scale=1.0, type='uniform'):
    if type == 'uniform':
        return (np.random.uniform(0, 1, (num_agents, num_items)) * scale).round(2)
    elif type == 'exponential':
        return (np.random.exponential(scale, (num_agents, num_items))).round(2)
    elif type == 'lognormal':
        return (np.random.lognormal(0, 1, (num_agents, num_items)) * scale).round(2)
    else:
        raise ValueError('Invalid type')
    