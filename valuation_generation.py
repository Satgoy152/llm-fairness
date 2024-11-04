import numpy as np

def generate_uniform_valuations(num_agents, num_items, scale=1.0):
    return (np.random.uniform(0, 1, (num_agents, num_items)) * scale).round(2)

def generate_exponential_valuations(num_agents, num_items, scale=1.0):
    return (np.random.exponential(scale, (num_agents, num_items))).round(2)

def generate_log_normal_valuations(num_agents, num_items, mean=0, sigma=1.0, scale=1.0):
    return (np.random.lognormal(mean, sigma, (num_agents, num_items)) * scale).round(2)