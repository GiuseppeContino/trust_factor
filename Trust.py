import numpy as np


def update_trust(actual_value, new_element, num):
    return (num * actual_value + new_element) / (num + 1), num + 1


class Trust:

    def __init__(self, num_agent, num_event):
        self.agents_trust = np.zeros((num_agent, num_event))
        self.n_values = np.zeros_like(self.agents_trust)

    def get_agents_trust(self):
        return self.agents_trust

    def get_agent_trust(self, idx):
        return self.agents_trust[idx]

    def get_n_values(self):
        return self.n_values

    def get_n_value(self, idx):
        return self.n_values[idx]

    def update_trust(self, agent_idx, new_element):
        self.agents_trust[agent_idx] = (
                (self.n_values[agent_idx] * self.agents_trust[agent_idx] + new_element) / (self.n_values[agent_idx] + 1)
        )
        self.n_values[agent_idx] += 1
