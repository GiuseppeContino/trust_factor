import numpy as np


starting_n_value = 7  # 1


def return_updated_trust(actual_value, new_element, num):
    return (num * actual_value + new_element) / (num + 1), num + 1


class Trust:

    def __init__(self, num_agent, events_dict, start_value=0.5):
        self.events_dict = events_dict
        self.agents_trust = np.ones((num_agent, len(self.events_dict))) * start_value
        self.n_values = np.ones_like(self.agents_trust) * starting_n_value

    def reset(self, start_value=0.5):
        self.agents_trust = np.ones_like(self.agents_trust) * start_value
        self.n_values = np.ones_like(self.agents_trust)

    def reset_n_values(self):
        self.n_values = np.ones_like(self.agents_trust) * starting_n_value

    def get_agents_trust(self):
        return self.agents_trust

    def get_agent_trust(self, idx):
        return self.agents_trust[idx]

    def get_n_values(self):
        return self.n_values

    def get_n_value(self, idx):
        return self.n_values[idx]

    def update_trust(self, agent_idx, event, new_element):
        self.agents_trust[agent_idx][self.events_dict[event]] = (
                (self.n_values[agent_idx][self.events_dict[event]] *
                 self.agents_trust[agent_idx][self.events_dict[event]] + new_element) /
                (self.n_values[agent_idx][self.events_dict[event]] + 1)
        )
        self.n_values[agent_idx][self.events_dict[event]] += 1
