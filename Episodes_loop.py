import numpy as np
import copy

import Policy

epsilon = 0.4

learning_rate = 0.7  # 0.8  # 0.7
gamma = 0.9  # 0.9  # 0.85
alpha = 0.9

train_transition = 0.98  # high valuer means less environment transitions


class Training:

    def __init__(self, env_size, agent_names, max_action):

        self.env_size = env_size
        self.agent_names = agent_names
        self.n_agents = len(agent_names)
        self.max_action = max_action

        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def env_step(self, environment, obs, q_tables, agent_states):

        state_1 = obs['agent_1'][1] * self.env_size + obs['agent_1'][0]
        state_2 = obs['agent_2'][1] * self.env_size + obs['agent_2'][0]

        actions = [
            Policy.greedy_policy(q_tables[0][agent_states[0]], state_1),
            Policy.greedy_policy(q_tables[1][agent_states[1]], state_2),
        ]

        if agent_states[0] == 6:  # 6 is the final state
            actions[0] = 6  # 6 action mean no action
        if agent_states[1] == 6:
            actions[1] = 6

        actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

        # Perform the environment step
        obs, rew, term, _, _ = environment.step(actions_dict)

        return obs, rew, term, actions

    @staticmethod
    def manual_env_step(environment):

        # select manually the agents action
        actions = []
        for i in range(2):
            ele = int(input())
            actions.append(ele)

        actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

        # Perform the environment step
        obs, rew, term, _, _ = environment.step(actions_dict)

        return obs, rew, term, actions

    def simple_train(self, q_tables, agent_idx, agent_state, state, new_state, actions, rew):

        # save the actual q_value and max q_value in the state for simplify the writing
        actual_q_value = q_tables[agent_idx][agent_state][state][actions[agent_idx]]

        # add for real RM learning
        max_near_q_value = np.max(q_tables[agent_idx][agent_state][new_state])

        # update the agent q_table
        q_tables[agent_idx][agent_state][state][actions[agent_idx]] = (
            min((actual_q_value + self.lr * (rew['agent_' + str(agent_idx + 1)] +
                                             self.gamma * max_near_q_value - actual_q_value)
                 ), 1))

    def dont_skip_random_train(self, environment, q_tables, agent_idx, agent_state, state, new_state, actions, rew):

        if not environment.unwrapped.get_random_event():
            new_agent_state = copy.copy(environment.unwrapped.get_next_flags()[agent_idx])
        else:
            new_agent_state = copy.copy(agent_state)

        # save the actual q_value and max q_value in the state for simplify the writing
        actual_q_value = q_tables[agent_idx][agent_state][state][actions[agent_idx]]

        # add for real RM learning
        max_near_q_value = np.max(q_tables[agent_idx][new_agent_state][new_state])

        # update the agent q_table
        q_tables[agent_idx][agent_state][state][actions[agent_idx]] = (
            min((actual_q_value + self.lr * (rew['agent_' + str(agent_idx + 1)] +
                                             self.gamma * max_near_q_value - actual_q_value)
                 ), 1))

    def skip_random_train(self, environment, q_tables, agent_idx, agent_state, state, new_state, actions, rew):

        new_agent_state = copy.copy(environment.unwrapped.get_next_flags()[agent_idx])

        # save the actual q_value and max q_value in the state for simplify the writing
        actual_q_value = q_tables[agent_idx][agent_state][state][actions[agent_idx]]

        # add for real RM learning
        max_near_q_value = np.max(q_tables[agent_idx][new_agent_state][new_state])

        # skip
        if not environment.unwrapped.get_random_event():

            # update the agent q_table
            q_tables[agent_idx][agent_state][state][actions[agent_idx]] = (
                min((actual_q_value + self.lr * (rew['agent_' + str(agent_idx + 1)] +
                                                 self.gamma * max_near_q_value - actual_q_value)
                     ), 1))

    def training_step(self, environment, max_steps, q_tables, ):

        # train loop for the agents
        for agent_idx, agent in enumerate(self.agent_names):

            # reset the environment for the single agent training
            obs, _ = environment.reset()
            agent_state = copy.copy(environment.unwrapped.get_next_flags()[agent_idx])

            # single agent train
            for _ in range(max_steps):

                # get the old state and clean the actions array
                state = obs[agent][1] * self.env_size + obs[agent][0]
                actions = []

                # set the prev agents to do nothing
                for elem in range(agent_idx):
                    actions.append(self.max_action)

                # compute the agent action
                actions.append(
                    Policy.epsilon_greedy_policy(
                        environment,
                        q_tables[agent_idx][agent_state],
                        state,
                        self.epsilon,
                    )
                )

                # set the next agents to do nothing
                for elem in range(self.n_agents - agent_idx - 1):
                    actions.append(self.max_action)

                # transform the list in a dictionary
                actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

                # Perform the environment step
                obs, rew, term, _, _ = environment.step(actions_dict)

                # compute the new state
                new_state = obs[agent][1] * self.env_size + obs[agent][0]

                self.simple_train(q_tables, agent_idx, agent_state, state, new_state, actions, rew)
                # self.skip_random_train(environment, q_tables, agent_idx, agent_state, state, new_state, actions, rew)
                # self.dont_skip_random_train(environment, q_tables, agent_idx, agent_state, state, new_state, actions, rew)

                # move up the agent_state to the next RM state
                agent_state = copy.copy(environment.unwrapped.get_next_flags()[agent_idx])

                # if the episode is terminated, break the loop
                if np.any(list(term.values())):
                    break

    def validation_step(self, environment, max_steps, q_tables, events_dict, trust):

        # set the value for the evaluation after the training step
        obs, _ = environment.reset()

        agent_states = copy.copy(environment.get_next_flags())

        # test policy with all agents
        for step in range(max_steps):

            # Perform the environment step
            obs, rew, term, actions = self.env_step(environment, obs, q_tables, agent_states)

            # update the trust
            for agent_idx in range(self.n_agents):

                # update the trust if an event is occurred
                if (list(rew.values())[agent_idx] and
                        not agent_states[agent_idx] == environment.unwrapped.get_next_flags()[agent_idx]):

                    events = environment.get_event()
                    common_events = list(set(events) & set(environment.agents[agent_idx].get_events()))

                    for common_event in common_events:

                        if not common_event == 'trust' and not common_event[:-1] == 'target_':

                            if (common_event == 'red' or common_event == 'blue') and actions[agent_idx] == 4:
                                trust.n_values[agent_idx][events_dict[common_event]] += 1
                                trust.agents_trust[agent_idx][events_dict[common_event]] = (
                                        self.alpha * trust.agents_trust[agent_idx][events_dict[common_event]] +
                                        (1 - self.alpha) * rew['agent_' + str(agent_idx + 1)]
                                )
                                break

                            elif common_event[:-1] == 'door_' and actions[agent_idx] == 5:
                                trust.n_values[agent_idx][events_dict[common_event]] += 1
                                trust.agents_trust[agent_idx][events_dict[common_event]] = (
                                        self.alpha * trust.agents_trust[agent_idx][events_dict[common_event]] +
                                        (1 - self.alpha) * rew['agent_' + str(agent_idx + 1)]
                                )
                                break

            agent_states = copy.copy(environment.get_next_flags())

            # if the episode is terminated, break the loop
            if np.all(list(term.values())):
                return step + 1

        return max_steps
