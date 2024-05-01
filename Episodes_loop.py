import numpy as np
import copy

import Policy

epsilon = 0.4

learning_rate = 0.7  # 0.8  # 0.7
gamma = 0.9  # 0.9  # 0.85
alpha = 0.9

train_transition = 0.99  # high valuer means less environment transitions


class Training:

    def __init__(
        self,
        env_size,
        agent_names,
        max_action
    ):

        self.env_size = env_size
        self.agent_names = agent_names
        self.n_agents = len(agent_names)
        self.max_action = max_action

        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def env_step(
        self,
        environment,
        obs,
        q_tables
    ):

        state_1 = obs['agent_1'][1] * self.env_size + obs['agent_1'][0]
        state_2 = obs['agent_2'][1] * self.env_size + obs['agent_2'][0]

        actions = [
            Policy.greedy_policy(q_tables[0][environment.agents[0].get_state()][environment.agents[0].get_selected_task_id()], state_1),
            Policy.greedy_policy(q_tables[1][environment.agents[1].get_state()][environment.agents[1].get_selected_task_id()], state_2),
        ]

        # set to do nothing all the agents in their final state
        if environment.agents[0].get_state() == 5:  # 6 is the final state
            actions[0] = 6  # 6 action mean no action
        if environment.agents[1].get_state() == 5:
            actions[1] = 6

        # generate action dict with all agents
        actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

        # Perform the environment step
        obs, rew, term, _, _ = environment.step(actions_dict)

        return obs, rew, term, actions

    @staticmethod
    def manual_env_step(
        environment,
    ):

        # select manually the agents action
        actions = []
        for i in range(2):
            action = int(input())
            actions.append(action)

        # generate action dict with all agents
        actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

        # perform the environment step
        obs, rew, term, _, _ = environment.step(actions_dict)

        return obs, rew, term, actions

    def update_agents_states(
        self,
        environment,
    ):

        for agent_idx in range(self.n_agents):
            environment.agents[agent_idx].update_state(environment.get_event())

    def simple_train(
        self,
        environment,
        q_tables,
        agent_idx,
        agent_state,
        state,
        new_state,
        actions,
        rew,
    ):

        # save the actual q_value and max q_value in the state for simplify the writing
        actual_q_value = q_tables[agent_idx][agent_state][state][actions[agent_idx]]

        # add for real RM learning
        max_near_q_value = np.max(q_tables[agent_idx][agent_state][new_state])

        # update the agent q_table
        q_tables[agent_idx][agent_state][state][actions[agent_idx]] = (
            min((actual_q_value + self.lr * (rew['agent_' + str(agent_idx + 1)] +
                                             self.gamma * max_near_q_value - actual_q_value)
                 ), 1))

        environment.agents[agent_idx].update_state(environment.get_event())

    def dont_skip_random_train(
        self,
        environment,
        q_tables,
        agent_idx,
        state,
        new_state,
        actions,
        rew,
    ):

        agent_state = copy.copy(environment.agents[agent_idx].get_state())
        agent_task_selected = copy.copy(environment.agents[agent_idx].get_selected_task_id())

        environment.agents[agent_idx].update_state(environment.get_event())
        new_agent_state = environment.agents[agent_idx].get_state()

        # save the actual q_value and max q_value in the state for simplify the writing
        actual_q_value = q_tables[agent_idx][agent_state][agent_task_selected][state][actions[agent_idx]]

        # add for real RM learning
        max_near_q_value = np.max(q_tables[agent_idx][new_agent_state][agent_task_selected][new_state])

        # update the agent q_table
        q_tables[agent_idx][agent_state][agent_task_selected][state][actions[agent_idx]] = (
            min((actual_q_value + self.lr * (rew['agent_' + str(agent_idx + 1)] +
                                             self.gamma * max_near_q_value - actual_q_value)
                 ), 1))

    def skip_random_train(
        self,
        environment,
        q_tables,
        agent_idx,
        state,
        new_state,
        actions,
        rew,
    ):

        agent_state = environment.agents[agent_idx].get_state()
        agent_task_selected = environment.agents[agent_idx].get_selected_task_id()

        # save the actual q_value and max q_value in the state for simplify the writing
        actual_q_value = q_tables[agent_idx][agent_state][agent_task_selected][state][actions[agent_idx]]

        # add for real RM learning
        max_near_q_value = np.max(q_tables[agent_idx][agent_state][agent_task_selected][new_state])

        # skip train if the event is generated by the environment
        if not environment.is_random_event():

            # update the agent q_table
            q_tables[agent_idx][agent_state][agent_task_selected][state][actions[agent_idx]] = (
                min((actual_q_value + self.lr * (rew['agent_' + str(agent_idx + 1)] +
                                                 self.gamma * max_near_q_value - actual_q_value)
                     ), 1))

        environment.agents[agent_idx].update_state(environment.get_event())

    def training_step(
        self,
        environment,
        max_steps,
        q_tables
    ):

        # train loop for the agents
        for agent_idx, agent in enumerate(self.agent_names):

            # reset the environment for the single agent training
            obs, _ = environment.reset()

            # single agent train
            for _ in range(max_steps):

                # get the old state and clean the actions array
                state = obs[agent][1] * self.env_size + obs[agent][0]
                actions = []

                # set the prev agents to do nothing
                for elem in range(agent_idx):
                    actions.append(self.max_action)

                # compute the agent action
                agent_state = environment.agents[agent_idx].get_state()
                agent_selected_task = environment.agents[agent_idx].get_selected_task_id()
                actions.append(
                    Policy.epsilon_greedy_policy(
                        environment,
                        q_tables[agent_idx][agent_state][agent_selected_task],
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

                # train only the agent with not the nothing action
                if not actions[agent_idx] == self.max_action:
                    # self.simple_train(
                    # self.dont_skip_random_train(
                    self.skip_random_train(
                        environment,
                        q_tables,
                        agent_idx,
                        state,
                        new_state,
                        actions,
                        rew,
                    )

                if environment.is_random_event() and environment.event[0] in environment.agents[agent_idx].events:
                    environment.tasks_trust.update_trust(
                        agent_idx,
                        environment.agents[agent_idx].get_selected_task(),
                        0.0,
                    )

                # if the episode is terminated, break the loop
                if np.any(list(term.values())):
                    break

    def validation_step(
        self,
        environment,
        max_steps,
        q_tables,
        events_dict
    ):

        # set the value for the evaluation after the training step
        obs, _ = environment.reset()

        # test policy with all agents
        for step in range(max_steps):

            # Perform the environment step
            obs, rew, term, actions = self.env_step(environment, obs, q_tables)

            # update the agents states
            self.update_agents_states(environment)

            # if the episode is terminated, break the loop
            if np.all(list(term.values())):
                return step + 1

        return max_steps
