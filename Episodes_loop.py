import random

import numpy as np

import Policy

epsilon = 0.4

learning_rate = 0.7
gamma = 0.9
alpha = 0.9

train_transition = 0.99  # high valuer means less environment transitions
dummy_value = 1.0  # 0.98  # 1.0 or greater if you don't want dummy event


class Training:

    def __init__(
        self,
        env_size,
        agent_names,
        max_action,
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
        env,
        obs,
        q_tables,
        end_events,
    ):

        if self.n_agents == 2:

            state_1 = obs['agent_1'][1] * self.env_size + obs['agent_1'][0]
            state_2 = obs['agent_2'][1] * self.env_size + obs['agent_2'][0]

            actions = self.two_agents_actions(
                q_tables,
                env,
                [state_1, state_2]
            )

            # set to do nothing all the agents in their final state
            if end_events[0] not in env.next_event:
                actions[0] = self.max_action
            if end_events[1] not in env.next_event:
                actions[1] = self.max_action

        elif self.n_agents == 3:

            state_1 = obs['agent_1'][1] * self.env_size + obs['agent_1'][0]
            state_2 = obs['agent_2'][1] * self.env_size + obs['agent_2'][0]
            state_3 = obs['agent_3'][1] * self.env_size + obs['agent_3'][0]

            actions = self.three_agents_actions(
                q_tables,
                env,
                [state_1, state_2, state_3]
            )

            # set to do nothing all the agents in their final state
            if end_events[0] not in env.next_event:
                actions[0] = self.max_action
            if end_events[1] not in env.next_event:
                actions[1] = self.max_action
            if end_events[2] not in env.next_event:
                actions[2] = self.max_action

        # generate action dict with all agents
        actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

        # Perform the environment step
        obs, rew, term, _, _ = env.step(actions_dict)

        return obs, rew, term, actions

    @staticmethod
    def two_agents_actions(
        q_table,
        env,
        state,
    ):

        actions = [
            Policy.greedy_policy(q_table[0][env.agents[0].get_state()][env.agents[0].get_selected_task_id()], state[0]),
            Policy.greedy_policy(q_table[1][env.agents[1].get_state()][env.agents[1].get_selected_task_id()], state[1]),
        ]

        return actions

    @staticmethod
    def three_agents_actions(
        q_table,
        env,
        state,
    ):

        actions = [
            Policy.greedy_policy(q_table[0][env.agents[0].get_state()][env.agents[0].get_selected_task_id()], state[0]),
            Policy.greedy_policy(q_table[1][env.agents[1].get_state()][env.agents[1].get_selected_task_id()], state[1]),
            Policy.greedy_policy(q_table[2][env.agents[2].get_state()][env.agents[2].get_selected_task_id()], state[2])
        ]

        return actions

    def manual_env_step(
        self,
        env,
    ):

        # select manually the agents action
        actions = []
        for i in range(self.n_agents):
            action = int(input())
            actions.append(action)

        # generate action dict with all agents
        actions_dict = {'agent_' + str(key + 1): value for key, value in enumerate(actions)}

        # perform the environment step
        obs, rew, term, _, _ = env.step(actions_dict)

        return obs, rew, term, actions

    def update_agents_states(
        self,
        env,
    ):

        for agent_idx in range(self.n_agents):
            env.agents[agent_idx].update_state(
                env.get_event(),
                env.training,
                env.tasks_trust.get_agent_trust(agent_idx),
                env.next_event,
            )

    def simple_train(
        self,
        env,
        q_tables,
        agent_idx,
        state,
        new_state,
        actions,
        rew,
    ):

        agent_state = env.agents[agent_idx].get_state()
        agent_task_selected = env.agents[agent_idx].get_selected_task_id()

        # save the actual q_value and max q_value in the state for simplify the writing
        actual_q_value = q_tables[agent_idx][agent_state][agent_task_selected][state][actions[agent_idx]]

        # add for real RM learning
        max_near_q_value = np.max(q_tables[agent_idx][agent_state][agent_task_selected][new_state])

        q_tables[agent_idx][agent_state][agent_task_selected][state][actions[agent_idx]] = (
                min((actual_q_value + self.lr * (rew['agent_' + str(agent_idx + 1)] +
                                                 self.gamma * max_near_q_value - actual_q_value)
                     ), 1))

        env.agents[agent_idx].update_state(
            env.get_event(),
            env.training,
            env.tasks_trust.get_agent_trust(agent_idx),
            env.next_event,
        )

    def training_step(
        self,
        env,
        max_steps,
        q_tables,
    ):

        # train loop for the agents
        for agent_idx, agent in enumerate(self.agent_names):

            # reset the environment for the single agent training
            obs, _ = env.reset()

            # single agent train
            for _ in range(max_steps):

                # get the old state and clean the actions array
                state = obs[agent][1] * self.env_size + obs[agent][0]
                actions = []

                # set the prev agents to do nothing
                for elem in range(agent_idx):
                    actions.append(self.max_action)

                # compute the agent action
                agent_state = env.agents[agent_idx].get_state()
                agent_selected_task = env.agents[agent_idx].get_selected_task_id()
                actions.append(
                    Policy.epsilon_greedy_policy(
                        env,
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
                obs, rew, term, _, _ = env.step(actions_dict)

                # compute the new state
                new_state = obs[agent][1] * self.env_size + obs[agent][0]

                # train only the agent with not the nothing action
                if not actions[agent_idx] == self.max_action:
                    self.simple_train(
                        env,
                        q_tables,
                        agent_idx,
                        state,
                        new_state,
                        actions,
                        rew,
                    )

                if env.is_random_event() and env.event[0] in env.agents[agent_idx].events:
                    env.tasks_trust.update_trust(
                        agent_idx,
                        env.agents[agent_idx].get_selected_task(),
                        0.0,
                    )

                # dump event for trust more near event
                if not env.event and random.uniform(0, 1) > dummy_value:
                    env.tasks_trust.update_trust(
                        agent_idx,
                        env.agents[agent_idx].get_selected_task(),
                        0.0,
                    )
                    env.agents[agent_idx].select_new_random_task(env.next_event)

                # if the episode is terminated, break the loop
                if list(term.values())[agent_idx]:
                    break

    def validation_step(
        self,
        env,
        max_steps,
        q_tables,
        end_events,
    ):

        # set the value for the evaluation after the training step
        obs, _ = env.reset()

        # test policy with all agents
        for step in range(max_steps):

            # Perform the environment step
            obs, rew, term, actions = self.env_step(env, obs, q_tables, end_events)

            # update the agents states
            self.update_agents_states(env)

            # if the episode is terminated, break the loop
            if np.all(list(term.values())):
                return step + 1

        return max_steps
