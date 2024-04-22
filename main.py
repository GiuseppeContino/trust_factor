import gymnasium as gym
import tqdm

import Trust
import Episodes_loop
import Environment_data

import copy

from matplotlib import pyplot as plt
import numpy as np
from gymnasium.envs.registration import register

epochs = 600  # 600
max_episode_steps = 600  # 600
test_num = 1

# Register the environment
register(
    id='GridWorld-v0',
    entry_point='Environment:GridWorldEnv',
    max_episode_steps=max_episode_steps,
)

q_tables = [
    np.zeros((6, Environment_data.size * Environment_data.size, len(Environment_data.actions))),
    np.zeros((6, Environment_data.size * Environment_data.size, len(Environment_data.actions))),
]

events = ['red', 'blue', 'door_1', 'door_2', 'door_3']

events_dict = {elem: idx for idx, elem in enumerate(events)}

# train_env = gym.make('GridWorld-v0', render_mode='human', training=True,
#                      train_transition=Episodes_loop.train_transition)
train_env = gym.make('GridWorld-v0', training=True, train_transition=Episodes_loop.train_transition)
# valid_env = gym.make('GridWorld-v0', render_mode='human')
valid_env = gym.make('GridWorld-v0')

# list for plotting
steps_list = []

trust_list = []
for agent_idx in range(len(Environment_data.agents)):
    trust_list.append([])
    for _ in range((len(Environment_data.events) + 1)):
        trust_list[agent_idx].append([])

train_wrapper = Episodes_loop.Training(Environment_data.size, Environment_data.agents, len(Environment_data.actions))
trust = Trust.Trust(len(Environment_data.agents), len(Environment_data.events))

# epoch training loop
for epoch in tqdm.tqdm(range(epochs)):

    train_wrapper.training_step(train_env, max_episode_steps, q_tables)

    # test training every test_num value
    if epoch % test_num == 0:

        epoch_step = train_wrapper.validation_step(valid_env, max_episode_steps, q_tables, events_dict, trust)

        # update the trust for event that are not occurred
        for agent_idx in range(len(Environment_data.agents)):
            for trust_idx in range(len(trust.agents_trust[agent_idx])):
                if trust.n_values[agent_idx][trust_idx] < epoch + 1:
                    trust.n_values[agent_idx][trust_idx] += 1
                    trust.agents_trust[agent_idx][trust_idx] = (
                            Episodes_loop.alpha * trust.agents_trust[agent_idx][trust_idx] +
                            (1 - Episodes_loop.alpha) * 0
                    )

        steps_list.append(epoch_step)
        for agent_idx in range(len(Environment_data.agents)):
            for trust_idx in range(len(trust.agents_trust[agent_idx])):
                trust_list[agent_idx][trust_idx].append(trust.agents_trust[agent_idx][trust_idx])

# plot the # of step during evaluation
plt.plot(steps_list)
plt.xlabel('Epochs')
plt.ylabel('# of steps')
plt.show()

# TEST WITH ONE TRUST

# show the result (pass to a not trainer environment and to a full greedy policy)
show_env = gym.make('GridWorld-v0', render_mode='human')
# show_env = gym.make('GridWorld-v0', render_mode='human', events=Utilities.events,
#                     rewarding_machines=agents_pythomata_rm, task_trust=test_env.unwrapped.task_trust)

# set the value for show after the training without the trust
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

episodic_trust = [
    [show_env.unwrapped.task_trust[0]],
    [show_env.unwrapped.task_trust[1]],
]

agent_states = copy.copy(show_env.unwrapped.get_next_flags())

print(q_tables[0][0])
print('#' * 20)
print(q_tables[0][1])
print('#' * 20)
print(q_tables[0][2])
print('#' * 20)
print(q_tables[0][3])

# start the steps loop
for step in tqdm.tqdm(range(max_episode_steps)):

    print(agent_states)

    obs, rew, term, actions = train_wrapper.env_step(show_env, obs, q_tables, agent_states)

    total_rew += sum(list(rew.values()))
    total_step += 1

    agent_states = copy.copy(show_env.unwrapped.get_next_flags())

    episodic_trust[0].append(show_env.unwrapped.task_trust[0])
    episodic_trust[1].append(show_env.unwrapped.task_trust[1])

    if np.all(list(term.values())):
        break
