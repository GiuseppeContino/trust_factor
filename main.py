import gymnasium as gym
import tqdm

import Trust
import Episodes_loop

from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from gymnasium.envs.registration import register

epochs = 1500
max_episode_steps = 600
test_num = 1

env_str = 'Environment_1'
# env_str = 'Environment_2'
# env_str = 'Environment_3'

match env_str:
    case 'Environment_1':
        import Environment_data_1 as Environment_data
        end_events = [
            Environment_data.agent_1_end,
            Environment_data.agent_2_end,
        ]
    case 'Environment_2':
        import Environment_data_2 as Environment_data
        end_events = [
            Environment_data.agent_1_end,
            Environment_data.agent_2_end,
        ]
    case 'Environment_3':
        import Environment_data_3 as Environment_data
        end_events = [
            Environment_data.agent_1_end,
            Environment_data.agent_2_end,
            Environment_data.agent_3_end,
        ]

# Register the environment
register(
    id='GridWorld-v0',
    entry_point=env_str+':GridWorldEnv',
    max_episode_steps=max_episode_steps,
)

events_dict = {elem: idx for idx, elem in enumerate(Environment_data.events)}
# print(events_dict)

q_tables = np.zeros((
    len(Environment_data.agents),
    10,  # # of states of the RM
    len(events_dict),
    Environment_data.size * Environment_data.size,
    len(Environment_data.actions),
))

trust = Trust.Trust(
    len(Environment_data.agents),
    events_dict,
)

arc_values = np.ones((
    len(Environment_data.agents),
    10,
    len(events_dict),
)) * 100

arc_waiting = np.ones_like(arc_values) * 100

train_env = gym.make(
    id='GridWorld-v0',
    # render_mode='human',
    training=True,
    train_transition=Episodes_loop.train_transition,
    tasks_trust=trust,
    tasks_value=arc_values,
    dict_event_to_state=events_dict,
)

valid_env = gym.make(
    id='GridWorld-v0',
    # render_mode='human',
    tasks_trust=trust,
    tasks_value=arc_values,
    dict_event_to_state=events_dict,
)

show_env = gym.make(
    id='GridWorld-v0',
    render_mode='human',
    tasks_trust=trust,
    tasks_value=arc_values,
    dict_event_to_state=events_dict,
)

# list for plotting
steps_list = []

train_wrapper = Episodes_loop.Training(
    Environment_data.size,
    Environment_data.agents,
    len(Environment_data.actions),
)

# start the trust lists for plotting
trust_lists = []
for agent_idx in range(len(Environment_data.agents)):
    trust_lists.append([])
    for _ in range(len(events_dict)):
        trust_lists[agent_idx].append([0.5])

# epoch training loop
for epoch in tqdm.tqdm(range(epochs)):

    train_wrapper.training_step(
        train_env,
        max_episode_steps,
        q_tables,
        arc_values,
        arc_waiting,
    )

    for agent_idx in range(len(Environment_data.agents)):
        for trust_idx in range(len(events_dict)):
            trust_lists[agent_idx][trust_idx].append(trust.agents_trust[agent_idx][trust_idx])

    # test training every test_num value
    if (epoch + 1) % test_num == 0:
        epoch_step = train_wrapper.validation_step(
            valid_env,
            max_episode_steps,
            q_tables,
            end_events,
            arc_values,
            arc_waiting,
        )
        steps_list.append(epoch_step)

# print(arc_values)
# print(arc_waiting)

for i, arc_value in enumerate(arc_values):
    for ii, elems in enumerate(arc_value):
        for iii, elem in enumerate(elems):
            if elem > arc_waiting[i][ii][iii]:
                arc_value[ii][iii] -= arc_waiting[i][ii][iii]
print(arc_values)

# plot the # of step during evaluation
fontsize = 16
y_lowess = sm.nonparametric.lowess(steps_list, range(len(steps_list)), frac=0.1)  # 30 % lowess smoothing
plt.plot(y_lowess[:, 0], y_lowess[:, 1])
plt.xlabel('Epochs', fontsize=fontsize)
plt.ylabel('# of steps', fontsize=fontsize)
plt.show()

plt.plot(steps_list)
plt.plot(y_lowess[:, 0], y_lowess[:, 1])
plt.xlabel('Epochs', fontsize=fontsize)
plt.ylabel('# of steps', fontsize=fontsize)
plt.show()

train_env.plot_trust(trust_lists)

# show the result (pass to a not trainer environment and to a full greedy policy)
# set the value for show after the training without the trust
obs, _ = show_env.reset()

# start the steps loop
for step in range(max_episode_steps):

    obs, rew, term, _ = train_wrapper.env_step(
        show_env,
        obs,
        q_tables,
        end_events,
    )

    print('agent_1:', show_env.agents[0].get_selected_task(), '  -  agent_2:', show_env.agents[1].get_selected_task())

    train_wrapper.update_agents_states(show_env, arc_values)

    if np.all(list(term.values())):
        print('end in # ', step + 1)
        break
