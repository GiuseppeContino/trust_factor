import gymnasium as gym
import tqdm

import Trust
import Episodes_loop
import Environment_data

from matplotlib import pyplot as plt
import numpy as np
from gymnasium.envs.registration import register

epochs = 1500
max_episode_steps = 600
test_num = 1

# Register the environment
register(
    id='GridWorld-v0',
    entry_point='Environment:GridWorldEnv',
    max_episode_steps=max_episode_steps,
)

events_dict = {elem: idx for idx, elem in enumerate(Environment_data.events)}

q_tables = np.zeros((
    len(Environment_data.agents),
    8,  # # of states of the RM
    len(events_dict),
    Environment_data.size * Environment_data.size,
    len(Environment_data.actions),
))

trust = Trust.Trust(
    len(Environment_data.agents),
    events_dict,
)

train_env = gym.make(
    id='GridWorld-v0',
    # render_mode='human',
    training=True,
    train_transition=Episodes_loop.train_transition,
    tasks_trust=trust,
    dict_event_to_state=events_dict,
)

valid_env = gym.make(
    id='GridWorld-v0',
    # render_mode='human',
    tasks_trust=trust,
    dict_event_to_state=events_dict,
)

show_env = gym.make(
    id='GridWorld-v0',
    render_mode='human',
    tasks_trust=trust,
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
        )
        steps_list.append(epoch_step)

# plot the # of step during evaluation
plt.plot(steps_list)
plt.xlabel('Epochs')
plt.ylabel('# of steps')
plt.show()

fig = plt.figure(figsize=(12, 8))

fig.legend(loc='upper right')

ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.title.set_text((Environment_data.events + ['target_1'])[0] + ' trust')
ax2.title.set_text((Environment_data.events + ['target_1'])[1] + ' trust')
ax3.title.set_text((Environment_data.events + ['target_1'])[2] + ' trust')
ax4.title.set_text((Environment_data.events + ['target_1'])[3] + ' trust')
ax5.title.set_text((Environment_data.events + ['target_1'])[4] + ' trust')
ax6.title.set_text((Environment_data.events + ['target_1'])[5] + ' trust')

ax1.set(xlabel='Epochs', ylabel='Trust')
ax2.set(xlabel='Epochs', ylabel='Trust')
ax3.set(xlabel='Epochs', ylabel='Trust')
ax4.set(xlabel='Epochs', ylabel='Trust')
ax5.set(xlabel='Epochs', ylabel='Trust')
ax6.set(xlabel='Epochs', ylabel='Trust')

ax1.set_ylim([-0.15, 1.15])
ax2.set_ylim([-0.15, 1.15])
ax3.set_ylim([-0.15, 1.15])
ax4.set_ylim([-0.15, 1.15])
ax5.set_ylim([-0.15, 1.15])
ax6.set_ylim([-0.15, 1.15])

ax1.plot(trust_lists[0][0], 'y', label='agent_1')
# ax1.plot(trust_lists[1][0], 'm', label='agent_2')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

# ax2.plot(trust_lists[0][1], 'y', label='agent_1')
ax2.plot(trust_lists[1][1], 'm', label='agent_2')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

# ax3.plot(trust_lists[0][2], 'y', label='agent_1')
ax3.plot(trust_lists[1][2], 'm', label='agent_2')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

ax4.plot(trust_lists[0][3], 'y', label='agent_1')
ax4.plot(trust_lists[1][3], 'm', label='agent_2')
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

ax5.plot(trust_lists[0][4], 'y', label='agent_1')
ax5.plot(trust_lists[1][4], 'm', label='agent_2')
ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

ax6.plot(trust_lists[0][5], 'y', label='agent_1')
# ax6.plot(trust_lists[1][5], 'm', label='agent_2')
ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

plt.ylim(-0.15, 1.15)
fig.tight_layout(pad=0.75)
plt.show()

# show the result (pass to a not trainer environment and to a full greedy policy)
# set the value for show after the training without the trust
obs, _ = show_env.reset()

# start the steps loop
for step in tqdm.tqdm(range(max_episode_steps)):

    obs, rew, term, _ = train_wrapper.env_step(
        show_env,
        obs,
        q_tables,
    )

    train_wrapper.update_agents_states(show_env)

    if np.all(list(term.values())):
        break
