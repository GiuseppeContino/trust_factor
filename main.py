import gymnasium as gym
import tqdm

import Trust
import Episodes_loop
import Environment_data

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

events_dict = {elem: idx for idx, elem in enumerate(Environment_data.events + ['target_1'])}

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
    dict_event_to_state=events_dict,
)

# list for plotting
steps_list = []

train_wrapper = Episodes_loop.Training(
    Environment_data.size,
    Environment_data.agents,
    len(Environment_data.actions),
)

# epoch training loop
for epoch in tqdm.tqdm(range(epochs)):

    train_wrapper.training_step(
        train_env,
        max_episode_steps,
        q_tables,
    )

    # test training every test_num value
    if (epoch + 1) % test_num == 0:

        epoch_step = train_wrapper.validation_step(
            valid_env,
            max_episode_steps,
            q_tables,
            events_dict,
        )
        steps_list.append(epoch_step)

print(Environment_data.events + ['target_1'])
for trusts in trust.agents_trust:
    print('#' * 10)
    for trust in trusts:
        print(np.format_float_positional(np.float32(trust), unique=False, precision=6))

# plot the # of step during evaluation
plt.plot(steps_list)
plt.xlabel('Epochs')
plt.ylabel('# of steps')
plt.show()

# show the result (pass to a not trainer environment and to a full greedy policy)
# set the value for show after the training without the trust
obs, _ = show_env.reset()

total_rew = 0
total_step = 0

# start the steps loop
for step in tqdm.tqdm(range(max_episode_steps)):

    obs, rew, term, _ = train_wrapper.env_step(
        show_env,
        obs,
        q_tables,
    )

    train_wrapper.update_agents_states(show_env)

    total_rew += sum(list(rew.values()))
    total_step += 1

    if np.all(list(term.values())):
        break
