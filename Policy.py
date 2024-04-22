import math
import random


# select greedily an action, when more than one have the highest value, pick randomly between them
# with a probability of slipping
def greedy_policy(q_table, act_state, slip=0.98):

    temp_action = []
    action_value = - math.inf

    for action_idx in range(len(q_table[act_state])):
        if q_table[act_state][action_idx] > action_value:
            action_value = q_table[act_state][action_idx]
            temp_action = [action_idx]

        elif q_table[act_state][action_idx] == action_value:
            temp_action.append(action_idx)

    action = temp_action[random.randint(0, len(temp_action) - 1)]

    # 2% random slip instead of selected action
    if action < 4 and random.uniform(0, 1) > slip:
        action = random.randint(0, 3)

    return action


# pick greedily a % of the time, randomly the other, with a probability of slipping
def epsilon_greedy_policy(environment, q_table, act_state, epsilon):

    random_float = random.uniform(0, 1)

    if random_float > epsilon:
        action = greedy_policy(q_table, act_state)

    else:
        action = environment.action_space.sample()

    return action
