import numpy as np
from os import path

import pythomata

# environment dimension
size = 10

# agent names and information
agents = ['agent_1', 'agent_2']

agents_initial_location = [
    np.array((0, 0)),
    np.array((7, 0)),
]

# agents_prob = [1, 0.5]
# agents_prob = [0.5, 1]
agents_prob = [1, 1]

# actions names
actions = ['up', 'right', 'down', 'left', 'push_button', 'open_pocket_door',]

# events that has to occur for complete the task
events = ['door_1', 'door_2', 'door_3', 'red', 'blue', 'target_1']

# important event for each agent
agent_1_events = ['door_1', 'red', 'blue', 'target_1',]
agent_2_events = ['door_2', 'red', 'door_3', 'blue',]

agent_1_end = 'target_1'
agent_2_end = 'blue'

# rewards
complete_reward = 1.0
subtask_reward = 1.0
action_reward = 0.0
impossible_reward = 0.0

# environment data
targets_location = [np.array((5, 9))]

# walls location
walls = (
    # First vertical wall
    ((2, 0), (3, 0)),
    ((2, 1), (3, 1)),
    ((2, 2), (3, 2)),
    ((2, 3), (3, 3)),
    ((2, 4), (3, 4)),
    ((2, 6), (3, 6)),
    ((2, 7), (3, 7)),
    ((2, 8), (3, 8)),
    ((2, 9), (3, 9)),
    # Second vertical wall
    ((5, 0), (6, 0)),
    ((5, 1), (6, 1)),
    ((5, 3), (6, 3)),
    ((5, 4), (6, 4)),
    ((5, 5), (6, 5)),
    ((5, 6), (6, 6)),
    ((5, 7), (6, 7)),
    ((5, 8), (6, 8)),
    ((5, 9), (6, 9)),
    # First horizontal wall
    ((0, 3), (0, 4)),
    ((2, 3), (2, 4)),
    ((3, 3), (3, 4)),
    ((4, 3), (4, 4)),
    ((5, 3), (5, 4)),
    ((6, 3), (6, 4)),
    ((8, 3), (8, 4)),
    ((9, 3), (9, 4)),
    # Second horizontal wall
    ((6, 5), (6, 6)),
    ((8, 5), (8, 6)),
    ((9, 5), (9, 6)),
    # third horizontal wall
    ((3, 6), (3, 7)),
    ((5, 6), (5, 7)),
    # forth horizontal wall
    ((1, 7), (1, 8)),
    ((2, 7), (2, 8)),
    # fifth horizontal wall
    ((0, 8), (0, 9)),
    ((1, 8), (1, 9)),
)

# Doors and relative button location
doors_location = [
    [np.array((7, 4))],
    [np.array((4, 7))],
]
doors_button = [
    [np.array((5, 4))],
    [np.array((0, 9)), np.array((6, 9)), ],
]

# Pocket doors and relative opening position
pocket_doors_location = [
    np.array((1, 4)),
    np.array((5, 2)),
    np.array((7, 6)),
]
pocket_doors_opener_position = [
    np.array((1, 3)),
    np.array((6, 2)),
    np.array((7, 5)),
]

# how many agents are needed to complete the open door task
doors_opener = [1, 1]

# state of the door, 1 for closed, 0 for open
# both doors and pocket doors start closed
initial_doors_flag = np.ones((len(doors_location)))
initial_pocket_doors_flag = np.ones((len(pocket_doors_opener_position)))

# Image for better drawing
red_door = path.join(path.dirname(__file__), 'images/red_door_1.png')
blue_door = path.join(path.dirname(__file__), 'images/blue_door_1.png')

doors_img_paths = [red_door, blue_door]

# images paths
red_button = path.join(path.dirname(__file__), 'images/red_button.png')
blue_button = path.join(path.dirname(__file__), 'images/blue_button.png')

agent_def_path = path.join(path.dirname(__file__), 'images/agent_def.png')
agent_red_path = path.join(path.dirname(__file__), 'images/agent_red.png')
agent_blue_path = path.join(path.dirname(__file__), 'images/agent_blue.png')
agent_one_path = path.join(path.dirname(__file__), 'images/agent_one.png')
agent_two_path = path.join(path.dirname(__file__), 'images/agent_two.png')
agent_three_path = path.join(path.dirname(__file__), 'images/agent_three.png')
agent_target_path = path.join(path.dirname(__file__), 'images/agent_target.png')

open_door_1_path = path.join(path.dirname(__file__), 'images/open_door_1.png')
closed_door_1_path = path.join(path.dirname(__file__), 'images/closed_door_1.png')
open_door_2_path = path.join(path.dirname(__file__), 'images/open_door_2.png')
closed_door_2_path = path.join(path.dirname(__file__), 'images/closed_door_2.png')
open_door_3_path = path.join(path.dirname(__file__), 'images/open_door_3.png')
closed_door_3_path = path.join(path.dirname(__file__), 'images/closed_door_3.png')

buttons_img_paths = [red_button, blue_button]


def create_first_individual_rm():

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_0 = automaton.create_state()
    state_1 = automaton.create_state()
    state_2 = automaton.create_state()
    state_3 = automaton.create_state()
    state_4 = automaton.create_state()
    state_5 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_5, True)

    automaton.add_transition((state_0, 'door_1', state_1))

    # automaton.add_transition((state_1, 'red & ~ blue', state_2))
    # automaton.add_transition((state_1, 'blue & ~ red', state_3))
    # automaton.add_transition((state_1, 'red & blue', state_4))

    automaton.add_transition((state_1, 'red', state_2))
    automaton.add_transition((state_1, 'blue', state_3))

    automaton.add_transition((state_2, 'blue', state_4))

    automaton.add_transition((state_3, 'red', state_4))
    automaton.add_transition((state_3, 'target_1', state_5))

    automaton.add_transition((state_4, 'target_1', state_5))

    return automaton


def create_second_individual_rm():

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_0 = automaton.create_state()
    state_1 = automaton.create_state()
    state_2 = automaton.create_state()
    state_3 = automaton.create_state()
    state_4 = automaton.create_state()
    state_5 = automaton.create_state()
    state_6 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_6, True)

    automaton.add_transition((state_0, 'door_2 & ~ red & ~ blue', state_1))
    automaton.add_transition((state_0, 'red & ~ door_2', state_2))
    automaton.add_transition((state_0, 'door_2 & red', state_3))

    automaton.add_transition((state_1, 'red', state_3))

    automaton.add_transition((state_2, 'door_2 & ~ blue', state_3))
    automaton.add_transition((state_2, 'door_3 & ~ blue', state_5))

    automaton.add_transition((state_3, 'door_3 & ~ blue', state_4))

    automaton.add_transition((state_5, 'door_2 & ~ blue', state_4))

    automaton.add_transition((state_0, 'blue', state_6))
    automaton.add_transition((state_1, 'blue', state_6))
    automaton.add_transition((state_2, 'blue', state_6))
    automaton.add_transition((state_3, 'blue', state_6))
    automaton.add_transition((state_4, 'blue', state_6))
    automaton.add_transition((state_5, 'blue', state_6))

    automaton.add_transition((state_6, 'red', state_6))

    return automaton
