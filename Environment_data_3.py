import numpy as np
from os import path

import pythomata

# environment dimension
size = 10

# agent names and information
agents = ['agent_1', 'agent_2', 'agent_3']

agents_initial_location = [
    np.array((0, 0)),
    np.array((9, 1)),
    np.array((9, 4)),
]

# agents_prob = [1, 0.5]
# agents_prob = [0.5, 1]
agents_prob = [1, 1, 1]

# actions names
actions = ['up', 'right', 'down', 'left', 'push_button', 'open_pocket_door',]

# events that has to occur for complete the task
events = ['door_1', 'door_2', 'door_3', 'door_4', 'red', 'blue', 'green', 'target_1', 'target_2', 'target_3',]

# important event for each agent
agent_1_events = ['door_1', 'red', 'green', 'target_1',]
agent_2_events = ['door_2', 'door_4', 'red', 'blue', 'green', 'target_2']
agent_3_events = ['door_3', 'door_4', 'blue', 'green', 'target_3',]

agent_1_end = 'target_1'
agent_2_end = 'target_2'
agent_3_end = 'target_3'

# rewards
complete_reward = 1.0
subtask_reward = 1.0
action_reward = 0.0
impossible_reward = 0.0

# environment data
targets_location = [
    np.array((5, 9)),
    np.array((5, 6)),
    np.array((3, 0)),
]

# walls location
walls = (
    # First vertical wall
    ((2, 0), (3, 0)),
    ((2, 1), (3, 1)),
    ((2, 2), (3, 2)),
    ((2, 3), (3, 3)),
    ((2, 4), (3, 4)),
    ((2, 5), (3, 5)),
    ((2, 6), (3, 6)),
    # First horizontal wall
    ((3, 6), (3, 7)),
    ((4, 6), (4, 7)),
    ((5, 6), (5, 7)),
    ((6, 6), (6, 7)),
    ((7, 6), (7, 7)),
    ((9, 6), (9, 7)),
    # Second vertical wall
    ((5, 0), (6, 0)),
    ((5, 1), (6, 1)),
    ((5, 6), (6, 6)),
    ((5, 7), (6, 7)),
    ((5, 8), (6, 8)),
    ((5, 9), (6, 9)),
    # Second horizontal wall
    ((6, 2), (6, 3)),
    ((7, 2), (7, 3)),
    ((8, 2), (8, 3)),
    ((9, 2), (9, 3)),
    # Third horizontal wall
    ((3, 1), (3, 2)),
    ((5, 1), (5, 2)),
    # Fourth vertical wall
    ((7, 0), (8, 0)),
    ((7, 2), (8, 2)),
    # Second horizontal wall
    ((0, 4), (0, 5)),
    ((2, 4), (2, 5)),
)

# Doors and relative button location
doors_location = [
    [np.array((5, 2))],
    [np.array((5, 3)), np.array((5, 4)), np.array((5, 5))],
    [np.array((3, 7)), np.array((3, 8)), np.array((3, 9))],
]
doors_button = [
    [np.array((2, 0)), ],
    [np.array((3, 6)), ],
    [np.array((5, 0)), ],
]

# Pocket doors and relative opening position
pocket_doors_location = [
    np.array((1, 5)),
    np.array((7, 1)),
    np.array((8, 7)),
    np.array((4, 1)),
]
pocket_doors_opener_position = [
    np.array((1, 4)),
    np.array((8, 1)),
    np.array((8, 6)),
    np.array((4, 2)),
]

# how many agents are needed to complete the open door task
doors_opener = [1, 1, 1]

# state of the door, 1 for closed, 0 for open
# both doors and pocket doors start closed
initial_doors_flag = np.ones((len(doors_location)))
initial_pocket_doors_flag = np.ones((len(pocket_doors_opener_position)))

# image for better drawing
red_door = path.join(path.dirname(__file__), 'images/red_door.png')
blue_door = path.join(path.dirname(__file__), 'images/blue_door.png')
green_door = path.join(path.dirname(__file__), 'images/green_door.png')

doors_img_paths = [red_door, blue_door, green_door]

# images paths
red_button = path.join(path.dirname(__file__), 'images/red_button.png')
blue_button = path.join(path.dirname(__file__), 'images/blue_button.png')
green_button = path.join(path.dirname(__file__), 'images/green_button.png')

buttons_img_paths = [red_button, blue_button, green_button]

agent_def_path = path.join(path.dirname(__file__), 'images/agent_def.png')
agent_red_path = path.join(path.dirname(__file__), 'images/agent_red.png')
agent_blue_path = path.join(path.dirname(__file__), 'images/agent_blue.png')
agent_green_path = path.join(path.dirname(__file__), 'images/agent_green.png')
agent_one_path = path.join(path.dirname(__file__), 'images/agent_one.png')
agent_two_path = path.join(path.dirname(__file__), 'images/agent_two.png')
agent_three_path = path.join(path.dirname(__file__), 'images/agent_three.png')
agent_four_path = path.join(path.dirname(__file__), 'images/agent_four.png')
agent_target_path = path.join(path.dirname(__file__), 'images/agent_target.png')

open_door_1_path = path.join(path.dirname(__file__), 'images/open_door_1.png')
closed_door_1_path = path.join(path.dirname(__file__), 'images/closed_door_1.png')
open_door_2_path = path.join(path.dirname(__file__), 'images/open_door_2.png')
closed_door_2_path = path.join(path.dirname(__file__), 'images/closed_door_2.png')
open_door_3_path = path.join(path.dirname(__file__), 'images/open_door_3.png')
closed_door_3_path = path.join(path.dirname(__file__), 'images/closed_door_3.png')
open_door_4_path = path.join(path.dirname(__file__), 'images/open_door_4.png')
closed_door_4_path = path.join(path.dirname(__file__), 'images/closed_door_4.png')


def create_first_individual_rm():

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

    automaton.add_transition((state_0, 'red', state_1))
    automaton.add_transition((state_0, 'door_1', state_2))

    automaton.add_transition((state_1, 'green & ~ door_1', state_4))
    automaton.add_transition((state_1, 'door_1 & green', state_5))
    automaton.add_transition((state_1, 'door_1 & ~ green', state_3))

    automaton.add_transition((state_2, 'red', state_3))

    automaton.add_transition((state_3, 'green', state_5))

    automaton.add_transition((state_4, 'door_1', state_5))

    automaton.add_transition((state_5, 'target_1', state_6))

    automaton.add_transition((state_6, 'target_1', state_6))

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
    state_7 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_7, True)

    automaton.add_transition((state_0, 'door_2 & ~ red', state_1))
    automaton.add_transition((state_0, 'red & ~ door_2', state_2))
    automaton.add_transition((state_0, 'red & door_2', state_3))

    automaton.add_transition((state_1, 'red', state_3))

    automaton.add_transition((state_2, 'door_2', state_3))

    automaton.add_transition((state_3, 'blue', state_4))

    automaton.add_transition((state_4, 'door_4 & ~ target_2', state_5))
    automaton.add_transition((state_4, 'target_2', state_7))

    automaton.add_transition((state_5, 'green & ~ target_2', state_6))
    automaton.add_transition((state_5, 'target_2', state_7))

    automaton.add_transition((state_6, 'target_2', state_7))

    automaton.add_transition((state_7, 'door_4 | green | target_2', state_7))

    return automaton


def create_third_individual_rm():

    automaton = pythomata.impl.symbolic.SymbolicDFA()

    state_0 = automaton.create_state()
    state_1 = automaton.create_state()
    state_2 = automaton.create_state()
    state_3 = automaton.create_state()
    state_4 = automaton.create_state()
    state_5 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_5, True)

    automaton.add_transition((state_0, 'door_3 & ~ blue', state_2))
    automaton.add_transition((state_0, 'blue', state_1))

    automaton.add_transition((state_1, 'door_4', state_3))

    automaton.add_transition((state_2, 'blue', state_1))

    automaton.add_transition((state_3, 'green & ~ target_3', state_4))
    automaton.add_transition((state_3, 'target_3', state_5))

    automaton.add_transition((state_4, 'target_3', state_5))

    automaton.add_transition((state_5, 'green | target_3', state_5))

    return automaton
