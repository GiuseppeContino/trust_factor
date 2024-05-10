import numpy as np
from os import path

import pythomata

# environment dimension
size = 10

# agent names and information
agents = ['agent_1', 'agent_2']

agents_initial_location = [
    np.array((4, 0)),
    np.array((5, 0)),
]

# agents_prob = [1, 0.5]
# agents_prob = [0.5, 1]
agents_prob = [0.5, 1]

# actions names
actions = ['up', 'right', 'down', 'left', 'push_button', 'open_pocket_door',]

# events that has to occur for complete the task
events = ['door_1', 'door_2', 'red', 'blue', 'target_1', 'target_2',]

# important event for each agent
agent_1_events = ['door_1', 'door_2', 'red', 'blue', 'target_1',]
agent_2_events = ['door_1', 'door_2', 'red', 'blue', 'target_2',]

agent_1_end = 'target_1'
agent_2_end = 'target_2'

# rewards
complete_reward = 1.0
subtask_reward = 1.0
action_reward = 0.0
impossible_reward = 0.0

# environment data
targets_location = [np.array((4, 9)), np.array((5, 9))]

# walls location
walls = (
    # First vertical wall
    ((2, 0), (3, 0)),
    ((2, 1), (3, 1)),
    # ((2, 2), (3, 2)),
    ((2, 3), (3, 3)),
    ((2, 4), (3, 4)),
    ((2, 5), (3, 5)),
    ((2, 6), (3, 6)),
    ((2, 7), (3, 7)),
    ((2, 8), (3, 8)),
    ((2, 9), (3, 9)),
    # Second vertical wall
    ((6, 0), (7, 0)),
    ((6, 1), (7, 1)),
    # ((6, 2), (7, 2)),
    ((6, 3), (7, 3)),
    ((6, 4), (7, 4)),
    ((6, 5), (7, 5)),
    ((6, 6), (7, 6)),
    ((6, 7), (7, 7)),
    ((6, 8), (7, 8)),
    ((6, 9), (7, 9)),
)

# Doors and relative button location
doors_location = [
    [np.array((3, 5)), np.array((4, 5)), np.array((5, 5)), np.array((6, 5))],
    [np.array((3, 6)), np.array((4, 6)), np.array((5, 6)), np.array((6, 6))],
]
doors_button = [
    [np.array((1, 9)),],
    [np.array((8, 9)),],
]

# Pocket doors and relative opening position
pocket_doors_location = [
    np.array((2, 2)),
    np.array((7, 2)),
]
pocket_doors_opener_position = [
    np.array((3, 2)),
    np.array((6, 2)),
]

# how many agents are needed to complete the open door task
doors_opener = [1, 1]

# state of the door, 1 for closed, 0 for open
# both doors and pocket doors start closed
initial_doors_flag = np.ones((len(doors_location)))
initial_pocket_doors_flag = np.ones((len(pocket_doors_opener_position)))

# Image for better drawing
red_door = path.join(path.dirname(__file__), 'images/red_door.png')
blue_door = path.join(path.dirname(__file__), 'images/blue_door.png')

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
    state_6 = automaton.create_state()
    state_7 = automaton.create_state()
    state_8 = automaton.create_state()
    state_9 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_9, True)

    automaton.add_transition((state_0, 'door_1 & ~ door_2', state_1))
    automaton.add_transition((state_0, 'door_2 & ~ door_1', state_2))
    automaton.add_transition((state_0, 'door_1 & door_2', state_4))

    automaton.add_transition((state_1, 'red & ~ door_2', state_3))
    automaton.add_transition((state_1, 'door_2 & ~ red', state_4))
    automaton.add_transition((state_1, 'door_2 & red', state_6))

    automaton.add_transition((state_2, 'door_1 & ~ blue', state_4))
    automaton.add_transition((state_2, 'blue & ~ door_1', state_5))
    automaton.add_transition((state_2, 'door_1 & blue', state_7))

    automaton.add_transition((state_3, 'door_2', state_6))

    automaton.add_transition((state_4, 'red & ~ blue', state_6))
    automaton.add_transition((state_4, 'blue & ~ red', state_7))
    automaton.add_transition((state_4, 'red & blue', state_8))

    automaton.add_transition((state_5, 'door_1', state_7))

    automaton.add_transition((state_6, 'blue', state_8))

    automaton.add_transition((state_7, 'red', state_8))

    automaton.add_transition((state_8, 'target_1', state_9))

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
    state_8 = automaton.create_state()
    state_9 = automaton.create_state()

    automaton.set_initial_state(state_0)
    automaton.set_accepting_state(state_9, True)

    automaton.add_transition((state_0, 'door_1 & ~ door_2', state_1))
    automaton.add_transition((state_0, 'door_2 & ~ door_1', state_2))
    automaton.add_transition((state_0, 'door_1 & door_2', state_4))

    automaton.add_transition((state_1, 'red & ~ door_2', state_3))
    automaton.add_transition((state_1, 'door_2 & ~ red', state_4))
    automaton.add_transition((state_1, 'door_2 & red', state_6))

    automaton.add_transition((state_2, 'door_1 & ~ blue', state_4))
    automaton.add_transition((state_2, 'blue & ~ door_1', state_5))
    automaton.add_transition((state_2, 'door_1 & blue', state_7))

    automaton.add_transition((state_3, 'door_2', state_6))

    automaton.add_transition((state_4, 'red & ~ blue', state_6))
    automaton.add_transition((state_4, 'blue & ~ red', state_7))
    automaton.add_transition((state_4, 'red & blue', state_8))

    automaton.add_transition((state_5, 'door_1', state_7))

    automaton.add_transition((state_6, 'blue', state_8))

    automaton.add_transition((state_7, 'red', state_8))

    automaton.add_transition((state_8, 'target_2', state_9))

    return automaton
