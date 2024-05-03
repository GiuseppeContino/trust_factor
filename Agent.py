import random
from sympy import *


class Agent:

    def __init__(
        self,
        position: list,
        temporal_goal,
        events,
        dict_event_to_state,
    ):
        self.position = position
        self.temporal_goal = temporal_goal
        self.state = temporal_goal.current_state - 1
        self.events = events
        self.selected_task = 0
        self.dict_event_to_state = dict_event_to_state
        self.dict_state_to_event = {value: key for key, value in dict_event_to_state.items()}

        self.select_new_task()

    def get_position(
        self,
    ):
        return self.position

    def get_state(
        self,
    ):
        return self.state

    def update_state(
        self,
        event,
    ):

        common_events = list(
            set(event) & set(self.events)
        )
        if not common_events == []:
            self.temporal_goal.step(common_events)
            self.state = self.temporal_goal.current_state - 1
            self.select_new_task()

    def reset_temporal_goal(
        self,
    ):

        self.temporal_goal.reset()
        self.state = self.temporal_goal.current_state - 1

    def get_temporal_goal(
        self,
    ):
        return self.temporal_goal

    def get_selected_task_id(
        self,
    ):
        return self.selected_task

    def get_selected_task(
        self,
    ):
        return self.dict_state_to_event[self.selected_task]

    def select_new_task(
        self,
    ):

        atoms = []
        for transition in self.temporal_goal.automaton.get_transitions_from(self.state + 1):
            for atom in str(sympify(transition[1])).replace(' ', '').split('&'):
                if not atom[0] == '~':
                    atoms.append(atom)

        if atoms:
            self.selected_task = self.dict_event_to_state[atoms[random.randint(0, len(atoms) - 1)]]

    def set_selected_task(
        self,
        new_selected_task,
    ):

        self.selected_task = new_selected_task

    def get_events(
        self,
    ):
        return self.events
