import random
from sympy import *
import numpy as np


class Agent:

    def __init__(
        self,
        position: list,
        temporal_goal,
        events,
        dict_event_to_state,
        training,
        tasks_trust,
        next_tasks,
    ):
        self.position = position
        self.temporal_goal = temporal_goal
        self.state = temporal_goal.current_state - 1
        self.events = events
        self.selected_task = 0
        self.dict_event_to_state = dict_event_to_state
        self.dict_state_to_event = {value: key for key, value in dict_event_to_state.items()}

        if training:
            self.select_new_random_task(next_tasks)
        else:
            self.select_new_trusted_task(tasks_trust, next_tasks)

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
        training,
        tasks_trust,
        next_tasks,
    ):

        common_events = list(
            set(event) & set(self.events)
        )
        if not common_events == []:
            # print(self.state)
            # print(event)
            # print(self.events)
            self.temporal_goal.step(common_events)
            self.state = self.temporal_goal.current_state - 1
            if training:
                self.select_new_random_task(next_tasks)
            else:
                self.select_new_trusted_task(tasks_trust, next_tasks)

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

    def get_selectionable_task(
        self,
        next_tasks,
    ):

        atoms = []
        for transition in self.temporal_goal.automaton.get_transitions_from(self.state + 1):
            for atom in str(sympify(transition[1])).replace(' ', '').split('&'):
                if not atom[0] == '~':
                    atoms.append(atom)
        return list(set(atoms) & set(next_tasks))

    def select_new_random_task(
        self,
        next_tasks,
    ):
        atoms = self.get_selectionable_task(next_tasks)
        if atoms:
            self.selected_task = self.dict_event_to_state[atoms[random.randint(0, len(atoms) - 1)]]

    def select_new_trusted_task(
        self,
        trust,
        next_tasks,
    ):
        atoms = self.get_selectionable_task(next_tasks)
        if atoms:
            atoms_state = [self.dict_event_to_state[item] for item in atoms if item in self.dict_event_to_state]
            atoms_trust = {item: trust[item] for item in atoms_state}
            # TODO: here I hypnotize all trust are different
            self.selected_task = list(atoms_trust.keys())[np.argmax(list(atoms_trust.values()))]

    def set_selected_task(
        self,
        new_selected_task,
    ):

        self.selected_task = new_selected_task

    def get_events(
        self,
    ):
        return self.events
