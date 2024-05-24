import random
import copy
from sympy import *
from matplotlib import pyplot as plt

import gymnasium as gym
from gymnasium import spaces
import pygame

import Agent
import Environment_data_2 as Environment_data

from temprl.reward_machines.automata import RewardAutomaton
from temprl.wrapper import TemporalGoal

import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        training=False,
        train_transition=0.0,
        tasks_trust=None,
        dict_event_to_state=None,
    ):

        self.size = Environment_data.size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.events = Environment_data.events
        self.training = training
        self.train_transition = train_transition

        self.event = []
        self.next_event = copy.copy(Environment_data.events)

        self.random_event = False

        self.tasks_trust = tasks_trust

        # agents initial position and colors
        agents_location = Environment_data.agents_initial_location

        self.agents = []

        rewarding_machines = [
            Environment_data.create_first_individual_rm(),
            Environment_data.create_second_individual_rm()
        ]

        # save the RM draw and create the agents
        for agent_idx, agent_events in enumerate([Environment_data.agent_1_events, Environment_data.agent_2_events]):

            agent_pythomata_rm = rewarding_machines[agent_idx]
            agent_graph = agent_pythomata_rm.to_graphviz()
            agent_graph.render('./images/agent_' + str(agent_idx + 1) + '_reward_machine')

            agent_automata = RewardAutomaton(agent_pythomata_rm, 1)
            agent_temp_goal = TemporalGoal(agent_automata)

            self.agents.append(Agent.Agent(
                agents_location[agent_idx],
                agent_temp_goal,
                agent_events,
                dict_event_to_state,
                training,
                self.next_event,
                tasks_trust.get_agent_trust(agent_idx),
            ))

        # observation space of the action
        self.observation_space = spaces.Dict(
            {
                'agent_1': spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                'agent_2': spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 6 actions, corresponding to 'right', 'up', 'left', 'down', 'push_button', 'open_pocket_door'
        self.action_space = spaces.Discrete(len(Environment_data.actions))

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Targets position
        self._targets_location = Environment_data.targets_location

        # Doors and relative button location and how many agents are needed for open a door
        self._doors_location = Environment_data.doors_location
        self._doors_button = Environment_data.doors_button
        self._doors_opener = Environment_data.doors_opener

        # initial door state
        self._doors_flag = Environment_data.initial_doors_flag

        # Pocket door and opening position location
        self._pocket_doors_location = Environment_data.pocket_doors_location
        self._pocket_doors_opener_position = Environment_data.pocket_doors_opener_position

        # initial pocket door state
        self._pocket_doors_flag = Environment_data.initial_pocket_doors_flag

        # image path of doors and buttons
        self._doors_img_paths = Environment_data.doors_img_paths
        self._buttons_img_paths = Environment_data.buttons_img_paths

        self._walls = Environment_data.walls

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def is_random_event(
        self,
    ):
        return self.random_event

    def get_event(
        self,
    ):
        return self.event

    def get_trust(
        self,
    ):
        return self.tasks_trust

    def _get_obs(
        self,
    ):
        return {
            'agent_1': self.agents[0].position,
            'agent_2': self.agents[1].position,
        }

    def _get_info(
        self,
    ):  # TODO: maybe to change to be more accurate
        return {
            'agent_1': [np.linalg.norm(self.agents[0].position - self._targets_location[0], ord=1)],
            'agent_2': [np.linalg.norm(self.agents[1].position - self._targets_location[0], ord=1)],
        }

    def reset(
        self,
        seed=None,
        options=None,
    ):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.next_event = copy.copy(Environment_data.events)
        self.random_event = False

        for agent_idx, agent in enumerate(self.agents):
            agent.reset_temporal_goal()
            if self.training:
                agent.select_new_random_task(self.next_event)
            else:
                agent.select_new_trusted_task(self.tasks_trust.get_agent_trust(agent_idx), self.next_event)
                # agent.select_new_random_task(self.next_event)

        # Reset agents position
        self.agents[0].position = Environment_data.agents_initial_location[0]
        self.agents[1].position = Environment_data.agents_initial_location[1]

        self.agents[0].events = Environment_data.agent_1_events
        self.agents[1].events = Environment_data.agent_2_events

        # Reset the doors flag to close all them
        self._doors_flag = copy.copy(Environment_data.initial_doors_flag)
        self._pocket_doors_flag = copy.copy(Environment_data.initial_pocket_doors_flag)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def agent_move(
        self,
        action,
        agent_idx,
    ):

        direction = self._action_to_direction[action]

        collision = False

        for doors, door_flag in zip(self._doors_location, self._doors_flag):
            for door in doors:
                if np.all(self.agents[agent_idx].position + direction == door) and door_flag == 1:
                    collision = True

        for pocket_door, pocket_door_flag in zip(self._pocket_doors_location, self._pocket_doors_flag):
            if np.all(self.agents[agent_idx].position + direction == pocket_door) and pocket_door_flag == 1:
                collision = True

        for wall in self._walls:
            if ((np.all(self.agents[agent_idx].position == wall[0]) and
                 np.all(self.agents[agent_idx].position + direction == wall[1])) or
                    (np.all(self.agents[agent_idx].position == wall[1]) and
                     np.all(self.agents[agent_idx].position + direction == wall[0]))):
                collision = True

        if (
                self.agents[agent_idx].position[0] + direction[0] > self.size - 1 or
                self.agents[agent_idx].position[1] + direction[1] > self.size - 1 or
                self.agents[agent_idx].position[0] + direction[0] < 0 or
                self.agents[agent_idx].position[1] + direction[1] < 0
        ):
            collision = True

        #  use np.clip to make sure we don't leave the grid
        if not collision:

            self.agents[agent_idx].position = np.clip(
                self.agents[agent_idx].position + direction, 0, self.size - 1
            )

            return Environment_data.action_reward

        else:
            return Environment_data.impossible_reward

    def open_door(
        self,
        agent_idx,
        door_idx,
        door_location,
        door_event,
        openers,
    ):

        if (np.all(self.agents[agent_idx].position == self._doors_button[door_idx][door_location]) and
                self._doors_flag[door_idx] == 1 and door_event in self.agents[agent_idx].events):

            openers[0] += 1
            if openers[0] >= self._doors_opener[0]:
                self.event.append(door_event)

                if self.tasks_trust and self.agents[agent_idx].get_selected_task() == door_event:
                    if self.training:
                        self.tasks_trust.update_trust(agent_idx, self.agents[agent_idx].get_selected_task(), 1.0)
                    return True
                elif self.tasks_trust and not self.agents[agent_idx].get_selected_task() == door_event:
                    if self.training:
                        self.tasks_trust.update_trust(agent_idx, self.agents[agent_idx].get_selected_task(), 0.0)

        return False

    def open_pocket_door(
        self,
        agent_idx,
        pocket_door_idx,
        pocket_door_event,
    ):

        random_float = random.uniform(0, 1)

        if (np.all(self.agents[agent_idx].position == self._pocket_doors_opener_position[pocket_door_idx]) and
                self._pocket_doors_flag[pocket_door_idx] == 1):

            if random_float < Environment_data.agents_prob[agent_idx]:
                self.event.append(pocket_door_event)
                if self.tasks_trust and self.agents[agent_idx].get_selected_task() == pocket_door_event:
                    if self.training:
                        self.tasks_trust.update_trust(agent_idx, self.agents[agent_idx].get_selected_task(), 1.0)
                    return True
                elif self.tasks_trust and not self.agents[agent_idx].get_selected_task() == pocket_door_event:
                    if self.training:
                        self.tasks_trust.update_trust(agent_idx, self.agents[agent_idx].get_selected_task(), 0.0)
            elif self.tasks_trust and self.agents[agent_idx].get_selected_task() == pocket_door_event:
                if self.training:
                    self.tasks_trust.update_trust(agent_idx, self.agents[agent_idx].get_selected_task(), 0.0)
                return True

            return False

    def reach_target(
        self,
        agent_idx,
        target_idx,
        target_event,
    ):

        # check of final position of the agents
        if np.array_equal(self.agents[agent_idx].position, self._targets_location[target_idx]):
            self.event.append(target_event)
            if self.tasks_trust and self.agents[agent_idx].get_selected_task() == target_event:
                if self.training:
                    self.tasks_trust.update_trust(agent_idx, self.agents[agent_idx].get_selected_task(), 1.0)
                return True
            elif self.tasks_trust and not self.agents[agent_idx].get_selected_task() == target_event:
                if self.training:
                    self.tasks_trust.update_trust(agent_idx, self.agents[agent_idx].get_selected_task(), 0.0)

        return False

    def step(
        self,
        actions,
    ):

        self.event = []
        self.random_event = False

        reward = [0.0, 0.0]
        openers = [0, 0]

        for agent_idx, action in enumerate(list(actions.values())):

            # Map the action (element of {0,1,2,3}) to the direction we walk in
            if action < 4:

                reward[agent_idx] = self.agent_move(action, agent_idx)

            # check for push button to open door
            elif action == 4:

                reward[agent_idx] = Environment_data.impossible_reward

                if (
                        self.open_door(agent_idx, 0, 0, 'red', openers) or
                        self.open_door(agent_idx, 1, 0, 'blue', openers)
                ):

                    reward[agent_idx] = Environment_data.subtask_reward

            # check for open pocket door
            elif action == 5:

                reward[agent_idx] = Environment_data.impossible_reward

                if (
                        self.open_pocket_door(agent_idx, 0, 'door_1') or
                        self.open_pocket_door(agent_idx, 1, 'door_2')
                ):

                    reward[agent_idx] = Environment_data.subtask_reward

            # check of final position of the agents
            if agent_idx == 0 and 'target_1' in self.next_event:
                if self.reach_target(agent_idx, 0, 'target_1'):
                    reward[agent_idx] = Environment_data.complete_reward
            if agent_idx == 1 and 'target_2' in self.next_event:
                if self.reach_target(agent_idx, 1, 'target_2'):
                    reward[agent_idx] = Environment_data.complete_reward

        # generate a fake event during training
        if self.training and not self.event and self.next_event:
            agent_idx = -1
            for action_idx, action in enumerate(actions.values()):
                if not action == 6:
                    agent_idx = action_idx
            random_uniform = random.uniform(0, 1)
            if (
                    random_uniform > self.train_transition and
                    list(set(self.agents[agent_idx].events) - set(self.next_event))
            ):
                temp_state = self.agents[agent_idx].state
                temp_goal = self.agents[agent_idx].temporal_goal.automaton.get_transitions_from(temp_state + 1)
                atoms = []
                for transition in temp_goal:
                    for atom in str(sympify(transition[1])).replace(' ', '').split('&'):
                        if not atom[0] == '~' and not atom == 'target_1' and not atom == 'target_2':
                            atoms.append(atom)
                if atoms:
                    self.event = [atoms[random.randint(0, len(atoms) - 1)]]
                    while self.event[0] not in self.next_event:
                        self.event = [atoms[random.randint(0, len(atoms) - 1)]]
                    self.random_event = True

        # from the event to the correlated action
        if 'red' in self.event:
            self._doors_flag[0] = 0
        if 'blue' in self.event:
            self._doors_flag[1] = 0
        if 'door_1' in self.event:
            self._pocket_doors_flag[0] = 0
        if 'door_2' in self.event:
            self._pocket_doors_flag[1] = 0

        if self.event:
            # remove from the next_event the pass event
            for item in self.event:
                if item in self.next_event:
                    self.next_event.remove(item)

        # compute the reward dict
        reward_dict = {'agent_' + str(key + 1): value for key, value in enumerate(reward)}

        # compute the termination dict
        if not self.training:
            terminated = [
                'target_1' not in self.next_event,
                'target_2' not in self.next_event,
            ]
        else:
            terminated = [
                'target_1' not in self.next_event,
                'target_2' not in self.next_event,
            ]

        terminated_dict = {'agent_' + str(key + 1): value for key, value in enumerate(terminated)}

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward_dict, terminated_dict, False, info

    def plot_trust(self, trust_list):

        fig = plt.figure(figsize=(8, 8))

        fig.legend(loc='upper right')

        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        ax1.title.set_text(Environment_data.events[0] + ' CTF')
        ax2.title.set_text(Environment_data.events[1] + ' CTF')
        ax3.title.set_text(Environment_data.events[2] + ' CTF')
        ax4.title.set_text(Environment_data.events[3] + ' CTF')
        ax5.title.set_text(Environment_data.events[4] + ' CTF')
        ax6.title.set_text(Environment_data.events[5] + ' CTF')

        ax1.set(xlabel='Epochs', ylabel='CTF')
        ax2.set(xlabel='Epochs', ylabel='CTF')
        ax3.set(xlabel='Epochs', ylabel='CTF')
        ax4.set(xlabel='Epochs', ylabel='CTF')
        ax5.set(xlabel='Epochs', ylabel='CTF')
        ax6.set(xlabel='Epochs', ylabel='CTF')

        ax1.set_ylim([-0.15, 1.15])
        ax2.set_ylim([-0.15, 1.15])
        ax3.set_ylim([-0.15, 1.15])
        ax4.set_ylim([-0.15, 1.15])
        ax5.set_ylim([-0.15, 1.15])
        ax6.set_ylim([-0.15, 1.15])

        ax1.plot(trust_list[0][0], 'yellowgreen', label='agent_1')
        ax1.plot(trust_list[1][0], 'orange', label='agent_2')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

        ax2.plot(trust_list[0][1], 'yellowgreen', label='agent_1')
        ax2.plot(trust_list[1][1], 'orange', label='agent_2')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

        ax3.plot(trust_list[0][2], 'yellowgreen', label='agent_1')
        ax3.plot(trust_list[1][2], 'orange', label='agent_2')
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

        ax4.plot(trust_list[0][3], 'yellowgreen', label='agent_1')
        ax4.plot(trust_list[1][3], 'orange', label='agent_2')
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

        ax5.plot(trust_list[0][4], 'yellowgreen', label='agent_1')
        ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

        ax6.plot(trust_list[1][5], 'orange', label='agent_2')
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

        plt.ylim(-0.15, 1.15)
        fig.tight_layout(pad=0.75)
        plt.show()

    def render(
        self,
    ):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(
        self,
    ):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.display.set_mode((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

        agent_def_img = pygame.transform.scale(
            pygame.image.load(Environment_data.agent_def_path), (pix_square_size, pix_square_size)
        )
        agent_red_img = pygame.transform.scale(
            pygame.image.load(Environment_data.agent_red_path), (pix_square_size, pix_square_size)
        )
        agent_blue_img = pygame.transform.scale(
            pygame.image.load(Environment_data.agent_blue_path), (pix_square_size, pix_square_size)
        )
        agent_one_img = pygame.transform.scale(
            pygame.image.load(Environment_data.agent_one_path), (pix_square_size, pix_square_size)
        )
        agent_two_img = pygame.transform.scale(
            pygame.image.load(Environment_data.agent_two_path), (pix_square_size, pix_square_size)
        )
        agent_three_img = pygame.transform.scale(
            pygame.image.load(Environment_data.agent_three_path), (pix_square_size, pix_square_size)
        )
        agent_target_img = pygame.transform.scale(
            pygame.image.load(Environment_data.agent_target_path), (pix_square_size, pix_square_size)
        )

        closed_door_1_img = pygame.transform.scale(
            pygame.image.load(Environment_data.closed_door_1_path), (pix_square_size, pix_square_size)
        )
        open_door_1_img = pygame.transform.scale(
            pygame.image.load(Environment_data.open_door_1_path), (pix_square_size, pix_square_size)
        )
        closed_door_2_img = pygame.transform.scale(
            pygame.image.load(Environment_data.closed_door_2_path), (pix_square_size, pix_square_size)
        )
        open_door_2_img = pygame.transform.scale(
            pygame.image.load(Environment_data.open_door_2_path), (pix_square_size, pix_square_size)
        )

        # Draw the target
        for target_location in self._targets_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw the doors and the buttons
        for door_idx, door_path, button_path in (
                zip(range(len(self._doors_location)), self._doors_img_paths, self._buttons_img_paths)
        ):

            for door in self._doors_location[door_idx]:

                if self._doors_flag[door_idx] == 1:

                    door_img = pygame.transform.scale(
                        pygame.image.load(door_path), (pix_square_size, pix_square_size)
                    )

                    # Draw the doors
                    canvas.blit(
                        door_img,
                        pygame.Rect(
                            pix_square_size * door,
                            (pix_square_size, pix_square_size),
                        )
                    )

            # Draw the buttons
            for button in self._doors_button[door_idx]:

                button_img = pygame.transform.scale(
                    pygame.image.load(button_path), (pix_square_size, pix_square_size)
                )

                canvas.blit(
                    button_img,
                    pygame.Rect(
                        pix_square_size * button,
                        (pix_square_size, pix_square_size),
                    )
                )

        # draw pocket door 1
        if self._pocket_doors_flag[0] == 1:
            # Draw the closed doors
            canvas.blit(
                closed_door_1_img,
                pygame.Rect(
                    pix_square_size * self._pocket_doors_location[0],
                    (pix_square_size, pix_square_size),
                )
            )
        else:
            # Draw the open doors
            canvas.blit(
                open_door_1_img,
                pygame.Rect(
                    pix_square_size * self._pocket_doors_location[0],
                    (pix_square_size, pix_square_size),
                )
            )

        # draw pocket door 2
        if self._pocket_doors_flag[1] == 1:
            # Draw the closed doors
            canvas.blit(
                closed_door_2_img,
                pygame.Rect(
                    pix_square_size * self._pocket_doors_location[1],
                    (pix_square_size, pix_square_size),
                )
            )
        else:
            # Draw the open doors
            canvas.blit(
                open_door_2_img,
                pygame.Rect(
                    pix_square_size * self._pocket_doors_location[1],
                    (pix_square_size, pix_square_size),
                )
            )

        # Now we draw the agents
        for agent in self.agents:
            if agent.get_state() == 5:
                agent_img = agent_def_img
            else:
                match agent.get_selected_task():
                    case 'blue':
                        agent_img = agent_blue_img
                    case 'red':
                        agent_img = agent_red_img
                    case 'door_1':
                        agent_img = agent_one_img
                    case 'door_2':
                        agent_img = agent_two_img
                    case 'target_1':
                        agent_img = agent_target_img
                    case 'target_2':
                        agent_img = agent_target_img
                    case _:
                        agent_img = agent_def_img

            canvas.blit(
                agent_img,
                pygame.Rect(
                    pix_square_size * agent.position,
                    (pix_square_size, pix_square_size),
                )
            )

        # Draw some gridlines for readability
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, pix_square_size * x - 2),
                (self.window_size, pix_square_size * x - 2),
                width=4,
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (pix_square_size * x - 2, 0),
                (pix_square_size * x - 2, self.window_size),
                width=4,
            )

        # Draw the walls
        for wall in self._walls:

            if wall[0][0] == wall[1][0]:
                pygame.draw.line(
                    canvas,
                    0,
                    (wall[0][0] * pix_square_size - 2,
                     wall[0][1] * pix_square_size + pix_square_size - 2),
                    (wall[1][0] * pix_square_size + pix_square_size - 2,
                     wall[1][1] * pix_square_size - 2),
                    width=4,
                )
            elif wall[0][1] == wall[1][1]:
                pygame.draw.line(
                    canvas,
                    0,
                    (wall[0][0] * pix_square_size + pix_square_size - 2,
                     wall[0][1] * pix_square_size - 2),
                    (wall[1][0] * pix_square_size - 2,
                     wall[1][1] * pix_square_size + pix_square_size - 2),
                    width=4,
                )

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined frame-rate.
            # The following line will automatically add a delay to keep the frame-rate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(
        self
    ):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
