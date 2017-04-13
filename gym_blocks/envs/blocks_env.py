import logging
import sys

import gym
import numpy as np
from gym.spaces import Discrete
from six import StringIO

# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MoveL = 4  # moves with cube
MoveD = 5
MoveR = 6
MoveU = 7

logger = logging.getLogger()


class BlocksEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self):
        self.action_space = Discrete(8)

    def _configure(self, raw_map_start=None, raw_map_final=None, state_size=None):
        self.state_size = state_size
        self.sit_start = self._preprocess_state(raw_map_start)
        self.sit_final = self._preprocess_state(raw_map_final)
        self.state = np.hstack((self.sit_start, self.sit_final))

        self.hand_position = self._find_hand()
        self.sit_current_map = raw_map_start
        self.sit_final_map = raw_map_final
        self.last_action = None
        self.reward = None
        self.step_n = 0

    def _reset(self):
        return self.state

    def _step(self, a):

        def build_new_state(action, map):
            current_map = map.copy()

            def make_zero(x, y):
                current_map[x][y] = 0
                current_map[x + 1][y] = 0
                current_map[x + 1][y + 1] = 0
                current_map[x + 1][y - 1] = 0
                current_map[x - 1][y] = 0

            def make_zero_with_cube(x, y):
                current_map[x][y] = 0
                current_map[x + 1][y] = 0
                current_map[x + 1][y + 1] = 0
                current_map[x + 1][y - 1] = 0
                current_map[x - 1][y] = 0
                current_map[x + 3][y] = 0
                current_map[x + 4][y] = 0
                current_map[x + 2][y] = 0
                current_map[x + 3][y + 1] = 0
                current_map[x + 4][y + 1] = 0
                current_map[x + 2][y + 1] = 0
                current_map[x + 4][y - 1] = 0
                current_map[x + 2][y - 1] = 0
                current_map[x + 3][y - 1] = 0

            def make_ones_with_cube(x, y):
                current_map[x][y] = 1
                current_map[x + 1][y] = 1
                current_map[x + 1][y + 1] = 1
                current_map[x + 1][y - 1] = 1
                current_map[x - 1][y] = 1
                current_map[x + 3][y] = 1
                current_map[x + 4][y] = 1
                current_map[x + 2][y] = 1
                current_map[x + 3][y + 1] = 1
                current_map[x + 4][y + 1] = 1
                current_map[x + 2][y + 1] = 1
                current_map[x + 4][y - 1] = 1
                current_map[x + 2][y - 1] = 1
                current_map[x + 3][y - 1] = 1

            def make_ones(x, y):
                current_map[x][y] = 1
                current_map[x + 1][y] = 1
                current_map[x + 1][y + 1] = 1
                current_map[x + 1][y - 1] = 1
                current_map[x - 1][y] = 1

            if action == 0:  # step left
                if self.hand_position[0] == 28:
                    if self.hand_position[1] > 1:
                        if current_map[self.hand_position[0]][self.hand_position[1] - 3] == 0:
                            make_zero(self.hand_position[0], self.hand_position[1])
                            self.hand_position[1] -= 3
                            make_ones(self.hand_position[0], self.hand_position[1])

                elif self.hand_position[0] == 25:
                    if self.hand_position[1] > 1:
                        if current_map[self.hand_position[0]][self.hand_position[1] - 3] == 0:
                            make_zero(self.hand_position[0], self.hand_position[1])
                            self.hand_position[1] = self.hand_position[1] - 3
                            make_ones(self.hand_position[0], self.hand_position[1])

                elif self.hand_position[1] > 1:
                    if current_map[self.hand_position[0] + 3][self.hand_position[1]] == 0:
                        if current_map[self.hand_position[0]][self.hand_position[1]-3] == 0:
                            make_zero(self.hand_position[0], self.hand_position[1])
                            self.hand_position[1] -= 3
                            make_ones(self.hand_position[0], self.hand_position[1])

                elif self.hand_position[1] > 1:
                    if current_map[self.hand_position[0] + 3][self.hand_position[1]] == 1 \
                     and current_map[self.hand_position[0] + 6][self.hand_position[1]] == 1:
                        if current_map[self.hand_position[0]][self.hand_position[1]-3] == 0:
                            make_zero(self.hand_position[0], self.hand_position[1])
                            self.hand_position[1] -= 3
                            make_ones(self.hand_position[0], self.hand_position[1])
                else:
                    pass

            elif action == 1:
                if self.hand_position[0] < 26:
                    if current_map[self.hand_position[0] + 3][self.hand_position[1]] == 0:
                        make_zero(self.hand_position[0], self.hand_position[1])
                        self.hand_position[0] = min(self.hand_position[0] + 3, 28)
                        make_ones(self.hand_position[0], self.hand_position[1])

            elif action == 2:  # right
                if self.hand_position[0] == 28 and self.hand_position[1] < 28:
                    if current_map[self.hand_position[0]][self.hand_position[1] + 3] == 0:
                        make_zero(self.hand_position[0], self.hand_position[1])
                        self.hand_position[1] = min(self.hand_position[1] + 3, 28)
                        make_ones(self.hand_position[0], self.hand_position[1])

                elif self.hand_position[0] == 25 and self.hand_position[1] < 28:
                    if current_map[self.hand_position[0]][self.hand_position[1] + 3] == 0:
                        make_zero(self.hand_position[0], self.hand_position[1])
                        self.hand_position[1] = min(self.hand_position[1] + 3, 28)
                        make_ones(self.hand_position[0], self.hand_position[1])

                elif self.hand_position[1] < 28:
                    if (current_map[self.hand_position[0]][self.hand_position[1] + 3] == 0) and (
                                (current_map[self.hand_position[0] + 3][self.hand_position[1]] == 0) or (
                                    (current_map[self.hand_position[0] + 3][self.hand_position[1]] == 1 and
                                             current_map[self.hand_position[0] + 6][
                                                 self.hand_position[1]] == 1))):
                        make_zero(self.hand_position[0], self.hand_position[1])
                        self.hand_position[1] = self.hand_position[1] + 3
                        make_ones(self.hand_position[0], self.hand_position[1])

            elif action == 3:  # up
                if self.hand_position[0] == 28:

                    make_zero(self.hand_position[0], self.hand_position[1])
                    self.hand_position[0] = max(self.hand_position[0] - 3, 1)
                    make_ones(self.hand_position[0], self.hand_position[1])

                elif (current_map[self.hand_position[0] + 3][self.hand_position[1]] == 0) or (
                                    self.hand_position[0] < 23 and current_map[self.hand_position[0] + 3][
                                self.hand_position[1]] == 1 and
                                current_map[self.hand_position[0] + 6][self.hand_position[1]] == 1) or (
                                current_map[self.hand_position[0] + 3][self.hand_position[1]] == 1 and
                                self.hand_position[0] == 25):

                    make_zero(self.hand_position[0], self.hand_position[1])
                    self.hand_position[0] = max(self.hand_position[0] - 3, 1)
                    make_ones(self.hand_position[0], self.hand_position[1])

            elif action == 4:  # MoveL
                if self.hand_position[0] < 27:
                    if current_map[self.hand_position[0] + 3][self.hand_position[1]] == 1 and \
                                    current_map[self.hand_position[0] + 3][
                                                self.hand_position[1] - 3] == 0 and self.hand_position[1] > 2 and \
                                    current_map[self.hand_position[0]][
                                                self.hand_position[1] - 3] == 0:
                        make_zero_with_cube(self.hand_position[0], self.hand_position[1])
                        self.hand_position[1] = self.hand_position[1] - 3
                        make_ones_with_cube(self.hand_position[0], self.hand_position[1])

            elif action == 5:  # Cube down
                if self.hand_position[0] < 24:
                    if (current_map[self.hand_position[0] + 3][self.hand_position[1]] == 1) and (
                                current_map[self.hand_position[0] + 6][self.hand_position[1]] == 0):
                        make_zero_with_cube(self.hand_position[0], self.hand_position[1])
                        self.hand_position[0] = self.hand_position[0] + 3
                        make_ones_with_cube(self.hand_position[0], self.hand_position[1])

            if action == 6:
                if self.hand_position[1] < 26 and self.hand_position[0] < 27:
                    if current_map[self.hand_position[0] + 3][self.hand_position[1]] == 1 and \
                                    current_map[self.hand_position[0] + 3][
                                                self.hand_position[1] + 3] == 0 and current_map[self.hand_position[0]][
                                self.hand_position[1] + 3] == 0:
                        if current_map[self.hand_position[0]][self.hand_position[1]] == 1:
                            make_zero_with_cube(self.hand_position[0], self.hand_position[1])
                            self.hand_position[1] = self.hand_position[1] + 3
                            make_ones_with_cube(self.hand_position[0], self.hand_position[1])

            if action == 7:  # Move cube up
                if self.hand_position[0] < 27 and self.hand_position[0] > 3:
                    if current_map[self.hand_position[0]][self.hand_position[1]] == 1:
                        if current_map[self.hand_position[0] + 3][self.hand_position[1]] == 1 and \
                                        current_map[self.hand_position[0] - 3][self.hand_position[1]] == 0:
                            make_zero_with_cube(self.hand_position[0], self.hand_position[1])
                            self.hand_position[0] = self.hand_position[0] - 3
                            make_ones_with_cube(self.hand_position[0], self.hand_position[1])

            return current_map

        def find_hand(x):  # returns current hand center's location
            for i in range(10):
                for k in range(10):
                    if (x[3 * i + 1][3 * k + 1] == 1) and (x[3 * i + 1][3 * k + 2] == 0) and (x[3 * i + 1][3 * k] == 0):
                        return i, k

        def find_cubes(x):  # returns N of cubes
            s = 0
            for i in range(10):
                for k in range(10):
                    if x[3 * i + 1][3 * k + 1] == 1:
                        s += 1
            return s

        logger.debug("N of cubes: {}".format(find_cubes(self.sit_current_map)))

        self.sit_current_map = build_new_state(a, self.sit_current_map)
        logger.debug("current location {0}".format(self.hand_position))

        map_rewarded = self.sit_current_map.copy()

        map_rewarded[self.hand_position[0]][self.hand_position[1]] = 0
        map_rewarded[self.hand_position[0] + 1][self.hand_position[1]] = 0
        map_rewarded[self.hand_position[0] - 1][self.hand_position[1]] = 0
        map_rewarded[self.hand_position[0] + 1][self.hand_position[1] - 1] = 0
        map_rewarded[self.hand_position[0] + 1][self.hand_position[1] + 1] = 0

        rew = np.sum(np.array(self.sit_final_map) * np.array(map_rewarded)) / np.sum(self.sit_final_map)
        if self.last_action:
            if (a == 3) and (int(self.last_action) % 8 > 3): rew = 1.05 * rew  # increase rew for rational movements
            # assign zero reward for absolutely silly steps
            if (a == 1) and (int(self.last_action) % 8 > 3): rew = 0
            if (a == 3) and (int(self.last_action) % 8 == 7): rew = 0
            if (a == 4) and (int(self.last_action) % 8 == 1): rew = 0
            if (a == 1) and (int(self.last_action) % 8 == 3): rew = 0
            if (a == 3) and (int(self.last_action) % 8 == 1): rew = 0
            if (a == 0) and (int(self.last_action) % 8 == 2): rew = 0
            if (a == 2) and (int(self.last_action) % 8 == 0): rew = 0

        self.reward = rew
        self.last_action = a

        self.state = self._preprocess_state(self.sit_current_map)
        self.state = np.hstack((self.state, self.sit_final))

        if self.reward == 1:
            self.reset()
            self.step_n = 1
            return self.state, self.reward, 1, 0  # third param means "game over"
        logger.debug("END STEP\n")


        if  self.step_n == 100:
            self.reset()
            self.step_n = 1
            return self.state, 0, 1, 0  # third param means "game over"


        self.step_n +=1
        return self.state, self.reward, 0, self.hand_position




    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        desc = self.sit_current_map.tolist()
        desc = [[c for c in line] for line in desc]
        outfile.write("\n".join(''.join(str(line)) for line in desc) + "\n")

        return outfile

    def _find_hand(self):
        return [10, 13]  # r(row, column)

    def _preprocess_state(self, map):
        raveled = map.astype(np.float).ravel()
        return np.reshape(raveled, [1, self.state_size])
