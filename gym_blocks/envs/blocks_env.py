import numpy as np
import pandas as pd
import sys
from six import StringIO
from gym import utils
import gym
from gym.envs.toy_text import discrete

# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MoveL = 4  # moves with cube
MoveD = 5
MoveR = 6
MoveU = 7


class BlocksEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None):

        self.task_set = None
        self.cur_task = None
        self.task_policy = None
        self.path_policy = None
        self.observation_space = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.done_reward = None


        mapp = pd.read_csv('gym_blocks/envs/map.csv', sep=';')
        mapp = np.asarray(mapp)
        self.mapp = mapp  # initial observation
        target = np.asarray(pd.read_csv('gym_blocks/envs/target.csv', sep=';'))  # final position of blocks
        desc = mapp  # this variable will be changed after each step

        self.nrow = len(desc[0])  # dimension of observations  space
        self.ncol = len(desc)  # dimension of observations  space
        hand_row = 10  # initial hand center's row
        hand_col = 13  # initial hand center's col

        self.desc = desc  # current observation, initially the same as mapp
        self.hand_row = hand_row
        self.hand_col = hand_col

        nA = 8  # eight actions
        nS = 100  # number of hand center's

        P = {s: {a: [] for a in range(nA)} for s in
             range(nS)}  # required for proper work of environment, actually is not used
        self.target = target  # assigning target

        super(BlocksEnv, self).__init__()


    def _reset(self):
        self.s = np.random.choice(60)  # required for proper work of env
        self.desc = self.mapp.copy()  # assign initial map
        self.hand_row = 10
        self.hand_col = 13
        return self.desc

    def _step(self, a):

        def inc(a, desc):
            def make_zero(x, y):
                desc[x][y] = 0
                desc[x + 1][y] = 0
                desc[x + 1][y + 1] = 0
                desc[x + 1][y - 1] = 0
                desc[x - 1][y] = 0
            def make_zero_with_cube(x,y):
                desc[x][y] = 0
                desc[x+ 1][y] = 0
                desc[x + 1][y + 1] = 0
                desc[x + 1][y - 1] = 0
                desc[x - 1][y] = 0
                desc[x + 3][y] = 0
                desc[x + 4][y] = 0
                desc[x + 2][y] = 0
                desc[x + 3][y + 1] = 0
                desc[x + 4][y + 1] = 0
                desc[x + 2][y + 1] = 0
                desc[x + 4][y - 1] = 0
                desc[x + 2][y - 1] = 0
                desc[x + 3][y - 1] = 0
            def make_ones_with_cube(x,y):
                desc[x][y] = 1
                desc[x+ 1][y] = 1
                desc[x + 1][y + 1] = 1
                desc[x + 1][y - 1] = 1
                desc[x - 1][y] = 1
                desc[x + 3][y] = 1
                desc[x + 4][y] = 1
                desc[x + 2][y] = 1
                desc[x + 3][y + 1] = 1
                desc[x + 4][y + 1] = 1
                desc[x + 2][y + 1] = 1
                desc[x + 4][y - 1] = 1
                desc[x + 2][y - 1] = 1
                desc[x + 3][y - 1] = 1
            def make_ones(x, y):
                desc[x][y] = 1
                desc[x + 1][y] = 1
                desc[x + 1][y + 1] = 1
                desc[x + 1][y - 1] = 1
                desc[x - 1][y] = 1

            if a == 0:  # step left
                if self.hand_row == 28:
                    if self.hand_col > 1:
                        if desc[self.hand_row][self.hand_col - 3] == 0:

                            make_zero(self.hand_row, self.hand_col)
                            self.hand_col = self.hand_col - 3
                            make_ones(self.hand_row, self.hand_col)

                elif self.hand_row == 25:
                    if self.hand_col > 1:
                        if desc[self.hand_row][self.hand_col - 3] == 0:
                            make_zero(self.hand_row, self.hand_col)
                            self.hand_col = self.hand_col - 3
                            make_ones(self.hand_row, self.hand_col)

                else:
                    if self.hand_col > 1:
                        if (desc[self.hand_row + 3][self.hand_col] == 0) or (desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row + 6][self.hand_col] == 1):

                            make_zero(self.hand_row, self.hand_col)
                            self.hand_col = self.hand_col - 3
                            make_ones(self.hand_row, self.hand_col)

            elif a == 1:
                if self.hand_row < 26:
                    if desc[self.hand_row + 3][self.hand_col] == 0:
                        make_zero(self.hand_row, self.hand_col)
                        self.hand_row = min(self.hand_row + 3, 28)
                        make_ones(self.hand_row, self.hand_col)

            elif a == 2:  # right
                if self.hand_row == 28 and self.hand_col < 28:
                    if desc[self.hand_row][self.hand_col + 3] == 0:
                        make_zero(self.hand_row, self.hand_col)
                        self.hand_col = min(self.hand_col + 3, 28)
                        make_ones(self.hand_row, self.hand_col)

                elif self.hand_row == 25 and self.hand_col < 28:
                    if desc[self.hand_row][self.hand_col + 3] == 0:
                        make_zero(self.hand_row, self.hand_col)
                        self.hand_col = min(self.hand_col + 3, 28)
                        make_ones(self.hand_row, self.hand_col)

                elif self.hand_col < 28:
                    if (desc[self.hand_row][self.hand_col + 3] == 0) and ((desc[self.hand_row + 3][self.hand_col] == 0) or ((desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row + 6][self.hand_col] == 1))):

                        make_zero(self.hand_row, self.hand_col)
                        self.hand_col = self.hand_col + 3
                        make_ones(self.hand_row, self.hand_col)

            elif a == 3:  # up
                if self.hand_row == 28:

                    make_zero(self.hand_row, self.hand_col)
                    self.hand_row = max(self.hand_row - 3, 1)
                    make_ones(self.hand_row, self.hand_col)

                elif (desc[self.hand_row + 3][self.hand_col] == 0) or (
                            self.hand_row < 23 and desc[self.hand_row + 3][self.hand_col] == 1 and
                        desc[self.hand_row + 6][self.hand_col] == 1) or (
                        desc[self.hand_row + 3][self.hand_col] == 1 and self.hand_row == 25):

                    make_zero(self.hand_row, self.hand_col)
                    self.hand_row = max(self.hand_row - 3, 1)
                    make_ones(self.hand_row, self.hand_col)

            elif a == 4:  # MoveL
                if self.hand_row < 27:
                    if desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row + 3][
                                self.hand_col - 3] == 0 and self.hand_col > 2 and desc[self.hand_row][
                                self.hand_col - 3] == 0:

                        make_zero_with_cube(self.hand_row,self.hand_col)
                        self.hand_col = self.hand_col - 3
                        make_ones_with_cube(self.hand_row,self.hand_col)

            elif a == 5: # Cube down
                if self.hand_row < 24:
                    if (desc[self.hand_row + 3][self.hand_col] == 1) and (desc[self.hand_row + 6][self.hand_col] == 0):

                        make_zero_with_cube(self.hand_row,self.hand_col)
                        self.hand_row = self.hand_row + 3
                        make_ones_with_cube(self.hand_row,self.hand_col)

            if a == 6:
                if self.hand_col < 26 and self.hand_row < 27:
                    if desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row + 3][self.hand_col + 3] == 0 and desc[self.hand_row][self.hand_col + 3] == 0:
                        if desc[self.hand_row][self.hand_col] == 1:
                            make_zero_with_cube(self.hand_row,self.hand_col)
                            self.hand_col = self.hand_col + 3
                            make_ones_with_cube(self.hand_row,self.hand_col)

            if a == 7:  # Move cube up
                if self.hand_row < 27 and self.hand_row > 3:
                    if desc[self.hand_row][self.hand_col] == 1:
                        if desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row - 3][self.hand_col] == 0:
                            make_zero_with_cube(self.hand_row, self.hand_col)
                            self.hand_row = self.hand_row - 3
                            make_ones_with_cube(self.hand_row, self.hand_col)

            self.lastaction = a
            return desc

        def find_hand(x):  # returns current hand center's location
            for i in range(10):
                for k in range(10):
                    if (x[3 * i + 1][3 * k + 1] == 1) and (x[3 * i + 1][3 * k + 2] == 0) and (x[3 * i + 1][3 * k] == 0):
                        return i, k
        def find_cubes(x):  # returns current hand center's location
            s=0
            for i in range(10):
                for k in range(10):
                    if x[3 * i + 1][3 * k + 1] == 1:
                        s+=1
            return s
        print("N of cubes: ", find_cubes(self.desc))

        inc(a, self.desc)
        print("current location", self.hand_row, self.hand_col)

        self.desc_for_rew = self.desc.copy()

        self.desc_for_rew[self.hand_row][self.hand_col] = 0
        self.desc_for_rew[self.hand_row + 1][self.hand_col] = 0
        self.desc_for_rew[self.hand_row - 1][self.hand_col] = 0
        self.desc_for_rew[self.hand_row + 1][self.hand_col - 1] = 0
        self.desc_for_rew[self.hand_row + 1][self.hand_col + 1] = 0

        rew = np.sum(np.array(self.target) * np.array(self.desc_for_rew)) / np.sum(self.target)
        if (a == 3) and (int(self.lastaction) % 8 > 3): rew = 1.05 * rew  # increase rew for rational movements
        # assign zero reward for absolutely silly steps
        if (a == 1) and (int(self.lastaction) % 8 > 3): rew = 0
        if (a == 3) and (int(self.lastaction) % 8 == 7): rew = 0
        if (a == 4) and (int(self.lastaction) % 8 == 1): rew = 0
        if (a == 1) and (int(self.lastaction) % 8 == 3): rew = 0
        if (a == 3) and (int(self.lastaction) % 8 == 1): rew = 0
        if (a == 0) and (int(self.lastaction) % 8 == 2): rew = 0
        if (a == 2) and (int(self.lastaction) % 8 == 0): rew = 0

        self.rew = rew

        if rew == 1:
            self.reset()
            return (self.desc, rew, 1, 0)  # third param means "game over"
        print("END STEP \n")
        return (self.desc, rew, 0, 0)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        desc = self.desc.tolist()
        desc = [[c for c in line] for line in desc]
        # desc[self.hand_row][self.hand_col] = utils.colorize(desc[self.hand_row][self.hand_col], "red", highlight=True)
        outfile.write("\n".join(''.join(str(line)) for line in desc) + "\n")

        # if self.lastaction is not None:
        #     outfile.write("  ({})\n\n".format(
        #         ["Left", "Down", "Right", "Up", "MoveL", "MoveD", "MoveR", "MoveU"][self.lastaction]))
        # else:
        #     outfile.write("\n")
        return outfile








