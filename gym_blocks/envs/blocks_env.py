import numpy as np
import pandas as pd
import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete

# Actions

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MoveL = 4 # moves with cube
MoveD = 5
MoveR = 6
MoveU = 7


class BlocksEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None):
        mapp = pd.read_csv('/Users/Edward/PyCharmProjects/Blocks/gym_blocks/envs/map.csv', sep=';')
        mapp = np.asarray(mapp)
        self.mapp = mapp  #initial observation
        target = np.asarray(pd.read_csv('/Users/Edward/PyCharmProjects/Blocks/gym_blocks/envs/target.csv', sep=';')) #final position of blocks
        desc = mapp # this variable will be changed after each step

        self.nrow = len(desc[0]) #dimension of observations  space
        self.ncol = len(desc) #dimension of observations  space
        hand_row = 10  #initial hand center's row
        hand_col = 13  #initial hand center's col

        self.desc = desc #current observation, initially the same as mapp
        self.hand_row = hand_row
        self.hand_col = hand_col
        nA = 8 # eight actions
        nS = 100  #number of hand center's
        
        P = {s : {a : [] for a in range(nA)} for s in range(nS)} #required for proper work of environment, actually is not used
        self.target = target #assigning target

        super(BlocksEnv, self).__init__(nS, nA, P, desc)

    def _reset(self):
         self.s = np.random.choice(60) #required for proper work of env
         self.desc = self.mapp.copy() #assign initial map
         self.hand_row = 10
         self.hand_col = 13
         return self.desc

    def _step(self, a):
        def inc(a, desc):
            if a == 0: #step left
                if (desc[self.hand_row+3][self.hand_col]==0) or (desc[self.hand_row+3][self.hand_col]==1 and self.hand_row ==28) or (desc[self.hand_row+3][self.hand_col]==1 and desc[self.hand_row+6][self.hand_col]==1):
                    desc[self.hand_row][self.hand_col]=0
                    desc[self.hand_row + 1][self.hand_col]=0
                    desc[self.hand_row + 1][self.hand_col + 1]=0
                    desc[self.hand_row + 1][self.hand_col -1]=0
                    desc[self.hand_row - 1][self.hand_col]=0

                    self.hand_col = max(self.hand_col - 3, 1)

                    desc[self.hand_row][self.hand_col]=1
                    desc[self.hand_row + 1][self.hand_col]=1
                    desc[self.hand_row + 1][self.hand_col + 1]=1
                    desc[self.hand_row + 1][self.hand_col -1]=1
                    desc[self.hand_row - 1][self.hand_col]=1

            elif a == 1:
                if desc[self.hand_row + 3][self.hand_col] == 0:
                    desc[self.hand_row][self.hand_col]=0
                    desc[self.hand_row + 1][self.hand_col] = 0
                    desc[self.hand_row + 1][self.hand_col + 1] = 0
                    desc[self.hand_row + 1][self.hand_col - 1] = 0
                    desc[self.hand_row - 1][self.hand_col] = 0
                    self.hand_row = min(self.hand_row + 3, 28)

                    desc[self.hand_row][self.hand_col]=1
                    desc[self.hand_row + 1][self.hand_col] = 1
                    desc[self.hand_row + 1][self.hand_col + 1] = 1
                    desc[self.hand_row + 1][self.hand_col - 1] = 1
                    desc[self.hand_row - 1][self.hand_col] = 1
            elif a == 2:  # right
                if (desc[self.hand_row + 3][self.hand_col] == 0) or (desc[self.hand_row + 3][self.hand_col] == 1 and self.hand_row == 28) or (desc[self.hand_row+3][self.hand_col] == 1 and desc[self.hand_row+6][self.hand_col] == 1):
                    desc[self.hand_row][self.hand_col] = 0
                    desc[self.hand_row + 1][self.hand_col] = 0
                    desc[self.hand_row + 1][self.hand_col + 1] = 0
                    desc[self.hand_row + 1][self.hand_col - 1] = 0
                    desc[self.hand_row - 1][self.hand_col] = 0

                    self.hand_col = min(self.hand_col + 3, 28)

                    desc[self.hand_row][self.hand_col] =1
                    desc[self.hand_row + 1][self.hand_col] = 1
                    desc[self.hand_row + 1][self.hand_col + 1] = 1
                    desc[self.hand_row + 1][self.hand_col - 1] = 1
                    desc[self.hand_row - 1][self.hand_col] = 1

            elif a == 3:  # up

                if (desc[self.hand_row + 3][self.hand_col] == 0) or (desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row + 6][self.hand_col] == 1) or (desc[self.hand_row + 3][self.hand_col] == 1 and self.hand_row ==25) or self.hand_row ==28:
                    desc[self.hand_row][self.hand_col]=0
                    desc[self.hand_row + 1][self.hand_col] = 0
                    desc[self.hand_row + 1][self.hand_col + 1] = 0
                    desc[self.hand_row + 1][self.hand_col - 1] = 0
                    desc[self.hand_row - 1][self.hand_col] = 0
                    self.hand_row = max(self.hand_row - 3, 0)
                    desc[self.hand_row][self.hand_col]=1
                    desc[self.hand_row + 1][self.hand_col] = 1
                    desc[self.hand_row + 1][self.hand_col + 1] = 1
                    desc[self.hand_row + 1][self.hand_col - 1] = 1
                    desc[self.hand_row - 1][self.hand_col] = 1
            elif a == 4: #MoveL
                    if desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row + 3][self.hand_col-3]==0 and self.hand_col>2:
                        desc[self.hand_row][self.hand_col]=0
                        desc[self.hand_row + 1][self.hand_col] = 0
                        desc[self.hand_row + 1][self.hand_col + 1] = 0
                        desc[self.hand_row + 1][self.hand_col - 1] = 0
                        desc[self.hand_row - 1][self.hand_col] = 0
                        desc[self.hand_row + 3][self.hand_col]=0
                        desc[self.hand_row + 4][self.hand_col] = 0
                        desc[self.hand_row + 2][self.hand_col] = 0
                        desc[self.hand_row + 3][self.hand_col + 1] = 0
                        desc[self.hand_row + 4][self.hand_col + 1] = 0
                        desc[self.hand_row + 2][self.hand_col + 1] = 0
                        desc[self.hand_row + 4][self.hand_col - 1] = 0
                        desc[self.hand_row + 2][self.hand_col - 1] = 0
                        desc[self.hand_row + 3][self.hand_col - 1] = 0

                        self.hand_col = max(self.hand_col - 3, 1)
                        desc[self.hand_row][self.hand_col]=1
                        desc[self.hand_row + 1][self.hand_col] = 1
                        desc[self.hand_row + 1][self.hand_col + 1] = 1
                        desc[self.hand_row + 1][self.hand_col - 1] = 1
                        desc[self.hand_row - 1][self.hand_col] = 1
                        desc[self.hand_row + 3][self.hand_col]=1
                        desc[self.hand_row + 4][self.hand_col] = 1
                        desc[self.hand_row + 2][self.hand_col] = 1
                        desc[self.hand_row + 3][self.hand_col + 1] = 1
                        desc[self.hand_row + 4][self.hand_col + 1] = 1
                        desc[self.hand_row + 2][self.hand_col + 1] = 1
                        desc[self.hand_row + 4][self.hand_col - 1] = 1
                        desc[self.hand_row + 2][self.hand_col - 1] = 1
                        desc[self.hand_row + 3][self.hand_col - 1] = 1
            elif a==5:
                    if (desc[self.hand_row + 3][self.hand_col] == 1) and (desc[self.hand_row + 6][
                                self.hand_col ]== 0) and (self.hand_row <25):
                        desc[self.hand_row][self.hand_col]=0
                        desc[self.hand_row + 1][self.hand_col] = 0
                        desc[self.hand_row + 1][self.hand_col + 1] = 0
                        desc[self.hand_row + 1][self.hand_col - 1] = 0
                        desc[self.hand_row - 1][self.hand_col] = 0
                        desc[self.hand_row + 3][self.hand_col]=0
                        desc[self.hand_row + 4][self.hand_col] = 0
                        desc[self.hand_row + 2][self.hand_col] = 0
                        desc[self.hand_row + 3][self.hand_col + 1] = 0
                        desc[self.hand_row + 4][self.hand_col + 1] = 0
                        desc[self.hand_row + 2][self.hand_col + 1] = 0
                        desc[self.hand_row + 4][self.hand_col - 1] = 0
                        desc[self.hand_row + 2][self.hand_col - 1] = 0
                        desc[self.hand_row + 3][self.hand_col - 1] = 0

                        self.hand_row = self.hand_col + 3

                        desc[self.hand_row][self.hand_col]=1
                        desc[self.hand_row + 1][self.hand_col] = 1
                        desc[self.hand_row + 1][self.hand_col + 1] = 1
                        desc[self.hand_row + 1][self.hand_col - 1] = 1
                        desc[self.hand_row - 1][self.hand_col] = 1
                        desc[self.hand_row + 3][self.hand_col]=1
                        desc[self.hand_row + 4][self.hand_col] = 1
                        desc[self.hand_row + 2][self.hand_col] = 1
                        desc[self.hand_row + 3][self.hand_col + 1] = 1
                        desc[self.hand_row + 4][self.hand_col + 1] = 1
                        desc[self.hand_row + 2][self.hand_col + 1] = 1
                        desc[self.hand_row + 4][self.hand_col - 1] = 1
                        desc[self.hand_row + 2][self.hand_col - 1] = 1
                        desc[self.hand_row + 3][self.hand_col - 1] = 1
            if a==6:
                if desc[self.hand_row + 3][self.hand_col] == 1 and desc[self.hand_row + 3][
                                                                        self.hand_col + 3] == 0 and self.hand_col <26:
                    desc[self.hand_row][self.hand_col]=0
                    desc[self.hand_row + 1][self.hand_col] = 0
                    desc[self.hand_row + 1][self.hand_col + 1] = 0
                    desc[self.hand_row + 1][self.hand_col - 1] = 0
                    desc[self.hand_row - 1][self.hand_col] = 0
                    desc[self.hand_row + 3][self.hand_col]=0
                    desc[self.hand_row + 4][self.hand_col] = 0
                    desc[self.hand_row + 2][self.hand_col] = 0
                    desc[self.hand_row + 3][self.hand_col + 1] = 0
                    desc[self.hand_row + 4][self.hand_col + 1] = 0
                    desc[self.hand_row + 2][self.hand_col + 1] = 0
                    desc[self.hand_row + 4][self.hand_col - 1] = 0
                    desc[self.hand_row + 2][self.hand_col - 1] = 0
                    desc[self.hand_row + 3][self.hand_col - 1] = 0

                    self.hand_col = self.hand_col + 3

                    desc[self.hand_row][self.hand_col]=1
                    desc[self.hand_row + 1][self.hand_col] = 1
                    desc[self.hand_row + 1][self.hand_col + 1] = 1
                    desc[self.hand_row + 1][self.hand_col - 1] = 1
                    desc[self.hand_row - 1][self.hand_col] = 1
                    desc[self.hand_row + 3][self.hand_col]=1
                    desc[self.hand_row + 4][self.hand_col] = 1
                    desc[self.hand_row + 2][self.hand_col] = 1
                    desc[self.hand_row + 3][self.hand_col + 1] = 1
                    desc[self.hand_row + 4][self.hand_col + 1] = 1
                    desc[self.hand_row + 2][self.hand_col + 1] = 1
                    desc[self.hand_row + 4][self.hand_col - 1] = 1
                    desc[self.hand_row + 2][self.hand_col - 1] = 1
                    desc[self.hand_row + 3][self.hand_col - 1] = 1
            if a==7: #Move cube up
                if desc[self.hand_row + 3][self.hand_col] == 1  and self.hand_row > 3:
                    desc[self.hand_row][self.hand_col]=0
                    desc[self.hand_row + 1][self.hand_col] = 0
                    desc[self.hand_row + 1][self.hand_col + 1] = 0
                    desc[self.hand_row + 1][self.hand_col - 1] = 0
                    desc[self.hand_row - 1][self.hand_col] = 0
                    desc[self.hand_row + 3][self.hand_col]=0
                    desc[self.hand_row + 4][self.hand_col] = 0
                    desc[self.hand_row + 2][self.hand_col] = 0
                    desc[self.hand_row + 3][self.hand_col + 1] = 0
                    desc[self.hand_row + 4][self.hand_col + 1] = 0
                    desc[self.hand_row + 2][self.hand_col + 1] = 0
                    desc[self.hand_row + 4][self.hand_col - 1] = 0
                    desc[self.hand_row + 2][self.hand_col - 1] = 0
                    desc[self.hand_row + 3][self.hand_col - 1] = 0

                    self.hand_row = self.hand_row - 3

                    desc[self.hand_row][self.hand_col]=1
                    desc[self.hand_row + 1][self.hand_col] = 1
                    desc[self.hand_row + 1][self.hand_col + 1] = 1
                    desc[self.hand_row + 1][self.hand_col - 1] = 1
                    desc[self.hand_row - 1][self.hand_col] = 1
                    desc[self.hand_row + 3][self.hand_col]=1
                    desc[self.hand_row + 4][self.hand_col] = 1
                    desc[self.hand_row + 2][self.hand_col] = 1
                    desc[self.hand_row + 3][self.hand_col + 1] = 1
                    desc[self.hand_row + 4][self.hand_col + 1] = 1
                    desc[self.hand_row + 2][self.hand_col + 1] = 1
                    desc[self.hand_row + 4][self.hand_col - 1] = 1
                    desc[self.hand_row + 2][self.hand_col - 1] = 1
                    desc[self.hand_row + 3][self.hand_col - 1] = 1
            self.lastaction = a
            return desc

        def find_hand(x): # returns current hand center's location
            for i in range(10):
                for k in range(10):
                    if (x[3*i+1][3*k+1]==1) and (x[3*i+1][3*k+2]==0): return (i, k)

        inc(a, self.desc)

        self.desc_for_rew = self.desc.copy()
        #Abolish hand from observation
        self.desc_for_rew[find_hand(self.desc)[0]*3+1][find_hand(self.desc)[1]*3+1]=0
        self.desc_for_rew[find_hand(self.desc)[0]*3+2][find_hand(self.desc)[1]*3+1]=0
        self.desc_for_rew[find_hand(self.desc)[0]*3][find_hand(self.desc)[1]*3+1]=0
        self.desc_for_rew[find_hand(self.desc)[0]*3+2][find_hand(self.desc)[1]*3]=0
        self.desc_for_rew[find_hand(self.desc)[0]*3+2][find_hand(self.desc)[1]*3+2]=0

        rew = np.sum(np.array(self.target)*np.array(self.desc_for_rew))/np.sum(self.target)
        if (a==3) and (int(self.lastaction)%8>3): rew=1.05*rew # increase rew for rational movements
        #assign zero reward for absolutely silly steps
        if (a==1) and (int(self.lastaction)%8>3): rew=0
        if (a==3) and (int(self.lastaction)%8==7): rew=0
        if (a==4) and (int(self.lastaction)%8==1): rew=0
        if (a==1) and (int(self.lastaction)%8==3): rew=0
        if (a==3) and (int(self.lastaction)%8==1): rew=0
        if (a==0) and (int(self.lastaction)%8==2): rew=0
        if (a==2) and (int(self.lastaction)%8==0): rew=0



        self.rew = rew

        if rew == 1:
            self.reset()
            return (self.desc, rew, 1, 0) #third param means "game over"
        return (self.desc, rew, 0, 0)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        desc = self.desc.tolist()
        desc = [[c for c in line] for line in desc]
        #desc[self.hand_row][self.hand_col] = utils.colorize(desc[self.hand_row][self.hand_col], "red", highlight=True)
        outfile.write("\n".join(''.join(str(line)) for line in desc)+"\n")

        if self.lastaction is not None:
            outfile.write("  ({})\n\n".format(["Left","Down","Right","Up","MoveL", "MoveD","MoveR","MoveU"][self.lastaction]))
        else:
            outfile.write("\n")
        return outfile
