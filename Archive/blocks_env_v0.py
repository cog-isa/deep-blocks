#старая версия среды

import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete
import tensorflow
import gym
from gym import spaces
from gym.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
GRAB = 4
PUT = 5

MAPS = {
    "cub2": [["OOOOOOOM",
        "OOOOOOOO",
        "OOOOOOOO",
        "OOOOOOOO",
        "EEEEEEEE",
        "EEEEEEEE",
        "EEEECEEE",
        "ECCCCCEE"
    ],False],"target": [["OOOOOOOM",
        "OOOOOOOO",
        "OOOOOOOO",
        "OOOOOOOO",
        "EEEEEEEE",
        "EEEEEEEE",
        "EEECCEEE",
        "EECCCCEE"
    ],False]
}


class BlocksEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="cub2"):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = np.array(MAPS[map_name])
        self.nrow = nrow = len(desc[0])/2
        self.ncol = ncol = len(desc[0][0])
        
        hand = "".join(desc[0][:4])
        self.hand = hand = np.asarray(hand,dtype='c')
        cubes = "".join(desc[0][4:])
        self.cubes = cubes = np.asarray(cubes,dtype='c')
        self.desc = desc[0] = np.asarray(desc[0],dtype='c')  

        self.desc_m = desc[0][:4] = np.asarray(desc[0],dtype='c')[:4]
        self.desc_c = desc[0][4:] = np.asarray(desc[0],dtype='c')[4:]  

        self.grabbed = grabbed = desc[1]

        nA = 6
        nS = int(2 * nrow * ncol )#способы расположения манипулятора
        
        hand_vector = np.array(hand == b'M').astype('float64').ravel()
        #start = np.argmax(start)
        
        cubes_vector = np.array(cubes == b'C').astype('float64').ravel()
        #ids /= ids.sum()
        ids = (hand_vector, cubes_vector, grabbed) 
        self.ids =ids
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        self.target  = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,
        1.,  1.,  1.,  1.,  0.,  0.])
        def to_s(row, col):
            return row*ncol + col

        def inc(a, ids):
            row, col = ids[0].argmax()//8, np.mod(ids[0].argmax(),8)
            if ids[2]==False: #рука свободна, значит просто двигается
                if a==0: #left
                    ids[0][to_s(row, col)]=0
                    col = max(col-1,0)
                    ids[0][to_s(row, col)]=1
                elif a==1: #down
                    ids[0][to_s(row, col)]=0
                    row = min(row+1,nrow-1)
                    ids[0][to_s(row, col)]=1
                elif a==2: #right
                    ids[0][to_s(row, col)]=0
                    col = min(col+1,ncol-1)
                    ids[0][to_s(row, col)]=1
                elif a==3: #up
                    ids[0][to_s(row, col)]=0
                    row = max(row-1,0)
                    ids[0][to_s(row, col)]=1
                elif a ==4:
                    if ids[1][to_s(row, col)]==1:
                        ids[2] = True
                        ids[1][to_s(row, col)]=0
            else: #рука занята
                if a==0:
                    ids[0][to_s(row, col)]=0
                    col = max(col-1,0)
                    ids[0][to_s(row, col)]=1
                elif a==1:
                    ids[0][to_s(row, col)]=0
                    row = min(row+1,nrow-1)
                    ids[0][to_s(row, col)]=1
                elif a==2:
                    ids[0][to_s(row, col)]=0
                    col = min(col+1,ncol-1)
                    ids[0][to_s(row, col)]=1
                elif a==3:
                    ids[0][to_s(row, col)]=0
                    row = max(row-1,0)
                    ids[0][to_s(row, col)]=1
                elif a==5:
                    if row ==0 or (ids[1][to_s(row, col)]==0 and ids[1][to_s(row-1, col)]==1):
                        ids[1][to_s(row, col)]=1
                        ids[2] = False
            return ids
                

        
        target = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,
        1.,  1.,  1.,  1.,  0.,  0.])
        
        for row in range(int(nrow)):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(6):
                    li = P[s][a]
                    #letter = desc[row, col]
                    letter = cubes[s]
                    if cubes_vector.all()==target.all():
                        li.append((1.0, s, 0, True))
                    else:
                        ids = inc(a, ids)
                        row, col = ids[0].argmax()//8, np.mod(ids[0].argmax(),8)

                        newstate = to_s(row, col)
                        letter = cubes[newstate]
                        done = False
                        rew = np.corrcoef(target, ids[1], rowvar=1)
                        li.append((1.0, newstate, rew, done))

        super(BlocksEnv, self).__init__(nS, nA, P, ids)

    def _reset(self):
         self.s = np.random.choice(32)
         return self.s

    def _step(self, a):
        #transitions = self.P[self.s][a]
        #i = categorical_sample([t[0] for t in transitions], self.np_random)
        #p, s, r, d= transitions[i]
        #self.s = s
        #self.lastaction=a
        def to_s(row, col):
            return row*self.ncol + col
        def inc(a, ids):
            row, col = ids[0].argmax() // 8, np.mod(ids[0].argmax(), 8)
            if ids[2] == False:  # рука свободна, значит просто двигается
                if a == 0:  # left
                    ids[0][to_s(row, col)] = 0
                    col = max(col - 1, 0)
                    ids[0][to_s(row, col)] = 1
                elif a == 1:  # down
                    ids[0][to_s(row, col)] = 0
                    row = min(row + 1, self.nrow - 1)
                    ids[0][to_s(row, col)] = 1
                elif a == 2:  # right
                    ids[0][to_s(row, col)] = 0
                    col = min(col + 1, self.ncol - 1)
                    ids[0][to_s(row, col)] = 1
                elif a == 3:  # up
                    ids[0][to_s(row, col)] = 0
                    row = max(row - 1, 0)
                    ids[0][to_s(row, col)] = 1
                elif a == 4:
                    if ids[1][to_s(row, col)] == 1:
                        ids[2] = True
                        ids[1][to_s(row, col)] = 0
            else:  # рука занята
                if a == 0:
                    ids[0][to_s(row, col)] = 0
                    col = max(col - 1, 0)
                    ids[0][to_s(row, col)] = 1
                elif a == 1:
                    ids[0][to_s(row, col)] = 0
                    row = min(row + 1, self.nrow - 1)
                    ids[0][to_s(row, col)] = 1
                elif a == 2:
                    ids[0][to_s(row, col)] = 0
                    col = min(col + 1, self.ncol - 1)
                    ids[0][to_s(row, col)] = 1
                elif a == 3:
                    ids[0][to_s(row, col)] = 0
                    row = max(row - 1, 0)
                    ids[0][to_s(row, col)] = 1
                elif a == 5:
                    if row == 0 or (ids[1][to_s(row, col)] == 0 and ids[1][to_s(row - 1, col)] == 1):
                        ids[1][to_s(row, col)] = 1
                        ids[2] = False
            return ids
        inc(a, self.ids)
        rew = np.mean(np.corrcoef(self.target, self.ids[1], rowvar=1))+0.25*(self.ids[2]==True)
        if rew.all() == 1:
            self.reset()
            return (self.isd, rew, 1,0)
        return (self.isd, rew, 0, 0)
        #return (s, r, d, {"prob" : p})
    
    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.ids[0].argmax() // self.ncol, self.ids[0].argmax() % self.ncol
        desc_m = self.desc_m.tolist()
        desc = self.desc.tolist()
        print(desc)
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc_m = [[c.decode('utf-8') for c in line] for line in desc_m]

        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up","Grab", "Put"][self.lastaction]))
        else:
            outfile.write("\n")
        return outfile
