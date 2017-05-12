import logging
import sys
import random
#configs/blocks30x30.yaml data/map.csv data/target.csv data/target01.csv data/target02.csv data/target04.csv
#  data/target05.csv
from rendering import draw

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
import scipy.misc
import yaml
#from slideshow import slideshows
from gym_blocks.dqn_agents import DQNAgent
from rendering import draw
from time import gmtime, strftime
current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

def get_distance(map, target, hand_position):
    map = map[:, 0:900]
    dif = map - target
    #print(map.shape)
    dif = dif.reshape(30,30)
    for i in range(10):
        for j in range(10):
            if dif[3*i+1,3*j+1]==1 and dif[3*i,3*j]==1:
                cube_position = (i,j)

    for i in range(10):
        for j in range(10):
            if dif[3*i+1,3*j+1]==-1:
                target_position = (i,j)

    cube_position = list(cube_position)
    target_position = list(target_position)
    cube_position[0]=cube_position[0]*3+1
    cube_position[1]=cube_position[1]*3+1
    target_position[0]=target_position[0]*3+1
    target_position[1]=target_position[1]*3+1

    if cube_position == target_position: return True
    elif cube_position[0]-3 == hand_position[0] and cube_position[1] == hand_position[1]: #проверяем чтобы кубик который нужно передвинуть находился под рукой
        return np.sqrt((cube_position[0]-target_position[0])**2 + (cube_position[0]-target_position[0])**2)/(np.sqrt(2)*30)
    else:
        return np.sqrt((cube_position[0]-hand_position[0])**2 + (cube_position[0]-hand_position[0])**2)/(np.sqrt(2)*30)

    print(len(dif), np.sum(np.abs(sum(dif)))) #reward calculation  for subtargets


def preprocess_state(map):
    raveled = map.astype(np.float).ravel()
    return np.reshape(raveled, [1, 900])

def load_from_yaml(file_name):
    with open(file_name, 'r') as f:
        return yaml.load(f)


def load_from_csv(file_name):
    return np.asarray(pd.read_csv(file_name, sep=';'))



if __name__ == "__main__":
    #logging.basicConfig(filename='example.log', level=logging.DEBUG)


    #hdlr = logging.FileHandler('/Users/Edward/PycharmProjects/New_Blocks/deep-blocks/myapp.log')


    # create logger with 'spam_application'
    logger = logging.getLogger('deep_blocks')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(current_time+'.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)





    #agent = DQNAgent(run_params['state_size'], run_params['action_size']) #agent's initialisation

    #Loading initial, final and intermidiate states
    raw_map_start = load_from_csv(sys.argv[2])
    raw_map_final = load_from_csv(sys.argv[3])
    raw_map_mid01 = preprocess_state(load_from_csv(sys.argv[4]))
    raw_map_mid02 = preprocess_state(load_from_csv(sys.argv[5]))
    raw_map_mid03 = preprocess_state(load_from_csv(sys.argv[6]))
    raw_map_mid04 = preprocess_state(load_from_csv(sys.argv[7]))

    env = gym.make('Blocks-v0')


    def get_distance(map, target, hand_position):
        map = map[:, 0:900]
        dif = map - target
        # print(map.shape)
        dif = dif.reshape(30, 30)
        for i in range(10):
            for j in range(10):
                if dif[3 * i + 1, 3 * j + 1] == 1 and dif[3 * i, 3 * j] == 1:
                    cube_position = (i, j)

        for i in range(10):
            for j in range(10):
                if dif[3 * i + 1, 3 * j + 1] == -1:
                    target_position = (i, j)

        cube_position = list(cube_position)
        target_position = list(target_position)
        cube_position[0] = cube_position[0] * 3 + 1
        cube_position[1] = cube_position[1] * 3 + 1
        target_position[0] = target_position[0] * 3 + 1
        target_position[1] = target_position[1] * 3 + 1

        if cube_position == target_position:
            return True
        elif cube_position[0] - 3 == hand_position[0] and cube_position[1] == hand_position[
            1]:  # проверяем чтобы кубик который нужно передвинуть находился под рукой
            return np.sqrt(
                (cube_position[0] - target_position[0]) ** 2 + (cube_position[0] - target_position[0]) ** 2) / (
                   np.sqrt(2) * 30)
        else:
            return np.sqrt((cube_position[0] - hand_position[0]) ** 2 + (cube_position[0] - hand_position[0]) ** 2) / (
            np.sqrt(2) * 30)

        print(len(dif), np.sum(np.abs(sum(dif))))  # reward calculation  for subtargets


    def preprocess_state(map):
        raveled = map.astype(np.float).ravel()
        raveled = raveled[:900]
        return np.reshape(raveled, [1, 900])

    run_params = load_from_yaml(sys.argv[1])

    env.configure(raw_map_start, raw_map_final, run_params['state_size'])

    #Initialize table with all zeros
    Q = np.zeros([100,8])
    # Set learning parameters
    lr = .85
    e=-.9
    y = .99
    num_episodes = 10
    #create lists to contain total rewards and steps per episode
    #jList = []
    rList = []
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        s=35
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            #Choose an action by greedily (with noise) picking from Q table
            if np.random.randn() < e:
                a = np.argmax(Q[s,:] + np.random.randn(1,8)*(1./(i+1)))
            else: a = np.random.randint(8)
            #Get new state and reward from environment
            s1,r,d,q = env.step(a)
            draw(s1,j,i)

            s1=int(10*(q[0]+2)/3-10+(q[1]+2)/3)-1
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            print(r)
            s = s1
            print('PLAYED')
            if d == True:
                break
        #jList.append(j)
        rList.append(rAll)
    print(rAll)
    print(Q)
    print("Score over time: " + str(sum(rList) / num_episodes))
