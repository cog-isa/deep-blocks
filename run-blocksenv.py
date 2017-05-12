import logging
import sys

#configs/blocks30x30.yaml data/map.csv data/target.csv data/target01.csv data/target02.csv data/target04.csv
#  data/target05.csv

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




    run_params = load_from_yaml(sys.argv[1])

    agent = DQNAgent(run_params['state_size'], run_params['action_size']) #agent's initialisation

    #Loading initial, final and intermidiate states
    raw_map_start = load_from_csv(sys.argv[2])
    raw_map_final = load_from_csv(sys.argv[3])
    raw_map_mid01 = preprocess_state(load_from_csv(sys.argv[4]))
    raw_map_mid02 = preprocess_state(load_from_csv(sys.argv[5]))
    raw_map_mid03 = preprocess_state(load_from_csv(sys.argv[6]))
    raw_map_mid04 = preprocess_state(load_from_csv(sys.argv[7]))

    env = gym.make('Blocks-v0')
    env.configure(raw_map_start, raw_map_final, run_params['state_size'])

    # counters for targtets
    games_won = 0
    subtarget1 = 0
    subtarget2 = 0
    subtarget3 = 0
    subtarget4 = 0

    actions = []

    for e in range(run_params['episodes']):
        observation = env.reset()
        current_rewards = [0]
        coefs =[0]
        #k = 11
        done1 = False
        done2 = False
        done3 = False
        done4 = False

        for time in range(run_params['max_steps']):
            observation = observation.reshape(1,1,1800)
            action = agent.act(observation)
            qvalues = agent.qvalues(observation)
            actions.append(action)
            #if(np.sum(observation)-104)/9 < k:
            #   env.render()
            #  k=(np.sum(observation)-104)/9
            logger.info("Start episode={}/{}, steps={}".format(e, run_params['episodes'], time))
            logger.info("Chosen: {}".format(action))
            logger.info("Q-values: {}".format(qvalues))

            next_observation, reward, done, hand_position = env.step(action)
            logger.info("Initial reward from env: {}".format(reward))

            draw(next_observation,time,e)
            if reward == 1: games_won += 1

            # if len(current_rewards) > 0:
            #     if action < 4 and reward < np.mean(current_rewards):
            #         reward = 0.05
            # if len(current_rewards)>5:
            #     if action > 4 and reward < current_rewards[-1]:
            #         reward = 0.05
            # if action == np.mean(actions[-4:-1]):
            #     reward = 0.05
            if np.sum(np.abs(next_observation-observation)) == 0:
                reward = 0.1
                print("NO CHANGES")
            else: print("OK")
            #if action == 1: reward = 0.8
            #if action == 0 or action == 2: reward = 0.4
            if done1 == False:
                if get_distance(next_observation, raw_map_mid01, hand_position)==True:
                    done1 = True
                    reward = 1
                    subtarget1 +=1
                    print("Done 1")
                else:
                    coef = 1 - get_distance(next_observation, raw_map_mid01, hand_position)
                    print(reward, coef)

            if done2 == False:
                if get_distance(next_observation, raw_map_mid02, hand_position)==True:
                    done2 = True
                    reward = 1
                    subtarget2 +=1

                else: coef = 1 - get_distance(next_observation, raw_map_mid02, hand_position)

            if done3 == False:
                if get_distance(next_observation, raw_map_mid03, hand_position)==True:
                    done3 = True
                    reward = 1
                    subtarget3 +=1

                else: coef = 1 - get_distance(next_observation, raw_map_mid03, hand_position)

            if done4 == False:
                if get_distance(next_observation, raw_map_mid04, hand_position)==True:
                    done4 = True
                    reward = 1
                    subtarget4 +=1

                else: coef = 1 - get_distance(next_observation, raw_map_mid04, hand_position)
            logger.info("Reward {} and coef: {}".format(reward, coef))

            agent.remember(observation, action, reward*coef-coefs[-1]*current_rewards[-1], next_observation, done)
            coefs.append(coef)
            current_rewards.append(reward)



            observation = next_observation

            if done:
                logger.info("Agent reach goal`: episode={}/{}, steps={}".format(e, run_params['episodes'], time))
                break

        logger.info("Agent didn't reach goal")
        #slideshows()
        agent.replay(150)
        print("End Replay: {}".format(e))
        print("Won:", games_won, "games", subtarget1, subtarget2, subtarget3, subtarget4)


    #env.render()
    #plt.show()
    print("Won:", games_won, "games", subtarget1, subtarget2, subtarget3, subtarget4)
    agent.plot_loss()
