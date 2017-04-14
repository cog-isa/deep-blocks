import logging
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gym_blocks.dqn_agents import DQNAgent


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
        return np.sqrt((cube_position[0]-target_position[0])**2 + (cube_position[0]-target_position[0])**2)/np.sqrt(2)*30
    else:
        return np.sqrt((cube_position[0]-hand_position[0])**2 + (cube_position[0]-hand_position[0])**2)/np.sqrt(2)*30

    print(len(dif), np.sum(np.abs(sum(dif))))


def preprocess_state(map):
    raveled = map.astype(np.float).ravel()
    return np.reshape(raveled, [1, 900])

def load_from_yaml(file_name):
    with open(file_name, 'r') as f:
        return yaml.load(f)


def load_from_csv(file_name):
    return np.asarray(pd.read_csv(file_name, sep=';'))


if __name__ == "__main__":
    logger = logging.getLogger()

    run_params = load_from_yaml(sys.argv[1])

    agent = DQNAgent(run_params['state_size'], run_params['action_size'])
    raw_map_start = load_from_csv(sys.argv[2])
    raw_map_final = load_from_csv(sys.argv[3])
    raw_map_mid01 = preprocess_state(load_from_csv(sys.argv[4]))
    raw_map_mid02 = preprocess_state(load_from_csv(sys.argv[5]))
    raw_map_mid03 = preprocess_state(load_from_csv(sys.argv[6]))
    raw_map_mid04 = preprocess_state(load_from_csv(sys.argv[7]))

    env = gym.make('Blocks-v0')
    env.configure(raw_map_start, raw_map_final, run_params['state_size'])
    games_won = 0
    subtarget1 = 0
    subtarget2 = 0
    subtarget3 = 0
    subtarget4 = 0

    mean_rewards = []
    actions = []
    for e in range(run_params['episodes']):
        observation = env.reset()
        current_rewards = [0]
        k = 11
        done1 = False
        done2 = False
        done3 = False
        done4 = False

        for time in range(run_params['max_steps']):
            observation = observation.reshape(1,1,1800)
            action = agent.act(observation)
            actions.append(action)
            if(np.sum(observation)-104)/9 < k:
                env.render()
                k=(np.sum(observation)-104)/9

            logger.info("Chosen: {}".format(action))
            next_observation, reward, done, hand_position = env.step(action)
            logger.info("Reward: {}".format(reward))
            if reward == 1: games_won += 1

            if len(current_rewards) > 0:
                if action < 4 and reward < np.mean(current_rewards):
                    reward = -1
            if len(current_rewards)>5:
                if action > 4 and reward < current_rewards[-1]:
                    reward = -1
            if action == np.mean(actions[-4:-1]):
                reward = -1
            if np.sum(next_observation-observation) == 0:
                reward = -1

            #
            # done1 = False
            # done2 = False
            # done3 = False
            # done4 = False


            if done1 == False:
                if get_distance(next_observation, raw_map_mid01, hand_position)==True:
                    done1 = True
                    reward = 1
                    subtarget1 +=1
                else: reward = get_distance(next_observation, raw_map_mid01, hand_position)

            if done2 == False:
                if get_distance(next_observation, raw_map_mid02, hand_position)==True:
                    done2 = True
                    reward = 1
                    subtarget2 +=1

                else: reward = get_distance(next_observation, raw_map_mid02, hand_position)

            if done3 == False:
                if get_distance(next_observation, raw_map_mid03, hand_position)==True:
                    done3 = True
                    reward = 1
                    subtarget3 +=1

                else: reward = get_distance(next_observation, raw_map_mid03, hand_position)

            if done4 == False:
                if get_distance(next_observation, raw_map_mid04, hand_position)==True:
                    done4 = True
                    reward = 1
                    subtarget4 +=1

                else: reward = get_distance(next_observation, raw_map_mid04, hand_position)


            agent.remember(observation, action, reward - current_rewards[-1], next_observation, done)
            current_rewards.append(reward)

            observation = next_observation
            if done:
                logger.info("Agent reach goal: episode={}/{}, steps={}".format(e, run_params['episodes'], time))
                break
        logger.info("Agent didn't reach goal")
        mean_rewards.append(np.mean(current_rewards))
        agent.replay(150)
        print("End Replay: {}".format(e))
        print("Won:", games_won, "games", subtarget1, subtarget2, subtarget3, subtarget4)

    #plt.plot(mean_rewards)
    #env.render()
    #plt.show()
    print("Won:", games_won, "games", subtarget1, subtarget2, subtarget3, subtarget4)
    agent.plot_loss()
