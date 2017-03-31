import logging
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gym_blocks.dqn_agents import DQNAgent


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
    env = gym.make('Blocks-v0')
    env.configure(raw_map_start, raw_map_final, run_params['state_size'])

    mean_rewards = []
    actions = []
    for e in range(run_params['episodes']):
        observation = env.reset()
        current_rewards = []
        k = 11
        for time in range(run_params['max_steps']):
            #x = np.ndarray(shape=(1, 1, 1800)).astype(K.floatx())
            observation = observation.reshape(1,1,1800)
            action = agent.act(observation)
            actions.append(action)
            #print("Action:", action, "Blocks: ", (np.sum(observation)-104)/9)
            if(np.sum(observation)-104)/9 < k:
                env.render()
                k=(np.sum(observation)-104)/9


            logger.info("Chosen: {}".format(action))
            next_observation, reward, done, _ = env.step(action)
            logger.info("Reward: {}".format(reward))
            #reward = reward if not done else 10
            current_rewards.append(reward)

            if len(current_rewards) > 0:
                if action < 4 and reward < np.mean(current_rewards):
                    reward = 0
            if action > 4 and reward < current_rewards[-1]:
                reward = 0
            if action == np.mean(actions[-4:-1]):
                reward = 0
            if np.sum(next_observation-observation) == 0:
                reward = 0
            current_rewards.append(reward)


            agent.remember(observation, action, reward - current_rewards[-2], next_observation, done)

            observation = next_observation
            if done:
                logger.info("Agent reach goal: episode={}/{}, steps={}".format(e, run_params['episodes'], time))
                break
        logger.info("Agent didn't reach goal")
        mean_rewards.append(np.mean(current_rewards))
        agent.replay(150)
        print("End Replay: {}".format(e))
    #plt.plot(mean_rewards)
    env.render()
    #plt.show()
    agent.plot_loss()
