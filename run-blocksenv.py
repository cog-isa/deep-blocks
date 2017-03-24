import sys
import pandas as pd
import numpy as np

import gym
import logging

import yaml

from gym_blocks.agent.dqn_agents import DQNAgent
import matplotlib.pyplot as plt


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
    for e in range(run_params['episodes']):
        observation = env.reset()
        current_rewards = []
        for time in range(run_params['max_steps']):
            action = agent.act(observation)
            logger.info("Chosen: {}".format(action))
            next_observation, reward, done, _ = env.step(action)
            logger.info("Reward: {}".format(reward))
            reward = reward if not done else -10

            agent.remember(observation, action, reward, next_observation, done)
            current_rewards.append(reward)
            observation = next_observation
            if done:
                logger.info("Agent reach goal: episode={}/{}, steps={}"
                            .format(e, run_params['episodes'], time))
                break
        logger.info("Agent didn't reach goal")
        mean_rewards.append(np.mean(current_rewards))
        agent.replay(32)
    plt.plot(mean_rewards)
    plt.show()
