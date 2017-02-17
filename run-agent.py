import gym
import logging
from sklearn.neural_network import MLPClassifier
import numpy as np
from gym_blocks.envs.blocks_env import BlocksEnv

logger = logging.getLogger(__name__)


env = gym.make('Blocks-v0')




def prepro(I):
    # """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    # I = I[35:195]  # crop
    # I = I[::2, ::2, 0]  # downsample by factor of 2
    # I[I == 144] = 0  # erase background (background type 1)
    # I[I == 109] = 0  # erase background (background type 2)
    # I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(6))
print(env.step(7))
print(env.step(4))
print(env.step(4))
print(env.step(3))
print(env.render())






# #MLPClassifier(hidden_layer_sizes=(100, ),
#                                      activation='relu', solver='adam',
#                                      alpha=0.0001, batch_size='auto',
#                                      beta_1=0.9, beta_2=0.999, epsilon=1e-08)

