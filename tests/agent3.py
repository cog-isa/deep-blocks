import random
import gym
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv1D

from gym_blocks.envs.blocks_env import BlocksEnv
import tensorflow  as tf
import matplotlib.pyplot as plt
#logger = logging.getLogger(__name__)

EPISODES = 100


def one_hot(x):
    c = np.zeros(900)
    c[x]=1
    return c.reshape(1,900)


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def find_hand_1d(x):  # returns current hand center's location
    for i in range(len(x)):
            if (x[i] == 1) and (x[i + 1]== 0): return int(i)

def prepro(I):
    return I.astype(np.float).ravel()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.e_decay = .99
        self.e_min = 0.05
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        #model.add(Conv1D(16, 1, input_shape = (1,1800)))
        #model.add(Flatten())
        model.add(Dense(20, input_dim=1800, activation='tanh'))
        model.add(Dropout(0.4))
        model.add(Dense(20, activation='tanh', init='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size, 1800))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            state.reshape(state.shape + (1,))
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                            np.amax(self.model.predict(next_state)[0])
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('Blocks-v0')
  #  state_size = env.observation_space.shape[0]
    state_size = 900

    #action_size = env.action_space.n
    action_size = 8

    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-master.h5")
    y = np.asarray(pd.read_csv('gym_blocks/envs/target.csv', sep=';'))  # final position of blocks
    y = prepro(y)
    y = np.reshape(y, [1, state_size])
    print(len(y), y.shape )
    mean_rewards = []
    for e in range(EPISODES):
        state = prepro(env.reset())
        state = np.reshape(state, [1, 900])
        state = np.hstack((state, y))
        current_rewards = []
        for time in range(1000):
            action = agent.act(state)
            print("Chosen: ", action)
            next_state, reward, done, _ = env.step(action)
            print("Reward: ", reward)
            reward = reward if not done else -10
            next_statestate = prepro(next_state)
            next_state = np.reshape(next_state, [1, 900])
            next_state = np.hstack((next_state,y))

            agent.remember(state, action, reward, next_state, done)
            current_rewards.append(reward)
            state = next_state
            if done or time == 999:
                print("episode: {}/{}, score: {}, e: {:.2}"
                        .format(e, EPISODES, time, agent.epsilon))
                break
        mean_rewards.append(np.mean(current_rewards))
        agent.replay(32)
        # if e % 10 == 0:
            # agent.save("./save/cartpole.h5")
    plt.plot(mean_rewards)
    plt.show()