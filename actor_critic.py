import gym
import logging
import math
import numpy as np
from gym_blocks.envs.blocks_env import BlocksEnv
import tensorflow  as tf
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
import pylab

env = gym.make('Blocks-v0')
def to_onehot(size,value):

  my_onehot = np.zeros((size))
  my_onehot[value] = 1.0
  return my_onehot
OBSERVATION_SPACE=100
ACTION_SPACE=8

from keras.models import Sequential
from keras.layers.core import Dense,  Activation
from keras.optimizers import RMSprop, SGD
from keras.layers import LSTM, Dropout
actor_model = Sequential()
actor_model.add(Dense(32, init='lecun_uniform', input_shape=(100,)))
actor_model.add(Activation('relu'))

actor_model.add(Dense(20, init='lecun_uniform'))
actor_model.add(Activation('relu'))
actor_model.add(Dropout(0.2))


actor_model.add(Dense(8, init='lecun_uniform'))
actor_model.add(Activation('linear'))

a_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
actor_model.compile(loss='mse', optimizer=a_optimizer)

critic_model = Sequential()

critic_model = Sequential()
critic_model.add(Dense(32, init='lecun_uniform', input_shape=(100,)))
critic_model.add(Activation('relu'))
critic_model.add(Dense(20, init='lecun_uniform'))
critic_model.add(Activation('relu'))
critic_model.add(Dense(1, init='lecun_uniform'))
critic_model.add(Activation('linear'))

c_optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
critic_model.compile(loss='mse', optimizer=c_optimizer)


# Plot out the values the critic gives for the agent being in
# a specific state, i.e. in a specific location in the e


import random
import time


def trainer(epochs=1000, batchSize=40,
            gamma=0.975, epsilon=1, min_epsilon=0.1,
            buffer=80):
    wins = 0
    losses = 0
    # Replay buffers
    actor_replay = []
    critic_replay = []

    def get_state(x):  # returns current hand center's location
        for i in range(10):
            for k in range(10):
                if (x[3 * i + 1][3 * k + 1] == 1) and (x[3 * i + 1][3 * k + 2] == 0) and (x[3 * i + 1][3 * k] == 0):
                    return 10*i +k+1

    for i in range(epochs):

        observation = 3*10+5
        done = False
        reward = 0
        info = None
        move_counter = 0



        while (not done):
            # Get original state, original reward, and critic's value for this state.
            orig_state = to_onehot(OBSERVATION_SPACE, observation)
            orig_reward = reward
            orig_val = critic_model.predict(orig_state.reshape(1, OBSERVATION_SPACE))

            if (random.random() < epsilon):  # choose random action
                action = np.random.randint(0, ACTION_SPACE)
            else:  # choose best action from Q(s,a) values
                qval = actor_model.predict(orig_state.reshape(1, OBSERVATION_SPACE))
                action = (np.argmax(qval))
            print("Action: :", action)
            # Take action, observe new state S'
            new_observation, new_reward, done, info = env.step(action)
            new_observation = get_state(new_observation)

            print("OBSERVATION: ", int(new_observation))
            env.render()
            new_state = to_onehot(OBSERVATION_SPACE, int(new_observation))
            # Critic's value for this new state.
            new_val = critic_model.predict(new_state.reshape(1, OBSERVATION_SPACE))

            if not done:  # Non-terminal state.
                target = orig_reward + (gamma * new_val)
            else:
                # In terminal states, the environment tells us
                # the value directly.
                target = orig_reward + (gamma * new_reward)

            # For our critic, we select the best/highest value.. The
            # value for this state is based on if the agent selected
            # the best possible moves from this state forward.
            #
            # BTW, we discount an original value provided by the
            # value network, to handle cases where its spitting
            # out unreasonably high values.. naturally decaying
            # these values to something reasonable.
            best_val = max((orig_val * gamma), target)

            # Now append this to our critic replay buffer.
            critic_replay.append([orig_state, best_val])
            # If we are in a terminal state, append a replay for it also.
            if done:
                critic_replay.append([new_state, float(new_reward)])

            # Build the update for the Actor. The actor is updated
            # by using the difference of the value the critic
            # placed on the old state vs. the value the critic
            # places on the new state.. encouraging the actor
            # to move into more valuable states.
            actor_delta = new_val - orig_val
            actor_replay.append([orig_state, action, actor_delta])

            # Critic Replays...
            while (len(critic_replay) > buffer):  # Trim replay buffer
                critic_replay.pop(0)
            # Start training when we have enough samples.
            if (len(critic_replay) >= buffer):
                minibatch = random.sample(critic_replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    m_state, m_value = memory
                    y = np.empty([1])
                    y[0] = m_value
                    X_train.append(m_state.reshape((OBSERVATION_SPACE,)))
                    y_train.append(y.reshape((1,)))
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                critic_model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)

            # Actor Replays...
            while (len(actor_replay) > buffer):
                actor_replay.pop(0)
            if (len(actor_replay) >= buffer):
                X_train = []
                y_train = []
                minibatch = random.sample(actor_replay, batchSize)
                for memory in minibatch:
                    m_orig_state, m_action, m_value = memory
                    old_qval = actor_model.predict(m_orig_state.reshape(1, OBSERVATION_SPACE, ))
                    y = np.zeros((1, ACTION_SPACE))
                    y[:] = old_qval[:]
                    y[0][m_action] = m_value
                    X_train.append(m_orig_state.reshape((OBSERVATION_SPACE,)))
                    y_train.append(y.reshape((ACTION_SPACE,)))
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                actor_model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)

            # Bookkeeping at the end of the turn.
            observation = new_observation
            reward = new_reward
            move_counter += 1
            if done:
                if new_reward > 0:  # Win
                    wins += 1
                else:  # Loss
                    losses += 1
        # Finised Epoch
        # clear_output(wait=True)
        print("Game #: %s" % (i,))
        print("Moves this round %s" % move_counter)
        print("Final Position:")
        print(env.render())
        print("Wins/Losses %s/%s" % (wins, losses))
        if epsilon > min_epsilon:
            epsilon -= (1 / epochs)


trainer()

