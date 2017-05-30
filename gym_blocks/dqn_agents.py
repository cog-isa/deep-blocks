import random
from collections import deque

import numpy as np
from keras.layers import Conv1D, Flatten, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.5  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.e_decay = .90
        self.e_min = 0.05
        self.learning_rate = 0.2
        self.model = self._build_model()
        self.output_dim = 8
        self.losses = []
        self.X = np.zeros((2000, 1800))
        self.Y = np.zeros((2000, 8))

    def _build_model(self):
        #Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(output_dim=(200),  return_sequences=True, input_shape = (1, 1800)))
        #model.add(LSTM(output_dim=(200),return_sequences=True))
        #model.add(LSTM(output_dim=(200)))
        #keras.initializers.RandomNormal(mean=5, stddev=1, seed=None)
        model.add(Conv1D(10, 1, input_shape=(1, 1800)))
        model.add(Flatten())
        model.add(Dense(10, input_dim=200, init='uniform', activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='sigmoid', init='uniform')) #was tanh

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

    def qvalues(self, state):
        act_values = self.model.predict(state)
        return act_values[0]  # returns action


    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = self.memory
        minibatch = list(minibatch)[-2000:-1]
        random.shuffle(minibatch, random.random)
        #minibatch = list(random.shuffle(self.memory))
        # X = np.zeros((2000, 1800))
        # Y = np.zeros((2000, self.action_size))
        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state.reshape(1,1,1800))[0]

            if done:
                 target[action] = reward
            else:
                 target[action] = reward + 5*self.gamma * \
                                           np.amax(self.model.predict(next_state.reshape(1,1,1800))[0])
            state = state.reshape(1,1,1800)
            self.X[i], self.Y[i] = state, target
        self.X = self.X.reshape(2000, 1,1800)

        self.fitted = self.model.fit(self.X, self.Y, batch_size=batch_size, verbose=0, nb_epoch=3)

        self.losses.append(self.fitted.history['loss'])
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay


    def plot_loss(self):
        # summarize history for loss
        plt.plot(self.losses)
        #plt.plot(history.losses)
        plt.title('model loss')
        plt.ylabel('loss')
        #plt.xlabel('episodes')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
