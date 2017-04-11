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
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.e_decay = .99
        self.e_min = 0.05
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.output_dim = 8

    def _build_model(self):
        #Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(output_dim=(100), input_shape = (1, 1800)))
        #model.add(Conv1D(4, 1, input_shape=(1, 1800)))
        #model.add(Flatten())
        model.add(Dense(1000, input_dim=100, activation='tanh'))
        model.add(Dropout(0.25))
        model.add(Dense(100, activation='tanh', init='uniform'))

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
        minibatch = self.memory
        minibatch = list(minibatch)[-2000:-1]
        random.shuffle(minibatch, random.random)
        #minibatch = list(random.shuffle(self.memory))
        X = np.zeros((2000, 1800))
        Y = np.zeros((2000, self.action_size))
        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state.reshape(1,1,1800))[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                                          np.amax(self.model.predict(next_state.reshape(1,1,1800))[0])
            state = state.reshape(1,1,1800)
            X[i], Y[i] = state, target
        X = X.reshape(2000,1,1800)
        self.fitted = self.model.fit(X, Y, batch_size=batch_size, verbose=0, nb_epoch=5)
        print("Fitted ")
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay


    # def replay(self, batch_size):
    #     batch_size = min(batch_size, len(self.memory))
    #     minibatch = random.sample(self.memory, batch_size)
    #     X = np.zeros((batch_size, 1800))
    #     Y = np.zeros((batch_size, self.action_size))
    #     for i in range(batch_size):
    #         state, action, reward, next_state, done = minibatch[i]
    #         target = self.model.predict(state.reshape(1,1,1800))[0]
    #         if done:
    #             target[action] = reward
    #         else:
    #             target[action] = reward + self.gamma * \
    #                                       np.amax(self.model.predict(next_state.reshape(1,1,1800))[0])
    #         state = state.reshape(1,1,1800)
    #         X[i], Y[i] = state, target
    #     X = X.reshape(150,1,1800)
    #     self.fitted =self.model.fit(X, Y, batch_size=batch_size, verbose=1, nb_epoch=5)
    #     if self.epsilon > self.e_min:
    #         self.epsilon *= self.e_decay


    def plot_loss(self):
        # summarize history for loss
        plt.plot(self.fitted.history['loss'])
        #plt.plot(history.losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('episodes')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
