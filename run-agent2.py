import gym
import logging
import numpy as np
from gym_blocks.envs.blocks_env import BlocksEnv
import tensorflow  as tf
import matplotlib.pyplot as plt
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

# Input and output size based on the Env
input_size = 900
output_size = 8
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32) # state input
W = tf.Variable(tf.random_uniform([input_size, output_size],0,0.01)) # weight

Qpred = tf.matmul(X, W) # Out Q prediction
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32) # Y label

loss = tf.reduce_sum(tf.square(Y - Qpred))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

init = tf.global_variables_initializer()

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

from random import randint

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = prepro(env.reset())
        lastrew = 0
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []
        k=1
        # The Q-Network training
        while not done:
            # Choose an action by greedily (with e chance of random action) from the Q-network
            Qs = sess.run(Qpred, feed_dict={X: one_hot(find_hand_1d(s))})
            #if np.random.rand(1) < e:
            if np.random.rand(1) < 0.2:
                a = randint(0,7)
            else:
                a = np.argmax(Qs)
            print(a)
            # Get new state and reward from environment
            s1, reward, done, _ = env.step(a)
            s1 = prepro(s1)
            if reward == lastrew:
                dreward = reward - lastrew*sigmoid_array(k/1000)
            else: dreward = reward - lastrew
            lastrew = reward
            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0, a] = dreward
            else:
                # Obtain the Q_s1 values by feeding the new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(find_hand_1d(s1))})
                # Update Q
                Qs[0, a] = dreward + dis * np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(find_hand_1d(s1)), Y: Qs})
            rAll += dreward
            s = s1
            k+=1
            print(lastrew,reward, dreward,k)
            print("\n","NEW  STEP", dreward)
            env.render()
        rList.append(rAll)
        print('Finish episode')

print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()


