import gym
env = gym.make('Blocks-v0')

import kerlym
agent = kerlym.agents.DQN(
                    env=env,
                    nframes=1,
                    epsilon=0.5,
                    discount=0.99,
                    modelfactory=kerlym.dqn.networks.simple_cnn,
                    batch_size=32,
                    dropout=0.1,
                    enable_plots = True,
                    epsilon_schedule=lambda episode,epsilon: max(0.1, epsilon*(1-1e-4)),
                    dufference_obs = True,
                    preprocessor = kerlym.preproc.karpathy_preproc,
                    learning_rate = 1e-4,
                    render=True
                    )
agent.train()