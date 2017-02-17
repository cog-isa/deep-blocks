from gym.envs.registration import register

register(
    id='Blocks-v0',
    entry_point='gym_blocks.envs.blocks_env:BlocksEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)
