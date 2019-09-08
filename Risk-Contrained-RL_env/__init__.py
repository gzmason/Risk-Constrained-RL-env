from gym.envs.registration import register

register(
    id='custom-v0',
    entry_point='Risk-Constrained-RL_env.envs:CustomEnv',
)
