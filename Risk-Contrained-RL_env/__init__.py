from gym.envs.registration import register

register(
    id='custom-env',
    entry_point='Risk-Constrained-RL_env.envs:CustomEnv',
)
