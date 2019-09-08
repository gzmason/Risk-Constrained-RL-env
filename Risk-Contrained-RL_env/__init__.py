from gym.envs.registration import register

register(
    id='rl_env',
    entry_point='Risk-Constrained-RL_env.envs:CustomEnv',
)
