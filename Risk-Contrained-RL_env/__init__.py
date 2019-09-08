from gym.envs.registration import register

register(
    id='CustomEnv',
    entry_point='Risk-Constrained-RL_env.envs:CustomEnv',
)
