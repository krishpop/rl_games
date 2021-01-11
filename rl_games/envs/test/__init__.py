import gym

gym.envs.register(
     id='TestRnnEnv-v0',
     entry_point='rl_games.envs.test.rnn_env:TestRNNEnv',
     max_episode_steps=50,
)

gym.envs.register(
     id='TestAsymmetricEnv-v0',
     entry_point='rl_games.envs.test.test_asymmetric_env:TestAsymmetricCritic'
)

gym.envs.register(
     id='TestUAV-v0',
     entry_point='rl_games.envs.test.uav_env:UAV_nav'
)