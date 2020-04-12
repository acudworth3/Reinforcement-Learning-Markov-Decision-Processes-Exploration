import gym
import gym_hanoi
import numpy as np
from toh_gym.envs import TohEnv




env = TohEnv()
# env = gym.make('Hanoi-v0')
# env.set_env_parameters(num_disks=5)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        # env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()