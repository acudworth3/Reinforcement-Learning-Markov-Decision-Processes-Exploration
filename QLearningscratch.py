import gym
import gym_hanoi
import numpy as np
from toh_gym.envs import TohEnv




env = TohEnv()
env.reset() # reset environment to a new, random state

# from gym.envs.toy_text.frozen_lake import generate_random_map
# np.random.seed(777)
# random_map = generate_random_map(size=20, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)


print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))



q_table = np.zeros([env.observation_space.n, env.action_space.n])


"""Training the agent"""

import random


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
training_episodes = 100

for i in range(1, training_episodes):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    terminal = False
    
    while not terminal:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, terminal, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward < 0:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 1000 == 0:

        print(f"Episode: {i}")

print("Training finished.\n")

total_epochs, total_rewards = 0, 0
test_episodes = 100

for _ in range(test_episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        total_rewards += reward

        epochs += 1
        if epochs > 5000:
            done = True


    total_epochs += epochs

print("training Episodes: ",training_episodes)
print(f"Results after {test_episodes} test episodes:")
print(f"Average timesteps per episode: {total_epochs / test_episodes}")
print(f"Average reward per episode: {total_rewards / test_episodes}")