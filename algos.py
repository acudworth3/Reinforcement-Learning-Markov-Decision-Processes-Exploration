#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning Overview

# It is the area of Machine learning that deal with how an agent behave in an enviorment. This notebook covers two fundamental algorithms to solve MDPs namely **Value Iteration** and **Policy Iteration**.

# ## Markov Decision Process - MDP

# In MDP, there is an agent. The agent choose an action $a_{t}$ at time $t$ and as a consequance the enviorment changes.
# Here The evniorment is world around the agent. After the action the enviorment state changes to $s_{t+1}$.
# A reward might be emitted assciated with what just happened and then this process repeats. ![](nb_images/mdp.png)
# 
# So, there is a feedback cycle in that the next action you take, the next decision you make is in a situation that's the consiquence of what you did before.

# ## 1. Import libraries

# In[1]:


import numpy as np
import gym
import gym.spaces as spaces
import time
# import gym_hanoi
# from toh_gym.envs import TohEnv
import random


# In[2]:


# action mapping for display the final result
# action_mapping = {
#     3: '\u2191', # UP
#     2: '\u2192', # RIGHT
#     1: '\u2193', # DOWN
#     0: '\u2190' # LEFT
# }
# print(' '.join([action_mapping[i] for i in range(4)]))


# ## 2. Setup GYM Env for playing

# we define a faction that will take a GYM enviorment and plays number of games according to given policy.

# In[3]:


def play_episodes(enviorment, n_episodes, policy, random = False):
    """
    This fucntion plays the given number of episodes given by following a policy or sample randomly from action_space.
    
    Parameters:
        enviorment: openAI GYM object
        n_episodes: number of episodes to run
        policy: Policy to follow while playing an episode
        random: Flag for taking random actions. if True no policy would be followed and action will be taken randomly
        
    Return:
        wins: Total number of wins playing n_episodes
        total_reward: Total reward of n_episodes
        avg_reward: Average reward of n_episodes
    
    """
    # intialize wins and total reward
    wins = 0
    total_reward = 0
    
    # loop over number of episodes to play
    for episode in range(n_episodes):
        
        # flag to check if the game is finished
        terminated = False
        
        # reset the enviorment every time when playing a new episode
        state = enviorment.reset()
        step_count = 0
        while not terminated:
            
            # check if the random flag is not true then follow the given policy other wise take random action
            if random:
                action = enviorment.action_space.sample()
            else:
                action = policy[state]

            # take the next step
            next_state, reward,  terminated, info = enviorment.step(action)
            step_count += 1
            
            # enviorment.render()
            
            # accumalate total reward
            total_reward += reward
            
            # change the state
            state = next_state
            
            # if game is over with positive reward then add 1.0 in wins
            if terminated and reward >= 1.0:
                wins += 1
            if step_count >= 1000:
                terminated = True
                
    # calculate average reward
    average_reward = total_reward / n_episodes
    
    return wins, total_reward, average_reward
            

def one_step_lookahead(env, state, V , discount_factor = 0.99):
    """
    Helper function to  calculate state-value function
    
    Arguments:
        env: openAI GYM Enviorment object
        state: state to consider
        V: Estimated Value for each state. Vector of length nS
        discount_factor: MDP discount factor
        
    Return:
        action_values: Expected value of each action in a state. Vector of length nA
    """
    
    # initialize vector of action values
    action_values = np.zeros(env.nA)
    
    # loop over the actions we can take in an enviorment 
    for action in range(env.nA):
        # loop over the P_sa distribution.
        for probablity, next_state, reward, info in env.P[state][action]:
             #if we are in state s and take action a. then sum over all the possible states we can land into.
            action_values[action] += probablity * (reward + (discount_factor * V[next_state]))
            
    return action_values


def update_policy(env, policy, V, discount_factor):
    
    """
    Helper function to update a given policy based on given value function.
    
    Arguments:
        env: openAI GYM Enviorment object.
        policy: policy to update.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy: Updated policy based on the given state-Value function 'V'.
    """
    
    for state in range(env.nS):
        # for a given state compute state-action value.
        action_values = one_step_lookahead(env, state, V, discount_factor)
        
        # choose the action which maximizez the state-action value.
        policy[state] =  np.argmax(action_values)
        
    return policy
    

def value_iteration(env, discount_factor = 0.999, max_iteration = 1000,timing = False):
    """
    Algorithm to solve MPD.
    
    Arguments:
        env: openAI GYM Enviorment object.
        discount_factor: MDP discount factor.
        max_iteration: Maximum No.  of iterations to run.
        
    Return:
        V: Optimal state-Value function. Vector of lenth nS.
        optimal_policy: Optimal policy. Vector of length nS.
    
    """
    # intialize value fucntion
    tic = time.time()
    V = np.zeros(env.nS)
    
    # iterate over max_iterations
    for i in range(max_iteration):
        
        #  keep track of change with previous value function
        prev_v = np.copy(V) 
    
        # loop over all states
        for state in range(env.nS):
            
            # Asynchronously update the state-action value
            #action_values = one_step_lookahead(env, state, V, discount_factor)
            
            # Synchronously update the state-action value
            action_values = one_step_lookahead(env, state, prev_v, discount_factor)
            
            # select best action to perform based on highest state-action value
            best_action_value = np.max(action_values)
            
            # update the current state-value fucntion
            V[state] =  best_action_value
            
        # if policy not changed over 10 iterations it converged.
        #TODO parametrize this
        if i % 10 == 0:
            # if values of 'V' not changing after one iteration
            if (np.all(np.isclose(V, prev_v))):
                # print('Value converged at iteration %d' %(i+1))
                break


    # intialize optimal policy
    optimal_policy = np.zeros(env.nS, dtype = 'int8')
    
    # update the optimal polciy according to optimal value function 'V'
    optimal_policy = update_policy(env, optimal_policy, V, discount_factor)
    
    if timing:
        toc = time.time()
        elapsed_time = (toc - tic) * 1000
        return V, optimal_policy, elapsed_time,i
    else:
        return V, optimal_policy

def policy_eval(env, policy, V, discount_factor):
    """
    Helper function to evaluate a policy.
    
    Arguments:
        env: openAI GYM Enviorment object.
        policy: policy to evaluate.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy_value: Estimated value of each state following a given policy and state-value 'V'. 
        
    """
    policy_value = np.zeros(env.nS)
    for state, action in enumerate(policy):
        for probablity, next_state, reward, info in env.P[state][action]:
            policy_value[state] += probablity * (reward + (discount_factor * V[next_state]))
            
    return policy_value


# In[11]:


def policy_iteration(env, discount_factor = 0.999, max_iteration = 1000,timing = False):
    """
    Algorithm to solve MPD.
    
    Arguments:
        env: openAI GYM Enviorment object.
        discount_factor: MDP discount factor.
        max_iteration: Maximum No.  of iterations to run.
        
    Return:
        V: Optimal state-Value function. Vector of lenth nS.
        new_policy: Optimal policy. Vector of length nS.
    
    """
    # intialize the state-Value function
    tic = time.time()
    V = np.zeros(env.nS)
    
    # intialize a random policy
    policy = np.random.randint(0, 4, env.nS)
    policy_prev = np.copy(policy)
    
    for i in range(max_iteration):
        
        # evaluate given policy
        V = policy_eval(env, policy, V, discount_factor)
        
        # improve policy
        policy = update_policy(env, policy, V, discount_factor)
        
        # if policy not changed over 10 iterations it converged.
        #TODO parametrize this
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                # print('policy converged at iteration %d' %(i+1))
                break
            policy_prev = np.copy(policy)
            

    if timing:
        toc = time.time()
        elapsed_time = (toc - tic) * 1000
        return V, policy, elapsed_time, i            
    else:
        return V, policy


def Q_Learn(env,alpha = 0.1,gamma = 0.6,epsilon = 0.1,training_episodes=100,test_episodes=100,decay_r = 1.0):
    all_epochs = []
    all_penalties = []
    tic = time.time()
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, training_episodes):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        terminal = False

        while not terminal:
            np.random.seed(777)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
                epsilon *= decay_r
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, terminal, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward < 0:
                penalties += 1

            state = next_state
            epochs += 1

            if epochs >= 5000:
                terminal = True
    tic_tock = (time.time()-tic)*1000

    # print("Training finished.\n")
    total_epochs, total_rewards = 0, 0
    # env.reset()
    for _ in range(test_episodes):
        env.reset()
        env.s = 0
        state = 0
        # assert(env.all_states[0] == env.initial_state)
        epochs, penalties, reward = 1, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            total_rewards += reward

            epochs += 1
            if epochs > 5000:
                done = True

        total_epochs += epochs

    # print("training Episodes: ", training_episodes)
    # print(f"Results after {test_episodes} test episodes:")
    # print(f"Average timesteps per episode: {total_epochs / test_episodes}")
    # print(f"Average reward per episode: {total_rewards / test_episodes}")
    params = {'alpha':alpha,'gamma':gamma,'epsilon':epsilon,'training_episodes':training_episodes,'test_episodes':test_episodes}
    results = {'time_per_ep':total_epochs / test_episodes,'rew_per_ep':total_rewards / test_episodes,'training_time': tic_tock/training_episodes}
    return params, results


#test QL
# print('---------------------QL----------------------------')
# env = TohEnv()
# env.reset() # reset environment to a new, random state
# params, results = Q_Learn(env = env)
# print(params)
# print((results))
# print('---------------------QL----------------------------')
#


#test value iteration
# print('---------------------VI----------------------------')
# # enviorment = gym.make('FrozenLake-v0')
# # env = enviorment.env
# env = TohEnv(initial_state=((3,2, 1, 0), (), ()), goal_state=((), (), (3,2, 1, 0)), noise=0)
# tic = time.time()
# opt_V, opt_Policy = value_iteration(env, max_iteration = 1000)
# toc = time.time()
# elapsed_time = (toc - tic) * 1000
# print (f"Time to converge: {elapsed_time: 0.3} ms")
# print('Optimal Value function: ')
# # print(opt_V.reshape((4, 4)))
# print('Final Policy: ')
# print(opt_Policy)
# n_episode = 10
# wins, total_reward, avg_reward = play_episodes(env, n_episode, opt_Policy, random = False)
# # print(' '.join([action_mapping[int(action)] for action in opt_Policy]))
# print(f'Total wins with value iteration: {wins}')
# print(f"Average rewards with value iteration: {avg_reward}")
# print('---------------------VI----------------------------')




# ## Test Policy Iteration
# print('---------------------PI----------------------------')
#
# # enviorment2 = gym.make('FrozenLake-v0')
# # env2 = enviorment2.env
# env2 = TohEnv(initial_state=((3,2, 1, 0), (), ()), goal_state=((), (), (3,2, 1, 0)), noise=0)
#
# tic = time.time()
# opt_V2, opt_policy2 = policy_iteration(env2, discount_factor = 0.999, max_iteration = 10000)
# toc = time.time()
# elapsed_time = (toc - tic) * 1000
# print (f"Time to converge: {elapsed_time: 0.3} ms")
# print('Optimal Value function: ')
# # print(opt_V2.reshape((4, 4)))
# print('Final Policy: ')
# print(opt_policy2)
# # print(' '.join([action_mapping[(action)] for action in opt_policy2]))
# n_episode = 10
# wins, total_reward, avg_reward = play_episodes(env2, n_episode, opt_policy2, random = False)
#
# print(f'Total wins with Policy iteration: {wins}')
# print(f"Average rewards with Policy iteration: {avg_reward}")
# print('---------------------PI----------------------------')

# ## Remarks

# Policy Iteration converge faster but takes more computation.

# In[ ]:




