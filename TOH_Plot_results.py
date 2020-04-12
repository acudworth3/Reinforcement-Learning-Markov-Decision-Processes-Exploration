from analysis import training_data
from toh_gym.envs import TohEnv
import numpy as np

toh_small = training_data(env ='TH',init=False)
toh_small.N_range = list(range(2,8))
toh_small.title = "TOH"
toh_small.training_length = [10,50, 100,500, 1000,5000, 10000]
# toh_small.training_length = [10,50]
toh_small.init_envs(env='TH')
toh_small.gamma = 0.9
toh_small.alpha = 0.05
toh_small.epsilon = 0.75
# toh_small.plot_time_to_conv()
toh_small.plot_TOH_Policy_Length()
#TODO set epsilon, gamma, and alpha
# toh_small.plot_QL_decay()
