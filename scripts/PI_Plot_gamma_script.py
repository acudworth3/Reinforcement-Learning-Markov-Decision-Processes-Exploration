from analysis import training_data
from toh_gym.envs import TohEnv
import numpy as np

toh_small = training_data(env ='TH',init=False)
toh_small.N_range = list(range(2,3))
toh_small.title = "TOH 3 Disks"
toh_small.training_length = [10,50, 100,500, 1000,5000, 10000]
# toh_small.training_length = [10,50]
toh_small.init_envs(env='TH')
toh_small.plot_PI_analysis()

# toh_large = training_data(env ='TH',init=False)
# toh_large.N_range = list(range(8,9))
# toh_large.title = "TOH 8 Disks"
# toh_large.training_length = [100,1000,5000, 10000,100000,100000]
# # toh_small.training_length = [10,50]
# toh_large.title = 'TOH 8 Disks'
# toh_large.init_envs(env='TH')
#
# toh_large.env = toh_large.env_list[-1]
# toh_large.plot_VI_analysis()

#
FL = training_data(env ='FL',init=False)
FL.N_range = list(range(8,101))
FL.title = "FL 8x8"
FL.training_length = [100,1000,5000, 10000,100000,100000]
# FL.training_length = [10,50]
FL.init_envs(env='FL')
FL.plot_PI_analysis()

FL.title='FL 100x100'
FL.env = FL.env_list[-1]
FL.legend = True
FL.plot_PI_analysis(np.linspace(0.01,0.99,5))


#TODO include each environment
#Do I care about training time?
