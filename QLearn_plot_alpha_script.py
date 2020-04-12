from analysis import training_data
from toh_gym.envs import TohEnv

# toh_small = training_data(env ='TH',init=False)
# toh_small.N_range = list(range(2,3))
# toh_small.title = "TOH 3 Disks"
# toh_small.training_length = [10,50, 100,500, 1000,5000, 10000]
# # toh_small.training_length = [10,50]
# toh_small.init_envs(env='TH')
# toh_small.plot_QL_alhpa()
#
FL = training_data(env ='FL',init=False)
FL.N_range = list(range(8,101))
FL.title = "FL 8x8"
FL.training_length = [100,1000,5000, 10000,100000,100000]
# FL.training_length = [10,50]
FL.init_envs(env='FL')
FL.plot_QL_alhpa()

FL.env = FL.env_list[-1]
FL.title = "FL 100x100"
FL.legend = True
FL.plot_QL_alhpa()


marker = 1



# toh_small.training_length = [10000,20000,30000]
