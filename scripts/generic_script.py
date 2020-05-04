from analysis import training_data
from toh_gym.envs import TohEnv
import numpy as np

#Tower of Hanoi Example
#Initiate training object
toh_small = training_data(env ='TH',init=False)
#Set range of state spaces 
toh_small.N_range = list(range(2,8))
toh_small.title = "TOH"
#set Range of Training Episodes (QL)
toh_small.training_length = [10,50, 100,500, 1000,5000, 10000]
#initiate Environment
toh_small.init_envs(env='TH')
#Object Level Paramters
toh_small.gamma = 0.9
toh_small.alpha = 0.05
toh_small.epsilon = 0.75

#Available Methods
# toh_small.plot_TOH_Policy_Length()
# toh_small.plot_time_to_conv()
# toh_small.plot_gamma_v_reward()
# toh_small.plot_VI_analysis()
# toh_small.plot_PI_analysis()
# toh_small.plot_QL_gamma()
# toh_small.plot_QL_alhpa()
# toh_small.plot_QL_decay()
# toh_small.plot_QL_epsilon()




# toh_small.plot_time_to_conv()
toh_small.plot_TOH_Policy_Length()
# toh_small.plot_QL_decay()

#Frozen Lake Example
#Initiate training object
FL = training_data(env ='FL',init=False)
#Set range of state space
FL.N_range = list(np.linspace(8,100,10,dtype=int))
FL.title = "FL"
#set Range of Training Episodes (QL)
FL.training_length = [100,1000,5000, 10000,100000,100000]

#Object Level Paramters
FL.gamma = 0.9
FL.alpha = 0.5
FL.epsilon = 0.75
FL.init_envs(env='FL')
#call plot methods here

#FL.plot_TOH_Policy_Length()
#FL.plot_time_to_conv()
#FL.plot_gamma_v_reward()
#FL.plot_VI_analysis()
#FL.plot_PI_analysis()
#FL.plot_QL_gamma()
#FL.plot_QL_alhpa()
#FL.plot_QL_decay()
#FL.plot_QL_epsilon()


#Plot for 100X100 FL
FL.title='FL 100x100'
FL.env = FL.env_list[-1]
#call plot methods here

#available Methods
