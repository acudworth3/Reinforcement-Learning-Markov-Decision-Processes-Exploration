import gym
# import gym_hanoi
import tqdm
import numpy as np
from toh_gym.envs import TohEnv
from gym.envs.toy_text.frozen_lake import generate_random_map
from algos import value_iteration as VI
from algos import policy_iteration as PI
from algos import Q_Learn as QL
from algos import play_episodes
import  matplotlib.pyplot as plt
import time

class training_data:
    def __init__(self,env,init = True):
        self.env = None
        self.N_range = list(range(2,6))
        self.noise = 0.0
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.6
        self.env_list = []
        self.training_length = [10, 100, 1000, 10000]
        self.title = 'NONE'
        self.legend = False
        if init: self.init_envs(env)


    def init_envs(self,env):    
        if env == 'TH':
            self.key = 'TH'
            self.noise = 0.0
            # self.N_range = list(range(2, 6))
            print("initiaiting envs: \n")
            for N in tqdm.tqdm(self.N_range):
                state_N = tuple(range(N,-1,-1))
                # env = TohEnv(initial_state=((3, 2, 1, 0), (), ()), goal_state=((), (), (3, 2, 1, 0)), noise=0)

                env = TohEnv(initial_state=(state_N, (), ()), goal_state=((), (), state_N), noise=self.noise)
                self.env_list.append(env)
                self.env = self.env_list[0]

        elif env =='FL':
            self.key = 'FL'
            self.FL_maps = {}
            # self.N_range = list(range(4,20))
            print("initiaiting envs: \n")
            for N in tqdm.tqdm(self.N_range):
                np.random.seed(777)
                self.noise = 0.8
                self.FL_maps[N] = generate_random_map(size=N, p=self.noise)
                self.env_list.append(gym.make("FrozenLake-v0", desc=self.FL_maps[N]))
                self.env = self.env_list[0]
        else:
            raise KeyError

    def run_VI(self):
        #TODO extend to full list
        opt_V, opt_Policy = VI(self.env, max_iteration=1000)

    def run_PI(self):
        # TODO extend to full list
        opt_V, opt_Policy = PI(self.env, max_iteration=1000)


    def plot_TOH_Policy_Length(self):
        VI_policy_length = []
        PI_policy_length = []
        opt_policy_length = [2**N-1 for N in self.N_range]
        QL_policy_length = []
        N = list(self.N_range)
        for env in tqdm.tqdm(self.env_list):
            VI_opt_V, VI_opt_Policy = VI(env, max_iteration=1000)
            PI_opt_V, PI_opt_Policy = PI(env, max_iteration=1000)
            QL_params, QL_results = QL(env,alpha = self.alpha,gamma = self.gamma,epsilon = self.epsilon,training_episodes=100,test_episodes=1,decay_r = 1.0)
            QL_policy_length.append(QL_results['time_per_ep'])
            VI_policy_length.append(len(VI_opt_Policy))
            PI_policy_length.append(len(PI_opt_Policy))
        marker = 1
        plt.plot(self.N_range,opt_policy_length,label='Opt',marker='o')
        plt.plot(self.N_range, VI_policy_length, label='VI',marker='o',linestyle=':', markersize=10)
        plt.plot(self.N_range, PI_policy_length, label='PI',marker='x',linestyle=':', markersize=10)
        plt.plot(self.N_range, QL_policy_length, label='QL',marker='+',linestyle=':', markersize=10)
        plt.title('TOH RL Performance vs Optimium')
        plt.xlabel('Disks (N)')
        plt.ylabel('Policy Length')
        plt.xticks(list(self.N_range))
        plt.legend()
        # plt.show()
        plt.savefig('results/TOH_Policy_L.png')
        plt.close()
    # print('success')

    def plot_time_to_conv(self):
        VI_time = []
        VI_iters = []
        VI_rewards = []

        PI_time = []
        PI_iters = []
        PI_rewards = []

        QL_time = []
        QL_iters = []
        QL_rewards = []

        for env in tqdm.tqdm(self.env_list):
            #TODO loop and make this an average
            #TODO adjust training times
            self.env = env
            VI_opt_V, VI_opt_Policy, VI_elapsed_t, VI_iterations = VI(env, max_iteration=1000,timing=True)
            PI_opt_V, PI_opt_Policy, PI_elapsed_t, PI_iterations = PI(env, max_iteration=1000,timing=True)
            VI_time.append(VI_elapsed_t), VI_iters.append(VI_iterations), PI_time.append(PI_elapsed_t), PI_iters.append(PI_iterations)

            wins, total_reward, avg_reward = play_episodes(env, 100, VI_opt_Policy, random = False)
            VI_rewards.append(avg_reward)
            wins2, total_reward2, avg_reward2 = play_episodes(env, 100, PI_opt_Policy, random=False)
            PI_rewards.append(avg_reward)

            #TODO get performance

            QL_params, QL_results = QL(env, alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon,
                                       training_episodes=10**4, test_episodes=100, decay_r=1.0)
            QL_time.append(QL_results['training_time'])
            QL_iters.append(QL_results['time_per_ep'])
            QL_rewards.append(QL_results['rew_per_ep'])
        marker = 1
        #plot training time vs state space
        #Plto reward vs state space
        plt.plot(self.N_range, QL_time, label='QL', marker='x', linestyle=':', markersize=10)
        plt.plot(self.N_range, VI_time, label='VI', marker='o', linestyle=':', markersize=10)
        plt.plot(self.N_range, PI_time, label='PI', marker='+', linestyle=':', markersize=10)
        plt.xlabel('State Space Multiple (N)')
        plt.ylabel('Avg. Training Time (s)')
        plt.xticks(list(self.N_range))

        plt.title(str(self.title) + ' Training Time vs State Space')
        plt.legend()
        plt.savefig('results/'+str(self.title) +'_traintime_v_N.png')
        plt.close()

        plt.plot(self.N_range, QL_rewards, label='QL', marker='x', linestyle=':', markersize=10)
        plt.plot(self.N_range, VI_rewards, label='VI', marker='o', linestyle=':', markersize=10)
        plt.plot(self.N_range, PI_rewards, label='PI', marker='+', linestyle=':', markersize=10)
        plt.xlabel('State Space Multiple (N)')
        plt.ylabel('Avg. Reward')
        plt.xticks(list(self.N_range))

        plt.title(str(self.title) + ' Avg. Rewards vs State Space')
        plt.legend()
        plt.savefig('results/'+str(self.title) +'_rew_v_N.png')
        plt.close()

    def plot_gamma_v_reward(self):
        self.n_episode = 100
        PI_win_perc = []
        VI_win_perc = []
        PI_rewards = []
        VI_rewards = []
        gammaL = []
        # self.env = self.env_list[-1]
        for gamma in np.linspace(0.01,0.99,100):
            VI_opt_V, VI_opt_Policy = VI(self.env,discount_factor=gamma, max_iteration=1000)
            PI_opt_V, PI_opt_Policy = PI(self.env,discount_factor=gamma, max_iteration=1000)
            wins, total_reward, avg_reward = play_episodes(self.env, self.n_episode, VI_opt_Policy, random = False)
            wins2, total_reward2, avg_reward2 = play_episodes(self.env, self.n_episode, PI_opt_Policy, random=False)
            VI_win_perc.append(wins/self.n_episode)
            VI_rewards.append(avg_reward)
            PI_win_perc.append(wins2/self.n_episode)
            PI_rewards.append(avg_reward2)
            gammaL.append(gamma)
        plt.plot(gammaL,VI_win_perc,label='VI_win%',marker='o')
        # plt.plot(gammaL, VI_rewards, label='VI_rew',marker='o')
        plt.plot(gammaL, VI_win_perc, label='VI_win%',marker='o')
        # plt.plot(gammaL, VI_rewards, label='VI_rew',marker='o')
        plt.legend()
        plt.title(str(self.key)+' Gamma vs Wins+reward')

        plt.xlabel('Gamma')
        plt.show()
        plt.close()
        marker = 1


    def plot_VI_analysis(self,gamma_range=np.linspace(0.01,0.99,100)):

        self.n_episode = 100

        VI_win_perc = []
        VI_rewards = []
        VI_iters = []
        gammaL = []
        # self.env = self.env_list[-1]
        for gamma in tqdm.tqdm(gamma_range):
            VI_opt_V, VI_opt_Policy, VI_elapsed_t, VI_iterations = VI(self.env, max_iteration=1000,timing=True, discount_factor=gamma)
            wins, total_reward, avg_reward = play_episodes(self.env, self.n_episode, VI_opt_Policy, random = False)
            VI_win_perc.append(wins/self.n_episode)
            VI_rewards.append(avg_reward)
            VI_iters.append(VI_iterations)
            gammaL.append(gamma)

        # plt.plot(gammaL,VI_win_perc,label='VI_win%',marker='o')
        # # plt.plot(gammaL, VI_rewards, label='VI_rew',marker='o')
        # plt.plot(gammaL, VI_win_perc, label='VI_win%',marker='o')
        # # plt.plot(gammaL, VI_rewards, label='VI_rew',marker='o')
        # plt.legend()
        # plt.title(str(self.key)+' Gamma vs Wins+reward')
        #
        # plt.xlabel('Gamma')
        # plt.show()
        # plt.close()
        fig, ax = plt.subplots()
        ax.plot(gammaL, VI_win_perc, label='Avg. Reward', marker='x', linestyle=':')
        ax2 = ax.twinx()
        ax2.plot(gammaL, VI_iters, label='Avg. Iterations', marker='o', linestyle='--', linewidth=0.5, markersize=1.0,
                 color='green')
        ax.set_title(self.title + ' Avg. Reward, Iterations vs $\gamma$')
        ax.set_xlabel('$\gamma$')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Average Test Reward')
        ax2.set_ylabel('Average Training Iterations to Convergence')
        ax.legend(loc='lower right')
        ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 0.10))
        # plt.show()
        plt.savefig('results/VI_' + self.title + '_perf_v_gamma.png')
        # plt.close()


        marker = 1


    def plot_PI_analysis(self,gamma_range = np.linspace(0.01,0.99,100)):

        self.n_episode = 100

        PI_win_perc = []
        PI_rewards = []
        PI_iters = []
        gammaL = []
        # self.env = self.env_list[-1]
        for gamma in tqdm.tqdm(gamma_range):
            PI_opt_V, PI_opt_Policy, PI_elapsed_t, PI_iterations = PI(self.env, max_iteration=1000,timing=True,discount_factor=gamma)
            wins, total_reward, avg_reward = play_episodes(self.env, self.n_episode, PI_opt_Policy, random = False)
            PI_win_perc.append(wins/self.n_episode)
            PI_rewards.append(avg_reward)
            PI_iters.append(PI_iterations)
            gammaL.append(gamma)


        fig, ax = plt.subplots()
        ax.plot(gammaL, PI_win_perc, label='Avg. Reward', marker='x', linestyle=':')
        ax2 = ax.twinx()
        ax2.plot(gammaL, PI_iters, label='Avg. Iterations', marker='o', linestyle='--', linewidth=0.5, markersize=1.0,
                 color='green')
        ax.set_title(self.title + ' Avg. Reward, Iterations vs $\gamma$')
        ax.set_xlabel('$\gamma$')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Average Test Reward')
        ax2.set_ylabel('Average Training Iterations to Convergence')
        ax.legend(loc='lower right')
        ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 0.10))
        # plt.show()
        plt.savefig('results/PI_' + self.title + '_perf_v_gamma.png')
        # plt.close()


    def plot_QL_gamma(self):
        alpha = self.alpha #TODO update these
        gamma = self.gamma 
        epsilon = self.epsilon
        training_episodes = 100
        test_episodes = 100
        # params, results = QL(self.env,alpha,gamma,epsilon,training_episodes,test_episodes)

        gamma_rewards = {}
        gamma_steps = {}
        import tqdm
        gl = [0.05, 0.25, 0.5, 0.75, 0.9]
        for gamma in tqdm.tqdm(gl):
            avg_reward = []
            avg_steps = []
            # self.training_length = [10, 100, 1000, 10000]
            for training_episodes in tqdm.tqdm(self.training_length):
                params, results = QL(self.env, alpha, gamma, epsilon, training_episodes, test_episodes)
                avg_reward.append(results['rew_per_ep'])
                avg_steps.append(results['time_per_ep'])
            gamma_rewards[str(gamma)] = avg_reward
            gamma_steps[str(gamma)] = avg_steps
        marker = 1

        fig, ax = plt.subplots()
        [plt.plot(self.training_length, gamma_rewards[str(gm)], label=' $\gamma$ = '+str(gm), marker='x', linestyle=':') for
         gm in gl]
        ax2 = ax.twinx()
        [plt.plot(self.training_length, gamma_steps[str(gm)], label=' $\gamma$ = '+str(gm), marker='.', linestyle='--',
                  linewidth=0.4) for
         gm in gl]
        # [gamma_steps[str(gm) for gm in gl]
        # set x-axis label
        # ax.legend(title="Rewards")
        ax.set_xscale('log')
        ax.set_title(self.title)
        ax.set_xlabel('Training Episodes')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Average Test Reward')
        # ax2.legend(title="Iterations")
        ax2.set_ylabel('Average Iterations to Convergence')

        if self.legend:
            ax.legend(bbox_to_anchor=(1.15, 0.65), loc="lower left", borderaxespad=0, title='Avg. Reward')
            ax2.legend(bbox_to_anchor=(1.15, 0), loc="lower left", borderaxespad=0, title='Avg. Iters')
            plt.savefig('results/QL_' + self.title + '_gamma_v_train.png', bbox_inches='tight')
        else:
            plt.savefig('results/QL_' + self.title + '_gamma_v_train.png')
        plt.close()


    def plot_QL_alhpa(self):
        alpha = self.alpha #TODO update these
        gamma = self.gamma 
        epsilon = self.epsilon
        training_episodes = 100
        test_episodes = 100
        # params, results = QL(self.env,alpha,gamma,epsilon,training_episodes,test_episodes)

        alpha_rewards = {}
        alpha_steps = {}
        import tqdm
        gl = [0.01, 0.05, 0.1, 0.15, 0.2]
        for alpha in tqdm.tqdm(gl):
            avg_reward = []
            avg_steps = []
            # self.training_length = [10, 100, 1000, 10000]
            for training_episodes in tqdm.tqdm(self.training_length):
                params, results = QL(self.env, alpha, gamma, epsilon, training_episodes, test_episodes)
                avg_reward.append(results['rew_per_ep'])
                avg_steps.append(results['time_per_ep'])
            alpha_rewards[str(alpha)] = avg_reward
            alpha_steps[str(alpha)] = avg_steps

        
        fig, ax = plt.subplots()
        [plt.plot(self.training_length, alpha_rewards[str(gm)],label='alpha = '+str(gm), marker='x', linestyle=':') for gm in gl]
        ax2 = ax.twinx()
        [plt.plot(self.training_length, alpha_steps[str(gm)], label='alpha = '+str(gm), marker='.', linestyle='--', linewidth=0.4) for gm in gl]

        ax.set_xscale('log')
        ax.set_title(self.title)
        ax.set_xlabel('Training Episodes')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Average Test Reward')
        # ax2.legend(title="Iterations")
        ax2.set_ylabel('Average Iterations to Convergence')
        # plt.show()
        if self.legend:
            ax.legend(bbox_to_anchor=(1.15, 0.65), loc="lower left", borderaxespad=0, title='Avg. Reward')
            ax2.legend(bbox_to_anchor=(1.15, 0), loc="lower left", borderaxespad=0, title='Avg. Iters')
            plt.savefig('results/QL_' + self.title + '_alpha_v_train.png', bbox_inches='tight')
        else:
            plt.savefig('results/QL_' + self.title + '_alpha_v_train.png')
  

 # plt.savefig('results/QL_' + self.title + '_alpha_v_train.png')
        plt.close()

    def plot_QL_decay(self):
        alpha = self.alpha #TODO update these
        gamma = self.gamma
        epsilon = self.epsilon
        training_episodes = 100
        test_episodes = 100
        # params, results = QL(self.env,alpha,gamma,epsilon,training_episodes,test_episodes)

        nu_rewards = {}
        nu_steps = {}
        import tqdm
        gl = [0.0001, 0.001, 0.0, 0.1,0.5 ]
        for decay_r in tqdm.tqdm(gl):
            avg_reward = []
            avg_steps = []
            # training_length = [10, 100, 1000, 10000]
            for training_episodes in tqdm.tqdm(self.training_length):
                params, results = QL(self.env, alpha, gamma, epsilon, training_episodes, test_episodes,decay_r=decay_r)
                avg_reward.append(results['rew_per_ep'])
                avg_steps.append(results['time_per_ep'])
            nu_rewards[str(decay_r)] = avg_reward
            nu_steps[str(decay_r)] = avg_steps

        fig, ax = plt.subplots()
        [plt.plot(self.training_length, nu_rewards[str(gm)], label=r'$\nu$ = ' + str(gm), marker='x', linestyle=':') for
         gm in gl]
        # [gamma_steps[str(gm) for gm in gl]
        # set x-axis label
        ax.legend(title="Decay Rate")
        ax.set_xscale('log')
        ax.set_title(self.title)
        ax.set_xlabel('Training Episodes')
        # ax.set_ylim(0, 1)
        ax.set_ylabel('Average Test Reward')
        # ax2.legend(title="Iterations")

        # plt.show()
        plt.savefig('results/QL_' + self.title + '_decay_v_train.png')



        # plt.savefig('results/QL_' + self.title + '_eps_v_train.png')
        plt.close()



    def plot_QL_epsilon(self):
        alpha = self.alpha #TODO update these
        gamma = self.gamma 
        epsilon = self.epsilon
        training_episodes = 100
        test_episodes = 100
        # params, results = QL(self.env,alpha,gamma,epsilon,training_episodes,test_episodes)

        epsilon_rewards = {}
        epsilon_steps = {}
        import tqdm
        gl = [0.05, 0.25, 0.5, 0.75, 0.9]
        for epsilon in tqdm.tqdm(gl):
            avg_reward = []
            avg_steps = []
            # training_length = [10, 100, 1000, 10000]
            for training_episodes in tqdm.tqdm(self.training_length):
                params, results = QL(self.env, alpha, gamma, epsilon, training_episodes, test_episodes)
                avg_reward.append(results['rew_per_ep'])
                avg_steps.append(results['time_per_ep'])
            epsilon_rewards[str(epsilon)] = avg_reward
            epsilon_steps[str(epsilon)] = avg_steps

        fig, ax = plt.subplots()
        [plt.plot(self.training_length, epsilon_rewards[str(gm)], label='$\epsilon$ = '+str(gm), marker='x', linestyle=':') for
         gm in gl]
        ax2 = ax.twinx()
        [plt.plot(self.training_length, epsilon_steps[str(gm)], label='$\epsilon$ = '+str(gm), marker='.', linestyle='--',
                  linewidth=0.4) for
         gm in gl]
        # [gamma_steps[str(gm) for gm in gl]
        # set x-axis label
        # ax.legend(title="Rewards")
        ax.set_xscale('log')
        ax.set_title(self.title)
        ax.set_xlabel('Training Episodes')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Average Test Reward')
        # ax2.legend(title="Iterations")
        ax2.set_ylabel('Average Iterations to Convergence')
        # plt.show()
        if self.legend:
            ax.legend(bbox_to_anchor=(1.15, 0.65), loc="lower left", borderaxespad=0, title='Avg. Reward')
            ax2.legend(bbox_to_anchor=(1.15, 0), loc="lower left", borderaxespad=0, title='Avg. Iters')
            plt.savefig('results/QL_' + self.title + '_eps_v_train.png', bbox_inches='tight')
        else:
            plt.savefig('results/QL_' + self.title + '_eps_v_train.png')



        # plt.savefig('results/QL_' + self.title + '_eps_v_train.png')
        plt.close()




