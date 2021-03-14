class parameters():
    def __init__(self):
        self.network = 'dueling_dqn'    #'dqn, dueling_dqn'
        self.buffer ='baseline' #'priority_replay'
        self.env_seed = 2
        self.n_episodes = 2000
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.double_dqn = 'enable' #enable, disable
        self.episode_len=1000
        self.actor_nw_size = [256,128]
        self.actor_nw_lr =1e-4
        self.critic_nw_size = [256, 128]
        self.critic_nw_lr = 1e-4
        self.OU_noise_sigma=0.07
        self.OU_noise_mu = 0.
        self.OU_noise_theta = 0.15
        self.actor_nw_seed=2
        self.critic_nw_seed = 2