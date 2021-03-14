
from unityagents import UnityEnvironment
import numpy as np


#env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe",no_graphics=True)


brain_name = env.brain_names[0]
brain = env.brains[brain_name]


from ddpg_agent import Agent
from collections import deque
import time
from paramutils import *
import pickle as pk
#from workspace_utils import active_session

params=parameters()
params.op_filename_prefix="Continuous_ctrl_"+time.strftime("%H_%M_%S",time.gmtime(time.time()))
print(params.op_filename_prefix)
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

params.n_episodes=200
params.OU_noise_sigma=0.01

agent = Agent(state_size, action_size, params.env_seed, params)


print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


#from workspace_utils import active_session

n_episodes=params.n_episodes
scores_window=deque(maxlen=100)
scores=[]
actor_loss_lst=[]
critic_loss_lst=[]
start_time=time.time()
#with active_session():
if 1:
    for i_episode in range(1,n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        per_agent_score = np.zeros(num_agents)                          # initialize the score (for each agent)
        for t in range(params.episode_len):#1000 Needed to allow agents to collect rewards
            actions = agent.act(states) #Generate action from each agent.
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            actor_loss,critic_loss=agent.step(states, actions, rewards, next_states, dones) #Update ALL agents
            actor_loss_lst.append(actor_loss)
            critic_loss_lst.append(critic_loss)
            per_agent_score += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        scores_window.append(np.mean(per_agent_score)) #Take mean accross the agents??
        scores.append(np.mean(per_agent_score))
        print('\rEpisode {} \tTime:{} \tAverage Score: {:.2f}'.format(i_episode, time.strftime("%H:%M:%S", time.gmtime((time.time() - start_time))),np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))


env.close()
pk.dump([scores, params],open(params.op_filename_prefix+'.pk','wb'))
