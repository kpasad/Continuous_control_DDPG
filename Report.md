# Unity-ML Reacher

This repo contains a working code for solving the Unity ML Agent called Reacher using a DDPG  Network.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


## The Task
The environment contains emulation of spheres floating randomly in space and a double jointed arm whose motion are controlled by the agent. The agent must learn to move the    arm so that the arm is in contact with the sphere , as the spheres move around. For every step that the agent's hand is in contact with the spheres location, an award of +0,1 is provided. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.\
## The Environment and the Agent
### The State Space
The environment is defined by a state vector composed of 33 variables. corresponding to position, rotation, velocity, and angular velocities of the arm. 
What do the element of the state indicate? No clear answers from Unity. The environment itself is no longer present on unity.com 

### The Action Space
In response to the observed state, the agent can take an action to  move in four direction in 2-Dimensions space. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## DDPG algorithm
DDPG algorithm was proposed in [this](https://arxiv.org/pdf/1509.02971.pdf) paper.
Q-Learning cannot be applied to continuous action space as the optimization of the action (argmax) is inefficient if not immposible. DDPG solves this problem by directly mapping the state to the optimal action. The mapping is a non linear function, that is typically implemented as a neural network.  Exploration is invoked by dithering the action with Ornstein-Uhlenbeck noise.

DDPG is often bucketed in family of actor-critic algorithm.The actor implements a current policy to map the state to optimal action. The critic implemets the Q function estimation, and estimates the Action value given a state and an action. The actor is trained by the maximising the gradient of action value. Critic is trained by minimizing the MSE between predicted Q-value and meausured Q value (as a surrogate for true long term).

Experinces are stored in a replay buffer. 
In order to encourage exploration during training, Ornstein-Uhlenbeck noise is added to the actors selected actions. 

## The Solution: DDPG
The Unity environment enables 20 agents that can learn simultaneously. The DDPG solition is implemented with a single agent.
The DDPG agent is composed of two actor and two critic networks. 
The actor networks are:
|Layer|Input size  |Output(Number of Neurons)|Nonlinearity|
|--|--|--|--|
|1|33|128| ReLu|
|2 |128|64|ReLu|
|3|64|4|Linear|

The critic networks are:
|Layer|Input size  |Output(Number of Neurons)|Nonlinearity|
|--|--|--|--|
|1|33|128| ReLu|
|2 |128+4|64|ReLu|
|3|64|1|Linear|

The critic network concatenates the action estimated by the critic network with a feature representation of the state instead of concatenating directly with the state. Critic learns faster on the feature representation instead of the raw state.

Key parametes are::
|Parameter	|Value|
|--|--|
|replay buffer size|	int(1e6)|
|minibatch size	|256|
|discount factor	|0.99|
|tau (soft update)|	1e-3|
|learning rate actor|	1e-5|
|learning rate critic|	1e-5|
|L2 weight decay|	0|
|UPDATE_EVERY	|20|
|NUM_UPDATES	|10|
|EPSILON	|1.0|
|EPSILON_DECAY|	1e-6|
|NOISE_SIGMA	|0.1|

## Running the simulation
Run the file continuous_control.py. The simulation will terminate when the agent gets a score of 30 over 100 episodes At the end of simulation, a pickle file is generated. The file contains a dump of parameters as the raw scores. The scores can be analyzed to create the plot below using the utility script plotres.py. The network weights are checkpointed as well.

## Performance
The learning for the succesful agent is shown below.
![Scores for succesful agent](https://github.com/kpasad/Continuous_control_DDPG/blob/main/Results/final_scores_faster.jpeg)

The numerical values are shown below: 
![Scores for succesful agent](https://github.com/kpasad/Continuous_control_DDPG/blob/main/Results/Learning_screenshot.JPG)


Succesful learning is defined as an average score of 30 over 100 episodes. For an OU noise variance of 0.1, the agent reaches the target score over the 100 episodes folowing the 324th episode episodes. 

### Score trajectory
The score for a longer trajectory and OU noise variance of 0.07 is shown below
![Scores for various learning rate](https://github.com/kpasad/Continuous_control_DDPG/blob/main/Results/final_scores.jpeg)

The scores improve until episode 500 and then vaccilate around a average score which itself diminishes. To quantify the long term variance, the learning must be simulated over multiple seeds.

### Sensitivity to Exploration model (Ornstein???Uhlenbeck noise):
The action determined by Action network is perturbed by the Ornstein???Uhlenbeck noise. Training the networks required change to the noise model, to change from uniform random to normal distribution. The variance of the distribution dictates the amount of exploration that the agent performs. The agent is very sensitive to the the variance. See below:

 ![Scores for sweep of variance of OU noise](https://github.com/kpasad/Continuous_control_DDPG/blob/main/Results/OU_noise_sweep.jpeg)
 
For the noise variance of 0.08 and 0.1, the agent learns approximately identically until  about 250 episodes. Noise variance of 0.1 outperforms variance of 0.08. It could be hypothesised that a slight nudge during exploration led the 
the agent to learn at a highly rewarding behaviour.  

### Sensitivity to Learning rate :
We sweep the learning rate for both Actor and Critic. Both are kept identical, and swept together.
![Scores for various learning rate](https://github.com/kpasad/Continuous_control_DDPG/blob/main/Results/scores.jpeg)

Notice that the agent learns for a very narrow range of learning rates.

### Actor and Critic Loss:
During the initial stage of agent development, the agent is not able to learn. To try to understand, why the agent does not learn, can we analyze the Actor and Critic loss?
Below are plots for Actor and Critic Loss. For reference, the scores are shown about.
![Actor Loss](https://github.com/kpasad/Continuous_control_DDPG/blob/main/Results/actor_loss.jpeg)

![Critic Loss](https://github.com/kpasad/Continuous_control_DDPG/blob/main/Results/critic_loss.jpeg)

 The actor 'loss' is  the average reward and the actor learns to maximize it. The Actor loss is a Mean square loss, we expect it to reduce as the agent learns. In the plot below we see, that as the agent learns (score increases) the critic loss increases. The actor loss, however, also increases. The reason is that the MSE is calculated against a moving target created by the actor. So, the critic loss is not really a meaningful metric. Even if the actor loss behaves as expected, the actor loss is not meaningful either.

## Conclusion
While the DDPG model is straigh forward it is very sensitive to hyperparameters.After a hyper parameter sweep, the DDPG agent learns and the scores reach the target.  Strating with a smaller network is advisable.
Some suggestions:
1. Annealing the OU noise to automate the exploration-exploitation.
2. Using a very small network to limit the variance in score.
3. Trying multi-agent set up to reduce sensitivity to parameters.
