[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


#  Continuous Control

### Introduction

This project attempts to solve the Reacher environment from Unity technologies
 [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) .

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The problem is solved when the agent receives an average reward of 30 over 100 episodes.

See Report.md for detailed description

## Installation

* continuous_control.py is the landing script
* This repo contains contains :
	1. The agent (agent.py) that implements the DDPG RL agent functionalities   
	2. Actor and Critic Modes : model.py
	3. Simulator helper (paramsutility.py) ; A parameters manager utility
	4. File plotres.py: A utility to plot the results from multiple runs of Navigation.py
## Requirements:
	* Python 3.6 or greater
	* pythorch 1.7.1
	* Unity ML Agents, Banana Environment
	
## Step 1: Instantiating the Python environment and dependencies
This installs all the required dependencies.
Please see the instruction [ Udacity DRLND GitHub repository.](https://github.com/udacity/deep-reinforcement-learning#dependencies)

## Step 2: Installing the Reacher Environment
Please see the instruction [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started)

## Running the code
* Set the parameters in the params object in continuous_control.py
* Run the continuous_control.py
* By default, the script ends when a score of 30 is met.
* Three files are generated:
	* The pickle file containing the parameters, actor and critic loss and the scores. The file may be analyzed via the plotres.py utility
	* Checkpoint of the actor and critic NN model.
## Baseline results
The default parameter/results are located in the folder '/Results'. They can be analyzed with the plotres.py utility.
Checkpoint files are located in the checkpoint folder/


