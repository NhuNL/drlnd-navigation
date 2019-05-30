import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from collections import namedtuple, deque
import argparse

import gym
from unityagents import UnityEnvironment

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from dqn import DQNAgent
from ddqn import DDQNAgent
from train import train_agent
from utils import save_learning_data

parser = argparse.ArgumentParser(description='Optional arguments for solving unitys banana collection environment.\
                 Supported learning algorithms: dqn, ddqn')
parser.add_argument('--buffer_size', default = int(1e6), help='replay buffer size')
parser.add_argument('--batch_size', default = 64, help='minibatch size')
parser.add_argument('--gamma', default = 0.99, help='discount factor')
parser.add_argument('--tau', default = 0.001, help='for soft update of target parameters')
parser.add_argument('--lr', default = 0.0001, help='learning rate')
parser.add_argument('--update_every', default = 5, help='how often to update the q-network')
parser.add_argument('--act_every', default = 1, help='how often to take new actions')
parser.add_argument('--fc1', default = 32, help='fc1 units')
parser.add_argument('--fc2', default = 64, help='fc2 units')
parser.add_argument('--BN', default = "True", help='flag for using batch norm')
parser.add_argument('--seed', default = 42, help='seed')
parser.add_argument('--algo', default = 'dqn', help='learning algorithm')
parser.add_argument('--eps_init', default = 1.0, help='initial epsilon value')
parser.add_argument('--eps_decay', default = 0.995, help='epsilon decay rate')
parser.add_argument('--train_episodes', default = 1000, help='Number of training episodes')
parser.add_argument('-t', '--test_name', help='Input test name is required', required=True)
n = parser.parse_args()

test_name = str(n.test_name)

root_path = os.getcwd()
test_results_path = os.path.join(root_path,'tests')

#Training kwargs
kwargs = {  'ALGO' : str(n.algo),
            'TRAIN_EPISODES' : int(n.train_episodes),    
            'BUFFER_SIZE' : int(n.buffer_size),  
            'BATCH_SIZE' : int(n.batch_size),                
            'GAMMA' : float(n.gamma) ,                 
            'TAU' :  float(n.tau),                    
            'EPS_INIT' : float(n.eps_init)  , 
            'EPS_DECAY' :  float(n.eps_decay),                    
            'LR' : float(n.lr)  ,                  
            'UPDATE_EVERY' : int(n.update_every) ,            
            'ACT_EVERY' : int(n.act_every) ,                
            'FC1' : int(n.fc1),
            'FC2' : int(n.fc2),
            'BN' : str(n.BN),
            'SEED' : int(n.seed)
        }

learning_algorithm = str(n.algo)
eps_init = float(n.eps_init)
eps_decay = float(n.eps_decay)
n_episodes = int(n.train_episodes)

#get environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# get environment state and action space sizes
state = env_info.vector_observations[0]
state_size = len(state)
action_size = brain.vector_action_space_size

#make agent
if learning_algorithm == 'dqn':
        agent = DQNAgent(state_size=state_size, action_size=action_size, **kwargs)
elif learning_algorithm == 'ddqn':
        agent = DDQNAgent(state_size=state_size, action_size=action_size, **kwargs)       
else:
        print('Error: learning algorithm not currently supported. See help for supported algorithms')
        exit()

#train agent
scores, last_eps, episode_goal, episode_min_eps = train_agent(agent, env, brain_name, 
                                                              n_episodes=n_episodes, 
                                                              eps_start = eps_init, 
                                                              eps_decay = eps_decay)

#save learning data
qnet= agent.qnetwork_local.state_dict()
save_learning_data(scores, test_name, test_results_path, qnet,  100, **kwargs)

#save log file
line = test_name+"\t"+str(episode_goal)+"\t"+str(last_eps)+"\t"+str(episode_min_eps)+' \n'
with open('log.txt', 'a') as file:
    file.write(line)

print("Training finished successfully!")
