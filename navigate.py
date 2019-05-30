
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple, deque
import csv
import os
import argparse

import gym
from unityagents import UnityEnvironment

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from dqn import DQNAgent
from utils import navigate

parser = argparse.ArgumentParser(description='Provide the name of the test to render')
parser.add_argument('-t', '--test_name', help='Input test name', required=True)

test_name = parser.parse_args().test_name

#open test kwargs
root_path = os.getcwd()
test_results_path = os.path.join(root_path,'tests')
kwargs_name = test_name+'-kwargs.csv'
kwargs_path = os.path.join(test_results_path, kwargs_name)

with open(kwargs_path) as f:
    kwargs = dict(filter(None, csv.reader(f)))

#casting
kwargs['BUFFER_SIZE']=int(kwargs['BUFFER_SIZE'])
kwargs['BATCH_SIZE']=int(kwargs['BATCH_SIZE'])
kwargs['GAMMA']=float(kwargs['GAMMA'])
kwargs['TAU']=float(kwargs['TAU'])
kwargs['LR']=float(kwargs['LR'])
kwargs['UPDATE_EVERY']=int(kwargs['UPDATE_EVERY'])
kwargs['ACT_EVERY']=int(kwargs['ACT_EVERY'])
kwargs['SEED']=int(kwargs['SEED'])

#get environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

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
agent = DQNAgent(state_size=state_size, action_size=action_size, **kwargs)       

#load trained agent's weights
weights_name = test_name+'-weights.pth'
weights_path = os.path.join(test_results_path, weights_name)

agent.qnetwork_local.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))

#navigate
navigate(env, agent, brain_name)

