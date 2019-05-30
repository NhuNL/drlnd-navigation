import random
import torch
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os


def save_learning_data(scores, test_name, test_path, network, smoothing_window=100, **kwargs):
    #save kwargs
    kwargs_name = test_path+'/'+test_name+'-kwargs.csv'
    with open(kwargs_name, 'w') as f:
        for key in kwargs.keys():
            f.write("%s,%s\n"%(key,kwargs[key]))
        f.close()
    print('Kwargs saved in', kwargs_name)

    #save scores
    scores = pd.DataFrame({'Scores':scores})
    scores_name = test_path+'/'+test_name+'-scores.csv'
    scores.to_csv(scores_name, sep=',')
    print('Learning curve saved in', scores_name)

    #make and save figure
    plot_name = test_path+'/'+test_name+'-plot.png'
    fig = plt.figure(figsize=(10,5))
    plt.grid(False)
    plt.style.use('seaborn-bright')
    rewards_smoothed = scores.rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend([test_name])
    plt.savefig(plot_name)
    #plt.show(fig)
    print('Plot saved in', plot_name)

    #save weights
    weights_name = test_path+'/'+test_name+'-weights.pth'
    torch.save(network, weights_name)
    print('Model weights saved in', weights_name)


#play sample episode
def navigate(env, agent, brain_name):
    env_info = env.reset(train_mode=False)[brain_name] 
    state = env_info.vector_observations[0]            
    score = 0                                          
    while True:
        action = agent.act(state)        
        action_array = np.array([action])
        env_info = env.step(action_array)[brain_name]        
        next_state = env_info.vector_observations[0]   
        reward = env_info.rewards[0]                   
        done = env_info.local_done[0]                  
        score += reward                                
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:                                       
            break
    print("Sample episode score: {}".format(score))