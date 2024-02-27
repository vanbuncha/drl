#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'greedy':
            q_values = self.Q_sa[s]
            a = np.argmax(q_values) # takes the index of max q-value
            
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # If random number is less than epsilon, select a random action
            if np.random.rand() < epsilon:
                a = np.random.randint(0, self.n_actions)
            else:
                # Otherwise, select the action with the highest Q-value for state s
                greedy_action = np.argmax(self.Q_sa[s, :])
                a = greedy_action
                 
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # Calculate the action probabilities using softmax
            action_probabilities = softmax(self.Q_sa[s, :], temp)
            
            # Select an action based on the probabilities
            a = np.random.choice(np.arange(self.n_actions), p=action_probabilities)
        return a
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
