#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import time

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions)) # sets initial Q-values to zero

        
        
    def select_action(self,s):
        ''' Returns the greedy best action ifn state s '''
        q_values = self.Q_sa[s]
        a = np.argmax(q_values) # takes the index of max q-value
        return a
    

        
    # Update function using in place update instead of using a copy of the Q-table
    # is faster but not so consistent - and is biased as first calc pairs might have bigger importnace

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        q_value = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1))) # Bellman equation
        error = np.abs(self.Q_sa[s, a] - q_value)
        self.Q_sa[s, a] = q_value
        return error  
    


    
def Q_value_iteration(env, gamma=1, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    
    iteration = 0  # Initialize iteration counter
    while True:
        max_error = 0  # Reset the max error for this sweep
        Q_sa_copy = np.copy(QIagent.Q_sa)  # Make a copy of the current Q-table
        
        # Sweep through the state space
        for s in range(env.n_states):
            for a in range(env.n_actions):
                # Obtain model outputs: transition probabilities and rewards
                p_sas, r_sas = env.model(s, a)
                
                # Update the agent based on the model's outputs using the copy of the Q-table
                # error = QIagent.update(s, a, p_sas, r_sas, Q_sa_copy)
                error = QIagent.update(s, a, p_sas, r_sas)
                
                # Track the maximum absolute error in this sweep
                max_error = max(max_error, error)

        # visualize the Q-value estimates at each iteration
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.1)
        print("Q-value iteration, iteration {}, max error {}".format(iteration, max_error))
        
        iteration += 1  # Increment iteration counter
        
        # Check for convergence
        if max_error < threshold:
            break
    
    return QIagent



def compute_optimal_value(QIagent, s):
    ''' Compute the optimal value for state s '''
    optimal_value = np.max(QIagent.Q_sa[s])
    return optimal_value



def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    V_star_s3 = compute_optimal_value(QIagent, 3)
    print(f"The converged optimal value at the start state (s = 3) is: {V_star_s3}")
    # view optimal policy
    done = False
    s = env.reset()
    total_reward = 0
    timesteps = 0
    while not done:
        
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        s = s_next
        total_reward += r
        timesteps += 1
        

        # visualize the optimal policy
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=5) 

    # TO DO: Compute mean reward per timestep under the optimal policy
    
    mean_reward_per_timestep = total_reward / timesteps
    
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()
    time.sleep(1) # Sleep for X seconds to allow the last plot to be displayed

