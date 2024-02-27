#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class QLearningAgent(BaseAgent):
        
    def update(self, s, a, r, s_next, done):
        # Calculate the backup estimate/target G_t
        G_t = r + self.gamma * np.max(self.Q_sa[s_next, :]) * (1 - done)
        
        # Apply the tabular learning update
        self.Q_sa[s, a] += self.learning_rate * (G_t - self.Q_sa[s, a])

 
def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    s = env.reset()

    for t in range(n_timesteps):

        # action with specific policy
        a = agent.select_action(s, policy, epsilon, temp)
        
        # action, observation, reward, done
        s_next, r, done = env.step(a)
        
        # Update Q-value
        agent.update(s, a, r, s_next, done)
        
        # Update reward and new state
        s = s_next if not done else env.reset()




        # Evaluate the agent every eval_interval timesteps
        if (t + 1) % eval_interval == 0:
         eval_timesteps.append(t + 1)
         mean_return = agent.evaluate(eval_env)
         eval_returns.append(mean_return)
        
    
        if plot:
            # Plot the Q-value estimates during Q-learning execution
            env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) 

    return np.array(eval_returns), np.array(eval_timesteps)  


def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.001
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
