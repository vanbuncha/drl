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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        G_t = 0
        T = len(states) - 1
        for t in reversed(range(T)):  # Iterate backwards through the episode
            # Calculate G_t: the discounted return from time step t
            G_t = self.gamma * G_t + rewards[t]
            # Update Q-value for the state-action pair (S_t, A_t)
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G_t - self.Q_sa[states[t], actions[t]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''  
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    for t in range(n_timesteps):
        states, actions, rewards = [], [], []
        state = env.reset()

        # Episode loop
        for _ in range(max_episode_length):
            action = pi.select_action(state, policy, epsilon, temp)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break

        # Update Q-values using the episode
        pi.update(states, actions, rewards)

        # Evaluate the agent every eval_interval timesteps
        if t != 0 and t % eval_interval == 0:
            returns = pi.evaluate(eval_env, n_eval_episodes=30, max_episode_length=100) 
            eval_returns.append(returns)
            eval_timesteps.append(t)

        # if plot:
        #     env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution
            
    return np.array(eval_returns), np.array(eval_timesteps)
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()