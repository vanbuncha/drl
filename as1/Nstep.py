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

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' 
        Update Q-values using n-step Q-learning.
        ''' 
        T = len(rewards)
        for t in range(T):
            m = min(n, T-t)
            if t+m == T and done:
                G = np.sum([self.gamma**i * rewards[t+i] for i in range(m)])
            else:
                # computation of G for non-terminal states
                G = np.sum([self.gamma**i * rewards[t+i] for i in range(m)]) + self.gamma**m * np.max(self.Q_sa[states[t+m]])
            
            # Update Q-values
            updated_value = self.Q_sa[states[t], actions[t]] + self.learning_rate * (G - self.Q_sa[states[t], actions[t]])
            self.Q_sa[states[t], actions[t]] = np.clip(updated_value, -1e6, 1e6)


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' 
    Runs a single repetition of an n-step Q-learning agent.

    Returns:
        eval_returns: A vector with the observed returns at each evaluation interval.
        eval_timesteps: A vector with the corresponding timesteps at which evaluations occurred.
    ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    timestep_counter = 0

    while timestep_counter < n_timesteps:
        actions, rewards, done = [], [], False
        states = [env.reset()]

        # Collect an episode
        for _ in range(max_episode_length):
            actions.append(pi.select_action(states[-1], policy, epsilon, temp))
            next_state, reward, done = env.step(actions[-1])
            states.append(next_state)
            rewards.append(reward)

            timestep_counter += 1

            # Evaluate the policy at intervals
            if timestep_counter != 0 and timestep_counter % eval_interval == 0:
                returns = pi.evaluate(eval_env)
                eval_returns.append(returns)
                eval_timesteps.append(timestep_counter)

            if done or timestep_counter >= n_timesteps:
                break

        # Update Q-values using n-step Q-learning
        pi.update(states, actions, rewards, done, n)

    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)
    print("done")
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                                             policy, epsilon, temp, plot, n=n)
    print("Evaluation Returns:", eval_returns)
    print("Evaluation Timesteps:", eval_timesteps)
    
if __name__ == '__main__':
    test()