#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

class QLearningLogger:
    def __init__(self, agent, states):
        self.agent = agent
        self.states = states

    def log_training(self, episodes, rewards, log_interval=1000):
        q_values_history = []
        exploration_rate_history = []

        for episode in range(episodes):
            state_index = episode % len(self.states)
            state = tuple(self.states[state_index])
            action = self.agent.choose_action(state)
            reward = rewards[episode % len(rewards)]
            next_state = tuple(self.states[(episode + 1) % len(self.states)])
            self.agent.update_q_table(state, action, reward, next_state)
            self.agent.decay_exploration()

            # Log data every 'log_interval' episodes
            if episode % log_interval == 0:
                q_values_history.append(self.agent.q_table[state])
                exploration_rate_history.append(self.agent.exploration_rate)
                print(f"Episode {episode}: Exploration Rate = {self.agent.exploration_rate:.4f}")
                print(f"Q-values for a selected state: {self.agent.q_table[state]}")
        
        return q_values_history, exploration_rate_history

    def plot_q_values(self, q_values_history):
        plt.figure(figsize=(12, 8))
        q_values_array = np.array(q_values_history)  # Convert list of arrays into a 2D array for easier plotting
        for action_index in range(q_values_array.shape[1]):
            plt.plot(q_values_array[:, action_index], label=f'Action {action_index}')
        plt.title('Q-values Over Time')
        plt.xlabel('Episode (x1000)')
        plt.ylabel('Q-value')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_exploration_rate(self, exploration_rate_history):
        plt.figure(figsize=(12, 8))
        plt.plot(exploration_rate_history, label='Exploration Rate')
        plt.title('Exploration Rate Decay Over Time')
        plt.xlabel('Episode (x1000)')
        plt.ylabel('Exploration Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

