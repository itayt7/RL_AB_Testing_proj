#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from collections import defaultdict

  
class QLearningAgent:
    def __init__(self, n_actions, state_size,
                 learning_rate=0.1,
                 discount_factor=0.9,
                 initial_exploration_rate=1.0,
                 exploration_decay=0.99995, ##0.99995
                 min_exploration_rate=0.05,
                 lambda_reg=0.01, beta=0.9):
        self.n_actions = n_actions
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = initial_exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        old_value = self.q_table[state][action]
        new_value = old_value + self.learning_rate * td_error
        self.q_table[state][action] =self.beta * old_value + (1 - self.beta) * new_value - self.lambda_reg * self.q_table[state][action]

    def decay_exploration(self):
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.min_exploration_rate)




# class QLearningAgent:
#     def __init__(self, n_actions, state_size,
#                  learning_rate=0.0.1, discount_factor=0.999,
#                  initial_exploration_rate=1.0, exploration_decay=0.995,
#                   min_exploration_rate=0.2 ,lambda_reg=0.01):
#         self.n_actions = n_actions
#         self.state_size = state_size
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = initial_exploration_rate
#         self.exploration_decay = exploration_decay
#         self.min_exploration_rate = min_exploration_rate
#         self.lambda_reg = lambda_reg
#         self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
    
#     def choose_action(self, state):
#         if np.random.rand() < self.exploration_rate:
#             return np.random.randint(self.n_actions)
#         return np.argmax(self.q_table[state])
    
#     def update_q_table(self, state, action, reward, next_state):
#         best_next_action = np.argmax(self.q_table[next_state])
#         td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
#         td_error = td_target - self.q_table[state][action]
#         self.q_table[state][action] += self.learning_rate * td_error - self.lambda_reg * self.q_table[state][action]
# #         self.q_table[state][action] += self.learning_rate * td_error

#     def decay_exploration(self):
#         self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.min_exploration_rate)
        