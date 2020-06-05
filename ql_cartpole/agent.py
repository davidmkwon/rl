import math
import random
import pandas as pd
import numpy as np

class Agent():
    
    def __init__(self, eps_start, eps_end, eps_decay, gamma, alpha, num_actions):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.alpha = alpha
        self.q_table = {}
        self.num_actions = num_actions
        self.current_step = 0 

    def select_action(self, state):
        eps = self.get_epsilon()
        if random.random() > eps:
            self.add_state(state)
            if self.q_table[state][0] == self.q_table[state][1]:
                return random.randrange(self.num_actions)
            else:
                return np.argmax(self.q_table[state])
        else:
            return random.randrange(self.num_actions)
        
        self.current_step += 1

    def update_q_table(self, old_state, action, reward, new_state):
        self.add_state(old_state)
        self.add_state(new_state)
        self.q_table[old_state][action] = (1 - self.alpha) * self.q_table[old_state][action] + \
                self.alpha * (reward + self.gamma * max(self.q_table[new_state]))

    def add_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0 for i in range(self.num_actions)]

    def get_epsilon(self):
        # eps = self.eps_start + (self.eps_end - self.eps_start) * math.exp(-self.eps_decay * self.current_step)
        self.eps_start *= 0.99
        return self.eps_start
