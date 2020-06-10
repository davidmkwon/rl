import math
import random
import pandas as pd
import numpy as np

class Agent():
    
    def __init__(self, eps=1, gamma=0.9, alpha=0.1, num_actions=2):
        '''
        Initalizes Agent.
        '''
        self.eps = eps
        self.eps_min = 0.1
        self.eps_max = 1
        self.eps_decay = 0.001
        self.gamma = gamma
        self.alpha = alpha
        self.q_table = {}
        self.num_actions = num_actions
        self.current_step = 0 

    def select_action(self, state):
        '''
        Selects and returns an action given a state.

        Q-value policy. Uses epsilon for exploit
        vs. explore tradeoff. Explore: returns random action.
        Exploit: returns highest value from q_table, if all
        q_table values are equal then a random action is returned.
        '''
        if random.random() > self.eps:
            self.add_state(state)
            if self.q_table[state][0] == self.q_table[state][1]:
                return random.randrange(self.num_actions)
            else:
                return np.argmax(self.q_table[state])
        else:
            return random.randrange(self.num_actions)
        
        self.current_step += 1

    def update_q_table(self, old_state, action, reward, new_state):
        '''
        Updates q_table value.

        Implements q[a|s] = (1 - alpha)(q[a|s]) + (alpha)(reward + discount * (max(q[a'|s'])))
        '''
        self.add_state(old_state)
        self.add_state(new_state)
        self.q_table[old_state][action] = (1 - self.alpha) * self.q_table[old_state][action] + \
                self.alpha * (reward + (self.gamma * max(self.q_table[new_state])))

    def add_state(self, state):
        '''
        Adds a state to q_table if it does not already exist.
        '''
        if state not in self.q_table:
            self.q_table[state] = [0 for i in range(self.num_actions)]

    def update_epsilon_complicated(self, episode):
        '''
        Exponential decay on epsilon.
        '''
        self.eps = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-self.eps_decay * episode)
