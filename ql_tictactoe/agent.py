import random
import math

class Agent():

    def __init__(self, num_actions, agent_id):
        '''
        Initializes Agent.

        Agent keeps track of state-value function
        rather than action-value, however usage is
        similar to q-learning approach.
        '''
        self.current_step = 0
        self.num_actions = num_actions
        self.agent_id = agent_id
        self.state_value = {}
        self.start = 0
        self.end = 0
        self.decay = 0

    def select_move(self, state, available_moves):
        epsilon = self.get_epsilon()
        if random.random() > epsilon:
            # exploit
        else:
            move = random.choice(available_moves)
            return move

    def get_epsilon(self):
        '''
        Returns value of epsilon, exponentially decaying
        as the number of steps increases.
        '''
        return self.start + (self.end - self.start) * math.exp(-self.decay * current_step)

    def update_params(start, end, decay):
        '''
        Updates epsilon parameters outside of the class
        as a level of abstraction.
        '''
        self.start = start
        self.end = end
        self.decay = decay
