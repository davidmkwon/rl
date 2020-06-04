import random
import math
from collections import namedtuple

'''
Experience tuple.
'''
Experience = namedtuple(
        'Experience',
        ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        '''
        Initializes ReplayMemory object, keeping
        store of memory capacity, the experiences,
        and number of items added.
        '''
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        '''
        Adds experience to memory list.

        If the capacity is already exceeded, we
        begin to replace the oldest items.
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[push_count * self.capacity] = experience

        self.push_count += 1

    def sample(self, batch_size):
        '''
        Returns a randomly selected batch from
        memory.

        If the batch_size is larger than the
        memory size, None is returned
        '''
        try:
            return random.sample(self.memory, batch_size)
        except ValueError:
            return None

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        '''
        Initializes EGS object.
        '''
        self.start = start
        self.end = end
        self.decay = decay

    def get_epsilon(self, current_step):
        '''
        Returns epsilon value for current step
        '''
        return self.start + (self.end - self.start) * math.exp(-self.decay * current_step)
