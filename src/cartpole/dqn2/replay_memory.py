import random
from collections import namedtuple

class ReplayMemory():
    def __init__(self, capacity):
        '''
        Initalizes ReplayMemory.
        '''
        self.capacity = capacity
        self.memory = [0 for i in range(self.capacity)]
        self.push_count = 0

    def push(self, experience):
        '''
        Adds an experience to the memory.

        If the current push_count exceeeds the
        capacity then we will start replacing
        the memory starting from the oldest experiences.
        '''
        self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        '''
        Returns a randomly selected batch from
        the memory list.

        If the batch size is larger than the memory
        size, None is returned
        '''
        try:
            return random.sample(self.memory, batch_size)
        except ValueError:
            return None
