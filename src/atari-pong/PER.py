import random
import numpy as np
from sumtree import SumTree

class PriorityReplayBuffer(object):
    # TODO: reference https://github.com/rlcode/per/blob/master/prioritized_memory.py

    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        '''
        Args:
            capacity: capacity of backing SumTree
        '''
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        '''
        Args:
            error: input error
        Returns:
            the associated priority
        '''
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, experience):
        '''
        Args:
            error: TD error of the sample
            sample: experience to enter
        '''
        p = self._get_priority(error)
        self.tree.add(p, experience)

    def sample(self, size):
        '''
        Args:
            size: the desired batch size to receive
        Returns:
            the batch of experiences, indexes, and importance sampling weights
        '''
        batch = []
        idxs = []
        segment = self.tree.total() / size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        '''
        Args:
            idx: the SumTree index to update
            error: the error of the experience
        '''
        p = self._get_priority(error)
        self.tree.update(idx, p)

class ReplayBuffer(object):
    def __init__(self, capacity):
        '''
        Initalizes ReplayMemory.
        '''
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, state, action, next_state, reward, done):
        '''
        Adds an experience to the memory.

        If the current push_count exceeeds the
        capacity then we will start replacing
        the memory starting from the oldest experiences.

        experience tuple - (state, action, next_state, reward, done)
        '''
        experience = (state, action, next_state, reward, done)
        if self.push_count < self.capacity:
            self.memory.append(experience)
        else:
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
