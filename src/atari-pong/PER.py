import random
import numpy as np
from sumtree import SumTree

class PriorityReplayBuffer(object):
    # TODO: reference https://github.com/rlcode/per/blob/master/prioritized_memory.py

    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    absolute_error_upper = 1.

    def __init__(self, capacity):
        '''
        Initializes PRB.

        Args:
            capacity: capacity of backing SumTree
        '''
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        '''
        Gets the priority associated with the error.

        Args:
            error: input error
        Returns:
            the associated priority
        '''
        priority = np.abs(error) + self.e
        priority = np.minimum(priority, self.absolute_error_upper)
        return priority ** self.a

    def add(self, error, experience):
        '''
        Adds the experience and error to the SumTree.

        Args:
            error: TD error of the sample
            sample: experience to enter
        '''
        priority = self._get_priority(error)
        self.tree.add(experience, priority)

    def sample(self, size):
        '''
        Returns a sample with given size following weighted distribution.

        Args:
            size: the desired batch size to receive
        Returns:
            the batch of experiences, indexes, and importance sampling weights
        '''
        batch = []
        idxs = []
        segment = self.tree.total_priority() / size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        # p_min = np.min(self.tree.tree[-self.tree.capacity:])
        # max_weight = (p_min * size) ** (-self.beta)

        for i in range(size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_priority()
        is_weight = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        '''
        Updates the tree index with the error.
        
        Args:
            idx: the SumTree index to update
            error: the error of the experience
        '''
        p = self._get_priority(error)
        self.tree.update(idx, p)
