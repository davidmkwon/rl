import numpy as np

class SumTree(object):
    # TODO: consider rewriting this with all the data stored in a SumNode object
    # TODO: reference implementation: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    # TODO: reference implementation: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb

    def __init__(self, capacity):
        '''
        Args:
            capacity: the desired capacity, or number of leaf nodes, in the SumTree
        '''
        self.capacity = capacity
        self.data_pointer = 0
        self.tree = np.zeros((2 * capacity) - 1)
        self.data = np.zeros(self.capacity, dtype=object)

    def add(self, experience, priority):
        '''
        Args:
            experience: the experience to add
            priority: the priority of the data
        '''
        self.data[self.data_pointer] = experience
        tree_index = self.data_pointer + self.capacity - 1
        self.update(tree_index, priority)
        self.data_pointer += 1

        # Bring pointer back to beginning if capacity exceeded
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, new_priority):
        '''
        Args:
            tree_index: the index to update respective to the backing array
            new_priority: the new priority value
        '''
        delta = new_priority - self.tree[tree_index]
        self.tree[tree_index] = new_priority
        pointer = tree_index

        while pointer != 0:
            pointer = (pointer - 1) // 2
            self.tree[pointer] += delta

    def get_leaf(self, v):
        '''
        Args:
            v: random value from 0 to E(priorities)
        Returns:
            something.
        '''
        pointer = 0

        while True:
            left, right = pointer * 2 + 1, pointer * 2 + 2
            if left >= len(self.tree):
                tree_index = pointer
                break
            else:
                if v <= self.tree[left]:
                    pointer = left
                else:
                    v -= self.tree[left]
                    pointer = right

        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]
