class ReplayBuffer():
    
    def __init__(self, capacity):
        '''
        Initializes ReplayBuffer with given capacity.

        Args:
            capacity: the desired capacity (max size) of buffer
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

        Args:
            state: old state
            action: action took
            next_state: resulting state from action
            reward: reward received from state
            done: whether the episode is terminal
        '''
        experience = (state, action, next_state, reward, done)
        if self.push_count < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        ''' 
        Returns a randomly selected batch from the memory list. If
        the batch size is larger than the memory size, None is returned.

        Args:
            batch_size: desired size of batch
        Returns:
            List of random memory samples
        '''
        try:
            return random.sample(self.memory, batch_size)
        except ValueError:
            return None