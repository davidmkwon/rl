import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, obs_space, num_actions):
        '''
        Initializes deep network, consisting of 3 layers.
        '''
        super().__init__()
        self.fc1 = nn.Linear(in_features=obs_space, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=24)
        self.out = nn.Linear(in_features=24, out_features=num_actions)

    def forward(self, t):
        '''
        Performs forward pass of network using Rectified
        Linear Unit activation function.
        '''
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
