import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        '''
        Initializes deep network, consisting of 3 layers.
        Note that the third layer outputs 2 values representing
        the 2 possible actions for the agent.
        '''
        super().__init__()
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        '''
        Completes a forward pass of the network using
        Recitified Linear Unit activation function.
        '''
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t
