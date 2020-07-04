import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):

    def __init__(self, num_frames, num_actions):
        '''
        Initialize neural net.

        Args:
            num_frames: number of frames in the stack, equivalant to number of channels
            num_actions: number of actions for agent, number of ouput neurons
        '''
        super(DDQN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_frames,
            out_channels=16,
            kernel_size=8,
            stride=4
            )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2
            )
        self.fc = nn.Linear(
            in_features=32*9*9,
            out_features=256
            )
        self.output = nn.Linear(
            in_features=256,
            out_features=num_actions
            )

    def forward(self, t):
        '''
        Forward pass of neural net. Relu activation function used.

        Args:
            t: the tensor to pass forward
        '''
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = torch.flatten(t)
        t = F.relu(self.fc(t))
        t = self.output(t)

        return t
