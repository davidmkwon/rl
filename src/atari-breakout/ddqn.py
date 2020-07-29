import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):

    def __init__(self, num_frames, num_actions):
        '''
        Initialize neural net.

        Args:
            num_frames: number of frames in the stack, equivalant to number of channels
            num_actions: number of actions for agent, number of ouput neurons
        '''
        super(DuelingDQN, self).__init__()
        self.num_actions = num_actions

        # Convolutional layers
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

        # Fully connected layer 1 for advantage and value function
        self.fc_adv = nn.Linear(
            in_features=32*9*9,
            out_features=256
            )
        self.fc_val = nn.Linear(
            in_features=32*9*9,
            out_features=256
            )

        # Fully connected layer 2 for advantage and value function
        self.output_adv = nn.Linear(
            in_features=256,
            out_features=num_actions
            )
        self.output_val = nn.Linear(
            in_features=256, out_features=1
            )

    def forward(self, t):
        '''
        Forward pass of neural net. Relu activation function used.

        Args:
            t: the tensor to pass forward
        Returns:
            q values for each action
        '''
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.view(t.size(0), -1)

        adv = F.relu(self.fc_adv(t))
        val = F.relu(self.fc_val(t))
        adv = self.output_adv(adv)
        val = self.output_val(val).expand(t.size(0), self.num_actions)

        # subtract mean advantage from q values
        t = val + adv - adv.mean(1).unsqueeze(1).expand(t.size(0), self.num_actions)

        return t
