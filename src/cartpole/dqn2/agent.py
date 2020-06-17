import math
import random
import torch

class Agent():
    def __init__(
            self, eps, eps_min, eps_max, eps_decay, num_actions, device
            ):
        '''
        Initializes Agent object.
        '''
        self.eps = eps
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.num_actions = num_actions
        self.current_episode = 0
        self.device = device

    def select_action(self, state, policy_net):
        '''
        Selections action given current state and policy net.

        Use epsilon greedy strategy to choose between exploration
        and exploitation. Note that gradients are temporarily set
        off in policy_net when completing a forward pass in order
        to get the maximum q-value
        '''
        self.update_epsilon()
        if random.random() > self.eps:
            with torch.no_grad():
                res = policy_net(state.float()).argmax(dim=1).to(self.device)
        else:
            action = random.randrange(self.num_actions)
            res = torch.tensor([action], device=self.device)

        self.current_episode += 1
        return res

    def update_epsilon(self):
        '''
        Updates epsilon value based on the current episode.
        '''
        self.eps = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-self.eps_decay * self.current_episode)
