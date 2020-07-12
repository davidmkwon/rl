import math
import random
import torch

class Agent():
    def __init__(self, eps, eps_min, eps_max, eps_decay, num_actions, device):
        '''
        Initializes Agent object.
        '''
        self.eps = eps
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.eps_off = False
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

        Args:
            state: current state of the environment
            policy_net: the network used to find q-values
        Returns:
            the index of the best action to take 
        '''
        if self.eps_off:
            with torch.no_grad():
                res = policy_net(state.float()).argmax(dim=1).item()
        else:
            self.update_epsilon()
            if random.random() > self.eps:
                with torch.no_grad():
                    # currently state is in [4,84,84] size input -> change to [1,4,84,84] for CNN
                    state = torch.from_numpy(state).to(self.device)
                    state = state.unsqueeze(0)
                    res = policy_net(state.float()).argmax(dim=1).item()
            else:
                res = random.randrange(self.num_actions)

            self.current_episode += 1

        return res

    def update_epsilon(self):
        '''
        Updates epsilon value based on the current episode.
        '''
        self.eps = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-self.eps_decay * self.current_episode)

    def turn_eps_off(self):
        self.eps_off = True
