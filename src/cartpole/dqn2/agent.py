import math
import random
import torch

class Agent():
    def __init__(
            self, eps=1, eps_min=0.1, eps_max=1, eps_decay=0.001, num_actions, device
            ):
        self.eps = eps
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.num_actions = num_actions
        self.current_episode = 0
        self.device = device

    def select_action(self, state, policy_net):
        self.update_epsilon()
        if random.random() > self.eps:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)
        else:
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)

        self.current_episode += 1

    def update_epsilon(self):
        self.eps = self.start + (self.end - self.start) * math.exp(-self.decay * self.current_episode)
