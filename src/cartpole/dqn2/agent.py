import math
import random
import torch

class Agent():
    def __init__(
            self, eps=1, eps_min=0.1, eps_max=1, eps_decay = 0.001, num_actions, device='cpu', policy_net, target_net
            ):
        self.eps = eps
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.num_actions = num_actions
        self.current_episode = 0
        self.device = device

        self.policy_net = policy_net
        self.target_net = target_net

    def select_action(self, state):
        self.update_epsilon()
        if random.random() > self.eps:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).to(self.device)
        else:
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)

        self.current_episode += 1

    def update_epsilon(self):
        self.eps = self.start + (self.end - self.start) * math.exp(-self.decay * self.current_episode)
