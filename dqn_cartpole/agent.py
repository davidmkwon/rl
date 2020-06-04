import random
import torch

class Agent():
    def __init__(self, strategy, num_actions, device):
        '''
        Initializes Agent.
        '''
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        '''
        Returns appropriate action given current
        state and DQN

        Uses epsilon greedy strategy to calculate
        explore vs exploit tradeoff.
        '''
        epsilon = self.strategy.get_epsilon(self.current_step)
        if random.random() > epsilon:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)
        else:
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)

        self.current_step += 1
