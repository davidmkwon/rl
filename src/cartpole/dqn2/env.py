import gym
import numpy as np
import torch

class Env():
    def __init__(self, device, env_name='CartPole-v0'):
        '''
        Initializes Env object.

        The state field is always stored as a tensor,
        updated immediately after receiving the next
        state from the gym environment.
        '''
        self.device = device
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.state = Env.state_to_tensor(self.state)
        self.done = False

        self.num_actions = self.env.action_space.n
        self.obs_space = self.env.observation_space.shape[0]

    def reset(self):
        '''
        Resets the environment.
        '''
        self.env.reset()
        self.done = False

    def play_action(self, action):
        '''
        Plays the given action in the environment.

        State is updated to tensor-form and returned.
        '''
        self.state, reward, self.done, info = self.env.step(action.item())
        self.state = Env.state_to_tensor(self.state).to(self.device)
        reward = torch.tensor([reward], device=self.device)
        return self.state, reward, self.done, info
    
    @staticmethod
    def state_to_tensor(state):
        '''
        Static method that converts state tuple to pytorch tensor.

        If the state is the initial state, it will be a single
        numpy float--we construct a pytorch tensor with the other
        fields set to 0.
        '''
        if type(state) is np.float64:
            arr = np.zeros(4)
            arr[0] = state
            return torch.tensor(arr, dtype=torch.float64).unsqueeze(0)
        else:
            return torch.tensor(state, dtype=torch.float64).unsqueeze(0)

    def close(self):
        '''
        Closes the environment.
        '''
        self.env.close()
