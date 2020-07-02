import gym
import torch

from skimage import transform
from skimage.color import rgb2gray

from matplotlib import pyplot as plt

class Env():
    
    def __init__(self, device, env_name='PongDeterministic-v4'):
        '''
        Initialize Env object.
        
        Args:
            env_name: name of Gym environment
            device: PyTorch device for computations
        '''
        self.env = gym.make(env_name)
        self.done = False
        self.state = None
        self.reset()
        self.device = device

        self.num_actions = self.env.action_space.n

    def play_action(self, action):
        '''
        Executes given action in the environment.
        
        Args:
            action: index of action to execute
        Returns:
            
        '''
        return

    @staticmethod
    def state_to_tensor(state):
        '''
        Processes state to PyTorch tensor, delegating to preprocess_state().
        
        Args:
            state: RGB array from rendered environment
        Returns:
            84 x 84 PyTorch tensor
        '''
        state = Env.preprocess_state(state)
        state = torch.tensor(state, dtype=torch.float64).to(self.device)
        return state

    @staticmethod
    def preprocess_state(state, RESIZE_HEIGHT=84, RESIZE_WIDTH=84):
        '''
        Converts state to Grayscale, crops top/bottom, and resizes
        
        Args:
            state: RGB array from rendered environment
        Returns:
            84 x 84 numpy array
        '''
        state = rgb2gray(state)
        state = state[20:-5]
        state = transform.resize(state, [RESIZE_HEIGHT, RESIZE_WIDTH])
        return state

    def reset(self):
        '''
        Resets the environment.
        '''
        self.done = False
        self.env.reset()
        self.state = self.render(mode='rgb_array')
        self.state = Env.state_to_tensor(self.state)

    def render(self, mode='human'):
        '''
        Renders environment.
        
        Args:
            mode: type of rendering (human or rgb_array)
        '''
        return self.env.render(mode)
    
    def show_state(self):
        '''
        Plots current state.
        '''
        plt.imshow(self.state.numpy(), cmap='gray')
        plt.show()
        
    def close(self):
        '''
        Closes the environment.
        '''
        self.env.close()
