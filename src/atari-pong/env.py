import gym
import torch
from skimage import transform
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np

class Env():
    
    def __init__(self, device, env_name='PongDeterministic-v4'):
        '''
        Initialize Env object.
        
        Args:
            env_name: name of Gym environment
            device: PyTorch device for computations
        '''
        self.env = gym.make(env_name)
        self.device = device
        self.done = False
        self.state = None
        self.rgb_state = None
        self.reset()

        self.num_actions = self.env.action_space.n

    def play_action(self, action):
        '''
        Executes given action in the environment.
        
        Args:
            action: index of action to execute
        Returns:
            (processed) new state, reward, done, and info
        '''
        self.rgb_state, reward, self.done, info = self.env.step(action)
        self.state = Env.preprocess_state(self.rgb_state)
        reward = np.array([reward])
        return self.state, reward, self.done, info

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
        self.rgb_state = self.render(mode='rgb_array')
        self.state = Env.preprocess_state(self.rgb_state)

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
