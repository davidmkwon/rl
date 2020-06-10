import gym
import pandas as pd
import numpy as np

class Env():

    def __init__(self, env_name='CartPole-v0'):
        '''
        Initialize Env.

        Bins are used for discretizing observation measurements.
        obs_state refers to the original observation from the Gym
        environment while state refers to the condensed version.
        '''
        self.env = gym.make(env_name)

        self.cart_position_bins = pd.cut([-2.4, 2.4], bins=10, labels=False, retbins=True)[1][1:-1]
        self.pole_angle_bins = pd.cut([-2, 2], bins=10, labels=False, retbins=True)[1][1:-1]
        self.cart_velocity_bins = pd.cut([-1, 1], bins=10, labels=False, retbins=True)[1][1:-1]
        self.pole_velocity_bins = pd.cut([-3.2, 3.2], bins=10, labels=False, retbins=True)[1][1:-1]

        self.obs_state = self.env.reset()
        self.state = self.get_condensed_state(self.obs_state)
        self.done = False
        self.num_actions = self.env.action_space.n

    def play_action(self, action):
        '''
        Executes given action. Wrapper for gym env.
        '''
        self.obs_state, reward, self.done, info = self.env.step(action)
        self.state = self.get_condensed_state(self.obs_state)
        return self.state, reward, self.done, info

    def get_condensed_state(self, observation):
        '''
        Condenses given observation.
        '''
        first = np.digitize(x=observation[0], bins=self.cart_position_bins)
        second = np.digitize(x=observation[1], bins=self.pole_angle_bins)
        third = np.digitize(x=observation[2], bins=self.cart_velocity_bins)
        fourth = np.digitize(x=observation[3], bins=self.pole_velocity_bins)

        state = first * 1000 + second * 100 + third * 10 + fourth
        return int(state)

    def reset(self):
        '''
        Resets Env. Wrapper for gym env.
        '''
        self.obs_state = self.env.reset()
        self.state = self.get_condensed_state(self.obs_state)
        self.done = False

    def close(self):
        '''
        Closes Env. Wrapper for gym env.
        '''
        self.env.close()
