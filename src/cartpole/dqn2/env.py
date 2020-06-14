import gym
import numpy as np

class Env():
    def __init__(self, device, env_name='CartPole-v0'):
        self.device = device
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.done = False
        self.num_actions = self.env.action_space.n
        self.obs_space = self.env.observation_space.shape[0]

    def reset(self):
        self.env.reset()
        self.done = False

    def play_action(self, action):
        self.state, reward, self.done, info = self.env.step(action)
        return self.state, reward, self.done, info

    def get_state(self):
        if type(self.state) is np.float64:
            arr = np.zeros(4)
            arr[0] = self.state
            self.state = arr
        return self.state

    def close(self):
        self.env.close()
