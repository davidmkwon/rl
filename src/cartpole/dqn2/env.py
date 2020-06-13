import gym

class Env():
    def __init__(self, device, env_name='CartPole-v0'):
        self.device = device
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.done = False
        self.num_actions = self.env.action_space.n

    def reset(self):
        self.env.reset()
        self.done = False

    def play_action(self, action):
        self.state, reward, self.done, info = self.env.step(action)
        return self.state, reward, self.done, info

    def close(self):
        self.env.close()
