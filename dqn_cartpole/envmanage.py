import gym
import numpy as np
import torch
import torchvision.transforms as T

class EnvManager():
    def __init__(self, device, env_name='CartPole-v0'):
        '''
        Initializes EnvManager.

        gym environment is reset upon initalization.
        '''
        self.device = device
        self.env = gym.make(env_name).unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        '''
        Resets environment and current_screen.
        '''
        self.env.reset()
        self.current_screen = None

    def take_action(self, action):
        '''
        Executes given action and returns reward
        (as Torch Tensor).
        '''
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def close(self):
        '''
        Closes environment.
        '''
        self.env.close()

    def render(self, mode='human'):
        '''
        Renders environment.
        '''
        return self.env.render(mode)

    def num_actions_available(self):
        '''
        Returns number of currently available actions.
        '''
        return self.env.action_space.n

    def get_state(self):
        '''
        Returns the difference between the current
        and previous screen.

        If environment is in an unitialized state,
        then a empty screen is returned.
        '''
        if self.current_screen is None or self.done:
            self.current_screen = self.get_processed_screen()
            empty_screen = torch.zeros_like(self.current_screen)
            return empty_screen
        else:
            old_screen = self.current_screen
            self.current_screen = self.get_processed_screen()
            return self.current_screen - old_screen

    def get_screen_height(self):
        '''
        Returns screen height.
        '''
        return self.get_processed_screen().shape[2]

    def get_screen_width(self):
        '''
        Returns screen width.
        '''
        return self.get_processed_screen().shape[3]

    def get_processed_screen(self):
        '''
        Returns processed version of screen.

        Renders current state as rgb array, crops top
        and bottom, and returns array in proper format.
        '''
        screen = self.render('rgb_array').transpose((2,0,1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        '''
        Crops top 40% and bottom 20% of screen.
        '''
        screen_height = screen.shape[1]

        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        '''
        Processes screen.

        Screen is resized to a 40 x 90 dimension, ultimately
        leaving the final dimensions of transformed screen
        as [1,3,40,90]
        '''
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40,90)),
            T.ToTensor()
        ])
        return resize(screen).unsqueeze(0).to(self.device)
