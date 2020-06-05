import random
import pandas as pd
import numpy as np

class Agent():
    
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay,
            gamma, alpha, num_actions):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.alpha = alpha
        self.q_table = {}
        self.num_actions = num_actions

    def create_state(self, observation):
        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, labels=False, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1.5, 1.5], bins=10, labels=False, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-0.21, 0.21], bins=10, labels=False, retbins=True)[1][1:-1]
        pole_velocity_bins = pd.cut([-3.2, 3.2], bins=10, labels=False, retbins=True)[1][1:-1]

        cp_bin = np.digitize(x=observation[0], bins=cart_position_bins)
        cv_bin = np.digitize(x=observation[1], bins=cart_velocity_bins)
        pa_bin = np.digitize(x=observation[2], bins=pole_angle_bins)
        pv_bin = np.digitize(x=observation[3], bins=pole_velocity_bins)

        state = cp_bin * 1000 + cv_bin * 100 + pa_bin * 10 + pv_bin
        return state
