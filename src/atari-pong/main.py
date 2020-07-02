from env import Env
from frstack import Frstack
from PER import PriorityReplayBuffer
from ddqn import DDQN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Hyperparameters
EPS_MAX = 1
EPS_MIN = 1e-2
EPS_DECAY = 1e-3
ALPHA = 1e-3
GAMMA = 0.99
RM_SIZE = 20000
BATCH_SIZE = 32
NUM_EPISODES = 800
NUM_TEST_EPISODES = 100
TARGET_UPDATE = 10
SAVE_UPDATE = 10

POLICY_NET_PATH = "res/policy_net.pt"
TARGET_NET_PATH = "res/target_net.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory = PriorityReplayBuffer(MEMORY_SIZE)
env = Env(device)

stack = Frstack(initial_frame=env.state)
# print(stack.get_stack())
# print(stack.get_stack().shape)

policy_net = DDQN(stack.frame_count, env.num_actions)
state = stack.get_stack()
state = state.unsqueeze(0)
output = policy_net(state.float())
print(output.shape)

env.close()