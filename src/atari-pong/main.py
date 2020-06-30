from env import Env
from frstack import Frstack

import torch
import numpy as np

# Hyperparameters
MEMORY_SIZE = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory = ReplayBuffer(MEMORY_SIZE)
env = Env(device)

stack = Frstack(initial_frame=env.state)
print(stack.get_stack())
print(stack.get_stack().shape)

t1 = torch.zeros((5, 5), dtype=torch.uint8)
t2 = torch.zeros((5, 5), dtype=torch.uint8)
tc = torch.stack((t1, t2), dim=2)

env.close()

#stack = np.stack((n1, n2), axis=2)
# print(n1)
# print(stack.shape)
# print(stack.shape)
