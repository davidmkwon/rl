from agent import Agent
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

# set up environment/agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(device)
agent = Agent(
        eps=EPS_MAX, eps_min=EPS_MIN, eps_max=EPS_MAX, eps_decay=EPS_DECAY, num_actions=env.num_actions, device=device
        )
memory = PriorityReplayBuffer(RM_SIZE)
stack = Frstack(initial_frame=env.state)

# initialize policy and target network
policy_net = DDQN(stack.frame_count, env.num_actions)
target_net = DDQN(stack.frame_count, env.num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# TODO: consider RMSProp vs Adam - DeepMind paper uses RMSProp
optimizer = optim.Adam(params=policy_net.parameters(), lr=ALPHA)

def experience_replay():
    batch, idxs, is_weights = memory.sample(BATCH_SIZE)
    batch = list(zip(*batch))

    state_tensors = torch.cat(batch[0])
    action_tensors = torch.cat(exp_zip[1])
    next_state_tensors = torch.cat(exp_zip[2])
    reward_tensors = torch.cat(exp_zip[3])
    dones_tensor = torch.FloatTensor(exp_zip[4])

    # find q-values for current states through policy_net
    actions_index = action_tensors.unsqueeze(1)
    current_q_values = torch.gather(policy_net(state_tensors.float()), dim=1, index=actions_index)

    # find optimal q-values by finding the maximum q-value action indices using the policy_net
    # and evaluating them with the target_net (Double DQN)
    best_action_indices = policy_net(next_state_tensors).detach().max(dim=1)[0]
    next_q_values = target_net(next_state_tensors).detach().gather(dim=1, index=best_action_indices.unsqueeze(1))
    # next_q_values = next_q_values.squeeze()
    optimal_q_values = reward_tensors + (1 - dones_tensor) * GAMMA * next_q_values

    # update the Prioritized Replay Buffer
    errors = torch.abs(current_q_values - optimal_q_values).data.numpy()
    for i in range(BATCH_SIZE):
        idx = idxs[i]
        self.memory.update(idx, errors[i])

    # backpropogate network with MSE loss
    optimizer.zero_grad()
    loss = (torch.FloatTensor(is_weights) * F.mse_loss(current_q_values, optimal_q_values)).mean()
    loss.backward()
    optimizer.step()

env.close()
