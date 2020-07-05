from agent import Agent
from env import Env
from frstack import Frstack
from PER import PriorityReplayBuffer
from ddqn import DDQN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from itertools import count

# Hyperparameters
EPS_MAX = 1
EPS_MIN = 1e-2
EPS_DECAY = 5e-5
ALPHA = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 100000
BATCH_SIZE = 32
NUM_EPISODES = 10000
NUM_TEST_EPISODES = 100
PRE_TRAIN_LENGTH = 30000
TAU = 10000
SAVE_UPDATE = 50
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

def train():
    average_rewards = deque(maxlen=50)
    max_reward = 0
    all_rewards = []

    for episode in range(NUM_EPISODES + PRE_TRAIN_LENGTH):
        env.reset()
        episode_reward = 0
        stack.push(env.state, True)
        curr_state = stack.get_stack()
        next_state = None

        for step in count():
            # if we are in pre-training phase/filling up memory, then pick
            # a random action
            if memory.tree.size <= PRE_TRAIN_LENGTH:
                action = random.choice(env.num_actions)
            else:
                action = agent.select_action(state=env.state, policy_net=policy_net)

            next_state, reward, done, _ = env.play_action(action)
            stack.push(next_state, False)
            next_state = stack.get_stack()

            # experience tuple - (state, action, next_state, reward, done)
            assert curr_state.shape == torch.Size((4,84,84))
            assert next_state.shape == torch.Size((4,84,84))
            memory.add(curr_state, action, new_state, reward, done)

            if memory.tree.size >= PRE_TRAIN_LENGTH:
                experience_replay()

            episode_reward += reward.item()
            curr_state = next_state

        if episode % TAU == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % SAVE_UPDATE == 0:
            torch.save(policy_net.state_dict(), POLICY_NET_PATH)
            torch.save(target_net.state_dict(), TARGET_NET_PATH)

if __name__ == '__main__':
    print("Training...")
    train()
    print("Done!")
    env.close()
