from agent import Agent
from env import Env
from frstack import Frstack
from PER import PriorityReplayBuffer
from ddqn import DDQN
import utils

from collections import deque
from itertools import count
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# hyperparameters
EPS_MAX = 1
EPS_MIN = 1e-2
EPS_DECAY = 5e-5
ALPHA = 2.5e-4
GAMMA = 0.99
MEMORY_SIZE = 100000
BATCH_SIZE = 32
NUM_EPISODES = 7500
PRE_TRAIN_LENGTH = 40
TAU = 10000
SAVE_UPDATE = 25
POLICY_NET_PATH = "res/policy_net.pt"
TARGET_NET_PATH = "res/target_net.pt"
LOG_EVERY = 100

# important variables for utilizing CUDA
USE_GPU = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# set up environment/agent
'''
Running env.env.unwrapped.get_action_meanings(), we get:
ACTIONS - ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
We will ignore actions 0 and 1.
'''
mod_action_space = [2,3,4,5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(device)
agent = Agent(
        eps=EPS_MAX, eps_min=EPS_MIN, eps_max=EPS_MAX, eps_decay=EPS_DECAY, num_actions=len(mod_action_space), device=device
        )
memory = PriorityReplayBuffer(MEMORY_SIZE)
stack = Frstack(initial_frame=env.state)

# initialize policy and target network
policy_net = DDQN(stack.frame_count, len(mod_action_space))
target_net = DDQN(stack.frame_count, len(mod_action_space))
if USE_GPU:
    policy_net.cuda()
    target_net.cuda()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# TODO: consider RMSProp vs Adam - DeepMind paper uses RMSProp
optimizer = optim.Adam(params=policy_net.parameters(), lr=ALPHA)

def experience_replay():
    # experience tuple - (state, action, next_state, reward, done)
    batch, idxs, is_weights = memory.sample(BATCH_SIZE)
    batch = list(zip(*batch))

    # convert experiences from numpy to CUDA (if available) tensors
    state_tensors = torch.from_numpy(np.stack(batch[0])).type(dtype)
    action_tensors = torch.from_numpy(np.stack(batch[1])).type(dlongtype)
    next_state_tensors = torch.from_numpy(np.stack(batch[2])).type(dtype)
    reward_tensors = torch.from_numpy(np.concatenate(batch[3])).type(dtype)
    dones_tensor = torch.tensor(batch[4]).type(dtype)

    # find q-values for current states through policy_net
    # actions_index = action_tensors.unsqueeze(1)
    current_q_values = torch.gather(policy_net(state_tensors.float()), dim=1, index=action_tensors)

    # find optimal q-values by finding the maximum q-value action indices using the policy_net and evaluating them with the target_net
    best_action_indices = policy_net(next_state_tensors.float()).detach().argmax(dim=1)
    optimal_q_values = target_net(next_state_tensors.float()).detach().gather(dim=1, index=best_action_indices.unsqueeze(1))
    # change dimensions of cq_values and oq_values from [32, 1] -> [32]
    current_q_values, optimal_q_values = current_q_values.squeeze(), optimal_q_values.squeeze()
    optimal_q_values = reward_tensors + (1 - dones_tensor) * GAMMA * optimal_q_values

    assert current_q_values.shape == torch.Size([32])
    assert optimal_q_values.shape == current_q_values.shape

    # update the Prioritized Replay Buffer
    errors = torch.abs(current_q_values - optimal_q_values).detach().cpu().numpy()
    for i in range(BATCH_SIZE):
        idx = idxs[i]
        memory.update(idx, errors[i])

    # backpropogate network with MSE loss
    optimizer.zero_grad()
    loss = (torch.tensor(is_weights).type(dtype) * F.mse_loss(current_q_values, optimal_q_values)).mean()
    loss.backward()
    optimizer.step()

def get_error(experience):
    with torch.no_grad():
        state, action, next_state, reward, done = experience
        state, next_state = torch.tensor(state).type(dtype), torch.tensor(next_state).type(dtype)
        state, next_state = state.unsqueeze(0), next_state.unsqueeze(0)
        action, reward = action.item(), torch.from_numpy(reward).type(dtype)

        current_q_value = policy_net(state.float())
        current_q_value = current_q_value[0][action]

        best_action_index = policy_net(next_state.float()).argmax(dim=1)
        optimal_q_value = target_net(next_state.float())[0][best_action_index.item()]
        optimal_q_value = reward + (1 - done) * GAMMA * optimal_q_value

        error = abs(current_q_value - optimal_q_value)
        return error.cpu().numpy()

def pre_train():
    print("pre-training: filling up replay memory")
    env.reset()
    stack.push(env.state, True)
    curr_state = stack.get_stack()
    next_state = None
    
    for step in range(PRE_TRAIN_LENGTH):
        action = random.randrange(len(mod_action_space))
        next_state, reward, done, _ = env.play_action(mod_action_space[action])
        stack.push(next_state, False)
        next_state = stack.get_stack()

        action = np.array([action])
        experience = (curr_state, action, next_state, reward, done)
        error = get_error(experience)
        memory.add(error=error, experience=experience)

        if env.done:
            env.reset()
            stack.push(env.state, True)
            curr_state = stack.get_stack()
            next_state = None

    assert memory.tree.size == PRE_TRAIN_LENGTH
    print("completed pre training")

def train():
    print("training")

    average_rewards = deque(maxlen=LOG_EVERY)
    all_rewards = []
    tau_count = 0

    for episode in range(NUM_EPISODES):
        env.reset()
        episode_reward = 0
        stack.push(env.state, True)
        curr_state = stack.get_stack()
        next_state = None

        while not env.done:
            # pick an action and execute
            action = agent.select_action(state=curr_state, policy_net=policy_net)
            next_state, reward, done, _ = env.play_action(mod_action_space[action])
            stack.push(next_state, False)
            next_state = stack.get_stack()
            tau_count += 1

            # store experience in memory -> (state, action, next_state, reward, done)
            action = np.array([action])
            experience = (curr_state, action, next_state, reward, done)
            error = get_error(experience)
            memory.add(error=error, experience=experience)

            episode_reward += reward
            curr_state = next_state

            # perform experience replay (replay buffer is already sufficient size)
            experience_replay()

        average_rewards.append(episode_reward)
        all_rewards.append(episode_reward)

        if episode % LOG_EVERY == 0:
            average_reward = sum(average_rewards) / LOG_EVERY
            print("Current episode: {}\nAverge reward: {}\n".format(episode, average_reward))

        if tau_count >= TAU:
            target_net.load_state_dict(policy_net.state_dict())
            tau_count = 0

        if episode % SAVE_UPDATE == 0:
            torch.save(policy_net.state_dict(), POLICY_NET_PATH)
            torch.save(target_net.state_dict(), TARGET_NET_PATH)

    utils.plot_training(all_rewards)
    torch.save(policy_net.state_dict(), POLICY_NET_PATH)
    torch.save(target_net.state_dict(), TARGET_NET_PATH)

    print("done!")

if __name__ == '__main__':
    pre_train()
    train()
    env.close()
