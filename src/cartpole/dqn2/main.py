from env import Env
from agent import Agent
from replay_memory import ReplayBuffer
from DQN import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

EPS_MAX = 1
EPS_MIN = 0.01
EPS_DECAY = 1e-3
ALPHA = 0.001
GAMMA = 0.99
RM_SIZE = 100000
BATCH_SIZE = 32
NUM_EPISODES = 1000
MAX_STEPS = 200
TARGET_UPDATE = 10

POLICY_NET_PATH = "res/policy_net.pt"
TARGET_NET_PATH = "res/target_net.pt"

# experience tuple - (state, action, next_state, reward, done)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(device)
agent = Agent(
        eps=EPS_MAX, eps_min=EPS_MIN, eps_max=EPS_MAX, eps_decay=EPS_DECAY, num_actions=env.num_actions, device=device
        )
memory = ReplayBuffer(RM_SIZE)

policy_net = DQN(obs_space=env.obs_space, num_actions=env.num_actions)
target_net = DQN(obs_space=env.obs_space, num_actions=env.num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=ALPHA)

def experience_replay(experiences):
    '''
    Performs experience replay using memory buffer and the given
    experience batch.

    Loss is calculated using Mean Squared Error between current q-values
    and optimal q-values calculated using Bellman Optimality.
    '''
    exp_zip = list(zip(*experiences))

    state_tensors = torch.cat(exp_zip[0])
    action_tensors = torch.cat(exp_zip[1])
    next_state_tensors = torch.cat(exp_zip[2])
    reward_tensors = torch.cat(exp_zip[3])
    done_list = exp_zip[4]

    actions_index = action_tensors.unsqueeze(1)
    current_q_values = torch.gather(policy_net(state_tensors.float()), dim=1, index=actions_index)
    next_q_values = torch.zeros(len(experiences))

    next_state_indices = []
    for i in range(next_state_tensors.size()[0]):
        if not done_list[i]:
            next_state_indices.append(i)
    next_state_tensors = next_state_tensors[next_state_indices]
    next_q_values[next_state_indices] = target_net(next_state_tensors.float()).max(dim=1)[0].detach()

    optimal_q_values = reward_tensors + (next_q_values * GAMMA)
    assert current_q_values.size() == optimal_q_values.unsqueeze(1).size()

    return current_q_values, optimal_q_values.unsqueeze(1)

average_rewards = deque(maxlen=50)

def train():
    for episode in range(NUM_EPISODES):
        env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state=env.state, policy_net=policy_net)
            curr_state = env.state
            new_state, reward, done, _ = env.play_action(action)

            # Decide whether to include experience for initial state.
            memory.push(curr_state, action, new_state, reward, done)
            episode_reward += reward.item()

            experiences = memory.sample(BATCH_SIZE)
            if experiences:
                current_q_values, optimal_q_values = experience_replay(experiences)
                loss = F.mse_loss(current_q_values, optimal_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if env.done:
                break

        average_rewards.append(episode_reward)

        if memory.is_full():
            print("Memory is full.")

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 50 == 0:
            print("Episode {}: {}".format(episode, sum(average_rewards) / 50))
            print("Epsilon: {}".format(agent.eps))

    torch.save(policy_net.state_dict(), POLICY_NET_PATH)
    torch.save(target_net.state_dict(), TARGET_NET_PATH)

    env.close()

if __name__ == '__main__':
    train()
