from env import Env
from agent import Agent
from replay_memory import ReplayMemory
from DQN import DQN

import torch
import torch.optim as optim

EPS_MAX = 1
EPS_MIN = 0.01
EPS_DECAY = 1e-3
ALPHA = 0.001
RM_SIZE = 100000
BATCH_SIZE = 32
NUM_EPISODES = 1000
MAX_STEPS = 200
TARGET_UPDATE = 10

# experience tuple - (state, action, next_state, reward, done)

device = torch.device("cpu")
env = Env(device)
agent = Agent(
        eps=EPS_MAX, eps_min=EPS_MIN, eps_max=EPS_MAX, eps_decay=EPS_DECAY, num_actions=env.num_actions, device=device
        )
memory = ReplayMemory(RM_SIZE)

policy_net = DQN(obs_space=env.obs_space, num_actions=env.num_actions)
target_net = DQN(obs_space=env.obs_space, num_actions=env.num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=ALPHA)

def extract_tensors(experiences):
    batch = list(zip(*experiences))

def experience_replay(experiences):
    for state, action, reward, next_state, done in experiences:


for episode in range(NUM_EPISODES):
    env.reset()

    for step in range(MAX_STEPS):
        action = agent.select_action(state=env.state, policy_net=policy_net)
        curr_state = env.get_state()
        new_state, reward, done, _ = env.play_action(action)

        if step != 0:
            memory.push(curr_state, action, new_state, reward, done)

        experiences = memory.sample(BATCH_SIZE)
        if experiences:
            # do cool stuff

        if env.done:
            break

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
