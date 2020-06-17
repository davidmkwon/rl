import agent
import tools
import envmanage
import DQN
from utils import plot

import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

# parameters
BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.001
TARGET_UPDATE = 10
MEMORY_SIZE = 100000
ALPHA = 0.001
NUM_EPISODES = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_manage = envmanage.EnvManager(device)
strategy = tools.EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)
memory = tools.ReplayMemory(MEMORY_SIZE)
agent = agent.Agent(strategy, env_manage.num_actions_available(), device)

policy_net = DQN.DQN(env_manage.get_screen_height(), env_manage.get_screen_width()).to(device)
target_net = DQN.DQN(env_manage.get_screen_height(), env_manage.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=ALPHA)

episode_durations = []

def extract_tensors(experiences):
    batch = tools.Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    print(t1.size())

    return (t1, t2, t3, t4)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        '''
        Returns the q value of the corresponding action
        index given the states
        '''
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        '''
        Returns the q-values for the states in next_states
        that are NOT in their final state. This is because
        the q-values for final states should be 0.

        Because final state tensors have all 0 values, we
        locate them by seeing if the maximum of the tensor
        is 0. If so, values[final_state] = 0
        '''
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

for episode in range(NUM_EPISODES):
    env_manage.reset()
    state = env_manage.get_state()

    for step in count():
        action = agent.select_action(state, policy_net)
        reward = env_manage.take_action(action)
        next_state = env_manage.get_state()
        memory.push(tools.Experience(state, action, next_state, reward))
        state = next_state

        experiences = memory.sample(BATCH_SIZE)
        if experiences:
            states, actions, rewards, next_states = extract_tensors(experiences)

            q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            optimal_q_values = (next_q_values * GAMMA) + rewards

            loss = F.mse_loss(q_values, optimal_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if env_manage.done:
            episode_durations.append(step)
            plot(episode_durations, 100, 50, episode)
            break

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env_manage.close()
