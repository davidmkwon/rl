from agent import Agent
from env import Env
from DQN import DQN
from main import EPS_MAX, EPS_MIN, EPS_DECAY, POLICY_NET_PATH, NUM_TEST_EPISODES, MAX_STEPS
import utils

import torch

SHOW_PLAYING = NUM_TEST_EPISODES - 2

# set up environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(device)
agent = Agent(
        eps=EPS_MAX, eps_min=EPS_MIN, eps_max=EPS_MAX, eps_decay=EPS_DECAY, num_actions=env.num_actions, device=device
        )
agent.turn_eps_off()

# initialize policy network (will copy the saved parameters from training)
policy_net = DQN(obs_space=env.obs_space, num_actions=env.num_actions)

def test():
    agent.eps = 0
    all_rewards = []
    policy_net.load_state_dict(torch.load(POLICY_NET_PATH))
    policy_net.eval()

    for episode in range(NUM_TEST_EPISODES):
        env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            if episode >= SHOW_PLAYING:
                env.render()

            action = agent.select_action(state=env.state, policy_net=policy_net)
            _, reward, done, _ = env.play_action(action)

            episode_reward += reward.item()

            if env.done:
                break

        all_rewards.append(episode_reward)

    utils.plot_testing(all_rewards)

if __name__ == '__main__':
    print('Testing...')
    test()
    print('Done!')
    env.close()