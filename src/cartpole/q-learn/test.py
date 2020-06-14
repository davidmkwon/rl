from agent import Agent
from env import Env
import utils

import pickle

agent = Agent(eps=0)
env = Env()

with open('res/q_table.pickle', 'rb') as handle:
    agent.q_table = pickle.load(handle)

rewards = []
max_reward = 0

print('testing')

for episode in range(100):
    env.reset()
    episode_reward = 0

    for step in range(200):
        curr_state = env.state
        action = agent.select_action(curr_state)
        new_state, reward, _, _ = env.play_action(action)
        episode_reward += reward

        if env.done:
            break

    rewards.append(episode_reward)
    if episode_reward > max_reward:
        max_reward = episode_reward

print("max reward:", max_reward)
print("average reward:", sum(rewards) / 100)

utils.plot_testing(rewards)
env.close()
