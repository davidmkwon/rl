from env import Env
from agent import Agent

import random
import pandas as pd
import numpy as np
from collections import deque

ALPHA = 0.9
GAMMA = 0.90

EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.001

NUM_EPISODES = 2000
MAX_STEPS = 200

env = Env('CartPole-v0')
agent = Agent(
        eps_start=EPSILON,eps_end=MIN_EPSILON,eps_decay=EPSILON_DECAY,gamma=GAMMA,alpha=ALPHA,num_actions=env.num_actions
        )
rewards = deque()
max_reward = 0

print("training...")

for episode in range(NUM_EPISODES):
    env.reset()
    episode_reward = 0

    if episode % 100 == 0:
        avg_reward = sum(rewards) / 1000
        print("episode {} reached: average reward {}".format(episode, avg_reward))
        print("current max reward: {}".format(max_reward))
        rewards.clear()

    for step in range(MAX_STEPS):
        curr_state = env.state
        action = agent.select_action(curr_state)

        new_state, reward, _, _ = env.play_action(action)
        if step != 0:
            agent.update_q_table(old_state=curr_state,action=action,reward=reward,new_state=new_state)

        episode_reward += reward

        if env.done:
            break

    rewards.append(episode_reward)
    if episode_reward > max_reward:
        max_reward = episode_reward

env.close()
