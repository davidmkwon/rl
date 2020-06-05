from agent import Agent

import gym
import random
import pandas as pd
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

new_state, _, _, _ = env.step(0)

agent = Agent(0,0,0,0,0,0)
agent.create_state(new_state)
