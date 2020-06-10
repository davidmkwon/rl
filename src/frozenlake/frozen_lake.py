import gym
import numpy as np
import random
import time
from IPython.display import clear_output

# create environment
env = gym.make("FrozenLake-v0")

# creating q-table
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))

# q-learning setup
num_episodes = 11000
max_steps_per_episode = 100

alpha = 0.1 # learning rate
gamma = 0.99 # discount rate

epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
epsilon_decay_rate = 0.001

rewards_all_episodes = []

print("training...")

# train over episodes
for episode in range(num_episodes):
    state = env.reset()
    done = False
    reward_current_episode = 0

    for step in range(max_steps_per_episode):

        # explore vs exploit / epsilon-greedy policy
        epsilon_threshold = random.random()
        if epsilon_threshold > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        # execute chosen action
        new_state, reward, done, _ = env.step(action)

        # update q-value in q-table
        q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * 
                np.max(q_table[new_state]))

        state = new_state
        reward_current_episode += reward
        
        if done:
            break
    
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
    rewards_all_episodes.append(reward_current_episode)

print("done!")

# get data on average rewards
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("AVERAGES:\n")
for each in rewards_per_thousand_episodes:
    print("{} : {}".format(count, str(sum(each) / 1000)))
    count += 1000
print("q_table:\n", q_table)

# visualize trained agent
for episode in range(3):
    state = env.reset()
    done = False
    print("episode {}:".format(episode + 1))

    for step in range(max_steps_per_episode):
        env.render()
        time.sleep(3)

        action = np.argmax(q_table[state])
        new_state, reward, done, _ = env.step(action)
        state = new_state

        if done:
            env.render()
            if reward == 1:
                print("goal reached!")
            else:
                print("fell in hole :(")
            time.sleep(3)
            break

env.close()
