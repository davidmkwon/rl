import matplotlib.pyplot as plt
import pickle
import random
import math
from collections import deque

x = [1,2,3,8]
y = [4,5,6,7]

def plot_training(rewards, moving_avg_period=100):
    plt.figure(1)
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    episodes = [i for i in range(1, len(rewards) + 1)]
    num_1 = plt.plot(episodes, rewards, 'b', label="Episode Reward")

    avg_queue = deque()
    avg_rewards = deque()
    for i in range(99):
        avg_rewards.append(0)
        avg_queue.append(rewards[i])
    for i in range(100, len(rewards)):
        avg_queue.append(rewards[i])
        avg = sum(avg_queue) / len(avg_queue)
        avg_rewards.append(avg)
        avg_queue.popleft()
    avg_rewards.append(avg_rewards[-1])
    num_2 = plt.plot(episodes, avg_rewards, 'm', label="100 Episode Moving Average")

    plt.savefig("res/training.png", bbox_inches="tight")
    plt.legend()
    plt.show()

def plot_testing(rewards):
    plt.figure(2)
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    episodes = [i for i in range(1, len(rewards) + 1)]
    num_1 = plt.plot(episodes, rewards, 'm', label="Episode Reward")

    avg_reward = sum(rewards) / len(rewards)
    avg_rewards = [avg_reward for i in range(len(rewards))]
    num_2 = plt.plot(episodes, avg_rewards,'--', label="Average Reward")

    plt.savefig("res/testing.png", bbox_inches='tight')
    plt.legend()
    plt.show()

def plot_epsilon(epsilons):
    plt.figure(3)
    episodes = [i for i in range(1, len(epsilons) + 1)]
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.plot(episodes, epsilons, 'c')
    plt.savefig("res/epsilon.png")
    plt.show()

def save_q_table(q_table):
    with open('res/q_table.pickle', 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

def dummy(rewards):
    plt.figure(5)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title("Testing Period")

    episodes = [i for i in range(1, 101)]
    num_1 = plt.plot(episodes, rewards, 'm', label="Episode Reward")

    avg_reward = sum(rewards) / 100
    avg_rewards = [avg_reward for i in range(100)]
    num_2 = plt.plot(episodes, avg_rewards,'y', label="Average Reward")

    plt.savefig("res/dummy.png", bbox_inches='tight')
    plt.legend()
    plt.show()
