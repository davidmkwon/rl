import matplotlib.pyplot as plt
import random
import math
from collections import deque

def plot_training(rewards, moving_avg_period=100):
    plt.figure(1)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title("Training")

    episodes = [i for i in range(1, len(rewards) + 1)]
    num_1 = plt.plot(episodes, rewards, 'b', label="Episode Reward")

    avg_queue = deque(maxlen=100)
    avg_rewards = deque()
    
    for i in range(99):
        avg_rewards.append(0)
        avg_queue.append(rewards[i])
    for i in range(100, len(rewards)):
        avg_queue.append(rewards[i])
        assert len(avg_queue) == 100
        avg = sum(avg_queue) / 100
        avg_rewards.append(avg)

    avg_rewards.append(avg_rewards[-1])
    num_2 = plt.plot(episodes, avg_rewards, 'm', label="100 Episode Moving Average")

    plt.savefig("res/training.png", bbox_inches="tight")
    plt.legend()
    # omit plt.show() line because running on VM

def plot_testing(rewards):
    plt.figure(2)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title("Testing")

    episodes = [i for i in range(1, len(rewards) + 1)]
    num_1 = plt.plot(episodes, rewards, 'm', label="Episode Reward")

    avg_reward = sum(rewards) / len(rewards)
    avg_rewards = [avg_reward for i in range(len(rewards))]
    num_2 = plt.plot(episodes, avg_rewards,'--', label="Average Reward")

    plt.savefig("res/testing.png", bbox_inches='tight')
    plt.legend()
    # omit plt.show() line because running on VM
    plt.show()
