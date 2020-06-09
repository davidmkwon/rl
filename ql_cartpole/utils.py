import matplotlib.pyplot as plt
import json
import random

x = [1,2,3,8]
y = [4,5,6,7]

def plot():
    plt.figure(1)
    plt.plot(x)
    plt.plot(y)
    plt.title("Training Period")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.savefig('res/training.png')
    plt.show()

def plot_training(values, moving_avg_period=100):
    plt.figure(2)
    plt.plot(x,y,'m')
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title("Testing Period")
    plt.savefig("res/testing.png")
    plt.show()

def plot_testing(rewards):
    plt.figure(3)
    episodes = [i for i in range(1, 101)]
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title("Testing Period")
    plt.plot(episodes, rewards, 'm')
    avg_reward = sum(rewards) / 100
    avg_rewards = [avg_reward for i in range(100)]
    plt.plot(episodes, avg_rewards,'--')
    plt.savefig("res/testing.png")
    plt.show()

def plot_epsilon(epsilons):
    plt.figure(4)
    episodes = [i for i in range(1, len(epsilons) + 1)]
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.plot(episodes, epsilons, 'c')
    plt.savefig("res/epsilon.png")
    plt.show()

def save_q_table(q_table):
    q_table_file = open("res/q_table.json", "w")
    json.dump(q_table, q_table_file)
    q_table_file.close()

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

rewards = [random.random() for i in range(100)]
dummy(rewards)
