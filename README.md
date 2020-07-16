# Overview

<p align="center">
    <img src="src/atari-pong/res/pong.gif" alt="pong gameplay"/>
</p>

This repository showcases my reinforcement learning explorations! A good amount of them will be using OpenAI's [gym](https://github.com/openai/gym), an open source library that contains various environments to explore rl algorithms on. I highly recommend this library as you don't need to worry about setting up a proper environment. I will also occasionally create and use my own environments.

Because of the various (and rapidily expanding) number of reinforcement learning algorithms, I try to implement a variety of them and compare performances. Most explorations will have some kind of final rewards graph(s) in their README that displays the progression of rewards over episodes

**currently working on**: [atari-pong](https://github.com/davidmkwon/rl/tree/master/src/atari-pong/)

# Structure

All code lies in the [src](https://github.com/davidmkwon/rl/tree/master/src/) directory. In src you will find directories that correspond to environment names. Within the environment directories, there will be one or more approaches to training an agent in the environment. Current layout  (within [src](https://github.com/davidmkwon/rl/tree/master/src/)):

Each approach to an environment has a README that details the algorithm used, hyperparameters, and graphs. They may also include some general takeaway thoughts I have.

Because of its large number of powerful libararies and general ease of use, all code is written in Python3.7. The necessary libraries needed for running are provided in the code's README.

# Algorithms

Reinforcement Learning encompasses many types and variations of algorithms. So far the following have been implemented:

- [Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Deep Q-learning (DQN) with Fixed Q-Targets](https://arxiv.org/pdf/1312.5602v1.pdf)
- [Double Deep Q-learning (DDQN) with Fixed Q-Targets](https://arxiv.org/pdf/1509.06461.pdf) and [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
- (WIP) [Dueling Deep Q-learning (DDQN)](https://arxiv.org/pdf/1511.06581.pdf) with [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)

# Environments

As mentioned earlier, most of the environments performed on are from OpenAI's gym. Current environment are:

- [Atari Pong](https://github.com/davidmkwon/rl/tree/master/src/atari-pong)
- [Cartpole](https://github.com/davidmkwon/rl/tree/master/src/cartpole)