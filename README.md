# Overview

This repository showcases my reinforcement learning explorations! A good amount of them will be using OpenAI's [Gym](https://github.com/openai/gym), an open source library that contains various environments to explore rl algorithms on. I highly recommend this library as you don't need to worry about setting up a proper environment. I will also occasionally create and use my own environments.

Because of the various (and rapidily expanding) number of reinforcement learning algorithms, I try to implement a variety of them and compare performances. Most explorations will have some kind of final rewards graph(s) in their README that displays the progression of rewards over episodes

currently working on: [atari-pong](https://github.com/davidmkwon/rl/tree/master/src/atari-pong/)

# Structure

All code lies in the [src](https://github.com/davidmkwon/rl/tree/master/src/) directory. In src you will find directories that correspond to environment names. Within the environment directories, there will be one or more approaches to training an agent in the environment.

Each approach to an environment has a README that details the algorithm used, hyperparameters, and graphs. They may also include some general takeaway thoughts I have.

Because of its large number of powerful libararies and general ease of use, all code is written in Python3.7. The necessary libraries needed for running are provided in the code's README.
