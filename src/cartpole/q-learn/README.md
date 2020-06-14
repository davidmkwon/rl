# q-learning cartpole

## Overview
A standard q-learning approach to solving OpenAI Gym's Cartpole environment. Because there are infinite potential states, we convert cart position, cart velocity, pole angle, and pole velocity measurements into a 4 digit number through discretized buckets (see `env.py` for exact sizes). We then map these 4 digit numbers to the two potential actions (left or right) in the q-table.

Currently, testing and training reside in different python files, so `pickle` is used for storing the q_table dictionary externally.

Looking at the performance graphs, the agent performs significantly better during testing as expected (epsilon is set to 0, so the agent always exploits). Interestingly, when the agent is trained for longer periods (~1000-1500 episodes) it will occasionally peform worse when testing. This could be due to the nature of using a q-table in this environment—the longer the agent is trained, the larger the q-table becomes, and the more likely untrained paths are taken. This is however all speculation.

Run:
```bash
python train.py
python test.py
```

## Hyperparameters
- ALPHA = 0.1
- GAMMA = 0.9
- NUM_EPISODES = 500
- EPSILON DECAY = 0.001

## Performance
<img src="res/training-pic.jpg" alt="drawing" width="550"/>
<img src="res/testing-pic.jpg" alt="drawing" width="550"/>

## Libraries
- pickle
- matplotlib
- pandas
- numpy
- gym