# q-learning cartpole

A standard q learning approach to solving OpenAI Gym's Cartpole environment. Because there are infinite potential states, we convert cart position, cart velocity, pole angle, and pole velocity measurements into a 4 digit number through discretized buckets. We then map these 4 digit numbers to the two potential actions (left or right) in the q-table.

## Current Hyperparameters
- ALPHA = 0.1
- GAMMA = 0.9
- NUM_EPISODES = 1000
- EPSILON DECAY = 0.001