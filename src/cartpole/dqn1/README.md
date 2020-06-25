# cartpole with DDQN

This is an approach to solving OpenAI Gym's Cartpole environment. The algorithm used is a Deep Q Learning with fixed q-targets approach, using a second target neural net to find the optimal q values. The Target net recopies the weights of the policy net every 10 episodes.

This approach also does not take in the traditional 4 inputs (cart position, cart velocity, angle position, angle velocity) but rather 40 x 90 x 3 inputs (dimensions of processed rgb image of current environment state)
