# TODO: make vm branch the master branch (all changes are on there now)

# TODO: add in-depth README for atari pong

# TODO: train agent on other games? -> Breakout

from agent import Agent
from env import Env
from ddqn import DDQN
from frstack import Frstack
import utils

from collections import deque
import torch
from PIL import Image

# for agent eps bounds
dum_val = 1
NUM_FRAMES = 4
NUM_TEST_EPISODES = 1
POLICY_NET_PATH = "res/trained_policy_net.pt"
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# set up environment tools
USE_GPU = torch.cuda.is_available()
mod_action_space = [2,3,4,5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Env(device)
agent = Agent(
        eps=dum_val, eps_min=dum_val, eps_max=dum_val, eps_decay=dum_val, num_actions=len(mod_action_space), device=device
        )
agent.turn_eps_off()
stack = Frstack(initial_frame=env.state)

# create policy net and load saved weights
policy_net = DDQN(NUM_FRAMES, len(mod_action_space))
if USE_GPU:
    policy_net.cuda()

def test():
    policy_net.load_state_dict(torch.load(POLICY_NET_PATH))
    policy_net.eval()

    print("testing...")
    all_rewards = []
    all_images = []

    for episode in range(NUM_TEST_EPISODES):
        env.reset()
        episode_reward = 0
        stack.push(env.state, True)
        curr_state = stack.get_stack()
        next_state = None

        while not env.done:
            action = agent.select_action(state=curr_state, policy_net=policy_net)
            next_state, reward, done, _ = env.play_action(mod_action_space[action])
            stack.push(next_state, False)
            curr_state = stack.get_stack()
            episode_reward += reward

            if episode == NUM_TEST_EPISODES - 1:
                im = Image.fromarray(env.rgb_state)
                all_images.append(im)

        print("Current episode reward:", episode_reward)
        all_rewards.append(episode_reward)

    utils.plot_testing(all_rewards)

    # note: using pillow to save GIFs caused some arbitrary blacking out of pixels so
    # re-edited the gif with ezgif.com
    all_images[0].save('res/pong.gif', save_all=True, append_images=all_images[1:])

    print('done testing!')

if __name__ == '__main__':
    test()
