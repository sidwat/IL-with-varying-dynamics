import gym
import gym.wrappers
import reacher
import driving
import time
from gym import make 
import numpy as np
import argparse
import pickle as pkl


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='reacher_custom-action1-v0')
parser.add_argument('--optimal', action='store_true',
                    help='use the optimal demonstrations of the demo or not')
args = parser.parse_args()
if 'action1' in args.env:
    DEMO_PATH = '/home/smart/Learn-Imperfect-Varying-Dynamics/demos/reacher_custom-action1-v0_1.0.pkl' if args.optimal else '/home/smart/Learn-Imperfect-Varying-Dynamics/demos/reacher_custom-action1-v0_0.0.pkl'
elif 'action2' in args.env:
    DEMO_PATH = '/home/smart/Learn-Imperfect-Varying-Dynamics/demos/reacher_custom-action2-v0_1.0.pkl' if args.optimal else '/home/smart/Learn-Imperfect-Varying-Dynamics/demos/reacher_custom-action2-v0_0.0.pkl'
else:
    print('Invalid environment')
with open(DEMO_PATH, 'rb') as f:
    data = pkl.load(f)

env = make(args.env, render_mode = 'human')
print(env.observation_space)
print(env.action_space)

env.reset()
for episode in range(10):
    for state in data[episode]['state']:
        env.reset_with_obs(state.squeeze())
        # print(state.squeeze())
        env.render()
        time.sleep(0.05)
    print('episode : {} done'.format(episode))
env.close()

