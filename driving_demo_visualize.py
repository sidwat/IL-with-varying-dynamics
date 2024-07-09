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
parser.add_argument('--env', type=str, default='ContinuousSlowRandom-v0')
parser.add_argument('--optimal', action='store_true',
                    help='use the optimal demonstrations of the demo or not')
args = parser.parse_args()
if 'Fast' in args.env:
    DEMO_PATH = 'demos/driving-fast.pkl' if args.optimal else 'demos/driving-fast-suboptimal.pkl'
elif 'Slow' in args.env:
    DEMO_PATH = 'demos/driving-slow.pkl' if args.optimal else 'demos/driving-slow-suboptimal.pkl'
else:
    print('Invalid environment')
with open(DEMO_PATH, 'rb') as f:
    data = pkl.load(f)

env = make(args.env)
print(env.observation_space)
print(env.action_space)
env.reset()
reward = 0
for episode in range(10):
    reward = 0
    env.reset_with_obs(data[episode]['state'][0].squeeze())
    for step in range(len(data[episode]['action'])):
        env.reset_with_obs(data[episode]['state'][step + 1].squeeze())
        reward += data[episode]['reward'][step]
        env.render()
        time.sleep(0.05)
    print('episode : {} done with reward as {}'.format(episode, reward))
env.close()

