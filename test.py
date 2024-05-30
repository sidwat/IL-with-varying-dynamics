import gym
import gym.wrappers
import reacher
import driving
import time
from gym import make 
import numpy as np
import argparse

# env = make("ContinuousSlowRandom-v0")
# # env = wrappers.SeedWrapper(env, 42)  # Set seed to 42
# # env = wrappers.SeedWrapper(env, 42)  # Set seed to 42

# # # env = gym.make('reacher-action', render_mode='human')
# # env = gym.make('Reacher-v4', render_mode = 'human')
# # env2 = gym.make('reacher_custom-action1-v0', render_mode = 'human')
# # env3 = gym.make('reacher_custom-action2-v0', render_mode = 'human')
# # print("observation space of environment 1 ", env1.observation_space)
# # print("observation space of environment 2 ", env2.observation_space)
# # print("observation space of environment 3 ", env3.observation_space)
# # print("action space of environment 1 ", env1.action_space)
# # print("action space of environment 2 ", env2.action_space)
# # print("action space of environment 3 ", env3.action_space)
# # print("specifications ", env2.spec)
# # env1 = gym.wrappers.SeedWrapper(env1, 42)  # Set seed to 42
# # # env2.reset()
# # # Uncomment following line to save video of our Agent interacting in this environment
# # # This can be used for debugging and studying how our agent is performing
# # # # env = gym.wrappers.Monitor(env, './video/', force = True)
# # env.reset()
# # print(env.action_space)
# # env.render()
# # time.sleep(1)
# # action = [1, -1]
# # observation, reward, done, _, info = env.step(action)
# # print(info)
# # env.render()
# # time.sleep(1)
# # env.seed(0,1)
# env.reset()
# # print(env.action_space)
# for i in range(100):
#    env.render()
#    # print(observation)
#    action = np.array([1])
#    observation, reward, done, _, info = env.step(action)
#    print('action : {} and reward : {}'.format(action, reward))
#    time.sleep(0.1)
# env.close()

# import pickle as pkl
# with open('demos/reacher_custom-action1-v0_1.0.pkl', 'rb') as f:
#     data = pkl.load(f)
# print(data[0])

parser = argparse.ArgumentParser(description='Test the model')
parser.add_argument('--num-episodes', type=int, default=10)
parser.add_argument('--seed', type=int, default=1001)
parser.add_argument('--use-sleep', action='store_true')
parser.add_argument('--env', type=str, default='ContinuousFastRandom-v0')
args = parser.parse_args()

environment = args.env
env1 = gym.make(environment)
# env1.set_goal(0.5, 0.8)
num_inputs = env1.observation_space.shape[0]
num_actions = env1.action_space.shape[0]
print(num_inputs, num_actions)
best_reward_list = []
for i in range(args.num_episodes):
    accumulator = 0
    env1.reset()
    for _ in range(10000):
        env1.render()
        action = env1.action_space.sample()
        next_state, reward, done, _, info= env1.step(action)
        print(next_state, reward, done, info)
        accumulator += reward
        if done :
            break
        if args.use_sleep: 
            time.sleep(0.05)
    print("episode {} done : reward {}".format(i, accumulator))
    best_reward_list.append(accumulator)
best_reward_list = np.array(best_reward_list)
env1.close()

