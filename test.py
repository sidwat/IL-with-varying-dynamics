import gym
import gym.wrappers
import reacher
import driving
import time
from gym import make 
import numpy as np

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

import random
random.seed(10)
print(random.random())