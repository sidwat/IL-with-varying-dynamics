import torch
from models import Policy, Value # Import your model module here
import gym
import time
import argparse
import numpy as np
import reacher

parser = argparse.ArgumentParser(description='Test the model')
parser.add_argument('--num-episodes', type=int, default=100)
parser.add_argument('--seed', type=int, default=1551)
args = parser.parse_args()

environment = 'reacher_custom-action1-v0'
snapshot_file_best = 'snapshot/reacher_snapshot/action1-action2-action1-feas_confbest.tar'
snapshot_file = 'snapshot/reacher_snapshot/action1-action2-action1-feas_conf.tar'
env1 = gym.make(environment)

def select_action(state, policy_net):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_mean, _, action_std = policy_net(state)
    action = torch.normal(action_mean, action_std)
    return action

num_inputs = env1.observation_space.shape[0]
num_actions = env1.action_space.shape[0]
print("num inputs ", num_inputs)
print("num actions ", num_actions)

hidden_dim = 100
policy_net = Policy(num_inputs, num_actions, hidden_dim).float()
value_net = Value(num_inputs, hidden_dim).float()
# snapshot_file = args.snapshot

# Step 2: Load the saved model parameters
checkpoint_best = torch.load(snapshot_file_best, map_location=torch.device('cpu'))
checkpoint = torch.load(snapshot_file, map_location=torch.device('cpu'))

# Step 3: Set the model parameters to the loaded state dictionary
policy_net.load_state_dict(checkpoint_best['policy_net'])
value_net.load_state_dict(checkpoint_best['value_net'])

env1.reset()
policy_net.eval()  # Set the model to evaluation mode
total_reward = 0
with torch.no_grad():
    for _ in range(args.num_episodes):
        state = env1.reset(seed = args.seed)[0]
        accumulator = 0
        for i in range(10000):
            # env1.render()
            action = select_action(state, policy_net=policy_net).detach().numpy().squeeze()
            # action = policy_net(torch.tensor(state).float()).detach().numpy()
            # print(action.shape)
            next_state, reward, done, _, info= env1.step(action)
            if done :
                break
            # time.sleep(0.1)
            accumulator += reward
            state = next_state
        total_reward += accumulator
print("best model average reward ", total_reward/args.num_episodes)

policy_net.load_state_dict(checkpoint['policy_net'])
value_net.load_state_dict(checkpoint['value_net'])

env1.reset()
policy_net.eval()  # Set the model to evaluation mode
total_reward = 0
with torch.no_grad():
    for _ in range(args.num_episodes):
        state = env1.reset(seed = args.seed)[0]
        accumulator = 0
        for i in range(10000):
            # print(state)
            # env1.render()
            action = select_action(state, policy_net=policy_net).detach().numpy().squeeze()
            # action = policy_net(torch.tensor(state).float()).detach().numpy()
            next_state, reward, done, _, info= env1.step(action)
            if done :
                break
            # time.sleep(0.1)
            accumulator += reward
            state = next_state
        total_reward += accumulator
print("other model average reward", total_reward/args.num_episodes)

