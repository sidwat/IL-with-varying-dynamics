import torch
from models import CloningPolicy, Value  # Import your model module here
import gym
import driving
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Test the model')
parser.add_argument('--num-episodes', type=int, default=100)
parser.add_argument('--seed', type=int, default=1001)
parser.add_argument('--use-sleep', action='store_true')
parser.add_argument('--render-best', action='store_true'
                    , help='render the best model')
parser.add_argument('--render-other', action='store_true'
                    , help='render the other model')

args = parser.parse_args()

environment = 'ContinuousFastRandom-v0'
snapshot_file_best = 'snapshot/driving_snapshot/fast_slow_fast-feas_confbest.tar'
snapshot_file = 'snapshot/driving_snapshot/fast_slow_fast-feas_conf.tar'
env1 = gym.make(environment)

num_inputs = env1.observation_space.shape[0]
num_actions = env1.action_space.shape[0]
hidden_dim = 100
policy_net = CloningPolicy(num_inputs, num_actions, hidden_dim).float()
value_net = Value(num_inputs, hidden_dim)
# snapshot_file = args.snapshot

# Step 2: Load the saved model parameters
checkpoint_best = torch.load(snapshot_file_best, map_location=torch.device('cpu'))
checkpoint = torch.load(snapshot_file, map_location=torch.device('cpu'))

# Step 3: Set the model parameters to the loaded state dictionary
policy_net.load_state_dict(checkpoint_best['policy_net'])
value_net.load_state_dict(checkpoint_best['value_net'])

policy_net.eval()  # Set the model to evaluation mode
best_reward_list = []
with torch.no_grad():
    for i in range(args.num_episodes):
        state = env1.reset(seed = (i+1)*args.seed)
        accumulator = 0
        while True:
            if args.render_best:
                env1.render()
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = policy_net(state)
            action = action.data[0].numpy()
            action = np.clip(action, env1.action_space.low, env1.action_space.high)
            next_state, reward, done, _, info= env1.step(action)
            accumulator += reward
            if done :
                break
            if args.use_sleep: 
                time.sleep(0.05)
            state = next_state
        best_reward_list.append(accumulator)
best_reward_list = np.array(best_reward_list)
if args.render_best:
    env1.close()
print("best model average reward ", np.mean(best_reward_list))

policy_net.load_state_dict(checkpoint['policy_net'])
value_net.load_state_dict(checkpoint['value_net'])

policy_net.eval()  # Set the model to evaluation mode
total_reward = 0

other_reward_list = []
with torch.no_grad():
    for i in range(args.num_episodes):
        state = env1.reset(seed = (i+1)*args.seed)
        accumulator = 0
        while True:
            # print(state)
            if args.render_other:
                env1.render()
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = policy_net(state)
            action = action.data[0].numpy()
            action = np.clip(action, env1.action_space.low, env1.action_space.high)
            next_state, reward, done, _, info= env1.step(action)
            accumulator += reward
            if done :
                break
            if args.use_sleep:
                time.sleep(0.05)
            state = next_state
        other_reward_list.append(accumulator)
other_reward_list = np.array(other_reward_list)
if args.render_other:
    env1.close()
print("other model average reward", np.mean(other_reward_list))
# print("best rewards ", best_reward_list)
# print("other rewards ", other_reward_list)  

