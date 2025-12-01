import numpy as np
import random
from ale_py import ALEInterface
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import argparse
import matplotlib.pyplot as plt
import pickle

# Printed Actions and got
# Actions: [<Action.NOOP: 0>, <Action.FIRE: 1>, <Action.UP: 2>, <Action.RIGHT: 3>, <Action.LEFT: 4>, 
#                 <Action.DOWN: 5>, <Action.UPRIGHT: 6>, <Action.UPLEFT: 7>, <Action.DOWNRIGHT: 8>, 
#                 <Action.DOWNLEFT: 9>, <Action.UPFIRE: 10>, <Action.RIGHTFIRE: 11>, <Action.LEFTFIRE: 12>, 
#                 <Action.DOWNFIRE: 13>, <Action.UPRIGHTFIRE: 14>, <Action.UPLEFTFIRE: 15>, 
#                 <Action.DOWNRIGHTFIRE: 16>, <Action.DOWNLEFTFIRE: 17>]

# For space invaders legal actions are:
# NOOP
# FIRE
# RIGHT
# LEFT
# RIGHTFIRE
# LEFTFIRE

VALID_ACTIONS = [0, 1, 3, 4, 11, 12]
ACTION_SIZE = len(VALID_ACTIONS)

# Hyperparameters
BATCH_SIZE = 64   # number of experience samples drawn from the replay memory
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1
DECAY_RATE = 0.995
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000 # max num transitions that can be stored in the replay memory.
LEARNING_RATE = 1e-6 # step size in each iteration when moving toward min loss. 

# data collection
# action count dictionary
action_counts = {action_code: 0 for action_code in VALID_ACTIONS}
total_rewards = []
loss_values = []
epsilon_values = []

# Define the DQN neural net
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Optimistic initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 10.0)  # Set biases to higher value
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        return self.layers(x)

# Experience Replay Buffer to make network updates stable
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def get_state(ram_state, approach):
    if approach == 1:
        # Full RAM state convert to tensor for network
        state = torch.tensor(ram_state, dtype=torch.float32)
    elif approach == 2:
        # Selected RAM variables
        player_x = ram_state[28]
        invaders_left = ram_state[17]
        enemies_x = ram_state[26]
        enemies_y = ram_state[24]
        state = torch.tensor([player_x, invaders_left, enemies_x, enemies_y], dtype=torch.float32)
    elif approach == 3:
        # Processed features
        player_x = ram_state[28] / 255
        invaders_left = ram_state[17] / 36
        enemies_x = ram_state[26] / 255
        enemies_y = ram_state[24] / 255
        state = torch.tensor([player_x, invaders_left, enemies_x, enemies_y], dtype=torch.float32)
    else:
        raise ValueError("Invalid approach")
    return state.unsqueeze(0)  # Adds batch dimension

def state_to_key(state):
    # Convert the state tensor to a tuple of integers by rounding
    return tuple(state.squeeze().numpy().round(decimals=0).astype(int))

def train_agent(ale, policy_net, target_net, optimizer, memory, episodes, approach, exploration):
    steps_done = 0
    N_sa = {}  # initialise the state-action visit counts
    c = 50  # exploration constant for count-based exploration
    epsilon = EPSILON_START  # decays across episodes instead of resetting each time

    for episode in range(episodes):
        ale.reset_game()
        ram_state = np.array(ale.getRAM())
        state = get_state(ram_state, approach)
        total_reward = 0

        while not ale.game_over():
            if exploration == 'epsilon':
                # Epsilon-greedy exploration
                epsilon = max(EPSILON_END, epsilon * DECAY_RATE)
                if random.random() < epsilon:
                    action_index = random.randint(0, ACTION_SIZE - 1)
                else:
                    with torch.no_grad():
                        q_values = policy_net(state)
                        action_index = q_values.argmax().item()
                action_code = VALID_ACTIONS[action_index]
                steps_done += 1
                
                # Record data
                action_counts[action_code] += 1
                epsilon_values.append(epsilon)
                
            elif exploration == 'count':
                with torch.no_grad():
                    q_values = policy_net(state)
                    q_values = q_values.squeeze()
                    # Generate state key
                    state_key = state_to_key(state)
                    q_values_with_bonus = []
                    for idx, action_code in enumerate(VALID_ACTIONS):
                        # Get N(s,a)
                        sa_key = (state_key, action_code)
                        N_sa_count = N_sa.get(sa_key, 1)  # Default to 1 to prevent division by zero
                        exploration_bonus = c / np.sqrt(N_sa_count)
                        # Compute Q(s,a) + exploration bonus
                        q_sa = q_values[idx].item() + exploration_bonus
                        q_values_with_bonus.append(q_sa)
                    # Select the action that maximizes Q(s,a) + exploration bonus
                    best_action_idx = np.argmax(q_values_with_bonus)
                    action_code = VALID_ACTIONS[best_action_idx]
                    # Update N(s,a)
                    sa_key = (state_key, action_code)
                    N_sa[sa_key] = N_sa.get(sa_key, 0) + 1
                    steps_done += 1
                    action_index = best_action_idx  # For consistency in memory storage
                    
                    # record action data
                    action_counts[action_code] += 1
            else:
                raise ValueError("Invalid exploration strategy. Choose 'epsilon' or 'count'.")

            # Take action
            reward = ale.act(action_code)
            total_reward += reward

            # Read new state
            next_ram_state = np.array(ale.getRAM())
            next_state = get_state(next_ram_state, approach)
            done = ale.game_over()

            # Stores transition in memory
            memory.push(state, action_index, reward, next_state, done)

            # Move to next state
            state = next_state

            # Perform the optimizations
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.cat(batch_state)
                batch_action = torch.tensor(batch_action).unsqueeze(1)
                batch_reward = torch.tensor(batch_reward)
                batch_next_state = torch.cat(batch_next_state)
                batch_done = torch.tensor(batch_done, dtype=torch.float32)

                # Compute Q(s_t, a)
                state_action_values = policy_net(batch_state).gather(1, batch_action)

                # Compute U(s_{t+1})
                next_state_values = target_net(batch_next_state).max(1)[0].detach()
                next_state_values = next_state_values * (1 - batch_done)

                # Compute expected Q values
                expected_state_action_values = batch_reward + (GAMMA * next_state_values)

                # Compute loss
                loss = nn.functional.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # record loss
                loss_values.append(loss.item())

            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
    
    # save data
    # end of training save data
    with open('total_rewards.pkl', 'wb') as f:
        pickle.dump(total_rewards, f)

    with open('loss_values.pkl', 'wb') as f:
        pickle.dump(loss_values, f)
        
    with open('action_counts.pkl', 'wb') as f:
        pickle.dump(action_counts, f)

    with open('epsilon_values.pkl', 'wb') as f:
        pickle.dump(epsilon_values, f)

    return total_rewards


if __name__ == "__main__":
    # Set up argument parsor
    parser = argparse.ArgumentParser(description='Train or evaluate the Space Invaders agent.')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode to run the script in: train or evaluate.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run.')
    parser.add_argument('--approach', type=int, choices=[1,2,3], default=1, help='State representation approach to use (1, 2, or 3).')
    parser.add_argument('--exploration', choices=['epsilon', 'count'], default='epsilon', help='Exploration strategy to use.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--rom', type=str, default='Space Invaders.bin', help='Path to a legal Space Invaders ROM file (not included).')
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize ALE
    ale = ALEInterface()
    ale.setInt('random_seed', args.seed)
    if not os.path.exists(args.rom):
        raise FileNotFoundError(f"ROM not found at {args.rom}. Please supply a legal copy of the game ROM.")
    ale.loadROM(args.rom)
    # Disable visuals during training for faster learning; re-enable in eval.
    ale.setBool('display_screen', False)
    ale.setBool('sound', False)
    
    if args.mode == 'train':
        # Training mode
        ale.setBool('display_screen', False)
        ale.setBool('sound', False)

        # Determine input size based on approach
        if args.approach == 1:
            input_size = 128
        elif args.approach in [2, 3]:
            input_size = 4
        else:
            raise ValueError("Invalid approach")

        policy_net = DQN(input_size, ACTION_SIZE)
        target_net = DQN(input_size, ACTION_SIZE)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        # package to implement optimization algos
        optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
        memory = ReplayMemory(MEMORY_SIZE)

        total_rewards = train_agent(ale, policy_net, target_net, optimizer, memory, args.episodes, args.approach, args.exploration)

        # Save the model
        torch.save(policy_net.state_dict(), 'dqn_model.pth')
        print("Model saved to 'dqn_model.pth'")

    elif args.mode == 'evaluate':
        # Evaluation mode
        ale.setBool('display_screen', True)
        ale.setBool('sound', True)

        # Determine input size based on approach
        if args.approach == 1:
            input_size = 128
        elif args.approach in [2, 3]:
            input_size = 4
        else:
            raise ValueError("Invalid approach")

        policy_net = DQN(input_size, ACTION_SIZE)
        if os.path.exists('dqn_model.pth'):
            policy_net.load_state_dict(torch.load('dqn_model.pth'))
            policy_net.eval()
            print("Loaded model from 'dqn_model.pth'")
        else:
            print("No trained model found. Please train the agent first.")
            exit()

        for episode in range(args.episodes):
            ale.loadROM(args.rom)
            ale.setBool('display_screen', True)
            ale.setBool('sound', True)
            ale.reset_game()
            ram_state = np.array(ale.getRAM())
            state = get_state(ram_state, args.approach)
            total_reward = 0

            while not ale.game_over():
                with torch.no_grad():
                    q_values = policy_net(state)
                    action_index = q_values.argmax().item()
                action = VALID_ACTIONS[action_index]

                # Take action
                reward = ale.act(action)
                total_reward += reward

                # Observe new state
                next_ram_state = np.array(ale.getRAM())
                next_state = get_state(next_ram_state, args.approach)

                # Update state
                state = next_state

            print(f"Evaluate Episode {episode + 1}/{args.episodes}, Total Reward: {total_reward}")
