# atariGamesRL

# **Reinforcement Learning Agent for Space Invaders**

This project implements a Deep Q-Network (DQN) agent that learns to play Space Invaders using ALE. The agent interacts with the game environment, learns optimal actions through trial and error, and can be trained and evaluated using this codebase.

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Usage](#usage)
  - [Training the Agent](#training-the-agent)
  - [Evaluating the Agent](#evaluating-the-agent)
- [Code Structure](#code-structure)
- [State Representations](#state-representations)
- [Exploration Strategies](#exploration-strategies)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [References](#references)

---

## **Project Overview**

The goal of this project is to develop a reinforcement learning agent capable of playing Space Invaders on the Atari 2600 console. The agent uses a Deep Q-Network (DQN), a neural network-based RL algorithm, to learn the optimal policy for maximizing the game score.

Key features:

- **Function Approximation:** Utilizes a neural network to approximate the Q-values, enabling the handling of high-dimensional and continuous state spaces.
- **State Representations:** Implements multiple state representations to explore different generalization approaches.
- **Exploration Strategies:** Incorporates various exploration methods, including epsilon-greedy and count-based exploration, to balance exploration and exploitation.
- **Optimistic Initialization:** Initializes the network with optimistic priors to encourage exploration of unvisited state-action pairs.
- **Training and Evaluation Modes:** The script supports separate modes for training the agent and evaluating its performance.

---

## **Usage**

The script can be run in two modes:

1. **Training Mode:** The agent learns to play the game through interactions.
2. **Evaluation Mode:** The trained agent plays the game, and you can observe its performance.

### **Training the Agent**

To train the agent, run the script with the `--mode train` argument:

python3 main.py --mode train --episodes <number_of_episodes> --approach <state_representation> --exploration <exploration_strategy> --seed <random_seed>

- **Example:**

```python3 main.py --mode train --episodes 1000 --approach 1 --exploration epsilon --seed 123```

- **Parameters:**
  - `--mode train`: Specifies that the agent should be trained.
  - `--episodes`: (Optional) Number of training episodes (default is 1000).
  - `--approach`: (Optional) State representation approach (1, 2, or 3).
  - `--exploration`: (Optional) Exploration strategy (`epsilon` or `count`).
  - `--seed`: (Optional) Random seed for reproducibility.

**Notes:**

- During training, the game screen and sound are disabled to speed up the process.
- The trained model will be saved to `dqn_model.pth` after training completes.

### **Evaluating the Agent**

To evaluate the trained agent and watch it play the game, run the script with the `--mode evaluate` argument:

python3 main.py --mode evaluate --episodes <number_of_episodes> --approach <state_representation> --seed <random_seed>

- **Example:**

  ```python3 main.py --mode evaluate --episodes 10 --approach 1 --seed 123```

- **Parameters:**
  - `--mode evaluate`: Specifies that the agent should be evaluated.
  - `--episodes`: (Optional) Number of evaluation episodes (default is 10).
  - `--approach`: (Optional) State representation approach used during training.
  - `--seed`: (Optional) Random seed for reproducibility.

**Notes:**

- During evaluation, the game screen and sound are enabled.
- The agent uses the learned model (`dqn_model.pth`) to make decisions without further learning.
- Ensure that `dqn_model.pth` exists in the directory (generated after training).

---

## **Code Structure**

- **`main.py`**: The main script containing the agent implementation.
- **Key Components:**
  - **Classes:**
    - `DQN`: Defines the neural network architecture for approximating Q-values.
    - `ReplayMemory`: Implements experience replay buffer for storing transitions.
  - **Functions:**
    - `get_state(ram_state, approach)`: Extracts and processes the RAM state based on the selected state representation approach.
    - `state_to_key(state)`: Converts a state tensor to a hashable key for counting state-action visits.
    - `train_agent(...)`: Contains the training loop where the agent learns from interactions.
    - `evaluate_agent(...)`: Runs the agent in evaluation mode without learning updates.

---

## **State Representations**

The agent supports multiple state representations to explore different generalization approaches:

1. **Approach 1: Full RAM State**

   - Uses the entire 128-byte RAM state as input.
   - Captures all available information from the game.

2. **Approach 2: Selected RAM Variables**

   - Uses specific RAM addresses corresponding to key game variables:
     - Player's X position (`ram_state[28]`)
     - Number of invaders left (`ram_state[17]`)
     - Enemies' average X position (`ram_state[26]`)
     - Enemies' average Y position (`ram_state[24]`)
   - Focuses on critical game elements affecting the agent's decisions.

3. **Approach 3: Processed Features**

   - Uses the normalized and processed features derived from selected RAM variables.
   - Scales variables to a range between 0 and 1 for better learning efficiency.

**Example of State Extraction:**

```python
def get_state(ram_state, approach):
    if approach == 1:
        state = torch.tensor(ram_state, dtype=torch.float32)
    elif approach == 2:
        player_x = ram_state[28]
        invaders_left = ram_state[17]
        enemies_x = ram_state[26]
        enemies_y = ram_state[24]
        state = torch.tensor([player_x, invaders_left, enemies_x, enemies_y], dtype=torch.float32)
    elif approach == 3:
        player_x = ram_state[28] / 255.0
        invaders_left = ram_state[17] / 36.0
        enemies_x = ram_state[26] / 255.0
        enemies_y = ram_state[24] / 255.0
        state = torch.tensor([player_x, invaders_left, enemies_x, enemies_y], dtype=torch.float32)
    return state.unsqueeze(0)
```

---

## **Exploration Strategies**

Two exploration strategies are implemented to balance exploration and exploitation:

1. **Epsilon-Greedy Exploration**

   - The agent selects a random action with probability epsilon.
   - Epsilon decays over time to reduce exploration as the agent learns.

2. **Count-Based Exploration**

   - Incorporates visit counts \( N(s,a) \) for state-action pairs.
   - Adds an exploration bonus \( c / \sqrt{N(s,a)} \) to the Q-values during action selection.
   - Encourages the agent to explore less-visited state-action pairs.

**Example of the Count-Based Exploration:**

```python
if exploration == 'count':
    with torch.no_grad():
        q_values = policy_net(state)
        q_values = q_values.squeeze()
        state_key = state_to_key(state)
        q_values_with_bonus = []
        for action_index in range(action_size):
            sa_key = (state_key, action_index)
            N_sa_count = N_sa.get(sa_key, 1)
            exploration_bonus = c / np.sqrt(N_sa_count)
            q_sa = q_values[action_index].item() + exploration_bonus
            q_values_with_bonus.append(q_sa)
        action_index = np.argmax(q_values_with_bonus)
    sa_key = (state_key, action_index)
    N_sa[sa_key] = N_sa.get(sa_key, 0) + 1
```

---

## **Hyperparameters**

The following hyperparameters are used in the DQN algorithm:

- **Network Parameters:**
  - Learning Rate (`LEARNING_RATE`): `1e-5`
  - Batch Size (`BATCH_SIZE`): `64`
  - Replay Memory Size (`MEMORY_SIZE`): `100000`
  - Target Network Update Frequency (`TARGET_UPDATE`): `10000`

- **Discount Factor (`GAMMA`):** `0.99`

- **Exploration Parameters:**
  - Epsilon-Greedy:
    - Initial Epsilon (`EPSILON_START`): `1.0`
    - Final Epsilon (`EPSILON_END`): `0.1`
    - Epsilon Decay Rate (`DECAY_RATE`): `0.995`
  - Count-Based Exploration:
    - Exploration Constant (`c`): `10` (Modifiable)

- **Number of Episodes:**
  - Default for training: `100`
  - Default for evaluation: `10`

---

## **Heursitic Mechnaisms**

- **Function Approximation Specialization:**
  - The neural network approximates Q-values, focusing on important state-action pairs based on the provided state representations.
  - Optimistic initialization of network biases encourages exploration of unvisited states.

- **Multiple Generalization Approaches:**
  - Three different state representations are implemented to study their effects on the agent's performance and generalization capabilities.

- **Exploration Functions:**
  - Two exploration strategies (epsilon-greedy and count-based exploration) are implemented.
  - The impact of each strategy on the agent's behavior is analyzed through experiments.

- **State-Action Distance Metric:**
  - A function `state_to_key` is used to discretize continuous states into hashable keys, enabling the computation of visit counts for state-action pairs.

- **Analysis over Multiple Seeds:**
  - The script allows setting a random seed to test the agent's performance consistency across different runs.

---

## **Results**

After training, the agent's performance is evaluated by observing:

- **Total Reward per Episode:** Printed during training and evaluation.
- **Agent Behavior:** During evaluation, watch how the agent moves, shoots, and avoids enemy fire.
- **Learning Progress:** Analyze learning curves and total rewards to assess improvement over training episodes.
- **Impact of State Representations and Exploration Strategies:**
  - Compare results from different state representations and exploration methods.
  - Discuss how these factors influence the agent's ability to learn effective policies.
