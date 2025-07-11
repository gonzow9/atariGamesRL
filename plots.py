import pickle
import matplotlib.pyplot as plt

# Load total rewards data
with open('total_rewards.pkl', 'rb') as f:
    total_rewards = pickle.load(f)

# Plot total rewards per episode
plt.figure(figsize=(10, 6))
plt.plot(total_rewards, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
plt.show()

# Load loss values data
with open('loss_values.pkl', 'rb') as f:
    loss_values = pickle.load(f)

# Plot loss values over time
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig('loss_over_time.png')
plt.show()

import pickle
import matplotlib.pyplot as plt

# # Load epsilon values data
# with open('epsilon_values.pkl', 'rb') as f:
#     epsilon_values = pickle.load(f)

# # Plot epsilon decay over time
# plt.figure(figsize=(10, 6))
# plt.plot(epsilon_values, label='Epsilon')
# plt.xlabel('Training Step')
# plt.ylabel('Epsilon Value')
# plt.title('Epsilon Decay Over Time')
# plt.legend()
# plt.grid(True)
# plt.savefig('epsilon_decay.png')
# plt.show()

# Load action counts data
with open('action_counts.pkl', 'rb') as f:
    action_counts = pickle.load(f)

# Map action codes to action names
action_names = {
    0: 'NOOP',
    1: 'FIRE',
    3: 'RIGHT',
    4: 'LEFT',
    11: 'RIGHTFIRE',
    12: 'LEFTFIRE'
}

# Prepare data for plotting
sorted_action_codes = sorted(action_counts.keys())
actions = [action_names[code] for code in sorted_action_codes]
counts = [action_counts[code] for code in sorted_action_codes]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(actions, counts, color='skyblue')
plt.xlabel('Action')
plt.ylabel('Count')
plt.title('Action Distribution During Training')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('action_distribution.png')
plt.show()


