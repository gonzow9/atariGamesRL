# atariGamesRL

Reinforcement learning experiments for Space Invaders using a Deep Q-Network on ALE RAM observations.

## What this project does
- DQN agent on Atari 2600 Space Invaders with RAM-only state (three representations: full RAM, selected bytes, normalised features).
- Two exploration strategies: epsilon-greedy with optimistic bias init, and a simple count-based bonus on state-action visits.
- Saves training artifacts (rewards, losses, epsilon schedule, action counts) plus a CLI plotting script.
- Reproducible runs via CLI arguments for ROM path, seed, state representation, exploration strategy, and episode counts.

## To start
1) Install dependencies
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) Get the Space Invaders ROM and its path.
3) Train (screen/sound are disabled during training for speed):
```
python main.py --mode train --rom /path/to/SpaceInvaders.bin --episodes 1000 --approach 1 --exploration epsilon --seed 123
```
4) Evaluate the trained agent (screen/sound enabled):
```
python main.py --mode evaluate --rom /path/to/SpaceInvaders.bin --episodes 10 --approach 1 --seed 123
```

## Key arguments
- `--mode`: `train` or `evaluate`.
- `--rom`: Path to Space Invaders ROM (must have).
- `--episodes`: Number of episodes (train default 1000, eval default 10).
- `--approach`: State representation.
  - `1`: Full 128-byte RAM.
  - `2`: Selected RAM bytes (player x, invaders left, enemies x/y).
  - `3`: Normalized version of the selected bytes.
- `--exploration`: `epsilon` (decaying epsilon-greedy) or `count` (visit-count bonus).
- `--seed`: Random seed (applied to Python, NumPy, PyTorch, and ALE).

## Training outputs
- `dqn_model.pth`: Trained network weights.
- `total_rewards.pkl`, `loss_values.pkl`, `epsilon_values.pkl`, `action_counts.pkl`: Training logs.
- Generate plots (saved to `plots/` by default):
```
python plots.py --log-dir . --out-dir plots
```
Add `--show` to view interactively.

## Implementation notes and limitations
- Uses a compact MLP on RAM; no frame stacking or image-based features.
- Target network sync every `TARGET_UPDATE` steps; replay buffer size and learning rate set for stability but not exhaustively tuned.
- Count-based exploration uses a fixed bonus `c = 50`; adjust to trade off exploration vs. exploitation.
- Optimistic bias initialisation accelerates early exploration; epsilon decays across the full training horizon rather than resetting each episode.
- No GPU-specific code is required; PyTorch will use CPU by default.

## Repo layout
- `main.py`: Agent, training/eval loops, state representations, exploration strategies, logging.
- `plots.py`: CLI to generate learning, loss, epsilon, and action-distribution plots.
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT license.

## References
- Mnih et al., 2015. Human-level control through deep reinforcement learning.
- Bellemare et al., 2013. The Arcade Learning Environment.

## Math appendix
See `theory.md` for my notes on the Bellman equations, loss definitions, and exploration bonus used.
