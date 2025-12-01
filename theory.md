# RL math notes

## Bellman optimality (action-value)
Q*(s, a) = E_{s'}[ r + gamma * max_{a'} Q*(s', a') ]

For a fixed policy pi, this is a contraction under gamma in (0, 1); the optimality operator shares that property, so repeated application converges in the tabular case.

## One-step TD target used in DQN
- Target network parameters: theta_target
- Online network parameters: theta
- Target y = r + gamma * max_{a'} Q_{theta_target}(s', a')
- Loss: smooth L1 on (y - Q_{theta}(s, a))
- Gradient step: update theta to minimize the TD error (y - Q_theta(s, a)); this is the stochastic approximation to the Bellman optimality operator.
- caveat found: using max for both action selection and evaluation can overestimate values; double DQN prevents by selecting the action with the online net and evaluating it with the target net.

Target net lags the online net to reduce moving-target instability. Replay buffer approximates i.i.d. draws to reduce correlation in gradient steps.

## Count-based bonus
Q_bonus(s, a) = Q(s, a) + c / sqrt(N(s, a))

Where N(s, a) is the visit count and c is a scalar hyperparameter (here 50). This is a simple optimism bonus kinda similar to UCB-style exploration.

## Epsilon-greedy schedule
- Start epsilon_0 = 1.0
- Each step: epsilon = max(epsilon_min, epsilon * decay)
- With decay < 1, epsilon decreases geometrically over steps, not per episode, so total random-action mass is spread across the full training horizon.

## State representations
All are RAM-based:
1) Full 128-byte RAM.
2) Selected bytes: player x (addr 28), invaders left (17), enemies x (26), enemies y (24).
3) Same selected bytes but scaled to [0, 1] for smoother optimization.

## Optimistic initialisation
Linear layer biases are initialized to a positive value (10.0). This encodes a prior that unseen state-action values are good, nudging exploration early; the bias is updated away as data arrives.
