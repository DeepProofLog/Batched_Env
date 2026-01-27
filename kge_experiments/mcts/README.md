# MCTS Module for DeepProofLog

MuZero-style Monte Carlo Tree Search implementation for knowledge graph reasoning.

## Overview

This module implements MCTS with **real environment dynamics** (no learned dynamics model). Instead of predicting state transitions, we use the actual `env.step()` during tree search, which provides accurate simulations for the logical reasoning domain.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCTS Module                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   config.py          MCTSConfig dataclass                           │
│        │                                                             │
│        ▼                                                             │
│   ┌─────────┐    ┌──────────┐    ┌───────────────┐                 │
│   │  tree   │◄───│ networks │◄───│ replay_buffer │                 │
│   └────┬────┘    └──────────┘    └───────┬───────┘                 │
│        │                                  │                          │
│        ▼                                  ▼                          │
│   ┌─────────────────────────────────────────┐                       │
│   │           fast_trainer.py               │                       │
│   │  (collect episodes + train + evaluate)  │                       │
│   └─────────────────────────────────────────┘                       │
│                        │                                             │
│                        ▼                                             │
│              PPO.evaluate() for ranking                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `config.py` | `MCTSConfig` and `FastMCTSConfig` dataclasses with hyperparameters |
| `tree.py` | Core MCTS: `Node`, `MinMaxStats`, `MCTS` search algorithm |
| `networks.py` | Neural networks: embedder, backbone, policy/value heads |
| `replay_buffer.py` | Off-policy experience storage with n-step return computation |
| `trainer.py` | Original `MuZeroTrainer` implementation |
| `fast_trainer.py` | Optimized trainer with PPO evaluation integration |
| `run_mcts_family.py` | Entry point script for family dataset |
| `tests/` | Unit tests (54 tests) |

## Algorithm

### MCTS Search (PUCT)

For each action selection, we run `num_simulations` tree traversals:

```
1. SELECTION: From root, select child with highest UCB score until leaf
   UCB(s,a) = Q(s,a) + c * prior(a) * sqrt(parent_visits) / (1 + child_visits)

2. EXPANSION: At leaf, use real env.step() to get next state
   - Get policy prior and value estimate from neural network
   - Create child node

3. BACKPROPAGATION: Update statistics along search path
   - Increment visit counts
   - Update Q-values (mean of backed-up values)
   - Update MinMaxStats for Q normalization

4. ACTION SELECTION: Choose action based on visit counts
   action ~ (visit_count)^(1/temperature)
```

### Training Loop

```python
for iteration in range(num_iterations):
    # 1. Collect episodes using MCTS
    for episode in range(episodes_per_iter):
        trajectory = collect_episode_with_mcts()
        replay_buffer.add(trajectory)

    # 2. Sample batch and train
    batch = replay_buffer.sample(batch_size)

    # Policy target: MCTS visit distribution (not raw policy)
    policy_target = normalize(visit_counts)

    # Value target: n-step bootstrapped returns
    value_target = compute_nstep_returns(rewards, values, gamma, n=5)

    # 3. Compute loss and update
    policy_loss = cross_entropy(policy_logits, policy_target)
    value_loss = mse(predicted_value, value_target)
    loss = policy_loss + value_coef * value_loss
```

### Key Differences from PPO

| Aspect | PPO | MCTS |
|--------|-----|------|
| Action Selection | Sample from policy | Tree search + visit counts |
| Policy Target | Advantage-weighted actions | MCTS visit distribution |
| Value Target | GAE returns | N-step bootstrapped returns |
| Data Usage | On-policy (discard) | Off-policy (replay buffer) |
| Exploration | Entropy bonus | UCB exploration in tree |

## Usage

### Quick Start

```bash
# Train with default settings (3000 timesteps, 10 simulations)
python kge_experiments/mcts/fast_trainer.py

# Custom settings
python kge_experiments/mcts/fast_trainer.py \
    --timesteps 5000 \
    --mcts_sims 25 \
    --eval_queries 500 \
    --n_corruptions 100
```

### Programmatic Usage

```python
from kge_experiments.mcts import FastMCTSConfig, FastMCTSTrainer, FastMCTS
from kge_experiments.builder import get_default_config, create_env, create_policy
from kge_experiments.ppo import PPO

# Setup
config = get_default_config(dataset='family', device='cuda', n_envs=1)
env = create_env(config)
policy = create_policy(config, env)

# MCTS config
mcts_config = FastMCTSConfig(
    num_simulations=10,      # Simulations per action
    c_puct=1.25,             # Exploration constant
    dirichlet_alpha=0.3,     # Root noise
    temperature=1.0,         # Visit count temperature
)

# Create trainer
ppo_agent = PPO(policy, env, config, eval_only=True)
trainer = FastMCTSTrainer(policy, env, mcts_config, ppo_agent)

# Train
trainer.learn(total_timesteps=3000, episodes_per_iter=10)

# Evaluate
results = trainer.evaluate(test_queries, sampler, n_corruptions=100)
print(f"MRR: {results['MRR']:.4f}")
```

## Configuration

### FastMCTSConfig

```python
@dataclass
class FastMCTSConfig:
    num_simulations: int = 25       # MCTS simulations per action
    c_puct: float = 1.25            # PUCT exploration constant
    dirichlet_alpha: float = 0.3    # Dirichlet noise alpha
    dirichlet_epsilon: float = 0.25 # Noise weight at root
    temperature: float = 1.0        # Action selection temperature
    discount: float = 0.99          # Value discount factor
    max_episode_steps: int = 20     # Max steps per episode
```

### Recommended Settings

| Dataset | Simulations | Timesteps | Expected MRR |
|---------|-------------|-----------|--------------|
| family  | 10          | 3000      | ~0.48-0.56   |
| family  | 25          | 5000      | ~0.50-0.58   |

## Data Flow

```
Query (head, rel, ?)
       │
       ▼
┌──────────────────┐
│   env.reset()    │  Initialize proof state
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   MCTS Search    │  Run simulations, select action
│                  │
│  ┌────────────┐  │
│  │   Root     │  │  Current state
│  │  ┌─┴─┐     │  │
│  │  │   │     │  │  Expand children via env.step()
│  │ ┌┴┐ ┌┴┐    │  │
│  │ │ │ │ │    │  │  Evaluate with policy network
│  └─┴─┴─┴─┴────┘  │
└────────┬─────────┘
         │ action (rule selection)
         ▼
┌──────────────────┐
│   env.step()     │  Apply rule, get reward
└────────┬─────────┘
         │
         ▼
    Store (obs, action, reward, visit_counts, value)
         │
         ▼
    Repeat until done or max_steps
         │
         ▼
┌──────────────────┐
│  Replay Buffer   │  Store trajectory
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Train Policy    │  policy_loss + value_loss
└──────────────────┘
```

## Evaluation

Evaluation uses PPO's compiled ranking infrastructure for speed:

```python
# Separate environments for train (n_envs=1) and eval (n_envs=75)
train_env = create_env(train_config)  # n_envs=1 for MCTS
eval_env = create_env(eval_config)    # n_envs=75 for batched eval

# PPO handles corruption ranking with CUDA graphs
ppo_agent = PPO(policy, eval_env, eval_config, eval_only=True)
results = ppo_agent.evaluate(queries, sampler, n_corruptions=100)
```

Evaluation metrics:
- **MRR**: Mean Reciprocal Rank
- **Hits@k**: Proportion of correct answers in top-k
- **Proven**: Percentage of queries successfully proven

## Tests

```bash
# Run all MCTS tests
pytest kge_experiments/mcts/tests/ -v

# Run specific test file
pytest kge_experiments/mcts/tests/test_mcts_tree.py -v

# Run with coverage
pytest kge_experiments/mcts/tests/ --cov=kge_experiments/mcts
```

Test coverage:
- `test_mcts_tree.py`: Node, MinMaxStats, UCB formula, backpropagation
- `test_mcts_networks.py`: Embedder, backbone, policy/value heads
- `test_mcts_replay.py`: Buffer operations, n-step returns
- `test_mcts_integration.py`: Full search, trainer, config integration

## Performance Notes

1. **Simulations**: 10-25 simulations per action balances quality vs speed
2. **Evaluation Speed**: Uses CUDA graphs (~43s for 500 queries vs 700s without)
3. **Memory**: Replay buffer stores trajectories; adjust size for GPU memory
4. **Training Duration**: 3000 timesteps is often sufficient; longer may overfit

## References

- [MuZero Paper](https://arxiv.org/abs/1911.08265) - Original algorithm
- [PUCT Formula](https://www.chessprogramming.org/UCT#PUCT) - Upper Confidence bounds for Trees
- [DeepProofLog](https://arxiv.org/abs/2511.08581) - Base system this extends
