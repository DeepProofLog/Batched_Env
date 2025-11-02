# PPO Module - Modular Implementation

This directory contains a modular implementation of Proximal Policy Optimization (PPO) for the Neural-guided Grounding project.

## Structure

```
ppo/
├── __init__.py           # Module exports
├── model.py              # Neural network architectures (PolicyNetwork, ValueNetwork, ActorCriticModel)
├── rollout.py            # Experience collection (RolloutCollector, collect_rollouts)
├── learner.py            # PPO learning algorithm (PPOLearner)
└── ppo_agent.py          # Main PPO agent class (PPOAgent)
```

## Components

### 1. Model (`model.py`)

Contains the neural network architectures:

- **PolicyNetwork**: Residual MLP that produces policy logits
- **ValueNetwork**: Residual MLP that estimates state values
- **EmbeddingExtractor**: Converts index-based observations to embeddings
- **ActorCriticModel**: Combined actor-critic model
- **TorchRLActorModule**: TorchRL-compatible actor wrapper
- **TorchRLValueModule**: TorchRL-compatible value wrapper
- **create_torchrl_modules()**: Factory function to create TorchRL modules

### 2. Rollout (`rollout.py`)

Handles experience collection from environment interactions:

- **RolloutCollector**: Manages rollout collection and episode statistics
- **collect_rollouts()**: Main function to collect experience data

### 3. Learner (`learner.py`)

Implements the PPO learning algorithm:

- **PPOLearner**: Handles advantage computation (GAE) and policy optimization
  - `compute_advantages()`: Computes advantages using Generalized Advantage Estimation
  - `learn()`: Performs PPO optimization on collected experiences

### 4. PPO Agent (`ppo_agent.py`)

Main agent class that coordinates training:

- **PPOAgent**: High-level agent that manages the training loop
  - `train()`: Main training loop
  - `_save_checkpoint()`: Saves model checkpoints
  - `load_checkpoint()`: Loads model checkpoints

## Usage

### Basic Usage

```python
from ppo import PPOAgent, create_torchrl_modules

# Create actor and critic networks
actor, critic = create_torchrl_modules(
    embedder=embedder,
    num_actions=num_actions,
    embed_dim=200,
    hidden_dim=128,
    num_layers=8,
    dropout_prob=0.2,
    device=device,
)

# Create optimizer
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()),
    lr=3e-4
)

# Create PPO agent
agent = PPOAgent(
    actor=actor,
    critic=critic,
    optimizer=optimizer,
    train_env=train_env,
    eval_env=eval_env,
    sampler=sampler,
    data_handler=data_handler,
    n_envs=128,
    n_steps=128,
    n_epochs=10,
    batch_size=128,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.5,
    device=device,
)

# Train the agent
actor, critic = agent.train(
    total_timesteps=400000,
    eval_callback=eval_callback,
    rollout_callback=rollout_callback,
    metrics_callback=metrics_callback,
    logger=logger,
)
```

### Advanced Usage

You can also use individual components:

```python
from ppo import PPOLearner, collect_rollouts

# Collect rollouts
experiences, stats = collect_rollouts(
    env=train_env,
    actor=actor,
    critic=critic,
    n_envs=128,
    n_steps=128,
    device=device,
)

# Create learner
learner = PPOLearner(
    actor=actor,
    critic=critic,
    optimizer=optimizer,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    n_epochs=10,
    batch_size=64,
    device=device,
)

# Learn from experiences
metrics = learner.learn(
    experiences=experiences,
    n_steps=128,
    n_envs=128,
)
```

## Key Features

### Modularity

- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **Reusability**: Components can be used independently or combined
- **Testability**: Easier to test individual components

### Flexibility

- **Customizable**: Easy to modify individual components without affecting others
- **Extensible**: Simple to add new features or algorithms

### Maintainability

- **Clear Structure**: Well-organized code with clear naming conventions
- **Documentation**: Comprehensive docstrings for all classes and methods
- **Type Hints**: Full type annotations for better IDE support

## Integration with Existing Code

The `model.py` file in the root directory now acts as a redirect to this module:

```python
from ppo.model import create_torchrl_modules
```

This ensures backward compatibility with existing code while providing the benefits of the modular structure.

## Configuration Parameters

### Model Parameters
- `embed_dim`: Embedding dimension (default: 200)
- `hidden_dim`: Hidden layer dimension (default: 128)
- `num_layers`: Number of residual layers (default: 8)
- `dropout_prob`: Dropout probability (default: 0.2)

### Training Parameters
- `n_envs`: Number of parallel environments (default: 128)
- `n_steps`: Steps per rollout (default: 128)
- `n_epochs`: Optimization epochs per update (default: 10)
- `batch_size`: Mini-batch size (default: 128)

### PPO Parameters
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda parameter (default: 0.95)
- `clip_range`: PPO clipping range (default: 0.2)
- `ent_coef`: Entropy coefficient (default: 0.01)
- `value_coef`: Value loss coefficient (default: 0.5)
- `max_grad_norm`: Maximum gradient norm (default: 0.5)

## Dependencies

- PyTorch
- TorchRL
- TensorDict

## Future Improvements

Potential enhancements for future versions:

1. **Parallel Rollouts**: Add support for distributed rollout collection
2. **Priority Experience Replay**: Implement prioritized sampling
3. **Multiple Policies**: Support for ensemble or mixture of policies
4. **Advanced GAE**: Alternative advantage estimation methods
5. **Hyperparameter Tuning**: Built-in support for hyperparameter optimization
