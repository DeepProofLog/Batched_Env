# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepProofLog (DPrL) is a neurosymbolic system that bridges probabilistic logic programming with deep learning for knowledge graph reasoning. It uses PPO (Proximal Policy Optimization) combined with a vectorized logical unification engine to scale neurosymbolic reasoning to large knowledge bases.

**Paper**: [arXiv:2511.08581](https://arxiv.org/abs/2511.08581)

## Common Commands

### Environment Setup
```bash
conda env create -f environment.yml
conda activate rl
```

### Running Experiments
```bash
# Basic training run
python runner.py --dataset countries_s3 --experiment kge --total_timesteps 50000

in day to day work and all agentic tasks, use runner_kge.py instead of runner.py

# With custom parameters
python runner.py --dataset family --n_envs 128 --n_steps 128 --batch_size 512 --device cuda
```

Available datasets: `countries_s1`, `countries_s2`, `countries_s3`, `family`, `fb15k237`, `wn18rr`

### Testing
```bash
# Run parity tests (compiled vs SB3 reference)
pytest kge_experiments/tests/parity_sb3/ -v

# Run compiled parity tests
pytest kge_experiments/tests/parity_tensor/ -v

# Run all tests
pytest kge_experiments/tests/ -v

# Profile training performance
python kge_experiments/tests/profile_learn.py

# Profile evaluation performance
python kge_experiments/tests/profile_eval.py
```

## Architecture

```
runner.py                      # Entry point: parses args, orchestrates training
    ↓
registry.py                    # Factory registry: load_config → build_env → build_policy → get_algorithm
    ↓
kge_experiments/builder.py     # KGE-specific factory functions
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ kge_experiments/                                                     │
│   ├── config.py       TrainConfig dataclass (100+ hyperparameters)  │
│   ├── env.py          EnvVec: vectorized KG reasoning environment   │
│   ├── ppo.py          PPO algorithm with torch.compile support      │
│   ├── policy.py       ActorCriticPolicy: actor/critic neural nets   │
│   ├── unification.py  Vectorized logical unification engine         │
│   ├── data_handler.py Dataset loading & preprocessing               │
│   ├── index_manager.py String-to-index vocabulary mapping           │
│   ├── rollout.py      Experience buffer for PPO                     │
│   └── callbacks.py    Training callbacks (metrics, checkpoints)     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Data Flow
1. **Config** (`TrainConfig`) holds all hyperparameters and caches shared components
2. **Environment** (`EnvVec`) represents the logical resolution MDP; actions are rule selections
3. **Policy** (`ActorCriticPolicy`) embeds states (logical terms) and outputs action probabilities
4. **Unification** (`UnificationEngineVectorized`) performs batched Prolog-style unification
5. **PPO** trains the policy using collected rollouts with CUDA graph compilation

### Tensor Conventions
- State tensors: `[B, S, A, 3]` where B=batch, S=states, A=atoms, 3=(predicate, arg1, arg2)
- Action masks: `[B, S, num_rules]` boolean mask for valid actions
- All operations are vectorized; no explicit Python loops in hot paths

## Coding Standards

From `.agent/workflows/CLAUDE.md` (PhD Mode):

- **Tensor Shape Annotations**: Always document tensor shapes as `[B, T, D]` with transformation comments
- **Type Hints**: All function signatures must include type hints
- **Vectorization First**: Avoid explicit loops; minimize GPU-CPU synchronization, graph replays and always use reduce-overhead with full graphs.
- **Config over Hardcoding**: Pass hyperparameters via params or configs, never hardcode
- **Verify Optimizations**: When replacing components for performance, verify parity or profile speedup
- **Concise Comments**: Explain functionality briefly; write script purpose at the top
- **Use conda environment**: `/home/castellanoontiv/miniconda3/envs/rl/bin/python`

## Key Technical Documentation

- `kge_experiments/docs/CUDA_compilation.md` - Comprehensive guide on torch.compile optimization, CUDA graphs, and performance debugging (83x speedup achieved)
- `kge_experiments/docs/compiled_rollout_deep_dive.md` - Experience collection optimization
- `kge_experiments/docs/compiled_eval_deep_dive.md` - Evaluation performance optimization

## Important Directories

### Reference Implementations (for parity testing)
- `kge_experiments/sb3/` - Older Stable-Baselines3 reference implementation
- `kge_experiments/tensor/` - Older tensor-based implementation

These folders contain legacy versions kept to ensure correctness via parity tests. The parity tests compare the current optimized code against these references.

### Paper Results
- `kge_experiments/AAAI26/` - Contains results from the published paper (if available) in case you are interested in the results of combining RL+KGE for link prediction at inference, otherwise not relevant.

### Dataset Location
All datasets are in `kge_experiments/data/{dataset_name}/` with files:
- `train.txt`, `valid.txt`, `test.txt` - KG triples
- `rules.txt` - Logical rules for reasoning
- `facts.txt` - Background facts (typically same as train.txt)

## Verification After Modifications

**IMPORTANT**: When modifying core modules (`policy.py`, `ppo.py`, `env.py`), always run the profiling scripts to verify correctness and performance:

```bash
# Verify training performance and correctness
python kge_experiments/tests/profile_learn.py

# Verify evaluation performance and correctness
python kge_experiments/tests/profile_eval.py
```

These scripts save results to `profile_learn.txt` and `profile_eval.txt` respectively, allowing comparison of metrics (MRR, runtime, steps/second) before and after changes.
