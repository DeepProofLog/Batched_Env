# Debug System for RL Pipeline

## Overview

A comprehensive debug configuration system has been implemented to help debug different components of the RL pipeline. The system is centralized in `debug_config.py` and integrated throughout the codebase.

## Components

### 1. DebugConfig (`debug_config.py`)

Central configuration class with the following debug categories:

- **debug_env**: Environment-level debugging (state transitions, rewards, etc.)
- **debug_prover**: Prover-level debugging (unification, proof search, etc.)
- **debug_agent**: Agent-level debugging (rollout statistics, training metrics, etc.)
- **debug_model**: Model-level debugging (logits, actions, distributions, etc.)
- **debug_rollouts**: Detailed rollout debugging (step-by-step info)

Each category supports verbosity levels (0=disabled, 1=basic, 2=detailed).

### 2. Integration Points

The debug system is integrated in:

- **`ppo/pposb3_model.py`**: Model-level debugging (logits, actions, entropy, distributions)
- **`ppo/pposb3.py`**: Agent-level debugging (rollout stats, training stats)
- **`tests/test_rollout.py`**: Command-line interface for debug modes

### 3. Preset Configurations

```python
# Entropy debugging (recommended for investigating low entropy)
debug_cfg = DebugConfig.create_entropy_debug()

# Agent debugging (rollout and training statistics)
debug_cfg = DebugConfig.create_agent_debug()

# Model debugging (logits, actions, distributions)
debug_cfg = DebugConfig.create_model_debug()

# Full debugging (all components)
debug_cfg = DebugConfig.create_full_debug()
```

## Usage

### Command Line (test_rollout.py)

```bash
# Use preset debug modes
python tests/test_rollout.py --debug_mode entropy
python tests/test_rollout.py --debug_mode agent
python tests/test_rollout.py --debug_mode model
python tests/test_rollout.py --debug_mode full

# Manual debug levels
python tests/test_rollout.py --debug_agent 2 --debug_model 1
```

### In Code

```python
from debug_config import DebugConfig

# Create debug config
debug_cfg = DebugConfig(
    debug_agent=2,
    debug_model=2,
    debug_model_entropy=True,
    debug_sample_envs=5  # Only show first 5 environments
)

# Pass to components
policy = create_actor_critic(..., debug_config=debug_cfg)
ppo = PPO(..., debug_config=debug_cfg)
```

## Debug Output Examples

### Model Forward Pass (Action Selection)
Shows:
- Valid actions per environment
- Logit ranges and statistics
- Distribution parameters (probs, entropy)
- Selected actions and their probabilities

### Agent Rollout Stats
Shows:
- Average actions available per step
- Distribution of action availability
- Episode completion statistics

### Agent Training Stats  
Shows:
- Training metrics (policy loss, value loss, entropy)
- Advantage statistics
- Value and return statistics
- Low entropy warnings

## Findings from Entropy Debugging

### Problem Identified

**Root Cause**: Low entropy is due to **lack of valid actions**, not model issues.

### Evidence

1. **Step 0** (initial state):
   - Valid actions: mean=3.00
   - Entropy: ~0.96-1.06 (reasonable)
   - Multiple actions with balanced probabilities

2. **Steps 1-3** (early proof search):
   - Valid actions drop to mean=1.48-1.55
   - Many environments have only 1 valid action
   - Entropy approaches 0 for those environments

3. **Steps 4+** (later proof search):
   - Valid actions: mean=1.00-1.08
   - Almost all environments have exactly 1 valid action
   - Entropy is exactly 0 (prob=1.0 for the only action)

### Implications

1. **The model is working correctly**: When multiple actions are available, the model produces reasonable probability distributions with decent entropy.

2. **The environment/prover is too restrictive**: After a few steps, the proof search space narrows down to typically only one valid action (likely the termination action).

3. **This is expected behavior**: In logical proof search, once you're near a proof or dead-end, there may genuinely be only one or very few valid next steps.

### Recommendations

1. **If low entropy is undesirable**:
   - Increase action space diversity (e.g., allow more unification choices)
   - Modify reward structure to encourage exploration before converging
   - Increase `ent_coef` to penalize low entropy (but this may hurt performance)
   - Consider adding action noise or epsilon-greedy exploration

2. **If low entropy is acceptable**:
   - This is normal for structured search tasks
   - The agent is behaving deterministically when the problem structure demands it
   - Monitor whether the agent learns effective strategies despite low entropy

## Future Extensions

The debug system can be easily extended:

1. **Add environment debugging**: Integrate with `debug_helper.py` for state/action visualization
2. **Add prover debugging**: Show unification steps, proof search trees
3. **Add rollout debugging**: Step-by-step environment transitions
4. **Add memory profiling**: Track GPU memory usage
5. **Add timing profiling**: Identify bottlenecks

## Files Modified

- `debug_config.py` (new): Central debug configuration
- `ppo/pposb3_model.py`: Added model debugging support
- `ppo/pposb3.py`: Added agent debugging support  
- `tests/test_rollout.py`: Added command-line debug options
