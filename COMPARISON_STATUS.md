# Batched vs SB3 Comparison Status

## Summary
Comparing batched (TorchRL) and SB3 (Stable-Baselines3) implementations to verify they produce identical results.

## Completed Work

### 1. File Renaming (✓ by User)
All sb3 files renamed with `sb3_` prefix to avoid import conflicts.

### 2. Import Fixes (✓ Completed)
Fixed all imports in sb3 folder to use new sb3_ prefixed names:
- `sb3_train.py`, `sb3_env.py`, `sb3_model.py`, `sb3_dataset.py`
- `sb3_callbacks.py`, `sb3_index_manager.py`, `sb3_unification.py`
- `sb3_custom_dummy_env.py`, `sb3_neg_sampling.py`

### 3. Comparison Infrastructure
- `runner_new.py`: Working batched runner
- `sb3/sb3_runner_simple.py`: SB3 runner (needs testing)
- `run_comparison.py`: Simple wrapper script

## Next Steps

###  1. Test SB3 Version
```bash
cd /home/castellanoontiv/Batched_env/sb3
/home/castellanoontiv/miniconda3/bin/conda run -n rl --no-capture-output \
  python sb3_runner_simple.py --n_envs 1 --n_steps 128 --timesteps_train 512 --seed 0
```

### 2. Compare Metrics
After both runs complete, compare:
- MRR, Hits@1, Hits@3, Hits@10
- Episode rewards and lengths
- Training losses

### 3. Fix Discrepancies
Per user requirement: "fix the one with the wrong logic, whether if it is sb3 or tensor version"

## Test Configuration
```
dataset: countries_s3
n_steps: 128  
batch_size/n_envs: 1
timesteps: 512
seed: 0
```
