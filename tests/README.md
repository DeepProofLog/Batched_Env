# Test Suite Organization

This directory contains reorganized tests for the Batched_Env project. The tests are now organized into modular components that can be tested individually or collectively.

## File Structure

### Individual Test Modules

1. **test_sb3_engine.py** - Tests for SB3 (string-based) unification engine
   - Setup and testing of string-based unification engine
   - Single query and batch query testing
   - Supports both deterministic (canonical) and random action selection

2. **test_tensor_engine.py** - Tests for tensor-based unification engine
   - Setup and testing of tensor-based unification engine (non-batched and batched)
   - Single query and batch query testing
   - Supports both deterministic and random action selection

3. **test_sb3_env.py** - Tests for SB3 (string-based) environment
   - Setup and testing of string-based Gym environment
   - Tests environment stepping, rewards, and termination
   - Supports both deterministic and random policies

4. **test_tensor_env.py** - Tests for batched tensor environment
   - Setup and testing of batched tensor environment
   - Tests both sequential (batch_size=1) and true batch modes
   - Supports both deterministic and random policies

### Main Orchestrator

5. **test_env_engines.py** - Main test orchestrator
   - Tests all 6 configurations:
     * sb3_engine (string-based unification engine only)
     * tensor_engine (tensor unification engine only)
     * batched_tensor_engine (batched tensor unification engine only)
     * sb3_env (string-based environment)
     * tensor_env (tensor environment, sequential mode)
     * batched_tensor_env (batched tensor environment)
   - Compares results across configurations
   - Generates comprehensive statistics and reports

### Legacy Test Files (for reference)

- **test_engines.py** - Original engine comparison tests
- **test_envs.py** - Original environment comparison tests (interleaved)
- **test_envs_v2.py** - Original environment comparison tests (batch mode)

## Usage

### Running All Configurations

```bash
# Test with random policy (default)
python tests/test_env_engines.py --num-queries 100 --random --seed 42

# Test with deterministic (canonical) policy
python tests/test_env_engines.py --num-queries 100 --deterministic --seed 42
```

### Running Specific Configurations

```bash
# Test only engines
python tests/test_env_engines.py --num-queries 100 --random --configs sb3_engine tensor_engine

# Test only environments
python tests/test_env_engines.py --num-queries 100 --random --configs sb3_env batched_tensor_env
```

### Command-Line Options

- `--dataset`: Dataset to use (default: 'countries_s3')
- `--num-queries`: Number of queries to test (default: all queries)
- `--deterministic`: Use deterministic (canonical) policy
- `--random`: Use random policy (default if neither specified)
- `--max-depth`: Maximum proof depth (default: 20)
- `--seed`: Random seed for reproducibility (default: 42)
- `--verbose`: Print detailed information during testing
- `--debug`: **DEBUG MODE** - Compare step-by-step and raise error on first mismatch
- `--configs`: Specific configurations to test (space-separated list)

## Test Modes

### Deterministic Policy

With deterministic policy (`--deterministic`), the test:
- Uses canonical ordering to select actions (first canonical state)
- Compares all configurations step-by-step
- Verifies that states, derived states, and actions match across configurations
- Reports any mismatches in detail

### Random Policy

With random policy (`--random`), the test:
- Uses random action selection at each step
- Compares average success rates across configurations
- Expected result: ~38-42% average success rate on countries_s3 dataset
- Verifies that all configurations achieve similar success rates

### Debug Mode (NEW!)

With debug mode (`--debug`, requires `--deterministic`), the test:
- Compares configurations query-by-query, step-by-step
- **Raises AssertionError immediately** on first mismatch
- Provides detailed error message showing:
  - Exact query and step where mismatch occurred
  - Type of mismatch (state, num_actions, derived_states, etc.)
  - Complete information from both configurations
  - Previous step context for debugging
- Useful for finding exact differences between configurations

**Example:**
```bash
# Debug mode to find first mismatch
python tests/test_env_engines.py --num-queries 20 --deterministic --debug \
  --configs sb3_engine tensor_engine

# If all match, you'll see:
# ✓ DEBUG MODE: ALL QUERIES MATCH ACROSS ALL CONFIGURATIONS!
```

## Expected Results

### Countries_s3 Dataset

With 100 queries and random policy (seed=42):

```
Configuration                    Success Rate    Avg Steps
----------------------------------------------------------
sb3_engine                       38.00%          3.53
tensor_engine                    38.00%          3.53
batched_tensor_engine            38.00%          3.53
sb3_env                          37.00%          3.78
tensor_env                       32.00%          3.91
batched_tensor_env               37.00%          3.50
----------------------------------------------------------
Overall Average:                 36.67% ± 2.13%
```

With deterministic policy (seed=42):

```
Configuration                    Success Rate    Avg Steps
----------------------------------------------------------
sb3_engine                       42.00%          3.12
tensor_engine                    42.00%          3.12
sb3_env                          46.00%          3.52
----------------------------------------------------------
Overall Average:                 43.33% ± 1.89%
```

**Note:** The engine-only configurations (sb3_engine, tensor_engine, batched_tensor_engine) 
should produce identical results with deterministic policy. The environment configurations 
(sb3_env, tensor_env, batched_tensor_env) may differ slightly due to additional environment 
logic (reward shaping, pruning strategies, etc.).

## Sanity Checks

The test suite performs the following sanity checks:

1. **Consistency Check (Deterministic)**: 
   - All engine configurations should produce identical results
   - Traces should match step-by-step
   
2. **Success Rate Check (Random)**:
   - Average success rate should be in range 35-50%
   - All configurations should achieve similar rates (within ±5%)

3. **Per-Configuration Checks**:
   - Each configuration reports: total queries, successful queries, avg reward, avg steps
   - Traces include: states, derived states, actions, rewards, done flags

## Integration with Existing Tests

The new test suite is designed to replace the old tests while maintaining compatibility:

- **test_engines.py** → **test_sb3_engine.py** + **test_tensor_engine.py**
- **test_envs.py** → **test_sb3_env.py** + **test_tensor_env.py**
- **test_envs_v2.py** → **test_tensor_env.py** (batch mode)
- All tests → **test_env_engines.py** (orchestrator)

The old test files are kept for reference but should not be used for new testing.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure you're using the correct Python environment:
   ```bash
   /home/castellanoontiv/miniconda3/envs/rl/bin/python tests/test_env_engines.py ...
   ```

2. **Success rate too low**: 
   - Check if queries are being excluded properly (train vs valid/test)
   - Verify max_depth is sufficient (default: 20)
   - Try increasing num_queries for more stable statistics

3. **Deterministic comparison fails**:
   - This is expected for engine vs environment comparisons
   - Use `--debug` to see **exact first mismatch** with full details
   - Use `--verbose` for summary of all mismatches
   - Compare only similar configurations (e.g., sb3_engine vs tensor_engine)

### Using Debug Mode to Find Issues

When you see mismatches, use debug mode to identify the exact problem:

```bash
# Step 1: Run debug mode with small query set
python tests/test_env_engines.py --num-queries 10 --deterministic --debug \
  --configs config1 config2

# Step 2: Error will show exact query and step where mismatch occurs
# Step 3: Analyze if difference is expected or a bug
```

**Expected differences:**
- Engines vs Environments: Trace length differences (environments add terminal states)
- Success rate variations of 3-5% are normal

**Unexpected differences:**
- Engines not matching each other (sb3 vs tensor vs batched)
- Large success rate differences (>10%)

## Future Enhancements

Possible improvements:
- Add support for more datasets (family, fb15k237, wn18rr)
- Add per-step reward tracking and comparison
- Add visualization of proof traces
- Add performance benchmarking (time, memory)
- Add support for different reward types
- Add statistical significance tests for random policy comparisons
