# Compiled Rollout Deep Dive
# Training vs Evaluation Architecture Analysis

This document provides an in-depth analysis of how rollout collection works in `ppo.py` and `env.py`, comparing it with the evaluation system, and discussing the feasibility of sharing environments and compilation strategies.

---

## Table of Contents

1. [Rollout vs Evaluation: Fundamental Differences](#1-rollout-vs-evaluation-fundamental-differences)
2. [Environment Architecture (env.py)](#2-environment-architecture-envpy)
3. [PPO Rollout Collection (ppo.py)](#3-ppo-rollout-collection-ppopy)
4. [Compilation Strategy Analysis](#4-compilation-strategy-analysis)
5. [Shared vs Separate Environments](#5-shared-vs-separate-environments)
6. [Fused Step Compilation](#6-fused-step-compilation)
7. [Compilation Problems and Solutions](#7-compilation-problems-and-solutions)
8. [TensorDict in Rollout](#8-tensordict-in-rollout)
9. [Performance Comparison](#9-performance-comparison)
10. [Recommendations](#10-recommendations)

---

## 1. Rollout vs Evaluation: Fundamental Differences

### 1.1 Purpose Comparison

| Aspect | Rollout (Training) | Evaluation |
|--------|-------------------|------------|
| **Goal** | Collect experiences for learning | Measure policy performance |
| **Action selection** | Stochastic (sample from probs) | Deterministic (argmax) |
| **Data needed** | obs, action, reward, log_prob, value | success/failure only |
| **Gradient tracking** | Forward pass for values | No gradients |
| **Query source** | Random/shuffled from pool | Fixed test set |
| **Auto-reset** | Yes (continuous) | Per-query (slot recycling) |
| **RolloutBuffer** | Yes | No |

### 1.2 Step Function Differences

**Rollout Step (Training):**
```python
def fused_step(obs, state):
    logits = policy.get_logits(obs)
    masked = logits.masked_fill(obs['action_mask'] == 0, -inf)
    probs = torch.softmax(masked, dim=-1)
    actions = torch.multinomial(probs, 1).squeeze(-1)  # SAMPLE!
    log_probs = torch.log_softmax(masked, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
    new_obs, new_state = env._step_and_reset_core(state, actions, pool, ptrs)
    return new_obs, new_state, actions, log_probs
```

**Eval Step (Inference):**
```python
def fused_eval(obs, state):
    logits = policy.get_logits(obs)
    masked = logits.masked_fill(obs['action_mask'] == 0, -inf)
    actions = masked.argmax(dim=-1)  # DETERMINISTIC!
    new_obs, new_state = env._step_core(state, actions)  # No auto-reset
    return new_obs, new_state
```

### 1.3 Data Flow Diagram

```
ROLLOUT (Training):
┌────────────────────────────────────────────────────────────────┐
│  Query Pool ──▶ Environment ──▶ Policy ──▶ RolloutBuffer       │
│     ▲                │             │              │            │
│     │                │             │              ▼            │
│     └────────────────┘             │         GAE + Returns     │
│     (auto-reset on done)           │              │            │
│                                    │              ▼            │
│                              Sample Actions    Train Loop      │
│                                    │              │            │
│                                    ▼              ▼            │
│                              log_probs       Loss + Update     │
└────────────────────────────────────────────────────────────────┘

EVALUATION:
┌────────────────────────────────────────────────────────────────┐
│  Candidate Pool ──▶ Environment ──▶ Policy ──▶ Results         │
│     ▲                    │             │           │           │
│     │                    │             │           ▼           │
│     └────────────────────┘             │       success[]       │
│     (slot recycling)              argmax           │           │
│                                        │           ▼           │
│                                        │    Ranking Metrics    │
│                                        │                       │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Environment Architecture (env.py)

### 2.1 EnvVec Class Structure

```python
class EnvVec:
    """Vectorized KG reasoning environment."""
    
    # Core components
    engine: UnificationEngineVectorized  # Unification logic
    batch_size: int                       # Number of parallel slots
    padding_atoms: int                    # Max atoms per state
    padding_states: int                   # Max derived states
    
    # Mode-specific state
    _query_pool: Tensor                   # Available queries
    _per_env_ptrs: Tensor                 # Current index per slot
    is_training: bool                     # Train vs eval mode
    
    # Compiled functions
    _reset_fn: CompiledFunction           # Reset from queries
    _step_fn: CompiledFunction            # Step without reset
    _step_and_reset_fn: CompiledFunction  # Fused step + reset
```

### 2.2 State Structure (TensorDict)

```python
state = TensorDict({
    # Proof state
    "current_states": [B, A, 3],          # Current proof state
    "derived_states": [B, S, A, 3],       # Possible next states
    "derived_counts": [B],                 # Valid derived count
    "original_queries": [B, A, 3],        # Initial query
    "next_var_indices": [B],              # Variable tracking
    
    # Progress tracking
    "depths": [B],                         # Current depth
    "done": [B],                           # Completion flag
    "success": [B],                        # Proof success
    "current_labels": [B],                 # Query labels
    
    # History (for pruning)
    "history_hashes": [B, max_history],   # Visited state hashes
    "history_count": [B],                  # History length
    
    # Rewards
    "step_rewards": [B],                   # Immediate reward
    "step_dones": [B],                     # Just-completed flag
    "cumulative_rewards": [B],             # Total reward
    
    # Pool management
    "per_env_ptrs": [B],                   # Pool indices
    "neg_counters": [B],                   # Negative sampling count
})
```

### 2.3 Key Environment Methods

**`_step_core`**: Pure step without reset
```python
def _step_core(self, state, actions):
    # 1. Apply action (select derived state)
    next_states = state['derived_states'][batch_idx, actions]
    new_current = torch.where(active, next_states, state['current_states'])
    
    # 2. Compute reward and termination
    rewards, terminated, truncated, is_success = self._compute_reward(new_current, ...)
    
    # 3. Update history
    new_hash = self._compute_hash(new_current)
    # ... update history_hashes
    
    # 4. Compute new derived states (engine call)
    new_derived, new_counts = self._compute_derived(new_current, ...)
    
    # 5. Return new state
    return obs, new_state
```

**`_step_and_reset_core`**: Fused step + auto-reset
```python
def _step_and_reset_core(self, state, actions, query_pool, per_env_ptrs):
    # 1. Do normal step
    _, next_state = self._step_core(state, actions)
    
    # 2. Get next query for done slots
    done_mask = next_state['step_dones'].bool()
    next_query = query_pool[per_env_ptrs % pool_size]
    
    # 3. Reset done slots
    reset_state = self._reset_from_queries(next_query, labels)
    
    # 4. Merge: reset slots get reset_state, others keep next_state
    mixed = TensorDict({
        "current_states": torch.where(done_mask, reset_state['...'], next_state['...']),
        ...
    })
    
    return obs, mixed
```

### 2.4 Mode Switching

```python
def train(self):
    """Switch to training mode."""
    self._set_queries_internal(training_queries)
    self.negative_ratio = config.negative_ratio  # Enable negative sampling
    self.order = False  # Random query order
    self.rejection_weight = config.rejection_weight

def eval(self, queries):
    """Switch to evaluation mode."""
    self._set_queries_internal(queries)
    self.negative_ratio = 0.0  # No negative sampling
    self.order = True  # Sequential order
```

---

## 3. PPO Rollout Collection (ppo.py)

### 3.1 Collect Rollouts Flow

```python
def collect_rollouts(self, state, obs, ...):
    """Collect n_steps of experience."""
    
    while n_collected < self.n_steps:
        # 1. Snapshot observation for storage
        obs_snap = {k: v.clone() for k, v in obs.items()}
        
        # 2. Get value estimate (for GAE)
        values = self.policy.predict_values(obs)
        
        # 3. Compiled fused step
        torch.compiler.cudagraph_mark_step_begin()
        new_obs, new_state, actions, log_probs = self._compiled_rollout_step(obs, state)
        
        # 4. Store in buffer
        self.rollout_buffer.add(
            sub_index=obs_snap['sub_index'],
            derived_sub_indices=obs_snap['derived_sub_indices'],
            action_mask=obs_snap['action_mask'],
            action=actions,
            reward=new_state['step_rewards'],
            value=values,
            log_prob=log_probs,
        )
        
        # 5. Handle done episodes (logging)
        done_idx = torch.nonzero(new_state['step_dones'])
        if done_idx.numel() > 0:
            self._handle_done_episodes(...)
        
        state, obs = new_state, new_obs
        n_collected += 1
    
    # Compute advantages
    self.rollout_buffer.compute_returns_and_advantage(last_values)
```

### 3.2 The Fused Rollout Step

```python
def _setup_fused_rollout_step(self):
    policy, env = self._uncompiled_policy, self.env
    
    def fused_step(obs, state):
        # Policy forward
        logits = policy.get_logits(obs)
        masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
        
        # Stochastic action selection
        probs = torch.softmax(masked, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        
        # Compute log probabilities
        log_probs = torch.log_softmax(masked, dim=-1)
        log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        log_probs = log_probs.masked_fill(state['done'].bool(), 0.0)
        
        # Environment step with auto-reset
        new_obs, new_state = env._step_and_reset_core(
            state, actions, env._query_pool, env._per_env_ptrs
        )
        
        return new_obs, new_state, actions, log_probs
    
    self._compiled_rollout_step = torch.compile(fused_step, mode='reduce-overhead')
```

### 3.3 RolloutBuffer Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                    ROLLOUT BUFFER                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 0:  obs₀, action₀, reward₀, value₀, log_prob₀         │
│  Step 1:  obs₁, action₁, reward₁, value₁, log_prob₁         │
│  ...                                                         │
│  Step n:  obsₙ, actionₙ, rewardₙ, valueₙ, log_probₙ         │
│                                                              │
│  After collection:                                          │
│  ┌────────────────────────────────────────┐                 │
│  │  compute_returns_and_advantage()       │                 │
│  │  → advantages, returns                 │                 │
│  └────────────────────────────────────────┘                 │
│                                                              │
│  During training:                                           │
│  ┌────────────────────────────────────────┐                 │
│  │  for batch in buffer.get(batch_size): │                 │
│  │      loss = compute_loss(batch)        │                 │
│  │      optimizer.step()                  │                 │
│  └────────────────────────────────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Compilation Strategy Analysis

### 4.1 What Gets Compiled

**For Rollout:**
```python
# 1. Fused rollout step (policy + env step + reset)
self._compiled_rollout_step = torch.compile(fused_step, mode='reduce-overhead')

# 2. Loss module (policy forward + loss computation)
self.loss_module = torch.compile(PPOLossModule(policy), mode='reduce-overhead')

# 3. Policy itself
self.policy = torch.compile(policy, mode='reduce-overhead')
```

**For Evaluation:**
```python
# 1. Fused eval step (policy + env step, no reset)
self._compiled_eval_step = torch.compile(fused_eval, mode='reduce-overhead')

# 2. Policy logits (if eval_only mode)
self._compiled_policy_fn = torch.compile(policy.get_logits, mode='reduce-overhead')
```

### 4.2 Compilation Graph Structure

```
ROLLOUT COMPILATION:
┌────────────────────────────────────────────────────────────────┐
│  fused_step() - Single Graph                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ policy.get_logits()                                       │ │
│  │     ↓                                                     │ │
│  │ softmax + multinomial                                     │ │
│  │     ↓                                                     │ │
│  │ log_softmax + gather                                      │ │
│  │     ↓                                                     │ │
│  │ env._step_and_reset_core()                                │ │
│  │     ├── _step_core()                                      │ │
│  │     │       ├── action selection                          │ │
│  │     │       ├── reward computation                        │ │
│  │     │       └── _compute_derived() (engine)               │ │
│  │     └── reset for done slots                              │ │
│  │             └── _reset_from_queries()                     │ │
│  │                     └── _compute_derived() (engine)       │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘

EVALUATION COMPILATION (V10):
┌────────────────────────────────────────────────────────────────┐
│  step_fn() - Single Graph + torch.cond                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ policy.get_logits()                                       │ │
│  │     ↓                                                     │ │
│  │ masked_fill + argmax                                      │ │
│  │     ↓                                                     │ │
│  │ action selection (derived[actions])                       │ │
│  │     ↓                                                     │ │
│  │ engine.get_derived()                                      │ │
│  │     ↓                                                     │ │
│  │ torch.cond(needs_reset.any(), reset_branch, no_reset)     │ │
│  │     ↓                                                     │ │
│  │ return new states                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 4.3 Engine Calls: Rollout vs Eval

**Rollout:** Two engine calls per step when reset happens
- `_step_core` → `_compute_derived` (for step)
- `_reset_from_queries` → `_compute_derived` (for reset)

**Eval (V10):** One or two engine calls depending on torch.cond
- Step always calls engine
- Reset branch calls engine only when needed

---

## 5. Shared vs Separate Environments

### 5.1 Current Architecture: Shared Environment

```python
class PPO:
    def __init__(self, policy, env, config):
        self.env = env  # Single environment for both
    
    def learn(self, ...):
        self.env.train()  # Switch to training mode
        # ... collect rollouts ...
    
    def evaluate(self, ...):
        self.env.eval(queries)  # Switch to eval mode
        # ... run evaluation ...
```

### 5.2 Pros of Shared Environment

| Advantage | Explanation |
|-----------|-------------|
| Memory efficiency | One set of buffers, one engine |
| Code simplicity | Single env initialization |
| Consistency | Same engine behavior for both |
| Compilation sharing | Engine compilation shared |

### 5.3 Cons of Shared Environment

| Disadvantage | Explanation |
|--------------|-------------|
| Mode switching | Must switch train()/eval() |
| State contamination | Risk of training state in eval |
| Compilation conflicts | Different step functions may conflict |
| Buffer size mismatch | Training batch != eval batch |

### 5.4 Separate Environments Analysis

**Architecture:**
```python
class PPO:
    def __init__(self, policy, train_env, eval_env, config):
        self.train_env = train_env
        self.eval_env = eval_env
    
    def learn(self, ...):
        # Always use train_env, no mode switching
        ...
    
    def evaluate(self, ...):
        # Always use eval_env, no mode switching
        ...
```

**Pros:**
- No mode switching needed
- Clear separation of concerns
- Can have different batch sizes
- No state contamination risk
- Can optimize each independently

**Cons:**
- 2x memory for engine embeddings (if not shared)
- More initialization code
- Potential for drift between envs

### 5.5 Hybrid Approach (Recommended)

```python
class PPO:
    def __init__(self, policy, env, config):
        self.env = env  # Shared for memory
        
        # Different compiled functions
        self._setup_fused_rollout_step()  # For training
        self._setup_fused_eval_step()      # For evaluation
```

The engine and core logic are shared, but compilation is separate.

---

## 6. Fused Step Compilation

### 6.1 Rollout Fused Step Details

```python
def fused_step(obs, state):
    """Everything in one compiled function."""
    
    # === POLICY FORWARD ===
    logits = policy.get_logits(obs)
    # Shape: [B, S] where S = padding_states
    
    # === MASKING ===
    masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
    
    # === STOCHASTIC ACTION ===
    probs = torch.softmax(masked, dim=-1)
    actions = torch.multinomial(probs, 1).squeeze(-1)
    # multinomial is non-deterministic, works in compile
    
    # === LOG PROBABILITY ===
    log_probs = torch.log_softmax(masked, dim=-1)
    log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    log_probs = log_probs.masked_fill(state['done'].bool(), 0.0)
    
    # === ENVIRONMENT STEP ===
    new_obs, new_state = env._step_and_reset_core(
        state, actions, env._query_pool, env._per_env_ptrs
    )
    
    return new_obs, new_state, actions, log_probs
```

### 6.2 Eval Fused Step Details

```python
def fused_eval(obs, state):
    """Simpler - no sampling, no log_probs."""
    
    # === POLICY FORWARD ===
    logits = policy.get_logits(obs)
    
    # === DETERMINISTIC ACTION ===
    masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
    actions = masked.argmax(dim=-1)  # No sampling!
    
    # === ENVIRONMENT STEP ===
    new_obs, new_state = env._step_core(state, actions)
    # No auto-reset - handled separately in V10
    
    return new_obs, new_state
```

### 6.3 Can We Use One Fused Step for Both?

**Attempt: Parameterized Fused Step**
```python
def unified_step(obs, state, deterministic, auto_reset):
    logits = policy.get_logits(obs)
    masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
    
    if deterministic:  # GRAPH BREAK!
        actions = masked.argmax(dim=-1)
    else:
        probs = torch.softmax(masked, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
    
    ...
```

**Problem:** `if deterministic:` causes graph breaks!

**Solution with torch.cond:**
```python
def unified_step(obs, state, deterministic_tensor, auto_reset_tensor):
    logits = policy.get_logits(obs)
    masked = logits.masked_fill(obs['action_mask'] == 0, -3.4e38)
    
    def deterministic_branch(m):
        return m.argmax(dim=-1)
    
    def stochastic_branch(m):
        return torch.multinomial(torch.softmax(m, dim=-1), 1).squeeze(-1)
    
    actions = torch.cond(
        deterministic_tensor,
        deterministic_branch,
        stochastic_branch,
        (masked,)
    )
```

**But:** This adds torch.cond overhead and complexity.

**Recommendation:** Keep separate compiled functions - simpler and cleaner.

---

## 7. Compilation Problems and Solutions

### 7.1 Problem: Double Compilation of Policy

```python
# In _compile_all():
self._compiled_policy_fn = torch.compile(policy.get_logits, ...)
self.policy = torch.compile(policy, ...)

# In fused_step:
def fused_step(obs, state):
    logits = policy.get_logits(obs)  # Called inside fused
```

**Issue:** If `policy` is already compiled, and we compile `fused_step`, we have nested compilation.

**Solution:** Use uncompiled policy in fused functions:
```python
def _setup_fused_rollout_step(self):
    policy = self._uncompiled_policy  # Keep uncompiled reference
    
    def fused_step(obs, state):
        logits = policy.get_logits(obs)  # Not compiled yet
        ...
    
    # Compile the whole thing
    self._compiled_rollout_step = torch.compile(fused_step, ...)
```

### 7.2 Problem: Engine Compilation Conflicts

**Scenario:**
```python
# Env compiles engine separately
self.engine.compile()

# Then PPO compiles fused step that uses engine
fused_step = torch.compile(...)  # Calls engine internally
```

**Issue:** Nested compilation can cause graph breaks or errors.

**Solution:** Don't compile engine separately when used in fused steps:
```python
class EnvVec:
    def compile(self, mode, fullgraph):
        # Compile reset and step functions
        # But DON'T compile engine - let fused step handle it
        self._reset_fn = torch.compile(self._reset_from_queries, ...)
        self._step_fn = torch.compile(self._step_core, ...)
```

### 7.3 Problem: Different Batch Sizes

**Scenario:**
- Training batch: 2048
- Eval batch: 256

```python
compiled_train_step = torch.compile(step, ...)  # Captures B=2048
compiled_eval_step = torch.compile(step, ...)   # Captures B=256
```

**Issue:** If using shared env, batch size is baked into compiled graph.

**Solution 1:** Separate environments
```python
train_env = EnvVec(..., batch_size=2048)
eval_env = EnvVec(..., batch_size=256)
```

**Solution 2:** Fixed batch with padding
```python
# Always use max batch size, pad smaller batches
padded_queries = torch.cat([queries, padding], dim=0)[:max_batch]
```

### 7.4 Problem: Cloning in Compiled Regions

```python
def collect_rollouts(self, ...):
    obs_in, state_in = {k: v.clone() for k, v in obs.items()}, state.clone()
    new_obs, new_state = self._compiled_rollout_step(obs_in, state_in)
```

**Why clone?** CUDA graphs capture memory addresses. Without cloning:
1. Compiled step modifies original tensors
2. Next iteration reads modified data unexpectedly

**Cost:** Clone overhead (~1-2% total time)

### 7.5 Problem: TensorDict Operations in Compiled Code

**Issue:** Some TensorDict methods may not compile cleanly.

```python
# May cause issues:
new_state = state.clone()  # TensorDict clone
state.update(new_state)    # TensorDict update
```

**Solution:** Use explicit tensor operations:
```python
# Instead of TensorDict.clone():
new_state = TensorDict({
    k: v.clone() for k, v in state.items()
}, batch_size=state.batch_size)

# Instead of TensorDict.update():
for k in new_state.keys():
    state[k].copy_(new_state[k])
```

---

## 8. TensorDict in Rollout

### 8.1 TensorDict Usage in EnvVec

```python
# State is a TensorDict
state = TensorDict({
    "current_states": torch.zeros(B, A, 3),
    "derived_states": torch.zeros(B, S, A, 3),
    ...
}, batch_size=[B])

# Access like dict
current = state['current_states']

# Operations
state_copy = state.clone()
state.update(new_values)
```

### 8.2 Advantages for Rollout

1. **Structured state:** All tensors grouped logically
2. **Easy cloning:** `state.clone()` copies everything
3. **Batch operations:** Apply operations to all tensors
4. **Indexing support:** `state[mask]` for subset selection

### 8.3 Compilation Compatibility

TensorDict is designed for torch.compile:
- Tensors are traced individually
- Operations translate to tensor ops
- No Python control flow issues

**Caveat:** Complex TensorDict operations (nested, custom methods) may cause graph breaks.

### 8.4 Memory Layout

```
TensorDict State:
┌─────────────────────────────────────────────────────────────┐
│  state                                                       │
│  ├── current_states    → GPU Address 0x7f001000             │
│  ├── derived_states    → GPU Address 0x7f002000             │
│  ├── derived_counts    → GPU Address 0x7f003000             │
│  ├── depths            → GPU Address 0x7f004000             │
│  ├── done              → GPU Address 0x7f005000             │
│  └── ...               → ...                                │
└─────────────────────────────────────────────────────────────┘

Each tensor has independent GPU memory.
TensorDict is just metadata grouping them.
```

---

## 9. Performance Comparison

### 9.1 Rollout vs Eval Performance

| Metric | Rollout (Training) | Eval (V10) |
|--------|-------------------|------------|
| ms/step | ~1.5 | ~0.88 |
| Engine calls/step | 1-2 | 1-2 |
| Policy forward | Yes | Yes |
| Log prob compute | Yes | No |
| Multinomial | Yes | No |
| Auto-reset overhead | Every done | Slot recycling |

### 9.2 Why Eval is Faster

1. **No multinomial:** argmax is cheaper than sampling
2. **No log_prob gather:** Skip log probability computation
3. **Slot recycling:** Better utilization than sequential reset
4. **torch.cond:** Skip engine when not needed

### 9.3 Potential Rollout Optimizations

**1. Batch value prediction outside step:**
```python
# Currently: value computed separately
values = self.policy.predict_values(obs)
# Then: fused step

# Better: Fuse value prediction too
def fused_step_with_value(obs, state):
    logits, values = policy.forward_with_value(obs)
    ...
```

**2. Remove clone overhead:**
```python
# Current: Clone before compiled step
obs_in = {k: v.clone() for k, v in obs.items()}

# Better: Use buffer-copy pattern like V10
# Pre-allocate buffers, copy outside compiled
```

**3. Conditional engine with torch.cond:**
```python
# Current: Engine always called in step_and_reset
new_derived = engine.get_derived(new_current)

# Better: Skip for done slots
new_derived = torch.cond(
    any_active,
    lambda: engine.get_derived(new_current),
    lambda: state['derived_states']
)
```

---

## 10. Recommendations

### 10.1 For Current System (env.py + ppo.py)

1. **Keep shared environment** - Memory efficiency outweighs mode switching cost
2. **Separate fused functions** - One for rollout, one for eval
3. **Use uncompiled policy in fused** - Avoid nested compilation
4. **Fixed batch sizes** - Avoid recompilation

### 10.2 For Maximum Performance

1. **Implement V10-style slot recycling for training** - Higher throughput
2. **Use torch.cond for conditional engine** - Skip unnecessary work
3. **Buffer-copy pattern** - Eliminate clone overhead
4. **Compile engine separately first** - 4.3x engine speedup

### 10.3 Architecture Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| Memory constrained | Shared env, mode switching |
| Max training speed | V10-style with slot recycling |
| Max eval speed | Separate eval env (eval_minimal_v10) |
| Simple implementation | Shared env, separate fused steps |
| Production | Shared engine, separate compiled steps |

### 10.4 Summary

```
┌─────────────────────────────────────────────────────────────┐
│                  RECOMMENDED ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Shared Components:                                          │
│  ├── UnificationEngine (one instance)                        │
│  ├── Policy weights (one model)                              │
│  └── Query/Knowledge base tensors                            │
│                                                              │
│  Separate Compiled Functions:                                │
│  ├── _compiled_rollout_step (for training)                   │
│  │   ├── Uses _uncompiled_policy                             │
│  │   ├── Uses env._step_and_reset_core                       │
│  │   └── Includes multinomial, log_prob                      │
│  │                                                           │
│  └── _compiled_eval_step (for evaluation)                    │
│      ├── Uses _uncompiled_policy                             │
│      ├── Uses env._step_core or V10 pattern                  │
│      └── Uses argmax, no log_prob                            │
│                                                              │
│  Mode Switching:                                             │
│  ├── env.train() - Set training query pool                   │
│  └── env.eval() - Set evaluation queries                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

This architecture balances memory efficiency with optimization flexibility, while avoiding the pitfalls of nested compilation and mode conflicts.
