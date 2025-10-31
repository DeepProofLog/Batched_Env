# Quick Reference: Gymnasium vs TorchRL

## Side-by-Side Comparison

### Creating an Environment

**Gymnasium:**
```python
import gymnasium as gym

class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(...)
        self.action_space = gym.spaces.Discrete(...)
```

**TorchRL:**
```python
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import CompositeSpec, DiscreteTensorSpec

class MyEnv(EnvBase):
    def __init__(self, device='cpu'):
        self.device = device  # MUST be before super().__init__()
        super().__init__(device=device, batch_size=torch.Size([]))
        self._make_spec()
    
    def _make_spec(self):
        self.observation_spec = CompositeSpec(...)
        self.action_spec = DiscreteTensorSpec(...)
        self.reward_spec = UnboundedContinuousTensorSpec(...)
        self.done_spec = BinaryDiscreteTensorSpec(...)
```

### Reset

**Gymnasium:**
```python
def reset(self, seed=None, options=None):
    # Setup
    obs = {'key1': np.array(...), 'key2': np.array(...)}
    info = {'extra': 'data'}
    return obs, info

# Usage
obs, info = env.reset()
```

**TorchRL:**
```python
def _reset(self, tensordict=None, **kwargs):
    # Setup
    td = TensorDict({
        'key1': torch.tensor(...),
        'key2': torch.tensor(...),
        'done': torch.tensor(False),
    }, batch_size=torch.Size([]))
    return td

# Usage
td = env.reset()
```

### Step

**Gymnasium:**
```python
def step(self, action):
    # Process action (int or array)
    obs = {'key': np.array(...)}
    reward = 1.0
    done = False
    truncated = False
    info = {'data': 'value'}
    return obs, reward, done, truncated, info

# Usage
obs, reward, done, truncated, info = env.step(action)
```

**TorchRL:**
```python
def _step(self, tensordict):
    action = tensordict["action"].item()  # Extract action
    # Process
    next_td = TensorDict({
        'key': torch.tensor(...),
        'reward': torch.tensor([reward]),  # Shape [1]!
        'done': torch.tensor(done),
        'data': 'value',  # Info integrated
    }, batch_size=torch.Size([]))
    return next_td

# Usage
action_td = TensorDict({"action": torch.tensor(action)}, batch_size=[])
next_td = env._step(action_td)
```

### Spaces vs Specs

**Gymnasium Spaces:**
```python
gym.spaces.Box(low, high, shape, dtype=np.float32)
gym.spaces.Discrete(n)
gym.spaces.MultiBinary(n)
gym.spaces.Dict({'key': space})
```

**TorchRL Specs:**
```python
BoundedTensorSpec(low, high, shape, dtype=torch.float32, device=device)
DiscreteTensorSpec(n, shape, dtype=torch.int64, device=device)
BinaryDiscreteTensorSpec(n, shape, dtype=torch.bool, device=device)
CompositeSpec(key=spec, shape=torch.Size([]))
```

### Action Masks

**Gymnasium:**
```python
# Usually in observation
obs = {
    'observation': np.array(...),
    'action_mask': np.array([1, 1, 0, 1, 0], dtype=np.uint8)
}
```

**TorchRL:**
```python
# In TensorDict with bool type
td = TensorDict({
    'observation': torch.tensor(...),
    'action_mask': torch.tensor([True, True, False, True, False], dtype=torch.bool)
}, batch_size=[])
```

### Seeding

**Gymnasium:**
```python
env.reset(seed=42)
# Or
env.seed(42)
```

**TorchRL:**
```python
env.reset(seed=42)
# Or
env.set_seed(42)

# Override _set_seed for custom behavior
def _set_seed(self, seed=None):
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    torch.manual_seed(seed)
    return seed
```

---

## Key Differences Summary

| Feature | Gymnasium | TorchRL |
|---------|-----------|---------|
| Base class | `gym.Env` | `EnvBase` |
| Data structure | dict/tuple with numpy | `TensorDict` with torch |
| Main methods | `reset()`, `step()` | `_reset()`, `_step()` |
| Observation | numpy arrays | torch tensors |
| Action input | int/array | TensorDict with "action" |
| Reward type | float/scalar | torch.Tensor with shape [1] |
| Done type | bool | torch.Tensor (bool) |
| Info | separate dict | integrated in TensorDict |
| Device | CPU (numpy) | Any torch device |
| Batching | Manual vectorization | Built-in with batch_size |
| Specs | Spaces | TensorSpecs |

---

## Common Gotchas

### 1. Reward Shape
❌ **Wrong:**
```python
td['reward'] = torch.tensor(1.0)  # Scalar
```

✅ **Correct:**
```python
td['reward'] = torch.tensor([1.0])  # Shape [1]
```

### 2. Device Consistency
❌ **Wrong:**
```python
td = TensorDict({'obs': torch.tensor(...).cuda()}, batch_size=[])
env.device = 'cpu'  # Mismatch!
```

✅ **Correct:**
```python
td = TensorDict({'obs': torch.tensor(..., device=env.device)}, batch_size=[])
```

### 3. Action Format
❌ **Wrong:**
```python
next_td = env._step(5)  # Raw action
```

✅ **Correct:**
```python
action_td = TensorDict({"action": torch.tensor(5)}, batch_size=[])
next_td = env._step(action_td)
```

### 4. Batch Size
❌ **Wrong:**
```python
td = TensorDict({...})  # No batch_size!
```

✅ **Correct:**
```python
td = TensorDict({...}, batch_size=torch.Size([]))  # Single env
# Or for batched:
td = TensorDict({...}, batch_size=torch.Size([batch_size]))
```

### 5. Method Names
❌ **Wrong (calling private methods):**
```python
# In external code
td = env._reset()  # Don't call _reset directly!
```

✅ **Correct:**
```python
# The base class provides public methods
td = env.reset()  # This calls _reset internally
```

---

## Migration Checklist

- [ ] Replace `gym.Env` with `EnvBase`
- [ ] Update imports (torchrl.envs, tensor_specs)
- [ ] Add `device` parameter to `__init__`
- [ ] Call `super().__init__(device=device, batch_size=...)`
- [ ] Replace spaces with specs in `_make_spec()`
- [ ] Update `reset()` to return TensorDict
- [ ] Rename `reset()` to `_reset()` (core implementation)
- [ ] Update `step()` to accept and return TensorDict
- [ ] Rename `step()` to `_step()` (core implementation)
- [ ] Change reward to shape `[1]`
- [ ] Change done to bool tensor
- [ ] Integrate info into TensorDict
- [ ] Update action mask to bool type
- [ ] Ensure all tensors on `self.device`
- [ ] Add `batch_size` to all TensorDicts
- [ ] Update `_set_seed()` to return seed value
- [ ] Test with test script

---

## Example: Full Episode

**Gymnasium:**
```python
obs, info = env.reset(seed=42)
done = False
total_reward = 0

while not done:
    action = policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    done = done or truncated
```

**TorchRL:**
```python
td = env.reset(seed=42)
total_reward = 0

while not td['done']:
    action = policy(td)
    action_td = TensorDict({"action": action}, batch_size=[])
    td = env._step(action_td)
    total_reward += td['reward'].item()
```

---

## Benefits of TorchRL

✅ **Native PyTorch**: No numpy conversion overhead  
✅ **GPU Support**: Run entire RL loop on GPU  
✅ **Batching**: Built-in support for parallel envs  
✅ **Composition**: Easy to chain env transforms  
✅ **Type Safety**: Specs enforce correct data types  
✅ **Modularity**: Cleaner separation of concerns  
✅ **Research**: Access to latest RL algorithms  

---

## Resources

- [TorchRL Documentation](https://pytorch.org/rl/)
- [TensorDict Tutorial](https://pytorch.org/tensordict/tutorials/)
- [EnvBase API](https://pytorch.org/rl/reference/envs.html)
