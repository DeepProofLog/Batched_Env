#!/usr/bin/env python
"""Quick test to verify debug output from ppo_rollout.py"""

print("Starting debug test...")

from ppo.ppo_rollout import collect_rollouts, _masked_policy_factory
import torch
import torch.nn as nn
from tensordict import TensorDict

print("Imports successful")

# Create minimal dummy actor
class MinimalActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = [nn.Linear(10, 5)]  # Minimal structure
        
actor = MinimalActor()
print(f"Actor created: {type(actor)}")

# Test the policy factory
policy = _masked_policy_factory(actor)
print(f"Policy factory created: {type(policy)}")

# Create a minimal test tensordict
td = TensorDict({
    "sub_index": torch.tensor([0]),
    "derived_sub_indices": torch.tensor([[0, 1, 2, -1]]),
    "action_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
}, batch_size=torch.Size([1]))

print("\nCalling policy with test tensordict...")
print(f"Input TD keys: {list(td.keys())}")

try:
    result = policy(td)
    print(f"\nPolicy call completed!")
    print(f"Result keys: {list(result.keys())}")
except Exception as e:
    print(f"\nPolicy call failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
