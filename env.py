# env.py — Contract-correct, SB3-free TorchRL environment (fast)  
# This file defines a single env class `LogicEnv` that follows TorchRL's EnvBase contract:
# - reset(): returns observations at ROOT (no 'next')
# - _step(): returns a TensorDict with 'reward', 'done', 'terminated', 'truncated' at ROOT
#            and the next observation under the 'next' sub-TensorDict.
# The heavy logic is inherited from env_original.LogicEnv_gym; we only change I/O layout and
# guarantee consistent keys (no 'next' flattening, no NonTensorData in policy paths).

from __future__ import annotations
from typing import Optional
import torch
from tensordict import TensorDict

# Import the original environment logic
from env_original import LogicEnv_gym as _CoreEnv


class LogicEnv(_CoreEnv):
    """Fast, contract-correct environment for TorchRL collectors."""

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        # Use the core reset (already returns obs at ROOT). Keep only tensors the policy/metrics need.
        td = super()._reset(tensordict, **kwargs)
        # Ensure termination keys exist and have correct shapes
        td.setdefault('terminated', torch.zeros(1, dtype=torch.bool, device=self.device))
        td.setdefault('truncated', torch.zeros(1, dtype=torch.bool, device=self.device))
        td.setdefault('done', (td['terminated'] | td['truncated']).to(torch.bool))
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Run core transition (returns new obs at ROOT in the original implementation)
        base_td = super()._step(tensordict)

        # Root keys
        reward = base_td.get('reward', torch.zeros(1, dtype=torch.float32, device=self.device))
        terminated = base_td.get('terminated', torch.zeros(1, dtype=torch.bool, device=self.device))
        truncated = base_td.get('truncated', torch.zeros(1, dtype=torch.bool, device=self.device))
        done = (terminated | truncated).to(torch.bool)

        out = TensorDict({}, batch_size=torch.Size([]), device=self.device)
        out.set('reward', reward)
        out.set('terminated', terminated)
        out.set('truncated', truncated)
        out.set('done', done)

        # Build 'next' observation from the new state's fields the policy needs.
        nxt = TensorDict({}, batch_size=torch.Size([]), device=self.device)
        for k in ('sub_index', 'derived_sub_indices', 'action_mask'):
            if k in base_td.keys():
                nxt.set(k, base_td.get(k))

        # Mirror reward/done under 'next' for downstream code that expects next/*.
        nxt.set('reward', reward)
        nxt.set('done', done)

        # Episode metadata for callbacks / metrics (kept under 'next')
        for k in ('label', 'query_depth', 'is_success', 'episode_idx'):
            if k in base_td.keys():
                nxt.set(k, base_td.get(k))

        out.set('next', nxt)
        return out