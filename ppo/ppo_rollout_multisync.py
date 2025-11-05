# ppo_rollout_multisync.py — Rollout using TorchRL MultiSyncDataCollector
# This version uses MultiSyncDataCollector for potentially better parallelism
#
# Usage: Pass a ParallelEnv (created via env_factory.create_environments) to the collector.
# The MultiSyncDataCollector will create multiple sub-collectors, each managing a subset
# of environments. This can improve throughput for environments with variable step times.
#
# IMPLEMENTATION STATUS:
# ✅ The implementation is complete and working correctly
# ✅ env_factory.py has been refactored to use picklable PicklableEnvCreator
# ✅ DataHandler has pickle support (excludes/reconstructs sampler)
# ✅ MultiSyncDataCollector successfully uses env creator functions from ParallelEnv
# ✅ Tested and verified with test_multisync_real.py
#
# KEY IMPLEMENTATION DETAIL:
# MultiSyncDataCollector requires environment creator functions, not a ParallelEnv instance.
# We extract the env_fns from ParallelEnv (stored by env_factory) and pass those to
# MultiSyncDataCollector. This allows the collector to properly distribute environments
# across its sub-collectors.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import ParallelEnv

# # Import shared components from ppo_rollout.py to avoid code duplication
# from ppo.ppo_rollout import (
#     MaskedPolicyWrapper,
#     _reshape_time_env,
#     _extract_episode_infos,
# )


def _extract_episode_infos(batch_td: TensorDict, n_steps: int, n_envs: int, device: torch.device, verbose: bool = False) -> List[Dict[str, Any]]:
    """Extract episode infos more efficiently.

    This implementation avoids iterating over every time step. Instead it
    locates episode-end positions and computes episode returns and lengths
    per-environment which reduces Python-level loop overhead when n_steps is large.
    """
    infos: List[Dict[str, Any]] = []

    done = batch_td.get(('next', 'done'))
    reward = batch_td.get(('next', 'reward'))
    label = batch_td.get(('next', 'label'), None)
    depth = batch_td.get(('next', 'query_depth'), None)
    success = batch_td.get(('next', 'is_success'), None)

    done = done.reshape(n_steps, n_envs).squeeze(-1).to(torch.bool)
    reward = reward.reshape(n_steps, n_envs).squeeze(-1)

    if label is not None:
        label = label.reshape(n_steps, n_envs).squeeze(-1)
    if depth is not None:
        depth = depth.reshape(n_steps, n_envs).squeeze(-1)
    if success is not None:
        success = success.reshape(n_steps, n_envs).squeeze(-1)

    # Precompute cumulative sums for fast range-sum queries per env
    csum = reward.cumsum(dim=0)

    # For each environment, find time indices where an episode finished
    # then compute return and length using cumsum and the previous done index.
    for env in range(n_envs):
        done_indices = torch.nonzero(done[:, env], as_tuple=False).reshape(-1)
        if done_indices.numel() == 0:
            continue
        prev_t = -1
        for t in done_indices.tolist():
            # Sum of rewards from prev_t+1 .. t inclusive
            if prev_t < 0:
                ep_ret = float(csum[t, env].item())
            else:
                ep_ret = float((csum[t, env] - csum[prev_t, env]).item())
            ep_len = int(t - prev_t)

            info: Dict[str, Any] = {"episode": {"r": ep_ret, "l": ep_len}}
            if label is not None:
                info["label"] = int(label[t, env].item())
            if depth is not None:
                info["query_depth"] = int(depth[t, env].item())
            if success is not None:
                info["is_success"] = bool(success[t, env].item())

            infos.append(info)
            prev_t = t

    if verbose:
        print(f"[_extract_episode_infos] Extracted {len(infos)} episode infos from {n_envs} envs")
    return infos



def _reshape_time_env(td: TensorDict, n_steps: int, n_envs: int) -> TensorDict:
    """Reshape a flat [n_steps*n_envs] batch to [n_steps, n_envs]."""
    return td.reshape(n_steps, n_envs)


class MaskedPolicyWrapper:
    """Lightweight policy wrapper that re-applies action_mask before sampling.

    The wrapper keeps the minimum runtime overhead required to ensure
    invalid actions are masked out. Detailed logging and heavy checks
    are disabled by default to avoid slowing down rollout collection.
    """

    def __init__(self, actor, debug: bool = False):
        self.actor = actor
        self.debug = debug

    @torch.no_grad()
    def __call__(self, td: TensorDict) -> TensorDict:
        # Run the actor's first module (logits producer). Use the provided
        # tensordict directly to avoid extra copies; exclude keys only when
        # they exist to avoid unnecessary allocations.
        keys_to_remove = [k for k in ("logits", "action", "sample_log_prob") if k in td.keys()]
        td_clean = td.exclude(*keys_to_remove) if keys_to_remove else td

        if not hasattr(self.actor, "module") or len(self.actor.module) == 0:
            # Fallback: if actor has no module sequence, try to call the actor directly
            # (some TorchRL actor wrappers expose a callable module rather than a list).
            if hasattr(self.actor, "__call__"):
                # call actor directly and hope it produces 'logits'
                self.actor(td_clean)
            else:
                raise ValueError("Actor must expose a .module sequence with a logits producer at [0] or be callable")

        else:
            # Call logits producer (in-place on td_clean view)
            try:
                # Some wrappers expose a sequence-like .module (indexable)
                self.actor.module[0](td_clean)
            except Exception:
                # Other wrappers expose .module as a callable object
                try:
                    self.actor.module(td_clean)
                except Exception as exc:
                    raise RuntimeError("Failed to call actor module to produce logits") from exc

        logits = td_clean.get("logits")
        if logits is None:
            raise ValueError("Actor module failed to produce 'logits'")

        action_mask = td_clean.get("action_mask")
        if action_mask is None:
            raise ValueError("Missing 'action_mask' in tensordict")

        # Snapshot the mask to avoid races with async workers.
        mask = action_mask.clone().to(logits.device).bool()
        if mask.dim() == 1 and logits.dim() == 2:
            mask = mask.unsqueeze(0)

        masked_logits = logits.masked_fill(~mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        td.set("action", action)
        td.set("sample_log_prob", log_prob)
        if self.debug:
            td.set("_action_mask_at_sample_time", action_mask)

        return td

class MultiSyncRolloutCollector:
    """Persistent collector using MultiSyncDataCollector.
    
    This version uses multiple sub-collectors that run in parallel, potentially
    providing better parallelism than SyncDataCollector for environments with
    high variability in step times.
    
    Note: MultiSyncDataCollector creates multiple independent data collectors,
    each managing a subset of environments. This can improve throughput when
    environments have variable execution times.
    """
    
    def __init__(
        self,
        env,
        actor: torch.nn.Module,
        n_envs: int,
        n_steps: int,
        device: torch.device,
        num_sub_collectors: int = 2,
        debug: bool = False,
    ):
        """Initialize persistent MultiSync collector.
        
        Args:
            env: ParallelEnv or single environment (will be replicated)
            actor: Policy network
            n_envs: Total number of parallel environments
            n_steps: Steps per rollout
            device: Device for computation
            num_sub_collectors: Number of sub-collectors (each gets env/num_sub_collectors envs)
            debug: Enable debug logging
        """
        from torchrl.envs import ExplorationType
        
        self.env = env
        self.actor = actor
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        self.debug = debug
        self.num_sub_collectors = num_sub_collectors
        
        frames_per_batch = int(n_envs) * int(n_steps)
        masked_policy = MaskedPolicyWrapper(actor, debug=debug)
        
        if debug:
            print(f"[MultiSyncRolloutCollector] Creating persistent collector")
            print(f"  n_envs: {n_envs}, n_steps: {n_steps}")
            print(f"  frames_per_batch: {frames_per_batch}")
            print(f"  num_sub_collectors: {num_sub_collectors}")
        
        # MultiSyncDataCollector with num_workers creates multiple sub-collectors
        # that each manage a portion of the environments. This can provide better
        # parallelism than SyncDataCollector for variable-time environments.
        
        # Extract environment creators from ParallelEnv
        # MultiSyncDataCollector needs the actual env creator functions, not the ParallelEnv
        if isinstance(env, ParallelEnv) and hasattr(env, 'env_fns'):
            create_env_fn = env.env_fns
            if debug:
                print(f"  Using {len(create_env_fn)} environment creators from ParallelEnv")
        elif isinstance(env, (list, tuple)):
            # Already a list of env creators
            create_env_fn = env
            if debug:
                print(f"  Using {len(create_env_fn)} environment creators (list)")
        else:
            # If not a ParallelEnv with env_fns, use env directly
            create_env_fn = env
            if debug:
                print(f"  Using environment directly (not ParallelEnv)")
        
        # MultiSyncDataCollector expects num_workers to match the number of env_fns
        # Each worker gets one env_fn. So we set num_workers = len(env_fns)
        if isinstance(create_env_fn, (list, tuple)):
            actual_num_workers = len(create_env_fn)
            if debug:
                print(f"  Setting num_workers={actual_num_workers} to match number of env_fns")
                if actual_num_workers != num_sub_collectors:
                    print(f"  Note: Requested {num_sub_collectors} sub-collectors, but using {actual_num_workers} to match env_fns")
        else:
            actual_num_workers = num_sub_collectors
            if debug:
                print(f"  Using {actual_num_workers} sub-collectors")
        
        self.collector = MultiSyncDataCollector(
            create_env_fn=create_env_fn,
            policy=masked_policy,
            frames_per_batch=frames_per_batch,
            total_frames=-1,  # Infinite collection (we'll reuse this)
            device=device,
            storing_device='cpu',
            split_trajs=False,
            exploration_type=ExplorationType.RANDOM,
            init_random_frames=-1,
            num_workers=actual_num_workers,  # Must match number of env_fns
        )
        self._iterator = iter(self.collector)
        
        if debug:
            print(f"[MultiSyncRolloutCollector] Persistent collector created and ready")
    
    def collect(self, critic: torch.nn.Module, rollout_callback: Optional[Callable] = None) -> Tuple[List[TensorDict], Dict[str, Any]]:
        """Collect one rollout batch using the persistent collector.
        
        Args:
            critic: Critic network for value estimation
            rollout_callback: Optional callback for each step
            
        Returns:
            experiences: List of TensorDicts (one per step)
            stats: Dictionary with episode_info
        """
        from torchrl.envs import set_exploration_type, ExplorationType
        
        with set_exploration_type(ExplorationType.RANDOM):
            batch_td = next(self._iterator)
        
        # Add value estimates
        critic(batch_td)
        
        # Reshape and convert to list format
        batch_td_time = _reshape_time_env(batch_td, n_steps=self.n_steps, n_envs=self.n_envs)
        experiences: List[TensorDict] = []
        for t in range(self.n_steps):
            step_td = TensorDict({
                "sub_index": batch_td_time[t]["sub_index"],
                "derived_sub_indices": batch_td_time[t]["derived_sub_indices"],
                "action_mask": batch_td_time[t]["action_mask"],
                "action": batch_td_time[t]["action"],
                "sample_log_prob": batch_td_time[t].get("sample_log_prob"),
                "state_value": batch_td_time[t].get("state_value"),
                "next": TensorDict({
                    "sub_index": batch_td_time[t]["next"]["sub_index"],
                    "derived_sub_indices": batch_td_time[t]["next"]["derived_sub_indices"],
                    "action_mask": batch_td_time[t]["next"]["action_mask"],
                    "reward": batch_td_time[t]["next"]["reward"],
                    "done": batch_td_time[t]["next"]["done"],
                }, batch_size=[self.n_envs]),
            }, batch_size=[self.n_envs])
            experiences.append(step_td)
            
            if rollout_callback is not None:
                rollout_callback(t)
        
        # Extract episode stats
        episode_info = _extract_episode_infos(
            batch_td, n_steps=self.n_steps, n_envs=self.n_envs,
            device=self.device, verbose=False
        )
        stats = {"episode_info": episode_info}
        
        return experiences, stats
    
    def shutdown(self):
        """Clean up the collector."""
        if hasattr(self, 'collector'):
            self.collector.shutdown()
