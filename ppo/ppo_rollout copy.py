
# ppo_rollout.py — fast rollout via TorchRL SyncDataCollector
# This file uses a masked policy wrapper to ensure action_mask is always respected.

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ParallelEnv


def _masked_policy_factory(actor):
    """
    Return a policy callable that re-applies action_mask before sampling.
    
    This ensures that the SyncDataCollector never samples invalid (padded) actions,
    even if the actor's internal distribution doesn't properly handle -inf masking.
    
    Args:
        actor: The ProbabilisticActor module
        
    Returns:
        A policy function that always respects the action_mask
    """
    call_count = [0]  # Mutable to track calls across invocations
    
    @torch.no_grad()
    def policy(td: TensorDict) -> TensorDict:
        call_count[0] += 1
        log_this = call_count[0] <= 20  # Log first 20 calls
        
        if log_this:
            print(f"\n[_masked_policy call {call_count[0]}]")
            print(f"  Input TD keys: {list(td.keys())}")
            print(f"  Batch size: {td.batch_size}")
        
        # Remove any existing action/logits keys to force recomputation
        keys_to_remove = []
        for key in ['logits', 'action', 'sample_log_prob']:
            if key in td.keys():
                keys_to_remove.append(key)
        
        if keys_to_remove:
            # Create a new TD without these keys
            td_clean = td.exclude(*keys_to_remove)
        else:
            td_clean = td
        
        # 1) Run the actor's logits-producing module without sampling
        #    ProbabilisticActor usually wraps a TensorDictModule in .module[0]
        
        # Check actor structure
        if hasattr(actor, "module"):
            # actor.module is a TensorDictSequential containing the logits module
            if len(actor.module) > 0:
                # Call just the first module (the logits producer)
                actor.module[0](td_clean)
            else:
                raise ValueError("Actor module is empty")
        else:
            raise ValueError(f"Actor has no 'module' attribute. Actor type: {type(actor)}")
        
        # 2) Get logits and action_mask from the tensordict
        logits = td_clean.get("logits")
        if logits is None:
            raise ValueError(f"Actor module did not produce 'logits' key. Available keys: {list(td_clean.keys())}")
        
        action_mask = td_clean.get("action_mask")
        if action_mask is None:
            raise ValueError(f"TensorDict missing 'action_mask' key. Available keys: {list(td_clean.keys())}")
        
        if log_this:
            print(f"  Logits shape: {logits.shape}")
            print(f"  Action mask shape: {action_mask.shape}")
            # Check all environments in batch
            for env_idx in range(min(action_mask.shape[0], 16)):
                valid_count = action_mask[env_idx].sum().item()
                valid_indices = torch.where(action_mask[env_idx])[0].tolist()
                print(f"  Env {env_idx}: {valid_count} valid actions - {valid_indices}")
        
        # 3) Re-apply the mask to ensure -inf for invalid actions
        mask = action_mask.to(logits.device).bool()
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        
        if log_this:
            # Verify masking worked
            for env_idx in range(min(masked_logits.shape[0], 3)):
                finite_count = torch.isfinite(masked_logits[env_idx]).sum().item()
                expected_count = action_mask[env_idx].sum().item()
                if finite_count != expected_count:
                    print(f"  WARNING: Env {env_idx} masking failed! {finite_count} finite logits but {expected_count} valid actions")
        
        # 4) Create distribution and sample action
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if log_this:
            print(f"  Sampled actions: {action.tolist() if action.dim() > 0 else [action.item()]}")
            # Verify ALL actions are valid
            for i in range(action.shape[0] if action.dim() > 0 else 1):
                act_idx = action[i].item() if action.dim() > 0 else action.item()
                is_valid = action_mask[i][act_idx].item() if action_mask.dim() > 1 else action_mask[act_idx].item()
                if not is_valid:
                    valid_indices = torch.where(action_mask[i])[0].tolist()
                    print(f"    Env {i}: action={act_idx} INVALID! Valid actions: {valid_indices}")
                    print(f"    Env {i} masked logits: {masked_logits[i].tolist()}")
                    print(f"    Env {i} action_mask: {action_mask[i].tolist()}")
        
        # 5) Write action and log_prob back to tensordict
        td.set("action", action)
        td.set("sample_log_prob", log_prob)
        
        return td
    
    return policy


def _reshape_time_env(td: TensorDict, n_steps: int, n_envs: int) -> TensorDict:
    """Reshape a flat [n_steps*n_envs] batch to [n_steps, n_envs]."""
    return td.reshape(n_steps, n_envs)


def _extract_episode_infos(batch_td: TensorDict, n_steps: int, n_envs: int, device: torch.device, verbose: bool = False) -> List[Dict[str, Any]]:
    """Lightweight episode stats extraction compatible with your callbacks."""
    infos: List[Dict[str, Any]] = []
    
    if verbose:
        print(f"[_extract_episode_infos] Batch TensorDict keys: {list(batch_td.keys())}")
        print(f"[_extract_episode_infos] Next keys: {list(batch_td.get('next', TensorDict({})).keys())}")
    
    done = batch_td.get(('next', 'done'))
    reward = batch_td.get(('next', 'reward'))
    label = batch_td.get(('next', 'label'), None)
    depth = batch_td.get(('next', 'query_depth'), None)
    success = batch_td.get(('next', 'is_success'), None)

    done = done.reshape(n_steps, n_envs).squeeze(-1).to(torch.bool)
    reward = reward.reshape(n_steps, n_envs).squeeze(-1)
    
    # Reshape label, depth, and success tensors if they exist
    if label is not None:
        label = label.reshape(n_steps, n_envs).squeeze(-1)
    if depth is not None:
        depth = depth.reshape(n_steps, n_envs).squeeze(-1)
    if success is not None:
        success = success.reshape(n_steps, n_envs).squeeze(-1)

    # For each env, accumulate until done (use the same device as reward)
    ep_ret = torch.zeros(n_envs, device=reward.device, dtype=reward.dtype)
    ep_len = torch.zeros(n_envs, device=reward.device, dtype=torch.long)
    
    if verbose:
        print(f"[_extract_episode_infos] Extracting from {n_steps} steps, {n_envs} envs")
        print(f"  label: {label is not None}, depth: {depth is not None}, success: {success is not None}")
    
    for t in range(n_steps):
        ep_ret += reward[t]
        ep_len += 1
        finished = done[t]
        if finished.any():
            idxs = torch.nonzero(finished, as_tuple=False).reshape(-1)
            if verbose:
                print(f"  Step {t}: {len(idxs)} episodes finished")
            for i in idxs.tolist():
                info = {
                    "episode": {"r": float(ep_ret[i].item()), "l": int(ep_len[i].item())}
                }
                if label is not None:
                    info["label"] = int(label[t, i].item())
                if depth is not None:
                    info["query_depth"] = int(depth[t, i].item())
                if success is not None:
                    info["is_success"] = bool(success[t, i].item())
                infos.append(info)
                if verbose:
                    print(f"    Env {i}: r={info['episode']['r']:.3f}, l={info['episode']['l']}, label={info.get('label', 'N/A')}")
                ep_ret[i] = 0.0
                ep_len[i] = 0
    
    if verbose:
        print(f"[_extract_episode_infos] Extracted {len(infos)} episode infos")
    return infos


@torch.inference_mode()
def collect_rollouts(
    env,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    n_envs: int,
    n_steps: int,
    device: torch.device,
    rollout_callback: Optional[Callable] = None,
) -> Tuple[List[TensorDict], Dict[str, Any]]:
    """Collect one batch (n_envs * n_steps) using SyncDataCollector.

    Returns
    -------
    experiences : List[TensorDict]
        Length n_steps; each TD has batch_size [n_envs] and contains:
          sub_index, derived_sub_indices, action_mask, action (one-hot),
          sample_log_prob, state_value, and a 'next' sub-TD with
          next/sub_index, next/derived_sub_indices, next/action_mask, next/reward, next/done.
    stats : Dict[str, Any]
        Contains 'episode_info' list for callbacks.
    """
    from torchrl.envs import set_exploration_type, ExplorationType
    
    frames_per_batch = int(n_envs) * int(n_steps)

    # Create a masked policy wrapper that ensures action_mask is always respected
    masked_policy = _masked_policy_factory(actor)

    # SyncDataCollector can accept either an env factory or an existing env.
    # If `env` is already a ParallelEnv, pass it directly to avoid double-batching
    # (returning a ParallelEnv from a create_env_fn would cause nested batching and shape mismatches).
    if isinstance(env, ParallelEnv):
        def create_env_fn():
            return env
        collector = SyncDataCollector(
            create_env_fn=create_env_fn,
            policy=masked_policy,  # Use masked policy instead of raw actor
            frames_per_batch=frames_per_batch,
            total_frames=frames_per_batch,
            device=device,
            storing_device='cpu',
            split_trajs=False,
            exploration_type=ExplorationType.RANDOM,
        )
    else:
        # SyncDataCollector requires a create_env_fn callable for single-worker envs
        def create_env_fn():
            return env

        collector = SyncDataCollector(
            create_env_fn=create_env_fn,
            policy=masked_policy,  # Use masked policy instead of raw actor
            frames_per_batch=frames_per_batch,
            total_frames=frames_per_batch,
            device=device,
            storing_device='cpu',
            split_trajs=False,
            exploration_type=ExplorationType.RANDOM,
        )

    # Wrap the rollout in RANDOM exploration mode context to ensure stochastic sampling
    with set_exploration_type(ExplorationType.RANDOM):
        batch_td = next(iter(collector))  # single batch
    collector.shutdown()

    # Add value estimates using the critic (operates on the observation at ROOT)
    critic(batch_td)  # adds 'state_value'

    # Reshape to [n_steps, n_envs] and split to list as expected by PPOAgent.learn
    batch_td_time = _reshape_time_env(batch_td, n_steps=n_steps, n_envs=n_envs)
    experiences: List[TensorDict] = []
    for t in range(n_steps):
        step_td = TensorDict({
            "sub_index": batch_td_time[t]["sub_index"],
            "derived_sub_indices": batch_td_time[t]["derived_sub_indices"],
            "action_mask": batch_td_time[t]["action_mask"],
            "action": batch_td_time[t]["action"],  # OneHotCategorical -> one-hot
            "sample_log_prob": batch_td_time[t].get("sample_log_prob"),
            "state_value": batch_td_time[t].get("state_value"),
            "next": TensorDict({
                "sub_index": batch_td_time[t]["next"]["sub_index"],
                "derived_sub_indices": batch_td_time[t]["next"]["derived_sub_indices"],
                "action_mask": batch_td_time[t]["next"]["action_mask"],
                "reward": batch_td_time[t]["next"]["reward"],
                "done": batch_td_time[t]["next"]["done"],
            }, batch_size=[n_envs]),
        }, batch_size=[n_envs])
        experiences.append(step_td)

        if rollout_callback is not None:
            rollout_callback(t)

    # Stats for callbacks
    episode_info = _extract_episode_infos(batch_td, n_steps=n_steps, n_envs=n_envs, device=device, verbose=False)
    stats = {
        "episode_info": episode_info,
    }
    return experiences, stats