
# ppo_rollout.py — fast rollout via TorchRL SyncDataCollector
# This file uses a masked policy wrapper to ensure action_mask is always respected.

from __future__ import annotations

import sys
print("="*80, flush=True)
print("DEBUG: ppo_rollout.py MODULE IS BEING IMPORTED/RELOADED", flush=True)
print("="*80, flush=True)
sys.stdout.flush()

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ParallelEnv


class MaskedPolicyWrapper:
    """
    Serializable policy wrapper that re-applies action_mask before sampling.
    
    This ensures that the data collector never samples invalid (padded) actions,
    even if the actor's internal distribution doesn't properly handle -inf masking.
    
    This class can be pickled for multiprocessing, unlike nested functions.
    """
    
    def __init__(self, actor):
        """Initialize the wrapper with an actor module.
        
        Args:
            actor: The ProbabilisticActor module
        """
        self.actor = actor
        self.call_count = 0
        self.previous_td_id = None
        self.previous_masks = {}
    
    @torch.no_grad()
    def __call__(self, td: TensorDict) -> TensorDict:
        """Apply the masked policy to the given tensordict.
        
        Args:
            td: Input tensordict from environment
            
        Returns:
            Updated tensordict with action and sample_log_prob
        """
        self.call_count += 1
        log_this = False  # Disable for cleaner output
        
        td_id = id(td)
        is_same_td = td_id == self.previous_td_id
        self.previous_td_id = td_id
        
        if log_this:
            print(f"\n{'='*80}")
            print(f"[MaskedPolicyWrapper call {self.call_count}] TD object id: {td_id} {'(SAME AS PREVIOUS!)' if is_same_td else '(new)'}")
            print(f"  Input TD keys: {list(td.keys())}")
            print(f"  Batch size: {td.batch_size}")
            print(f"  TD device: {td.device if hasattr(td, 'device') else 'N/A'}")
            
            # Check if this is a view or shared data
            if 'action_mask' in td.keys():
                mask_tensor = td.get('action_mask')
                print(f"  action_mask tensor id: {id(mask_tensor)}")
                print(f"  action_mask is_contiguous: {mask_tensor.is_contiguous()}")
                print(f"  action_mask storage id: {id(mask_tensor.storage()) if hasattr(mask_tensor, 'storage') else 'N/A'}")
        
        # CRITICAL: Check for stale/cached action_mask
        action_mask_input = td.get("action_mask", None)
        if action_mask_input is not None and log_this:
            print(f"\n  [INITIAL ACTION_MASK CHECK]")
            print(f"  Action mask shape: {action_mask_input.shape}")
            
            # Handle both batched [batch, actions] and unbatched [actions] masks
            if action_mask_input.dim() == 1:
                # Single environment, unbatched
                valid_count = action_mask_input.sum().item()
                valid_indices = torch.where(action_mask_input)[0].tolist()
                
                prev_mask = self.previous_masks.get(0)
                mask_changed = "NEW" if prev_mask is None else ("CHANGED" if not torch.equal(prev_mask, action_mask_input) else "SAME")
                self.previous_masks[0] = action_mask_input.clone()
                
                print(f"    Single env: {valid_count} valid actions {valid_indices} [{mask_changed}]")
            else:
                # Multiple environments, batched
                for env_idx in range(min(action_mask_input.shape[0], 4)):  # Only show first 4 envs
                    valid_count = action_mask_input[env_idx].sum().item()
                    valid_indices = torch.where(action_mask_input[env_idx])[0].tolist()
                    
                    prev_mask = self.previous_masks.get(env_idx)
                    mask_changed = "NEW" if prev_mask is None else ("CHANGED" if not torch.equal(prev_mask, action_mask_input[env_idx]) else "SAME")
                    self.previous_masks[env_idx] = action_mask_input[env_idx].clone()
                    
                    print(f"    Env {env_idx}: {valid_count} valid actions {valid_indices} [{mask_changed}]")
        
        # Remove any existing action/logits keys to force recomputation
        keys_to_remove = []
        for key in ['logits', 'action', 'sample_log_prob']:
            if key in td.keys():
                keys_to_remove.append(key)
                if log_this:
                    print(f"  Removing existing key: {key}")
        
        if keys_to_remove:
            # Create a new TD without these keys
            td_clean = td.exclude(*keys_to_remove)
        else:
            td_clean = td
        
        # 1) Run the actor's logits-producing module without sampling
        #    ProbabilisticActor usually wraps a TensorDictModule in .module[0]
        
        if log_this:
            print(f"\n  [ACTOR STRUCTURE]")
            print(f"  Actor type: {type(self.actor)}")
            print(f"  Has 'module': {hasattr(self.actor, 'module')}")
        
        # Check actor structure
        if hasattr(self.actor, "module"):
            # actor.module is a TensorDictSequential containing the logits module
            if len(self.actor.module) > 0:
                if log_this:
                    print(f"  Actor module length: {len(self.actor.module)}")
                    print(f"  Actor module[0] type: {type(self.actor.module[0])}")
                # Call just the first module (the logits producer)
                self.actor.module[0](td_clean)
            else:
                raise ValueError("Actor module is empty")
        else:
            raise ValueError(f"Actor has no 'module' attribute. Actor type: {type(self.actor)}")
        
        # 2) Get logits and action_mask from the tensordict
        logits = td_clean.get("logits")
        if logits is None:
            raise ValueError(f"Actor module did not produce 'logits' key. Available keys: {list(td_clean.keys())}")
        
        action_mask = td_clean.get("action_mask")
        if action_mask is None:
            raise ValueError(f"TensorDict missing 'action_mask' key. Available keys: {list(td_clean.keys())}")
        
        # CRITICAL FIX: Clone the action_mask to prevent it from being modified
        # by parallel workers between sampling and stepping.
        # This is the root cause of invalid actions - the mask changes after we sample!
        action_mask_snapshot = action_mask.clone()
        
        if log_this:
            print(f"\n  [AFTER ACTOR FORWARD]")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits device: {logits.device}")
            print(f"  Action mask shape: {action_mask.shape}")
            print(f"  Action mask device: {action_mask.device}")
            print(f"  Action mask tensor id: {id(action_mask)} -> cloned to {id(action_mask_snapshot)}")
            
            # CRITICAL: Check if action_mask changed after actor forward
            if action_mask_input is not None and not torch.equal(action_mask, action_mask_input):
                print(f"  WARNING: action_mask CHANGED after actor forward!")
                for env_idx in range(min(action_mask.shape[0], 3)):
                    if not torch.equal(action_mask[env_idx], action_mask_input[env_idx]):
                        print(f"    Env {env_idx} BEFORE: {torch.where(action_mask_input[env_idx])[0].tolist()}")
                        print(f"    Env {env_idx} AFTER:  {torch.where(action_mask[env_idx])[0].tolist()}")
        
        # 3) Re-apply the mask to ensure -inf for invalid actions
        # Use the cloned snapshot to ensure consistency
        # Handle both batched [batch, actions] and unbatched [actions] masks
        mask = action_mask_snapshot.to(logits.device).bool()
        
        # Ensure mask has the same shape as logits
        if mask.dim() == 1 and logits.dim() == 2:
            # Unbatched mask [actions] -> broadcast to [1, actions]
            mask = mask.unsqueeze(0)
        elif mask.dim() == 2 and logits.dim() == 2:
            # Already batched [batch, actions]
            pass
        else:
            raise ValueError(f"Unexpected shapes: mask {mask.shape}, logits {logits.shape}")
        
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        
        if log_this:
            print(f"\n  [MASKING VERIFICATION]")
            # Verify masking worked
            batch_size = masked_logits.shape[0]
            for env_idx in range(min(batch_size, 3)):  # Only check actual batch elements
                finite_count = torch.isfinite(masked_logits[env_idx]).sum().item()
                expected_count = mask[env_idx].sum().item()
                finite_indices = torch.where(torch.isfinite(masked_logits[env_idx]))[0].tolist()
                
                if finite_count != expected_count:
                    print(f"    Env {env_idx}: MASKING FAILED! {finite_count} finite but {expected_count} expected")
                    print(f"      Valid logit values: {masked_logits[env_idx][torch.isfinite(masked_logits[env_idx])].tolist()}")
                else:
                    print(f"    Env {env_idx}: OK - {finite_count} finite logits at {finite_indices}")
        
        # 4) Create distribution and sample action
        if log_this:
            print(f"\n  [SAMPLING]")
        
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if log_this:
            print(f"  Sampled actions shape: {action.shape}")
            print(f"  Sampled actions: {action.tolist() if action.dim() > 0 else [action.item()]}")
            
            # Verify ALL actions are valid ACCORDING TO THE SNAPSHOT
            print(f"\n  [ACTION VALIDATION]")
            batch_size = action.shape[0] if action.dim() > 0 else 1
            for i in range(min(batch_size, 3)):  # Only show first 3
                act_idx = action[i].item() if action.dim() > 0 else action.item()
                
                # Handle both batched and unbatched masks
                if mask.dim() == 1:
                    # Unbatched: mask shape [actions]
                    is_valid = mask[act_idx].item()
                    valid_indices = torch.where(mask)[0].tolist()
                else:
                    # Batched: mask shape [batch, actions]
                    is_valid = mask[i][act_idx].item()
                    valid_indices = torch.where(mask[i])[0].tolist()
                
                status = "✓ VALID" if is_valid else "✗ INVALID"
                print(f"    Env {i}: action={act_idx} {status} (valid: {valid_indices})")
                
                if not is_valid:
                    print(f"    !! CRITICAL ERROR !!")
                    print(f"      Env {i} masked logits: {masked_logits[i].tolist()}")
                    print(f"      Env {i} action_mask_snapshot: {action_mask_snapshot[i].tolist()}")
                    print(f"      Env {i} finite logits at: {torch.where(torch.isfinite(masked_logits[i]))[0].tolist()}")
                    print(f"      Distribution probs: {dist.probs[i].tolist()}")
                    
                    # This should NEVER happen - raise error immediately
                    raise RuntimeError(
                        f"Masked policy sampled INVALID action {act_idx} for env {i}! "
                        f"Valid actions: {valid_indices}. This indicates a fundamental "
                        f"issue with the masking or sampling logic."
                    )
        
        # 5) Write action and log_prob back to tensordict
        # Also store the action_mask_snapshot so we can verify it wasn't modified
        td.set("action", action)
        td.set("sample_log_prob", log_prob)
        td.set("_action_mask_at_sample_time", action_mask_snapshot)  # For debugging
        
        if log_this:
            print(f"\n  [FINAL STATE]")
            print(f"  TD keys after policy: {list(td.keys())}")
            print(f"{'='*80}\n")
        
        return td


def _masked_policy_factory(actor):
    """
    Create a serializable masked policy wrapper.
    
    This factory function creates a MaskedPolicyWrapper instance that can be pickled
    for use with data collectors.
    
    Args:
        actor: The ProbabilisticActor module
        
    Returns:
        A MaskedPolicyWrapper instance that always respects the action_mask
    """
    return MaskedPolicyWrapper(actor)


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
    debug=True  # Disable debug for cleaner output
    if debug:
        print(f"\n{'='*80}")
        print(f"[collect_rollouts] Starting collection")
        print(f"  n_envs: {n_envs}, n_steps: {n_steps}")
        print(f"  frames_per_batch: {n_envs * n_steps}")
        print(f"  Environment type: {type(env)}")
        print(f"  Environment batch_size: {env.batch_size if hasattr(env, 'batch_size') else 'N/A'}")
        print(f"  Is ParallelEnv: {isinstance(env, ParallelEnv)}")
    
    frames_per_batch = int(n_envs) * int(n_steps)

    # Create a masked policy wrapper that ensures action_mask is always respected
    masked_policy = _masked_policy_factory(actor)
    if debug:
        print(f"\n[collect_rollouts] Created masked policy wrapper")
        print(f"  Policy type: {type(masked_policy)}")

    # SyncDataCollector can work directly with an already-created environment
    # No need to wrap in a lambda or pass factories
    if debug:
        print(f"\n[collect_rollouts] Setting up SyncDataCollector")
        print(f"  Total environments: {n_envs}")
        print(f"  Frames per batch: {frames_per_batch}")
        print(f"  Passing environment directly (not as factory)")

    print(f"Running SyncDataCollector with device: {device}")
    collector = SyncDataCollector(
        create_env_fn=env,  # Pass the environment directly
        policy=masked_policy,  # Use masked policy instead of raw actor
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch,
        device=device,
        storing_device=device,  # Store on CPU to avoid GPU OOM with large batches
        split_trajs=False,
        exploration_type=ExplorationType.RANDOM,
        init_random_frames=-1,  # Don't collect random frames at start
        use_buffers=False,  # Disable pre-allocated buffers due to dynamic specs
    )

    if debug:
        print(f"\n[collect_rollouts] Collector created, starting rollout...")
        print(f"  Collector type: {type(collector)}")
        if hasattr(collector, 'device'):
            print(f"  Collector device: {collector.device}")
        print("=" * 80 + "\n")

    # Wrap the rollout in RANDOM exploration mode context to ensure stochastic sampling
    try:
        with set_exploration_type(ExplorationType.RANDOM):
            if debug:
                print("[collect_rollouts] Starting batch collection...")
            batch_td = next(iter(collector))  # single batch
            if debug:
                print(f"[collect_rollouts] Batch collected successfully!")
                print(f"  Batch shape: {batch_td.batch_size}")
                print(f"  Batch keys: {list(batch_td.keys())}")
    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"[collect_rollouts] ERROR during collection:")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {str(e)}")
        print(f"{'!'*80}\n")
        raise
    finally:
        collector.shutdown()
        if debug:
            print("[collect_rollouts] Collector shutdown complete")


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