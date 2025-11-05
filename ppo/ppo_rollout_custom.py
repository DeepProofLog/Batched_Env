from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict


class CustomRolloutCollector:
    """Custom rollout collector that doesn't use TorchRL collectors.
    
    This implementation manually steps through parallel environments and collects
    experiences, providing a baseline to compare against TorchRL's optimized collectors.
    """
    
    def __init__(
        self,
        env,
        actor: torch.nn.Module,
        n_envs: int,
        n_steps: int,
        device: torch.device,
        debug: bool = False,
    ):
        """Initialize custom collector.
        
        Args:
            env: ParallelEnv or environment
            actor: Policy network
            n_envs: Number of parallel environments
            n_steps: Steps per rollout
            device: Device for computation
            debug: Enable debug logging
        """
        self.env = env
        self.actor = actor
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        self.debug = debug
        
        if debug:
            print(f"[CustomRolloutCollector] Initialized")
            print(f"  n_envs: {n_envs}, n_steps: {n_steps}")
            print(f"  device: {device}")
    
    @torch.no_grad()
    def _select_action(self, obs_td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action using the actor policy.
        
        Args:
            obs_td: TensorDict with observation data
            
        Returns:
            Tuple of (action, log_prob)
        """
        # Get logits from actor
        # Call the actor's first module to get logits
        if hasattr(self.actor, "module") and len(self.actor.module) > 0:
            self.actor.module[0](obs_td)
        else:
            self.actor(obs_td)
        
        logits = obs_td.get("logits")
        if logits is None:
            raise ValueError("Actor failed to produce logits")
        
        action_mask = obs_td.get("action_mask")
        if action_mask is None:
            raise ValueError("Missing action_mask in observation")
        
        # Mask invalid actions
        mask = action_mask.clone().to(logits.device).bool()
        if mask.dim() == 1 and logits.dim() == 2:
            mask = mask.unsqueeze(0)
        
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        
        # Sample action
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def collect(self, critic: torch.nn.Module, rollout_callback: Optional[Callable] = None) -> Tuple[List[TensorDict], Dict[str, Any]]:
        """Collect one rollout batch by manually stepping through environments.
        
        Args:
            critic: Critic network for value estimation
            rollout_callback: Optional callback for each step
            
        Returns:
            experiences: List of TensorDicts (one per step)
            stats: Dictionary with episode_info
        """
        experiences: List[TensorDict] = []
        episode_infos: List[Dict[str, Any]] = []
        
        # Reset environments to get initial observations
        obs_td = self.env.reset()
        
        # Rollout loop
        for step in range(self.n_steps):
            # Select actions using policy
            action, log_prob = self._select_action(obs_td)
            
            # Create tensordict with current state and action
            step_td = TensorDict({
                "sub_index": obs_td["sub_index"].clone(),
                "derived_sub_indices": obs_td["derived_sub_indices"].clone(),
                "action_mask": obs_td["action_mask"].clone(),
                "action": action,
                "sample_log_prob": log_prob,
            }, batch_size=[self.n_envs])
            
            # Get value estimate from critic
            critic(step_td)
            
            # Step environments using step_and_maybe_reset
            # Returns (step_td_with_next, next_obs_td)
            # step_td_with_next contains the current step with reward/done in "next" key
            # next_obs_td contains the observation for the next step (already reset if episode ended)
            step_td_complete, next_obs_td = self.env.step_and_maybe_reset(step_td)
            
            # Debug: check what keys are in the returned tensordicts
            if step == 0 and self.debug:
                print(f"[DEBUG] step_td_complete keys: {step_td_complete.keys()}")
                if "next" in step_td_complete.keys():
                    print(f"[DEBUG] step_td_complete['next'] keys: {step_td_complete['next'].keys()}")
                print(f"[DEBUG] next_obs_td keys: {next_obs_td.keys()}")
            
            # The step_td_complete already has the "next" key with reward/done
            # We just need to copy that to our step_td
            if "next" in step_td_complete.keys():
                step_td.set("next", step_td_complete["next"])
            else:
                # Fallback: construct next from step_td_complete directly
                step_td.set("next", TensorDict({
                    "sub_index": step_td_complete["sub_index"].clone(),
                    "derived_sub_indices": step_td_complete["derived_sub_indices"].clone(),
                    "action_mask": step_td_complete["action_mask"].clone(),
                    "reward": step_td_complete["reward"].clone(),
                    "done": step_td_complete["done"].clone(),
                }, batch_size=[self.n_envs]))
            
            experiences.append(step_td)
            
            # Extract the next state data for episode info extraction
            # The next state info is in step_td["next"]
            next_state_data = step_td["next"]
            
            # Check for completed episodes and extract info
            done_mask = next_state_data["done"].squeeze(-1).bool()
            for env_idx in range(self.n_envs):
                if done_mask[env_idx].item():
                    # Episode completed - extract episode info if available
                    if "label" in next_state_data.keys():
                        info = {
                            "episode": {
                                "r": 0.0,  # Need to track cumulative reward separately
                                "l": 0,    # Need to track steps separately
                            }
                        }
                        # Add optional fields if present
                        info["label"] = int(next_state_data["label"][env_idx].item())
                        if "query_depth" in next_state_data.keys():
                            info["query_depth"] = int(next_state_data["query_depth"][env_idx].item())
                        if "is_success" in next_state_data.keys():
                            info["is_success"] = bool(next_state_data["is_success"][env_idx].item())
                        
                        episode_infos.append(info)
            
            # Update observation for next step
            # Use next_obs_td which already has the proper observation for next iteration
            obs_td = next_obs_td
            
            if rollout_callback is not None:
                rollout_callback(step)
        
        stats = {"episode_info": episode_infos}
        
        if self.debug:
            print(f"[CustomRolloutCollector] Collected {len(experiences)} steps, {len(episode_infos)} episodes")
        
        return experiences, stats
    
    def shutdown(self):
        """Clean up resources."""
        if hasattr(self.env, 'shutdown'):
            try:
                self.env.shutdown()
            except (AttributeError, EOFError) as e:
                # Ignore errors during shutdown - workers may already be dead
                pass


def _extract_episode_infos(batch_td: TensorDict, n_steps: int, n_envs: int, device: torch.device, verbose: bool = False) -> List[Dict[str, Any]]:
    """Extract episode infos from a batch tensordict.
    
    This is kept for compatibility but the custom collector extracts episode info during collection.
    """
    infos: List[Dict[str, Any]] = []
    
    done = batch_td.get(('next', 'done'))
    reward = batch_td.get(('next', 'reward'))
    label = batch_td.get(('next', 'label'), None)
    depth = batch_td.get(('next', 'query_depth'), None)
    success = batch_td.get(('next', 'is_success'), None)
    
    if done is None or reward is None:
        return infos
    
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
