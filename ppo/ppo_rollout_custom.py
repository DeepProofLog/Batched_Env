from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
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
        debug_action_space: bool = False,
    ):
        """Initialize custom collector.
        
        Args:
            env: ParallelEnv or environment
            actor: Policy network
            n_envs: Number of parallel environments
            n_steps: Steps per rollout
            device: Device for computation
            debug: Enable debug logging
            debug_action_space: Enable detailed action space diagnostics
        """
        self.env = env
        self.actor = actor
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        self.debug = debug
        self.debug_action_space = debug_action_space
        
        # Action space diagnostics
        if self.debug_action_space:
            self.action_space_stats = {
                'total_measurements': 0,
                'single_action_count': 0,
                'zero_action_count': 0,
                'action_distribution': {},
                'entropy_values': [],
            }
        
        if debug:
            print(f"[CustomRolloutCollector] Initialized")
            print(f"  n_envs: {n_envs}, n_steps: {n_steps}")
            print(f"  device: {device}")
            print(f"  debug_action_space: {debug_action_space}")
    
    @torch.no_grad()
    def _select_action(self, obs_td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action using the policy (actor).
        
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
        
        # ACTION SPACE DIAGNOSTICS
        if self.debug_action_space:
            self._diagnose_action_space(action_mask, obs_td)
        
        if self.debug:
            print(f"\n[DEBUG _select_action]")
            print(f"  action_mask shape: {action_mask.shape}")
            print(f"  action_mask:\n{action_mask}")
            valid_counts = action_mask.sum(dim=1)
            print(f"  valid_counts: {valid_counts.tolist()}")
        
        # Mask invalid actions
        mask = action_mask.clone().to(logits.device).bool()
        if mask.dim() == 1 and logits.dim() == 2:
            mask = mask.unsqueeze(0)
        
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        
        # Sample action
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Track entropy for diagnostics
        if self.debug_action_space:
            entropy = dist.entropy()
            self.action_space_stats['entropy_values'].extend(entropy.detach().cpu().tolist())
        
        if self.debug:
            print(f"  sampled actions: {action.tolist()}")
        
        # Verify actions are valid (safety check)
        if self.debug:
            valid_counts = mask.sum(dim=1)
            invalid_mask = action >= valid_counts
            if invalid_mask.any():
                print(f"\n[ERROR] Invalid action selected DURING action selection!")
                for i in range(len(action)):
                    if invalid_mask[i]:
                        print(f"  Env {i}: action={action[i].item()}, valid_range=0-{valid_counts[i].item()-1}")
                        print(f"    action_mask: {mask[i].tolist()}")
                        print(f"    masked_logits: {masked_logits[i].tolist()}")
                raise ValueError(f"Invalid actions selected after masking: {action[invalid_mask].tolist()}")
        
        return action, log_prob
    
    def _diagnose_action_space(self, action_mask: torch.Tensor, obs_td: TensorDict):
        """Diagnose action space constraints and collect statistics."""
        valid_counts = action_mask.sum(dim=1)  # [n_envs]
        
        # Update statistics
        self.action_space_stats['total_measurements'] += self.n_envs
        self.action_space_stats['single_action_count'] += (valid_counts == 1).sum().item()
        self.action_space_stats['zero_action_count'] += (valid_counts == 0).sum().item()
        
        # Track distribution
        for count in valid_counts.tolist():
            self.action_space_stats['action_distribution'][count] = \
                self.action_space_stats['action_distribution'].get(count, 0) + 1
        
        # Compute entropy (if logits available after actor call)
        # We'll store entropy after action selection
        
    def get_action_space_diagnostics(self) -> Dict[str, Any]:
        """Get action space diagnostic summary."""
        if not self.debug_action_space:
            return {}
        
        stats = self.action_space_stats
        total = stats['total_measurements']
        
        if total == 0:
            return {"message": "No measurements collected"}
        
        single_pct = 100.0 * stats['single_action_count'] / total
        zero_pct = 100.0 * stats['zero_action_count'] / total
        
        # Average actions
        avg_actions = sum(count * freq for count, freq in stats['action_distribution'].items()) / total
        
        # Median actions
        sorted_dist = sorted(stats['action_distribution'].items())
        cumsum = 0
        median_actions = 0
        for count, freq in sorted_dist:
            cumsum += freq
            if cumsum >= total / 2:
                median_actions = count
                break
        
        diagnostics = {
            'total_measurements': total,
            'single_action_count': stats['single_action_count'],
            'single_action_pct': single_pct,
            'zero_action_count': stats['zero_action_count'],
            'zero_action_pct': zero_pct,
            'avg_actions': avg_actions,
            'median_actions': median_actions,
            'action_distribution': dict(sorted_dist),
            'is_constrained': single_pct > 50.0,
            'has_zero_actions': zero_pct > 0,
        }
        
        if stats['entropy_values']:
            diagnostics['avg_entropy'] = np.mean(stats['entropy_values'])
            diagnostics['std_entropy'] = np.std(stats['entropy_values'])
        
        return diagnostics
    
    def print_action_space_diagnostics(self):
        """Print action space diagnostic summary."""
        if not self.debug_action_space:
            return
        
        diag = self.get_action_space_diagnostics()
        if 'message' in diag:
            print(f"\n[ACTION SPACE DIAGNOSTICS] {diag['message']}")
            return
        
        print("\n" + "="*80)
        print("ACTION SPACE DIAGNOSTICS")
        print("="*80)
        
        print(f"\nTotal measurements: {diag['total_measurements']}")
        print(f"Average actions per state: {diag['avg_actions']:.2f}")
        print(f"Median actions per state: {diag['median_actions']}")
        
        print(f"\n[CONSTRAINT ANALYSIS]")
        print(f"  Single-action states: {diag['single_action_count']} ({diag['single_action_pct']:.1f}%)")
        print(f"  Zero-action states: {diag['zero_action_count']} ({diag['zero_action_pct']:.1f}%)")
        
        if diag['is_constrained']:
            print(f"\n⚠️  WARNING: Action space is HEAVILY CONSTRAINED!")
            print(f"     Over 50% of states have only 1 action available.")
        
        if diag['has_zero_actions']:
            print(f"\n⚠️  WARNING: Zero-action states detected!")
            print(f"     This indicates a serious bug in action generation.")
        
        print(f"\n[ACTION DISTRIBUTION]")
        dist = diag['action_distribution']
        for n_actions in sorted(dist.keys())[:10]:  # Show top 10
            count = dist[n_actions]
            pct = 100.0 * count / diag['total_measurements']
            bar = "█" * int(pct / 2)  # Scale to 50 chars max
            print(f"  {n_actions:3d} actions: {count:6d} ({pct:5.1f}%) {bar}")
        
        if len(dist) > 10:
            print(f"  ... and {len(dist) - 10} more")
        
        if 'avg_entropy' in diag:
            print(f"\n[ENTROPY ANALYSIS]")
            print(f"  Average entropy: {diag['avg_entropy']:.4f}")
            print(f"  Std entropy: {diag['std_entropy']:.4f}")
            if diag['avg_entropy'] < 0.1:
                print(f"  ⚠️  Very low entropy - policy is extremely confident/deterministic")
        
        print("="*80 + "\n")

    
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
            if self.debug: print(f"\n[CustomRolloutCollector] Step {step+1}/{self.n_steps}", end="\r")
            # print(f"\n[CustomRolloutCollector] Step {step+1}/{self.n_steps}", end="\r")
            # Select actions using policy
            action, log_prob = self._select_action(obs_td)
            
            # Extract logits from obs_td (was set by _select_action)
            logits = obs_td.get("logits")
            
            if step == 0 and self.debug:
                print(f"[DEBUG collect] Step 0: logits present? {logits is not None}")
                if logits is not None:
                    print(f"  logits shape: {logits.shape}")
                action_mask = obs_td.get("action_mask")
                if action_mask is not None:
                    num_valid = action_mask.sum(dim=-1)
                    print(f"  num valid actions - min: {num_valid.min()}, max: {num_valid.max()}, mean: {num_valid.float().mean():.2f}")
                    print(f"  num envs with >1 valid action: {(num_valid > 1).sum()}/{len(num_valid)}")
            
            # Create tensordict with current state and action
            step_td = TensorDict({
                "sub_index": obs_td["sub_index"].clone(),
                "derived_sub_indices": obs_td["derived_sub_indices"].clone(),
                "action_mask": obs_td["action_mask"].clone(),
                "action": action,
                "sample_log_prob": log_prob,
                "logits": logits.clone() if logits is not None else None,  # Store logits for entropy computation
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
            done_mask = next_state_data["done"].squeeze(-1).to(torch.bool)
            if done_mask.any() and "label" in next_state_data.keys():
                label_tensor = next_state_data["label"].squeeze(-1)
                depth_tensor = next_state_data.get("query_depth", None)
                success_tensor = next_state_data.get("is_success", None)

                label_vals = label_tensor[done_mask].detach().cpu().tolist()
                depth_vals = (
                    depth_tensor.squeeze(-1)[done_mask].detach().cpu().tolist()
                    if depth_tensor is not None else None
                )
                success_vals = (
                    success_tensor.squeeze(-1)[done_mask].detach().cpu().tolist()
                    if success_tensor is not None else None
                )

                for i, lbl in enumerate(label_vals):
                    info = {"episode": {"r": 0.0, "l": 0}, "label": int(lbl)}
                    if depth_vals is not None:
                        info["query_depth"] = int(depth_vals[i])
                    if success_vals is not None:
                        info["is_success"] = bool(success_vals[i])
                    episode_infos.append(info)
            
            # Update observation for next step
            # Extract only observation keys from next_obs_td (it may have action/value data too)
            obs_td = TensorDict({
                "sub_index": next_obs_td["sub_index"],
                "derived_sub_indices": next_obs_td["derived_sub_indices"],
                "action_mask": next_obs_td["action_mask"],
            }, batch_size=[self.n_envs])
            
            if rollout_callback is not None:
                rollout_callback(step)
        
        stats = {"episode_info": episode_infos}
        
        if self.debug:
            print(f"[CustomRolloutCollector] Collected {len(experiences)} steps, {len(episode_infos)} episodes")
        
        # Print action space diagnostics if enabled
        if self.debug_action_space:
            self.print_action_space_diagnostics()
        
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
    
    done = done.reshape(n_steps, n_envs).squeeze(-1)
    reward = reward.reshape(n_steps, n_envs).squeeze(-1)
    
    label = label.reshape(n_steps, n_envs).squeeze(-1) if label is not None else None
    depth = depth.reshape(n_steps, n_envs).squeeze(-1) if depth is not None else None
    success = success.reshape(n_steps, n_envs).squeeze(-1) if success is not None else None
    
    done_np = done.detach().cpu().numpy().astype(bool)
    reward_np = reward.detach().cpu().numpy()
    csum_np = np.cumsum(reward_np, axis=0)
    label_np = label.detach().cpu().numpy() if label is not None else None
    depth_np = depth.detach().cpu().numpy() if depth is not None else None
    success_np = success.detach().cpu().numpy() if success is not None else None
    
    # For each environment, find time indices where an episode finished
    for env in range(n_envs):
        done_indices = np.nonzero(done_np[:, env])[0]
        if done_indices.size == 0:
            continue
        prev_t = -1
        for t in done_indices.tolist():
            if prev_t < 0:
                ep_ret = float(csum_np[t, env])
            else:
                ep_ret = float(csum_np[t, env] - csum_np[prev_t, env])
            ep_len = int(t - prev_t)
            
            info: Dict[str, Any] = {"episode": {"r": ep_ret, "l": ep_len}}
            if label_np is not None:
                info["label"] = int(label_np[t, env])
            if depth_np is not None:
                info["query_depth"] = int(depth_np[t, env])
            if success_np is not None:
                info["is_success"] = bool(success_np[t, env])
            
            infos.append(info)
            prev_t = t
    
    if verbose:
        print(f"[_extract_episode_infos] Extracted {len(infos)} episode infos from {n_envs} envs")
    return infos
