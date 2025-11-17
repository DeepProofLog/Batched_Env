from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict


class RolloutCollector:
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
        verbose: int = 0,
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
        self.verbose = max(int(verbose), 1 if debug else 0)
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

    def _vlog(self, level: int, message: str) -> None:
        if self.verbose >= level:
            print(f"[CustomRolloutCollector] {message}")
    
    @torch.no_grad()
    def _select_action(self, obs_td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action using the policy (actor).
        
        Args:
            obs_td: TensorDict with observation data
            
        Returns:
            Tuple of (action, log_prob)
        """
        actor_input = obs_td.clone().to(self.device)
        actor_out = self.actor(actor_input)
        if isinstance(actor_out, TensorDict):
            action_td = actor_out
        else:
            action_td = actor_input
        logits = action_td.get("logits")
        if logits is not None:
            obs_td.set("logits", logits.to(self.env._device))
        action_mask = obs_td.get("action_mask")
        if action_mask is None:
            raise ValueError("Missing action_mask in observation")
        
        # ACTION SPACE DIAGNOSTICS
        if self.debug_action_space:
            self._diagnose_action_space(action_mask, obs_td)
        
        # Use actor-provided action if available
        env_device = self.env._device
        if "action" in action_td.keys():
            action = action_td.get("action").long()
            log_prob = None
            for key in ("sample_log_prob", "log_prob"):
                if key in action_td.keys():
                    log_prob = action_td.get(key)
                    if log_prob.dim() > 1:
                        log_prob = log_prob.squeeze(-1)
                    break
            if log_prob is None:
                raise ValueError("Actor returned actions without log probabilities.")
            action = action.to(env_device)
            log_prob = log_prob.to(env_device)
            if self.verbose >= 2:
                self._vlog(2, f"_select_action actor_output action={action.tolist()}")
            return action, log_prob
        
        if logits is None:
            raise ValueError("Actor failed to produce logits for sampling")
        
        if self.verbose >= 2:
            self._vlog(2, f"_select_action logits shape={logits.shape}")
        
        mask = action_mask.clone().to(logits.device).bool()
        if mask.dim() == 1 and logits.dim() == 2:
            mask = mask.unsqueeze(0)
        
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if self.debug_action_space:
            entropy = dist.entropy()
            self.action_space_stats['entropy_values'].extend(entropy.detach().cpu().tolist())
        
        if self.verbose >= 2:
            self._vlog(2, f"_select_action sampled actions={action.tolist()}")
        
        valid_counts = mask.sum(dim=1)
        invalid_mask = action >= valid_counts
        if invalid_mask.any():
            raise ValueError(f"Invalid actions selected: {action[invalid_mask].tolist()}")
        action = action.to(env_device)
        log_prob = log_prob.to(env_device)
        
        return action, log_prob
    
    def _diagnose_action_space(self, action_mask: torch.Tensor, obs_td: TensorDict):
        """Diagnose action space constraints and collect statistics."""
        valid_counts = action_mask.sum(dim=1)  # [n_envs]
        
        # Update statistics (keep on GPU to avoid transfers)
        self.action_space_stats['total_measurements'] += self.n_envs
        self.action_space_stats['single_action_count'] += int((valid_counts == 1).sum())
        self.action_space_stats['zero_action_count'] += int((valid_counts == 0).sum())
        
        # Track distribution - only transfer unique counts
        # Use bincount for efficient histogram computation on GPU
        max_count = int(valid_counts.max()) if valid_counts.numel() > 0 else 0
        if max_count >= 0:
            # Create histogram on GPU then transfer once
            hist = torch.bincount(valid_counts, minlength=max_count + 1)
            for count_val, freq in enumerate(hist.tolist()):
                if freq > 0:
                    self.action_space_stats['action_distribution'][count_val] = \
                        self.action_space_stats['action_distribution'].get(count_val, 0) + freq
        
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
        running_returns = torch.zeros(self.n_envs, dtype=torch.float32, device=self.env._device)
        running_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=self.env._device)

        # Batch buffers for completed episodes to minimize cpu() synchronizations
        completed_rewards: List[torch.Tensor] = []
        completed_lengths: List[torch.Tensor] = []
        completed_labels: List[torch.Tensor] = []
        completed_depths: List[torch.Tensor] = []
        completed_success: List[torch.Tensor] = []
        labels_available = True
        depths_available = True
        success_available = True

        # Reset environments to get initial observations
        obs_td = self.env.reset()

        # Rollout loop
        for step in range(self.n_steps):
            self._vlog(2, f"Step {step+1}/{self.n_steps}")
            # Select actions using policy
            action, log_prob = self._select_action(obs_td)

            # Extract logits from obs_td (was set by _select_action)
            logits = obs_td.get("logits")

            if step == 0 and self.verbose >= 2:
                self._vlog(2, f"Step 0 logits present? {logits is not None}")
                if logits is not None:
                    self._vlog(2, f"  logits shape: {logits.shape}")
                action_mask = obs_td.get("action_mask")
                if action_mask is not None:
                    num_valid = action_mask.sum(dim=-1)
                    self._vlog(2, f"  valid actions stats min={num_valid.min()}, max={num_valid.max()}, mean={num_valid.float().mean():.2f}")

            # Create tensordict with current state and action
            step_td = TensorDict({
                "sub_index": obs_td["sub_index"].clone(),
                "derived_sub_indices": obs_td["derived_sub_indices"].clone(),
                "action_mask": obs_td["action_mask"].clone(),
                "action": action,
                "sample_log_prob": log_prob,
            }, batch_size=[self.n_envs])
            if logits is not None:
                step_td.set("logits", logits.clone())

            # Get value estimate from critic (on its device)
            critic_input = step_td.clone().to(self.device)
            critic_out = critic(critic_input)
            if "state_value" in critic_out.keys():
                step_td.set("state_value", critic_out["state_value"].to(self.env._device))

            # Step environments using step_and_maybe_reset (TorchRL method)
            # This calls step() and automatically resets done environments
            # Returns (step_result_td, next_obs_td)
            # - step_result_td: contains current step data with "next" key (reward, done, next obs)
            # - next_obs_td: observation for next iteration (already reset if done)
            step_td_with_next, next_obs_td = self.env.step_and_maybe_reset(step_td)

            # Debug: check what keys are in the returned tensordicts
            if step == 0 and self.verbose >= 2:
                self._vlog(2, f"step_td_with_next keys: {step_td_with_next.keys()}")
                if "next" in step_td_with_next.keys():
                    next_data = step_td_with_next["next"]
                    self._vlog(2, f"step_td_with_next['next'] keys: {next_data.keys()}")
                    if "reward" in next_data.keys():
                        self._vlog(2, f"  reward shape: {next_data['reward'].shape}, sample: {next_data['reward'][:2]}")
                self._vlog(2, f"next_obs_td keys: {next_obs_td.keys()}")

            # Copy the step result to our step_td (preserves current obs + action, adds next with reward/done)
            step_td = step_td_with_next

            experiences.append(step_td)

            # Extract the next state data for episode info extraction
            # The next state info is in step_td["next"]
            next_state_data = step_td["next"]
            reward_next = next_state_data["reward"].view(-1).to(self.env._device)
            running_returns.add_(reward_next)
            running_lengths.add_(1)

            # Check for completed episodes and extract info
            done_mask = next_state_data["done"].squeeze(-1).to(torch.bool)
            if step == 0 and self.verbose >= 1:
                self._vlog(1, f"Step {step}: done_mask.any()={done_mask.any()}, done_count={done_mask.sum()}")
                if "reward" in next_state_data.keys():
                    rew = next_state_data["reward"]
                    self._vlog(1, f"  rewards: mean={rew.mean():.4f}, nonzero={(rew != 0).sum()}")
            if done_mask.any():
                done_rows = done_mask.nonzero(as_tuple=False).view(-1)
                if self.verbose >= 1:
                    self._vlog(1, f"Step {step}: {done_rows.numel()} episodes completed")
                label_tensor = next_state_data.get("label", None)
                depth_tensor = next_state_data.get("query_depth", None)
                success_tensor = next_state_data.get("is_success", None)

                # Buffer GPU clones; convert to CPU once after rollout finishes
                completed_rewards.append(running_returns[done_rows].detach().clone())
                completed_lengths.append(running_lengths[done_rows].detach().clone())

                if labels_available and label_tensor is not None:
                    completed_labels.append(label_tensor.view(-1)[done_rows].detach().clone())
                elif label_tensor is None:
                    labels_available = False
                    completed_labels.clear()

                if depths_available and depth_tensor is not None:
                    completed_depths.append(depth_tensor.view(-1)[done_rows].detach().clone())
                elif depth_tensor is None:
                    depths_available = False
                    completed_depths.clear()

                if success_available and success_tensor is not None:
                    completed_success.append(success_tensor.view(-1)[done_rows].detach().clone())
                elif success_tensor is None:
                    success_available = False
                    completed_success.clear()

                running_returns[done_rows] = 0.0
                running_lengths[done_rows] = 0

            # Update observation for next step
            # Extract only observation keys from next_obs_td (it may have action/value data too)
            obs_td = TensorDict({
                "sub_index": next_obs_td["sub_index"],
                "derived_sub_indices": next_obs_td["derived_sub_indices"],
                "action_mask": next_obs_td["action_mask"],
            }, batch_size=[self.n_envs])

            if rollout_callback is not None:
                rollout_callback(step)

        if completed_rewards:
            reward_tensor = torch.cat(completed_rewards, dim=0)
            length_tensor = torch.cat(completed_lengths, dim=0)

            episode_info: Dict[str, torch.Tensor] = {
                "reward": reward_tensor.detach(),
                "length": length_tensor.detach(),
            }

            if labels_available and completed_labels:
                episode_info["label"] = torch.cat(completed_labels, dim=0).detach()
            if depths_available and completed_depths:
                episode_info["query_depth"] = torch.cat(completed_depths, dim=0).detach()
            if success_available and completed_success:
                episode_info["is_success"] = torch.cat(completed_success, dim=0).detach()
        else:
            device = self.env._device
            episode_info = {
                "reward": torch.zeros(0, dtype=torch.float32, device=device),
                "length": torch.zeros(0, dtype=torch.long, device=device),
            }

        stats = {"episode_info": episode_info}

        episode_count = int(episode_info["reward"].shape[0]) if episode_info.get("reward") is not None else 0
        self._vlog(1, f"Collected {len(experiences)} steps, {episode_count} episodes")
        
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
