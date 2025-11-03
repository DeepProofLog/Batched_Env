"""
Rollout Collection for PPO

This module handles the collection of experience data from environment interactions.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict


class RolloutCollector:
    """
    Collects experience data from environment interactions.
    
    This class manages the rollout collection process, tracking episode statistics
    and storing transitions for later training.
    """
    
    def __init__(
        self,
        n_envs: int,
        n_steps: int,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize the rollout collector.
        
        Args:
            n_envs: Number of parallel environments
            n_steps: Number of steps to collect per rollout
            device: Device to use for tensors
        """
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device
        
        # Episode tracking
        self.env_episode_rewards = torch.zeros(n_envs, device=device)
        self.env_episode_lengths = torch.zeros(n_envs, dtype=torch.int, device=device)
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_info = []
    
    def reset_episode_stats(self) -> None:
        """Reset episode statistics for a new rollout."""
        self.env_episode_rewards.zero_()
        self.env_episode_lengths.zero_()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_info = []
    
    def update_episode_stats(
        self,
        env_idx: int,
        reward: float,
        done: bool,
        next_td: TensorDict,
    ) -> Optional[Dict[str, Any]]:
        """
        Update episode statistics for a specific environment.
        
        Args:
            env_idx: Environment index
            reward: Reward received
            done: Whether episode is done
            next_td: Next state TensorDict
            
        Returns:
            Episode info dict if episode finished, None otherwise
        """
        self.env_episode_rewards[env_idx] += reward
        self.env_episode_lengths[env_idx] += 1
        
        if done:
            # Episode finished, record info
            self.episode_rewards.append(self.env_episode_rewards[env_idx])
            self.episode_lengths.append(self.env_episode_lengths[env_idx])
            
            # Extract episode info for callbacks
            episode_data = {
                "r": self.env_episode_rewards[env_idx],
                "l": self.env_episode_lengths[env_idx],
            }
            
            # Extract metadata from tensordict
            metadata_source = next_td.get("next", next_td)
            
            # Extract label
            label = self._extract_scalar(metadata_source, "label", env_idx, default=1)
            
            # Extract query_depth
            # Note: -1 is used as a sentinel for "no depth" in the environment
            # The metrics collector will handle -1 as "unknown" depth
            query_depth = self._extract_scalar(metadata_source, "query_depth", env_idx, default=-1)
            
            # Extract is_success
            is_success = bool(self._extract_scalar(metadata_source, "is_success", env_idx, default=False))
            
            info = {
                "episode": episode_data,
                "label": label,
                "query_depth": query_depth,
                "is_success": is_success,
            }
            
            # Reset counters
            self.env_episode_rewards[env_idx] = 0.0
            self.env_episode_lengths[env_idx] = 0
            
            return info
        
        return None
    
    def update_episode_stats_batched(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_td: TensorDict,
    ) -> List[Dict[str, Any]]:
        """
        Update episode statistics for all environments in batch.
        
        Args:
            rewards: Batch of rewards received (n_envs,)
            dones: Batch of done flags (n_envs,)
            next_td: Next state TensorDict
            
        Returns:
            List of episode info dicts for finished episodes
        """
        self.env_episode_rewards += rewards
        self.env_episode_lengths += 1
        
        finished_infos = []
        finished_mask = dones.bool()
        
        if finished_mask.any():
            # Record finished episodes
            finished_rewards = self.env_episode_rewards[finished_mask]
            finished_lengths = self.env_episode_lengths[finished_mask]
            
            self.episode_rewards.extend(finished_rewards.tolist())
            self.episode_lengths.extend(finished_lengths.tolist())
            
            # Extract metadata for finished episodes
            metadata_source = next_td.get("next", next_td)
            
            # Extract all metadata in batch, handling missing keys
            labels_all = torch.full((self.n_envs,), 1, device=self.device)  # Default to 1
            if "label" in metadata_source.keys():
                labels_all = self._extract_scalars_batched(metadata_source, "label")
            
            query_depths_all = torch.full((self.n_envs,), -1, device=self.device)  # Default to -1 (no depth)
            if "query_depth" in metadata_source.keys():
                query_depths_all = self._extract_scalars_batched(metadata_source, "query_depth")
            
            is_success_all = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)  # Default to False
            if "is_success" in metadata_source.keys():
                is_success_all = self._extract_scalars_batched(metadata_source, "is_success").bool()
            
            # Filter for finished episodes
            labels_finished = labels_all[finished_mask]
            query_depths_finished = query_depths_all[finished_mask]
            is_success_finished = is_success_all[finished_mask]
            
            # Create episode info dicts using list comprehension
            finished_rewards_list = finished_rewards.tolist()
            finished_lengths_list = finished_lengths.tolist()
            labels_list = labels_finished.tolist()
            query_depths_list = query_depths_finished.tolist()
            is_success_list = is_success_finished.tolist()
            
            finished_infos = [
                {
                    "episode": {
                        "r": r,
                        "l": l,
                    },
                    "label": int(label),
                    "query_depth": qd,
                    "is_success": bool(success),
                }
                for r, l, label, qd, success in zip(
                    finished_rewards_list,
                    finished_lengths_list,
                    labels_list,
                    query_depths_list,
                    is_success_list
                )
            ]
            
            # Reset counters for finished environments
            self.env_episode_rewards[finished_mask] = 0.0
            self.env_episode_lengths[finished_mask] = 0
        
        return finished_infos
    
    def _extract_scalars_batched(
        self,
        tensordict: TensorDict,
        key: str,
    ) -> torch.Tensor:
        """
        Extract scalar values from a TensorDict for all environments.
        
        Args:
            tensordict: TensorDict to extract from
            key: Key to extract
            
        Returns:
            Tensor of extracted scalar values (n_envs,) on self.device
            
        Raises:
            KeyError: If the key is not found in the tensordict
            RuntimeError: If extraction fails for any other reason
        """
        if key not in tensordict.keys():
            raise KeyError(f"Key '{key}' not found in tensordict")
        
        try:
            tensor = tensordict[key]
            if tensor.dim() > 0:
                tensor_reshaped = tensor.reshape(self.n_envs, -1)
                values = tensor_reshaped[:, 0].to(self.device)
            else:
                values = torch.full((self.n_envs,), tensor.item(), device=self.device)
            
            return values
        except Exception as e:
            raise RuntimeError(f"Failed to extract scalar values for key '{key}': {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collected episode statistics.
        
        Returns:
            Dict containing episode statistics
        """
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_info": self.episode_info,
        }


def collect_rollouts(
    env,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    n_envs: int,
    n_steps: int,
    device: torch.device,
    rollout_callback: Optional[Callable] = None,
) -> Tuple[List[TensorDict], Dict[str, Any]]:
    """
    Collect rollout data from environment interactions.
    
    Args:
        env: Environment to collect from
        actor: Actor model
        critic: Critic model
        n_envs: Number of parallel environments
        n_steps: Number of steps to collect
        device: Device to use
        rollout_callback: Optional callback for rollout progress
        
    Returns:
        Tuple of (experiences, stats) where experiences is a list of TensorDicts
        and stats contains episode statistics
    """
    collector = RolloutCollector(n_envs=n_envs, n_steps=n_steps, device=device)
    experiences = []
    
    # Reset environment
    td = env.reset()
    
    # Get the underlying actor-critic model
    critic_inner = critic._module if hasattr(critic, '_module') else critic.module
    actor_critic_model = critic_inner.actor_critic_model
    
    for step in range(n_steps):
        # Update rollout progress
        if rollout_callback is not None:
            rollout_callback(step)
        
        # Get actions from policy
        with torch.no_grad():
            # Get logits and values directly from the model
            logits = actor_critic_model.forward_actor(td.clone())
            values = actor_critic_model.forward_critic(td.clone())
            
            # # Reapply the mask defensively
            # action_mask = td["action_mask"].to(logits.device)
            # masked_logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
            # masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))
            # dist = torch.distributions.Categorical(logits=masked_logits)

            # raise an error if the mask doesnt agree with logits
            action_mask = td["action_mask"].to(logits.device)
            if not torch.all((action_mask.sum(dim=-1) > 0) | torch.isinf(logits).all(dim=-1)):
                raise ValueError("Action mask is inconsistent with logits: some valid actions have -inf logits.")

            # Sample actions directly from masked logits
            dist = torch.distributions.Categorical(logits=logits)
            action_indices = dist.sample()
            log_probs = dist.log_prob(action_indices)
            
            # Convert to one-hot for storage
            action_one_hot = torch.nn.functional.one_hot(
                action_indices, num_classes=logits.shape[-1]
            ).float()
            
            # Store current state with action and value
            experience = TensorDict({
                "sub_index": td["sub_index"],
                "derived_sub_indices": td["derived_sub_indices"],
                "action_mask": td["action_mask"],
                "action": action_one_hot,
                "sample_log_prob": log_probs,
                "state_value": values,
            }, batch_size=torch.Size([n_envs]))
        
        # Step environment
        td["action"] = action_indices
        next_td = env.step(td)
        
        # Extract reward and done from nested structure
        if "next" in next_td.keys():
            next_obs = next_td["next"]
            reward = next_td["next"]["reward"] if "reward" in next_td["next"].keys() else next_td.get("reward")
            done = next_td["next"]["done"] if "done" in next_td["next"].keys() else next_td.get("done")
        else:
            raise NotImplementedError("Expected 'next' key in next_td for reward and done extraction.")
            # next_obs = next_td
            # reward = next_td["reward"]
            # done = next_td["done"]
        
        # Add next state info to experience
        experience["next"] = TensorDict({
            "sub_index": next_obs["sub_index"].clone(),
            "derived_sub_indices": next_obs["derived_sub_indices"].clone(),
            "action_mask": next_obs["action_mask"].clone(),
            "reward": reward.clone(),
            "done": done.clone(),
        }, batch_size=torch.Size([n_envs]))
        
        experiences.append(experience)
        
        # Track episode statistics
        reward_tensor = reward.reshape(n_envs, -1).squeeze(-1).to(device)
        done_tensor = done.reshape(n_envs, -1).squeeze(-1).to(device)

        finished_infos = collector.update_episode_stats_batched(
            reward_tensor, done_tensor, next_td
        )
        collector.episode_info.extend(finished_infos)
        
        # Update state for next iteration
        if "next" in next_td.keys():
            next_obs = next_td["next"]
        else:
            raise NotImplementedError("Expected 'next' key in next_td for next_obs extraction.")
            # next_obs = next_td
            
        td = TensorDict({
            "sub_index": next_obs["sub_index"].clone(),
            "derived_sub_indices": next_obs["derived_sub_indices"].clone(),
            "action_mask": next_obs["action_mask"].clone(),
        }, batch_size=torch.Size([n_envs]))
    
    return experiences, collector.get_stats()
