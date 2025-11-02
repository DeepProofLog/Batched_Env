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
        self.env_episode_rewards = [0.0] * n_envs
        self.env_episode_lengths = [0] * n_envs
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_info = []
    
    def reset_episode_stats(self) -> None:
        """Reset episode statistics for a new rollout."""
        self.env_episode_rewards = [0.0] * self.n_envs
        self.env_episode_lengths = [0] * self.n_envs
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
            query_depth_val = self._extract_scalar(metadata_source, "query_depth", env_idx, default=None)
            query_depth = query_depth_val if query_depth_val not in (0, -1, None) else None
            
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
    
    def _extract_scalar(
        self,
        tensordict: TensorDict,
        key: str,
        env_idx: int,
        default: Any = None,
    ) -> Any:
        """
        Extract a scalar value from a TensorDict.
        
        Args:
            tensordict: TensorDict to extract from
            key: Key to extract
            env_idx: Environment index
            default: Default value if key not found
            
        Returns:
            Extracted scalar value or default
        """
        try:
            if key not in tensordict.keys():
                return default
            
            tensor = tensordict[key]
            if tensor.dim() > 0:
                tensor_reshaped = tensor.reshape(self.n_envs, -1)
                value = tensor_reshaped[env_idx, 0].item()
            else:
                value = tensor.item()
            
            return value
        except Exception as e:
            return default
    
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
            
            # Reapply the mask defensively
            action_mask = td["action_mask"].to(logits.device)
            masked_logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
            masked_logits = torch.nan_to_num(masked_logits, nan=float("-inf"))

            # Sample actions directly from masked logits
            dist = torch.distributions.Categorical(logits=masked_logits)
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
            next_obs = next_td
            reward = next_td["reward"]
            done = next_td["done"]
        
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
        reward_tensor = reward.reshape(n_envs, -1)
        done_tensor = done.reshape(n_envs, -1)

        for env_idx in range(n_envs):
            env_reward = reward_tensor[env_idx, 0].item()
            env_done = bool(done_tensor[env_idx, 0].item())
            
            episode_info = collector.update_episode_stats(
                env_idx, env_reward, env_done, next_td
            )
            
            if episode_info is not None:
                collector.episode_info.append(episode_info)
        
        # Update state for next iteration
        if "next" in next_td.keys():
            next_obs = next_td["next"]
        else:
            next_obs = next_td
            
        td = TensorDict({
            "sub_index": next_obs["sub_index"].clone(),
            "derived_sub_indices": next_obs["derived_sub_indices"].clone(),
            "action_mask": next_obs["action_mask"].clone(),
        }, batch_size=torch.Size([n_envs]))
    
    return experiences, collector.get_stats()
