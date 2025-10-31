"""
model_eval_torchrl.py — TorchRL-compatible evaluation utilities

This module provides TorchRL-compatible versions of the evaluation functions
originally designed for Stable-Baselines3. It wraps TorchRL actor modules to
provide an SB3-like policy interface for seamless integration with existing
evaluation infrastructure.

Key components:
1. TorchRLPolicyWrapper - Adapts TorchRL actors to SB3 policy interface
2. TorchRLEnvWrapper - Adapts TorchRL batched envs to SB3 VecEnv interface
3. evaluate_policy_torchrl - TorchRL-specific policy evaluation
4. eval_corruptions_torchrl - TorchRL-compatible corruption-based evaluation

All other evaluation utilities from model_eval.py are re-exported for convenience.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from tensordict import TensorDict
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

# Import all the helper functions and utilities from model_eval
from sb3_model_eval import (
    _BATCH_METRIC_KEYS,
    _GLOBAL_METRICS_TEMPLATE,
    _ensure_vec_env,
    _init_batch_metrics,
    _init_global_metrics,
    _compute_targets_and_mask,
    _configure_env_batch,
    _combine_hybrid_scores,
    kge_eval,
    _extract_and_accumulate_metrics,
    _report_batch_metrics,
    _finalize_and_get_results,
    prepare_batch_data,
    plot_logprob_heatmap,
)


class TorchRLEnvWrapper(VecEnv):
    """
    Wrapper to adapt TorchRL batched environment to SB3 VecEnv interface.
    
    This allows CustomBatchedEnv to work with the existing evaluate_policy
    infrastructure from model_eval.py which expects SB3-style VecEnv.
    """
    
    def __init__(self, torchrl_env):
        """
        Initialize wrapper around TorchRL batched environment.
        
        Args:
            torchrl_env: TorchRL batched environment (e.g., CustomBatchedEnv)
        """
        self.torchrl_env = torchrl_env
        self.num_envs = torchrl_env.num_envs
        
        # Get individual environment reference for specs
        env0 = torchrl_env.envs[0]
        
        # Create Gymnasium-compatible spaces from TorchRL specs
        # For now, use simple Box/Discrete spaces as placeholders
        # The actual observation/action handling is done via dicts/tensors
        from gymnasium import spaces
        
        # Create dummy observation space (Dict space with sub_index, derived_sub_indices, action_mask)
        self.observation_space = spaces.Dict({
            'sub_index': spaces.Box(low=0, high=1000, shape=(1, env0.padding_atoms, 3), dtype=np.int32),
            'derived_sub_indices': spaces.Box(low=0, high=1000, shape=(env0.padding_states, env0.padding_atoms, 3), dtype=np.int32),
            'action_mask': spaces.Box(low=0, high=1, shape=(env0.padding_states,), dtype=np.bool_),
        })
        
        # Create dummy action space
        self.action_space = spaces.Discrete(env0.padding_states)
        
        # Copy custom attributes needed by evaluate_policy
        self.type_ = "custom_dummy"  # Pretend to be custom_dummy for compatibility
        self._episode_count = torchrl_env._episode_count.numpy()
        self._episode_target = torchrl_env._episode_target.numpy()
        self.active_envs = torchrl_env.active_envs.numpy()
        
        # Track episode statistics for info callback
        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        
        # Access to underlying envs
        self.envs = [type('EnvWrapper', (), {'env': env})() for env in torchrl_env.envs]
        
    def reset(self):
        """Reset the environment and return observations."""
        td = self.torchrl_env.reset()
        # Reset episode tracking
        self._episode_rewards[:] = 0.0
        self._episode_lengths[:] = 0
        return self._tensordict_to_obs(td)
    
    def _get_last_valid_obs(self):
        """Get the last valid observation (used for error recovery)."""
        # Try to get current tensordict from the batched environment
        try:
            if hasattr(self.torchrl_env, 'tensordict'):
                td = self.torchrl_env.tensordict
                return self._tensordict_to_obs(td)
        except Exception:
            pass
        
        # Fallback: create dummy observation
        env0 = self.torchrl_env.envs[0]
        return {
            'sub_index': np.zeros((self.num_envs, 1, env0.padding_atoms, 3), dtype=np.int32),
            'derived_sub_indices': np.zeros((self.num_envs, env0.padding_states, env0.padding_atoms, 3), dtype=np.int32),
            'action_mask': np.zeros((self.num_envs, env0.padding_states), dtype=bool),
        }

    
    def step_async(self, actions):
        """Store actions to be executed on step_wait."""
        self._async_actions = actions
    
    def step_wait(self):
        """Execute stored actions and return results."""
        actions = self._async_actions
        
        # Convert numpy actions to tensor, ensure proper shape
        if isinstance(actions, np.ndarray):
            actions_tensor = torch.from_numpy(actions).flatten()
        else:
            actions_tensor = torch.tensor(actions).flatten()

        
        # Ensure we have exactly num_envs actions
        assert len(actions_tensor) == self.num_envs, \
            f"Expected {self.num_envs} actions, got {len(actions_tensor)}"
        
        # Create action TensorDict
        batch_size = self.torchrl_env.num_envs
        action_td = TensorDict({
            "action": torch.as_tensor(actions_tensor, dtype=torch.long),
        }, batch_size=torch.Size([batch_size]))
        
        # Step the environment
        td = self.torchrl_env.step(action_td)
        
        # TorchRL stores step results in 'next' subdictionary
        # Extract reward and done from the correct location
        if "next" in td.keys():
            next_td = td["next"]
            reward_tensor = next_td.get("reward")
            done_tensor = next_td.get("done")
        else:
            reward_tensor = td.get("reward")
            done_tensor = td.get("done")
        
        if reward_tensor is None:
            raise ValueError(f"TensorDict has no 'reward'. TD keys: {list(td.keys())}, "
                           f"next keys: {list(td['next'].keys()) if 'next' in td.keys() else 'N/A'}")
        if done_tensor is None:
            raise ValueError(f"TensorDict has no 'done'. TD keys: {list(td.keys())}, "
                           f"next keys: {list(td['next'].keys()) if 'next' in td.keys() else 'N/A'}")
        
        # Extract components - ensure 1D arrays
        obs = self._tensordict_to_obs(td)
        rewards = reward_tensor.cpu().numpy().flatten()[:self.num_envs]
        dones = done_tensor.cpu().numpy().flatten()[:self.num_envs].astype(bool)
        
        # Accumulate episode statistics
        self._episode_rewards += rewards
        self._episode_lengths += 1
        
        # Build info dicts
        infos = []
        for i in range(self.num_envs):
            info = {}
            if dones[i]:
                # Add terminal observation if available
                if "next" in td.keys():
                    info["terminal_observation"] = {
                        key: td["next"][key][i].cpu().numpy() 
                        for key in ["sub_index", "derived_sub_indices", "action_mask"]
                        if key in td["next"].keys()
                    }
                # Add success flag if available
                if "is_success" in td.keys():
                    info["is_success"] = bool(td["is_success"][i].item())
                elif "next" in td.keys() and "is_success" in td["next"].keys():
                    info["is_success"] = bool(td["next"]["is_success"][i].item())
                else:
                    info["is_success"] = False
                
                # Add episode statistics for depth tracking
                info["episode"] = {
                    "r": float(self._episode_rewards[i]),  # Total episode reward
                    "l": int(self._episode_lengths[i]),    # Total episode length
                }
                
                # Extract label from TensorDict
                if "label" in td.keys():
                    info["label"] = int(td["label"][i].item())
                elif "next" in td.keys() and "label" in td["next"].keys():
                    info["label"] = int(td["next"]["label"][i].item())
                
                # Extract query_depth from TensorDict
                if "query_depth" in td.keys():
                    info["query_depth"] = int(td["query_depth"][i].item())
                elif "next" in td.keys() and "query_depth" in td["next"].keys():
                    info["query_depth"] = int(td["next"]["query_depth"][i].item())
                
                # Add episode index for deduplication
                if "episode_idx" in td.keys():
                    info["episode_idx"] = int(td["episode_idx"][i].item())
                elif "next" in td.keys() and "episode_idx" in td["next"].keys():
                    info["episode_idx"] = int(td["next"]["episode_idx"][i].item())
                
                # Reset episode statistics for this environment
                self._episode_rewards[i] = 0.0
                self._episode_lengths[i] = 0
            infos.append(info)
        
    # Update episode tracking
        self._episode_count = self.torchrl_env._episode_count.numpy()
        self.active_envs = self.torchrl_env.active_envs.numpy()
        
        return obs, rewards, dones, infos
    
    def step(self, actions):
        """Execute actions and return results (combines step_async and step_wait)."""
        self.step_async(actions)
        return self.step_wait()
    
    def _tensordict_to_obs(self, td: TensorDict) -> dict:
        """Convert TensorDict observation to dict of numpy arrays."""
        # After step(), observations are in the 'next' subdictionary
        if "next" in td.keys():
            obs_td = td["next"]
        else:
            obs_td = td
            
        obs = {}
        for key in ["sub_index", "derived_sub_indices", "action_mask"]:
            if key in obs_td.keys():
                tensor_val = obs_td[key]
                # Convert to numpy - should already be batched correctly
                np_val = tensor_val.cpu().numpy()
                
                # Ensure batch dimension is first and matches num_envs
                if np_val.shape[0] != self.num_envs:
                    raise ValueError(f"Expected batch size {self.num_envs} for {key}, got {np_val.shape[0]}")
                
                obs[key] = np_val
        return obs
    
    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from underlying environments."""
        if indices is None:
            indices = range(self.num_envs)
        
        results = []
        for idx in indices:
            env = self.torchrl_env.envs[idx]
            if hasattr(env, attr_name):
                results.append(getattr(env, attr_name))
            else:
                results.append(None)
        return results

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call method on underlying environments."""
        if indices is None:
            indices = range(self.num_envs)
        
        results = []
        for idx in indices:
            env = self.torchrl_env.envs[idx]
            method = getattr(env, method_name)
            results.append(method(*method_args, **method_kwargs))
        return results
    
    def seed(self, seed=None):
        """Set seed for all environments."""
        if seed is None:
            return
        seeds = []
        for i, env in enumerate(self.torchrl_env.envs):
            env_seed = seed + i
            env.seed(env_seed)
            seeds.append(env_seed)
        return seeds
    
    def close(self):
        """Close all environments."""
        for env in self.torchrl_env.envs:
            env.close()
    
    def set_attr(self, attr_name: str, value: Any, indices=None):
        """Set attribute on underlying environments."""
        if indices is None:
            indices = range(self.num_envs)
        
        for idx in indices:
            env = self.torchrl_env.envs[idx]
            setattr(env, attr_name, value)
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped with a given wrapper."""
        if indices is None:
            indices = range(self.num_envs)
        
        results = []
        for idx in indices:
            env = self.torchrl_env.envs[idx]
            # Check if env is instance of wrapper_class
            is_wrapped = isinstance(env, wrapper_class)
            results.append(is_wrapped)
        return results


class TorchRLPolicyWrapper:
    """
    Wrapper to adapt TorchRL actor module to SB3-style policy interface.
    
    This allows using the existing model_eval.py infrastructure with TorchRL models.
    The wrapper maintains compatibility with the evaluate_policy function which
    expects a model.policy(obs, deterministic) -> (actions, values, log_probs) interface.
    """
    
    def __init__(self, actor: nn.Module, device: torch.device):
        """
        Initialize the TorchRL policy wrapper.
        
        Args:
            actor: TorchRL actor module (ProbabilisticActor)
            device: Device to run inference on
        """
        self.actor = actor
        self.device = device
        self._was_training = False
        
    def policy(self, obs_tensor: dict, deterministic: bool = True):
        """
        SB3-style policy interface: (obs) -> (actions, values, log_probs)
        
        Args:
            obs_tensor: Dict with 'sub_index', 'derived_sub_indices', 'action_mask'
                       Can be either numpy arrays or torch tensors
            deterministic: Whether to use deterministic actions (always True for eval)
            
        Returns:
            actions: Action indices (as 1D tensors)
            values: Dummy values (not used in evaluation)
            log_probs: Log probabilities of selected actions
        """
        try:
            # Ensure eval mode
            self._was_training = self.actor.training
            self.actor.eval()
            
            with torch.no_grad():
                # Convert observations to tensors if needed
                sub_index = torch.as_tensor(obs_tensor['sub_index'], device=self.device)
                derived_sub_indices = torch.as_tensor(obs_tensor['derived_sub_indices'], device=self.device)
                action_mask = torch.as_tensor(obs_tensor['action_mask'], device=self.device)
                
                # Create TensorDict from obs
                batch_size = sub_index.shape[0]
                td = TensorDict({
                    'sub_index': sub_index,
                    'derived_sub_indices': derived_sub_indices,
                    'action_mask': action_mask,
                }, batch_size=torch.Size([batch_size]))
                
                # Get action distribution from actor
                td_with_action = self.actor(td)
                
                # Extract action - check if it's already indices or one-hot
                action_output = td_with_action.get("action")
                
                if action_output is None:
                    raise ValueError("Actor did not produce 'action' in output TensorDict. "
                                   f"Available keys: {list(td_with_action.keys())}")
                
                # Handle different action output shapes:
                # - [batch, num_actions]: one-hot encoded actions, argmax to get indices
                # - [batch, time, num_actions]: batched time series, take last time step
                # - [batch]: already action indices
                if action_output.dim() == 3:
                    # Shape [batch, time, num_actions] - take last time step
                    action_output = action_output[:, -1, :]
                
                if action_output.dim() == 2:
                    # One-hot encoded: shape (batch, num_actions)
                    actions = torch.argmax(action_output, dim=-1)
                elif action_output.dim() == 1:
                    # Already indices: shape (batch,)
                    actions = action_output.long()
                else:
                    raise ValueError(f"Unexpected action dimension: {action_output.dim()}, shape: {action_output.shape}")
                
                # Get log probability
                log_probs = td_with_action.get("sample_log_prob", None)
                if log_probs is None:
                    # Fallback: compute from action distribution if available
                    log_probs = torch.zeros(batch_size, device=self.device)
                elif log_probs.dim() == 2:
                    # If log_probs is [batch, time], take last time step
                    log_probs = log_probs[:, -1]
                
                # Ensure 1D tensors
                actions = actions.flatten()
                log_probs = log_probs.flatten()
                
                # --- Safety: enforce action mask ---
                # Sometimes numerical or masking issues can lead to invalid
                # actions being selected. Make sure the returned actions are
                # within the valid set from action_mask. If not, replace with
                # the first valid action and set a reasonable fallback log-prob.
                try:
                    # action_mask is available in the observation tensor
                    am = action_mask.to(torch.bool)
                    if am.dim() == 2 and am.shape[0] == actions.shape[0]:
                        for i in range(actions.shape[0]):
                            a = int(actions[i].item())
                            # If action index out of range, treat as invalid
                            if a < 0 or a >= am.shape[1] or not bool(am[i, a].item()):
                                valid = torch.where(am[i])[0]
                                if valid.numel() > 0:
                                    new_a = int(valid[0].item())
                                    actions[i] = new_a
                                    # Fallback log-prob: uniform over valid actions
                                    fallback_lp = float(torch.log(torch.tensor(1.0 / float(valid.numel()), device=self.device)))
                                    log_probs[i] = fallback_lp
                                else:
                                    # No valid actions: fallback to 0
                                    actions[i] = 0
                                    log_probs[i] = float('-inf')
                except Exception:
                    # If anything goes wrong in mask enforcement, do not crash evaluation
                    pass

                # Verify shapes match batch_size
                if actions.shape[0] != batch_size:
                    raise ValueError(f"Actions shape {actions.shape} doesn't match batch_size {batch_size}. "
                                   f"Action output shape was {action_output.shape}")
                if log_probs.shape[0] != batch_size:
                    raise ValueError(f"Log probs shape {log_probs.shape} doesn't match batch_size {batch_size}")
                
                # Dummy values (not used in evaluation)
                values = torch.zeros(batch_size, device=self.device)
                
                # Ensure all return values are valid tensors
                assert actions is not None, "Actions tensor is None"
                assert values is not None, "Values tensor is None"
                assert log_probs is not None, "Log probs tensor is None"
            
            # Restore training mode if it was training
            if self._was_training:
                self.actor.train()
                
            return actions, values, log_probs
        except Exception as e:
            import traceback
            print(f"Error in TorchRLPolicyWrapper.policy(): {e}")
            print(traceback.format_exc())
            raise


@torch.inference_mode()
def evaluate_policy_torchrl(
    actor: nn.Module,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    target_episodes: np.ndarray | None = None,
    verbose: int = 0,
    track_logprobs: bool = False,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    TorchRL-specific version of evaluate_policy.
    
    This function wraps a TorchRL actor in TorchRLPolicyWrapper and TorchRL
    batched environment in TorchRLEnvWrapper, then delegates to the standard
    evaluate_policy implementation from model_eval.py.
    
    Args:
        actor: TorchRL actor module
        env: Vectorized environment (will be wrapped if TorchRL batched env)
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        target_episodes: Per-env episode targets
        verbose: Verbosity level
        track_logprobs: Whether to track per-step log probabilities
        info_callback: Optional callback for info dicts
        
    Returns:
        Same as evaluate_policy: (rewards, lengths, logps, mask, proof_successful)
        or with histories if track_logprobs=True
    """
    # Import here to avoid circular dependency
    from sb3_model_eval import evaluate_policy
    
    # Get device from actor parameters
    device = next(actor.parameters()).device
    
    # Wrap TorchRL batched environment if needed
    if hasattr(env, 'type_') and env.type_ == "custom_batched":
        env = TorchRLEnvWrapper(env)
    
    # Create policy wrapper
    policy_wrapper = TorchRLPolicyWrapper(actor, device)
    
    # Delegate to standard evaluate_policy
    return evaluate_policy(
        model=policy_wrapper,
        env=env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        target_episodes=target_episodes,
        verbose=verbose,
        track_logprobs=track_logprobs,
        info_callback=info_callback,
    )


def eval_corruptions_torchrl(
    actor: nn.Module,
    env: Union[gym.Env, VecEnv],
    data: List[Any],
    sampler: Any,
    n_corruptions: Optional[int] = None,
    deterministic: bool = True,
    verbose: int = 1,
    plot: bool = False,
    kge_inference_engine = None,
    evaluation_mode: str = 'rl_only',
    corruption_scheme: List[str] | None = None,
    info_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    data_depths: Optional[List[int]] = None,
    hybrid_kge_weight: float = 2.0,
    hybrid_rl_weight: float = 1.0,
    hybrid_success_only: bool = True,
) -> Dict[str, Any]:
    """
    TorchRL-specific version of eval_corruptions.
    
    This function wraps a TorchRL actor in TorchRLPolicyWrapper and TorchRL
    batched environment in TorchRLEnvWrapper, then delegates to the standard
    eval_corruptions implementation from model_eval.py.
    
    Args:
        actor: TorchRL actor module
        env: Vectorized environment (will be wrapped if TorchRL batched env)
        data: List of positive queries to evaluate
        sampler: Negative sampler for corruptions
        n_corruptions: Number of corruptions per query (None = all)
        deterministic: Whether to use deterministic actions
        verbose: Verbosity level
        plot: Whether to generate plots
        kge_inference_engine: Optional KGE model for hybrid evaluation
        evaluation_mode: One of 'rl_only', 'kge_only', 'hybrid'
        corruption_scheme: List of corruption types ('head', 'tail')
        info_callback: Optional callback for info dicts
        data_depths: Optional depth information for queries
        hybrid_kge_weight: Weight for KGE scores in hybrid mode
        hybrid_rl_weight: Weight for RL scores in hybrid mode
        hybrid_success_only: Whether to only add RL bonus for successful proofs
        
    Returns:
        Dict of evaluation metrics (MRR, Hits@K, etc.)
    """
    # Import here to avoid circular dependency
    from sb3_model_eval import eval_corruptions
    
    # Get device from actor parameters
    device = next(actor.parameters()).device
    
    # Wrap TorchRL batched environment if needed
    if hasattr(env, 'type_') and env.type_ == "custom_batched":
        env = TorchRLEnvWrapper(env)
    
    # Create policy wrapper
    policy_wrapper = TorchRLPolicyWrapper(actor, device)
    
    # Delegate to standard eval_corruptions
    return eval_corruptions(
        model=policy_wrapper,
        env=env,
        data=data,
        sampler=sampler,
        n_corruptions=n_corruptions,
        deterministic=deterministic,
        verbose=verbose,
        plot=plot,
        kge_inference_engine=kge_inference_engine,
        evaluation_mode=evaluation_mode,
        corruption_scheme=corruption_scheme,
        info_callback=info_callback,
        data_depths=data_depths,
        hybrid_kge_weight=hybrid_kge_weight,
        hybrid_rl_weight=hybrid_rl_weight,
        hybrid_success_only=hybrid_success_only,
    )


# Re-export commonly used utilities for convenience
__all__ = [
    'TorchRLPolicyWrapper',
    'TorchRLEnvWrapper',
    'evaluate_policy_torchrl',
    'eval_corruptions_torchrl',
    # Re-exported from model_eval
    'kge_eval',
    'plot_logprob_heatmap',
]
