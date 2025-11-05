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
from sb3_code.sb3_model_eval import (
    kge_eval,
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
        from torchrl.envs import EnvBase
        
        self.torchrl_env = torchrl_env

        # Try to infer the batch size / number of environments by inspecting
        # the result of a reset. This works for both custom batched envs and
        # TorchRL ParallelEnv/EnvBase instances.
        td = None
        try:
            td = self.torchrl_env.reset()
        except Exception:
            # If reset fails here, we'll fallback to assumptions below
            td = None

        # Determine num_envs from returned TensorDict if available
        self.num_envs = 1
        if isinstance(td, TensorDict):
            # For TorchRL envs, batch_size attribute tells us if it's batched
            td_batch_size = td.batch_size
            if len(td_batch_size) > 0:
                # Batched environment - first dimension is num_envs
                self.num_envs = int(td_batch_size[0])
            else:
                # Single (unbatched) environment
                self.num_envs = 1

        # If the wrapped env exposes a batch_size or num_workers, prefer it
        if hasattr(torchrl_env, "num_envs"):
            try:
                self.num_envs = int(getattr(torchrl_env, "num_envs"))
            except Exception:
                pass
        elif hasattr(torchrl_env, "num_workers"):
            try:
                self.num_envs = int(getattr(torchrl_env, "num_workers"))
            except Exception:
                pass
        elif hasattr(torchrl_env, "batch_size"):
            try:
                bs = getattr(torchrl_env, "batch_size")
                if isinstance(bs, (list, tuple)) and len(bs) > 0:
                    self.num_envs = int(bs[0])
                elif isinstance(bs, torch.Size) and len(bs) > 0:
                    self.num_envs = int(bs[0])
            except Exception:
                pass

        # Build lightweight Gym spaces from the observed TensorDict shape when possible
        from gymnasium import spaces
        sample_obs = None
        if isinstance(td, TensorDict):
            obs_td = td.get("next") if "next" in td.keys() else td
            sample_obs = {}
            for key in ("sub_index", "derived_sub_indices", "action_mask"):
                if key in obs_td.keys():
                    v = obs_td.get(key)
                    npv = v.cpu().numpy()
                    # sample shape excludes batch dim
                    sample_shape = tuple(npv.shape[1:])
                    sample_obs[key] = (npv.dtype, sample_shape)

        # Fallback assumptions if we couldn't infer shapes
        if sample_obs is None:
            # Conservative defaults
            sample_obs = {
                'sub_index': (np.int32, (1, 6, 3)),
                'derived_sub_indices': (np.int32, (20, 6, 3)),
                'action_mask': (np.bool_, (24,)),
            }

        # Build observation_space dict
        obs_spaces = {}
        for key, (dtype, shape) in sample_obs.items():
            high = 1 if np.issubdtype(dtype, np.bool_) else 100000
            # Prepend no batch dim in gym space definition; evaluate_policy expects per-env shapes
            obs_spaces[key] = spaces.Box(low=0, high=high, shape=shape, dtype=dtype)

        self.observation_space = spaces.Dict(obs_spaces)

        # Derive action space size from action_mask shape
        action_mask_info = sample_obs.get('action_mask', (np.bool_, (24,)))
        action_mask_shape = action_mask_info[1]  # Extract shape tuple from (dtype, shape)
        # action_mask_shape is a tuple like (24,) - get the first element
        if isinstance(action_mask_shape, tuple) and len(action_mask_shape) > 0:
            n_actions = int(action_mask_shape[0])
        elif isinstance(action_mask_shape, int):
            n_actions = int(action_mask_shape)
        else:
            n_actions = 24  # Fallback
        self.action_space = spaces.Discrete(n_actions)

        # Compatibility attributes expected by SB3-style evaluators
        # Always set to "custom_dummy" for SB3 compatibility
        self.type_ = 'custom_dummy'
        # Episode counters: try to copy if available, else initialize
        try:
            self._episode_count = (torchrl_env._episode_count.numpy()
                                   if hasattr(torchrl_env, '_episode_count') else np.zeros(self.num_envs, dtype=np.int32))
        except Exception:
            self._episode_count = np.zeros(self.num_envs, dtype=np.int32)
        try:
            self._episode_target = (torchrl_env._episode_target.numpy()
                                    if hasattr(torchrl_env, '_episode_target') else np.full(self.num_envs, 2, dtype=np.int32))
        except Exception:
            self._episode_target = np.full(self.num_envs, 2, dtype=np.int32)
        try:
            self.active_envs = (torchrl_env.active_envs.numpy() if hasattr(torchrl_env, 'active_envs') else np.ones(self.num_envs, dtype=bool))
        except Exception:
            self.active_envs = np.ones(self.num_envs, dtype=bool)

        self._episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        # Create a list-like `envs` attribute for compatibility with SB3 code that
        # expects env wrappers to expose an `envs` sequence of objects with `.env`.
        # Import here to avoid circular dependencies
        from torchrl.envs import ParallelEnv
        
        if isinstance(torchrl_env, ParallelEnv):
            # ParallelEnv runs envs in separate processes - no direct access
            # Create placeholders
            self.envs = [type('EnvWrapper', (), {'env': None})() for _ in range(self.num_envs)]
            self._needs_alternative_env_access = True
        elif hasattr(torchrl_env, 'envs') and not isinstance(torchrl_env, ParallelEnv):
            # Custom batched env with direct access to underlying envs
            self.envs = [type('EnvWrapper', (), {'env': env})() for env in getattr(torchrl_env, 'envs')]
        elif self.num_envs == 1 and isinstance(torchrl_env, EnvBase):
            # Single EnvBase instance - wrap it so env.envs[0].env gives the actual env
            self.envs = [type('EnvWrapper', (), {'env': torchrl_env})()]
        else:
            # Fallback: Parallel env with subprocess workers - no direct access
            # Create placeholders
            self.envs = [type('EnvWrapper', (), {'env': None})() for _ in range(self.num_envs)]
            self._needs_alternative_env_access = True
    
    def configure_env(self, env_idx: int, **config_kwargs):
        """Configure a specific environment in the batch.
        
        For ParallelEnv, this sends a configure message to the worker process.
        For direct access envs, this sets attributes directly or calls configure.
        
        Args:
            env_idx: Index of the environment to configure
            **config_kwargs: Configuration parameters (mode, queries, labels, etc.)
        """
        from torchrl.envs import ParallelEnv
        
        if isinstance(self.torchrl_env, ParallelEnv):
            # For ParallelEnv, call the configure method on the worker
            # ParallelEnv supports calling methods on workers via attribute access
            try:
                # Try to call configure on the worker environment
                # This will be sent to the subprocess
                self.torchrl_env[env_idx].configure(**config_kwargs)
            except Exception as e:
                # If that doesn't work, fall back to setting attributes
                # This approach stores configs and applies on reset
                if not hasattr(self, '_pending_configs'):
                    self._pending_configs = [{}  for _ in range(self.num_envs)]
                self._pending_configs[env_idx] = config_kwargs
        elif self.envs[env_idx].env is not None:
            # Direct access - try configure method first, then fall back to attributes
            inner_env = self.envs[env_idx].env
            if hasattr(inner_env, 'configure'):
                inner_env.configure(**config_kwargs)
            else:
                for key, value in config_kwargs.items():
                    setattr(inner_env, key, value)
        else:
            raise RuntimeError(f"Cannot configure environment {env_idx} - no access method available")
        
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
        
        # Create action TensorDict with proper batch size
        # For single unbatched env, use empty batch_size; for batched, use [num_envs]
        if self.num_envs == 1 and hasattr(self.torchrl_env, 'batch_size'):
            env_batch_size = self.torchrl_env.batch_size
        else:
            env_batch_size = torch.Size([self.num_envs])
        
        # For unbatched env, action should be scalar
        if env_batch_size == torch.Size([]):
            action_val = actions_tensor[0]  # Extract scalar from 1-element tensor
        else:
            action_val = actions_tensor
        
        action_td = TensorDict({
            "action": torch.as_tensor(action_val, dtype=torch.long),
        }, batch_size=env_batch_size)
        
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
        
        # Helper function to safely extract tensor values (handles both batched and unbatched)
        def safe_tensor_extract(tensor_val, index):
            """Extract value from tensor, handling both batched and unbatched cases."""
            if tensor_val.dim() == 0:
                # 0-dim tensor (unbatched environment)
                return tensor_val.item()
            else:
                # Batched tensor
                return tensor_val[index].item()
        
        # Build info dicts
        infos = []
        for i in range(self.num_envs):
            info = {}
            if dones[i]:
                # Add terminal observation if available
                if "next" in td.keys():
                    if self.num_envs == 1:
                        # Unbatched - no indexing needed
                        info["terminal_observation"] = {
                            key: td["next"][key].cpu().numpy() 
                            for key in ["sub_index", "derived_sub_indices", "action_mask"]
                            if key in td["next"].keys()
                        }
                    else:
                        # Batched - need indexing
                        info["terminal_observation"] = {
                            key: td["next"][key][i].cpu().numpy() 
                            for key in ["sub_index", "derived_sub_indices", "action_mask"]
                            if key in td["next"].keys()
                        }
                # Add success flag if available
                if "is_success" in td.keys():
                    info["is_success"] = bool(safe_tensor_extract(td["is_success"], i))
                elif "next" in td.keys() and "is_success" in td["next"].keys():
                    info["is_success"] = bool(safe_tensor_extract(td["next"]["is_success"], i))
                else:
                    info["is_success"] = False
                
                # Add episode statistics for depth tracking
                info["episode"] = {
                    "r": float(self._episode_rewards[i]),  # Total episode reward
                    "l": int(self._episode_lengths[i]),    # Total episode length
                }
                
                # Extract label from TensorDict
                if "label" in td.keys():
                    info["label"] = int(safe_tensor_extract(td["label"], i))
                elif "next" in td.keys() and "label" in td["next"].keys():
                    info["label"] = int(safe_tensor_extract(td["next"]["label"], i))
                
                # Extract query_depth from TensorDict
                if "query_depth" in td.keys():
                    info["query_depth"] = int(safe_tensor_extract(td["query_depth"], i))
                elif "next" in td.keys() and "query_depth" in td["next"].keys():
                    info["query_depth"] = int(safe_tensor_extract(td["next"]["query_depth"], i))
                
                # Add episode index for deduplication
                if "episode_idx" in td.keys():
                    info["episode_idx"] = int(safe_tensor_extract(td["episode_idx"], i))
                elif "next" in td.keys() and "episode_idx" in td["next"].keys():
                    info["episode_idx"] = int(safe_tensor_extract(td["next"]["episode_idx"], i))
                
                # Reset episode statistics for this environment
                self._episode_rewards[i] = 0.0
                self._episode_lengths[i] = 0
            infos.append(info)
        
        # Update episode tracking (only if the wrapped env has these attributes)
        if hasattr(self.torchrl_env, '_episode_count'):
            try:
                self._episode_count = self.torchrl_env._episode_count.numpy()
            except Exception:
                pass
        if hasattr(self.torchrl_env, 'active_envs'):
            try:
                self.active_envs = self.torchrl_env.active_envs.numpy()
            except Exception:
                pass
        
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
                # Convert to numpy
                np_val = tensor_val.cpu().numpy()
                
                # For unbatched envs (num_envs==1 and no batch dim), add batch dimension
                if self.num_envs == 1 and (len(np_val.shape) < 2 or np_val.shape[0] != 1):
                    # Check if this is actually unbatched by looking at expected shapes
                    # sub_index should be (1, atoms, arity+1), derived_sub_indices (states, atoms, arity+1), action_mask (states,)
                    np_val = np.expand_dims(np_val, axis=0)  # Add batch dimension
                
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
            env_obj = None
            if hasattr(self.torchrl_env, 'envs'):
                try:
                    env_obj = self.torchrl_env.envs[idx]
                except Exception:
                    env_obj = None
            elif hasattr(self, 'envs') and idx < len(self.envs):
                env_obj = getattr(self.envs[idx], 'env', None)

            if env_obj is not None and hasattr(env_obj, attr_name):
                results.append(getattr(env_obj, attr_name))
            else:
                results.append(None)
        return results

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call method on underlying environments."""
        if indices is None:
            indices = range(self.num_envs)
        
        results = []
        for idx in indices:
            env_obj = None
            if hasattr(self.torchrl_env, 'envs'):
                try:
                    env_obj = self.torchrl_env.envs[idx]
                except Exception:
                    env_obj = None
            elif hasattr(self, 'envs') and idx < len(self.envs):
                env_obj = getattr(self.envs[idx], 'env', None)

            if env_obj is None:
                results.append(None)
            else:
                method = getattr(env_obj, method_name)
                results.append(method(*method_args, **method_kwargs))
        return results
    
    def seed(self, seed=None):
        """Set seed for all environments."""
        if seed is None:
            return
        seeds = []
        if hasattr(self.torchrl_env, 'envs'):
            for i, env in enumerate(self.torchrl_env.envs):
                env_seed = seed + i
                try:
                    env.seed(env_seed)
                except Exception:
                    pass
                seeds.append(env_seed)
        elif hasattr(self, 'envs'):
            for i, wrapper in enumerate(self.envs):
                env_obj = getattr(wrapper, 'env', None)
                env_seed = seed + i
                if env_obj is not None:
                    try:
                        env_obj.seed(env_seed)
                    except Exception:
                        pass
                seeds.append(env_seed)
        else:
            # Try generic setter on the wrapped env
            if hasattr(self.torchrl_env, 'set_seed'):
                try:
                    self.torchrl_env.set_seed(seed)
                except Exception:
                    pass
            seeds = [seed + i for i in range(self.num_envs)]
        return seeds
    
    def close(self):
        """Close all environments."""
        if hasattr(self.torchrl_env, 'envs'):
            for env in self.torchrl_env.envs:
                try:
                    env.close()
                except Exception:
                    pass
        elif hasattr(self.torchrl_env, 'close'):
            try:
                self.torchrl_env.close()
            except Exception:
                pass
        elif hasattr(self, 'envs'):
            for wrapper in self.envs:
                env_obj = getattr(wrapper, 'env', None)
                if env_obj is not None:
                    try:
                        env_obj.close()
                    except Exception:
                        pass
    
    def set_attr(self, attr_name: str, value: Any, indices=None):
        """Set attribute on underlying environments."""
        if indices is None:
            indices = range(self.num_envs)
        
        for idx in indices:
            env_obj = None
            if hasattr(self.torchrl_env, 'envs'):
                try:
                    env_obj = self.torchrl_env.envs[idx]
                except Exception:
                    env_obj = None
            elif hasattr(self, 'envs') and idx < len(self.envs):
                env_obj = getattr(self.envs[idx], 'env', None)

            if env_obj is not None:
                try:
                    setattr(env_obj, attr_name, value)
                except Exception:
                    pass
            else:
                # Try to set attribute on the parallel wrapper if present
                if hasattr(self.torchrl_env, 'set_attr'):
                    try:
                        self.torchrl_env.set_attr(attr_name, value, indices=[idx])
                    except Exception:
                        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped with a given wrapper."""
        if indices is None:
            indices = range(self.num_envs)
        
        results = []
        for idx in indices:
            env_obj = None
            if hasattr(self.torchrl_env, 'envs'):
                try:
                    env_obj = self.torchrl_env.envs[idx]
                except Exception:
                    env_obj = None
            elif hasattr(self, 'envs') and idx < len(self.envs):
                env_obj = getattr(self.envs[idx], 'env', None)

            is_wrapped = isinstance(env_obj, wrapper_class) if env_obj is not None else False
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
    from sb3_code.sb3_model_eval import evaluate_policy
    from torchrl.envs import EnvBase
    
    # Get device from actor parameters
    device = next(actor.parameters()).device
    
    # Wrap TorchRL environment if needed (both custom batched and plain EnvBase)
    if isinstance(env, EnvBase):
        # It's a TorchRL environment, wrap it
        env = TorchRLEnvWrapper(env)
    elif hasattr(env, 'type_') and env.type_ == "custom_batched":
        # Legacy custom batched env
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
    from sb3_code.sb3_model_eval import eval_corruptions
    from torchrl.envs import EnvBase
    
    # Get device from actor parameters
    device = next(actor.parameters()).device
    
    # Wrap TorchRL environment if needed (both custom batched and plain EnvBase)
    # Check if it's a TorchRL environment that needs wrapping
    if isinstance(env, EnvBase):
        # It's a TorchRL environment, wrap it
        env = TorchRLEnvWrapper(env)
    elif hasattr(env, 'type_') and env.type_ == "custom_batched":
        # Legacy custom batched env
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
