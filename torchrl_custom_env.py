import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional, List

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv

from torchrl_env import LogicEnv_gym


def create_environments(args, data_handler, index_manager, kge_engine=None, detailed_eval_env=False):
    """
    Creates and seeds the training, evaluation, and callback environments for TorchRL.
    
    Returns TorchRL-compatible batched environments.
    """
    facts_set = set(data_handler.facts)
    shaping_gamma = args.pbrs_gamma if args.pbrs_gamma is not None else args.gamma

    def make_env(mode='train', seed=0, queries=None, labels=None, query_depths=None, 
                 facts=None, verbose=0, prover_verbose=0):
        """Factory function to create a single environment instance."""
        def _init():
            env = LogicEnv_gym(
                index_manager=index_manager,
                data_handler=data_handler,
                queries=queries,
                labels=labels,
                query_depths=query_depths,
                facts=facts,
                mode=mode,
                corruption_mode=args.corruption_mode,
                corruption_scheme=args.corruption_scheme,
                train_neg_ratio=args.train_neg_ratio,
                seed=seed,
                max_depth=args.max_depth,
                memory_pruning=args.memory_pruning,
                endt_action=args.endt_action,
                endf_action=args.endf_action,
                skip_unary_actions=args.skip_unary_actions,
                padding_atoms=args.padding_atoms,
                padding_states=args.padding_states,
                device='cpu',  # Individual envs on CPU, batching handles device placement
                engine=args.engine,
                kge_action=args.kge_action,
                reward_type=args.reward_type,
                shaping_beta=args.pbrs_beta,
                shaping_gamma=shaping_gamma,
                kge_inference_engine=kge_engine,
                verbose=verbose,
                prover_verbose=prover_verbose,
            )
            return env
        return _init

    # Generate seeds for different environment sets
    ss = np.random.SeedSequence(args.seed_run_i)
    child_seeds = ss.spawn(3)
    rng_env = np.random.Generator(np.random.PCG64(child_seeds[0]))
    rng_eval = np.random.Generator(np.random.PCG64(child_seeds[1]))
    rng_callback = np.random.Generator(np.random.PCG64(child_seeds[2]))

    env_seeds = rng_env.integers(0, 2**10, size=args.n_envs)
    eval_env_seeds = rng_eval.integers(0, 2**10, size=args.n_eval_envs)
    callback_env_seeds = rng_callback.integers(0, 2**10, size=1)

    # Create environment factory functions for each set
    env_fns = [make_env(
        mode='train',
        seed=int(env_seeds[i]),
        queries=data_handler.train_queries,
        labels=[1] * len(data_handler.train_queries),
        query_depths=data_handler.train_queries_depths,
        facts=facts_set,
        verbose=0,
        prover_verbose=0,
    ) for i in range(args.n_envs)]

    eval_env_fns = [make_env(
        mode='eval',
        seed=int(eval_env_seeds[i]),
        queries=data_handler.valid_queries,
        labels=[1] * len(data_handler.valid_queries),
        query_depths=data_handler.valid_queries_depths,
        facts=facts_set,
        verbose=0,
        prover_verbose=0,
    ) for i in range(args.n_eval_envs)]
    
    callback_env_fns = [make_env(
        mode='eval_with_restart',
        seed=int(callback_env_seeds[i]),
        queries=data_handler.valid_queries,
        labels=[1] * len(data_handler.valid_queries),
        query_depths=data_handler.valid_queries_depths,
        facts=facts_set,
        verbose=0,
        prover_verbose=0,
    ) for i in range(1)]

    # Create TorchRL batched environments
    # Use CustomBatchedEnv for all environments to preserve metadata like query_depth, label, etc.
    env = CustomBatchedEnv(args.n_envs, env_fns)
    eval_env = CustomBatchedEnv(args.n_eval_envs, eval_env_fns)
    
    if detailed_eval_env:
        callback_env = CustomBatchedEnv(1, callback_env_fns)
    else:
        callback_env = CustomBatchedEnv(1, callback_env_fns)

    # Mark the environment type for compatibility
    env.type_ = "torchrl_batched"
    eval_env.type_ = "custom_batched"
    callback_env.type_ = "custom_batched" if detailed_eval_env else "torchrl_batched"

    return env, eval_env, callback_env


class CustomBatchedEnv(EnvBase):
    """
    TorchRL-compatible batched environment that manages multiple environments.
    
    This replaces the SB3's CustomDummyVecEnv with TorchRL's batching paradigm.
    Supports episode counting and selective environment activation.
    """

    def __init__(self, num_envs: int, env_fns: List[Callable[[], LogicEnv_gym]]):
        """
        Initialize batched environment.
        
        Args:
            num_envs: Number of parallel environments
            env_fns: List of callables that create environment instances
        """
        self.num_envs = num_envs
        self.envs = [fn() for fn in env_fns]
        
        if len({id(env) for env in self.envs}) != len(self.envs):
            raise ValueError("Each env_fn must return a fresh env instance")
        
        # Get specs from first environment
        env0 = self.envs[0]
        device = env0.device
        
        # Initialize parent with batched specs
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        
        # Copy specs from base environment but with batch dimension
        self.observation_spec = env0.observation_spec.expand(num_envs)
        self.action_spec = env0.action_spec.expand(num_envs)
        self.reward_spec = env0.reward_spec.expand(num_envs)
        self.done_spec = env0.done_spec.expand(num_envs)
        
        # Track per-env episode counts/targets (to be set externally)
        self._episode_count = torch.zeros(num_envs, dtype=torch.int32)
        # Use maximum int32 value instead of infinity for integer tensor
        self._episode_target = torch.full((num_envs,), fill_value=2147483647, dtype=torch.int32)
        self._episode_step_id = torch.zeros(num_envs, dtype=torch.int32)
        
        # Only step/reset active envs
        self.active_envs = torch.ones(num_envs, dtype=torch.bool)
        
        # Metadata
        self.type_ = "custom_batched"
        self._seeds = [None] * num_envs
        self._current_tds = [None] * num_envs
        
        # Store NonTensorData separately to avoid stacking issues
        # These are for debugging only and not needed by the policy
        self._debug_states = [None] * num_envs
        self._debug_derived_states = [None] * num_envs

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """Reset all active environments and return batched TensorDict."""
        batch_td_list = []
        
        for idx in range(self.num_envs):
            if self.active_envs[idx]:
                seed = self._seeds[idx] if self._seeds else None
                td = self.envs[idx].reset(seed=seed)
                self._current_tds[idx] = td
            else:
                # For inactive envs, use the last known state
                td = self._current_tds[idx] if self._current_tds[idx] is not None else self.envs[idx].reset()
            
            # Extract and store NonTensorData separately (for debugging)
            if "state" in td.keys():
                self._debug_states[idx] = td.get("state")
            if "derived_states" in td.keys():
                self._debug_derived_states[idx] = td.get("derived_states")
            
            # Remove NonTensorData from TensorDict before stacking
            # This prevents stacking errors when NonTensorData types are inconsistent
            td_clean = td.select(
                "sub_index", "derived_sub_indices", "action_mask", 
                "label", "done", "terminated",
                strict=False  # Don't fail if some keys are missing
            )
            
            batch_td_list.append(td_clean)
        
        # Stack all TensorDicts into a batch (now safe without NonTensorData)
        batched_td = torch.stack(batch_td_list, dim=0)
        
        # Debug: verify NonTensorData was removed
        if "state" in batched_td.keys() or "derived_states" in batched_td.keys():
            print(f"WARNING: NonTensorData still in batched TensorDict! Keys: {list(batched_td.keys())}")
        
        # Reset seeds after use
        self._seeds = [None] * self.num_envs
        
        return batched_td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Step all active environments with their respective actions.
        
        Args:
            tensordict: Batched TensorDict containing actions for each environment
            
        Returns:
            Batched TensorDict with next observations, rewards, dones, etc.
        """
        batch_td_list = []
        
        for idx in range(self.num_envs):
            if self.active_envs[idx]:
                # Extract action for this environment
                action_td = TensorDict(
                    {"action": tensordict["action"][idx]},
                    batch_size=torch.Size([])
                )
                
                # Step the environment
                next_td = self.envs[idx]._step(action_td)
                
                # Ensure consistent keys across all TensorDicts
                # Add is_success if missing (defaulting to False)
                if "is_success" not in next_td.keys():
                    next_td["is_success"] = torch.tensor([False], dtype=torch.bool, device=self.device)
                
                # Check if episode is done
                if next_td["done"]:
                    # Increment episode count
                    self._episode_count[idx] += 1
                    self._episode_step_id[idx] += 1
                    
                    # Add episode metadata
                    next_td["episode_idx"] = torch.tensor([self._episode_step_id[idx].item()], dtype=torch.int32, device=self.device)
                    
                    # If reached target, deactivate
                    if self._episode_count[idx] >= self._episode_target[idx]:
                        self.active_envs[idx] = False
                    
                    # Auto-reset for next step
                    reset_td = self.envs[idx].reset()
                    self._current_tds[idx] = reset_td
                else:
                    self._current_tds[idx] = next_td
                    # Add episode_idx even when not done, for consistency
                    if "episode_idx" not in next_td.keys():
                        next_td["episode_idx"] = torch.tensor([0], dtype=torch.int32, device=self.device)
                
                batch_td_list.append(next_td)
            else:
                # Inactive env: return last state with zero reward
                td = self._current_tds[idx].clone()
                # Add zero reward and set done=True for inactive envs
                td["reward"] = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                td["done"] = torch.tensor([True], dtype=torch.bool, device=self.device)
                # Ensure episode_idx and is_success exist for consistency
                if "episode_idx" not in td.keys():
                    td["episode_idx"] = torch.tensor([0], dtype=torch.int32, device=self.device)
                if "is_success" not in td.keys():
                    td["is_success"] = torch.tensor([False], dtype=torch.bool, device=self.device)
                batch_td_list.append(td)
        
        # Stack into batched TensorDict
        # First, extract and store NonTensorData separately
        for idx, td in enumerate(batch_td_list):
            if "state" in td.keys():
                self._debug_states[idx] = td.get("state")
            if "derived_states" in td.keys():
                self._debug_derived_states[idx] = td.get("derived_states")
        
        # Remove NonTensorData from all TensorDicts before stacking
        batch_td_clean = []
        
        # Define required keys that must be present in all TensorDicts
        required_keys = {
            "sub_index", "derived_sub_indices", "action_mask",
            "label", "done", "terminated", "reward",
            "episode_idx", "is_success"
        }
        
        # Define optional keys that might not be present in all TensorDicts
        optional_keys = {"query_type", "query_depth", "max_depth_reached", "truncated"}
        
        # Define scalar keys that need to be unsqueezed
        scalar_keys = {'label', 'done', 'terminated', 'truncated', 'reward', 
                      'max_depth_reached', 'query_depth', 'query_type', 'episode_idx', 'is_success'}
        
        # First pass: determine which optional keys are present and normalize all tensor shapes
        keys_to_include = set(required_keys)
        for td in batch_td_list:
            # Ensure all tensors have consistent shapes first
            for key in list(td.keys()):
                value = td[key]
                # Only process torch tensors, skip lists and other types (Non TensorData)
                if not isinstance(value, torch.Tensor):
                    continue
                
                # For scalar tensors in scalar_keys, reshape to [1] for consistency
                if value.ndim == 0 and key in scalar_keys:
                    td.set(key, value.unsqueeze(0))
            
            # Check which optional keys are present
            for key in optional_keys:
                if key in td.keys():
                    keys_to_include.add(key)
        
        # Second pass: add missing keys with proper shapes and select
        for td in batch_td_list:
            # Add missing optional keys with default values (all with shape [1])
            for key in keys_to_include:
                if key not in td.keys():
                    if key == "query_type":
                        td.set(key, torch.tensor([0], dtype=torch.long))
                    elif key == "query_depth":
                        td.set(key, torch.tensor([0], dtype=torch.long))
                    elif key == "max_depth_reached":
                        td.set(key, torch.tensor([False], dtype=torch.bool))
                    elif key == "truncated":
                        td.set(key, torch.tensor([False], dtype=torch.bool))
            
            # Select only the keys we want to include (all TensorDicts now have the same keys and shapes)
            td_clean = td.select(*keys_to_include, strict=True)
            batch_td_clean.append(td_clean)
        
        batched_td = torch.stack(batch_td_clean, dim=0)
        
        # Debug: verify NonTensorData was removed  
        if "state" in batched_td.keys() or "derived_states" in batched_td.keys():
            print(f"WARNING: NonTensorData still in batched TensorDict after step! Keys: {list(batched_td.keys())}")
        
        return batched_td

    def _set_seed(self, seed: Optional[int] = None) -> Optional[int]:
        """Set seed for all environments."""
        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()
        
        # Set different seeds for each environment
        torch.manual_seed(seed)
        seeds = torch.randint(0, 2**31, (self.num_envs,)).tolist()
        
        for idx, env in enumerate(self.envs):
            env.set_seed(seeds[idx])
        
        return seed
    
    def set_episode_targets(self, targets: List[int]):
        """Set the number of episodes each environment should run."""
        assert len(targets) == self.num_envs
        self._episode_target = torch.tensor(targets, dtype=torch.int32)
    
    def get_episode_counts(self) -> torch.Tensor:
        """Get the current episode count for each environment."""
        return self._episode_count.clone()
    
    def reset_episode_counts(self):
        """Reset episode counters and reactivate all environments."""
        self._episode_count.zero_()
        self._episode_step_id.zero_()
        self.active_envs.fill_(True)
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:
        """Set attribute in specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        
        for idx in indices:
            setattr(self.envs[idx], attr_name, value)
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get attribute from specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        
        return [getattr(self.envs[idx], attr_name) for idx in indices]
    
    def env_method(self, method_name: str, *method_args, 
                   indices: Optional[List[int]] = None, **method_kwargs) -> List[Any]:
        """Call method on specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        
        return [getattr(self.envs[idx], method_name)(*method_args, **method_kwargs) 
                for idx in indices]
    
    def get_debug_states(self, idx: Optional[int] = None):
        """
        Retrieve debug state information (NonTensorData).
        
        Args:
            idx: Environment index. If None, returns all.
            
        Returns:
            Dict with 'state' and 'derived_states' for the specified environment(s).
        """
        if idx is not None:
            return {
                "state": self._debug_states[idx],
                "derived_states": self._debug_derived_states[idx]
            }
        else:
            return {
                "states": self._debug_states,
                "derived_states": self._debug_derived_states
            }
