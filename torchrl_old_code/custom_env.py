import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional, List

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv

# from env import LogicEnv_gym
from env import LogicEnv_gym


def create_environments(args, data_handler, index_manager, kge_engine=None, detailed_eval_env=False, device=None):
    """
    Creates and seeds the training, evaluation, and callback environments for TorchRL.
    
    Returns TorchRL-compatible batched environments.
    """
    # Default to CPU if no device is provided
    if device is None:
        device = torch.device('cpu')
    
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
                device=device,  # Use the provided device instead of hardcoding to CPU
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
        verbose=getattr(args, 'verbose_env', 0),
        prover_verbose=getattr(args, 'verbose_prover', 0),
    ) for i in range(args.n_envs)]

    eval_env_fns = [make_env(
        mode='eval',
        seed=int(eval_env_seeds[i]),
        queries=data_handler.valid_queries,
        labels=[1] * len(data_handler.valid_queries),
        query_depths=data_handler.valid_queries_depths,
        facts=facts_set,
        verbose=getattr(args, 'verbose_env', 0),
        prover_verbose=getattr(args, 'verbose_prover', 0),
    ) for i in range(args.n_eval_envs)]
    
    callback_env_fns = [make_env(
        mode='eval_with_restart',
        seed=int(callback_env_seeds[i]),
        queries=data_handler.valid_queries,
        labels=[1] * len(data_handler.valid_queries),
        query_depths=data_handler.valid_queries_depths,
        facts=facts_set,
        verbose=getattr(args, 'verbose_env', 0),
        prover_verbose=getattr(args, 'verbose_prover', 0),
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
            env_fns: List of callable environment factories
        """
        self.num_envs = num_envs
        self.env_fns = env_fns  # Store env_fns for MultiaSyncDataCollector
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
                if "next" in td.keys():
                    next_branch = td.get("next")
                    for key in next_branch.keys():
                        td.set(key, next_branch.get(key))
                    td = td.exclude("next")
                self._current_tds[idx] = td
            else:
                # For inactive envs, use the last known state
                if self._current_tds[idx] is not None:
                    td = self._current_tds[idx]
                else:
                    td = self.envs[idx].reset()
                    if "next" in td.keys():
                        next_branch = td.get("next")
                        for key in next_branch.keys():
                            td.set(key, next_branch.get(key))
                        td = td.exclude("next")
            
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
            
            # Ensure all tensors are on the correct device
            td_clean = td_clean.to(self.device)
            
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
                action_val = tensordict["action"][idx].item()
                
                # CRITICAL FIX: The batched tensordict's action_mask can become stale as we iterate
                # through environments. Each env.step() modifies its internal action_mask, but the
                # batched tensordict may not reflect these changes immediately.
                #
                # Solution: Use the snapshot mask from sampling time to temporarily restore the
                # environment's action_mask to what it was when the action was sampled.
                
                action_mask_at_sample = tensordict.get("_action_mask_at_sample_time", None)
                
                if action_mask_at_sample is not None:
                    sample_mask = action_mask_at_sample[idx]
                    
                    # Check environment's CURRENT internal mask
                    env_current_mask = self.envs[idx].tensordict.get("action_mask", None)
                    
                    if env_current_mask is not None:
                        # Ensure both masks are on the same device for comparison
                        sample_mask_cpu = sample_mask.cpu() if sample_mask.is_cuda else sample_mask
                        env_mask_cpu = env_current_mask.cpu() if env_current_mask.is_cuda else env_current_mask
                        
                        masks_equal = torch.equal(sample_mask_cpu, env_mask_cpu)
                        
                        if not masks_equal:
                            # Environment's mask has changed since sampling!
                            valid_at_sample = sample_mask[action_val].item() if action_val < len(sample_mask) else False
                            valid_now = env_current_mask[action_val].item() if action_val < len(env_current_mask) else False
                            
                            print(f"\n{'='*80}")
                            print(f"[ParallelEnvWrapper] MASK DESYNC DETECTED FOR ENV {idx}")
                            print(f"  Action: {action_val}")
                            print(f"  Valid at sample time: {valid_at_sample} (from {torch.where(sample_mask)[0].tolist()})")
                            print(f"  Valid now: {valid_now} (from {torch.where(env_current_mask)[0].tolist()})")
                            
                            if valid_at_sample and not valid_now:
                                print(f"  FIX: Restoring environment's action_mask to sample-time state")
                                # Restore the snapshot
                                self.envs[idx].tensordict.set("action_mask", sample_mask.clone())
                            else:
                                print(f"  No fix needed - action is {'valid' if valid_now else 'invalid'} in current state")
                            
                            print("="*80 + "\n")
                
                action_td = TensorDict(
                    {"action": tensordict["action"][idx]},
                    batch_size=torch.Size([])
                )
                
                # Step the single env
                next_td = self.envs[idx]._step(action_td)

                # Flatten TorchRL-style ("next", key) entries to root level for downstream code
                # IMPORTANT: Preserve episode metadata (label, query_depth, etc.) from completed episodes
                # These are set at the root level by the env when done=True for callback tracking
                if "next" in next_td.keys():
                    # First, check if this is a completed episode and preserve its metadata
                    metadata_keys = ["label", "query_depth", "is_success", "query_type"]
                    preserved_metadata = {}
                    
                    # Check if episode is done (metadata is only relevant when done=True)
                    is_done = next_td.get("done", torch.tensor([False]))
                    is_done_scalar = bool(is_done.reshape(-1)[0].item())
                    
                    if is_done_scalar:
                        # Preserve metadata from the COMPLETED episode (stored at root level)
                        for key in metadata_keys:
                            if key in next_td.keys():
                                preserved_metadata[key] = next_td[key].clone()
                    
                    # Now flatten the "next" branch (which contains the new episode's observation)
                    next_branch = next_td.get("next")
                    for key in next_branch.keys():
                        next_td.set(key, next_branch.get(key))
                    next_td = next_td.exclude("next")
                    
                    # Restore the completed episode's metadata so callbacks can access it
                    if is_done_scalar:
                        for key, value in preserved_metadata.items():
                            next_td[key] = value
                
                # Ensure consistent keys across all TensorDicts
                # Add is_success if missing (defaulting to False)
                if "is_success" not in next_td.keys():
                    next_td["is_success"] = torch.tensor([False], dtype=torch.bool, device=self.device)
                

                # --- robust scalar done check ---
                done_tensor = next_td["done"]
                done_scalar = bool(done_tensor.reshape(-1)[0].item())

                if done_scalar:
                    # Increment episode count
                    self._episode_count[idx] += 1
                    self._episode_step_id[idx] += 1
                    
                    # Add episode metadata
                    next_td["episode_idx"] = torch.tensor(
                        [int(self._episode_step_id[idx].item())],
                        dtype=torch.int32,
                        device=self.device
                    )
                    # If reached target, deactivate
                    if self._episode_count[idx] >= self._episode_target[idx]:
                        self.active_envs[idx] = False
                    
                    # NOTE: The individual environment already auto-resets in its _step method
                    # when done=True. The next_td already contains the reset state with the
                    # completed episode's metadata preserved. We don't need to reset again here.
                    # Just update our cached state.
                    self._current_tds[idx] = next_td
                else:
                    self._current_tds[idx] = next_td
                    # Add episode_idx even when not done, for consistency
                    if "episode_idx" not in next_td.keys():
                        next_td["episode_idx"] = torch.tensor([0], dtype=torch.int32, device=self.device)
                
                # Ensure all tensors in next_td are on the correct device
                next_td = next_td.to(self.device)
                batch_td_list.append(next_td)
            else:
                # Inactive envs: return last known state with zeroed reward & done=True
                td = self._current_tds[idx].clone()
                # Add zero reward and set done=True for inactive envs
                td["reward"] = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                td["done"] = torch.tensor([True], dtype=torch.bool, device=self.device)
                # Ensure episode_idx and is_success exist for consistency
                if "episode_idx" not in td.keys():
                    td["episode_idx"] = torch.tensor([0], dtype=torch.int32, device=self.device)
                if "is_success" not in td.keys():
                    td["is_success"] = torch.tensor([False], dtype=torch.bool, device=self.device)
                # Ensure all tensors are on the correct device
                td = td.to(self.device)
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
        # First pass: ensure scalars are [1]-shaped and detect present optional keys

        for td in batch_td_list:
            for k in scalar_keys:
                if k in td.keys():
                    v = td.get(k)
                    if v.dim() == 0:
                        td.set(k, v.unsqueeze(0))
            for ok in optional_keys:
                if ok in td.keys():
                    keys_to_include.add(ok)
        
        # Second pass: add missing keys with proper shapes and select
        for td in batch_td_list:
            for k in keys_to_include:
                if k not in td.keys():
                    if k == "query_type":
                        td.set(k, torch.tensor([0], dtype=torch.long, device=self.device))
                    elif k == "query_depth":
                        td.set(k, torch.tensor([0], dtype=torch.long, device=self.device))
                    elif k == "max_depth_reached":
                        td.set(k, torch.tensor([False], dtype=torch.bool, device=self.device))
                    elif k == "truncated":
                        td.set(k, torch.tensor([False], dtype=torch.bool, device=self.device))
            td_clean = td.select(*keys_to_include, strict=True)
            # Ensure all tensors in the cleaned TD are on the correct device
            td_clean = td_clean.to(self.device)
            batch_td_clean.append(td_clean)
        
        batched_td = torch.stack(batch_td_clean, dim=0)
        
        # Sanity: NonTensorData must not be in the stacked batch
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
