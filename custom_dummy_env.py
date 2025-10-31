import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional, List

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv

from env import LogicEnv_gym


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
    # Use SerialEnv for sequential execution (like DummyVecEnv)
    env = SerialEnv(args.n_envs, env_fns)
    eval_env = CustomBatchedEnv(args.n_eval_envs, eval_env_fns)
    
    if detailed_eval_env:
        callback_env = CustomBatchedEnv(1, callback_env_fns)
    else:
        callback_env = SerialEnv(1, callback_env_fns)

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
            
            batch_td_list.append(td)
        
        # Stack all TensorDicts into a batch
        batched_td = torch.stack(batch_td_list, dim=0)
        
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
                
                # Check if episode is done
                if next_td["done"]:
                    # Increment episode count
                    self._episode_count[idx] += 1
                    self._episode_step_id[idx] += 1
                    
                    # Add episode metadata
                    next_td["episode_idx"] = self._episode_step_id[idx]
                    
                    # If reached target, deactivate
                    if self._episode_count[idx] >= self._episode_target[idx]:
                        self.active_envs[idx] = False
                    
                    # Auto-reset for next step
                    reset_td = self.envs[idx].reset()
                    self._current_tds[idx] = reset_td
                else:
                    self._current_tds[idx] = next_td
                
                batch_td_list.append(next_td)
            else:
                # Inactive env: return last state with zero reward
                td = self._current_tds[idx].clone()
                td["reward"] = torch.zeros_like(td["reward"])
                td["done"] = torch.tensor(True, dtype=torch.bool, device=self.device)
                batch_td_list.append(td)
        
        # Stack into batched TensorDict
        batched_td = torch.stack(batch_td_list, dim=0)
        
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
