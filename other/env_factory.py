# env_factory.py — Environment factory for TorchRL (supports both parallel and sequential)
# Returns (train_env, valid_env, callback_env) — ParallelEnv or single EnvBase based on use_parallel_envs.

from __future__ import annotations
from typing import Any, Callable, Optional, List, Tuple
import sys
import numpy as np
import torch
from torchrl.envs import ParallelEnv, EnvBase

from env import LogicEnv_gym as LogicEnv


def _seed_stream(n: int, base_seed: int) -> List[int]:
    ss = np.random.SeedSequence(int(base_seed))
    rng = np.random.Generator(np.random.PCG64(ss))
    return rng.integers(0, 2**31 - 1, size=int(n), dtype=np.int64).tolist()


class PicklableSamplerWrapper:
    """Wrapper to make sampler picklable by storing reconstruction parameters.
    
    The sampler itself uses types.MethodType which isn't picklable, so we store
    the parameters needed to reconstruct it after unpickling.
    """
    
    def __init__(self, data_handler, index_manager, corruption_scheme, device):
        # Store only what's needed to reconstruct the sampler
        self.data_handler = data_handler
        self.index_manager = index_manager
        self.corruption_scheme = corruption_scheme
        self.device = device
        self._sampler = None
    
    def get_sampler(self):
        """Lazy initialization of sampler."""
        if self._sampler is None:
            from other.sampler_original import get_sampler
            self._sampler = get_sampler(
                data_handler=self.data_handler,
                index_manager=self.index_manager,
                corruption_scheme=self.corruption_scheme,
                device=self.device,
            )
        return self._sampler
    
    def __getstate__(self):
        """Pickle support - exclude the actual sampler."""
        state = self.__dict__.copy()
        # Don't pickle the sampler itself, it will be reconstructed
        state['_sampler'] = None
        return state
    
    def __setstate__(self, state):
        """Unpickle support - restore state without sampler."""
        self.__dict__.update(state)
        # _sampler is None and will be reconstructed on first use


class PicklableEnvCreator:
    """Picklable environment creator for multiprocessing compatibility.
    
    This class wraps all the parameters needed to create a LogicEnv instance
    and can be pickled/unpickled for use with multiprocessing-based collectors
    like MultiSyncDataCollector.
    
    Strategy: Instead of trying to pickle data_handler with its unpicklable sampler,
    we use a PicklableSamplerWrapper that stores reconstruction parameters and
    lazily creates the sampler when needed.
    """
    
    def __init__(self, index_manager, data_handler, queries, labels, query_depths, 
                 facts, mode: str, seed: Optional[int], args: Any, kge_engine=None):
        # Store direct parameters
        self.queries = queries
        self.labels = labels
        self.query_depths = query_depths
        self.facts = facts
        self.mode = mode
        self.seed = seed
        self.kge_engine = kge_engine
        
        # Store references to objects
        self.index_manager = index_manager
        self.data_handler = data_handler
        
        # Extract and store individual args values to avoid pickling the args object
        self.corruption_mode = args.corruption_mode
        self.corruption_scheme = args.corruption_scheme
        self.train_neg_ratio = args.train_neg_ratio
        self.max_depth = args.max_depth
        self.memory_pruning = args.memory_pruning
        self.endt_action = args.endt_action
        self.endf_action = args.endf_action
        self.skip_unary_actions = args.skip_unary_actions
        self.padding_atoms = args.padding_atoms
        self.padding_states = args.padding_states
        self.engine = args.engine
        self.kge_action = args.kge_action
        self.reward_type = args.reward_type
        self.pbrs_beta = getattr(args, 'pbrs_beta', 0.0)
        self.pbrs_gamma = getattr(args, 'pbrs_gamma', args.gamma)
        self.verbose_env = getattr(args, 'verbose_env', 0)
        self.verbose_prover = getattr(args, 'verbose_prover', 0)
    
    def __call__(self) -> EnvBase:
        """Create and return a LogicEnv instance."""
        # Ensure data_handler has a sampler if corruption_mode is enabled
        # The sampler may have been excluded during pickling
        if self.corruption_mode and (not hasattr(self.data_handler, 'sampler') or self.data_handler.sampler is None):
            # Import here to avoid circular dependencies
            from other.sampler_original import get_sampler
            from index_manager import IndexManager
            
            # Reconstruct the sampler
            self.data_handler.sampler = get_sampler(
                data_handler=self.data_handler,
                index_manager=self.index_manager,
                corruption_scheme=self.corruption_scheme,
                device=torch.device('cpu'),
            )
        
        env = LogicEnv(
            index_manager=self.index_manager,
            data_handler=self.data_handler,
            queries=self.queries,
            labels=self.labels,
            query_depths=self.query_depths,
            facts=self.facts,
            mode=self.mode,
            corruption_mode=self.corruption_mode,
            corruption_scheme=self.corruption_scheme,
            train_neg_ratio=self.train_neg_ratio,
            seed=int(self.seed) if self.seed is not None else None,
            max_depth=self.max_depth,
            memory_pruning=self.memory_pruning,
            endt_action=self.endt_action,
            endf_action=self.endf_action,
            skip_unary_actions=self.skip_unary_actions,
            padding_atoms=self.padding_atoms,
            padding_states=self.padding_states,
            device=torch.device('cpu'),
            engine=self.engine,
            kge_action=self.kge_action,
            reward_type=self.reward_type,
            shaping_beta=self.pbrs_beta,
            shaping_gamma=self.pbrs_gamma,
            kge_inference_engine=self.kge_engine,
            verbose=self.verbose_env,
            prover_verbose=self.verbose_prover,
        )
        return env


def _make_single_env(index_manager, data_handler, *, queries, labels, query_depths, facts,
                     mode: str, seed: Optional[int], args: Any, kge_engine=None) -> Callable[[], EnvBase]:
    """Create a picklable environment creator.
    
    Returns a PicklableEnvCreator instance that can be called to create an environment.
    This function is kept for backward compatibility but now returns a picklable object.
    """
    return PicklableEnvCreator(
        index_manager=index_manager,
        data_handler=data_handler,
        queries=queries,
        labels=labels,
        query_depths=query_depths,
        facts=facts,
        mode=mode,
        seed=seed,
        args=args,
        kge_engine=kge_engine,
    )


def _create_parallel_env(queries, labels, query_depths, mode, n_workers, seeds, args, index_manager, data_handler, kge_engine, mp_start_method):
    """Helper function to create a parallel environment with given configuration."""
    facts_set = getattr(data_handler, "facts_set", None)
    if facts_set is None:
        facts_set = frozenset(data_handler.facts)
    env_fns = [
        _make_single_env(
            index_manager, data_handler,
            queries=queries,
            labels=labels,
            query_depths=query_depths,
            facts=facts_set,
            mode=mode, seed=int(seeds[i]), args=args, kge_engine=kge_engine,
        )
        for i in range(n_workers)
    ]
    parallel_env = ParallelEnv(
        num_workers=n_workers,
        create_env_fn=env_fns,
        shared_memory=True,
        mp_start_method=mp_start_method,
        pin_memory=False,  # Don't use pinned memory - can cause issues with CPU tensors
    )
    # Store env_fns as an attribute for SyncDataCollector
    parallel_env.env_fns = env_fns
    return parallel_env


def _create_single_env(queries, labels, query_depths, mode, seed, args, index_manager, data_handler, kge_engine):
    """Helper function to create a single environment."""
    facts_set = getattr(data_handler, "facts_set", None)
    if facts_set is None:
        facts_set = frozenset(data_handler.facts)
    env_fn = _make_single_env(
        index_manager, data_handler,
        queries=queries,
        labels=labels,
        query_depths=query_depths,
        facts=facts_set,
        mode=mode, seed=int(seed), args=args, kge_engine=kge_engine,
    )
    return env_fn()  # Call the function to get the actual environment


def create_environments(args: Any, data_handler, index_manager, kge_engine=None, 
                        detailed_eval_env: bool = False,
                        device: torch.device = torch.device('cpu')) -> Tuple[EnvBase, EnvBase, EnvBase]:
    """Create TorchRL environments for training/validation/callback.

    Returns
    -------
    (train_env, valid_env, callback_env)
        train_env: ParallelEnv or single EnvBase with args.n_envs workers over TRAIN split.
        valid_env: ParallelEnv or single EnvBase with args.n_eval_envs workers over VALID split.
        callback_env: ParallelEnv or single EnvBase with 1 worker over VALID (restart) for detailed metrics.
    """
    # Generate seeds for different environment types
    n_train_envs = getattr(args, 'n_envs', 1)
    n_eval_envs = getattr(args, 'n_eval_envs', 1)
    
    seeds_train = _seed_stream(n_train_envs, args.seed_run_i)
    seeds_eval = _seed_stream(n_eval_envs, args.seed_run_i + 10_000)
    seeds_cb = _seed_stream(1, args.seed_run_i + 20_000)

    use_parallel = getattr(args, 'use_parallel_envs', True)

    if use_parallel:
        preferred_mp_method: Optional[str] = getattr(args, "parallel_env_start_method", None)
        if preferred_mp_method is None or preferred_mp_method == "auto":
            mp_start_method = "fork" if sys.platform.startswith("linux") else None
        else:
            mp_start_method = preferred_mp_method

        # Create training environment
        train_env = _create_parallel_env(
            queries=data_handler.train_queries,
            labels=[1] * len(data_handler.train_queries),
            query_depths=data_handler.train_queries_depths,
            mode='train',
            n_workers=n_train_envs,
            seeds=seeds_train,
            args=args,
            index_manager=index_manager,
            data_handler=data_handler,
            kge_engine=kge_engine,
            mp_start_method=mp_start_method,
        )

        # Create evaluation environment
        eval_env = _create_parallel_env(
            queries=data_handler.valid_queries,
            labels=[1] * len(data_handler.valid_queries),
            query_depths=data_handler.valid_queries_depths,
            mode='eval',
            n_workers=n_eval_envs,
            seeds=seeds_eval,
            args=args,
            index_manager=index_manager,
            data_handler=data_handler,
            kge_engine=kge_engine,
            mp_start_method=mp_start_method,
        )

        # Create callback environment
        callback_env = _create_parallel_env(
            queries=data_handler.valid_queries,
            labels=[1] * len(data_handler.valid_queries),
            query_depths=data_handler.valid_queries_depths,
            mode='eval_with_restart',
            n_workers=1,
            seeds=seeds_cb,
            args=args,
            index_manager=index_manager,
            data_handler=data_handler,
            kge_engine=kge_engine,
            mp_start_method=mp_start_method,
        )
    else:
        # Create single environments
        train_env = _create_single_env(
            queries=data_handler.train_queries,
            labels=[1] * len(data_handler.train_queries),
            query_depths=data_handler.train_queries_depths,
            mode='train',
            seed=seeds_train[0],  # Use first seed
            args=args,
            index_manager=index_manager,
            data_handler=data_handler,
            kge_engine=kge_engine,
        )

        eval_env = _create_single_env(
            queries=data_handler.valid_queries,
            labels=[1] * len(data_handler.valid_queries),
            query_depths=data_handler.valid_queries_depths,
            mode='eval',
            seed=seeds_eval[0],
            args=args,
            index_manager=index_manager,
            data_handler=data_handler,
            kge_engine=kge_engine,
        )

        callback_env = _create_single_env(
            queries=data_handler.valid_queries,
            labels=[1] * len(data_handler.valid_queries),
            query_depths=data_handler.valid_queries_depths,
            mode='eval_with_restart',
            seed=seeds_cb[0],
            args=args,
            index_manager=index_manager,
            data_handler=data_handler,
            kge_engine=kge_engine,
        )

    return train_env, eval_env, callback_env