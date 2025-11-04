# env_factory.py — Environment factory for TorchRL (supports both parallel and sequential)
# Returns (train_env, valid_env, callback_env) — ParallelEnv or single EnvBase based on use_parallel_envs.

from __future__ import annotations
from typing import Any, Callable, Optional, List
import sys
import numpy as np
import torch
from torchrl.envs import ParallelEnv, EnvBase

from env import LogicEnv_gym as LogicEnv


def _seed_stream(n: int, base_seed: int) -> List[int]:
    ss = np.random.SeedSequence(int(base_seed))
    rng = np.random.Generator(np.random.PCG64(ss))
    return rng.integers(0, 2**31 - 1, size=int(n), dtype=np.int64).tolist()


def _make_single_env(index_manager, data_handler, *, queries, labels, query_depths, facts,
                     mode: str, seed: Optional[int], args: Any, kge_engine=None) -> Callable[[], EnvBase]:
    def _init():
        env = LogicEnv(
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
            seed=int(seed) if seed is not None else None,
            max_depth=args.max_depth,
            memory_pruning=args.memory_pruning,
            endt_action=args.endt_action,
            endf_action=args.endf_action,
            skip_unary_actions=args.skip_unary_actions,
            padding_atoms=args.padding_atoms,
            padding_states=args.padding_states,
            device=torch.device('cpu'),
            engine=args.engine,
            kge_action=args.kge_action,
            reward_type=args.reward_type,
            shaping_beta=getattr(args, 'pbrs_beta', 0.0),
            shaping_gamma=getattr(args, 'pbrs_gamma', args.gamma),
            kge_inference_engine=kge_engine,
            verbose=getattr(args, 'verbose_env', 0),
            prover_verbose=getattr(args, 'verbose_prover', 0),
        )
        return env
    return _init


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
    return ParallelEnv(
        num_workers=n_workers,
        create_env_fn=env_fns,
        shared_memory=True,
        mp_start_method=mp_start_method,
    )


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


def create_environments(args: Any, data_handler, index_manager, kge_engine=None, detailed_eval_env: bool = False):
    """Create TorchRL environments for training/validation/callback.

    Returns
    -------
    (train_env, valid_env, callback_env)
        train_env: ParallelEnv or single EnvBase with args.n_envs_train workers over TRAIN split.
        valid_env: ParallelEnv or single EnvBase with args.n_envs_eval workers over VALID split.
        callback_env: ParallelEnv or single EnvBase with args.n_envs_cb workers over VALID (restart) for detailed metrics.
    """
    # Generate seeds for different environment types
    seeds_train = _seed_stream(args.n_envs_train, args.seed_run_i)
    seeds_eval = _seed_stream(args.n_envs_eval, args.seed_run_i + 10_000)
    seeds_cb = _seed_stream(args.n_envs_cb, args.seed_run_i + 20_000)

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
            n_workers=args.n_envs_train,
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
            n_workers=args.n_envs_eval,
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
            n_workers=args.n_envs_cb,
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
