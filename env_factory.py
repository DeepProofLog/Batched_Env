# env_factory.py — Parallel environment factory (SB3-free)
# Returns (train_env, valid_env, callback_env) — all built with ParallelEnv.

from __future__ import annotations
from typing import Any, Callable, Optional, List
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


def _create_parallel_env(queries, labels, query_depths, mode, n_workers, seeds, args, index_manager, data_handler, kge_engine):
    """Helper function to create a parallel environment with given configuration."""
    facts_set = set(data_handler.facts)
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
    return ParallelEnv(num_workers=n_workers, create_env_fn=env_fns, shared_memory=True)


def create_environments(args: Any, data_handler, index_manager, kge_engine=None, detailed_eval_env: bool = False):
    """Create parallel TorchRL environments for training/validation/callback.

    Returns
    -------
    (train_env, valid_env, callback_env)
        train_env: ParallelEnv with args.n_envs_train workers over TRAIN split.
        valid_env: ParallelEnv with args.n_envs_eval workers over VALID split.
        callback_env: ParallelEnv with args.n_envs_cb workers over VALID (restart) for detailed metrics.
    """
    # Generate seeds for different environment types
    seeds_train = _seed_stream(args.n_envs_train, args.seed_run_i)
    seeds_eval = _seed_stream(args.n_envs_eval, args.seed_run_i + 10_000)
    seeds_cb = _seed_stream(args.n_envs_cb, args.seed_run_i + 20_000)

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
        kge_engine=kge_engine
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
        kge_engine=kge_engine
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
        kge_engine=kge_engine
    )

    return train_env, eval_env, callback_env