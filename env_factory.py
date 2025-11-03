# env_factory.py — Parallel environment factory (SB3-free)
# Returns (train_env, valid_env, callback_env) — all built with ParallelEnv.

from __future__ import annotations
from typing import Any, Callable, Optional, List
import numpy as np
import torch
from torchrl.envs import ParallelEnv, EnvBase

from env import LogicEnv


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


def create_environments(args: Any, data_handler, index_manager, kge_engine=None, detailed_eval_env: bool = False):
    """Create parallel TorchRL environments for training/validation/callback.

    Returns
    -------
    (train_env, valid_env, callback_env)
        train_env: ParallelEnv with args.n_envs workers over TRAIN split.
        valid_env: ParallelEnv with args.n_eval_envs workers over VALID split.
        callback_env: ParallelEnv with 1 worker over VALID (restart) for detailed metrics.
    """
    facts_set = set(data_handler.facts)

    seeds_train = _seed_stream(args.n_envs, args.seed_run_i)
    seeds_eval = _seed_stream(args.n_eval_envs, args.seed_run_i + 10_000)
    seeds_cb = _seed_stream(1, args.seed_run_i + 20_000)

    # TRAIN — Parallel
    train_fns = [
        _make_single_env(
            index_manager, data_handler,
            queries=data_handler.train_queries,
            labels=[1] * len(data_handler.train_queries),
            query_depths=data_handler.train_queries_depths,
            facts=facts_set,
            mode='train', seed=int(seeds_train[i]), args=args, kge_engine=kge_engine,
        )
        for i in range(int(args.n_envs))
    ]
    train_env = ParallelEnv(num_workers=int(args.n_envs), create_env_fn=train_fns, shared_memory=True)

    # VALIDATION (mid-training eval) — Parallel
    eval_fns = [
        _make_single_env(
            index_manager, data_handler,
            queries=data_handler.valid_queries,
            labels=[1] * len(data_handler.valid_queries),
            query_depths=data_handler.valid_queries_depths,
            facts=facts_set,
            mode='eval', seed=int(seeds_eval[i]), args=args, kge_engine=kge_engine,
        )
        for i in range(int(args.n_eval_envs))
    ]
    eval_env = ParallelEnv(num_workers=int(args.n_eval_envs), create_env_fn=eval_fns, shared_memory=True)

    # CALLBACK (per-iter detailed eval with restart) — Parallel (1 worker)
    cb_fn = _make_single_env(
        index_manager, data_handler,
        queries=data_handler.valid_queries,
        labels=[1] * len(data_handler.valid_queries),
        query_depths=data_handler.valid_queries_depths,
        facts=facts_set,
        mode='eval_with_restart', seed=int(seeds_cb[0]), args=args, kge_engine=kge_engine,
    )
    callback_env = ParallelEnv(num_workers=1, create_env_fn=[cb_fn], shared_memory=True)

    return train_env, eval_env, callback_env