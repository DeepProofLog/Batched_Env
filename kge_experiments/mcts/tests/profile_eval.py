"""
Profile MCTS evaluation using the trainer's evaluate_batched() interface.

Usage:
    python kge_experiments/mcts/tests/profile_eval.py
"""

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'kge_experiments'))

import cProfile
import pstats
import io
from time import time

import torch

from kge_experiments.mcts.config import MCTSConfig
from kge_experiments.mcts.trainer import MuZeroTrainer
from kge_experiments.builder import create_env, create_policy, KGEConfig
from kge_experiments.data_handler import DataHandler
from kge_experiments.nn.sampler import Sampler
from kge_experiments.index_manager import IndexManager


def main():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Configuration
    B = 100  # Batch size for batched evaluation
    n_sims = 25
    n_queries = 50  # Number of test queries
    n_corruptions = 10  # Corruptions per query
    dataset = 'family'
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Num simulations: {n_sims}")
    print(f"  N queries: {n_queries}")
    print(f"  N corruptions: {n_corruptions}")
    print(f"  Total candidates: {n_queries * (1 + n_corruptions) * 2}")  # 2 modes

    # Build environment and policy
    print("\nBuilding environment and policy...")
    kge_config = KGEConfig(
        dataset=dataset,
        data_path=data_path,
        device=str(device),
        n_envs=B,
    )
    env = create_env(kge_config)
    policy = create_policy(kge_config, env)

    # MCTS-specific config - get max_actions from env
    max_actions = env.padding_states  # Actual action space size
    mcts_config = MCTSConfig(
        num_simulations=n_sims,
        max_episode_steps=20,
        mcts_batch_size=B,
        max_actions=max_actions,
        use_batched_mcts=True,  # Use batched MCTS
        device=str(device),
        compile=False,  # Disable compilation for now (TensorDict key access issue)
    )

    # Create trainer
    trainer = MuZeroTrainer(
        config=mcts_config,
        env=env,
        policy=policy,
        device=device,
    )

    # Load test data and sampler
    dh = DataHandler(
        dataset_name=dataset,
        base_path=data_path,
    )

    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=100,
        max_arity=dh.max_arity,
        padding_atoms=kge_config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)

    # Get test queries - convert strings to indices using IndexManager
    test_queries_raw = dh.test_queries[:n_queries]
    test_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        for q in test_queries_raw
    ]).to(device)

    # Create sampler
    d2i, e2d = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode='both',
        seed=42,
        domain2idx=d2i,
        entity2domain=e2d,
    )

    # Set env mode
    env.train()

    # Warmup (minimal)
    print("\nRunning warmup...")
    trainer.evaluate_batched(
        queries=test_queries[:5],
        sampler=sampler,
        n_corruptions=5,
        corruption_modes=('head',),
        verbose=False,
    )
    torch.cuda.synchronize()

    # Profile evaluation
    print(f"\nProfiling evaluation: {n_queries} queries...")

    profiler = cProfile.Profile()
    profiler.enable()

    torch.cuda.synchronize()
    start = time()

    # Run batched evaluation
    results = trainer.evaluate_batched(
        queries=test_queries,
        sampler=sampler,
        n_corruptions=n_corruptions,
        corruption_modes=('head', 'tail'),
        verbose=False,
    )

    torch.cuda.synchronize()
    elapsed = time() - start

    profiler.disable()

    # Calculate metrics
    total_candidates = n_queries * (1 + n_corruptions) * 2  # 2 modes
    queries_per_sec = n_queries / elapsed
    candidates_per_sec = total_candidates / elapsed

    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.3f}s for {n_queries} queries")
    print(f"Time per query: {elapsed/n_queries*1000:.2f} ms")
    print(f"Queries/sec: {queries_per_sec:.1f}")
    print(f"Candidates/sec: {candidates_per_sec:.1f}")
    print(f"\nResults:")
    print(f"  MRR: {results['MRR']:.4f}")
    print(f"  Hits@1: {results['Hits@1']:.4f}")
    print(f"  Hits@10: {results['Hits@10']:.4f}")
    print(f"  proven_pos: {results['proven_pos']:.4f}")
    print(f"  proven_neg: {results['proven_neg']:.4f}")

    print(f"\n{'='*60}")
    print("CPU BOTTLENECKS (top 20 by cumulative time)")
    print(f"{'='*60}")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    main()
