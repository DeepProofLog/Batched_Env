"""
Profile MCTS evaluation using evaluate_batched().

Usage:
    python kge_experiments/mcts/tests/profile_eval.py
    python kge_experiments/mcts/tests/profile_eval.py --n-queries 100 --n-corruptions 100
"""

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'kge_experiments'))

import argparse
import cProfile
import pstats
import io
from datetime import datetime
from time import time

import torch

from kge_experiments.mcts.config import MCTSConfig
from kge_experiments.mcts.trainer import MuZeroTrainer
from kge_experiments.builder import create_env, create_policy, KGEConfig
from kge_experiments.data_handler import DataHandler
from kge_experiments.nn.sampler import Sampler
from kge_experiments.index_manager import IndexManager


def main():
    parser = argparse.ArgumentParser(description='Profile MCTS evaluation')
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--n-queries', type=int, default=100)
    parser.add_argument('--n-corruptions', type=int, default=100)
    parser.add_argument('--num-simulations', type=int, default=25,
                        help='Number of MCTS simulations for training')
    parser.add_argument('--eval-num-simulations', type=int, default=None,
                        help='Number of MCTS simulations for evaluation (default: uses num-simulations)')
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--cpu-profile', action='store_true')
    parser.add_argument('--use-mcts-search', action='store_true',
                        help='Use MCTS search for action selection (slower but more accurate)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    print(f"MCTS Evaluation Profile")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    B = args.batch_size
    n_queries = args.n_queries
    n_corruptions = args.n_corruptions
    total_candidates = n_queries * (1 + n_corruptions) * 2  # 2 modes
    dataset = args.dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Num simulations: {args.num_simulations}")
    print(f"  N queries: {n_queries}")
    print(f"  N corruptions: {n_corruptions}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Compile: {args.compile}")
    print(f"  Use MCTS search: {args.use_mcts_search}")
    if args.use_mcts_search:
        eval_sims = args.eval_num_simulations if args.eval_num_simulations else args.num_simulations
        print(f"  Eval simulations: {eval_sims}")

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

    max_actions = env.padding_states
    eval_sims = args.eval_num_simulations if args.eval_num_simulations else args.num_simulations
    mcts_config = MCTSConfig(
        num_simulations=args.num_simulations,
        eval_num_simulations=eval_sims,
        max_episode_steps=20,
        mcts_batch_size=B,
        max_actions=max_actions,
        use_batched_mcts=True,
        device=str(device),
        compile=args.compile,
    )

    trainer = MuZeroTrainer(
        config=mcts_config,
        env=env,
        policy=policy,
    )

    # Load test data and sampler
    dh = DataHandler(dataset_name=dataset, base_path=data_path)
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

    test_queries = torch.stack([
        im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
        for q in dh.test_queries[:n_queries]
    ]).to(device)

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

    env.train()

    # Warmup
    print("\nRunning warmup...")
    trainer.evaluate_batched(
        queries=test_queries[:5],
        sampler=sampler,
        n_corruptions=5,
        corruption_modes=('head',),
        verbose=False,
        use_mcts_search=args.use_mcts_search,
    )
    torch.cuda.synchronize()

    # Profile evaluation
    print(f"\nProfiling evaluation: {n_queries} queries...")

    if args.cpu_profile:
        profiler = cProfile.Profile()
        profiler.enable()

    torch.cuda.synchronize()
    start = time()

    results = trainer.evaluate_batched(
        queries=test_queries,
        sampler=sampler,
        n_corruptions=n_corruptions,
        corruption_modes=('head', 'tail'),
        verbose=False,
        use_mcts_search=args.use_mcts_search,
    )

    torch.cuda.synchronize()
    elapsed = time() - start

    if args.cpu_profile:
        profiler.disable()

    # Calculate metrics
    ms_per_candidate = (elapsed / total_candidates) * 1000
    candidates_per_sec = total_candidates / elapsed

    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Runtime:         {elapsed:.3f}s")
    print(f"Candidates:      {total_candidates}")
    print(f"ms/candidate:    {ms_per_candidate:.4f}")
    print(f"Candidates/sec:  {candidates_per_sec:.1f}")

    print(f"\nResults:")
    print(f"  MRR:        {results['MRR']:.4f}")
    print(f"  Hits@1:     {results['Hits@1']:.4f}")
    print(f"  Hits@10:    {results['Hits@10']:.4f}")
    print(f"  proven_pos: {results['proven_pos']:.4f}")
    print(f"  proven_neg: {results['proven_neg']:.4f}")

    if args.cpu_profile:
        print(f"\n{'='*60}")
        print("CPU BOTTLENECKS (top 20 by cumulative time)")
        print(f"{'='*60}")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs().sort_stats('cumulative').print_stats(20)
        print(s.getvalue())


if __name__ == "__main__":
    main()
