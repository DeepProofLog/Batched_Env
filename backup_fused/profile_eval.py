"""
Profile the PPO.evaluate() function.

This script profiles the  evaluation to verify performance
matches or exceeds the V10 implementation.

Targets:
- ms/candidate: ≤0.88 (matching V10)
- MRR: ≈0.16 (correctness)

Usage:
    conda activate rl
    python tests/profile_eval.py
    python tests/profile_eval.py --n-queries 100 --n-corruptions 100

Results are saved to profile_eval.txt for comparison.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cProfile
import pstats
import io
from datetime import datetime
from time import time
from types import SimpleNamespace

import torch


class Tee(object):
    """Write to multiple streams simultaneously."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return any(getattr(f, 'isatty', lambda: False)() for f in self.files)


def setup(device, config):
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable
    from policy import ActorCriticPolicy
    from nn.sampler import Sampler
    from env import EnvVec
    from ppo import PPO
    
    import unification
    unification.COMPILE_MODE = config.compile
    torch.set_float32_matmul_precision('high')
    
    dh = DataHandler(
        dataset_name=config.dataset, base_path=config.data_path,
        train_file="train.txt", valid_file="valid.txt", test_file="test.txt",
        rules_file="rules.txt", facts_file="train.txt", corruption_mode='dynamic',
    )
    
    im = IndexManager(
        constants=dh.constants, predicates=dh.predicates,
        max_total_runtime_vars=100, max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms, device=device, rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    d2i, e2d = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no, num_relations=im.predicate_no,
        device=device, default_mode='both', seed=42, domain2idx=d2i, entity2domain=e2d,
    )
    
    torch.manual_seed(42)
    
    embedder = EmbedderLearnable(
        n_constants=im.constant_no, n_predicates=im.predicate_no, n_vars=1000,
        max_arity=dh.max_arity, padding_atoms=config.padding_atoms,
        atom_embedder='transe', state_embedder='mean',
        constant_embedding_size=250, predicate_embedding_size=250, atom_embedding_size=250,
        device=str(device),
    )
    embedder.embed_dim = 250
    
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im, padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states, end_proof_action=True,
        max_fact_pairs_cap=config.max_fact_pairs_cap,
    )
    
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    train_queries = convert_queries_unpadded(dh.train_queries)
    test_queries = convert_queries_unpadded(dh.test_queries)
    
    # Use Env
    env = EnvVec(
        vec_engine=vec_engine,
        batch_size=config.batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=True,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=True,
        train_queries=train_queries,
        valid_queries=test_queries,
    )
    
    policy = ActorCriticPolicy(
        embedder=embedder, embed_dim=250, action_dim=config.padding_states,
        hidden_dim=256, num_layers=8, dropout_prob=0.0, device=device, compile_policy=False,
    ).to(device)
    
    # Create config for PPO
    ppo_config = SimpleNamespace(
        n_envs=config.batch_size,
        batch_size_env=config.batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        n_steps=32,
        learning_rate=1e-4,
        n_epochs=5,
        batch_size=config.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=False,
        seed=42,
        parity=False,
        eval_only=True,
        compile=config.compile,
        fixed_batch_size=config.batch_size,
        ranking_compile_mode='reduce-overhead',
    )
    
    # Use PPO
    ppo = PPO(policy, env, ppo_config, device=device)
    
    return {
        'ppo': ppo,
        'env': env,
        'sampler': sampler,
        'queries': test_queries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wn18rr')
    parser.add_argument('--n-queries', type=int, default=10)
    parser.add_argument('--n-corruptions', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--gpu-profile', action='store_true',  help='Run GPU profiler')
    parser.add_argument('--cpu-profile', action='store_true', help='Run CPU profiler')
    parser.add_argument('--no-profile', action='store_true', help='Run without any profiler (pure timing, default)')
    parser.add_argument('--max-fact-pairs-cap', type=int, default=None, help='Cap max_fact_pairs to limit tensor sizes')
    parser.add_argument('--eval-padding-states', type=int, default=None, help='Padding states for evaluation (default: same as padding_states)')
    parser.add_argument('--max-depth', type=int, default=20, help='Max steps per episode')
    args = parser.parse_args()

    # Setup output file for logging results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_eval.txt')
    log_file = open(output_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout

    try:
        if args.max_fact_pairs_cap is None and args.dataset == 'wn18rr':
            print("Setting max_fact_pairs_cap to 1000 for wn18rr")
            args.max_fact_pairs_cap = 1000

        # Default eval_padding_states based on dataset
        if args.eval_padding_states is None:
            args.eval_padding_states = {'wn18rr': 120, 'fb15k237': 120, 'family': 130}.get(args.dataset, 262)

        config = SimpleNamespace(
            dataset=args.dataset,
            data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
            padding_atoms=6, padding_states=args.eval_padding_states, max_depth=args.max_depth,
            batch_size=args.batch_size, compile=args.compile,
            max_fact_pairs_cap=args.max_fact_pairs_cap,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total = args.n_queries * (1 + args.n_corruptions) * 2

        print(f"Profile Eval Results")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")
        print(f"Dataset: {config.dataset}")
        print(f"\nConfiguration:")
        print(f"  N queries: {args.n_queries}")
        print(f"  N corruptions: {args.n_corruptions}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Padding states: {config.padding_states}")
        print(f"  Max depth: {config.max_depth}")
        print(f"  Total candidates: {total}")
        print(f"")

        print("Setup...")
        c = setup(device, config)

        print("Compile + warmup...")
        t0 = time()
        c['env'].compile(mode='reduce-overhead', fullgraph=True)
        # Warmup with small evaluation
        c['ppo'].evaluate(c['queries'][:5].to(device), c['sampler'], n_corruptions=5, corruption_modes=('head',))
        torch.cuda.synchronize()
        warmup = time() - t0
        print(f"Warmup: {warmup:.2f}s")

        print("Profiling...")

        if args.gpu_profile:
            # GPU Profiler
            torch.cuda.synchronize()
            t0 = time()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                results = c['ppo'].evaluate(
                    c['queries'][:args.n_queries].to(device),
                    c['sampler'], n_corruptions=args.n_corruptions,
                    corruption_modes=('head', 'tail'), verbose=True,
                )
            torch.cuda.synchronize()
            runtime = time() - t0

            ms_cand = (runtime / total) * 1000
            print(f"\n{'='*50}")
            print(f"TIMING SUMMARY")
            print(f"{'='*50}")
            print(f"Runtime:      {runtime:.4f}s")
            print(f"ms/candidate: {ms_cand:.4f}  (target: ≤0.88)")
            print(f"MRR:          {results['MRR']:.4f}  (target: ≈0.16)")
            print(f"\nStatus: {'PASS ✓' if ms_cand <= 0.88 else 'REVIEW'}")

            print(f"\n=== GPU PROFILE: Top 30 by CUDA time ===")
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30))

            print(f"\n=== GPU PROFILE: Top 20 by Self CUDA time ===")
            print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=20))
        elif args.cpu_profile:
            # CPU Profiler
            prof = cProfile.Profile()
            prof.enable()
            torch.cuda.synchronize()
            t0 = time()
            results = c['ppo'].evaluate(
                c['queries'][:args.n_queries].to(device),
                c['sampler'], n_corruptions=args.n_corruptions,
                corruption_modes=('head', 'tail'), verbose=True,
            )
            torch.cuda.synchronize()
            runtime = time() - t0
            prof.disable()

            ms_cand = (runtime / total) * 1000
            print(f"\n{'='*50}")
            print(f"TIMING SUMMARY")
            print(f"{'='*50}")
            print(f"Runtime:      {runtime:.4f}s")
            print(f"ms/candidate: {ms_cand:.4f}  (target: ≤0.88)")
            print(f"MRR:          {results['MRR']:.4f}  (target: ≈0.16)")
            print(f"\nStatus: {'PASS ✓' if ms_cand <= 0.88 else 'REVIEW'}")

            print(f"\nTop 20:")
            s = io.StringIO()
            pstats.Stats(prof, stream=s).strip_dirs().sort_stats('tottime').print_stats(20)
            print(s.getvalue())
        else:
            # Pure Timing (default)
            torch.cuda.synchronize()
            t0 = time()
            results = c['ppo'].evaluate(
                c['queries'][:args.n_queries].to(device),
                c['sampler'], n_corruptions=args.n_corruptions,
                corruption_modes=('head', 'tail'), verbose=True,
            )
            torch.cuda.synchronize()
            runtime = time() - t0

            ms_cand = (runtime / total) * 1000
            print(f"\n{'='*50}")
            print(f"TIMING SUMMARY")
            print(f"{'='*50}")
            print(f"Runtime:      {runtime:.4f}s")
            print(f"ms/candidate: {ms_cand:.4f}  (target: ≤0.88)")
            print(f"MRR:          {results['MRR']:.4f}  (target: ≈0.16)")
            print(f"\nStatus: {'PASS ✓' if ms_cand <= 0.88 else 'REVIEW'}")

        print(f"\nResults saved to {output_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == '__main__':
    main()
