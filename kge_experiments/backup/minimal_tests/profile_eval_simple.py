"""
Simple Evaluation Profiler.

Minimal profiling script to measure pure evaluation performance without PPO overhead.

Usage:
    python tests/profile_eval_simple.py --n-queries 100 --n-corruptions 100
    python tests/profile_eval_simple.py --use-gpu-profiler

Target: 0.3 ms/candidate with 100 queries x 100 corruptions = 20,200 candidates
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

try:
    from torch.profiler import profile, ProfilerActivity
except ImportError:
    profile = None
    ProfilerActivity = None


def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize minimal components for evaluation."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngineVectorized
    from nn.embeddings import EmbedderLearnable as TensorEmbedder
    from policy import ActorCriticPolicy as TensorPolicy
    from nn.sampler import Sampler
    from env_evaluate import EnvEval
    from ppo_evaluate import PPOEval
    
    # Enable compile mode
    import unification
    unification.COMPILE_MODE = config.compile
    
    # Load data
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic',
    )
    
    # Index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    dh.materialize_indices(im=im, device=device)
    
    # Sampler
    default_mode = 'tail' if 'countries' in config.dataset else 'both'
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    torch.manual_seed(config.seed)
    
    # Embedder
    embedder = TensorEmbedder(
        n_constants=im.constant_no,
        n_predicates=im.predicate_no,
        n_vars=1000,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        atom_embedder='transe',
        state_embedder='mean',
        constant_embedding_size=config.atom_embedding_size,
        predicate_embedding_size=config.atom_embedding_size,
        atom_embedding_size=config.atom_embedding_size,
        device=str(device),
    )
    embedder.embed_dim = config.atom_embedding_size
    
    # Vectorized engine
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        padding_atoms=config.padding_atoms,
        max_derived_per_state=config.padding_states,
        end_proof_action=config.end_proof_action,
    )
    
    # Clean up index manager
    im.facts_idx = None
    im.rules_idx = None
    im.rule_lens = None
    im.rules_heads_idx = None
    
    # Convert queries
    def convert_queries(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    test_queries = convert_queries(dh.test_queries)
    
    # Minimal evaluation environment
    env = EnvEval(
        vec_engine=vec_engine,
        batch_size=config.batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        end_proof_action=config.end_proof_action,
    )
    
    # Policy
    policy = TensorPolicy(
        embedder=embedder,
        embed_dim=config.atom_embedding_size,
        action_dim=config.padding_states,
        hidden_dim=256,
        num_layers=8,
        dropout_prob=0.0,
        device=device,
        compile_policy=config.compile,
    ).to(device)
    
    # Minimal PPO wrapper
    ppo = PPOEval(policy, env, device)
    
    return {
        'ppo': ppo,
        'policy': policy,
        'env': env,
        'sampler': sampler,
        'test_queries': test_queries,
        'dh': dh,
        'im': im,
    }


def warmup(components, config) -> float:
    """Run warmup (compilation + first run). Returns warmup time."""
    ppo = components['ppo']
    queries = components['test_queries'][:min(5, config.n_queries)].to(ppo.device)
    sampler = components['sampler']
    
    warmup_start = time()
    
    # Compile eval step
    ppo.compile(mode=config.compile_mode)
    
    # Run minimal eval to trigger CUDA graph capture
    _ = ppo.evaluate(
        queries=queries,
        sampler=sampler,
        n_corruptions=5,
        corruption_modes=('head',),
        verbose=False,
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return time() - warmup_start


def run_evaluation(components, config):
    """Run actual evaluation."""
    ppo = components['ppo']
    queries = components['test_queries'][:config.n_queries].to(ppo.device)
    sampler = components['sampler']
    
    return ppo.evaluate(
        queries=queries,
        sampler=sampler,
        n_corruptions=config.n_corruptions,
        corruption_modes=tuple(config.corruption_modes),
        verbose=config.verbose,
    )


def profile_cprofile(config: SimpleNamespace):
    """Profile with cProfile."""
    output_path = os.path.join(os.path.dirname(__file__), 'profile_eval_simple_results.txt')
    
    # Open log file
    with open(output_path, 'w') as log:
        def log_print(msg):
            print(msg)
            log.write(msg + '\n')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log_print(f"Simple Eval Profile Results")
        log_print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Device: {device}")
        log_print(f"")
        log_print(f"Configuration:")
        log_print(f"  Dataset: {config.dataset}")
        log_print(f"  Queries: {config.n_queries}")
        log_print(f"  Corruptions: {config.n_corruptions}")
        log_print(f"  Corruption modes: {config.corruption_modes}")
        log_print(f"  Batch size: {config.batch_size}")
        log_print(f"  Compile mode: {config.compile_mode}")
        
        n_modes = len(config.corruption_modes)
        total_candidates = config.n_queries * (1 + config.n_corruptions) * n_modes
        log_print(f"  Total candidates: {total_candidates}")
        log_print(f"")
        
        # Setup
        log_print("Setting up components...")
        init_start = time()
        components = setup_components(device, config)
        init_time = time() - init_start
        log_print(f"Init time: {init_time:.2f}s")
        
        # Warmup
        log_print("")
        log_print("Running warmup (compilation)...")
        warmup_time = warmup(components, config)
        log_print(f"Warmup time: {warmup_time:.2f}s")
        
        # Profile
        log_print("")
        log_print("Profiling evaluation...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time()
        
        results = run_evaluation(components, config)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime = time() - start_time
        
        profiler.disable()
        
        # Results
        ms_per_query = (runtime / config.n_queries) * 1000
        ms_per_candidate = (runtime / total_candidates) * 1000
        
        log_print(f"")
        log_print("=" * 60)
        log_print("TIMING SUMMARY")
        log_print("=" * 60)
        log_print(f"Warmup time:      {warmup_time:.4f}s")
        log_print(f"Runtime:          {runtime:.4f}s")
        log_print(f"Total time:       {warmup_time + runtime:.4f}s")
        log_print(f"")
        log_print(f"ms/query:         {ms_per_query:.3f}")
        log_print(f"ms/candidate:     {ms_per_candidate:.3f}  (target: 0.3)")
        log_print(f"")
        log_print(f"MRR:              {results.get('MRR', 0.0):.4f}")
        log_print(f"Hits@1:           {results.get('Hits@1', 0.0):.4f}")
        log_print(f"Hits@3:           {results.get('Hits@3', 0.0):.4f}")
        log_print(f"Hits@10:          {results.get('Hits@10', 0.0):.4f}")
        
        # Profile stats
        log_print("")
        log_print("=" * 60)
        log_print("TOP FUNCTIONS BY CUMULATIVE TIME")
        log_print("=" * 60)
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(30)
        log_print(s.getvalue())
        
        log_print("")
        log_print("=" * 60)
        log_print("TOP FUNCTIONS BY TOTAL TIME")
        log_print("=" * 60)
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('tottime')
        ps.print_stats(30)
        log_print(s.getvalue())
        
        log_print(f"\nResults saved to {output_path}")


def profile_gpu(config: SimpleNamespace):
    """Profile with torch.profiler."""
    if profile is None:
        print("torch.profiler not available")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA not available")
        return
    
    output_path = os.path.join(os.path.dirname(__file__), 'profile_eval_simple_gpu_results.txt')
    
    with open(output_path, 'w') as log:
        def log_print(msg):
            print(msg)
            log.write(msg + '\n')
        
        log_print(f"Simple Eval GPU Profile")
        log_print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        n_modes = len(config.corruption_modes)
        total_candidates = config.n_queries * (1 + config.n_corruptions) * n_modes
        
        log_print("Setting up components...")
        components = setup_components(device, config)
        
        log_print("Running warmup...")
        warmup_time = warmup(components, config)
        log_print(f"Warmup: {warmup_time:.2f}s")
        
        log_print("GPU Profiling...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            start_time = time()
            results = run_evaluation(components, config)
            torch.cuda.synchronize()
            runtime = time() - start_time
        
        ms_per_candidate = (runtime / total_candidates) * 1000
        
        log_print("")
        log_print("=" * 60)
        log_print("TIMING")
        log_print("=" * 60)
        log_print(f"Runtime:          {runtime:.4f}s")
        log_print(f"ms/candidate:     {ms_per_candidate:.3f}")
        log_print(f"MRR:              {results.get('MRR', 0.0):.4f}")
        
        log_print("")
        log_print("=" * 60)
        log_print("TOP CUDA TIME")
        log_print("=" * 60)
        log_print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        
        log_print("")
        log_print("=" * 60)
        log_print("TOP CPU TIME")
        log_print("=" * 60)
        log_print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        
        log_print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Simple Evaluation Profiler')
    
    # Profiler
    parser.add_argument('--use-gpu-profiler', action='store_true')
    
    # Eval config
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--n-queries', type=int, default=100)
    parser.add_argument('--n-corruptions', type=int, default=100)
    parser.add_argument('--corruption-modes', type=str, nargs='+', default=['head', 'tail'])
    parser.add_argument('--batch-size', type=int, default=256)
    
    # Compile
    parser.add_argument('--compile', default=True, type=lambda x: x.lower() != 'false')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead')
    
    # Other
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    config = SimpleNamespace(
        dataset=args.dataset,
        data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
        n_queries=args.n_queries,
        n_corruptions=args.n_corruptions,
        corruption_modes=args.corruption_modes,
        batch_size=args.batch_size,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        max_total_vars=100,
        atom_embedding_size=250,
        seed=42,
        compile=args.compile,
        compile_mode=args.compile_mode,
        verbose=args.verbose,
    )
    
    if args.use_gpu_profiler:
        profile_gpu(config)
    else:
        profile_cprofile(config)


if __name__ == '__main__':
    main()
